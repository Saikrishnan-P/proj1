"""
LangGraph agentic pipeline for CodeSage.

Four nodes, same shape as the OrbitDesk pipeline this reuses patterns from:

    triage -> retrieve -> generate -> verify

- triage:   classify the query (needs code lookup vs. answerable from
            conversation alone) and lightly rewrite it for retrieval.
- retrieve: hybrid dense+sparse search via HybridRetriever (RRF fusion).
- generate: call the LLM (Groq) grounded on retrieved chunks, with citations.
- verify:   check the answer actually cites real retrieved chunks; if it
            hallucinated a citation or came back empty, loop back to
            retrieve once with a broadened query before giving up.
"""
from __future__ import annotations

import os
import re
from typing import Iterator, TypedDict

from langgraph.graph import StateGraph, END

from app.config import settings
from app.retriever import HybridRetriever, RetrievedChunk


class AgentState(TypedDict, total=False):
    query: str
    repo_filter: str | None
    user_id: str | None
    search_query: str
    needs_retrieval: bool
    retrieved: list[RetrievedChunk]
    answer: str
    citations: list[str]
    verified: bool
    retry_count: int
    # Prior turns in this conversation, oldest first, each
    # {"role": "user"|"assistant", "content": str} -- see main.py, which
    # loads these from query_history for a given conversation_id before
    # calling ask()/ask_stream(). Empty/absent for a one-off question
    # with no conversation attached, which behaves exactly as before.
    history: list[dict]
    # Set by verify_node: "high" (verified first try, has citations),
    # "medium" (only passed after a broadened retry), or "low" (still
    # unverified/hallucinated/empty after the retry budget is spent, or
    # no chunks were ever found). Surfaced to the client in
    # QueryResponse.confidence (see main.py) and streamed as the final
    # SSE event's "confidence" field.
    confidence: str


_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def triage_node(state: AgentState) -> AgentState:
    query = state["query"].strip()
    # Simple heuristic triage: greetings/meta questions skip retrieval.
    trivial_patterns = ("hi", "hello", "thanks", "who are you", "what can you do")
    needs_retrieval = not any(query.lower().startswith(p) for p in trivial_patterns)

    search_query = query
    if needs_retrieval and _looks_like_followup(query):
        # Elliptical follow-ups ("what about error handling there?", "and
        # the tests for it?") retrieve poorly on their own -- the code
        # they're pointing at is described mostly in the PREVIOUS
        # question, not this one. Folding the last user turn into the
        # search string (not the whole history -- older turns dilute
        # relevance more than they add) gives the retriever something
        # concrete to match against, while state["query"] (unchanged)
        # still goes to the LLM as the actual question to answer.
        last_user_turn = _last_user_message(state.get("history") or [])
        if last_user_turn:
            search_query = f"{last_user_turn} {query}"

    return {
        **state,
        "needs_retrieval": needs_retrieval,
        "search_query": search_query,
        "retry_count": state.get("retry_count", 0),
    }


_FOLLOWUP_STARTS = (
    "what about", "and ", "also ", "what if", "why not", "how about",
    "does it", "is it", "can it", "does that", "is that",
)


def _looks_like_followup(query: str) -> bool:
    lowered = query.lower()
    if lowered.startswith(_FOLLOWUP_STARTS):
        return True
    # A short question leaning on a pronoun almost always refers back to
    # whatever the previous turn was about ("where's it called?", "why
    # does that fail?") rather than standing alone.
    return len(query.split()) <= 8 and bool(re.search(r"\b(it|that|this|those|these)\b", lowered))


def _last_user_message(history: list[dict]) -> str | None:
    for turn in reversed(history):
        if turn.get("role") == "user" and turn.get("content"):
            return turn["content"]
    return None


def retrieve_node(state: AgentState) -> AgentState:
    if not state.get("needs_retrieval", True):
        return {**state, "retrieved": []}

    retriever = _get_retriever()
    chunks = retriever.retrieve(
        query=state["search_query"],
        repo_filter=state.get("repo_filter"),
        user_id=state.get("user_id"),
        top_k=settings.top_k_final,
    )
    return {**state, "retrieved": chunks}


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] {c.citation}\n{c.text}")
    return "\n\n".join(blocks)


_TRIVIAL_SYSTEM_PROMPT = "You are CodeSage, an assistant for exploring a codebase."
_GROUNDED_SYSTEM_PROMPT = (
    "You are CodeSage, a code Q&A assistant. Answer ONLY using the numbered "
    "code excerpts below. Cite the excerpt number(s) you used like [1], [2]. "
    "If the excerpts don't contain the answer, say so plainly instead of guessing. "
    "Earlier turns in this conversation are provided for context (e.g. resolving "
    "'it'/'that' or avoiding repeating yourself) -- the excerpts below, not the "
    "earlier turns, are still your only source for facts about the code."
)


def _build_messages(state: AgentState, system: str, user: str) -> list[dict]:
    """Prepends prior conversation turns (if any) as real chat messages
    ahead of the current one, rather than flattening them into the system
    or user string -- keeps each turn's role distinct for the model, and
    means the existing single-turn behavior (empty/absent history) needs
    no special-casing here at all, it's just an empty list to prepend."""
    history = state.get("history") or []
    trimmed = history[-2 * settings.conversation_history_turns:]
    return (
        [{"role": "system", "content": system}]
        + [{"role": h["role"], "content": h["content"]} for h in trimmed]
        + [{"role": "user", "content": user}]
    )


def generate_node(state: AgentState) -> AgentState:
    chunks = state.get("retrieved", [])

    if not chunks and state.get("needs_retrieval", True):
        return {
            **state,
            "answer": "I couldn't find anything relevant in the indexed repo for that question.",
            "citations": [],
        }

    if not state.get("needs_retrieval", True):
        messages = _build_messages(state, _TRIVIAL_SYSTEM_PROMPT, state["query"])
        answer = _call_llm(messages)
        return {**state, "answer": answer, "citations": []}

    context = _build_context_block(chunks)
    user = f"Question: {state['query']}\n\nCode excerpts:\n{context}"
    messages = _build_messages(state, _GROUNDED_SYSTEM_PROMPT, user)
    answer = _call_llm(messages)

    citations = [c.citation for c in chunks]
    return {**state, "answer": answer, "citations": citations}


def verify_node(state: AgentState) -> AgentState:
    """Guard against hallucinated citations: if the model cited a bracket
    number that doesn't exist in the retrieved set, or produced an empty
    answer while chunks *were* available, retry retrieval once with a
    broadened query before surfacing the (unverified) answer as a fallback.

    Also assigns a user-facing confidence label -- "high" / "medium" /
    "low" -- since an unverified answer is still returned rather than
    hidden (a wrong-but-plausible-looking answer with no context is worse
    than the same answer flagged as uncertain), so the caller needs a way
    to tell the two apart. See main.py/QueryResponse.confidence."""
    answer = state.get("answer", "")
    chunks = state.get("retrieved", [])
    retry_count = state.get("retry_count", 0)

    cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    valid_range = set(range(1, len(chunks) + 1))
    hallucinated = bool(cited_numbers - valid_range) if chunks else False
    empty = not answer.strip()

    if (hallucinated or empty) and chunks and retry_count < 1:
        broadened = " ".join(state["search_query"].split()[:3]) or state["search_query"]
        return {**state, "search_query": broadened, "retry_count": retry_count + 1, "verified": False}

    confidence = _score_confidence(
        answer, chunks, state.get("needs_retrieval", True), retried=retry_count > 0
    )
    return {**state, "verified": True, "confidence": confidence}


def _route_after_verify(state: AgentState) -> str:
    return "retrieve" if not state.get("verified", True) else END


def _call_llm(messages: list[dict]) -> str:
    """Thin wrapper around Groq's OpenAI-compatible chat completion API.
    Kept isolated so it's trivial to swap providers or mock in tests."""
    if not settings.groq_api_key:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            "[No GROQ_API_KEY set — returning retrieved context only]\n\n"
            f"{last_user[:2000]}"
        )
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
    )
    return completion.choices[0].message.content


def _call_llm_stream(messages: list[dict]) -> Iterator[str]:
    """Same call as _call_llm, but yields text deltas as they arrive
    instead of waiting for the full completion -- what powers
    POST /query/stream (see main.py). Only ever used for the grounded
    "generate" step's final answer text; retrieval and verification
    still happen synchronously either way (see ask_stream() below)."""
    if not settings.groq_api_key:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        yield (
            "[No GROQ_API_KEY set — returning retrieved context only]\n\n"
            f"{last_user[:2000]}"
        )
        return
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    stream = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", _route_after_verify, {"retrieve": "retrieve", END: END})

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(
    query: str,
    repo_filter: str | None = None,
    user_id: str | None = None,
    history: list[dict] | None = None,
) -> AgentState:
    graph = get_graph()
    return graph.invoke({
        "query": query, "repo_filter": repo_filter, "user_id": user_id,
        "history": history or [],
    })


def ask_stream(
    query: str,
    repo_filter: str | None = None,
    user_id: str | None = None,
    history: list[dict] | None = None,
) -> Iterator[dict]:
    """Streaming counterpart to ask() -- yields dicts of the shape
    {"type": "token", "text": ...} as the answer is generated, then
    exactly one final {"type": "done", "answer", "citations",
    "confidence"} once the whole answer is in. Used by
    POST /query/stream (see main.py), which turns each yielded dict into
    one SSE event.

    Deliberately reuses triage_node/retrieve_node as-is (retrieval isn't
    what a user is waiting on token-by-token for), but does NOT reuse
    verify_node's retry-on-hallucination loop: that loop re-runs
    *generation* from scratch with a broadened query, which only makes
    sense before anything has been shown to the user. Once tokens are
    streaming, a retry would mean either silently discarding what
    they've already seen or visibly restarting the answer under them --
    both worse than just being honest about lower confidence on the
    first pass. Confidence is still computed (via _score_confidence,
    shared with verify_node's own logic) so the client gets the same
    high/medium/low signal either way, just without the retry."""
    state: AgentState = {
        "query": query, "repo_filter": repo_filter, "user_id": user_id,
        "history": history or [], "retry_count": 0,
    }
    state = triage_node(state)
    state = retrieve_node(state)

    chunks = state.get("retrieved", [])
    if not chunks and state.get("needs_retrieval", True):
        answer = "I couldn't find anything relevant in the indexed repo for that question."
        yield {"type": "token", "text": answer}
        yield {"type": "done", "answer": answer, "citations": [], "confidence": "low"}
        return

    if not state.get("needs_retrieval", True):
        messages = _build_messages(state, _TRIVIAL_SYSTEM_PROMPT, state["query"])
        citations: list[str] = []
    else:
        context = _build_context_block(chunks)
        user = f"Question: {state['query']}\n\nCode excerpts:\n{context}"
        messages = _build_messages(state, _GROUNDED_SYSTEM_PROMPT, user)
        citations = [c.citation for c in chunks]

    answer_parts: list[str] = []
    for delta in _call_llm_stream(messages):
        answer_parts.append(delta)
        yield {"type": "token", "text": delta}

    answer = "".join(answer_parts)
    confidence = _score_confidence(answer, chunks, state.get("needs_retrieval", True), retried=False)
    yield {"type": "done", "answer": answer, "citations": citations, "confidence": confidence}


def _score_confidence(answer: str, chunks: list[RetrievedChunk], needs_retrieval: bool, retried: bool) -> str:
    """Shared by verify_node (post-retry) and ask_stream (no retry) so
    both paths agree on what "high"/"medium"/"low" means."""
    cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    valid_range = set(range(1, len(chunks) + 1))
    hallucinated = bool(cited_numbers - valid_range) if chunks else False
    empty = not answer.strip()

    if not needs_retrieval:
        return "high" if not empty else "low"
    if not chunks or hallucinated or empty:
        return "low"
    return "medium" if retried else "high"