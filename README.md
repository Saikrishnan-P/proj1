# CodeSage

RAG system for querying GitHub repositories in natural language, built around
AST-based chunking, hybrid retrieval, and a LangGraph agentic pipeline.

## Architecture

```
Ingestion:  GitHub repo -> AST chunker -> ChromaDB (dense) + BM25 (sparse)
Query:      User query -> Hybrid retrieval (RRF fusion) -> LangGraph agent -> Answer
```

- **AST chunker** (`app/chunker.py`) — parses each Python file's AST and emits
  one chunk per function/method/class, instead of blindly splitting by lines.
  Falls back to a sliding-window text splitter for unparseable or non-Python
  files.
- **Hybrid retrieval** (`app/retriever.py`) — combines ChromaDB dense vector
  search (`sentence-transformers/all-MiniLM-L6-v2` by default) with BM25
  keyword search, merged via **Reciprocal Rank Fusion (RRF)**. Dense catches
  conceptual/paraphrased questions; BM25 catches exact identifiers, error
  strings, and decorators. RRF combines the two rank lists without needing to
  normalize incomparable similarity scores.
- **LangGraph pipeline** (`app/graph.py`) — a `triage -> retrieve -> generate
  -> verify` agent graph. `verify` checks the generated answer's citations
  against the retrieved chunk set and loops back to `retrieve` once with a
  broadened query if it detects a hallucinated citation or an empty answer.
- **API** (`app/main.py`) — FastAPI app with `/ingest` and `/query` endpoints,
  deployable to Railway like your other projects.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your GROQ_API_KEY
```

## Usage

### Option A — CLI (fastest for a demo)

Run it with no arguments and it'll prompt you to paste a GitHub link:

```bash
python scripts/ingest_repo.py
Paste a GitHub link (or a local repo path): https://github.com/someuser/somerepo
```

Or pass the link directly (a local folder path also works):

```bash
python scripts/ingest_repo.py https://github.com/someuser/somerepo
python scripts/ingest_repo.py https://github.com/someuser/somerepo my-repo-name
python scripts/query.py "How does the retriever combine dense and sparse results?"
```

Once you've ingested more than one repo, list what's indexed and scope a query to just one of them:

```bash
python scripts/list_repos.py
python scripts/query.py "How does file upload work?" --repo my-repo-name
```

Without `--repo`, retrieval searches across every repo you've ever ingested — fine with one repo, but it can blend unrelated chunks into an answer once you have two or more indexed at once.

### Option B — API

```bash
uvicorn app.main:app --reload
```

Then paste a link into the same `source` field the CLI uses:

```bash
curl -X POST localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "https://github.com/someuser/somerepo"}'

curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is the database connection configured?"}'
```

`source` accepts a GitHub link (https, `.git`, or ssh) or a local folder path — it's auto-detected either way, so the same field works for both the demo and local dev.

Swagger UI is at `localhost:8000/docs`.

## Selling API access (usage-based billing)

Third-party developers can use `/ingest` and `/query` without a CodeSage
login at all, authenticating with an API key instead of a JWT:

```bash
# 1. Sign up as an API customer (org) -- returns an API key, shown once
curl -X POST localhost:8000/developers/signup \
  -H "Content-Type: application/json" \
  -d '{"org_name": "Acme Dev Tools", "email": "eng@acme.dev"}'

# 2. Buy credits (Stripe Checkout) -- see "packs" for pricing
curl localhost:8000/developers/billing/packs
curl -X POST localhost:8000/developers/billing/checkout \
  -H "X-API-Key: cs_live_..." -H "Content-Type: application/json" \
  -d '{"pack_id": "starter"}'
# -> {"checkout_url": "https://checkout.stripe.com/..."}, open it and pay

# 3. Use the same /ingest and /query endpoints, with X-API-Key instead of a Bearer token
curl -X POST localhost:8000/ingest \
  -H "X-API-Key: cs_live_..." -H "Content-Type: application/json" \
  -d '{"source": "https://github.com/someuser/somerepo"}'
curl -X POST localhost:8000/query \
  -H "X-API-Key: cs_live_..." -H "Content-Type: application/json" \
  -d '{"question": "Where is the database connection configured?"}'
```

- Each `/query` (or `/query/stream`) call costs a flat `QUERY_CREDIT_COST`
  credits, deducted before any retrieval/LLM work happens. Each `/ingest`
  costs credits proportional to repo size (`INGEST_CREDIT_COST_PER_CHUNKS`),
  deducted after the job completes since the true cost isn't known until
  then. Logged-in web users (email/password or GitHub OAuth) are never
  charged -- only `X-API-Key` requests are metered. See `app/billing.py`.
- Credits are prepaid via Stripe Checkout (one-time purchases, not a
  subscription) -- see `app/billing.py`'s `CREDIT_PACKS`. Requires
  `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` (a webhook endpoint at
  `{BACKEND_PUBLIC_URL}/webhooks/stripe` listening for
  `checkout.session.completed`) -- see `.env.example`.
- An API customer is its own principal, separate from `users` -- see
  `GET /developers/api-keys`, `POST /developers/api-keys` (mint another
  key for your org), and `GET /developers/usage` (balance + transaction
  history). `/repos` also works with an API key, scoped to that org the
  same way it's scoped per-user for a logged-in account.

## Deploying to Railway

The repo builds as a Docker image (`Dockerfile`) with the embedding model
baked in at build time, so cold starts don't need a live model download.

1. **Push this repo to GitHub**, then in Railway: New Project → Deploy from
   GitHub repo.
2. **Attach a Volume** — this is the step it's easy to skip and then wonder
   why your index disappears on every redeploy. Railway's filesystem is
   ephemeral outside of a Volume. In your service settings: Volumes → New
   Volume → mount path `/app/data`. This is where both ChromaDB
   (`/app/data/chroma`) and the BM25 pickle (`/app/data/bm25_index.pkl`)
   persist.
3. **Set environment variables** (Service → Variables): at minimum
   `GROQ_API_KEY`. Everything else in `.env.example` has a working default.
   Railway sets `$PORT` itself — don't set it manually.
4. **Deploy.** Railway builds the Dockerfile and starts the service per
   `railway.json` (`healthcheckPath: /health`, 120s timeout to cover the
   image's first boot).
5. **Ingest your repo(s) against the live URL** once it's up:
   ```bash
   curl -X POST https://<your-app>.up.railway.app/ingest \
     -H "Content-Type: application/json" \
     -d '{"repo_url": "https://github.com/someuser/somerepo"}'
   ```
   Do this once per repo after each deploy that touches the Volume, or
   automate it as a post-deploy step if you want ingestion to survive
   redeploys automatically.

**Things to know before you rely on this for the live panel demo:**
- Free/small Railway plans have limited RAM — `sentence-transformers` +
  ChromaDB + a loaded BM25 index for a large repo can get memory-hungry.
  Test against your actual repo size before demo day, not a toy repo.
- `/ingest` clones via `git clone` inside the container — private repos
  need a `GIT_ASKPASS`/token setup this project doesn't include yet.
- CORS is wide open (`allow_origins=["*"]`) for demo convenience. Tighten
  this in `app/main.py` before pointing real traffic at it.

## Tests

```bash
pytest tests/
```

`tests/test_chunker.py` covers chunking correctness and the RRF fusion math
without needing ChromaDB, sentence-transformers, or Groq installed — good for
a quick CI check or a live demo of "here's how I verified retrieval quality."

## Notes for the panel demo

- Swap `GROQ_API_KEY` for any Groq-hosted model; if it's unset, `/query`
  still returns the retrieved, cited context so retrieval quality can be
  demoed independently of generation.
- Currently only `.py` files are AST-chunked (`app/config.py ->
  supported_extensions`). Extend with a JS/TS/Go parser (e.g. `tree-sitter`)
  for polyglot repos — the chunker's `Chunk` dataclass and metadata shape
  don't need to change.
- BM25 is rebuilt in memory on each ingest (see the docstring in
  `indexer.py`); fine at repo scale, but call that out if asked about
  scaling to a multi-repo index.
