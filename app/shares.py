"""
Shareable answer permalinks -- GET /share/{id} (see main.py) is a public,
unauthenticated endpoint, so this module is careful to only ever store
and return what's safe for anyone with the link to see: the question,
answer, citations, and confidence. Never the owning user_id, their email,
or which OTHER repos they have indexed.
"""
from __future__ import annotations

import time
import uuid

from app.db import get_cursor


def create_share(
    owner_user_id: str,
    question: str,
    answer: str,
    citations: list[str],
    confidence: str | None,
    repo_filter: str | None,
) -> dict:
    import json
    share_id = uuid.uuid4().hex[:12]  # short enough for a clean-looking /share/<id> URL
    now = time.time()
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO shared_answers "
            "(id, owner_user_id, question, answer, citations, confidence, repo_filter, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (share_id, owner_user_id, question, answer, json.dumps(citations), confidence, repo_filter, now),
        )
    return {
        "id": share_id, "question": question, "answer": answer, "citations": citations,
        "confidence": confidence, "repo_filter": repo_filter, "created_at": now,
    }


def get_share(share_id: str) -> dict | None:
    """Public lookup -- deliberately does not filter or join on
    owner_user_id, since anyone with the link is meant to be able to view
    it. Returns None for an unknown or since-deleted id."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, question, answer, citations, confidence, repo_filter, created_at "
            "FROM shared_answers WHERE id = %s",
            (share_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def delete_share(owner_user_id: str, share_id: str) -> bool:
    """Owner-only revocation. Returns False (rather than raising) for an
    id that doesn't exist or belongs to someone else -- main.py maps that
    to a 404 without distinguishing the two, so a guessed id can't be
    used to confirm another user's share exists."""
    with get_cursor(dict_rows=False) as cur:
        cur.execute(
            "DELETE FROM shared_answers WHERE id = %s AND owner_user_id = %s",
            (share_id, owner_user_id),
        )
        return cur.rowcount > 0
