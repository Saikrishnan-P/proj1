"""
Conversation threads -- the grouping that makes multi-turn memory
possible. Each row here is just a label + timestamps; the actual
question/answer content still lives in query_history (see app/auth.py),
tagged with this conversation's id.
"""
from __future__ import annotations

import time
import uuid

from app.db import get_cursor

MAX_TITLE_CHARS = 80


def create_conversation(user_id: str, repo_filter: str | None = None, title: str | None = None) -> dict:
    conversation_id = uuid.uuid4().hex
    now = time.time()
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (id, user_id, title, repo_filter, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (conversation_id, user_id, title, repo_filter, now, now),
        )
    return {"id": conversation_id, "user_id": user_id, "title": title, "repo_filter": repo_filter,
            "created_at": now, "updated_at": now}


def get_conversation(user_id: str, conversation_id: str) -> dict | None:
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, user_id, title, repo_filter, created_at, updated_at "
            "FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, title, repo_filter, created_at, updated_at "
            "FROM conversations WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
            (user_id, limit),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def touch_conversation(conversation_id: str, title_if_unset: str | None = None) -> None:
    """Bumps updated_at (so the conversation list sorts most-recently-
    active first) after every turn, and backfills a title from the
    conversation's first question if it was created without one --
    trimmed to MAX_TITLE_CHARS so a long first question doesn't blow out
    whatever UI renders a conversation list."""
    with get_cursor(dict_rows=False) as cur:
        if title_if_unset:
            trimmed = title_if_unset[:MAX_TITLE_CHARS]
            cur.execute(
                "UPDATE conversations SET updated_at = %s, "
                "title = COALESCE(title, %s) WHERE id = %s",
                (time.time(), trimmed, conversation_id),
            )
        else:
            cur.execute(
                "UPDATE conversations SET updated_at = %s WHERE id = %s",
                (time.time(), conversation_id),
            )
