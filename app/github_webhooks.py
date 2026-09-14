"""
GitHub push webhooks -- what makes incremental re-indexing on push work.

Flow:
    1. Right after a full ingest of a github.com source succeeds (see
       app/jobs.py's run_ingest_job), if the ingesting user authenticated
       via GitHub (so we have a token that might have admin rights on
       that repo) we call register_webhook_if_possible(). That POSTs a
       webhook to GitHub asking it to notify BACKEND_PUBLIC_URL on every
       `push` event, and stores the returned webhook id + a freshly
       generated per-repo secret in app/repos.py's indexed_repos table.
    2. On every push, GitHub POSTs the payload to
       POST /webhooks/github/{indexed_repo_id} (see main.py), signed with
       that same secret via the X-Hub-Signature-256 header.
    3. main.py verifies the signature with verify_signature() below,
       parses out changed/removed file paths with parse_push_event(), and
       enqueues an incremental re-index job (app/jobs.py) instead of a
       full one.

Registration is best-effort everywhere: no BACKEND_PUBLIC_URL configured,
no GitHub token on the ingesting user, or a 403 because they're not an
admin on that repo (common for "let me test this on some public repo I
don't own") all just mean "skip it, no incremental re-indexing for this
repo" -- never a reason to fail the ingest itself, since the repo is
still fully usable via manual re-ingestion either way.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

import httpx

from app.config import settings

GITHUB_API_BASE = "https://api.github.com"


def register_webhook_if_possible(
    owner: str, repo: str, github_token: str | None, indexed_repo_id: str
) -> tuple[int, str] | None:
    """Returns (webhook_id, webhook_secret) on success, None if skipped
    or it failed for any reason (logged, never raised -- see module
    docstring for why this must never block ingestion)."""
    if not settings.backend_public_url or not github_token:
        return None

    secret = secrets.token_hex(32)
    callback_url = f"{settings.backend_public_url}/webhooks/github/{indexed_repo_id}"

    try:
        resp = httpx.post(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/hooks",
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "name": "web",
                "active": True,
                "events": ["push"],
                "config": {
                    "url": callback_url,
                    "content_type": "json",
                    "secret": secret,
                    "insecure_ssl": "0",
                },
            },
            timeout=10,
        )
    except httpx.HTTPError as e:
        print(f"[github_webhooks] Couldn't reach GitHub to register a webhook for {owner}/{repo}: {e}")
        return None

    if resp.status_code not in (200, 201):
        # Most commonly a 403 (not an admin on this repo) or a 422 (a
        # webhook to this exact URL already exists) -- either way, not
        # something the ingesting request should surface as an error.
        print(
            f"[github_webhooks] Webhook registration for {owner}/{repo} returned "
            f"{resp.status_code}: {resp.text[:200]}"
        )
        return None

    webhook_id = resp.json().get("id")
    if webhook_id is None:
        return None
    return webhook_id, secret


def verify_signature(secret: str, payload_body: bytes, signature_header: str | None) -> bool:
    """GitHub signs the raw request body with HMAC-SHA256 keyed on the
    per-webhook secret, sent as `X-Hub-Signature-256: sha256=<hexdigest>`.
    Constant-time comparison (hmac.compare_digest) so response timing
    can't be used to guess the correct digest byte-by-byte."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), payload_body, hashlib.sha256).hexdigest()
    provided = signature_header[len("sha256="):]
    return hmac.compare_digest(expected, provided)


def parse_push_event(payload: dict) -> dict:
    """Extracts what indexer.incremental_index_repo() actually needs from
    a GitHub push webhook payload: the new HEAD commit and the union of
    every added/modified/removed file path across all commits in the
    push (a push can carry more than one commit -- e.g. someone pushing
    three local commits at once -- and a file touched in an earlier
    commit but not the latest still needs picking up)."""
    added_or_modified: set[str] = set()
    removed: set[str] = set()

    for commit in payload.get("commits", []):
        added_or_modified.update(commit.get("added", []))
        added_or_modified.update(commit.get("modified", []))
        removed.update(commit.get("removed", []))

    # A file removed in one commit of the push and re-added in a later
    # one should end up as "changed", not "removed".
    removed -= added_or_modified

    return {
        "after_sha": payload.get("after"),
        "changed_files": sorted(added_or_modified),
        "removed_files": sorted(removed),
        "ref": payload.get("ref"),
    }
