"""
API keys for the Developer API. Every key is minted for an api_customer
row, which is auto-created transparently the first time a LOGGED-IN
CodeSage user (any of: password, GitHub, Google) touches any
/developers/* endpoint -- see get_or_create_api_customer_for_user below
and main.py's get_or_create_current_api_customer dependency. There's no
separate "developer signup" step for someone who already has a CodeSage
account; POST /developers/signup still exists for the one case that
genuinely needs it (a third-party integrator who wants API-only access
with no CodeSage login at all), but the web app's own Developer API page
never calls it.

Key format: `cbi_<32 random url-safe chars>`. The `cbi_` prefix is never
secret (it's shown back in list_api_keys for the customer to tell keys
apart) and lets us cheaply reject obviously-malformed keys before even
hitting the database. Only a sha256 hash of the full key is ever stored --
see app/db.py's api_keys table docstring for why sha256 (not PBKDF2) is
the right choice here.
"""
from __future__ import annotations

import hashlib
import secrets
import time
import uuid

import psycopg2

from app.db import get_cursor

KEY_PREFIX = "cbi_"
KEY_RANDOM_CHARS = 32


class ApiKeyError(Exception):
    """Raised for any api-customer signup/key-management failure; main.py
    maps this to a 400."""


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """Returns (raw_key, key_hash, key_prefix). raw_key is returned to
    the caller exactly once -- neither this function nor anything that
    stores its output ever needs to see it again."""
    raw_key = KEY_PREFIX + secrets.token_urlsafe(KEY_RANDOM_CHARS)
    return raw_key, _hash_key(raw_key), raw_key[:len(KEY_PREFIX) + 6]


# ---------------------------------------------------------------------------
# API customers (orgs)
# ---------------------------------------------------------------------------

def create_api_customer(org_name: str, email: str) -> tuple[dict, str]:
    """Creates a new api_customer and its first API key. Returns
    (api_customer, raw_key) -- raw_key must be shown to the caller now;
    it can never be retrieved again (see create_api_key's docstring for
    how to mint a replacement/additional key later)."""
    org_name = org_name.strip()
    email = email.strip().lower()
    if not org_name:
        raise ApiKeyError("org_name must not be empty.")
    if not email or "@" not in email:
        raise ApiKeyError("Please provide a valid email address.")

    customer_id = uuid.uuid4().hex
    now = time.time()
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO api_customers (id, org_name, email, credit_balance, created_at) "
            "VALUES (%s, %s, %s, 0, %s)",
            (customer_id, org_name, email, now),
        )
    customer = {"id": customer_id, "org_name": org_name, "email": email, "credit_balance": 0}

    raw_key = create_api_key(customer_id, label="default")
    return customer, raw_key


def get_api_customer(customer_id: str) -> dict | None:
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, org_name, email, credit_balance, created_at "
            "FROM api_customers WHERE id = %s",
            (customer_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_or_create_api_customer_for_user(user_id: str, email: str) -> dict:
    """Auto-creates (or fetches) the api_customer record backing a
    logged-in CodeSage web user's Developer API access. Deliberately
    reuses the SAME id as the user's own users.id rather than minting a
    separate uuid -- exactly one api_customer per web account, created
    lazily and transparently the first time they touch any /developers/*
    endpoint. This is what main.py's get_or_create_current_api_customer
    dependency calls; see that dependency's docstring for the full
    "why no separate developer signup" reasoning."""
    existing = get_api_customer(user_id)
    if existing:
        return existing

    now = time.time()
    org_name = email  # no separate "org name" concept for an auto-created personal account
    try:
        with get_cursor() as cur:
            cur.execute(
                "INSERT INTO api_customers (id, org_name, email, credit_balance, created_at) "
                "VALUES (%s, %s, %s, 0, %s)",
                (user_id, org_name, email, now),
            )
    except psycopg2.IntegrityError:
        # Race: two concurrent first-touch requests (e.g. two tabs both
        # loading the Developer API page at once) both saw "missing" and
        # both tried to insert -- the loser's insert conflicts on the
        # primary key (id = user_id). Letting the exception propagate out
        # of the `with get_cursor()` block above (rather than catching it
        # INSIDE that block) matters: it's what makes get_conn() roll the
        # aborted transaction back correctly before we touch the
        # connection again below -- catching it inside the block instead
        # would leave that transaction aborted, and the implicit commit()
        # on a clean exit would itself then raise InFailedSqlTransaction.
        pass

    return get_api_customer(user_id) or {
        "id": user_id, "org_name": org_name, "email": email, "credit_balance": 0,
    }


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def create_api_key(api_customer_id: str, label: str | None = None) -> str:
    """Mints an additional key for an existing api_customer (e.g. one
    per environment/service on their end) -- callers authenticate this
    with any one of their existing valid keys (see main.py's
    POST /developers/api-keys), so losing one key doesn't lock them out
    of minting a replacement. Returns the raw key -- shown once."""
    raw_key, key_hash, key_prefix = generate_api_key()
    key_id = uuid.uuid4().hex
    now = time.time()
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (id, api_customer_id, key_hash, key_prefix, label, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (key_id, api_customer_id, key_hash, key_prefix, label, now),
        )
    return raw_key


def authenticate_api_key(raw_key: str) -> dict | None:
    """Looks up the api_customer owning a raw key, or None if it's
    unknown, revoked, or malformed. Also bumps last_used_at (best-effort,
    non-blocking -- see below) so list_api_keys can show "last used"
    without needing separate request logging."""
    if not raw_key or not raw_key.startswith(KEY_PREFIX):
        return None

    key_hash = _hash_key(raw_key)
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, api_customer_id FROM api_keys "
            "WHERE key_hash = %s AND revoked_at IS NULL",
            (key_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute(
            "UPDATE api_keys SET last_used_at = %s WHERE id = %s",
            (time.time(), row["id"]),
        )

    return get_api_customer(row["api_customer_id"])


def list_api_keys(api_customer_id: str) -> list[dict]:
    """Masked key list -- key_prefix only, never the full key or its
    hash, since neither should ever leave the process that stored it."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, key_prefix, label, created_at, last_used_at, revoked_at "
            "FROM api_keys WHERE api_customer_id = %s ORDER BY created_at DESC",
            (api_customer_id,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def revoke_api_key(api_customer_id: str, key_id: str) -> bool:
    """Owner-scoped revocation. Returns False for an id that doesn't
    exist, is already revoked, or belongs to a different api_customer --
    same non-distinguishing posture as the rest of this codebase's
    ownership checks (see app/shares.py's delete_share).

    No longer refuses to revoke your last active key. That guard existed
    back when the ONLY way to manage your keys at all was presenting a
    valid X-API-Key -- revoking your one remaining key would have meant
    no way back in. Now that key management is authenticated with your
    normal CodeSage login instead (see main.py's
    get_or_create_current_api_customer), that lockout is impossible: you
    can always log back in and mint a fresh key, so there's nothing left
    for this guard to protect against, only friction it would add."""
    with get_cursor(dict_rows=False) as cur:
        cur.execute(
            "UPDATE api_keys SET revoked_at = %s "
            "WHERE id = %s AND api_customer_id = %s AND revoked_at IS NULL",
            (time.time(), key_id, api_customer_id),
        )
        return cur.rowcount > 0
