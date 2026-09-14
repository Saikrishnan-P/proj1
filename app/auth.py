"""
User accounts, password hashing, JWT issuance/validation, and per-user query
history -- all backed by Postgres now (see app/db.py) instead of a SQLite
file on the same local volume as the old vector/BM25 indexes. This is what
makes it safe to run more than one API instance (or an RQ worker process,
coming in stage 2): every process talks to the same DATABASE_URL instead of
each having its own local users.db.

Every function signature here is unchanged from the SQLite version --
main.py doesn't need to change at all for this migration.

Password hashing still uses PBKDF2-HMAC-SHA256 (Python's stdlib hashlib, no
extra dependency) with a random per-user salt and 260k iterations --
OWASP's current minimum recommendation for PBKDF2-SHA256 as of their latest
cheat sheet revision.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import uuid

import jwt
import psycopg2
from cryptography.fernet import Fernet
from fastapi import Header, HTTPException

from app.config import settings
from app.db import get_cursor, init_db as _init_db

PBKDF2_ITERATIONS = 260_000


def init_db() -> None:
    """Kept as a thin re-export so main.py's `auth.init_db()` call site
    doesn't need to change -- the actual DDL now lives in app/db.py
    because chunks (indexer.py/retriever.py) need the same call at
    startup, and duplicating the table-creation logic in two places would
    just be a second place for it to drift out of sync."""
    _init_db()


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Returns (salt, hash), both hex-encoded. Pass an existing salt to
    check a login attempt against a stored hash; omit it to hash a brand
    new password at signup (a fresh random salt is generated)."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    )
    return salt, digest.hex()


def _verify_password(password: str, salt: str, expected_hash: str) -> bool:
    _, computed_hash = _hash_password(password, salt=salt)
    return hmac.compare_digest(computed_hash, expected_hash)  # constant-time, avoids timing attacks


# ---------------------------------------------------------------------------
# GitHub access token encryption at rest
#
# A GitHub access token is a live credential (it can clone your private
# repos), not just an identifier -- worth meaningfully more protection than
# a password hash, since a password hash is one-way and useless to an
# attacker without cracking it, while a leaked plaintext GitHub token is
# immediately usable as-is. Fernet (symmetric, authenticated encryption)
# from the `cryptography` package keeps it unreadable at rest in Postgres;
# only this process, holding TOKEN_ENCRYPTION_KEY, can decrypt it back.
# ---------------------------------------------------------------------------

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        if not settings.token_encryption_key:
            raise RuntimeError(
                "TOKEN_ENCRYPTION_KEY is not set -- required to store or read "
                "GitHub access tokens. Generate one with: "
                "python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\""
            )
        _fernet = Fernet(settings.token_encryption_key.encode())
    return _fernet


def encrypt_token(raw: str) -> str:
    return _get_fernet().encrypt(raw.encode()).decode()


def decrypt_token(encrypted: str) -> str:
    return _get_fernet().decrypt(encrypted.encode()).decode()


# ---------------------------------------------------------------------------
# User accounts
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """Raised for any signup/login failure; main.py maps this to a 400/401."""


def create_user(email: str, password: str) -> dict:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise AuthError("Please provide a valid email address.")
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")

    salt, password_hash = _hash_password(password)
    user_id = uuid.uuid4().hex

    with get_cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            raise AuthError("An account with that email already exists.")
        try:
            cur.execute(
                "INSERT INTO users (id, email, password_hash, password_salt, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user_id, email, password_hash, salt, time.time()),
            )
        except psycopg2.IntegrityError:
            raise AuthError("An account with that email already exists.")

    return {"id": user_id, "email": email}


def authenticate_user(email: str, password: str) -> dict:
    email = email.strip().lower()
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, email, password_hash, password_salt FROM users WHERE email = %s",
            (email,),
        )
        row = cur.fetchone()

    if row is None or not _verify_password(password, row["password_salt"], row["password_hash"]):
        raise AuthError("Incorrect email or password.")

    return {"id": row["id"], "email": row["email"]}


def get_user_by_id(user_id: str) -> dict | None:
    with get_cursor() as cur:
        cur.execute("SELECT id, email, email_verified FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    return {"id": row["id"], "email": row["email"], "email_verified": row["email_verified"]} if row else None


def get_user_by_email(email: str) -> dict | None:
    """Used by POST /auth/resend-verification (see main.py) -- kept
    separate from authenticate_user, which also checks a password and
    raises on a bad one; this just looks the account up, no password
    involved."""
    email = email.strip().lower()
    with get_cursor() as cur:
        cur.execute("SELECT id, email, email_verified FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
    return {"id": row["id"], "email": row["email"], "email_verified": row["email_verified"]} if row else None


# ---------------------------------------------------------------------------
# GitHub OAuth accounts
# ---------------------------------------------------------------------------

def find_or_create_github_user(
    github_id: int, github_login: str, email: str | None, access_token: str
) -> dict:
    """Called after a successful GitHub OAuth callback (see
    app/github_oauth.py). Links to an existing account by github_id if
    we've seen this GitHub user before -- refreshing their stored token,
    since GitHub tokens can be revoked/rotated -- otherwise creates a new
    account.

    Deliberately links by github_id, never by email: if some other
    account already used this email (a password signup, say), a stranger
    who happens to control a GitHub account with a matching public email
    must never be able to silently take over that existing account just
    by clicking "Continue with GitHub"."""
    encrypted = encrypt_token(access_token)

    with get_cursor() as cur:
        cur.execute("SELECT id, email FROM users WHERE github_id = %s", (github_id,))
        row = cur.fetchone()
        if row:
            cur.execute(
                "UPDATE users SET github_access_token_encrypted = %s, github_login = %s WHERE id = %s",
                (encrypted, github_login, row["id"]),
            )
            return {"id": row["id"], "email": row["email"]}

        user_id = uuid.uuid4().hex
        # GitHub's user:email scope can still come back with no public
        # email (github_oauth.py already tries the dedicated emails
        # endpoint first) -- fall back to a synthetic, GitHub-reserved
        # noreply address rather than leaving email NULL, since the
        # `users.email` column is NOT NULL UNIQUE and other code paths
        # (e.g. displaying it in the UI) assume every account has one.
        resolved_email = email or f"{github_id}+{github_login}@users.noreply.github.com"
        try:
            cur.execute(
                "INSERT INTO users "
                "(id, email, github_id, github_login, github_access_token_encrypted, "
                "email_verified, created_at) "
                "VALUES (%s, %s, %s, %s, %s, TRUE, %s)",
                (user_id, resolved_email, github_id, github_login, encrypted, time.time()),
            )
        except psycopg2.IntegrityError:
            # A password account already exists with this exact email
            # (rare, but possible with a real public GitHub email) --
            # refuse to auto-link accounts across sign-in methods, same
            # reasoning as the github_id-only lookup above.
            raise AuthError(
                f"An account already exists for {resolved_email}. "
                "Log in with your password instead, or use a different GitHub account."
            )

    return {"id": user_id, "email": resolved_email}


def link_github_account(user_id: str, github_id: int, github_login: str, access_token: str) -> None:
    """Attaches a GitHub account to an ALREADY-LOGGED-IN user's existing
    account, rather than creating (or finding) a separate one -- this is
    what /auth/github/link/start + the "link" branch of /auth/github/callback
    use (see main.py), as opposed to find_or_create_github_user above, which
    is for signing in via GitHub when there's no existing session at all.

    Lets a user who originally signed up with a password later connect
    GitHub too -- e.g. specifically to get private-repo cloning access via
    the stored token -- without ending up with two separate accounts."""
    with get_cursor() as cur:
        cur.execute("SELECT id FROM users WHERE github_id = %s", (github_id,))
        row = cur.fetchone()
        if row and row["id"] != user_id:
            # This GitHub account is already tied to a DIFFERENT CodeSage
            # account. Silently re-pointing it here would let user_id's
            # session start impersonating that other account's GitHub
            # identity (and clone their private repos) -- refuse instead.
            raise AuthError(
                "This GitHub account is already linked to a different CodeSage account."
            )

        encrypted = encrypt_token(access_token)
        try:
            cur.execute(
                "UPDATE users SET github_id = %s, github_login = %s, "
                "github_access_token_encrypted = %s WHERE id = %s",
                (github_id, github_login, encrypted, user_id),
            )
        except psycopg2.IntegrityError:
            # Backstop for a race between the SELECT above and this UPDATE
            # -- someone else linked the same github_id in that tiny
            # window. The UNIQUE constraint on github_id is the actual
            # source of truth; this check is just for a clearer error
            # message than a raw constraint-violation would give.
            raise AuthError(
                "This GitHub account is already linked to a different CodeSage account."
            )


def get_github_token(user_id: str) -> str | None:
    """Decrypted GitHub access token for this user, or None if they never
    connected GitHub (a password-only account) or their token was cleared.
    Called by the ingest worker (see jobs.py) right before cloning, so a
    private repo can be cloned with the owner's own GitHub credentials."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT github_access_token_encrypted FROM users WHERE id = %s", (user_id,)
        )
        row = cur.fetchone()
    if not row or not row["github_access_token_encrypted"]:
        return None
    return decrypt_token(row["github_access_token_encrypted"])


# ---------------------------------------------------------------------------
# Google OAuth accounts
#
# Deliberately simpler than the GitHub versions above: no token is stored
# (nothing here calls a Google API on the user's behalf after login, so
# there's nothing worth keeping), and email_verified is taken directly from
# Google's own claim rather than always TRUE -- Google's OIDC userinfo
# genuinely reports whether the address was itself verified, so that's
# trusted as-is instead of assuming it.
# ---------------------------------------------------------------------------

def find_or_create_google_user(google_id: str, email: str | None, email_verified: bool) -> dict:
    """Mirrors find_or_create_github_user's shape and reasoning exactly --
    see that function's docstring for why this links by provider id only,
    never by email."""
    with get_cursor() as cur:
        cur.execute("SELECT id, email FROM users WHERE google_id = %s", (google_id,))
        row = cur.fetchone()
        if row:
            return {"id": row["id"], "email": row["email"]}

        user_id = uuid.uuid4().hex
        # Google's userinfo endpoint reliably includes email when the
        # `email` scope was granted (unlike GitHub, which can omit it even
        # with user:email) -- the synthetic fallback here is defensive,
        # not the common case.
        resolved_email = email or f"{google_id}@users.noreply.google.com"
        try:
            cur.execute(
                "INSERT INTO users (id, email, google_id, email_verified, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user_id, resolved_email, google_id, email_verified, time.time()),
            )
        except psycopg2.IntegrityError:
            # Same reasoning as find_or_create_github_user: a password (or
            # GitHub) account already owns this exact email -- refuse to
            # silently merge across sign-in methods.
            raise AuthError(
                f"An account already exists for {resolved_email}. "
                "Log in that way instead, or use a different Google account."
            )

    return {"id": user_id, "email": resolved_email}


def link_google_account(user_id: str, google_id: str) -> None:
    """Attaches Google to an ALREADY-LOGGED-IN user's existing account --
    same "link" concept and same conflict handling as link_github_account
    above, just without a token to store."""
    with get_cursor() as cur:
        cur.execute("SELECT id FROM users WHERE google_id = %s", (google_id,))
        row = cur.fetchone()
        if row and row["id"] != user_id:
            raise AuthError(
                "This Google account is already linked to a different CodeSage account."
            )

        try:
            cur.execute("UPDATE users SET google_id = %s WHERE id = %s", (google_id, user_id))
        except psycopg2.IntegrityError:
            raise AuthError(
                "This Google account is already linked to a different CodeSage account."
            )


# ---------------------------------------------------------------------------
# Email verification (password signups only -- see module docstring above
# link_github_account/find_or_create_google_user for why OAuth accounts
# skip this entirely).
# ---------------------------------------------------------------------------

def create_email_verification_token(user_id: str) -> str:
    """A one-time, expiring token -- deliberately separate from the JWT
    system (create_access_token/_decode_token below), since a verification
    link needs to be single-use and short-lived, not a reusable bearer
    credential valid for JWT_EXPIRE_DAYS."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO email_verification_tokens (token, user_id, created_at, expires_at) "
            "VALUES (%s, %s, %s, %s)",
            (token, user_id, now, now + settings.email_verification_ttl_hours * 3600),
        )
    return token


def verify_email_token(token: str) -> str:
    """Marks the token's user as verified and consumes the token in the
    same transaction (get_cursor commits both statements together or
    neither -- see db.py's get_conn) -- so a token can never be used twice,
    even under a race. Returns the user_id. Raises AuthError if the token
    is unknown, already used, or expired."""
    with get_cursor() as cur:
        cur.execute(
            "DELETE FROM email_verification_tokens WHERE token = %s "
            "RETURNING user_id, expires_at",
            (token,),
        )
        row = cur.fetchone()
        if not row:
            raise AuthError("This verification link is invalid or has already been used.")
        if row["expires_at"] < time.time():
            # Deliberately still consumed (deleted) above even though
            # expired -- an expired link should never work twice either;
            # the user just needs to request a fresh one.
            raise AuthError("This verification link has expired -- request a new one.")
        cur.execute("UPDATE users SET email_verified = TRUE WHERE id = %s", (row["user_id"],))
    return row["user_id"]


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

def create_access_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.jwt_expire_days * 86400,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def _decode_token(token: str) -> str:
    """Returns the user_id encoded in a valid token, or raises AuthError."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise AuthError("Session expired -- please log in again.")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid authentication token.")
    return payload["sub"]


# ---------------------------------------------------------------------------
# FastAPI dependency -- add `current_user: dict = Depends(get_current_user)`
# as a parameter to any endpoint that should require login.
# ---------------------------------------------------------------------------

def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header. Expected: Bearer <token>")

    token = authorization[len("bearer "):].strip()
    try:
        user_id = _decode_token(token)
    except AuthError as e:
        raise HTTPException(401, str(e))

    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(401, "User no longer exists.")
    return user


# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------

def log_query(
    user_id: str,
    question: str,
    answer: str,
    repo_filter: str | None,
    citations: list[str],
    conversation_id: str | None = None,
    confidence: str | None = None,
) -> str:
    """Returns the new query_history row's id -- POST /answers/{id}/share
    (see main.py) needs it to know which history entry to snapshot into a
    permalink right after a /query call, without a second round trip to
    look it up."""
    history_id = uuid.uuid4().hex
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO query_history "
            "(id, user_id, question, answer, repo_filter, citations, created_at, "
            " conversation_id, confidence) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                history_id, user_id, question, answer, repo_filter, json.dumps(citations),
                time.time(), conversation_id, confidence,
            ),
        )
    return history_id


def get_query_history(user_id: str, limit: int = 50, conversation_id: str | None = None) -> list[dict]:
    with get_cursor() as cur:
        if conversation_id:
            cur.execute(
                "SELECT id, question, answer, repo_filter, citations, created_at, "
                "conversation_id, confidence "
                "FROM query_history WHERE user_id = %s AND conversation_id = %s "
                "ORDER BY created_at ASC LIMIT %s",
                (user_id, conversation_id, limit),
            )
        else:
            cur.execute(
                "SELECT id, question, answer, repo_filter, citations, created_at, "
                "conversation_id, confidence "
                "FROM query_history WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                (user_id, limit),
            )
        rows = cur.fetchall()

    return [
        {
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "repo_filter": r["repo_filter"],
            "citations": r["citations"],
            "created_at": r["created_at"],
            "conversation_id": r["conversation_id"],
            "confidence": r["confidence"],
        }
        for r in rows
    ]


def get_history_entry(user_id: str, history_id: str) -> dict | None:
    """A single query_history row, scoped to its owner -- used by
    POST /answers/{id}/share (see main.py) to snapshot an existing
    answer into a public permalink. Returns None (not another user's
    row) if history_id belongs to someone else, same "can't confirm it
    exists" behavior as get_job_status()."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, question, answer, repo_filter, citations, confidence "
            "FROM query_history WHERE id = %s AND user_id = %s",
            (history_id, user_id),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def conversation_history_for_llm(user_id: str, conversation_id: str, limit: int = 20) -> list[dict]:
    """Turns a conversation's past query_history rows into the flat
    [{"role", "content"}, ...] shape app/graph.py's generate_node expects
    -- each row becomes one "user" turn (the question) followed by one
    "assistant" turn (the answer), oldest first."""
    entries = get_query_history(user_id, limit=limit, conversation_id=conversation_id)
    turns: list[dict] = []
    for e in entries:
        turns.append({"role": "user", "content": e["question"]})
        turns.append({"role": "assistant", "content": e["answer"]})
    return turns