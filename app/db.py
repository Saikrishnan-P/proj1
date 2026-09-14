"""
Shared Postgres connection pool + schema for CodeSage.

Stage 1 of the scale-out migration: this replaces two things that used to
be fragile local files —

  - the SQLite users/query_history DB (app/auth.py used to open
    ./data/users.db directly with sqlite3)
  - the BM25 pickle (app/indexer.py used to pickle.dump a rank_bm25 index
    to ./data/bm25_index.pkl, app/retriever.py unpickled it on every query)

— with tables in one Postgres instance, so every replica/worker (including
the RQ workers coming in stage 2) reads and writes the same data instead of
each process having its own local file. Chroma (dense vectors) is
unchanged here; that's stage 3 (pluggable Chroma client).

Sparse search is now Postgres full-text search instead of BM25: a
GENERATED tsvector column + a GIN index over it, ranked with ts_rank_cd.
We use the 'simple' text-search config (no English stemming/stopwords) on
purpose -- code identifiers aren't English prose, and stemming would drop
or mangle exactly the exact-match tokens the old regex-based BM25
tokenizer was there to catch.

Set DATABASE_URL to point this at your Postgres instance (see
.env.example / docker-compose.yml for a local one).
"""
from __future__ import annotations

from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from app.config import settings

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(1, 10, dsn=settings.database_url)
    return _pool


@contextmanager
def get_conn():
    """Borrow a connection from the pool. Commits on a clean exit, rolls
    back on an exception, always returns the connection to the pool
    afterwards -- same shape as the old SQLite `_db()` helper in auth.py,
    so callers barely change."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def get_cursor(dict_rows: bool = True):
    """Convenience wrapper around get_conn() for the common case of "one
    cursor, one or more statements". dict_rows=True (the default) gives
    you row["col"] access via RealDictCursor, matching how auth.py used to
    read sqlite3.Row objects."""
    with get_conn() as conn:
        cursor_factory = psycopg2.extras.RealDictCursor if dict_rows else None
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur


def init_db() -> None:
    """Creates every table/index this app needs, if they don't already
    exist. Called once at startup (see main.py) -- same call site and
    same idempotent "safe to call on every boot" contract the old
    auth.init_db() had, just backed by Postgres DDL now."""
    with get_cursor(dict_rows=False) as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL
            )
        """)

        # GitHub OAuth ("Continue with GitHub") support -- additive columns
        # plus relaxing password_hash/password_salt to nullable, since a
        # GitHub-only account never sets a password at all. ADD COLUMN IF
        # NOT EXISTS and DROP NOT NULL (on an already-nullable column) are
        # both no-ops on a second run, same idempotent "safe on every boot"
        # contract as the CREATE TABLE IF NOT EXISTS calls throughout this
        # function -- no separate migration tool/step needed.
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS github_id BIGINT UNIQUE")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS github_login TEXT")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS github_access_token_encrypted TEXT")
        cur.execute("ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL")
        cur.execute("ALTER TABLE users ALTER COLUMN password_salt DROP NOT NULL")

        # Google OAuth ("Continue with Google") -- same additive-column
        # pattern as GitHub above. google_id is TEXT, not BIGINT: Google's
        # `sub` claim is a numeric-looking string but isn't documented or
        # guaranteed to fit in a 64-bit int the way GitHub's user id is.
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id TEXT UNIQUE")

        # Email verification for password signups (OAuth accounts are
        # marked verified immediately at creation -- see auth.py's
        # find_or_create_github_user / find_or_create_google_user).
        cur.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE"
        )
        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at DOUBLE PRECISION NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                repo_filter TEXT,
                citations JSONB NOT NULL,
                created_at DOUBLE PRECISION NOT NULL
            )
        """)
        # user_id here is deliberately NOT a foreign key to users(id).
        # Since the API platform (app/api_keys.py, app/billing.py), this
        # column is a polymorphic principal id -- either a users.id (a
        # logged-in web user) or an api_customers.id (a third-party
        # developer authenticating with an API key; see main.py's
        # get_principal()) -- and Postgres has no single-column FK that
        # can point at either table. Referential integrity here is
        # enforced at the application layer instead (every write to this
        # column goes through auth.log_query, called only after
        # get_principal has already validated the id belongs to one of
        # the two). DROP CONSTRAINT IF EXISTS is a no-op on a fresh
        # database (the CREATE TABLE above never adds this constraint on
        # a first run) and only does real work against an existing
        # database created before this column stopped being a strict FK.
        cur.execute("ALTER TABLE query_history DROP CONSTRAINT IF EXISTS query_history_user_id_fkey")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_user "
            "ON query_history(user_id, created_at DESC)"
        )

        # conversation_id: NULL for a one-off /query call with no
        # conversation attached (still fully supported, unchanged
        # behavior); set once multi-turn conversations exist (see the
        # `conversations` table below). confidence: the verify node's
        # own read on the answer -- "high" / "medium" / "low" -- kept as
        # plain text history a user can filter/sort on, not just baked
        # into the answer string.
        cur.execute("ALTER TABLE query_history ADD COLUMN IF NOT EXISTS conversation_id TEXT")
        cur.execute("ALTER TABLE query_history ADD COLUMN IF NOT EXISTS confidence TEXT")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_conversation "
            "ON query_history(conversation_id, created_at)"
        )

        # ---------------------------------------------------------------
        # Multi-turn conversations. query_history above is still the
        # source of truth for "everything this user has ever asked" (the
        # existing GET /history endpoint keeps working untouched); this
        # table exists so a *sequence* of turns can be grouped, named,
        # and replayed as context for the next question in the same
        # thread (see app/graph.py's generate_node, which reads recent
        # turns for a conversation_id straight out of query_history).
        # ---------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                repo_filter TEXT,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )
        """)
        # Same polymorphic-principal reasoning as query_history above --
        # user_id may be an api_customers.id now, not just a users.id.
        cur.execute("ALTER TABLE conversations DROP CONSTRAINT IF EXISTS conversations_user_id_fkey")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user "
            "ON conversations(user_id, updated_at DESC)"
        )

        # ---------------------------------------------------------------
        # Shareable answer permalinks. Deliberately a copy of the answer
        # at share time (question/answer/citations/confidence), not a
        # foreign key to query_history -- a shared link should keep
        # working and showing the same content even if the user later
        # deletes that history entry or the underlying repo, and it must
        # never expose which user_id asked it to the public viewer (see
        # GET /share/{id} in main.py, which intentionally omits user_id
        # from its response).
        # ---------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shared_answers (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations JSONB NOT NULL,
                confidence TEXT,
                repo_filter TEXT,
                created_at DOUBLE PRECISION NOT NULL
            )
        """)
        # Same polymorphic-principal reasoning as query_history above --
        # owner_user_id may be an api_customers.id now, not just a users.id.
        cur.execute("ALTER TABLE shared_answers DROP CONSTRAINT IF EXISTS shared_answers_owner_user_id_fkey")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_shared_owner "
            "ON shared_answers(owner_user_id, created_at DESC)"
        )

        # ---------------------------------------------------------------
        # One row per (user, ingested repo) that came from a real GitHub
        # URL (local-folder ingests never get a row here -- there's no
        # webhook to receive for those). Lets a push webhook find every
        # user who has this exact GitHub repo indexed, and lets the
        # incremental-ingest job know what commit it last saw and how to
        # authenticate a `git fetch` against that repo again. See
        # app/github_webhooks.py and app/indexer.py's
        # incremental_index_repo().
        # ---------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS indexed_repos (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                repo_name TEXT NOT NULL,
                clone_url TEXT NOT NULL,
                github_owner TEXT,
                github_repo TEXT,
                last_commit_sha TEXT,
                local_clone_path TEXT,
                webhook_id BIGINT,
                webhook_secret TEXT,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                UNIQUE (user_id, repo_name)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_indexed_repos_owner_repo "
            "ON indexed_repos(github_owner, github_repo)"
        )
        # Same polymorphic-principal reasoning as query_history above --
        # user_id may be an api_customers.id now, not just a users.id
        # (an API customer's GitHub-sourced ingest still registers a
        # webhook row here, same as a logged-in user's).
        cur.execute("ALTER TABLE indexed_repos DROP CONSTRAINT IF EXISTS indexed_repos_user_id_fkey")

        # One row per chunk, mirroring what used to go into the BM25
        # pickle's `docs` list. `text` is the same chunk text that's also
        # embedded into Chroma for dense search -- kept here too because
        # full-text search needs the raw text to build/query the tsvector
        # against, not just an id.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                repo TEXT NOT NULL,
                file_path TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv)"
        )
        # Every chunk lookup/delete in indexer.py and retriever.py filters
        # by (user_id, repo) first -- this index makes that filter cheap
        # before Postgres even has to touch the tsvector for ranking.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_repo_user "
            "ON chunks(user_id, repo)"
        )

        # ---------------------------------------------------------------
        # API platform (usage-based billing for third-party developers).
        # An api_customer is a completely separate principal from `users`
        # above -- orgs/developers who want RAG-over-code without a login
        # of their own, authenticating with an API key instead of a JWT.
        # Deliberately its own table, not a flag on `users`: the two have
        # almost nothing in common (no email/password, no GitHub OAuth,
        # no web session) and giving api_customers its own id namespace
        # means app/api_keys.py and app/billing.py never have to worry
        # about colliding with a `users.id`. See app/main.py's
        # get_principal(), which accepts EITHER a user JWT OR an API key
        # and passes whichever id it resolves to as `user_id` into the
        # existing indexer/retriever code -- repo/chunk scoping is
        # already keyed by an opaque user_id string, so an api_customer
        # is indistinguishable from a regular user at that layer.
        # ---------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_customers (
                id TEXT PRIMARY KEY,
                org_name TEXT NOT NULL,
                email TEXT NOT NULL,
                credit_balance BIGINT NOT NULL DEFAULT 0,
                created_at DOUBLE PRECISION NOT NULL
            )
        """)

        # Only key_hash is ever stored -- the raw key is shown exactly
        # once, at creation time, same as e.g. Stripe/GitHub PAT UX. A
        # sha256 hash (not PBKDF2) is intentional and fine here: unlike a
        # user-chosen password, an API key is already a high-entropy
        # random token (see app/api_keys.py's generate_api_key), so
        # there's no low-entropy-guessing risk a slow hash defends
        # against -- and a fast hash is what makes "look this key up by
        # its hash" cheap on every request.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                api_customer_id TEXT NOT NULL REFERENCES api_customers(id) ON DELETE CASCADE,
                key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL,
                label TEXT,
                created_at DOUBLE PRECISION NOT NULL,
                last_used_at DOUBLE PRECISION,
                revoked_at DOUBLE PRECISION
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_api_keys_customer ON api_keys(api_customer_id)"
        )

        # Append-only audit ledger backing api_customers.credit_balance.
        # The balance column is the fast path every request checks;
        # this table exists so that number is always explainable after
        # the fact (a purchase, a query debit, a manual grant, etc.)
        # rather than being an opaque counter. delta is signed: positive
        # for a grant/purchase, negative for a debit.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id TEXT PRIMARY KEY,
                api_customer_id TEXT NOT NULL REFERENCES api_customers(id) ON DELETE CASCADE,
                delta BIGINT NOT NULL,
                reason TEXT NOT NULL,
                reference TEXT,
                created_at DOUBLE PRECISION NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_credit_tx_customer "
            "ON credit_transactions(api_customer_id, created_at DESC)"
        )
        # A Stripe checkout.session.completed webhook can be delivered
        # more than once (Stripe's own retry policy, or just a flaky
        # network) -- this makes granting credits for a given Stripe
        # session idempotent: the second delivery's INSERT conflicts and
        # is ignored, rather than crediting the purchase twice. See
        # app/billing.py's fulfill_checkout_session().
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_tx_reference "
            "ON credit_transactions(reference) WHERE reference IS NOT NULL"
        )