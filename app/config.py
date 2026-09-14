"""
Central configuration for CodeSage.

All values can be overridden via environment variables (see .env.example).
"""
import os
import secrets
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env into the process environment BEFORE any os.getenv() calls below
# run -- without this, .env is never actually read and every setting silently
# falls back to its default (this is why GROQ_API_KEY was coming back empty).
load_dotenv()

# Disable ChromaDB's anonymous telemetry. Harmless when it fails (just a
# noisy "Failed to send telemetry event" line), but distracting in a live
# demo. Set here in config.py -- not indexer.py -- because both the ingest
# path (via indexer.py) and the query path (via retriever.py) import this
# module first, so this is the one place that covers both.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# JWT_SECRET_KEY MUST be set explicitly in production (Railway's Variables
# tab, your local .env, etc.). Without it, a random secret is generated here
# instead -- the app still boots, but every previously issued login token
# silently stops working on the next restart/redeploy (a random secret means
# a new one every process start), forcing everyone to log in again. Fine for
# a first local test; not fine to leave unset once this is actually deployed.
_env_jwt_secret = os.getenv("JWT_SECRET_KEY", "")
if not _env_jwt_secret:
    print(
        "[config] WARNING: JWT_SECRET_KEY is not set. Using a random secret "
        "for this process only -- all existing login tokens will be "
        "invalidated on the next restart. Set JWT_SECRET_KEY to fix this."
    )
    _env_jwt_secret = secrets.token_hex(32)

# DATABASE_URL MUST be set in production too -- there's no local-file
# fallback anymore (see app/db.py). Falls back to the docker-compose.yml
# Postgres service for local dev, since that's what `docker compose up`
# next to this repo gives you out of the box.
_env_database_url = os.getenv(
    "DATABASE_URL", "postgresql://codesage:codesage@localhost:5432/codesage"
)
if not os.getenv("DATABASE_URL"):
    print(
        "[config] WARNING: DATABASE_URL is not set. Falling back to "
        "postgresql://codesage:codesage@localhost:5432/codesage (the "
        "docker-compose.yml default) -- set DATABASE_URL explicitly once "
        "this is deployed anywhere but your laptop."
    )

# REDIS_URL -- same "no local fallback in production" story as DATABASE_URL
# above. Used by app/jobs.py (the web process enqueueing ingest jobs) and
# worker.py (the separate process picking them up); both must point at the
# same Redis or jobs enqueued by one are never seen by the other.
_env_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
if not os.getenv("REDIS_URL"):
    print(
        "[config] WARNING: REDIS_URL is not set. Falling back to "
        "redis://localhost:6379/0 (the docker-compose.yml default) -- set "
        "REDIS_URL explicitly once this is deployed anywhere but your "
        "laptop."
    )


@dataclass
class Settings:
    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Vector store (dense). Local-file Chroma client by default (fine for
    # your laptop or a single-instance deploy). Set CHROMA_SERVER_HOST to
    # switch to a networked Chroma server instead -- required once you run
    # more than one worker/web replica, since a local PersistentClient on
    # each machine's own disk means ingest-here is invisible query-there.
    # See app/chroma_client.py.
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION", "codesage_chunks")
    chroma_server_host: str = os.getenv("CHROMA_SERVER_HOST", "")
    chroma_server_port: int = int(os.getenv("CHROMA_SERVER_PORT", "8000"))
    chroma_server_ssl: bool = os.getenv("CHROMA_SERVER_SSL", "false").lower() == "true"

    # Postgres -- users, query_history, and chunks (full-text/sparse search)
    # all live here now. See app/db.py for the schema. Replaces the old
    # users_db_path (SQLite) and bm25_index_path (pickle) settings.
    database_url: str = _env_database_url

    # Redis -- backs the RQ ingest job queue (app/jobs.py, worker.py), and
    # also the short-lived GitHub OAuth CSRF state (app/main.py) so that
    # state check works correctly even with more than one web replica.
    redis_url: str = _env_redis_url

    # Chunking
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "1500"))
    min_chunk_chars: int = int(os.getenv("MIN_CHUNK_CHARS", "40"))

    # Retrieval
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "10"))
    top_k_sparse: int = int(os.getenv("TOP_K_SPARSE", "10"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "6"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))  # standard RRF damping constant

    # Generation (Groq, matching kk's existing stack)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Auth (see app/auth.py) -- JWT bearer tokens, backed by the `users`
    # table in Postgres (database_url above).
    jwt_secret_key: str = _env_jwt_secret
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = int(os.getenv("JWT_EXPIRE_DAYS", "7"))

    # GitHub OAuth ("Continue with GitHub") -- see app/github_oauth.py.
    # Register an OAuth App at https://github.com/settings/developers with
    # callback URL {your backend url}/auth/github/callback.
    github_client_id: str = os.getenv("GITHUB_CLIENT_ID", "")
    github_client_secret: str = os.getenv("GITHUB_CLIENT_SECRET", "")
    github_redirect_uri: str = os.getenv(
        "GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback"
    )

    # Google OAuth ("Continue with Google") -- see app/google_oauth.py.
    # Authentication only: unlike GitHub, no token is stored after login,
    # since nothing here calls a Google API on the user's behalf later
    # (there's no Google equivalent of "clone a private repo"). Register
    # credentials at https://console.cloud.google.com/apis/credentials
    # with authorized redirect URI {your backend url}/auth/google/callback.
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    google_redirect_uri: str = os.getenv(
        "GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback"
    )

    # Where to redirect the browser back to after a successful GitHub or
    # Google login, with the issued JWT attached as a query param.
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Publicly reachable base URL of THIS backend (no trailing slash) --
    # needed for exactly one thing: telling GitHub where to POST push
    # webhooks when we auto-register one after a GitHub-sourced ingest
    # (see app/github_webhooks.py). GITHUB_REDIRECT_URI already implies
    # this same host, but it's the callback *path*, not the bare origin,
    # so it's kept as its own setting rather than string-munged out of
    # that one. Leave unset (default) to simply skip webhook
    # registration -- incremental re-indexing then just never triggers
    # for that repo, which is a safe, silent no-op, not an error.
    backend_public_url: str = os.getenv("BACKEND_PUBLIC_URL", "")

    # How many previous turns (question+answer pairs) from a conversation
    # get replayed to the LLM as context on a follow-up question (see
    # app/graph.py's generate_node). Kept small on purpose -- Groq's
    # context window is generous, but a long history is mostly noise for
    # THIS kind of Q&A, since each turn is answered fresh from newly
    # retrieved code, not from the conversation itself.
    conversation_history_turns: int = int(os.getenv("CONVERSATION_HISTORY_TURNS", "4"))

    # Symmetric key used to encrypt GitHub access tokens at rest (see
    # auth.py's encrypt_token/decrypt_token) -- generate one with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # MUST be set before any GitHub login succeeds in production; unlike
    # JWT_SECRET_KEY there's no random per-process fallback here, because a
    # fallback that changes on every restart would mean already-stored
    # tokens become silently undecryptable garbage the moment the process
    # restarts -- better to fail loudly at first use (see auth.py) than
    # corrupt stored tokens.
    token_encryption_key: str = os.getenv("TOKEN_ENCRYPTION_KEY", "")

    # -----------------------------------------------------------------
    # Email verification (see app/email_sender.py, app/auth.py's
    # create_email_verification_token/verify_email_token). Only applies
    # to password signups (POST /auth/signup) -- GitHub and Google OAuth
    # accounts are marked verified immediately, since successfully
    # completing that provider's login already proves control of the
    # account, without needing our own separate email loop on top.
    # -----------------------------------------------------------------

    # How long a verification link stays valid before someone has to
    # request a new one.
    email_verification_ttl_hours: int = int(os.getenv("EMAIL_VERIFICATION_TTL_HOURS", "24"))

    # Plain SMTP -- works with Gmail (an app password), SendGrid, Mailgun,
    # AWS SES, or any other provider that exposes an SMTP relay, without
    # pulling in a provider-specific SDK. Leave SMTP_HOST unset to skip
    # actually sending mail (the link is printed to the server log
    # instead) -- lets signup/login work end-to-end in local dev without
    # needing real email credentials.
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from_email: str = os.getenv("SMTP_FROM_EMAIL", "no-reply@codesage.local")
    smtp_use_tls: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    # Repo ingestion
    # AST chunking applies to: .py (Python's own `ast` module), .ipynb
    # (parsed cell-by-cell, with code cells re-using the .py AST chunker --
    # see chunk_notebook_file() in chunker.py), and
    # .js/.jsx/.ts/.tsx/.java/.html/.htm/.css/.scss (tree-sitter grammars,
    # see ts_chunker.py). Everything else here still gets walked and
    # indexed via the plain-text sliding-window fallback in chunker.py.
    supported_extensions: tuple = (
        ".py",
        ".ipynb",
        ".ts", ".tsx",
        ".js", ".jsx",
        ".java",
        ".html", ".htm",
        ".css", ".scss",
        ".json", ".md",
    )
    ignored_dirs: tuple = (
        ".git", "__pycache__", "node_modules", "venv", ".venv",
        "dist", "build", ".next", "coverage",
    )

    # -----------------------------------------------------------------
    # API platform -- usage-based billing for third-party developers
    # calling /ingest and /query directly with an API key (see
    # app/api_keys.py, app/billing.py). Logged-in web users (app/auth.py)
    # are never charged; only requests authenticated via X-API-Key are.
    # -----------------------------------------------------------------

    # Flat cost of one /query or /query/stream call. Deliberately not
    # metered by tokens: a customer integrating against this API needs
    # to be able to estimate their bill from call volume alone, without
    # having to model retry/verify-loop token variance -- see
    # app/graph.py's verify_node, which can silently double generation
    # cost on a hallucination retry. That variance is CodeSage's cost to
    # absorb, not something to pass through per-call.
    query_credit_cost: int = int(os.getenv("QUERY_CREDIT_COST", "1"))

    # Ingest cost scales with repo size (real embedding compute), but is
    # still a flat multiple rather than literally 1 credit/chunk, so a
    # customer can roughly guess "a repo this size will cost about this
    # much" before ingesting. See app/billing.py's ingest_credit_cost().
    ingest_credit_cost_per_chunks: int = int(os.getenv("INGEST_CREDIT_COST_PER_CHUNKS", "100"))
    ingest_credit_cost_minimum: int = int(os.getenv("INGEST_CREDIT_COST_MINIMUM", "1"))

    # Stripe -- prepaid credit purchases only (Checkout, one-time
    # payments), not subscriptions or usage-based metered billing. See
    # app/billing.py. Get these from https://dashboard.stripe.com/apikeys
    # and https://dashboard.stripe.com/webhooks (endpoint:
    # {BACKEND_PUBLIC_URL}/webhooks/stripe, event: checkout.session.completed).
    stripe_secret_key: str = os.getenv("STRIPE_SECRET_KEY", "")
    stripe_webhook_secret: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    # Where Stripe Checkout sends the customer's browser after payment
    # (success) or if they back out (cancel). Point these at pages your
    # developer-facing dashboard actually serves.
    billing_success_url: str = os.getenv(
        "BILLING_SUCCESS_URL", "http://localhost:3000/billing/success"
    )
    billing_cancel_url: str = os.getenv(
        "BILLING_CANCEL_URL", "http://localhost:3000/billing/cancel"
    )


settings = Settings()
