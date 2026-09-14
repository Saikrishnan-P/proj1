"""
Redis + RQ job queue for repo ingestion.

Replaces the old in-process `_jobs` dict + `threading`/`BackgroundTasks`
setup in main.py: POST /ingest used to hand work to this same process's
threadpool, which meant an in-flight job was silently lost on every
restart/redeploy, and the job status dict never existed anywhere another
replica could see it. Now the web process just enqueues a job (this
module's enqueue_ingest) and returns immediately; a separate worker
process (see worker.py, run via `rq worker` or Railway's Procfile) picks
it up from Redis and runs it. Status/result live in Redis via RQ's own Job
object, so any web replica can poll get_job_status() and see the same
answer, and a job survives a web-process restart -- only losing a worker
mid-job loses it now, same class of risk any queue has.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from redis import Redis
from rq import Queue
from rq.job import Job
from rq.job import JobStatus as RQJobStatus

from app import auth, billing, github_webhooks
from app import repos as repo_registry
from app.config import settings
from app.indexer import RepoIndexer

_redis: Redis | None = None
_queue: Queue | None = None
_indexer: RepoIndexer | None = None

QUEUE_NAME = "codesage-ingest"


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url)
    return _redis


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(QUEUE_NAME, connection=get_redis())
    return _queue


def _get_indexer() -> RepoIndexer:
    """RepoIndexer.__init__ loads the sentence-transformers embedding
    model, the slow part to construct -- build it once and reuse it across
    every job THIS WORKER PROCESS handles. Separate from (and unrelated
    to) any indexer instance the web process keeps for /repos -- workers
    and the web server are different processes now, each with their own
    singleton."""
    global _indexer
    if _indexer is None:
        _indexer = RepoIndexer()
    return _indexer


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


# Maps RQ's own status enum onto the same four-value vocabulary the API
# already returns (main.py's IngestStatusResponse/pollIngest on the
# frontend expect exactly "pending" | "running" | "done" | "error") --
# QUEUED/DEFERRED/SCHEDULED are all "still waiting" from the caller's
# point of view, and STOPPED/CANCELED are folded into "error" since
# neither ever produces a result.
_RQ_STATUS_MAP = {
    RQJobStatus.QUEUED: JobStatus.PENDING,
    RQJobStatus.DEFERRED: JobStatus.PENDING,
    RQJobStatus.SCHEDULED: JobStatus.PENDING,
    RQJobStatus.STARTED: JobStatus.RUNNING,
    RQJobStatus.FINISHED: JobStatus.DONE,
    RQJobStatus.FAILED: JobStatus.ERROR,
    RQJobStatus.STOPPED: JobStatus.ERROR,
    RQJobStatus.CANCELED: JobStatus.ERROR,
}


def run_ingest_job(source: str, repo_name: str | None, user_id: str, billed: bool = False) -> dict:
    """The actual ingestion work -- this is what a worker process
    (worker.py) executes for a queued job. Returning a dict here is what
    RQ stores as job.result once it marks the job FINISHED; raising marks
    it FAILED, with the traceback captured in job.exc_info (surfaced via
    get_job_status()'s "error" field below).

    Looks up the user's GitHub token (if they connected one via OAuth)
    fresh from Postgres right here, at execution time, rather than
    accepting it as a parameter passed in from enqueue_ingest() below --
    RQ persists a job's arguments in Redis for the job's result_ttl (a
    full day here), so threading a live credential through as an argument
    would mean a plaintext GitHub token sitting in Redis for that whole
    window. Looking it up by user_id, decrypting only in-memory in this
    worker process, avoids that entirely.

    billed=True means user_id is actually an api_customer_id (see
    app/main.py's POST /ingest, which sets this whenever the caller
    authenticated with an API key rather than a login). Credits are
    deducted here, AFTER ingestion, because the true cost
    (billing.ingest_credit_cost) depends on chunks_indexed, which isn't
    known until the work is done -- see deduct_credits' allow_negative
    docstring for why that deduction is allowed to push the balance
    negative in the rare large-repo case, rather than refusing to record
    the usage that already happened."""
    github_token = auth.get_github_token(user_id)
    result = _get_indexer().ingest(
        source, repo_name=repo_name, user_id=user_id, github_token=github_token
    )

    if billed:
        cost = billing.ingest_credit_cost(result.get("chunks_indexed", 0))
        try:
            new_balance = billing.deduct_credits(
                user_id, cost, reason=f"ingest:{result.get('repo')}", allow_negative=True
            )
            result["credits_charged"] = cost
            result["credit_balance"] = new_balance
        except billing.BillingError as e:
            # api_customer row vanished mid-job (shouldn't happen -- FK
            # cascade would also have deleted this job's own queued
            # state) -- log and move on rather than failing an otherwise
            # successful ingest over a billing bookkeeping error.
            print(f"[jobs] Couldn't charge ingest for '{user_id}': {e}")

    # Best-effort: register a push webhook for incremental re-indexing,
    # now that we know this ingest was a real github.com source (see
    # RepoIndexer.index_repo -- it only sets these keys when it is) and
    # we have this user's GitHub token in hand right here in the worker
    # process. Skipped silently (see github_webhooks.py's docstring) if
    # there's no token, no BACKEND_PUBLIC_URL configured, or GitHub
    # refuses (e.g. not an admin on that repo) -- none of that should
    # affect the ingest result the caller already has.
    owner, repo = result.get("github_owner"), result.get("github_repo")
    if owner and repo and github_token and not result.get("webhook_registered"):
        try:
            registered = github_webhooks.register_webhook_if_possible(
                owner, repo, github_token, result["indexed_repo_id"]
            )
            if registered:
                webhook_id, webhook_secret = registered
                repo_registry.set_webhook(result["indexed_repo_id"], webhook_id, webhook_secret)
                result["webhook_registered"] = True
        except Exception as e:
            print(f"[jobs] Webhook registration for {owner}/{repo} failed: {e}")

    return result


def run_incremental_ingest_job(indexed_repo_id: str, push_event: dict) -> dict:
    """What a worker runs for a webhook-triggered incremental job (see
    enqueue_incremental_ingest below and app/main.py's
    POST /webhooks/github/{indexed_repo_id}). Looks up the indexed_repos
    row fresh here (same "don't thread a live token through Redis as a
    job argument" reasoning as run_ingest_job above -- github_token is
    fetched by user_id at execution time, not passed in)."""
    row = repo_registry.get_indexed_repo_by_id(indexed_repo_id)
    if row is None:
        raise ValueError(f"No indexed_repos row for id '{indexed_repo_id}' -- was it deleted?")

    github_token = auth.get_github_token(row["user_id"])
    result = _get_indexer().ingest_incremental(
        clone_url=row["clone_url"],
        branch=push_event.get("ref"),
        repo_name=row["repo_name"],
        user_id=row["user_id"],
        github_token=github_token,
        changed_files=push_event.get("changed_files", []),
        removed_files=push_event.get("removed_files", []),
    )

    commit_sha = result.get("commit_sha") or push_event.get("after_sha")
    if commit_sha:
        repo_registry.update_last_commit_sha(indexed_repo_id, commit_sha)

    return result


def enqueue_incremental_ingest(indexed_repo_id: str, push_event: dict) -> str:
    job = get_queue().enqueue(
        run_incremental_ingest_job,
        indexed_repo_id,
        push_event,
        job_timeout="10m",  # a handful of changed files, not a whole repo -- much faster than a full ingest
        result_ttl=86400,
        failure_ttl=86400,
        meta={"indexed_repo_id": indexed_repo_id, "incremental": True},
    )
    return job.id


def enqueue_ingest(source: str, repo_name: str | None, user_id: str, billed: bool = False) -> str:
    """Queues an ingest job and returns its id right away -- the RQ
    equivalent of the old background_tasks.add_task() call in main.py.
    job_timeout is generous (cloning + AST-chunking + CPU embeddings on a
    real-sized repo can take minutes, same reasoning main.py's docstring
    always had for why this is a background job at all). meta stores the
    request fields RQ doesn't track natively, keyed by user_id so
    get_job_status() can enforce the same per-user ownership check the old
    _jobs dict did. billed is threaded straight through to run_ingest_job
    -- see that function's docstring for why the actual credit deduction
    happens post-ingest rather than here."""
    job = get_queue().enqueue(
        run_ingest_job,
        source,
        repo_name,
        user_id,
        billed,
        job_timeout="30m",
        result_ttl=86400,  # keep the result pollable for a day after it finishes
        failure_ttl=86400,
        meta={"source": source, "repo_name": repo_name, "user_id": user_id},
    )
    return job.id


def get_job_status(job_id: str, user_id: str) -> dict | None:
    """Fetches a job's current status/result from Redis. Returns None if
    the job doesn't exist, has expired (past result_ttl/failure_ttl), OR
    belongs to a different user -- same "can't tell the difference"
    behavior main.py's 404 branch relied on before, so a guessed job_id
    still can't be used to confirm another user's job exists."""
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:
        return None

    if job.meta.get("user_id") != user_id:
        return None

    status = _RQ_STATUS_MAP.get(job.get_status(refresh=True), JobStatus.ERROR)

    return {
        "job_id": job.id,
        "status": status,
        "source": job.meta.get("source"),
        "repo_name": job.meta.get("repo_name"),
        "result": job.result if status == JobStatus.DONE else None,
        "error": _format_error(job) if status == JobStatus.ERROR else None,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.ended_at),
    }


def _format_error(job: Job) -> str:
    """job.exc_info is the full worker-side traceback -- useful in worker
    logs, too noisy/leaky to hand back verbatim over the API. The last
    non-empty line is normally the exception's own message (e.g. the
    ValueError text index.py's ingest() raises), which is what the old
    in-process version's `str(e)` used to surface."""
    if not job.exc_info:
        return "Ingestion failed."
    lines = [ln for ln in job.exc_info.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "Ingestion failed."


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
