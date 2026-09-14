"""
Entry point for a CodeSage ingest worker process.

Run one (or several, for parallel ingestion) alongside the web process:

    python worker.py

Each worker pulls jobs off the same Redis queue (app/jobs.py's
"codesage-ingest" queue) that main.py's POST /ingest enqueues onto, and
runs run_ingest_job() -- the actual git clone / AST chunk / embed work
that used to happen in main.py's own threadpool.

Uses SimpleWorker rather than RQ's default Worker: the default Worker
forks a fresh child process per job (via os.fork) for isolation, which
Windows doesn't support at all. SimpleWorker runs each job in this same
process instead, which works identically on Windows, Linux, and Mac.

On top of that, RQ's job-timeout enforcement is ALSO Unix-only by
default -- it uses signal.SIGALRM, which doesn't exist on Windows either.
On Windows we swap in a no-op death penalty class (as a subclass, since
this RQ version takes death_penalty_class as a class attribute, not a
constructor argument) so a hung job just won't be force-killed at the
timeout mark; fine for local dev. On a real Linux/Mac deployment, the
normal signal-based Worker/enforcement still applies unchanged.
"""
import platform

from redis import Redis
from rq.timeouts import BaseDeathPenalty

from app.config import settings
from app.jobs import get_queue


class _NoOpDeathPenalty(BaseDeathPenalty):
    def setup_death_penalty(self):
        pass

    def cancel_death_penalty(self):
        pass


if __name__ == "__main__":
    from rq import SimpleWorker

    redis_conn = Redis.from_url(settings.redis_url)

    worker_class = SimpleWorker
    if platform.system() == "Windows":
        class WindowsSimpleWorker(SimpleWorker):
            death_penalty_class = _NoOpDeathPenalty
        worker_class = WindowsSimpleWorker

    worker = worker_class([get_queue()], connection=redis_conn)
    print(f"[worker] listening on queue '{get_queue().name}' at {settings.redis_url}")
    worker.work()