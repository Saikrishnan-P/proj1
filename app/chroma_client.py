"""
Shared ChromaDB client factory -- pluggable between a local persistent
client and a networked Chroma server, switched by an env var.

Why this needs to exist at all: app/indexer.py now runs inside RQ worker
processes (see worker.py) and app/retriever.py runs inside the web process
(main.py) -- two different processes already, and scaling to more than one
worker replica means potentially different machines/containers entirely.
chromadb.PersistentClient writes to a local disk path; if two processes
each have their own local disk, they each get their own separate, unsynced
Chroma database -- ingested-here becomes invisible queried-there. This is
the same class of problem Postgres and Redis already solved for
chunks/users and job state in the earlier stages of this migration; a
networked Chroma server is what makes it true for vectors too.

Set CHROMA_SERVER_HOST (and optionally CHROMA_SERVER_PORT / CHROMA_SERVER_SSL)
to switch to networked mode -- e.g. pointing at the official `chromadb/chroma`
Docker image, or a managed Chroma Cloud instance. Leave it unset for
local-file mode, which is what your laptop and a single-instance deploy
still use, completely unchanged from before this file existed.
"""
from __future__ import annotations

import chromadb

from app.config import settings


def get_chroma_client() -> chromadb.ClientAPI:
    if settings.chroma_server_host:
        return chromadb.HttpClient(
            host=settings.chroma_server_host,
            port=settings.chroma_server_port,
            ssl=settings.chroma_server_ssl,
        )
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)
