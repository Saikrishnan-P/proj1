"""
Ingestion pipeline: walk a repo -> chunk files -> embed + store in ChromaDB
-> write the same chunks into Postgres for full-text (sparse) search.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid

import chromadb
import psycopg2.extras
from chromadb.utils import embedding_functions

from app.chunker import chunk_file
from app.chroma_client import get_chroma_client
from app.config import settings
from app.db import get_cursor
from app import repos as repo_registry

DEFAULT_CLI_USER_ID = "local-cli"


def is_github_url(source: str) -> bool:
    source = source.strip()
    return bool(
        re.match(r"^(https?://|git@)", source) or source.endswith(".git")
    )


def _clone_repo(repo_url: str, github_token: str | None = None, branch: str | None = None) -> str:
    tmp_dir = os.path.join(tempfile.gettempdir(), f"codesage_{uuid.uuid4().hex[:8]}")

    # Embed the token in the clone URL for authenticated access -- this is
    # what actually makes private-repo cloning work, and it also lifts
    # GitHub's much stiffer unauthenticated rate limit for public repos.
    # Never logged/returned as-is (see the scrub below) since it's a live
    # credential, not just an identifier.
    clone_url = repo_url
    if github_token and repo_url.startswith("https://github.com/"):
        clone_url = repo_url.replace("https://github.com/", f"https://{github_token}@github.com/")

    # branch is only ever passed by the incremental (push-webhook) path,
    # so a checkout of the pushed branch's tip is used instead of
    # whatever the repo's default branch happens to be -- a push to a
    # non-default branch would otherwise be diffed against the wrong
    # commit entirely.
    cmd = ["git", "clone", "--depth", "1", clone_url, tmp_dir]
    if branch:
        cmd[4:4] = ["--branch", branch]  # insert right before clone_url

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # git's own error output echoes back the URL it tried, which would
        # otherwise leak the raw token verbatim into whatever surfaces this
        # ValueError (API error response, logs, etc).
        stderr = e.stderr.replace(github_token, "***") if github_token else e.stderr
        raise ValueError(f"git clone failed for '{repo_url}': {stderr.strip()}")
    return tmp_dir


def _current_commit_sha(repo_path: str) -> str | None:
    """Best-effort HEAD sha for whatever's checked out at repo_path --
    used to remember "what we last indexed" (see app/repos.py's
    last_commit_sha) so a later push webhook knows this repo is
    incrementally trackable. None (not an error) for a plain local
    folder that isn't a git repo at all."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _default_repo_name(source: str) -> str:
    cleaned = source.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    return os.path.basename(cleaned) or cleaned


def _iter_repo_files(repo_path: str):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in settings.ignored_dirs and not d.startswith(".")]
        for fname in files:
            if fname.endswith(settings.supported_extensions) or fname in ("README.md", "readme.md"):
                yield os.path.join(root, fname)


class RepoIndexer:
    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)  # no-op in networked mode, harmless

        self.chroma_client = get_chroma_client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=self.embedding_fn,
        )

    def ingest(
        self,
        source: str,
        repo_name: str | None = None,
        user_id: str | None = None,
        github_token: str | None = None,
    ) -> dict:
        source = source.strip()
        if not source:
            raise ValueError("Please paste a GitHub link or a local repo path.")
        user_id = user_id or DEFAULT_CLI_USER_ID

        if is_github_url(source):
            repo_name = repo_name or _default_repo_name(source)
            tmp_dir = _clone_repo(source, github_token=github_token)
            try:
                return self.index_repo(
                    tmp_dir, repo_name=repo_name, user_id=user_id, source_url=source
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if not os.path.isdir(source):
            raise ValueError(
                f"'{source}' doesn't look like a GitHub link or an existing local folder."
            )
        return self.index_repo(source, repo_name=repo_name, user_id=user_id)

    def index_repo(
        self,
        repo_path: str,
        repo_name: str | None = None,
        user_id: str | None = None,
        source_url: str | None = None,
    ) -> dict:
        repo_name = repo_name or os.path.basename(os.path.normpath(repo_path))
        user_id = user_id or DEFAULT_CLI_USER_ID
        all_docs = []
        files_seen = 0

        for file_path in _iter_repo_files(repo_path):
            files_seen += 1
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except (UnicodeDecodeError, OSError):
                continue

            rel_path = os.path.relpath(file_path, repo_path)
            for chunk in chunk_file(rel_path, source):
                doc = chunk.to_document()
                doc["metadata"]["repo"] = repo_name
                doc["metadata"]["user_id"] = user_id
                all_docs.append(doc)

        self._delete_repo_chroma(repo_name, user_id=user_id)
        self._delete_repo_chunks_table(repo_name, user_id=user_id)

        if all_docs:
            self._upsert_in_batches(all_docs)
            self._upsert_chunks_table(all_docs)

        result = {
            "repo": repo_name,
            "files_seen": files_seen,
            "chunks_indexed": len(all_docs),
        }

        # Register/refresh this (user, repo) pairing for incremental
        # re-indexing on push -- only meaningful for a real github.com
        # source (a local folder has no webhook to receive). Best-effort:
        # a registry write failing here shouldn't fail an otherwise
        # successful ingest, it just means this repo won't pick up
        # incremental updates until the next full re-ingest.
        if source_url and is_github_url(source_url):
            try:
                commit_sha = _current_commit_sha(repo_path)
                registered = repo_registry.upsert_indexed_repo(
                    user_id=user_id, repo_name=repo_name, clone_url=source_url,
                    commit_sha=commit_sha,
                )
                result["indexed_repo_id"] = registered["id"]
                result["github_owner"] = registered["owner"]
                result["github_repo"] = registered["repo"]
                result["webhook_registered"] = bool(registered["webhook_id"])
            except Exception as e:
                print(f"[indexer] Couldn't register '{repo_name}' for incremental re-indexing: {e}")

        return result

    def ingest_incremental(
        self,
        clone_url: str,
        branch: str | None,
        repo_name: str,
        user_id: str,
        github_token: str | None,
        changed_files: list[str],
        removed_files: list[str],
    ) -> dict:
        """Entry point for a webhook-triggered incremental job (see
        app/jobs.py's run_incremental_ingest_job) -- clones just enough
        of the pushed branch to read the changed files' new content,
        then hands off to incremental_index_repo() for the actual
        chunk/embed/store work. Kept separate from incremental_index_repo
        itself so that method stays testable against a plain local
        directory, with no git/network involved."""
        branch_name = (branch or "").removeprefix("refs/heads/") or None
        tmp_dir = _clone_repo(clone_url, github_token=github_token, branch=branch_name)
        try:
            result = self.incremental_index_repo(
                tmp_dir, repo_name=repo_name, user_id=user_id,
                changed_files=changed_files, removed_files=removed_files,
            )
            result["commit_sha"] = _current_commit_sha(tmp_dir)
            return result
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def incremental_index_repo(
        self,
        repo_path: str,
        repo_name: str,
        user_id: str,
        changed_files: list[str],
        removed_files: list[str],
    ) -> dict:
        """Re-chunks and re-embeds only the files a push actually
        touched, instead of the full delete-everything-and-rebuild
        index_repo() does. repo_path is a fresh shallow checkout of the
        pushed commit (see app/jobs.py's run_incremental_ingest_job) --
        changed_files/removed_files come straight from the webhook
        payload (see app/github_webhooks.py's parse_push_event), so a
        rename shows up as one remove + one add, which this handles
        correctly since both are keyed by file_path.

        Every existing chunk for a changed file is deleted before its
        new chunks are inserted (rather than upserted in place) because
        a file's chunk ids are derived from (file_path, qualified_name,
        start_line) -- a function moving to a different line, or being
        removed entirely, would otherwise leave stale chunks behind that
        upsert-by-id would never touch.
        """
        touched_files = [f for f in changed_files + removed_files if f]
        new_docs = []
        files_indexed = 0

        for rel_path in changed_files:
            abs_path = os.path.join(repo_path, rel_path)
            if not os.path.isfile(abs_path):
                continue  # deleted again by a later commit in the same push, or otherwise gone
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except OSError:
                continue

            files_indexed += 1
            for chunk in chunk_file(rel_path, source):
                doc = chunk.to_document()
                doc["metadata"]["repo"] = repo_name
                doc["metadata"]["user_id"] = user_id
                new_docs.append(doc)

        if touched_files:
            self._delete_files_chroma(repo_name, user_id, touched_files)
            self._delete_files_chunks_table(repo_name, user_id, touched_files)

        if new_docs:
            self._upsert_in_batches(new_docs)
            self._upsert_chunks_table(new_docs)

        return {
            "repo": repo_name,
            "files_changed": len(changed_files),
            "files_removed": len(removed_files),
            "files_indexed": files_indexed,
            "chunks_indexed": len(new_docs),
        }

    def list_repos(self, user_id: str | None = None) -> list[dict]:
        user_id = user_id or DEFAULT_CLI_USER_ID
        all_docs = self.collection.get(where={"user_id": user_id})
        metadatas = all_docs.get("metadatas", [])

        counts: dict[str, int] = {}
        for m in metadatas:
            repo_name = m.get("repo", "(unknown)")
            counts[repo_name] = counts.get(repo_name, 0) + 1

        return [
            {"repo": repo_name, "chunks_indexed": count}
            for repo_name, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        ]

    def delete_repo(self, repo_name: str, user_id: str | None = None) -> int:
        user_id = user_id or DEFAULT_CLI_USER_ID
        where = {"$and": [{"repo": repo_name}, {"user_id": user_id}]}
        try:
            existing = self.collection.get(where=where)
            deleted_count = len(existing.get("ids", []))
        except Exception:
            deleted_count = 0

        if deleted_count == 0:
            return 0

        self._delete_repo_chroma(repo_name, user_id=user_id)
        self._delete_repo_chunks_table(repo_name, user_id=user_id)
        return deleted_count

    def _delete_repo_chroma(self, repo_name: str, user_id: str) -> None:
        try:
            self.collection.delete(where={"$and": [{"repo": repo_name}, {"user_id": user_id}]})
        except Exception:
            pass

    def _delete_repo_chunks_table(self, repo_name: str, user_id: str) -> None:
        with get_cursor(dict_rows=False) as cur:
            cur.execute(
                "DELETE FROM chunks WHERE repo = %s AND user_id = %s",
                (repo_name, user_id),
            )

    def _delete_files_chroma(self, repo_name: str, user_id: str, file_paths: list[str]) -> None:
        try:
            self.collection.delete(
                where={"$and": [
                    {"repo": repo_name},
                    {"user_id": user_id},
                    {"file_path": {"$in": file_paths}},
                ]}
            )
        except Exception:
            pass

    def _delete_files_chunks_table(self, repo_name: str, user_id: str, file_paths: list[str]) -> None:
        with get_cursor(dict_rows=False) as cur:
            cur.execute(
                "DELETE FROM chunks WHERE repo = %s AND user_id = %s AND file_path = ANY(%s)",
                (repo_name, user_id, file_paths),
            )

    def _upsert_in_batches(self, docs: list[dict]) -> None:
        try:
            max_batch_size = self.chroma_client.get_max_batch_size()
        except Exception:
            max_batch_size = 100

        for start in range(0, len(docs), max_batch_size):
            batch = docs[start:start + max_batch_size]
            self.collection.upsert(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
            )

    def _upsert_chunks_table(self, docs: list[dict]) -> None:
        rows = [
            (
                d["id"],
                d["metadata"]["user_id"],
                d["metadata"]["repo"],
                d["metadata"]["file_path"],
                d["metadata"]["qualified_name"],
                d["metadata"]["start_line"],
                d["metadata"]["end_line"],
                d["text"],
            )
            for d in docs
        ]

        with get_cursor(dict_rows=False) as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks
                    (id, user_id, repo, file_path, qualified_name, start_line, end_line, text)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    repo = EXCLUDED.repo,
                    file_path = EXCLUDED.file_path,
                    qualified_name = EXCLUDED.qualified_name,
                    start_line = EXCLUDED.start_line,
                    end_line = EXCLUDED.end_line,
                    text = EXCLUDED.text
                """,
                rows,
            )