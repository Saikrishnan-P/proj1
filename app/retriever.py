"""
Hybrid retrieval: dense (ChromaDB embeddings) + sparse (Postgres full-text
search), combined with Reciprocal Rank Fusion.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from app.chroma_client import get_chroma_client
from app.config import settings
from app.db import get_cursor


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def citation(self) -> str:
        m = self.metadata
        return f"{m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')} ({m.get('qualified_name')})"


class HybridRetriever:
    def __init__(self):
        self.chroma_client = get_chroma_client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=self.embedding_fn,
        )

    def _dense_search(self, query: str, k: int, repo_filter: str | None, user_id: str | None) -> list[str]:
        conditions = []
        if repo_filter:
            conditions.append({"repo": repo_filter})
        if user_id:
            conditions.append({"user_id": user_id})

        if not conditions:
            where = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        results = self.collection.query(query_texts=[query], n_results=k, where=where)
        return results["ids"][0] if results["ids"] else []

    def _sparse_search(self, query: str, k: int, repo_filter: str | None, user_id: str | None) -> list[str]:
        clauses = ["tsv @@ websearch_to_tsquery('simple', %(query)s)"]
        params: dict = {"query": query, "k": k}
        if repo_filter:
            clauses.append("repo = %(repo)s")
            params["repo"] = repo_filter
        if user_id:
            clauses.append("user_id = %(user_id)s")
            params["user_id"] = user_id

        sql = f"""
            SELECT id
            FROM chunks
            WHERE {' AND '.join(clauses)}
            ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('simple', %(query)s)) DESC
            LIMIT %(k)s
        """

        with get_cursor(dict_rows=False) as cur:
            cur.execute(sql, params)
            return [row[0] for row in cur.fetchall()]

    def retrieve(
        self,
        query: str,
        repo_filter: str | None = None,
        user_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_final

        dense_ids = self._dense_search(query, settings.top_k_dense, repo_filter, user_id)
        sparse_ids = self._sparse_search(query, settings.top_k_sparse, repo_filter, user_id)

        fused_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}

        for rank, doc_id in enumerate(dense_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            dense_ranks[doc_id] = rank + 1

        for rank, doc_id in enumerate(sparse_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            sparse_ranks[doc_id] = rank + 1

        top_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)[:top_k]
        if not top_ids:
            return []

        fetched = self.collection.get(ids=top_ids)
        by_id = {
            fetched["ids"][i]: (fetched["documents"][i], fetched["metadatas"][i])
            for i in range(len(fetched["ids"]))
        }

        results = []
        for doc_id in top_ids:
            if doc_id not in by_id:
                continue
            text, metadata = by_id[doc_id]
            results.append(RetrievedChunk(
                id=doc_id,
                text=text,
                metadata=metadata,
                dense_rank=dense_ranks.get(doc_id),
                sparse_rank=sparse_ranks.get(doc_id),
                rrf_score=fused_scores[doc_id],
            ))
        return results