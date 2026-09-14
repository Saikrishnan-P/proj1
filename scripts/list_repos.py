"""
Usage:
    python scripts/list_repos.py

Prints every distinct repo name currently in the index, with a chunk count
for each — useful for getting the exact spelling to pass to
`python scripts/query.py "..." --repo <name>`.
"""
import sys
from collections import Counter

sys.path.insert(0, ".")
from app.retriever import HybridRetriever  # noqa: E402


def main():
    retriever = HybridRetriever()
    all_docs = retriever.collection.get()
    metadatas = all_docs.get("metadatas", [])

    if not metadatas:
        print("No repos indexed yet. Run scripts/ingest_repo.py first.")
        return

    counts = Counter(m.get("repo", "(unknown)") for m in metadatas)

    print("Indexed repos:")
    for repo_name, count in counts.most_common():
        print(f"  - {repo_name}  ({count} chunks)")


if __name__ == "__main__":
    main()
