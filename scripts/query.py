"""
Usage:
    python scripts/query.py "How does the retriever fuse dense and sparse results?"
    python scripts/query.py "How does file upload work?" --repo Cloud-storage-management

If you've ingested more than one repo, --repo scopes retrieval to just that
one (matches the repo_name you gave at ingest time). Without it, retrieval
searches across every repo you've ever indexed, which can mix unrelated
chunks into one answer once you have 2+ repos in the same index.
"""
import sys

sys.path.insert(0, ".")
from app.graph import ask  # noqa: E402


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    repo_filter = None
    if "--repo" in args:
        idx = args.index("--repo")
        try:
            repo_filter = args[idx + 1]
        except IndexError:
            print("Error: --repo needs a value, e.g. --repo my-repo-name")
            sys.exit(1)
        args = args[:idx] + args[idx + 2:]

    question = " ".join(args).strip()
    if not question:
        print(__doc__)
        sys.exit(1)

    state = ask(question, repo_filter=repo_filter)

    print("\n--- Answer ---")
    print(state.get("answer", "(no answer)"))

    citations = state.get("citations", [])
    if citations:
        print("\n--- Sources ---")
        for c in citations:
            print(f"  - {c}")


if __name__ == "__main__":
    main()
