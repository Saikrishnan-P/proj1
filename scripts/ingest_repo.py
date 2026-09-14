"""
Ingest a repo by pasting a GitHub link, or a local folder path.

Usage:
    python scripts/ingest_repo.py                                  (prompts you to paste a link)
    python scripts/ingest_repo.py https://github.com/user/repo
    python scripts/ingest_repo.py https://github.com/user/repo my-repo-name
    python scripts/ingest_repo.py /path/to/local/repo
"""
import sys

sys.path.insert(0, ".")
from app.indexer import RepoIndexer  # noqa: E402


def main():
    if len(sys.argv) >= 2:
        source = sys.argv[1]
        repo_name = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        source = input("Paste a GitHub link (or a local repo path): ").strip()
        repo_name = None

    if not source:
        print("Nothing entered — aborting.")
        sys.exit(1)

    indexer = RepoIndexer()
    print(f"Ingesting '{source}'... (first run also downloads the embedding model, so this may take a minute)")
    try:
        result = indexer.ingest(source, repo_name=repo_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Indexed '{result['repo']}': {result['chunks_indexed']} chunks "
          f"from {result['files_seen']} files.")


if __name__ == "__main__":
    main()
