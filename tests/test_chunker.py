"""
Tests that don't require ChromaDB/sentence-transformers/Groq to be
installed or reachable — pure logic checks for chunking and RRF math.
Run with: pytest tests/
"""
import sys

sys.path.insert(0, ".")
from app.chunker import chunk_python_file  # noqa: E402


SAMPLE_SOURCE = '''
"""Module docstring for testing."""

def top_level_func(x):
    """Doubles x."""
    return x * 2


class Greeter:
    """Greets people."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        """Returns a greeting string."""
        return f"Hello, {self.name}!"
'''


def test_chunk_python_file_splits_functions_and_methods():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    names = {c.qualified_name for c in chunks}

    assert "top_level_func" in names
    assert "Greeter" in names
    assert "Greeter.__init__" in names
    assert "Greeter.greet" in names


def test_chunk_kinds_are_correct():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    by_name = {c.qualified_name: c for c in chunks}

    assert by_name["top_level_func"].kind == "function"
    assert by_name["Greeter"].kind == "class"
    assert by_name["Greeter.greet"].kind == "method"


def test_chunk_captures_docstrings():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    by_name = {c.qualified_name: c for c in chunks}

    assert by_name["top_level_func"].docstring == "Doubles x."


def test_syntax_error_falls_back_to_text_chunking():
    broken_source = "def broken(:\n    pass"
    chunks = chunk_python_file("broken.py", broken_source)
    # Should not raise, and should still produce something indexable.
    assert isinstance(chunks, list)


def test_rrf_fusion_rewards_items_ranked_in_both_lists():
    """Standalone RRF math check mirroring HybridRetriever.retrieve's logic,
    without needing a live ChromaDB/BM25 index."""
    rrf_k = 60
    dense_ids = ["a", "b", "c"]
    sparse_ids = ["c", "a", "d"]

    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)

    # "a" and "c" each appear in both lists, so they should outrank "b" and
    # "d", which only appear in one list each.
    assert ranked[0] in ("a", "c")
    assert ranked[1] in ("a", "c")
    assert scores["a"] > scores["b"]
    assert scores["c"] > scores["d"]
