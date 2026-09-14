"""
AST-based chunking.

Instead of splitting files by a fixed number of lines/characters (which cuts
functions and classes in half and destroys retrieval quality), we parse each
file's AST and emit one chunk per top-level function, method, or class. This
keeps each chunk semantically whole and lets us attach rich, useful metadata
(qualified name, kind, line range, docstring) for filtering and citation.

Falls back to a plain sliding-window text splitter for files that fail to
parse (syntax errors, non-Python files) so ingestion never hard-fails.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterator

from app.config import settings


@dataclass
class Chunk:
    id: str
    file_path: str
    kind: str            # "function" | "method" | "class" | "module" | "text_block"
    qualified_name: str
    start_line: int
    end_line: int
    code: str
    docstring: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_document(self) -> dict:
        """Flatten into the (id, text, metadata) shape ChromaDB expects."""
        return {
            "id": self.id,
            "text": self.code,
            "metadata": {
                "file_path": self.file_path,
                "kind": self.kind,
                "qualified_name": self.qualified_name,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "docstring": self.docstring or "",
                **self.metadata,
            },
        }


def _make_id(file_path: str, qualified_name: str, start_line: int) -> str:
    raw = f"{file_path}:{qualified_name}:{start_line}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    """Python 3.8+ ast.get_source_segment is convenient but re-derive from
    lineno/end_lineno directly so this also works on partially-typed nodes."""
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(source_lines[start:end])


def _iter_top_level_defs(tree: ast.Module) -> Iterator[tuple[ast.AST, str, str | None]]:
    """Yield (node, qualified_name, parent_class_or_None) for every top-level
    function/class and every method inside a top-level class."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, node.name, None
        elif isinstance(node, ast.ClassDef):
            yield node, node.name, None
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield sub, f"{node.name}.{sub.name}", node.name


def chunk_python_file(file_path: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_plain_text(file_path, source)

    source_lines = source.splitlines()
    seen_any = False

    for node, qname, parent_class in _iter_top_level_defs(tree):
        seen_any = True
        code = _get_source_segment(source_lines, node)
        if len(code.strip()) < settings.min_chunk_chars:
            continue

        kind = "class" if isinstance(node, ast.ClassDef) else ("method" if parent_class else "function")
        docstring = ast.get_docstring(node)

        # Large functions/classes get split further so no single chunk blows
        # past the embedding model's effective context window.
        if len(code) > settings.max_chunk_chars:
            for i, sub_code in enumerate(_split_by_chars(code, settings.max_chunk_chars)):
                chunks.append(Chunk(
                    id=_make_id(file_path, f"{qname}#part{i}", node.lineno),
                    file_path=file_path,
                    kind=kind,
                    qualified_name=qname,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    code=sub_code,
                    docstring=docstring,
                    metadata={"part": i, "parent_class": parent_class or ""},
                ))
        else:
            chunks.append(Chunk(
                id=_make_id(file_path, qname, node.lineno),
                file_path=file_path,
                kind=kind,
                qualified_name=qname,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                code=code,
                docstring=docstring,
                metadata={"parent_class": parent_class or ""},
            ))

    # Module-level code (imports, constants, top-level script logic) not
    # captured above still has retrieval value, so index it as one chunk.
    if not seen_any and source.strip():
        chunks.extend(chunk_plain_text(file_path, source))
    elif source.strip():
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            chunks.append(Chunk(
                id=_make_id(file_path, "__module__", 1),
                file_path=file_path,
                kind="module",
                qualified_name=os.path.basename(file_path),
                start_line=1,
                end_line=1,
                code=module_docstring,
                docstring=module_docstring,
            ))

    return chunks


def chunk_plain_text(file_path: str, source: str, window_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Fallback sliding-window splitter for non-Python or unparseable files."""
    chunks = []
    if not source.strip():
        return chunks
    step = max(window_chars - overlap, 1)
    for i, start in enumerate(range(0, len(source), step)):
        piece = source[start:start + window_chars]
        if len(piece.strip()) < settings.min_chunk_chars:
            continue
        approx_line = source[:start].count("\n") + 1
        chunks.append(Chunk(
            id=_make_id(file_path, f"block#{i}", approx_line),
            file_path=file_path,
            kind="text_block",
            qualified_name=f"{os.path.basename(file_path)}#block{i}",
            start_line=approx_line,
            end_line=approx_line + piece.count("\n"),
            code=piece,
        ))
    return chunks


def _split_by_chars(text: str, size: int) -> Iterator[str]:
    for start in range(0, len(text), size):
        yield text[start:start + size]


def _strip_ipython_magics(code: str) -> str:
    """Comment out IPython magic lines (%matplotlib inline, %%time, ...) and
    shell-escape lines (!pip install ...) so the rest of a notebook cell can
    still be parsed as plain Python. Without this, a single magic/shell line
    anywhere in a cell makes ast.parse() raise SyntaxError and the *entire*
    cell falls back to the much less useful plain-text chunker."""
    cleaned_lines = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            cleaned_lines.append("# " + line)
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _cell_source_text(cell: dict) -> str:
    """Notebook cell 'source' is either a list of lines or a single string
    depending on the tool that wrote the .ipynb — normalize to one string."""
    raw = cell.get("source", "")
    if isinstance(raw, list):
        return "".join(raw)
    return str(raw)


def chunk_notebook_file(file_path: str, source: str) -> list[Chunk]:
    """Chunk a Jupyter notebook (.ipynb) cell-by-cell.

    Code cells are run back through chunk_python_file() so functions/classes
    defined inside a notebook still get proper AST-level chunks; anything
    that isn't a whole function/class (loose script-style cell code) falls
    back to a single text block for that cell, same as a .py file would.
    Markdown cells are chunked as text, since notebook prose is often the
    only explanation of what the adjacent code cell actually does.
    Raw cells and empty cells are skipped.

    Falls back to the plain-text chunker for the whole file if the notebook
    JSON itself can't be parsed (corrupted file, or not actually a notebook).
    """
    try:
        notebook = json.loads(source)
    except (json.JSONDecodeError, ValueError):
        return chunk_plain_text(file_path, source)

    cells = notebook.get("cells", [])
    chunks: list[Chunk] = []

    for cell_index, cell in enumerate(cells):
        cell_type = cell.get("cell_type")
        cell_text = _cell_source_text(cell)
        if not cell_text.strip():
            continue

        if cell_type == "code":
            cell_chunks = chunk_python_file(file_path, _strip_ipython_magics(cell_text))
        elif cell_type == "markdown":
            cell_chunks = chunk_plain_text(file_path, cell_text)
        else:
            continue  # raw cells carry no queryable content

        for sub in cell_chunks:
            sub.id = _make_id(file_path, f"cell{cell_index}:{sub.qualified_name}", cell_index)
            sub.qualified_name = f"cell[{cell_index}].{sub.qualified_name}"
            sub.kind = "markdown_cell" if cell_type == "markdown" else sub.kind
            sub.metadata["cell_index"] = cell_index
            sub.metadata["cell_type"] = cell_type
            chunks.append(sub)

    # Notebook parsed but produced nothing chunkable (e.g. all-output, no
    # source) — fall back rather than silently indexing zero chunks for it.
    if not chunks and source.strip():
        return chunk_plain_text(file_path, source)

    return chunks


def chunk_file(file_path: str, source: str) -> list[Chunk]:
    """Dispatch by extension. Python gets its own AST chunker (below).
    Jupyter notebooks are parsed cell-by-cell (see chunk_notebook_file()).
    JS/TS/TSX/Java/HTML/CSS/SCSS get tree-sitter-based AST chunking (see
    ts_chunker.py). Everything else gets the plain-text fallback (still
    useful for READMEs, config files, etc)."""
    if file_path.endswith(".py"):
        return chunk_python_file(file_path, source)

    if file_path.endswith(".ipynb"):
        return chunk_notebook_file(file_path, source)

    from app.ts_chunker import chunk_with_tree_sitter  # local import avoids a
    # hard dependency on tree-sitter for pure-Python-repo users who never hit this path
    ts_chunks = chunk_with_tree_sitter(file_path, source)
    if ts_chunks is not None:
        return ts_chunks

    return chunk_plain_text(file_path, source)
