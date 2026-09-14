"""
AST-based chunking for languages other than Python, using tree-sitter.

Python's own `ast` module only understands Python, so every other language
needs its own grammar. tree-sitter is a parser generator with prebuilt,
pip-installable grammars for most mainstream languages — this is the same
approach GitHub's code navigation and Neovim's syntax highlighting use.

Each language gets its own extraction function because "what counts as a
chunkable unit" differs by language: JS/TS chunk on functions/classes/
methods/interfaces, Java on classes/interfaces/enums/methods, HTML on
top-level body elements, CSS on rules and at-rules. All of them funnel into
the same `Chunk` dataclass from chunker.py so the rest of the pipeline
(indexer, retriever) doesn't need to know or care which language produced
a given chunk.

Falls back to None (caller uses the plain-text sliding-window splitter)
whenever parsing fails or a file has no chunkable top-level constructs —
ingestion should never hard-fail because one file didn't parse cleanly.
"""
from __future__ import annotations

from tree_sitter import Language, Node, Parser

from app.chunker import Chunk, _make_id, chunk_plain_text
from app.config import settings

# --------------------------------------------------------------------------
# Parser setup — built lazily so importing this module doesn't pay the cost
# of loading every grammar until a file of that language actually shows up.
# --------------------------------------------------------------------------

_parsers: dict[str, Parser] = {}


def _get_parser(language_key: str) -> Parser:
    if language_key in _parsers:
        return _parsers[language_key]

    if language_key == "javascript":
        import tree_sitter_javascript as g
        lang = Language(g.language())
    elif language_key == "typescript":
        import tree_sitter_typescript as g
        lang = Language(g.language_typescript())
    elif language_key == "tsx":
        import tree_sitter_typescript as g
        lang = Language(g.language_tsx())
    elif language_key == "java":
        import tree_sitter_java as g
        lang = Language(g.language())
    elif language_key == "html":
        import tree_sitter_html as g
        lang = Language(g.language())
    elif language_key == "css":
        import tree_sitter_css as g
        lang = Language(g.language())
    elif language_key == "scss":
        import tree_sitter_scss as g
        lang = Language(g.language())
    else:
        raise ValueError(f"No tree-sitter grammar wired up for '{language_key}'")

    parser = Parser(lang)
    _parsers[language_key] = parser
    return parser


_EXTENSION_TO_LANGUAGE = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
}


def language_for_extension(file_path: str) -> str | None:
    for ext, lang in _EXTENSION_TO_LANGUAGE.items():
        if file_path.endswith(ext):
            return lang
    return None


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _child_name(node: Node, source_bytes: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    return _node_text(name_node, source_bytes) if name_node else None


def _make_chunk(file_path: str, kind: str, qualified_name: str, node: Node,
                 source_bytes: bytes, parent: str = "") -> Chunk:
    code = _node_text(node, source_bytes)
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    return Chunk(
        id=_make_id(file_path, qualified_name, start_line),
        file_path=file_path,
        kind=kind,
        qualified_name=qualified_name,
        start_line=start_line,
        end_line=end_line,
        code=code,
        metadata={"parent_class": parent} if parent else {},
    )


def _unwrap_export(node: Node) -> Node:
    """`export function foo() {}` parses as export_statement -> function_declaration.
    Unwrap it so callers see the real declaration, matching how a non-exported
    version of the same code would parse."""
    if node.type == "export_statement" and node.named_children:
        return node.named_children[0]
    return node


# --------------------------------------------------------------------------
# JavaScript / TypeScript / TSX
# --------------------------------------------------------------------------

_JS_FUNCTION_VALUE_TYPES = ("arrow_function", "function_expression", "function")


def _extract_js_like(root: Node, file_path: str, source_bytes: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []

    for top in root.named_children:
        node = _unwrap_export(top)

        if node.type == "function_declaration":
            name = _child_name(node, source_bytes) or "anonymous"
            if len(_node_text(node, source_bytes).strip()) >= settings.min_chunk_chars:
                chunks.append(_make_chunk(file_path, "function", name, node, source_bytes))

        elif node.type == "class_declaration":
            class_name = _child_name(node, source_bytes) or "AnonymousClass"
            chunks.append(_make_chunk(file_path, "class", class_name, node, source_bytes))

            body = node.child_by_field_name("body")
            if body:
                for member in body.named_children:
                    if member.type == "method_definition":
                        m_name_node = member.child_by_field_name("name")
                        m_name = _node_text(m_name_node, source_bytes) if m_name_node else "anonymous"
                        chunks.append(_make_chunk(
                            file_path, "method", f"{class_name}.{m_name}",
                            member, source_bytes, parent=class_name,
                        ))
                    elif member.type == "field_definition":
                        value = member.child_by_field_name("value")
                        if value is not None and value.type in _JS_FUNCTION_VALUE_TYPES:
                            f_name_node = member.child_by_field_name("property")
                            f_name = _node_text(f_name_node, source_bytes) if f_name_node else "anonymous"
                            chunks.append(_make_chunk(
                                file_path, "method", f"{class_name}.{f_name}",
                                member, source_bytes, parent=class_name,
                            ))

        elif node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in node.named_children:
                if declarator.type != "variable_declarator":
                    continue
                value = declarator.child_by_field_name("value")
                if value is not None and value.type in _JS_FUNCTION_VALUE_TYPES:
                    name = _child_name(declarator, source_bytes) or "anonymous"
                    if len(_node_text(declarator, source_bytes).strip()) >= settings.min_chunk_chars:
                        chunks.append(_make_chunk(file_path, "function", name, declarator, source_bytes))

        elif node.type == "interface_declaration":
            name = _child_name(node, source_bytes) or "AnonymousInterface"
            chunks.append(_make_chunk(file_path, "interface", name, node, source_bytes))

        elif node.type == "type_alias_declaration":
            name = _child_name(node, source_bytes) or "AnonymousType"
            chunks.append(_make_chunk(file_path, "type_alias", name, node, source_bytes))

    return chunks


# --------------------------------------------------------------------------
# Java
# --------------------------------------------------------------------------

_JAVA_TYPE_DECLARATIONS = ("class_declaration", "interface_declaration", "enum_declaration")
_JAVA_METHOD_LIKE = ("method_declaration", "constructor_declaration")


def _extract_java(root: Node, file_path: str, source_bytes: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []

    def visit(node: Node):
        if node.type in _JAVA_TYPE_DECLARATIONS:
            type_name = _child_name(node, source_bytes) or "AnonymousType"
            kind = {"class_declaration": "class", "interface_declaration": "interface",
                    "enum_declaration": "enum"}[node.type]
            chunks.append(_make_chunk(file_path, kind, type_name, node, source_bytes))

            body = node.child_by_field_name("body")
            if body:
                for member in body.named_children:
                    if member.type in _JAVA_METHOD_LIKE:
                        m_name = _child_name(member, source_bytes) or "anonymous"
                        m_kind = "constructor" if member.type == "constructor_declaration" else "method"
                        chunks.append(_make_chunk(
                            file_path, m_kind, f"{type_name}.{m_name}",
                            member, source_bytes, parent=type_name,
                        ))
                    elif member.type in _JAVA_TYPE_DECLARATIONS:
                        visit(member)  # nested/inner classes
        else:
            for child in node.named_children:
                visit(child)

    for top in root.named_children:
        visit(top)

    return chunks


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def _html_tag_name(element: Node, source_bytes: bytes) -> str | None:
    start_tag = next((c for c in element.children if c.type in ("start_tag", "self_closing_tag")), None)
    if not start_tag:
        return None
    tag_name_node = next((c for c in start_tag.children if c.type == "tag_name"), None)
    return _node_text(tag_name_node, source_bytes) if tag_name_node else None


def _html_attr(element: Node, source_bytes: bytes, attr: str) -> str | None:
    start_tag = next((c for c in element.children if c.type in ("start_tag", "self_closing_tag")), None)
    if not start_tag:
        return None
    for c in start_tag.children:
        if c.type != "attribute":
            continue
        name_node = next((gc for gc in c.children if gc.type == "attribute_name"), None)
        if name_node and _node_text(name_node, source_bytes) == attr:
            value_node = next((gc for gc in c.children if gc.type == "quoted_attribute_value"), None)
            if value_node:
                return _node_text(value_node, source_bytes).strip("\"'")
    return None


def _find_element_by_tag(node: Node, tag: str, source_bytes: bytes) -> Node | None:
    if node.type == "element" and _html_tag_name(node, source_bytes) == tag:
        return node
    for child in node.children:
        found = _find_element_by_tag(child, tag, source_bytes)
        if found:
            return found
    return None


def _extract_html(root: Node, file_path: str, source_bytes: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []

    body = _find_element_by_tag(root, "body", source_bytes)
    container = body if body else root

    for child in container.children:
        if child.type != "element":
            continue
        tag = _html_tag_name(child, source_bytes) or "element"
        el_id = _html_attr(child, source_bytes, "id")
        el_class = _html_attr(child, source_bytes, "class")
        qualifier = f"#{el_id}" if el_id else (f".{el_class.split()[0]}" if el_class else "")
        qualified_name = f"{tag}{qualifier}"

        code = _node_text(child, source_bytes)
        if len(code.strip()) < settings.min_chunk_chars:
            continue
        chunks.append(_make_chunk(file_path, "element", qualified_name, child, source_bytes))

    return chunks


# --------------------------------------------------------------------------
# CSS / SCSS
# --------------------------------------------------------------------------

_CSS_AT_RULE_TYPES = ("media_statement", "supports_statement", "keyframes_statement")


def _extract_css(root: Node, file_path: str, source_bytes: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []

    for top in root.named_children:
        if top.type == "rule_set":
            selectors = top.child_by_field_name("selectors") or next(
                (c for c in top.children if c.type == "selectors"), None
            )
            selector_text = _node_text(selectors, source_bytes).strip() if selectors else "rule"
            selector_text = " ".join(selector_text.split())[:80]
            chunks.append(_make_chunk(file_path, "rule", selector_text, top, source_bytes))

        elif top.type in _CSS_AT_RULE_TYPES:
            # Whole at-rule block (e.g. an entire @media query) as one chunk —
            # splitting further would separate a breakpoint from the rules
            # that only apply inside it.
            prelude = _node_text(top, source_bytes).split("{")[0].strip()
            prelude = " ".join(prelude.split())[:80]
            chunks.append(_make_chunk(file_path, "at_rule", prelude or top.type, top, source_bytes))

    return chunks


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

_EXTRACTORS = {
    "javascript": _extract_js_like,
    "typescript": _extract_js_like,
    "tsx": _extract_js_like,
    "java": _extract_java,
    "html": _extract_html,
    "css": _extract_css,
    "scss": _extract_css,
}


def chunk_with_tree_sitter(file_path: str, source: str) -> list[Chunk] | None:
    """Returns AST-based chunks for a tree-sitter-supported language, or
    None if this file's extension isn't wired up here (caller should fall
    back to chunk_plain_text) or parsing produced nothing usable."""
    language_key = language_for_extension(file_path)
    if language_key is None:
        return None

    try:
        parser = _get_parser(language_key)
        source_bytes = source.encode("utf-8", errors="ignore")
        tree = parser.parse(source_bytes)
    except Exception:
        return None

    extractor = _EXTRACTORS[language_key]
    chunks = extractor(tree.root_node, file_path, source_bytes)

    if not chunks:
        return chunk_plain_text(file_path, source)

    return chunks
