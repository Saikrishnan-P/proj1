"""
Tests for app/ts_chunker.py — verifies AST-based chunking actually extracts
the right units for each non-Python language it supports.
"""
import sys

sys.path.insert(0, ".")
from app.ts_chunker import chunk_with_tree_sitter  # noqa: E402


JS_SAMPLE = '''
function add(a, b) {
  // Adds two numbers together and returns the sum.
  return a + b;
}

class Greeter {
  constructor(name) {
    this.name = name;
  }

  greet() {
    return `Hello, ${this.name}`;
  }
}

const multiply = (a, b) => {
  // Multiplies two numbers together and returns the product.
  return a * b;
};
'''

TS_SAMPLE = '''
interface Point {
  x: number;
  y: number;
}

export function distance(a: Point, b: Point): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

export class Shape {
  area(): number {
    return 0;
  }
}
'''

JAVA_SAMPLE = '''
public class Greeter {
    private String name;

    public Greeter(String name) {
        this.name = name;
    }

    public String greet() {
        return "Hello, " + name;
    }
}

interface Shape {
    double area();
}
'''

HTML_SAMPLE = '''<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
  <header class="nav">
    <h1>Title</h1>
  </header>
  <section id="main-content">
    <p>Hello world, this is enough text to clear the minimum chunk size.</p>
  </section>
</body>
</html>
'''

CSS_SAMPLE = '''
.header {
  color: red;
  font-size: 14px;
}

#main-content {
  margin: 0 auto;
  padding: 20px;
}

@media (max-width: 600px) {
  .header { color: blue; }
}
'''


def test_js_extracts_function_and_class_and_methods():
    chunks = chunk_with_tree_sitter("sample.js", JS_SAMPLE)
    names = {c.qualified_name for c in chunks}
    assert "add" in names
    assert "Greeter" in names
    assert "Greeter.greet" in names
    assert "multiply" in names  # const arrow function


def test_ts_extracts_interface_function_and_class():
    chunks = chunk_with_tree_sitter("sample.ts", TS_SAMPLE)
    kinds_by_name = {c.qualified_name: c.kind for c in chunks}
    assert kinds_by_name.get("Point") == "interface"
    assert kinds_by_name.get("distance") == "function"
    assert kinds_by_name.get("Shape") == "class"
    assert kinds_by_name.get("Shape.area") == "method"


def test_java_extracts_class_interface_and_members():
    chunks = chunk_with_tree_sitter("Sample.java", JAVA_SAMPLE)
    names = {c.qualified_name for c in chunks}
    assert "Greeter" in names
    assert "Greeter.Greeter" in names  # constructor
    assert "Greeter.greet" in names
    assert "Shape" in names


def test_html_extracts_top_level_body_elements():
    chunks = chunk_with_tree_sitter("sample.html", HTML_SAMPLE)
    names = {c.qualified_name for c in chunks}
    assert "header.nav" in names
    assert "section#main-content" in names


def test_css_extracts_rules_and_media_query():
    chunks = chunk_with_tree_sitter("sample.css", CSS_SAMPLE)
    kinds = [c.kind for c in chunks]
    names = {c.qualified_name for c in chunks}
    assert ".header" in names
    assert "#main-content" in names
    assert "at_rule" in kinds  # the @media block


def test_unsupported_extension_returns_none():
    assert chunk_with_tree_sitter("sample.go", "func main() {}") is None
