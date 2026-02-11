#!/usr/bin/env python3
"""Visualize pycparser AST trees for C programs and invariant expressions.

Prints box-drawing tree diagrams showing how C code is represented
internally as Abstract Syntax Trees, and how normalization rewrites
those trees.

Run with:
    uv run python -m wonda.demos.ast_tree_demo

No external tools needed — all examples are embedded.
"""
from __future__ import annotations

import io
from typing import Optional

from pycparser import c_ast, c_generator, c_parser

from wonda.preprocess.gt_invariant_normalization import (
    normalize_invariant_ast,
    parse_invariant_expr,
    invariant_ast_to_pretty_src,
)

SEPARATOR = "=" * 70
THIN_SEP = "-" * 70

# ---------------------------------------------------------------------------
# Tree renderer
# ---------------------------------------------------------------------------

def _node_label(node: c_ast.Node) -> str:
    """One-line description of an AST node."""
    gen = c_generator.CGenerator()
    if isinstance(node, c_ast.ID):
        return f"ID \u00ab{node.name}\u00bb"
    if isinstance(node, c_ast.Constant):
        return f"Constant \u00ab{node.value}\u00bb"
    if isinstance(node, c_ast.BinaryOp):
        return f"BinaryOp \u00ab{node.op}\u00bb"
    if isinstance(node, c_ast.UnaryOp):
        return f"UnaryOp \u00ab{node.op}\u00bb"
    if isinstance(node, c_ast.Cast):
        ty = gen.visit(node.to_type)
        return f"Cast \u00ab{ty}\u00bb"
    if isinstance(node, c_ast.FuncCall):
        name = gen.visit(node.name) if node.name else "?"
        return f"FuncCall \u00ab{name}\u00bb"
    if isinstance(node, c_ast.FuncDef):
        name = node.decl.name if node.decl else "?"
        return f"FuncDef \u00ab{name}\u00bb"
    if isinstance(node, c_ast.Decl):
        return f"Decl \u00ab{node.name or '?'}\u00bb"
    if isinstance(node, c_ast.TypeDecl):
        return f"TypeDecl"
    if isinstance(node, c_ast.IdentifierType):
        return f"IdentifierType \u00ab{' '.join(node.names)}\u00bb"
    if isinstance(node, c_ast.If):
        return "If"
    if isinstance(node, c_ast.While):
        return "While"
    if isinstance(node, c_ast.For):
        return "For"
    if isinstance(node, c_ast.Return):
        return "Return"
    if isinstance(node, c_ast.Compound):
        return "Compound"
    if isinstance(node, c_ast.Assignment):
        return f"Assignment \u00ab{node.op}\u00bb"
    if isinstance(node, c_ast.ExprList):
        return "ExprList"
    if isinstance(node, c_ast.Typename):
        return "Typename"
    if isinstance(node, c_ast.FileAST):
        return "FileAST"
    if isinstance(node, c_ast.FuncDecl):
        return "FuncDecl"
    if isinstance(node, c_ast.ParamList):
        return "ParamList"
    if isinstance(node, c_ast.EmptyStatement):
        return "EmptyStatement"
    return type(node).__name__


# Nodes whose children are type-system boilerplate (not interesting for
# the expression-level view).
_SKIP_CHILDREN_FOR = {"Cast"}


def render_tree(
    node: c_ast.Node,
    *,
    prefix: str = "",
    is_last: bool = True,
    child_label: str = "",
    max_depth: int = 20,
    _depth: int = 0,
    collapse_cast_types: bool = True,
) -> str:
    """Return a Unicode box-drawing representation of *node*."""
    buf = io.StringIO()
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
    tag = f"[{child_label}] " if child_label else ""
    buf.write(f"{prefix}{connector}{tag}{_node_label(node)}\n")

    if _depth >= max_depth:
        ext = "    " if is_last else "\u2502   "
        buf.write(f"{prefix}{ext}    ... (max depth)\n")
        return buf.getvalue()

    # Collect children, optionally collapsing Cast type internals.
    children: list[tuple[str, c_ast.Node]] = []
    if collapse_cast_types and isinstance(node, c_ast.Cast):
        # Only show the inner expression, not the Typename subtree.
        if node.expr is not None:
            children.append(("expr", node.expr))
    else:
        children = list(node.children())

    ext = "    " if is_last else "\u2502   "
    for i, (cname, child) in enumerate(children):
        child_is_last = i == len(children) - 1
        buf.write(
            render_tree(
                child,
                prefix=prefix + ext,
                is_last=child_is_last,
                child_label=cname,
                max_depth=max_depth,
                _depth=_depth + 1,
                collapse_cast_types=collapse_cast_types,
            )
        )
    return buf.getvalue()


def print_tree(node: c_ast.Node, **kwargs) -> None:
    print(render_tree(node, **kwargs), end="")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def parse_program(code: str) -> c_ast.FileAST:
    return c_parser.CParser().parse(code)


def parse_expr(src: str) -> Optional[c_ast.Node]:
    return parse_invariant_expr(src)


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def demo_simple_expressions() -> None:
    """Show AST trees of basic C expressions."""
    section("1. SIMPLE EXPRESSION TREES")
    print("  Each C expression is parsed into an AST whose interior nodes are\n"
          "  operators and leaves are identifiers / constants.\n")

    exprs = [
        ("a + b", "Simple addition"),
        ("a + b * c", "Precedence: * binds tighter than +"),
        ("(a + b) * c", "Parentheses override precedence"),
        ("x <= n && i >= 0", "Logical conjunction of comparisons"),
        ("!(a || b)", "Negation of a disjunction"),
    ]

    for src, desc in exprs:
        ast = parse_expr(src)
        if ast is None:
            continue
        print(f"  {desc}")
        print(f"  Expression: {src}\n")
        print_tree(ast)
        print()


def demo_program_tree() -> None:
    """Show the AST of a small C function."""
    section("2. PROGRAM AST TREE")
    print("  A complete C function parsed into a full AST.\n")

    code = """\
int main() {
    int i = 0;
    int sum = 0;
    while (i < n) {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}
"""
    print(f"  Source:\n")
    for line in code.strip().splitlines():
        print(f"    {line}")
    print()

    ast = parse_program(code)
    print("  AST:\n")
    print_tree(ast, max_depth=10)


def demo_invariant_tree() -> None:
    """Show a raw verifier invariant and its AST."""
    section("3. RAW INVARIANT AST")
    raw = "(x <= x) && ((long long) prime_count + 1 <= x) && (0 < 1)"
    print(f"  Raw invariant (from UAutomizer):\n    {raw}\n")
    ast = parse_expr(raw)
    if ast is None:
        return
    print("  AST:\n")
    print_tree(ast)


def demo_normalization_trees() -> None:
    """Show before/after ASTs for normalization."""
    section("4. NORMALIZATION: BEFORE vs AFTER")
    print("  The normalization pass rewrites the AST bottom-up:\n"
          "    - Tautologies (x <= x, 0 < 1) are removed\n"
          "    - Integral casts ((long long)) are stripped\n"
          "    - Conjunction collapses after tautologous conjuncts are dropped\n")

    raw = "(x <= x) && ((long long) prime_count + 1 <= x) && (0 < 1)"
    ast = parse_expr(raw)
    if ast is None:
        return

    print(f"  BEFORE normalization: {raw}\n")
    print_tree(ast)

    norm = normalize_invariant_ast(
        ast, remove_integral_casts=True, remove_tautologies=True
    )
    norm_src = invariant_ast_to_pretty_src(norm)

    print(f"\n  AFTER normalization: {norm_src}\n")
    print_tree(norm)


def demo_cast_stripping_tree() -> None:
    """Show how cast nodes are removed from the tree."""
    section("5. CAST STRIPPING: TREE DIFF")
    print("  Integral casts add extra nodes to the AST that obscure the logic.\n"
          "  Stripping them simplifies the tree structure.\n")

    raw = "((__int128) x + 1) > 0"
    ast = parse_expr(raw)
    if ast is None:
        return

    print(f"  Expression: {raw}\n")
    print("  Before (cast nodes visible, type subtree collapsed):\n")
    print_tree(ast)

    print("  Before (cast nodes with full type subtree):\n")
    print_tree(ast, collapse_cast_types=False)

    norm = normalize_invariant_ast(
        ast, remove_integral_casts=True, remove_tautologies=False
    )
    norm_src = invariant_ast_to_pretty_src(norm)
    print(f"  After cast stripping: {norm_src}\n")
    print_tree(norm)


def demo_tautology_tree() -> None:
    """Show how tautology elimination prunes branches."""
    section("6. TAUTOLOGY ELIMINATION: BRANCH PRUNING")
    print("  Tautologous conjuncts are dropped, collapsing the && chain.\n")

    raw = "(i >= i) && (i <= n) && (sum >= 0) && (1 == 1)"
    ast = parse_expr(raw)
    if ast is None:
        return

    print(f"  Expression: {raw}\n")
    print("  Before (4 conjuncts, 2 are tautologies):\n")
    print_tree(ast)

    norm = normalize_invariant_ast(
        ast, remove_integral_casts=False, remove_tautologies=True
    )
    norm_src = invariant_ast_to_pretty_src(norm)
    print(f"  After (2 conjuncts remain): {norm_src}\n")
    print_tree(norm)


def demo_complex_tree() -> None:
    """Show a more complex invariant with disjunction."""
    section("7. COMPLEX INVARIANT TREE (DISJUNCTION)")
    raw = (
        "(sum <= 2 * threshold && b * 43 <= sum && 1 <= b) "
        "|| (sum == 0 && steps == 0)"
    )
    ast = parse_expr(raw)
    if ast is None:
        return

    print(f"  Expression:\n    {raw}\n")
    print("  AST (disjunction of two conjunctions):\n")
    print_tree(ast)


def demo_side_by_side() -> None:
    """Show multiple invariants side by side: raw vs normalized."""
    section("8. GALLERY: RAW vs NORMALIZED TREES")

    cases = [
        (
            "(n >= n) && ((long long) sum >= 0)",
            "Reflexive tautology + cast",
        ),
        (
            "(x < x) || (y > 0)",
            "Contradiction in disjunction branch",
        ),
        (
            "((unsigned int) a + (long long) b) <= ((long long) c * 2)",
            "Multiple casts on both sides",
        ),
    ]

    for raw, desc in cases:
        ast = parse_expr(raw)
        if ast is None:
            continue

        norm = normalize_invariant_ast(
            ast, remove_integral_casts=True, remove_tautologies=True
        )
        norm_src = invariant_ast_to_pretty_src(norm)

        print(f"\n{THIN_SEP}")
        print(f"  {desc}")
        print(f"  Raw:        {raw}")
        print(f"  Normalized: {norm_src}")
        print(f"{THIN_SEP}")
        print(f"\n  Before:\n")
        # Re-parse to get a fresh copy (normalize mutates in-place)
        print_tree(parse_expr(raw))
        print(f"  After:\n")
        print_tree(norm)


def main() -> None:
    section("AST TREE VISUALIZATION DEMO")
    print(
        "  This demo visualizes pycparser Abstract Syntax Trees as\n"
        "  box-drawing tree diagrams. It shows how C expressions and\n"
        "  programs are represented internally, and how the WONDA\n"
        "  normalization pass rewrites invariant ASTs.\n"
    )

    demo_simple_expressions()
    demo_program_tree()
    demo_invariant_tree()
    demo_normalization_trees()
    demo_cast_stripping_tree()
    demo_tautology_tree()
    demo_complex_tree()
    demo_side_by_side()

    section("SUMMARY")
    print(
        "  Key takeaways:\n"
        "\n"
        "  - C code is parsed into a tree of typed AST nodes\n"
        "  - Operator precedence is encoded in the tree structure,\n"
        "    not by parentheses (parentheses are only in the source text)\n"
        "  - Normalization rewrites the tree bottom-up:\n"
        "      Cast nodes  -> removed (child promoted)\n"
        "      Tautologies -> dropped from conjunctions\n"
        "      Contradictions -> propagated (short-circuit)\n"
        "  - The pretty-printer reconstructs source from the tree,\n"
        "    adding parentheses only where precedence requires them\n"
    )


if __name__ == "__main__":
    main()
