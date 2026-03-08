#!/usr/bin/env python3
"""Demo of the WONDA pipeline Stage 1: invariant normalization.

Demonstrates how raw verifier-generated invariants are cleaned up through
AST-based normalization — stripping redundant casts, removing tautologies,
propagating contradictions, and minimizing parentheses.

Run with:
    uv run python -m wonda.demos.normalization_demo

No external tools (UAutomizer, runlim) are needed — all examples are embedded.
"""
from __future__ import annotations

from wonda.preprocess.gt_invariant_normalization import (
    normalize_invariant,
    normalize_invariant_ast,
    parse_invariant_expr,
    invariant_ast_to_pretty_src,
    invariant_ast_to_src,
)

SEPARATOR = "=" * 70
THIN_SEP = "-" * 70

# ---------------------------------------------------------------------------
# Example invariants — raw UAutomizer output
# ---------------------------------------------------------------------------
EXAMPLES: list[dict] = [
    {
        "title": "Tautology elimination + cast stripping",
        "description": (
            "UAutomizer output for a prime-counting loop. Contains reflexive\n"
            "  comparisons (x <= x), always-true constant comparisons (0 < 1),\n"
            "  and redundant (long long) casts."
        ),
        "raw": "(x <= x) && ((long long) prime_count + 1 <= x) && (0 < 1)",
    },
    {
        "title": "Multiple integral cast types",
        "description": (
            "Invariant with __int128 and unsigned __int128 casts that are\n"
            "  redundant for the logical meaning."
        ),
        "raw": "(((__int128) x + 1) > 0) && ((unsigned __int128) y >= 0)",
    },
    {
        "title": "Nested tautologies in conjunction",
        "description": (
            "A conjunction where multiple conjuncts are trivially true,\n"
            "  leaving only the meaningful constraint."
        ),
        "raw": "(i >= i) && (n <= n) && (i <= n) && (1 == 1) && (sum >= 0)",
    },
    {
        "title": "Contradiction propagation in conjunction",
        "description": (
            "A single contradictory conjunct (x < x) makes the entire\n"
            "  conjunction false — short-circuits the whole expression."
        ),
        "raw": "(x < x) && (y > 0) && (z <= n)",
    },
    {
        "title": "Contradiction in disjunction",
        "description": (
            "An always-false disjunct (x != x) is dropped, keeping only\n"
            "  the meaningful branch."
        ),
        "raw": "(x != x) || (y > 0 && z <= n)",
    },
    {
        "title": "Tautology in disjunction",
        "description": (
            "A tautologous disjunct (0 <= 1) makes the entire disjunction\n"
            "  trivially true."
        ),
        "raw": "(y > 100) || (0 <= 1) || (z < 0)",
    },
    {
        "title": "Parenthesis minimization (arithmetic precedence)",
        "description": (
            "Excessive parentheses from pycparser's default generator.\n"
            "  The pretty-printer uses C operator precedence to keep only\n"
            "  the ones that are actually required."
        ),
        "raw": "((a * b) + c) <= (((d - 1) * (e + f)))",
    },
    {
        "title": "Parenthesis preservation (required grouping)",
        "description": (
            "Some parentheses ARE needed to preserve semantics:\n"
            "  (a + b) * c is different from a + b * c."
        ),
        "raw": "((a + b) * c) > ((x || y) && z)",
    },
    {
        "title": "Complex real-world invariant (conjunctive)",
        "description": (
            "Realistic UAutomizer invariant with many casts and a few\n"
            "  tautologies mixed in with the meaningful constraints."
        ),
        "raw": (
            "((((long long) sum <= ((long long) 2 * threshold)) "
            "&& (1 <= b)) "
            "&& (((long long) b * 43) <= sum)) "
            "&& (sum <= sum) "
            "&& (steps <= 43) "
            "&& (threshold <= 100) "
            "&& (0 < 1)"
        ),
    },
    {
        "title": "Disjunctive normal form with casts",
        "description": (
            "UAutomizer output containing casts and parentheses across\n"
            "  a disjunction of conjunctions (common pattern)."
        ),
        "raw": (
            "(((prime_count <= x) && ((((long long) prime_count + 1) % 4294967296) <= x)) "
            "&& ((((long long) 2 + prime_count) % 4294967296) <= ((long long) n + 1))) "
            "|| ((prime_count == 0) && (n == 0))"
        ),
    },
]


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def show_example(idx: int, ex: dict) -> None:
    """Process and display a single normalization example."""
    raw = ex["raw"]

    # Normalize with all features: cast stripping + tautology elimination + pretty
    normalized = normalize_invariant(raw, remove_integral_casts=True, remove_tautologies=True, pretty=True)

    # Also show intermediate variants
    casts_only = normalize_invariant(raw, remove_integral_casts=True, remove_tautologies=False, pretty=False)
    taut_only = normalize_invariant(raw, remove_integral_casts=False, remove_tautologies=True, pretty=False)
    pretty_only = normalize_invariant(raw, remove_integral_casts=False, remove_tautologies=False, pretty=True)

    print(f"\n{THIN_SEP}")
    print(f"  Example {idx}: {ex['title']}")
    print(f"  {ex['description']}")
    print(THIN_SEP)
    print(f"  Raw ({len(raw)} chars):")
    print(f"    {raw}")
    print()

    # Show intermediate steps only when they differ from each other
    steps_shown = 0
    if casts_only != raw:
        print(f"  After cast stripping ({len(casts_only)} chars):")
        print(f"    {casts_only}")
        steps_shown += 1

    if taut_only != raw and taut_only != casts_only:
        print(f"  After tautology elimination ({len(taut_only)} chars):")
        print(f"    {taut_only}")
        steps_shown += 1

    if pretty_only != raw and pretty_only != casts_only and pretty_only != taut_only:
        print(f"  After parenthesis minimization ({len(pretty_only)} chars):")
        print(f"    {pretty_only}")
        steps_shown += 1

    if steps_shown > 0:
        print()

    print(f"  Normalized ({len(normalized)} chars):")
    print(f"    {normalized}")

    reduction = (1 - len(normalized) / len(raw)) * 100 if len(raw) > 0 else 0
    print(f"\n  Reduction: {len(raw)} -> {len(normalized)} chars ({reduction:+.1f}%)")


def show_parenthesis_tests() -> None:
    """Demonstrate parenthesis minimization with precedence-aware rendering."""
    section("PARENTHESIS MINIMIZATION TESTS")
    print("  The pretty-printer uses C operator precedence to insert parens")
    print("  only when they are actually needed to preserve meaning.\n")

    cases = [
        # (input, expected, description)
        ("((a * b) + c)", "a * b + c", "* binds tighter than + (no parens needed)"),
        ("((a + b) * c)", "(a + b) * c", "+ is weaker than * (parens required)"),
        ("(a - (b - c))", "a - (b - c)", "Right-associative subtraction (parens required)"),
        ("((a + b) + c)", "a + b + c", "Left-associative addition (no parens needed)"),
        ("((a && b) || c)", "a && b || c", "&& binds tighter than || (no parens needed)"),
        ("((a || b) && c)", "(a || b) && c", "|| is weaker than && (parens required)"),
        ("((a + 1) > b)", "a + 1 > b", "+ binds tighter than > (no parens needed)"),
        ("(!(a && b))", "!(a && b)", "Negation of conjunction (parens required)"),
        ("((!a) && b)", "!a && b", "! binds tighter than && (no parens needed)"),
        ("((a + b) * (c + d))", "(a + b) * (c + d)", "Both sides need parens"),
    ]

    for raw, expected, desc in cases:
        result = normalize_invariant(raw, remove_integral_casts=False, remove_tautologies=False, pretty=True)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {desc}")
        print(f"         Input:    {raw}")
        print(f"         Expected: {expected}")
        if result != expected:
            print(f"         Got:      {result}")
        print()


def show_step_by_step() -> None:
    """Show the internal transformation pipeline on a single invariant."""
    section("STEP-BY-STEP: INTERNAL PIPELINE")

    raw = "(x <= x) && ((long long) prime_count + 1 <= x) && (0 < 1)"
    print(f"  Input: {raw}\n")

    # Step 1: Parse
    print(f"  Step 1 — Parse C expression into pycparser AST")
    ast = parse_invariant_expr(raw)
    if ast is None:
        print("    [FAILED to parse]")
        return
    default_src = invariant_ast_to_src(ast)
    print(f"    Default rendering: {default_src}")

    # Step 2: Normalize AST (all rewrites)
    print(f"\n  Step 2 — Normalize AST (bottom-up rewrite)")
    print(f"    Rules applied:")
    print(f"      - (x <= x)       => tautology, removed from conjunction")
    print(f"      - (0 < 1)        => tautology (constant comparison), removed")
    print(f"      - (long long)    => integral cast, stripped")

    norm_ast = normalize_invariant_ast(ast, remove_integral_casts=True, remove_tautologies=True)
    norm_default = invariant_ast_to_src(norm_ast)
    print(f"    Default rendering: {norm_default}")

    # Step 3: Pretty-print
    print(f"\n  Step 3 — Pretty-print with minimal parentheses")
    norm_pretty = invariant_ast_to_pretty_src(norm_ast)
    print(f"    Pretty rendering:  {norm_pretty}")

    # Combined
    print(f"\n  Full pipeline result:")
    print(f"    Before: {raw}")
    print(f"    After:  {norm_pretty}")


def main() -> None:
    section("WONDA NORMALIZATION DEMO")
    print(
        "  Stage 1 of the WONDA data pipeline: AST-based invariant normalization.\n"
        "\n"
        "  Raw invariants from automated verifiers (e.g. UAutomizer) are correct\n"
        "  by construction, but cluttered with syntactic noise:\n"
        "    - Redundant integral casts: (long long), (__int128), (unsigned int)\n"
        "    - Tautologies: (x <= x), (0 < 1), (1 == 1)\n"
        "    - Excessive parentheses from code generation\n"
        "\n"
        "  Normalization cleans these up while preserving semantic equivalence.\n"
        "  The normalized invariant phi' === phi (logically equivalent).\n"
    )

    # Part 1: Step-by-step internal pipeline
    show_step_by_step()

    # Part 2: Full example gallery
    section("NORMALIZATION EXAMPLES")
    for i, ex in enumerate(EXAMPLES, 1):
        show_example(i, ex)

    # Part 3: Parenthesis precision tests
    show_parenthesis_tests()

    # Summary
    section("SUMMARY")
    total_raw = sum(len(ex["raw"]) for ex in EXAMPLES)
    total_norm = sum(
        len(normalize_invariant(ex["raw"], remove_integral_casts=True, remove_tautologies=True, pretty=True))
        for ex in EXAMPLES
    )
    reduction = (1 - total_norm / total_raw) * 100 if total_raw > 0 else 0
    print(f"  Processed {len(EXAMPLES)} example invariants.")
    print(f"  Total characters: {total_raw} -> {total_norm} ({reduction:+.1f}% reduction)")
    print(
        f"\n"
        f"  Normalization is the first stage of the WONDA data pipeline:\n"
        f"\n"
        f"    Raw invariant (UAutomizer)\n"
        f"        | Stage 1: NORMALIZE (this demo)\n"
        f"        |   - Strip integral casts\n"
        f"        |   - Eliminate tautologies / contradictions\n"
        f"        |   - Minimize parentheses\n"
        f"        v\n"
        f"    Normalized invariant\n"
        f"        | Stage 2: SIMPLIFY (LLM-based rewriting)\n"
        f"        v\n"
        f"    Final curated invariant (training target)\n"
    )


if __name__ == "__main__":
    main()
