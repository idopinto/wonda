import os
import sys

# Keep the test style consistent with existing tests in this repo.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wonda.data_pipeline.prune_invariant import (
    parse_invariant_expr,
    pretty_invariant_src,
    strip_unnecessary_typing_casts,
)


def test_strip_unnecessary_typing_casts_removes_long_long():
    inv = "((((prime_count <= x) && ((((long long) prime_count + 1) % 4294967296) <= x)) && ((((long long) 2 + prime_count) % 4294967296) <= ((long long) n + 1))) || ((prime_count == 0) && (n == 0)))"
    simplified = strip_unnecessary_typing_casts(inv)

    assert "(long long)" not in simplified

    # Ensure we can still parse the simplified expression.
    ast = parse_invariant_expr(simplified)
    assert ast is not None


def test_strip_unnecessary_typing_casts_removes_int128():
    inv = "(((__int128) x + 1) > 0) && ((unsigned __int128) y >= 0)"
    simplified = strip_unnecessary_typing_casts(inv)

    assert "(__int128)" not in simplified
    assert "(unsigned __int128)" not in simplified

    ast = parse_invariant_expr(simplified)
    assert ast is not None


def test_pretty_invariant_src_reduces_parentheses_but_keeps_required_ones():
    inv = "(((a + 1) % 10) <= x) && ((b * (c + 2)) > 0)"
    pretty = pretty_invariant_src(inv, strip_casts=False)

    # Must keep (a + 1) for the modulo, but should drop outermost noise.
    assert "(a + 1) % 10 <= x" in pretty
    assert "(((" not in pretty
