import os
import sys
import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.validate import syntactic_validation


@pytest.mark.parametrize(
    "expr",
    [
        # Nested boolean logic
        "(x > 0 && y < 10) || (z == 3 && !(w != 0))",
        # Arithmetic with precedence and parentheses
        "(a + b*2 - (c/3)) >= 0",
        # Array indexing and pointer-like expressions
        "arr[i+1] == arr[i] + 1",
        # Ternary used within comparison (valid expression context)
        "(flag ? x : y) >= 0",
        # Chained comparisons are not C, so use valid equivalent
        "x < y && y < z",
        # Bitwise in boolean context
        "(mask & 1) == 1",
        # Logical implication encoded
        "(!p) || q",
    ],
)
def test_valid_complex_expressions(expr):
    assert syntactic_validation(expr), f"Should be valid: {expr}"


@pytest.mark.parametrize(
    "expr",
    [
        # Assignments inside larger boolean expressions
        "(x > 0 && (y = 5))",
        "(a == b) || (c += 1)",
        # Side effects via increments/decrements
        "(i++) < n",
        "(--k) >= 0",
        # Function calls that include assignment-like side effects (heuristic)
        # We cannot know side-effects, but we can reject obvious assignments around calls
        "(foo(), x = 1) == 1",
        # Comma expressions that change state before a boolean
        "(x = 0, y == 1)",
        # Multiple assignments
        "(x = y) == (y = z)",
    ],
)
def test_invalid_with_side_effects(expr):
    assert not syntactic_validation(expr), f"Should be invalid (side effects): {expr}"


@pytest.mark.parametrize(
    "expr",
    [
        # Syntax errors
        "(x > 0 && y < 10",  # unclosed
        "x >>> 2",           # invalid token for C here
        "&& x == 0",         # operators only leading
        ")x == 0(",          # misplaced parens
    ],
)
def test_malformed_inputs(expr):
    assert not syntactic_validation(expr), f"Should be rejected (malformed): {expr}"


def test_whitespace_and_formatting():
    assert not syntactic_validation("   \t  \n"), "Whitespace-only must be invalid"
    assert syntactic_validation("  (  x  ==  0  )  &&  (  y  !=  1  ) "), "Spaces should not matter"


