"""
Semantic equivalence checking for C boolean expressions using Z3.

This module provides functionality to check if two C-style invariants
are logically equivalent using the Z3 SMT solver.
"""

from typing import Optional
from z3 import (
    Int,
    And,
    Or,
    Not,
    Xor,
    Solver,
    unsat,
    ArithRef,
    BoolRef,
    ExprRef,
)
from pycparser import c_ast

from src.preprocess.clean_invariants import parse_invariant_expr


class Z3ConversionError(Exception):
    """Raised when conversion from C AST to Z3 fails."""
    pass


def _to_bool(expr: ExprRef) -> BoolRef:
    """
    Convert an expression to boolean for use in And/Or/Not.
    
    In C semantics, an integer is true if non-zero.
    If expr is already a BoolRef, return as-is.
    If expr is an ArithRef (integer), return expr != 0.
    """
    if isinstance(expr, BoolRef):
        return expr
    # Assume it's an arithmetic expression - C semantics: non-zero is true
    return expr != 0


def ast_to_z3(
    node: c_ast.Node,
    variables: dict[str, ArithRef],
) -> ExprRef:
    """
    Recursively convert a pycparser AST node to a Z3 expression.

    Args:
        node: pycparser AST node
        variables: Dictionary mapping variable names to Z3 Int variables

    Returns:
        Z3 expression (ArithRef for arithmetic, BoolRef for boolean)

    Raises:
        Z3ConversionError: If the AST node cannot be converted
    """
    if isinstance(node, c_ast.ID):
        name = node.name
        if name not in variables:
            # Auto-create variable if not provided
            variables[name] = Int(name)
        return variables[name]

    if isinstance(node, c_ast.Constant):
        # Handle integer constants
        val_str = node.value.rstrip("uUlL")
        try:
            if val_str.startswith("0x") or val_str.startswith("0X"):
                return int(val_str, 16)
            elif val_str.startswith("0") and len(val_str) > 1 and val_str[1:].isdigit():
                return int(val_str, 8)
            return int(val_str)
        except ValueError:
            raise Z3ConversionError(f"Cannot convert constant: {node.value}")

    if isinstance(node, c_ast.UnaryOp):
        operand = ast_to_z3(node.expr, variables)
        op = node.op
        if op == "-":
            return -operand
        if op == "!":
            # Logical NOT requires boolean operand
            return Not(_to_bool(operand))
        if op == "+":
            return operand
        raise Z3ConversionError(f"Unsupported unary operator: {op}")

    if isinstance(node, c_ast.BinaryOp):
        left = ast_to_z3(node.left, variables)
        right = ast_to_z3(node.right, variables)
        op = node.op

        # Arithmetic operators
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "%":
            return left % right

        # Comparison operators
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right

        # Logical operators - convert operands to boolean (C semantics: non-zero is true)
        if op == "&&":
            return And(_to_bool(left), _to_bool(right))
        if op == "||":
            return Or(_to_bool(left), _to_bool(right))

        # Bitwise operators (treat as arithmetic for now)
        if op == "&":
            return left & right
        if op == "|":
            return left | right
        if op == "^":
            return left ^ right
        if op == "<<":
            return left << right
        if op == ">>":
            return left >> right

        raise Z3ConversionError(f"Unsupported binary operator: {op}")

    if isinstance(node, c_ast.TernaryOp):
        cond = ast_to_z3(node.cond, variables)
        iftrue = ast_to_z3(node.iftrue, variables)
        iffalse = ast_to_z3(node.iffalse, variables)
        # Z3 If: If(cond, then, else)
        from z3 import If
        return If(cond, iftrue, iffalse)

    if isinstance(node, c_ast.Cast):
        # Ignore casts for Z3 conversion (treat as no-op)
        return ast_to_z3(node.expr, variables)

    raise Z3ConversionError(f"Unsupported AST node type: {type(node).__name__}")


def parse_invariant_to_z3(
    inv_str: str,
    variables: Optional[dict[str, ArithRef]] = None,
) -> tuple[ExprRef, dict[str, ArithRef]]:
    """
    Parse a C-style invariant string to a Z3 expression.

    Args:
        inv_str: C boolean expression string (e.g., "x > 0 && y <= 10")
        variables: Optional dict of pre-defined Z3 variables.
                   If None, variables are auto-created as Z3 Ints.

    Returns:
        Tuple of (Z3 expression, variables dict)

    Raises:
        Z3ConversionError: If parsing or conversion fails
    """
    if variables is None:
        variables = {}

    ast = parse_invariant_expr(inv_str)
    if ast is None:
        raise Z3ConversionError(f"Failed to parse invariant: {inv_str}")

    z3_expr = ast_to_z3(ast, variables)
    return z3_expr, variables


def are_semantically_equivalent(
    inv1: str,
    inv2: str,
    var_names: Optional[list[str]] = None,
) -> bool:
    """
    Check if two invariants are logically equivalent using Z3.

    Two formulas are equivalent iff their XOR is unsatisfiable
    (i.e., there's no assignment that makes one true and the other false).

    Args:
        inv1: First C boolean expression
        inv2: Second C boolean expression
        var_names: Optional list of variable names. If None, variables
                   are auto-discovered from the expressions.

    Returns:
        True if the invariants are semantically equivalent, False otherwise
    """
    # Create shared variables dict
    variables: dict[str, ArithRef] = {}
    if var_names:
        for name in var_names:
            variables[name] = Int(name)

    try:
        expr1, variables = parse_invariant_to_z3(inv1, variables)
        expr2, variables = parse_invariant_to_z3(inv2, variables)
    except Z3ConversionError as e:
        print(f"Warning: Z3 conversion failed: {e}")
        return False

    # Check equivalence: XOR(expr1, expr2) is UNSAT means they're equivalent
    solver = Solver()
    solver.add(Xor(expr1, expr2))

    return solver.check() == unsat


def is_implied_by(
    assumption: str,
    conclusion: str,
    var_names: Optional[list[str]] = None,
) -> bool:
    """
    Check if assumption implies conclusion (assumption => conclusion).

    Args:
        assumption: The premise expression
        conclusion: The conclusion expression
        var_names: Optional list of variable names

    Returns:
        True if assumption logically implies conclusion
    """
    variables: dict[str, ArithRef] = {}
    if var_names:
        for name in var_names:
            variables[name] = Int(name)

    try:
        expr_assumption, variables = parse_invariant_to_z3(assumption, variables)
        expr_conclusion, variables = parse_invariant_to_z3(conclusion, variables)
    except Z3ConversionError as e:
        print(f"Warning: Z3 conversion failed: {e}")
        return False

    # Convert to boolean for logical operations
    bool_assumption = _to_bool(expr_assumption)
    bool_conclusion = _to_bool(expr_conclusion)

    # Check implication: assumption && !conclusion is UNSAT means implication holds
    solver = Solver()
    solver.add(And(bool_assumption, Not(bool_conclusion)))

    return solver.check() == unsat


def deduplicate_by_equivalence(
    invariants: list[str],
    var_names: Optional[list[str]] = None,
) -> list[str]:
    """
    Remove semantically equivalent invariants, keeping only unique ones.

    Args:
        invariants: List of invariant strings
        var_names: Optional list of variable names

    Returns:
        List of semantically unique invariants
    """
    if not invariants:
        return []

    unique: list[str] = []
    for inv in invariants:
        # Check if this invariant is equivalent to any already in unique
        is_duplicate = False
        for existing in unique:
            if are_semantically_equivalent(inv, existing, var_names):
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(inv)

    return unique


# =============================================================================
# Tests
# =============================================================================

def test_basic_equivalence():
    """Test basic semantic equivalence cases."""
    print("=" * 60)
    print("Testing Semantic Equivalence")
    print("=" * 60)

    test_cases = [
        # (inv1, inv2, expected_equivalent, description)
        ("x > 0", "x >= 1", True, "x > 0 <=> x >= 1 (for integers)"),
        ("x >= 1", "1 <= x", True, "x >= 1 <=> 1 <= x (operand swap)"),
        ("x > 0", "x >= 0", False, "x > 0 NOT<=> x >= 0"),
        ("a && b", "b && a", True, "a && b <=> b && a (commutativity)"),
        ("a || b", "b || a", True, "a || b <=> b || a (commutativity)"),
        ("!(x < 5)", "x >= 5", True, "!(x < 5) <=> x >= 5"),
        ("!(x <= 5)", "x > 5", True, "!(x <= 5) <=> x > 5"),
        ("x + 1 > 0", "x >= 0", True, "x + 1 > 0 <=> x >= 0"),
        ("x - 1 >= 0", "x >= 1", True, "x - 1 >= 0 <=> x >= 1"),
        ("1 <= i && i <= 6", "i >= 1 && i <= 6", True, "Different syntax, same meaning"),
        ("a == b && b == c", "b == a && c == b", True, "Reordered equalities"),
        ("x > 0 && x < 10", "x >= 1 && x <= 9", True, "Equivalent bounds"),
        ("x * 2 > 0", "x > 0", True, "x * 2 > 0 <=> x > 0 (for positive multiplier)"),
    ]

    passed = 0
    failed = 0

    for inv1, inv2, expected, description in test_cases:
        try:
            result = are_semantically_equivalent(inv1, inv2)
            status = "[PASS]" if result == expected else "[FAIL]"
            if result == expected:
                passed += 1
            else:
                failed += 1
            print(f"{status} {description}")
            print(f"   '{inv1}' vs '{inv2}'")
            print(f"   Expected: {expected}, Got: {result}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {description}")
            print(f"   Error: {e}")
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_implication():
    """Test logical implication."""
    print("\n" + "=" * 60)
    print("Testing Logical Implication")
    print("=" * 60)

    test_cases = [
        ("x > 5", "x > 0", True, "x > 5 => x > 0"),
        ("x > 0", "x > 5", False, "x > 0 =/=> x > 5"),
        ("x == 5", "x >= 0 && x <= 10", True, "x == 5 => 0 <= x <= 10"),
        ("a && b", "a", True, "a && b => a"),
        ("a", "a || b", True, "a => a || b"),
    ]

    passed = 0
    failed = 0

    for assumption, conclusion, expected, description in test_cases:
        try:
            result = is_implied_by(assumption, conclusion)
            status = "[PASS]" if result == expected else "[FAIL]"
            if result == expected:
                passed += 1
            else:
                failed += 1
            print(f"{status} {description}")
            print(f"   Expected: {expected}, Got: {result}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {description}")
            print(f"   Error: {e}")
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_deduplication():
    """Test deduplication of equivalent invariants."""
    print("\n" + "=" * 60)
    print("Testing Deduplication")
    print("=" * 60)

    invariants = [
        "x > 0",
        "x >= 1",          # Equivalent to x > 0
        "y <= 10",
        "10 >= y",         # Equivalent to y <= 10
        "a && b",
        "b && a",          # Equivalent to a && b
        "z == 5",
    ]

    print(f"Input invariants ({len(invariants)}):")
    for inv in invariants:
        print(f"  - {inv}")

    unique = deduplicate_by_equivalence(invariants)

    print(f"\nUnique invariants ({len(unique)}):")
    for inv in unique:
        print(f"  - {inv}")

    expected_count = 4  # x > 0, y <= 10, a && b, z == 5
    status = "[PASS]" if len(unique) == expected_count else "[FAIL]"
    print(f"\n{status} Expected {expected_count} unique, got {len(unique)}")

    return len(unique) == expected_count


if __name__ == "__main__":
    all_passed = True
    all_passed &= test_basic_equivalence()
    all_passed &= test_implication()
    all_passed &= test_deduplication()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)
