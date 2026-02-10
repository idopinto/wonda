"""
Comprehensive pytest test suite for wonda/preprocess/gt_invariant_normalization.py

Tests are organized from easy to hard:
1. Basic parsing tests
2. AST to source conversion tests
3. Cast stripping tests (integral, pointer, combinations)
4. Parentheses preservation tests (all from paren_tests + edge cases)
5. Pretty printing tests (complex expressions, operators, associativity)
6. Helper function tests
7. Integration tests with real-world invariants
"""

import os
import sys

import pytest

# Keep the test style consistent with existing tests in this repo.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycparser import c_ast, c_parser

from wonda.preprocess.gt_invariant_normalization import (
    _extract_identifier_type_names,
    _is_integral_typename,
    _set_child,
    invariant_ast_to_pretty_src,
    invariant_ast_to_src,
    parse_invariant_expr,
    pretty_invariant_src,
    normalize_invariant,
    normalize_invariant_ast,
)


# =============================================================================
# SECTION 1: Basic Parsing Tests (Easy)
# =============================================================================


class TestParseInvariantExpr:
    """Test parsing of C expressions into AST nodes."""

    def test_parse_simple_identifier(self):
        """Parse a single identifier."""
        ast = parse_invariant_expr("x")
        assert ast is not None
        assert isinstance(ast, c_ast.ID)
        assert ast.name == "x"

    def test_parse_integer_constant(self):
        """Parse an integer constant."""
        ast = parse_invariant_expr("42")
        assert ast is not None
        assert isinstance(ast, c_ast.Constant)
        assert ast.value == "42"

    def test_parse_negative_constant(self):
        """Parse a negative constant."""
        ast = parse_invariant_expr("-1")
        assert ast is not None
        assert isinstance(ast, c_ast.UnaryOp)
        assert ast.op == "-"

    def test_parse_simple_binary_op(self):
        """Parse a simple binary operation."""
        ast = parse_invariant_expr("a + b")
        assert ast is not None
        assert isinstance(ast, c_ast.BinaryOp)
        assert ast.op == "+"

    def test_parse_comparison(self):
        """Parse a comparison expression."""
        ast = parse_invariant_expr("x > 0")
        assert ast is not None
        assert isinstance(ast, c_ast.BinaryOp)
        assert ast.op == ">"

    def test_parse_logical_and(self):
        """Parse a logical AND expression."""
        ast = parse_invariant_expr("a && b")
        assert ast is not None
        assert isinstance(ast, c_ast.BinaryOp)
        assert ast.op == "&&"

    def test_parse_logical_or(self):
        """Parse a logical OR expression."""
        ast = parse_invariant_expr("a || b")
        assert ast is not None
        assert isinstance(ast, c_ast.BinaryOp)
        assert ast.op == "||"

    def test_parse_nested_expression(self):
        """Parse a nested expression."""
        ast = parse_invariant_expr("(a + b) * c")
        assert ast is not None
        assert isinstance(ast, c_ast.BinaryOp)
        assert ast.op == "*"

    def test_parse_function_call(self):
        """Parse a function call expression."""
        ast = parse_invariant_expr("foo(x, y)")
        assert ast is not None
        assert isinstance(ast, c_ast.FuncCall)

    def test_parse_array_ref(self):
        """Parse an array reference."""
        ast = parse_invariant_expr("arr[i]")
        assert ast is not None
        assert isinstance(ast, c_ast.ArrayRef)

    def test_parse_unary_not(self):
        """Parse a unary NOT expression."""
        ast = parse_invariant_expr("!x")
        assert ast is not None
        assert isinstance(ast, c_ast.UnaryOp)
        assert ast.op == "!"

    def test_parse_ternary(self):
        """Parse a ternary expression."""
        ast = parse_invariant_expr("a ? b : c")
        assert ast is not None
        assert isinstance(ast, c_ast.TernaryOp)

    def test_parse_cast_expression(self):
        """Parse a cast expression."""
        ast = parse_invariant_expr("(long long) x")
        assert ast is not None
        assert isinstance(ast, c_ast.Cast)

    def test_parse_with_reused_parser(self):
        """Parse multiple expressions with the same parser instance."""
        parser = c_parser.CParser()
        ast1 = parse_invariant_expr("a + b", parser=parser)
        ast2 = parse_invariant_expr("c * d", parser=parser)
        assert ast1 is not None
        assert ast2 is not None

    def test_parse_deeply_nested(self):
        """Parse a deeply nested expression."""
        ast = parse_invariant_expr("((((a + b))))")
        assert ast is not None
        assert isinstance(ast, c_ast.BinaryOp)


# =============================================================================
# SECTION 2: AST to Source Conversion Tests (Easy)
# =============================================================================


class TestInvariantAstToSrc:
    """Test converting AST nodes back to C source."""

    def test_simple_identifier(self):
        """Convert a simple identifier back to source."""
        ast = parse_invariant_expr("x")
        src = invariant_ast_to_src(ast)
        assert "x" in src

    def test_binary_op(self):
        """Convert a binary operation back to source."""
        ast = parse_invariant_expr("a + b")
        src = invariant_ast_to_src(ast)
        assert "a" in src
        assert "b" in src
        assert "+" in src

    def test_preserves_cast(self):
        """Verify that casts are preserved when not stripped."""
        ast = parse_invariant_expr("(long long) x")
        src = invariant_ast_to_src(ast)
        assert "long long" in src


class TestInvariantAstToPrettySrc:
    """Test pretty printing AST nodes with minimal parentheses."""

    def test_simple_identifier(self):
        """Pretty print a simple identifier."""
        ast = parse_invariant_expr("x")
        src = invariant_ast_to_pretty_src(ast)
        assert src == "x"

    def test_constant(self):
        """Pretty print a constant."""
        ast = parse_invariant_expr("42")
        src = invariant_ast_to_pretty_src(ast)
        assert src == "42"

    def test_binary_op_no_extra_parens(self):
        """Binary op should not have extra parentheses."""
        ast = parse_invariant_expr("a + b")
        src = invariant_ast_to_pretty_src(ast)
        assert src == "a + b"

    def test_unary_op(self):
        """Pretty print unary operations."""
        ast = parse_invariant_expr("!x")
        src = invariant_ast_to_pretty_src(ast)
        assert src == "!x"

    def test_function_call_no_args(self):
        """Pretty print function call with no arguments."""
        ast = parse_invariant_expr("foo()")
        src = invariant_ast_to_pretty_src(ast)
        assert src == "foo()"

    def test_function_call_with_args(self):
        """Pretty print function call with arguments."""
        ast = parse_invariant_expr("foo(a, b, c)")
        src = invariant_ast_to_pretty_src(ast)
        assert src == "foo(a, b, c)"

    def test_array_ref(self):
        """Pretty print array reference."""
        ast = parse_invariant_expr("arr[i]")
        src = invariant_ast_to_pretty_src(ast)
        assert src == "arr[i]"

    def test_ternary(self):
        """Pretty print ternary expression."""
        ast = parse_invariant_expr("a ? b : c")
        src = invariant_ast_to_pretty_src(ast)
        assert src == "a ? b : c"


# =============================================================================
# SECTION 3: Cast Stripping Tests (Medium)
# =============================================================================


class TestStripUnnecessaryTypingCasts:
    """Test stripping of type casts from expressions."""

    # --- Integral cast removal ---

    def test_strip_long_long_cast(self):
        """Strip (long long) cast."""
        result = normalize_invariant("(long long) x")
        assert "long long" not in result
        assert "x" in result

    def test_strip_unsigned_int_cast(self):
        """Strip (unsigned int) cast."""
        result = normalize_invariant("(unsigned int) y")
        assert "unsigned" not in result
        assert "int" not in result

    def test_strip_signed_char_cast(self):
        """Strip (signed char) cast."""
        result = normalize_invariant("(signed char) z")
        assert "signed" not in result
        assert "char" not in result

    def test_strip_int128_cast(self):
        """Strip (__int128) cast."""
        result = normalize_invariant("(__int128) x + 1")
        assert "__int128" not in result

    def test_strip_unsigned_int128_cast(self):
        """Strip (unsigned __int128) cast."""
        result = normalize_invariant("(unsigned __int128) y >= 0")
        assert "__int128" not in result

    def test_strip_bool_cast(self):
        """Strip (_Bool) cast."""
        result = normalize_invariant("(_Bool) flag")
        assert "_Bool" not in result

    # def test_strip_bool_lowercase_cast(self):
    #     """Strip (bool) cast."""
    #     result = normalize_invariant("(bool) flag")
    #     assert "bool" not in result

    def test_strip_short_cast(self):
        """Strip (short) cast."""
        result = normalize_invariant("(short) val")
        assert "short" not in result

    def test_strip_unsigned_short_cast(self):
        """Strip (unsigned short) cast."""
        result = normalize_invariant("(unsigned short) val")
        assert "unsigned" not in result
        assert "short" not in result

    def test_strip_long_cast(self):
        """Strip (long) cast."""
        result = normalize_invariant("(long) val")
        assert "(long)" not in result

    def test_strip_unsigned_long_cast(self):
        """Strip (unsigned long) cast."""
        result = normalize_invariant("(unsigned long) val")
        assert "unsigned long" not in result

    def test_strip_unsigned_long_long_cast(self):
        """Strip (unsigned long long) cast."""
        result = normalize_invariant("(unsigned long long) val")
        assert "unsigned long long" not in result

    # --- Multiple casts in one expression ---

    def test_strip_multiple_casts(self):
        """Strip multiple casts in one expression."""
        inv = "((long long) a + 1) > ((unsigned int) b - 1)"
        result = normalize_invariant(inv)
        assert "long long" not in result
        assert "unsigned" not in result

    def test_strip_nested_casts(self):
        """Strip nested casts."""
        result = normalize_invariant("(long) (int) x")
        assert "(long)" not in result
        assert "(int)" not in result

    # --- Pointer cast preservation ---

    def test_preserve_pointer_cast_by_default(self):
        """Pointer casts should be preserved by default."""
        inv = "(int *) ptr"
        result = normalize_invariant(inv)
        # Pointer cast should be preserved
        assert "*" in result

    def test_remove_pointer_cast_when_remove_all(self):
        """Pointer casts should be removed when remove_all_casts=True."""
        inv = "(int *) ptr"
        result = normalize_invariant(inv, remove_all_casts=True)
        assert "*" not in result or result == "ptr"

    # --- Configuration options ---

    def test_remove_all_casts_option(self):
        """Test remove_all_casts option."""
        inv = "(long long) x"
        result = normalize_invariant(inv, remove_all_casts=True)
        assert "long" not in result

    def test_remove_integral_casts_false(self):
        """Test remove_integral_casts=False keeps integral casts."""
        inv = "(long long) x"
        result = normalize_invariant(inv, remove_integral_casts=False)
        # Should keep the cast
        assert "long long" in result

    # --- Pretty option ---

    def test_pretty_option(self):
        """Test pretty=True produces minimal parentheses output."""
        inv = "((a + b))"
        result = normalize_invariant(inv, pretty=True)
        assert result == "a + b"

    def test_not_pretty_option(self):
        """Test pretty=False uses default pycparser generator."""
        inv = "a + b"
        result = normalize_invariant(inv, pretty=False)
        # May have parentheses from pycparser
        assert "a" in result and "b" in result


class TestStripUnnecessaryTypingCastsAst:
    """Test the AST-level cast stripping function."""

    def test_strip_cast_returns_inner_expr(self):
        """Stripping a cast should return the inner expression."""
        ast = parse_invariant_expr("(long long) x")
        stripped = normalize_invariant_ast(ast)
        assert isinstance(stripped, c_ast.ID)
        assert stripped.name == "x"

    def test_strip_preserves_non_cast_nodes(self):
        """Non-cast nodes should be preserved."""
        ast = parse_invariant_expr("a + b")
        stripped = normalize_invariant_ast(ast)
        assert isinstance(stripped, c_ast.BinaryOp)


# =============================================================================
# SECTION 4: Parentheses Preservation Tests (Medium-Hard)
# =============================================================================


class TestParenthesesPreservationArithmetic:
    """Test parentheses preservation for arithmetic expressions."""

    def test_addition_before_multiplication_needs_parens(self):
        """(a + b) * c - parens needed."""
        result = normalize_invariant("((a + b) * c)", pretty=True)
        assert result == "(a + b) * c"

    def test_division_with_grouped_multiply_needs_parens(self):
        """a / (b * c) - parens needed."""
        result = normalize_invariant("(a / (b * c))", pretty=True)
        assert result == "a / (b * c)"

    def test_right_associative_subtraction_needs_parens(self):
        """a - (b - c) - parens needed."""
        result = normalize_invariant("(a - (b - c))", pretty=True)
        assert result == "a - (b - c)"

    def test_modulo_with_grouped_addition_needs_parens(self):
        """a % (b + c) - parens needed."""
        result = normalize_invariant("(a % (b + c))", pretty=True)
        assert result == "a % (b + c)"

    def test_multiplication_higher_precedence_no_parens(self):
        """(a * b) + c - parens NOT needed."""
        result = normalize_invariant("((a * b) + c)", pretty=True)
        assert result == "a * b + c"

    def test_multiplication_in_addition_no_parens(self):
        """a + (b * c) - parens NOT needed."""
        result = normalize_invariant("(a + (b * c))", pretty=True)
        assert result == "a + b * c"

    def test_left_associative_addition_no_parens(self):
        """(a + b) + c - parens NOT needed (left associative)."""
        result = normalize_invariant("((a + b) + c)", pretty=True)
        assert result == "a + b + c"

    def test_left_associative_multiplication(self):
        """(a * b) * c - parens NOT needed."""
        result = normalize_invariant("((a * b) * c)", pretty=True)
        assert result == "a * b * c"

    def test_right_associative_division_needs_parens(self):
        """a / (b / c) - parens needed."""
        result = normalize_invariant("(a / (b / c))", pretty=True)
        assert result == "a / (b / c)"

    def test_both_sides_need_parens_multiplication(self):
        """(a + b) * (c + d) - both sides need parens."""
        result = normalize_invariant("((a + b) * (c + d))", pretty=True)
        assert result == "(a + b) * (c + d)"


class TestParenthesesPreservationLogical:
    """Test parentheses preservation for logical expressions."""

    def test_or_before_and_needs_parens(self):
        """(a || b) && c - parens needed."""
        result = normalize_invariant("((a || b) && c)", pretty=True)
        assert result == "(a || b) && c"

    def test_or_grouped_in_and_needs_parens(self):
        """a && (b || c) - parens needed."""
        result = normalize_invariant("(a && (b || c))", pretty=True)
        assert result == "a && (b || c)"

    def test_and_higher_precedence_no_parens_left(self):
        """(a && b) || c - parens NOT needed."""
        result = normalize_invariant("((a && b) || c)", pretty=True)
        assert result == "a && b || c"

    def test_and_higher_precedence_no_parens_right(self):
        """a || (b && c) - parens NOT needed."""
        result = normalize_invariant("(a || (b && c))", pretty=True)
        assert result == "a || b && c"

    def test_both_or_need_parens_in_and(self):
        """(a || b) && (c || d) - both OR need parens."""
        result = normalize_invariant("((a || b) && (c || d))", pretty=True)
        assert result == "(a || b) && (c || d)"

    def test_chain_of_and(self):
        """a && b && c - no parens needed."""
        result = normalize_invariant("((a && b) && c)", pretty=True)
        assert result == "a && b && c"

    def test_chain_of_or(self):
        """a || b || c - no parens needed."""
        result = normalize_invariant("((a || b) || c)", pretty=True)
        assert result == "a || b || c"


class TestParenthesesPreservationComparison:
    """Test parentheses preservation for comparison expressions."""

    def test_addition_higher_than_comparison(self):
        """(a + 1) > b - parens NOT needed."""
        result = normalize_invariant("((a + 1) > b)", pretty=True)
        assert result == "a + 1 > b"

    def test_subtraction_higher_than_comparison(self):
        """a > (b - 1) - parens NOT needed."""
        result = normalize_invariant("(a > (b - 1))", pretty=True)
        assert result == "a > b - 1"

    def test_comparison_in_logical(self):
        """(a > b) && (c < d) - parens NOT needed."""
        result = normalize_invariant("((a > b) && (c < d))", pretty=True)
        assert result == "a > b && c < d"

    def test_equality_comparison(self):
        """a == b - no extra parens."""
        result = normalize_invariant("(a == b)", pretty=True)
        assert result == "a == b"

    def test_inequality_comparison(self):
        """a != b - no extra parens."""
        result = normalize_invariant("(a != b)", pretty=True)
        assert result == "a != b"


class TestParenthesesPreservationNegation:
    """Test parentheses preservation for negation expressions."""

    def test_negation_of_conjunction_needs_parens(self):
        """!(a && b) - needs parens."""
        result = normalize_invariant("(!(a && b))", pretty=True)
        assert result == "!(a && b)"

    def test_negation_binds_tighter_no_parens(self):
        """(!a) && b - no parens needed around !a."""
        result = normalize_invariant("((!a) && b)", pretty=True)
        assert result == "!a && b"

    def test_negation_of_disjunction_needs_parens(self):
        """!(a || b) - needs parens."""
        result = normalize_invariant("(!(a || b))", pretty=True)
        assert result == "!(a || b)"

    def test_double_negation(self):
        """!!a - no parens needed."""
        result = normalize_invariant("!!a", pretty=True)
        assert result == "!!a"

    def test_negation_of_comparison(self):
        """!(a > b) - needs parens."""
        result = normalize_invariant("(!(a > b))", pretty=True)
        assert result == "!(a > b)"


class TestParenthesesPreservationBitwise:
    """Test parentheses preservation for bitwise expressions."""

    def test_bitwise_and_in_or(self):
        """(a & b) | c - parens NOT needed (& has higher precedence)."""
        result = normalize_invariant("((a & b) | c)", pretty=True)
        assert result == "a & b | c"

    def test_bitwise_or_in_and(self):
        """a & (b | c) - parens needed."""
        result = normalize_invariant("(a & (b | c))", pretty=True)
        assert result == "a & (b | c)"

    def test_bitwise_xor(self):
        """a ^ b - no extra parens."""
        result = normalize_invariant("(a ^ b)", pretty=True)
        assert result == "a ^ b"

    def test_shift_operations(self):
        """a << b - no extra parens."""
        result = normalize_invariant("(a << b)", pretty=True)
        assert result == "a << b"

    def test_right_shift(self):
        """a >> b - no extra parens."""
        result = normalize_invariant("(a >> b)", pretty=True)
        assert result == "a >> b"

    def test_shift_associativity(self):
        """a << (b << c) - parens needed for right-side same-precedence."""
        result = normalize_invariant("(a << (b << c))", pretty=True)
        assert result == "a << (b << c)"


class TestParenthesesPreservationMixed:
    """Test parentheses preservation for mixed operator expressions."""

    def test_arithmetic_and_comparison_and_logical(self):
        """Complex mixed expression."""
        result = normalize_invariant(
            "((a + 1 > 0) && (b * 2 < 10))", pretty=True
        )
        assert result == "a + 1 > 0 && b * 2 < 10"

    def test_deep_nesting(self):
        """Deeply nested expression."""
        result = normalize_invariant("(((((a + b)))))", pretty=True)
        assert result == "a + b"

    def test_complex_invariant_style(self):
        """Invariant-style complex expression."""
        result = normalize_invariant(
            "((x >= 0) && (x <= n) && (y > 0))", pretty=True
        )
        assert result == "x >= 0 && x <= n && y > 0"


# =============================================================================
# SECTION 5: Pretty Invariant Src Tests (Medium)
# =============================================================================


class TestPrettyInvariantSrc:
    """Test the convenience pretty_invariant_src function."""

    def test_simple_expression(self):
        """Simple expression pretty printed."""
        result = pretty_invariant_src("a + b")
        assert result == "a + b"

    def test_with_cast_stripping(self):
        """Pretty print with cast stripping."""
        result = pretty_invariant_src("(long long) x + 1", strip_casts=True)
        assert "long long" not in result
        assert "x + 1" in result

    def test_without_cast_stripping(self):
        """Pretty print without cast stripping."""
        result = pretty_invariant_src("(long long) x", strip_casts=False)
        assert "long long" in result

    def test_remove_all_casts_option(self):
        """Test remove_all_casts option in pretty_invariant_src."""
        result = pretty_invariant_src(
            "(int *) ptr", strip_casts=True, remove_all_casts=True
        )
        assert result == "ptr"


# =============================================================================
# SECTION 6: Helper Function Tests (Medium)
# =============================================================================


class TestIsIntegralTypename:
    """Test _is_integral_typename helper function."""

    def test_long_long_is_integral(self):
        """long long is integral."""
        ast = parse_invariant_expr("(long long) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_unsigned_int_is_integral(self):
        """unsigned int is integral."""
        ast = parse_invariant_expr("(unsigned int) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_int_is_integral(self):
        """int is integral."""
        ast = parse_invariant_expr("(int) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_char_is_integral(self):
        """char is integral."""
        ast = parse_invariant_expr("(char) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_short_is_integral(self):
        """short is integral."""
        ast = parse_invariant_expr("(short) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_signed_is_integral(self):
        """signed is integral."""
        ast = parse_invariant_expr("(signed) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_unsigned_is_integral(self):
        """unsigned is integral."""
        ast = parse_invariant_expr("(unsigned) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_int128_is_integral(self):
        """__int128 is integral."""
        ast = parse_invariant_expr("(__int128) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_bool_is_integral(self):
        """_Bool is integral."""
        ast = parse_invariant_expr("(_Bool) x")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is True

    def test_pointer_is_not_integral(self):
        """Pointer types are not integral."""
        ast = parse_invariant_expr("(int *) ptr")
        assert isinstance(ast, c_ast.Cast)
        assert _is_integral_typename(ast.to_type) is False


class TestExtractIdentifierTypeNames:
    """Test _extract_identifier_type_names helper function."""

    def test_extract_long_long(self):
        """Extract (long, long) from long long."""
        ast = parse_invariant_expr("(long long) x")
        assert isinstance(ast, c_ast.Cast)
        names = _extract_identifier_type_names(ast.to_type.type)
        assert names == ("long", "long")

    def test_extract_unsigned_int(self):
        """Extract (unsigned, int) from unsigned int."""
        ast = parse_invariant_expr("(unsigned int) x")
        assert isinstance(ast, c_ast.Cast)
        names = _extract_identifier_type_names(ast.to_type.type)
        assert names == ("unsigned", "int")

    def test_extract_int(self):
        """Extract (int,) from int."""
        ast = parse_invariant_expr("(int) x")
        assert isinstance(ast, c_ast.Cast)
        names = _extract_identifier_type_names(ast.to_type.type)
        assert names == ("int",)


class TestSetChild:
    """Test _set_child helper function."""

    def test_set_simple_child(self):
        """Set a simple child attribute."""
        ast = parse_invariant_expr("a + b")
        new_child = c_ast.ID("c")
        _set_child(ast, "left", new_child)
        assert ast.left.name == "c"

    def test_set_list_child(self):
        """Set a child in a list (like exprs[0])."""
        ast = parse_invariant_expr("foo(a, b)")
        new_child = c_ast.ID("c")
        _set_child(ast.args, "exprs[0]", new_child)
        assert ast.args.exprs[0].name == "c"


# =============================================================================
# SECTION 7: Integration Tests with Real-World Invariants (Hard)
# =============================================================================


class TestRealWorldInvariants:
    """Test with real-world invariant examples from UAutomizer."""

    def test_uautomizer_prime_count_invariant(self):
        """Test a real UAutomizer-style invariant."""
        inv = (
            "((((prime_count <= x) && ((((long long) prime_count + 1) % 4294967296) <= x)) "
            "&& ((((long long) 2 + prime_count) % 4294967296) <= ((long long) n + 1))) "
            "|| ((prime_count == 0) && (n == 0)))"
        )
        result = normalize_invariant(inv, pretty=True)

        # All casts should be removed
        assert "(long long)" not in result
        assert "long long" not in result

        # The result should still be parseable
        ast = parse_invariant_expr(result)
        assert ast is not None

    def test_int128_mixed_invariant(self):
        """Test invariant with __int128 casts."""
        inv = "(((__int128) x + 1) > 0) && ((unsigned __int128) y >= 0)"
        result = normalize_invariant(inv, pretty=True)

        assert "__int128" not in result
        assert "unsigned" not in result

        # Should still be parseable
        ast = parse_invariant_expr(result)
        assert ast is not None

    def test_multiple_comparison_chain(self):
        """Test chain of comparisons in logical expression."""
        inv = "(((x >= 0) && (x <= 100)) && ((y > x) || (z == 0)))"
        result = normalize_invariant(inv, pretty=True)

        # Should have minimal parens but preserve meaning
        ast = parse_invariant_expr(result)
        assert ast is not None
        assert "x >= 0" in result

    def test_complex_arithmetic_with_modulo(self):
        """Test complex arithmetic with modulo operations."""
        inv = "((((a + 1) % 4294967296) <= b) && ((c * 2) > 0))"
        result = normalize_invariant(inv, pretty=True)

        assert "(a + 1) % 4294967296" in result

    def test_nested_function_calls(self):
        """Test expression with nested function calls."""
        inv = "(foo(bar(x), y) > 0) && (baz(a, b, c) == 1)"
        result = normalize_invariant(inv, pretty=True)

        assert "foo(bar(x), y)" in result
        assert "baz(a, b, c)" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_variable(self):
        """Single variable expression."""
        result = normalize_invariant("x", pretty=True)
        assert result == "x"

    def test_single_constant(self):
        """Single constant expression."""
        result = normalize_invariant("42", pretty=True)
        assert result == "42"

    def test_negative_number(self):
        """Negative number."""
        result = normalize_invariant("-1", pretty=True)
        assert result == "-1"

    def test_hexadecimal_constant(self):
        """Hexadecimal constant."""
        result = normalize_invariant("0xDEADBEEF", pretty=True)
        assert "0xDEADBEEF" in result or "DEADBEEF" in result.upper()

    def test_large_number(self):
        """Large number constant."""
        result = normalize_invariant("4294967296", pretty=True)
        assert result == "4294967296"

    def test_empty_function_args(self):
        """Function call with no arguments."""
        result = normalize_invariant("func()", pretty=True)
        assert result == "func()"

    def test_array_with_expression_index(self):
        """Array with expression as index."""
        result = normalize_invariant("arr[i + 1]", pretty=True)
        assert result == "arr[i + 1]"

    def test_multiple_array_refs(self):
        """Multiple array references."""
        result = normalize_invariant("arr[i][j]", pretty=True)
        assert "arr" in result and "i" in result and "j" in result

    def test_unary_plus(self):
        """Unary plus operator."""
        result = normalize_invariant("+x", pretty=True)
        assert result == "+x"

    def test_bitwise_not(self):
        """Bitwise NOT operator."""
        result = normalize_invariant("~x", pretty=True)
        assert result == "~x"

    def test_increment_postfix(self):
        """Postfix increment."""
        result = normalize_invariant("x++", pretty=True)
        assert result == "x++"

    def test_decrement_postfix(self):
        """Postfix decrement."""
        result = normalize_invariant("x--", pretty=True)
        assert result == "x--"

    def test_ternary_nested(self):
        """Nested ternary expression."""
        result = normalize_invariant("a ? b ? c : d : e", pretty=True)
        assert "?" in result and ":" in result

    def test_very_long_identifier(self):
        """Very long identifier name."""
        long_name = "very_long_variable_name_that_goes_on_and_on"
        result = normalize_invariant(long_name, pretty=True)
        assert result == long_name

    def test_cast_of_expression(self):
        """Cast of a complex expression."""
        result = normalize_invariant("(long long) (a + b)", pretty=True)
        assert "long long" not in result
        assert "a + b" in result

    def test_multiple_casts_same_subexpr(self):
        """Multiple casts on the same sub-expression."""
        result = normalize_invariant("(long) (int) (short) x", pretty=True)
        assert result == "x"


class TestAssociativityChains:
    """Test handling of associativity in operator chains."""

    def test_long_and_chain(self):
        """Long chain of AND operations."""
        result = normalize_invariant("(((a && b) && c) && d)", pretty=True)
        assert result == "a && b && c && d"

    def test_long_or_chain(self):
        """Long chain of OR operations."""
        result = normalize_invariant("(((a || b) || c) || d)", pretty=True)
        assert result == "a || b || c || d"

    def test_long_add_chain(self):
        """Long chain of additions."""
        result = normalize_invariant("(((a + b) + c) + d)", pretty=True)
        assert result == "a + b + c + d"

    def test_long_mult_chain(self):
        """Long chain of multiplications."""
        result = normalize_invariant("(((a * b) * c) * d)", pretty=True)
        assert result == "a * b * c * d"

    def test_mixed_add_sub_chain(self):
        """Mixed addition and subtraction (not associative)."""
        result = normalize_invariant("a - (b - c)", pretty=True)
        assert result == "a - (b - c)"

    def test_mixed_mult_div_chain(self):
        """Mixed multiplication and division (not associative for div)."""
        result = normalize_invariant("a / (b / c)", pretty=True)
        assert result == "a / (b / c)"


class TestSpecialOperators:
    """Test handling of special operators."""

    def test_struct_ref_dot(self):
        """Struct member access with dot."""
        result = normalize_invariant("s.field", pretty=True)
        assert result == "s.field"

    def test_struct_ref_arrow(self):
        """Struct member access with arrow."""
        result = normalize_invariant("p->field", pretty=True)
        assert result == "p->field"

    def test_sizeof_style_expression(self):
        """Expression that might look like sizeof."""
        result = normalize_invariant("size + 1", pretty=True)
        assert result == "size + 1"


class TestAllParenTestsFromMain:
    """Run all the test cases from the paren_tests list in __main__."""

    @pytest.mark.parametrize(
        "original,expected,description",
        [
            # Arithmetic precedence - parens needed
            ("((a + b) * c)", "(a + b) * c", "Addition before multiplication"),
            ("(a / (b * c))", "a / (b * c)", "Division with grouped multiply"),
            ("(a - (b - c))", "a - (b - c)", "Right-associative subtraction"),
            ("(a % (b + c))", "a % (b + c)", "Modulo with grouped addition"),
            # Arithmetic precedence - parens NOT needed
            ("((a * b) + c)", "a * b + c", "Multiplication already higher precedence"),
            (
                "(a + (b * c))",
                "a + b * c",
                "Multiplication already higher precedence v2",
            ),
            ("((a + b) + c)", "a + b + c", "Left-associative addition"),
            # Logical precedence - parens needed (|| vs &&)
            ("((a || b) && c)", "(a || b) && c", "OR before AND"),
            ("(a && (b || c))", "a && (b || c)", "OR grouped in AND"),
            # Logical precedence - parens NOT needed
            ("((a && b) || c)", "a && b || c", "AND already higher precedence"),
            ("(a || (b && c))", "a || b && c", "AND already higher precedence v2"),
            # Comparison with arithmetic
            ("((a + 1) > b)", "a + 1 > b", "Addition already higher than comparison"),
            (
                "(a > (b - 1))",
                "a > b - 1",
                "Subtraction already higher than comparison",
            ),
            # Negation
            ("(!(a && b))", "!(a && b)", "Negation of conjunction - needs parens"),
            ("((!a) && b)", "!a && b", "Negation binds tighter - no parens needed"),
            # Mixed - complex cases
            ("((a + b) * (c + d))", "(a + b) * (c + d)", "Both sides need parens"),
            (
                "((a || b) && (c || d))",
                "(a || b) && (c || d)",
                "Both OR need parens in AND",
            ),
        ],
    )
    def test_paren_preservation(self, original, expected, description):
        """Parameterized test for all paren_tests cases."""
        result = normalize_invariant(original, pretty=True)
        assert result == expected, f"Failed: {description}"


class TestCombinedCastAndParens:
    """Test combinations of cast stripping and parenthesis handling."""

    def test_cast_in_arithmetic(self):
        """Cast inside arithmetic expression."""
        result = normalize_invariant("((long long) a + b) * c", pretty=True)
        assert "long long" not in result
        assert "(a + b) * c" in result

    def test_cast_on_both_sides(self):
        """Casts on both sides of operator."""
        result = normalize_invariant(
            "((long long) a > (int) b)", pretty=True
        )
        assert "long long" not in result
        assert "(int)" not in result
        assert "a > b" in result

    def test_cast_in_complex_expression(self):
        """Cast in a complex expression with multiple precedence levels."""
        result = normalize_invariant(
            "(((long long) x + 1) > 0) && (y < ((unsigned) z * 2))",
            pretty=True,
        )
        assert "long long" not in result
        assert "unsigned" not in result
        # Check structure is preserved
        assert "x + 1 > 0" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
