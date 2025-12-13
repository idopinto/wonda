import sys
import os
import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.equivalence import check_semantic_equivalence


class TestSemanticEquivalence:
    """Test suite for check_semantic_equivalence function."""
    
    def test_identical_predicates(self):
        """Test that identical predicates are equivalent."""
        assert check_semantic_equivalence("x > 0", "x > 0")
        assert check_semantic_equivalence("a == b", "a == b")
        assert check_semantic_equivalence("x + y == 10", "x + y == 10")
    
    def test_commutative_operators(self):
        """Test that predicates with commutative operators are equivalent regardless of order."""
        # Addition is commutative
        assert check_semantic_equivalence("x + y == 10", "y + x == 10")
        
        # AND is commutative
        assert check_semantic_equivalence("x > 0 && y > 0", "y > 0 && x > 0")
        assert check_semantic_equivalence("a == 1 && b == 2 && c == 3", "c == 3 && a == 1 && b == 2")
        
        # OR is commutative
        assert check_semantic_equivalence("x > 0 || y > 0", "y > 0 || x > 0")
        
        # Multiplication is commutative
        assert check_semantic_equivalence("a * b == 10", "b * a == 10")
        assert check_semantic_equivalence("x * y * z == 0", "z * x * y == 0")
    
    def test_equality_vs_inequality_equivalence(self):
        """Test equivalence between equality and double inequality."""
        # The original problematic case
        assert check_semantic_equivalence(
            "q + a * b * p == (long long)x * y",
            "a * b * p + q <= (long long)x * y && a * b * p + q >= (long long)x * y"
        )
        
        # Simpler case
        assert check_semantic_equivalence("x == 5", "x <= 5 && x >= 5")
        assert check_semantic_equivalence("a + b == c", "a + b <= c && a + b >= c")
    
    def test_arithmetic_rearrangement(self):
        """Test that arithmetic rearrangements are equivalent."""
        # Addition associativity
        assert check_semantic_equivalence("(x + y) + z == 10", "x + (y + z) == 10")
        assert check_semantic_equivalence("x + y + z == 10", "x + (y + z) == 10")
        
        # Multiplication associativity
        assert check_semantic_equivalence("(a * b) * c == 10", "a * (b * c) == 10")
        
        # Distributivity
        assert check_semantic_equivalence("x * (y + z) == 10", "x * y + x * z == 10")
        assert check_semantic_equivalence("(a + b) * c == 10", "a * c + b * c == 10")
    
    def test_logical_equivalences(self):
        """Test logical equivalences (De Morgan's laws, etc.)."""
        # De Morgan's law: !(A && B) ≡ !A || !B
        assert check_semantic_equivalence("!(x > 0 && y > 0)", "!(x > 0) || !(y > 0)")
        assert check_semantic_equivalence("!(x > 0 && y > 0)", "x <= 0 || y <= 0")
        
        # De Morgan's law: !(A || B) ≡ !A && !B
        assert check_semantic_equivalence("!(x > 0 || y > 0)", "!(x > 0) && !(y > 0)")
        assert check_semantic_equivalence("!(x > 0 || y > 0)", "x <= 0 && y <= 0")
        
        # Double negation
        assert check_semantic_equivalence("!!(x > 0)", "x > 0")
    
    def test_inequality_equivalences(self):
        """Test that different inequality formulations are equivalent."""
        # x > y ≡ y < x
        assert check_semantic_equivalence("x > y", "y < x")
        
        # x >= y ≡ y <= x
        assert check_semantic_equivalence("x >= y", "y <= x")
        
        # x != y ≡ !(x == y)
        assert check_semantic_equivalence("x != y", "!(x == y)")
    
    def test_type_casts_ignored(self):
        """Test that type casts are properly ignored (since we use unbounded Ints)."""
        assert check_semantic_equivalence("(long long)x == 5", "x == 5")
        assert check_semantic_equivalence("(int)x * (int)y == 10", "x * y == 10")
    
    def test_non_equivalent_predicates(self):
        """Test that non-equivalent predicates are correctly identified as not equivalent."""
        # Strict inequality vs non-strict
        assert not check_semantic_equivalence("x > 0", "x >= 0")
        assert not check_semantic_equivalence("x < 0", "x <= 0")
        
        # Different constants
        assert not check_semantic_equivalence("x == 5", "x == 10")
        
        # Different operators
        assert not check_semantic_equivalence("x > 0", "x == 0")
        assert not check_semantic_equivalence("x + y == 10", "x - y == 10")
        
        # AND vs OR
        assert not check_semantic_equivalence("x > 0 && y > 0", "x > 0 || y > 0")
        
        # Different arithmetic
        assert not check_semantic_equivalence("x * y == 10", "x + y == 10")
    
    def test_complex_real_world_cases(self):
        """Test complex cases that might appear in actual loop invariants."""
        # Product computation invariant (from the original problem)
        assert check_semantic_equivalence(
            "q + a * b * p == x * y",
            "a * b * p + q == x * y"
        )
        
        # Sum of array elements
        assert check_semantic_equivalence(
            "sum == 0",
            "0 == sum"
        )
        
        # Range checks
        assert check_semantic_equivalence(
            "i >= 0 && i < n",
            "i < n && i >= 0"
        )
        
        # Multiple conditions
        assert check_semantic_equivalence(
            "x > 0 && y > 0 && z > 0",
            "z > 0 && x > 0 && y > 0"
        )
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single variable comparisons
        assert check_semantic_equivalence("x == 0", "0 == x")
        
        # Constants only
        assert check_semantic_equivalence("1 == 1", "1 == 1")
        
        # Negative numbers
        assert check_semantic_equivalence("x == -5", "-5 == x")
        assert check_semantic_equivalence("x > -10", "-10 < x")
        
        # Zero comparisons
        assert check_semantic_equivalence("x == 0", "x <= 0 && x >= 0")
        assert not check_semantic_equivalence("x == 0", "x > 0")
    
    def test_arithmetic_with_constants(self):
        """Test arithmetic operations with constants."""
        # Addition with zero
        assert check_semantic_equivalence("x + 0 == y", "x == y")
        assert check_semantic_equivalence("0 + x == y", "x == y")
        
        # Multiplication with one
        assert check_semantic_equivalence("x * 1 == y", "x == y")
        assert check_semantic_equivalence("1 * x == y", "x == y")
        
        # Multiplication with zero
        assert check_semantic_equivalence("x * 0 == 0", "0 == 0")
    
    def test_parsing_failures(self):
        """Test that malformed predicates are handled gracefully."""
        # These should return False (not equivalent) rather than crash
        result = check_semantic_equivalence("invalid syntax here", "x > 0")
        assert result is False
        
        result = check_semantic_equivalence("x > 0", "also invalid")
        assert result is False
        
        # Empty strings should be handled
        result = check_semantic_equivalence("", "x > 0")
        assert result is False
    
    def test_nested_expressions(self):
        """Test predicates with nested parentheses and complex structure."""
        assert check_semantic_equivalence(
            "((x + y) * z) == 10",
            "(x + y) * z == 10"
        )
        
        # Using explicit comparisons for boolean contexts
        assert check_semantic_equivalence(
            "(a > 0 && b > 0) || (c > 0 && d > 0)",
            "(c > 0 && d > 0) || (a > 0 && b > 0)"
        )
        
        assert check_semantic_equivalence(
            "((x > 0) && (y > 0)) && (z > 0)",
            "x > 0 && y > 0 && z > 0"
        )
    
    def test_modulo_operator(self):
        """Test predicates involving modulo operator."""
        # Modulo is not commutative, so these should NOT be equivalent
        assert not check_semantic_equivalence("x % y == 0", "y % x == 0")
        
        # But identical modulo expressions are equivalent
        assert check_semantic_equivalence("x % 2 == 0", "x % 2 == 0")
        
        # Even/odd checks
        assert check_semantic_equivalence("x % 2 == 0", "(x % 2) == 0")


def test_original_problematic_case():
    """Test the specific case that revealed the equivalence issue."""
    # This is the case where LLM generated equivalent invariant
    pred1 = "q + a * b * p == (long long)x * y"
    pred2 = "a * b * p + q <= (long long)x * y && a * b * p + q >= (long long)x * y"
    
    assert check_semantic_equivalence(pred1, pred2), \
        "The two predicates should be logically equivalent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
