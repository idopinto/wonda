import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.validate import syntactic_validation

print("pwd:", os.getcwd())

def test_valid_boolean_expressions():
    """Test that valid boolean expressions without assignments pass validation"""
    print("\n=== Testing Valid Boolean Expressions ===")
    
    # Simple comparisons
    assert syntactic_validation("x > 0"), "x > 0 should be valid"
    print("✓ Test 1 passed: x > 0")
    
    assert syntactic_validation("i < n"), "i < n should be valid"
    print("✓ Test 2 passed: i < n")
    
    assert syntactic_validation("k == 0"), "k == 0 should be valid"
    print("✓ Test 3 passed: k == 0")
    
    assert syntactic_validation("x != y"), "x != y should be valid"
    print("✓ Test 4 passed: x != y")
    
    # Complex boolean expressions
    assert syntactic_validation("x > 0 && y < 10"), "Compound condition should be valid"
    print("✓ Test 5 passed: x > 0 && y < 10")
    
    assert syntactic_validation("(x == 0) || (y > 5)"), "Parenthesized OR condition should be valid"
    print("✓ Test 6 passed: (x == 0) || (y > 5)")
    
    assert syntactic_validation("i >= 0 && i < SIZE"), "Complex condition should be valid"
    print("✓ Test 7 passed: i >= 0 && i < SIZE")
    
    # Arithmetic in comparisons
    assert syntactic_validation("x + y == 10"), "Arithmetic in comparison should be valid"
    print("✓ Test 8 passed: x + y == 10")
    
    assert syntactic_validation("a[i] == 0"), "Array access in condition should be valid"
    print("✓ Test 9 passed: a[i] == 0")
    
    assert syntactic_validation("arr[0] < arr[1]"), "Multiple array accesses should be valid"
    print("✓ Test 10 passed: arr[0] < arr[1]")


def test_invalid_assignments():
    """Test that expressions with assignments fail validation"""
    print("\n=== Testing Invalid Assignments ===")
    
    # Direct assignments
    assert not syntactic_validation("x = 5"), "x = 5 should be invalid (assignment)"
    print("✓ Test 1 passed: x = 5 rejected")
    
    assert not syntactic_validation("y = x + 1"), "y = x + 1 should be invalid"
    print("✓ Test 2 passed: y = x + 1 rejected")
    
    # Compound assignments
    assert not syntactic_validation("x += 1"), "x += 1 should be invalid"
    print("✓ Test 3 passed: x += 1 rejected")
    
    assert not syntactic_validation("i -= 5"), "i -= 5 should be invalid"
    print("✓ Test 4 passed: i -= 5 rejected")
    
    assert not syntactic_validation("k *= 2"), "k *= 2 should be invalid"
    print("✓ Test 5 passed: k *= 2 rejected")
    
    assert not syntactic_validation("n /= 3"), "n /= 3 should be invalid"
    print("✓ Test 6 passed: n /= 3 rejected")
    
    # Assignment in complex expression
    assert not syntactic_validation("x > 0 && (y = 5)"), "Assignment in compound should be invalid"
    print("✓ Test 7 passed: x > 0 && (y = 5) rejected")


def test_increment_decrement():
    """Test that increment/decrement operators fail validation"""
    print("\n=== Testing Increment/Decrement Operators ===")
    
    # Standalone increment/decrement
    assert not syntactic_validation("i++"), "i++ should be invalid"
    print("✓ Test 1 passed: i++ rejected")
    
    assert not syntactic_validation("j--"), "j-- should be invalid"
    print("✓ Test 2 passed: j-- rejected")
    
    assert not syntactic_validation("++k"), "++k should be invalid"
    print("✓ Test 3 passed: ++k rejected")
    
    assert not syntactic_validation("--n"), "--n should be invalid"
    print("✓ Test 4 passed: --n rejected")
    
    # Note: The current implementation has a heuristic that allows ++/-- if part of comparison
    # This is a known limitation - the AST checker should catch these, but this tests current behavior
    # If i++ is part of a comparison, it might still be caught by AST, but standalone should fail


def test_malformed_expressions():
    """Test that malformed or unparseable expressions fail validation"""
    print("\n=== Testing Malformed Expressions ===")
    
    # Unclosed parentheses
    assert not syntactic_validation("x > 0 && (y < 10"), "Unclosed paren should be invalid"
    print("✓ Test 1 passed: unclosed paren rejected")
    
    # Invalid syntax
    assert not syntactic_validation("x > > y"), "Invalid syntax should fail"
    print("✓ Test 2 passed: invalid syntax rejected")
    
    # Empty expression
    assert not syntactic_validation(""), "Empty expression should be invalid"
    print("✓ Test 3 passed: empty expression rejected")
    
    # Just operators
    assert not syntactic_validation("&& ||"), "Just operators should be invalid"
    print("✓ Test 4 passed: just operators rejected")


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n=== Testing Edge Cases ===")
    
    # Single variable (some systems might accept this as "true")
    assert syntactic_validation("x"), "Single variable should be valid (implicit != 0)"
    print("✓ Test 1 passed: single variable accepted")
    
    # Negation
    assert syntactic_validation("!flag"), "Negation should be valid"
    print("✓ Test 2 passed: !flag accepted")
    
    assert syntactic_validation("!(x > 0)"), "Negated condition should be valid"
    print("✓ Test 3 passed: !(x > 0) accepted")
    
    # Function calls (should be valid if they don't have side effects)
    # Note: This depends on how pycparser handles function calls
    # Most function calls in conditions are treated as expressions
    try:
        result = syntactic_validation("func() == 0")
        # Function calls might be syntactically valid but we can't know side effects
        print(f"✓ Test 4: func() == 0 returned {result} (depends on implementation)")
    except:
        print("⚠ Test 4: func() == 0 failed (might be expected)")


def run_all_tests():
    """Run all test suites"""
    print("=" * 60)
    print("Running Unit Tests for syntactic_validation()")
    print("=" * 60)
    
    try:
        test_valid_boolean_expressions()
        test_invalid_assignments()
        test_increment_decrement()
        test_malformed_expressions()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR RUNNING TESTS: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()