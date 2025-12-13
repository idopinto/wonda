import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.equivalence import check_syntactic_equivalence
print("pwd:", os.getcwd())
def test_check_equivalence():
    s1 = "x == SIZE - i"
    s2 = "x == (SIZE - i)"
    assert(check_syntactic_equivalence(s1, s2))
    print("Test 1 passed!")
    s1 = "x == SIZE - i"
    s2 = "x==(SIZE -i)"
    assert(check_syntactic_equivalence(s1, s2))
    print("Test 2 passed!")
    s1 = "x == SIZE - i"
    s2 = "x == (SIZE + i)"
    assert(not check_syntactic_equivalence(s1, s2))
    print("Test 3 passed!")
    s1 = "x == SIZE - i && a[i] == 1"
    s2 = "x == (SIZE - i) &&a[i]==1"
    assert(check_syntactic_equivalence(s1, s2))
    print("Test 4 passed!")

test_check_equivalence()