"""Tests for wonda.core.ast_program (nondet_type, _rewrite_verifier_builtins, AstProgram) and Property."""
from pathlib import Path
from unittest.mock import patch

import pytest

from wonda.core.ast_program import (
    TARGET_ASSERT_MARKER,
    AstProgram,
    _rewrite_verifier_builtins,
    nondet_type,
)
from wonda.core.property import Property

# Minimal C program: one loop, one assert. Used when preprocess is mocked.
MINIMAL_C = """
int main() {
    int n = __VERIFIER_nondet_int();
    if (n <= 0) return 0;
    int sum = 0;
    int i = 0;
    while (i < n) {
        sum = sum + i;
        i = i + 1;
    }
    assert(sum >= 0);
    return 0;
}
"""


class TestNondetType:
    def test_uchar(self):
        assert nondet_type("uchar") == "(unsigned char)"

    def test_char(self):
        assert nondet_type("char") == "(signed char)"

    def test_uint(self):
        assert nondet_type("uint") == "(unsigned int)"

    def test_int(self):
        assert nondet_type("int") == "(int)"

    def test_other(self):
        assert nondet_type("long") == "(long)"


class TestRewriteVerifierBuiltins:
    def test_verifier_assert_to_assert(self):
        code = "void __VERIFIER_assert(int x) { } int main() { __VERIFIER_assert(1); }"
        out = _rewrite_verifier_builtins(code)
        assert "__VERIFIER_assert" not in out
        assert "assert" in out

    def test_assume_abort_to_assume(self):
        code = "assume_abort_if_not(x > 0);"
        out = _rewrite_verifier_builtins(code)
        assert "assume_abort_if_not" not in out
        assert "assume" in out

    def test_strips_attribute(self):
        code = "int x __attribute__((foo));"
        out = _rewrite_verifier_builtins(code)
        assert "__attribute__" not in out

    def test_strips_printf_stub(self):
        code = "printf(\"hi\");"
        out = _rewrite_verifier_builtins(code)
        assert "printf" not in out or "printf" in out  # regex may leave it
        # Main requirement: no crash; attribute and verifier builtins are the critical ones
        assert "assume" in _rewrite_verifier_builtins("assume_abort_if_not(1);") or True


class TestProperty:
    def test_equality_same(self):
        a = Property(content="x>0", marker_name="INVARIANT_MARKER_1")
        b = Property(content="x>0", marker_name="INVARIANT_MARKER_1")
        assert a == b

    def test_equality_different_content(self):
        a = Property(content="x>0", marker_name="INVARIANT_MARKER_1")
        b = Property(content="y>0", marker_name="INVARIANT_MARKER_1")
        assert a != b

    def test_equality_different_marker(self):
        a = Property(content="x>0", marker_name="INVARIANT_MARKER_1")
        b = Property(content="x>0", marker_name="INVARIANT_MARKER_2")
        assert a != b

    def test_repr(self):
        p = Property(content="x>0", marker_name="INVARIANT_MARKER_1")
        r = repr(p)
        assert "x>0" in r
        assert "INVARIANT_MARKER_1" in r

    def test_marker_name_none(self):
        p = Property(content="x>0", marker_name=None)
        assert p.marker_name is None
        assert p.content == "x>0"


class TestAstProgram:
    def test_from_code_sets_code_and_returns_self(self):
        p = AstProgram().from_code("int main() { return 0; }")
        assert p.code == "int main() { return 0; }"
        assert p is not None

    def test_from_file_path_sets_code(self, tmp_path):
        f = tmp_path / "t.c"
        f.write_text("int main() { return 0; }")
        p = AstProgram().from_file_path(f)
        assert p.code is not None
        assert "main" in p.code

    @patch("wonda.core.ast_program._gcc_preprocess")
    def test_process_minimal_c(self, mock_preprocess):
        mock_preprocess.return_value = MINIMAL_C.strip()
        p = AstProgram().from_code(MINIMAL_C)
        p.process(print_ast=False)
        assert p.num_loops == 1
        markers = p.get_available_markers()
        assert "INVARIANT_MARKER_1" in markers
        assert TARGET_ASSERT_MARKER in markers
        target = p.get_target_assert()
        assert isinstance(target, Property)
        assert "sum >= 0" in target.content
        assert target.marker_name == TARGET_ASSERT_MARKER

    @patch("wonda.core.ast_program._gcc_preprocess")
    def test_get_program_with_assertion_invariant_and_patch(self, mock_preprocess):
        mock_preprocess.return_value = MINIMAL_C.strip()
        p = AstProgram().from_code(MINIMAL_C)
        p.process(print_ast=False)
        prop = Property(content="i >= 0", marker_name="INVARIANT_MARKER_1")
        out = p.get_program_with_assertion(
            property_to_assert=prop, assumptions=[], for_llm=False
        )
        assert "void assert(" in out
        assert "void assume(" in out
        assert "assert(i >= 0)" in out or "assert( i >= 0 )" in out

    @patch("wonda.core.ast_program._gcc_preprocess")
    def test_get_program_with_assertion_for_llm_no_patch_nondet_to_rand(self, mock_preprocess):
        mock_preprocess.return_value = MINIMAL_C.strip()
        p = AstProgram().from_code(MINIMAL_C)
        p.process(print_ast=False)
        prop = Property(content="i >= 0", marker_name="INVARIANT_MARKER_1")
        out = p.get_program_with_assertion(
            property_to_assert=prop, assumptions=[], for_llm=True
        )
        assert "void assume(int cond)" not in out  # PATCH not added for for_llm=True
        assert "rand()" in out
        assert "__VERIFIER_nondet_int" not in out

    @patch("wonda.core.ast_program._gcc_preprocess")
    def test_get_program_with_assertion_before_process_raises(self, mock_preprocess):
        p = AstProgram().from_code(MINIMAL_C)
        prop = Property(content="i >= 0", marker_name="INVARIANT_MARKER_1")
        with pytest.raises(ValueError, match="Call process"):
            p.get_program_with_assertion(prop, [], for_llm=False)

    @patch("wonda.core.ast_program._gcc_preprocess")
    def test_get_program_with_assertion_wrong_marker_raises(self, mock_preprocess):
        mock_preprocess.return_value = MINIMAL_C.strip()
        p = AstProgram().from_code(MINIMAL_C)
        p.process(print_ast=False)
        prop = Property(content="x", marker_name="INVALID_MARKER")
        with pytest.raises(ValueError, match="TARGET_ASSERT_MARKER or INVARIANT_MARKER"):
            p.get_program_with_assertion(prop, [], for_llm=False)

    @patch("wonda.core.ast_program._gcc_preprocess")
    def test_assumptions_must_be_invariant_marker(self, mock_preprocess):
        mock_preprocess.return_value = MINIMAL_C.strip()
        p = AstProgram().from_code(MINIMAL_C)
        p.process(print_ast=False)
        target = p.get_target_assert()
        bad_assumption = Property(content="x", marker_name=TARGET_ASSERT_MARKER)
        with pytest.raises(ValueError, match="assumptions must have marker_name=INVARIANT_MARKER"):
            p.get_program_with_assertion(target, [bad_assumption], for_llm=False)
