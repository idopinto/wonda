"""Tests for parse_model_answer and validate_model_answer (wonda.eval.validate)."""
import pytest

from wonda.core.property import Property
from wonda.eval.validate import parse_model_answer, validate_model_answer


class TestParseModelAnswer:
    def test_valid_bare_json(self):
        raw = '{"marker":"INVARIANT_MARKER_1","content":"x >= 0"}'
        result = parse_model_answer(raw)
        assert result == Property(content="x >= 0", marker_name="INVARIANT_MARKER_1")

    def test_valid_json_code_block(self):
        raw = '```json\n{"marker":"INVARIANT_MARKER_2","content":"i <= n"}\n```'
        result = parse_model_answer(raw)
        assert result == Property(content="i <= n", marker_name="INVARIANT_MARKER_2")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse model answer"):
            parse_model_answer("not json at all")

    def test_missing_marker_raises(self):
        with pytest.raises(ValueError, match="must contain 'marker' and 'content'"):
            parse_model_answer('{"content":"x >= 0"}')

    def test_missing_content_raises(self):
        with pytest.raises(ValueError, match="must contain 'marker' and 'content'"):
            parse_model_answer('{"marker":"INVARIANT_MARKER_1"}')


class TestValidateModelAnswer:
    def test_valid_answer_matching_marker(self):
        raw = '{"marker":"INVARIANT_MARKER_1","content":"x >= 0"}'
        prop, d = validate_model_answer(raw, target_marker="INVARIANT_MARKER_1")
        assert prop is not None
        assert prop.content == "x >= 0"
        assert prop.marker_name == "INVARIANT_MARKER_1"
        assert d["valid_json_format"] is True
        assert d["valid_content"] is True
        assert d["valid_marker"] is True

    def test_valid_answer_target_marker_none(self):
        raw = '{"marker":"INVARIANT_MARKER_1","content":"x >= 0"}'
        prop, d = validate_model_answer(raw, target_marker=None)
        assert d["valid_marker"] is True

    def test_invalid_json(self):
        prop, d = validate_model_answer("not json", target_marker="INVARIANT_MARKER_1")
        assert prop is None
        assert d["valid_json_format"] is False
        assert d["valid_content"] is False
        assert d["valid_marker"] is False

    def test_valid_json_invalid_content(self):
        # content that fails syntactic_validation (e.g. assignment)
        raw = '{"marker":"INVARIANT_MARKER_1","content":"x = 5"}'
        prop, d = validate_model_answer(raw, target_marker="INVARIANT_MARKER_1")
        assert prop is not None
        assert d["valid_json_format"] is True
        assert d["valid_content"] is False
        assert d["valid_marker"] is True

    def test_wrong_marker(self):
        raw = '{"marker":"INVARIANT_MARKER_2","content":"x >= 0"}'
        prop, d = validate_model_answer(raw, target_marker="INVARIANT_MARKER_1")
        assert prop is not None
        assert d["valid_json_format"] is True
        assert d["valid_content"] is True
        assert d["valid_marker"] is False
