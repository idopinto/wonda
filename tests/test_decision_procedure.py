"""Tests for decision procedure calculus (compute_decision)."""
import sys
from unittest.mock import MagicMock

import pytest

# weave is optional; avoid import error when testing compute_decision only
sys.modules["weave"] = MagicMock()
from wonda.eval.decision_procedure import compute_decision


class TestComputeDecision:
    def test_dec_false_db_false(self):
        final, rule = compute_decision("TRUE", "FALSE")
        assert final == "FALSE"
        assert rule == "DEC-FALSE"

    def test_dec_false_db_false_da_unknown(self):
        final, rule = compute_decision("UNKNOWN", "FALSE")
        assert final == "FALSE"
        assert rule == "DEC-FALSE"

    def test_dec_false_db_false_da_false(self):
        final, rule = compute_decision("FALSE", "FALSE")
        assert final == "FALSE"
        assert rule == "DEC-FALSE"

    def test_dec_false_db_false_da_timeout(self):
        final, rule = compute_decision("TIMEOUT", "FALSE")
        assert final == "FALSE"
        assert rule == "DEC-FALSE"

    def test_dec_prop_da_true_db_true(self):
        final, rule = compute_decision("TRUE", "TRUE")
        assert final == "TRUE"
        assert rule == "DEC-PROP"

    def test_dec_prop_da_true_db_unknown(self):
        final, rule = compute_decision("TRUE", "UNKNOWN")
        assert final == "UNKNOWN"
        assert rule == "DEC-PROP"

    def test_dec_prop_da_true_db_timeout(self):
        final, rule = compute_decision("TRUE", "TIMEOUT")
        assert final == "UNKNOWN"
        assert rule == "DEC-PROP"

    def test_dec_prop_da_true_db_error(self):
        final, rule = compute_decision("TRUE", "ERROR")
        assert final == "UNKNOWN"
        assert rule == "DEC-PROP"

    def test_dec_u_da_unknown_db_unknown(self):
        final, rule = compute_decision("UNKNOWN", "UNKNOWN")
        assert final == "UNKNOWN"
        assert rule == "DEC-U"

    def test_dec_u_da_false_db_true(self):
        final, rule = compute_decision("FALSE", "TRUE")
        assert final == "UNKNOWN"
        assert rule == "DEC-U"

    def test_dec_u_da_false_db_unknown(self):
        final, rule = compute_decision("FALSE", "UNKNOWN")
        assert final == "UNKNOWN"
        assert rule == "DEC-U"

    def test_dec_u_da_timeout_db_true(self):
        final, rule = compute_decision("TIMEOUT", "TRUE")
        assert final == "UNKNOWN"
        assert rule == "DEC-U"

    def test_dec_u_da_error_db_unknown(self):
        final, rule = compute_decision("ERROR", "UNKNOWN")
        assert final == "UNKNOWN"
        assert rule == "DEC-U"

    def test_fallback_no_rule(self):
        # da TRUE, db something else not in {TRUE,UNKNOWN,TIMEOUT,ERROR} -> no DEC-U (db != FALSE), no DEC-PROP
        # Actually db "FALSE" is handled first. So da TRUE, db "OTHER" -> DEC-PROP? No, db not in set. So fallback UNKNOWN.
        final, rule = compute_decision("TRUE", "OTHER")
        assert final == "UNKNOWN"
        assert rule == ""
