"""Tests for UAutomizer output parsing (_parse_runlim_output, _parse_uautomizer_output)."""
from pathlib import Path

import pytest

from wonda.verifiers.uautomizer import UAutomizerVerifier

# Verifier instance only used to call parsing methods; paths are not used.
def _verifier():
    return UAutomizerVerifier(
        uautomizer_path=Path("/tmp/Ultimate.py"),
        property_file_path=Path("/tmp/unreach-call.prp"),
        arch="32bit",
        timeout_seconds=60.0,
        version="25",
    )


RUNLIM_SAMPLE = """
[runlim] host: somehost
[runlim] time limit: 311040000 seconds
[runlim] real time limit: 60 seconds
[runlim] space limit: 16384 MB
[runlim] status: ok
[runlim] real: 12.34 seconds
[runlim] time: 10.50 seconds
[runlim] space: 400 MB
"""


class TestParseRunlimOutput:
    def test_parses_ok_status_and_times(self):
        v = _verifier()
        out = v._parse_runlim_output(RUNLIM_SAMPLE)
        assert out["status"] == "ok"
        assert out["real_time"] == 12.34
        assert out["cpu_time"] == 10.50
        assert out["real_time_limit"] == 60
        assert out["space_limit"] == 16384
        assert out["space"] == 400

    def test_empty_string_safe(self):
        v = _verifier()
        out = v._parse_runlim_output("")
        assert "status" in out
        assert out.get("real_time") is None or out.get("real_time") == 0
        assert out.get("real_time_limit") is None or isinstance(out.get("real_time_limit"), (int, type(None)))

    def test_malformed_no_crash(self):
        v = _verifier()
        out = v._parse_runlim_output("garbage\n[runlim] status: out of time\n")
        assert out["status"] == "out of time"


class TestParseUautomizerOutput:
    def test_result_true(self):
        v = _verifier()
        decision, reason = v._parse_uautomizer_output("Result:\nTRUE")
        assert decision == "TRUE"
        assert reason == ""

    def test_result_false_with_reason(self):
        v = _verifier()
        decision, reason = v._parse_uautomizer_output("Result:\nFALSE: some reason")
        assert decision == "FALSE"
        assert "some reason" in reason

    def test_result_unknown(self):
        v = _verifier()
        decision, reason = v._parse_uautomizer_output("Result:\nUNKNOWN")
        assert decision == "UNKNOWN"

    def test_result_error(self):
        v = _verifier()
        decision, reason = v._parse_uautomizer_output("Result:\nERROR: parse error")
        assert decision == "ERROR"

    def test_no_result_returns_error(self):
        v = _verifier()
        decision, reason = v._parse_uautomizer_output("no Result line here")
        assert decision == "ERROR"
        assert "Unable to parse" in reason
