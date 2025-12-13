import sys
from pathlib import Path
import pytest


from src.utils.program import Program, AssertionPointAttributes
from src.utils.rewriter import Rewriter


@pytest.mark.parametrize(
    "c_filename,expected_points",
    [
        ("benchmark24_conjunctive_1.c", {7, 10}),
    ],
)
def test_program_structure_basic(c_filename, expected_points):
    c_program = Path("dataset") / "evaluation" / "easy" / "c" / c_filename
    if not c_program.exists():
        pytest.skip(f"Missing dataset file: {c_program}")
    r = Rewriter(c_program)
    program = Program(r.lines_to_verify, r.replacement)
    assert program.assertion_points.keys() == expected_points
    assert AssertionPointAttributes.BeginningOfLoop in program.assertion_points[min(expected_points)]
    assert AssertionPointAttributes.BeforeAssertion in program.assertion_points[max(expected_points)]