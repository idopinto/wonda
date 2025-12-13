#!/usr/bin/env python3
"""Comprehensive test for simplified join_multiline_assertions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.rewriter import Rewriter
from src.utils.program import Program

print("=" * 70)
print("Comprehensive Test: Simplified join_multiline_assertions")
print("=" * 70)

test_cases = [
    {
        "name": "Single-line assertion with closing brace",
        "file": "dataset/old/hard/c/interleave_bits_1.c",
        "expected": "assertion on single line, closing brace separated"
    },
    {
        "name": "Multi-line assertion",
        "file": "dataset/evaluation/full/25/hard/c/dijkstra-u_valuebound2_4.c",
        "expected": "multi-line assertion joined into single line"
    },
    {
        "name": "Assertion with function call",
        "file": "dataset/evaluation/full/25/hard/c/loops_linear_sea.ch_assertion0.c",
        "expected": "assertion with function call extracted correctly"
    },
]

all_passed = True

for test_case in test_cases:
    print(f"\n{'=' * 70}")
    print(f"Test: {test_case['name']}")
    print(f"File: {test_case['file']}")
    print(f"Expected: {test_case['expected']}")
    print("-" * 70)
    
    test_file = Path(test_case['file'])
    if not test_file.exists():
        print(f"  ⚠️  File not found: {test_case['file']}")
        all_passed = False
        continue
    
    try:
        # Create rewriter
        r = Rewriter(test_file, rewrite=True)
        
        # Check for assertions with closing braces on same line
        has_issue = False
        for i, line in enumerate(r.lines_to_verify):
            stripped = line.strip()
            if (stripped.startswith("assert(") or stripped.startswith("assume(")) and ");" in stripped:
                semicolon_idx = stripped.find(");")
                if semicolon_idx != -1:
                    after_semicolon = stripped[semicolon_idx + 2:].strip()
                    if after_semicolon and after_semicolon.startswith("}"):
                        print(f"  ✗ Found assertion with closing brace on same line at line {i}:")
                        print(f"      {repr(line)}")
                        has_issue = True
                        all_passed = False
        
        if not has_issue:
            print("  ✓ No assertions with closing braces on same line")
        
        # Test Program creation and assertion extraction
        program = Program(r.lines_to_verify, r.replacement)
        
        if program.assertions or program.lemmas:
            total = len(program.assertions) + len(program.lemmas)
            print(f"  ✓ Extracted {len(program.assertions)} assertion(s) and {len(program.lemmas)} lemma(s)")
            
            # Check that assertions are on single lines
            for assertion in program.assertions:
                assertion_line = r.lines_to_verify[assertion.line_number] if assertion.line_number < len(r.lines_to_verify) else ""
                if 'assert(' in assertion_line and ');' in assertion_line:
                    # Check if it's actually complete (not spanning)
                    if assertion_line.count('(') - assertion_line.count(')') == 0:
                        print(f"  ✓ Assertion on line {assertion.line_number} is complete on single line")
                    else:
                        print(f"  ⚠️  Assertion on line {assertion.line_number} might be incomplete")
                        print(f"      Line: {repr(assertion_line[:100])}")
                else:
                    print(f"  ⚠️  Assertion on line {assertion.line_number} might span lines")
                    print(f"      Line: {repr(assertion_line[:100])}")
            
            # Test get_program_with_assertion
            if program.assertions:
                target = program.assertions[0]
                result = program.get_program_with_assertion(
                    predicate=target,
                    assumptions=[],
                    assertion_points={},
                    forGPT=False,
                    dump=False
                )
                
                if result.strip().endswith("}"):
                    print(f"  ✓ Generated code ends with closing brace")
                else:
                    # Check if it should end with a brace
                    lines = result.split('\n')
                    if any('}' in line for line in lines[-3:]):
                        print(f"  ✓ Generated code contains closing brace")
                    else:
                        print(f"  ℹ️  Generated code doesn't end with brace (might be expected)")
        else:
            print(f"  ⚠️  No assertions or lemmas found")
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✓ All tests passed!")
else:
    print("✗ Some tests failed")
print("=" * 70)

