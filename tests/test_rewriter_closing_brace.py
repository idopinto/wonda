#!/usr/bin/env python3
"""Test Rewriter to verify closing braces are preserved after assertions."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.rewriter import Rewriter
from src.utils.program import Program

# Test files that had the issue
test_files = [
    "dataset/old/hard/c/interleave_bits_1.c",
    "dataset/evaluation/full/25/hard/c/dijkstra-u_valuebound2_4.c",
]

print("Testing Rewriter with closing brace preservation")
print("=" * 70)

for test_file_path in test_files:
    test_file = Path(test_file_path)
    if not test_file.exists():
        print(f"\n⚠️  File not found: {test_file_path}")
        continue
    
    print(f"\n📁 Testing: {test_file_path}")
    print("-" * 70)
    
    # Create rewriter
    r = Rewriter(test_file, rewrite=True)
    
    # Check if assertions are on single lines
    print("\nLines after Rewriter (last 5 lines):")
    for i, line in enumerate(r.lines_to_verify[-5:], len(r.lines_to_verify)-5):
        print(f"  {i}: {repr(line)}")
    
    # Check for assertions with closing braces on same line
    has_issue = False
    for i, line in enumerate(r.lines_to_verify):
        stripped = line.strip()
        if stripped.startswith("assert(") and ");" in stripped:
            # Check if there's content after the semicolon
            semicolon_idx = stripped.find(");")
            if semicolon_idx != -1:
                after_semicolon = stripped[semicolon_idx + 2:].strip()
                if after_semicolon and after_semicolon.startswith("}"):
                    print(f"\n  ⚠️  Found assertion with closing brace on same line at line {i}:")
                    print(f"      {repr(line)}")
                    has_issue = True
    
    if not has_issue:
        print("\n  ✓ No assertions with closing braces on same line found")
    
    # Test Program creation
    try:
        program = Program(r.lines_to_verify, r.replacement)
        
        # Check if assertions were extracted
        if program.assertions:
            print(f"\n  ✓ Program created successfully")
            print(f"  ✓ Extracted {len(program.assertions)} assertion(s)")
            for assertion in program.assertions:
                print(f"      - Line {assertion.line_number}: {assertion.content[:60]}...")
        else:
            print(f"\n  ⚠️  No assertions found")
        
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
            
            # Check if closing brace is present
            if result.strip().endswith("}"):
                print(f"  ✓ Generated code ends with closing brace")
            else:
                print(f"  ✗ Generated code is missing closing brace!")
                print(f"  Last 3 lines of generated code:")
                lines = result.split('\n')
                for line in lines[-3:]:
                    print(f"      {repr(line)}")
        
    except Exception as e:
        print(f"\n  ✗ Error creating Program: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("Test completed!")

