#!/usr/bin/env python3
"""Test Program.py with reverted regex pattern on multi-line assertions."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.rewriter import Rewriter
from src.utils.program import Program

# Test with the dijkstra file
test_file = Path("dataset/evaluation/full/25/hard/c/dijkstra-u_valuebound2_4.c")

if not test_file.exists():
    print(f"Error: Test file not found: {test_file}")
    print(f"Current directory: {Path.cwd()}")
    sys.exit(1)

print(f"Testing Program.py with reverted regex pattern")
print(f"File: {test_file}")
print("=" * 70)

# Create rewriter (this will join multi-line assertions)
rewriter = Rewriter(test_file, rewrite=True)

# Create Program instance
try:
    program = Program(rewriter.lines_to_verify, rewriter.replacement)
    
    print("\n✓ Program created successfully!")
    print("\nExtracted Assertions:")
    print("=" * 70)
    
    if program.assertions:
        for i, assertion in enumerate(program.assertions, 1):
            print(f"\nAssertion {i}:")
            print(f"  Line: {assertion.line_number}")
            print(f"  Content (first 100 chars): {assertion.content[:100]}...")
            if len(assertion.content) > 100:
                print(f"  Content (last 50 chars): ...{assertion.content[-50:]}")
            else:
                print(f"  Content (full): {assertion.content}")
            print(f"  Length: {len(assertion.content)} chars")
            
            # Check if it contains expected content
            if "h * h * n" in assertion.content and "q * r * r == 0" in assertion.content:
                print("  ✓ Contains expected assertion content")
            else:
                print("  ✗ Missing expected assertion content")
    else:
        print("  ✗ No assertions found!")
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

