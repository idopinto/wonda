#!/usr/bin/env python3
"""Test Rewriter with join_multiline_assertions before and after clang_format."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.rewriter import Rewriter
from src.utils.program import Program

# Test file that had the issue
test_file_path = "dataset/old/hard/c/interleave_bits_1.c"
test_file = Path(test_file_path)

if not test_file.exists():
    print(f"⚠️  File not found: {test_file_path}")
    sys.exit(1)

print("Testing Rewriter: join_multiline_assertions BEFORE vs AFTER clang_format")
print("=" * 70)

# Read the original file
original_content = test_file.read_text()
print(f"\n📁 Testing: {test_file_path}")
print("-" * 70)

# Test 1: Current implementation (AFTER clang_format)
print("\n1️⃣  CURRENT: join_multiline_assertions AFTER clang_format")
print("-" * 70)
r1 = Rewriter(test_file, rewrite=True)
print("\nLast 5 lines after Rewriter:")
for i, line in enumerate(r1.lines_to_verify[-5:], len(r1.lines_to_verify)-5):
    print(f"  {i}: {repr(line)}")

# Check for assertions with closing braces on same line
has_issue_1 = False
for i, line in enumerate(r1.lines_to_verify):
    stripped = line.strip()
    if stripped.startswith("assert(") and ");" in stripped:
        semicolon_idx = stripped.find(");")
        if semicolon_idx != -1:
            after_semicolon = stripped[semicolon_idx + 2:].strip()
            if after_semicolon and after_semicolon.startswith("}"):
                print(f"\n  ⚠️  Found assertion with closing brace on same line at line {i}:")
                print(f"      {repr(line)}")
                has_issue_1 = True

if not has_issue_1:
    print("\n  ✓ No assertions with closing braces on same line")

# Test Program creation
try:
    program1 = Program(r1.lines_to_verify, r1.replacement)
    if program1.assertions:
        print(f"  ✓ Program created, extracted {len(program1.assertions)} assertion(s)")
        
        # Test get_program_with_assertion
        target = program1.assertions[0]
        result1 = program1.get_program_with_assertion(
            predicate=target,
            assumptions=[],
            assertion_points={},
            forGPT=False,
            dump=False
        )
        
        if result1.strip().endswith("}"):
            print(f"  ✓ Generated code ends with closing brace")
        else:
            print(f"  ✗ Generated code is missing closing brace!")
            print(f"  Last 3 lines:")
            lines = result1.split('\n')
            for line in lines[-3:]:
                print(f"      {repr(line)}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: Modified implementation (BEFORE clang_format)
print("\n\n2️⃣  ALTERNATIVE: join_multiline_assertions BEFORE clang_format")
print("-" * 70)

# Temporarily modify the Rewriter to test before clang_format
# We'll need to manually test this by reading the file and applying the logic
from src.utils.rewriter import Rewriter

# Create a test version that does join_multiline_assertions before clang_format
class TestRewriter(Rewriter):
    def __init__(self, base_file: Path, rewrite: bool = False, handle_reach_error: bool = True):
        self.base_file = base_file
        self.rewrite = rewrite
        self.handle_reach_error = handle_reach_error
        
        if rewrite:
            self.new_code = base_file.read_text()
            self.remove_comments()
            self.new_code = self.new_code.replace("__VERIFIER_assert", "assert")
            self.new_code = self.new_code.replace("assume_abort_if_not", "assume")
            
            # Join multi-line assertions BEFORE clang_format
            self.join_multiline_assertions()
            
            self.clang_format()
            self.remove_empty_lines()
            self.has_reach_error = False
            if handle_reach_error:
                try:
                    self.replace_reach_error_with_assertion()
                except:
                    # If replace_reach_error fails, continue anyway
                    pass

        self.lines_to_verify = self.new_code.split("\n")
        # Replace nondeterministic functions with random values
        self.remove_verifier_nondet()
        self.lines_for_gpt = self.new_code.split("\n") 
        self.replacement: dict = {}
        assert(len(self.lines_for_gpt) == len(self.lines_to_verify))

r2 = TestRewriter(test_file, rewrite=True)
print("\nLast 5 lines after Rewriter (join BEFORE clang_format):")
for i, line in enumerate(r2.lines_to_verify[-5:], len(r2.lines_to_verify)-5):
    print(f"  {i}: {repr(line)}")

# Check for assertions with closing braces on same line
has_issue_2 = False
for i, line in enumerate(r2.lines_to_verify):
    stripped = line.strip()
    if stripped.startswith("assert(") and ");" in stripped:
        semicolon_idx = stripped.find(");")
        if semicolon_idx != -1:
            after_semicolon = stripped[semicolon_idx + 2:].strip()
            if after_semicolon and after_semicolon.startswith("}"):
                print(f"\n  ⚠️  Found assertion with closing brace on same line at line {i}:")
                print(f"      {repr(line)}")
                has_issue_2 = True

if not has_issue_2:
    print("\n  ✓ No assertions with closing braces on same line")

# Test Program creation
try:
    program2 = Program(r2.lines_to_verify, r2.replacement)
    if program2.assertions:
        print(f"  ✓ Program created, extracted {len(program2.assertions)} assertion(s)")
        
        # Test get_program_with_assertion
        target = program2.assertions[0]
        result2 = program2.get_program_with_assertion(
            predicate=target,
            assumptions=[],
            assertion_points={},
            forGPT=False,
            dump=False
        )
        
        if result2.strip().endswith("}"):
            print(f"  ✓ Generated code ends with closing brace")
        else:
            print(f"  ✗ Generated code is missing closing brace!")
            print(f"  Last 3 lines:")
            lines = result2.split('\n')
            for line in lines[-3:]:
                print(f"      {repr(line)}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 70)
print("Comparison completed!")

