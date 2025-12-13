#!/usr/bin/env python3
"""Test script for multi-line assertion joining in Rewriter."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.rewriter import Rewriter

# Test with the dijkstra file
test_file = Path("dataset/evaluation/full/25/hard/c/dijkstra-u_valuebound2_4.c")

if not test_file.exists():
    print(f"Error: Test file not found: {test_file}")
    print(f"Current directory: {Path.cwd()}")
    sys.exit(1)

print(f"Testing multi-line assertion joining with: {test_file}")
print("=" * 70)

# Read original file to see the assertion
print("\nOriginal file (lines 42-46):")
with open(test_file) as f:
    lines = f.readlines()
    for i in range(41, min(47, len(lines))):
        print(f"{i+1:3d}: {lines[i].rstrip()}")

print("\n" + "=" * 70)

# Create rewriter
rewriter = Rewriter(test_file, rewrite=True)

# Show the lines after processing - look for assertion
print("\nLines after Rewriter processing (searching for assertion):")
print("=" * 70)
found_assertion = False
for i, line in enumerate(rewriter.lines_to_verify):
    if "assert" in line.lower():
        found_assertion = True
        print(f"\nLine {i}:")
        # Show first 150 chars
        if len(line) > 150:
            print(f"  {line[:150]}...")
            print(f"  ...{line[-50:]}")
        else:
            print(f"  {line}")
        print(f"  Length: {len(line)} chars")
        # Check if it contains the expected content
        if "h * h * n" in line and "q * r * r == 0" in line:
            print("  ✓ Contains expected assertion content")
        else:
            print("  ✗ Missing expected assertion content")

if not found_assertion:
    print("  ✗ No assertion found!")

print("\n" + "=" * 70)
print("Test completed!")
