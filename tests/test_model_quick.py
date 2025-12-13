#!/usr/bin/env python3
"""
Quick test script for generate_candidate_invariant using problem.c
"""
import sys
from pathlib import Path

# Add project root to path (go up from tests/ directory)
project_root = Path(__file__).parent.parent  # Go from tests/ to RLInv/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.rewriter import Rewriter
from src.utils.program import Program
from src.utils.predicate import Predicate
from src.eval.model import Model
import os

def main():
    # Use the problem.c file
    problem_file = project_root / "dataset" / "evaluation" / "problem.c"
    property_file = project_root / "dataset" / "properties" / "unreach-call.prp"
    
    if not problem_file.exists():
        print(f"Error: {problem_file} not found")
        return 1
    
    print("="*80)
    print("Testing generate_candidate_invariant")
    print("="*80)
    print(f"Program file: {problem_file}")
    print()
    
    try:
        # Create program from the C file
        print("Step 1: Creating Program from C file...")
        rewriter = Rewriter(problem_file)
        program = Program(rewriter.lines_to_verify, rewriter.replacement)
        print(f"✓ Program created with {len(program.assertion_points)} assertion points")
        print(f"  Assertion points: {sorted(program.assertion_points.keys())}")
        
        # Show some program info
        if program.assertions:
            print(f"  Target assertions: {len(program.assertions)}")
            for i, assertion in enumerate(program.assertions):
                print(f"    Assertion {i+1}: assert({assertion.content}) at line {assertion.line_number}")
        
        print()
        
        # Create model
        print("Step 2: Initializing Model...")
        model_name = os.getenv("TOGETHER_MODEL", "openai/gpt-oss-20b")
        model = Model(model_path_or_name=model_name)
        print(f"✓ Model initialized: {model.model_path_or_name}")
        print()
        
        # Create a dummy task (minimal - just for API compatibility)
        class DummyTask:
            def __init__(self):
                self.source_code_path = problem_file
                self.property_path = property_file
        
        task = DummyTask()
        
        # Generate invariant
        print("Step 3: Generating candidate invariant...")
        print("-" * 80)
        
        result = model.generate_candidate_invariant(task, program)
        
        print("-" * 80)
        print()
        print("="*80)
        print("RESULT:")
        print("="*80)
        print(f"✓ Generated Predicate:")
        print(f"  Content: {result.content}")
        print(f"  Line number: {result.line_number}")
        
        # Check if line number is a valid assertion point
        if result.line_number in program.assertion_points:
            attrs = program.assertion_points[result.line_number]
            attr_names = [attr.name for attr in attrs]
            print(f"  ✓ Valid assertion point with attributes: {', '.join(attr_names)}")
        else:
            print(f"  ⚠ Warning: Line {result.line_number} is not a listed assertion point")
        
        print("="*80)
        return 0
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

