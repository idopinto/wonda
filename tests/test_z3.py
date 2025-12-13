#!/usr/bin/env python3
import os
import subprocess

# Make z3 executable
z3_path = "/cs/labs/guykatz/idopinto12/projects/loop_invariant_generation/RLInv/tools/uautomaizer/z3"
os.chmod(z3_path, 0o755)

# Test if z3 works
try:
    result = subprocess.run([z3_path, "--version"], capture_output=True, text=True, timeout=10)
    print(f"Z3 version: {result.stdout}")
    print(f"Exit code: {result.returncode}")
except Exception as e:
    print(f"Error running z3: {e}")

# Test Ultimate with the updated config
try:
    cmd = [
        "python3", 
        "/cs/labs/guykatz/idopinto12/projects/loop_invariant_generation/RLInv/tools/uautomaizer/Ultimate.py",
        "--spec", "/cs/labs/guykatz/idopinto12/projects/loop_invariant_generation/RLInv/data/properties/unreach-call.prp",
        "--architecture", "64bit",
        "--file", "/cs/labs/guykatz/idopinto12/projects/loop_invariant_generation/RLInv/data/evaluation/problem.c"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    print(f"Ultimate stdout: {result.stdout}")
    print(f"Ultimate stderr: {result.stderr}")
    print(f"Ultimate exit code: {result.returncode}")
except Exception as e:
    print(f"Error running Ultimate: {e}")
