#!/usr/bin/env python3
"""End-to-end demo: load an evaluation program, process it, and run verification.

Demonstrates the full WONDA verification pipeline on a real evaluation program:
  1. Load a program from the HuggingFace evaluation dataset
  2. Process it through AstProgram (preprocessing, AST instrumentation)
  3. Run **baseline** verification (target assertion only, no invariant)
  4. Run **invariant-assisted** verification (correctness + usefulness queries)
  5. Apply the decision procedure to combine results

Prerequisites:
  - UAutomizer installed at tools/UAutomizer25/ (see README)
  - runlim installed at tools/runlim/runlim (see README)

Run with:
    uv run python -m wonda.demos.direct_verification_demo

Options:
    --timeout   Verifier timeout in seconds (default: 120)
    --limit     How many programs to load from dataset (default: 5)
    --index     Which program index to use from loaded dataset (default: 0)
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from configs import global_config as GC
from wonda.core.ast_program import AstProgram
from wonda.core.property import Property
from wonda.eval.eval_data import get_evaluation_dataset, preprocess_example
from wonda.verifiers.uautomizer import UAutomizerVerifier, VerifierCallReport, write_file

SEPARATOR = "=" * 70
THIN_SEP = "-" * 70


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def run_single_verification(
    verifier: UAutomizerVerifier,
    program_code: str,
    reports_dir: Path,
    label: str,
) -> VerifierCallReport:
    """Write a C program to disk and run UAutomizer on it."""
    program_path = reports_dir / f"{label}.c"
    write_file(program_path, program_code)
    report = verifier.verify(
        program_path=program_path,
        reports_dir=reports_dir,
        label=label,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120.0, help="Verifier timeout (seconds)")
    parser.add_argument("--limit", type=int, default=5, help="Programs to load from dataset")
    parser.add_argument("--index", type=int, default=0, help="Index of program to verify")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load evaluation program from HuggingFace
    # ------------------------------------------------------------------
    section("1. LOADING EVALUATION PROGRAM")
    dataset = get_evaluation_dataset(
        dataset_name="idopinto/wonda-eval-benchmark-full",
        limit=args.limit,
        split="easy",
        re_split=True,
        eval_per_marker=True,
    )
    if not dataset:
        print("ERROR: No programs loaded from dataset.")
        return

    idx = min(args.index, len(dataset) - 1)
    example = dataset[idx]
    file_name = example.get("file", "unknown")
    baseline_timing = example.get("median_timing", 0)
    target_marker = example.get("target_marker", None)

    print(f"  Dataset size:      {len(dataset)} examples")
    print(f"  Selected index:    {idx}")
    print(f"  File:              {file_name}")
    print(f"  Baseline timing:   {baseline_timing:.2f}s")
    if target_marker:
        print(f"  Target marker:     {target_marker}")

    # ------------------------------------------------------------------
    # 2. Process through AstProgram
    # ------------------------------------------------------------------
    section("2. AST PROCESSING")
    original_code = str(example["original_program"])
    print(f"  Original code length: {len(original_code)} chars")

    program = AstProgram().from_code(original_code)
    pp_code, llm_code = program.process(print_ast=False)

    target_assert = program.get_target_assert()
    markers = program.get_available_markers()

    print(f"  Loops found:       {program.num_loops}")
    print(f"  Target assertion:  {target_assert}")
    print(f"  Available markers: {markers}")

    print(f"\n{THIN_SEP}")
    print("  LLM-facing code (what the model sees):")
    print(THIN_SEP)
    print(llm_code)

    # ------------------------------------------------------------------
    # 3. Set up the verifier
    # ------------------------------------------------------------------
    section("3. VERIFIER SETUP")
    uautomizer_path = GC.UAUTOMIZER_PATHS["25"]
    property_file = GC.PROPERTIES_DIR / "unreach-call.prp"

    # Check prerequisites
    missing = []
    if not uautomizer_path.exists():
        missing.append(f"UAutomizer not found at {uautomizer_path}")
    runlim_path = GC.TOOLS_DIR / "runlim" / "runlim"
    if not runlim_path.exists():
        missing.append(f"runlim not found at {runlim_path}")
    if not property_file.exists():
        missing.append(f"Property file not found at {property_file}")

    if missing:
        for m in missing:
            print(f"  [MISSING] {m}")
        print("\n  Cannot run verification — install prerequisites first (see README).")
        print("  The AST processing above still works without a verifier.")
        return

    verifier = UAutomizerVerifier(
        uautomizer_path=uautomizer_path,
        property_file_path=property_file,
        arch="32bit",
        timeout_seconds=args.timeout,
        version="25",
        memory_limit_mb=GC.MEMORY_LIMIT_MB,
        use_semaphore=False,  # Single-threaded demo, no need for semaphore
    )
    print(f"  UAutomizer:  {uautomizer_path}")
    print(f"  Property:    {property_file}")
    print(f"  Timeout:     {args.timeout}s")

    # ------------------------------------------------------------------
    # 4. Baseline verification (no invariant — just the target assertion)
    # ------------------------------------------------------------------
    section("4. BASELINE VERIFICATION (target assertion only)")
    baseline_code = program.get_program_with_assertion(
        property_to_assert=target_assert,
        assumptions=[],
        for_llm=False,
    )
    print("  Generated baseline program (with PATCH header, no invariant).")
    print(f"  Code length: {len(baseline_code)} chars\n")
    print(f"{THIN_SEP}")
    print(baseline_code)
    print(THIN_SEP)

    with tempfile.TemporaryDirectory(prefix="wonda_demo_") as tmp:
        reports_dir = Path(tmp) / "baseline"
        reports_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Running UAutomizer (timeout={args.timeout}s) ...")
        baseline_report = run_single_verification(
            verifier, baseline_code, reports_dir, label="baseline"
        )
        print(f"\n  Decision:       {baseline_report.decision}")
        print(f"  Reason:         {baseline_report.decision_reason}")
        print(f"  Time taken:     {baseline_report.time_taken:.2f}s")

    # ------------------------------------------------------------------
    # 5. Invariant-assisted verification (decision procedure)
    # ------------------------------------------------------------------
    # Use a simple candidate invariant based on the program structure.
    # In the real pipeline, the LLM proposes this.
    inv_marker = target_marker or (f"INVARIANT_MARKER_1" if program.num_loops > 0 else None)
    if inv_marker is None or inv_marker not in markers:
        print(f"\n  No suitable loop marker for invariant placement — skipping invariant demo.")
        return

    section("5. INVARIANT-ASSISTED VERIFICATION (decision procedure)")
    # We use a trivially true invariant "1" (always true) as a safe demo.
    # Replace with a real invariant expression for meaningful results.
    candidate = Property(content="1", marker_name=inv_marker)
    print(f"  Candidate invariant: {candidate}")
    print(f"  Target assertion:    {target_assert}\n")

    # 5a. Correctness query — is the candidate a valid invariant?
    correctness_code = program.get_program_with_assertion(
        property_to_assert=candidate,
        assumptions=[],
        for_llm=False,
    )

    # 5b. Usefulness query — does the target hold assuming the candidate?
    usefulness_code = program.get_program_with_assertion(
        property_to_assert=target_assert,
        assumptions=[candidate],
        for_llm=False,
    )

    print(f"  Correctness program ({len(correctness_code)} chars):")
    print(f"{THIN_SEP}")
    print(correctness_code)
    print(THIN_SEP)

    print(f"\n  Usefulness program ({len(usefulness_code)} chars):")
    print(f"{THIN_SEP}")
    print(usefulness_code)
    print(THIN_SEP)

    with tempfile.TemporaryDirectory(prefix="wonda_demo_") as tmp:
        cor_dir = Path(tmp) / "correctness"
        use_dir = Path(tmp) / "usefulness"
        cor_dir.mkdir(parents=True, exist_ok=True)
        use_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Running correctness check (V(P, {{}}, q)) ...")
        cor_report = run_single_verification(
            verifier, correctness_code, cor_dir, label="correctness"
        )
        print(f"    da = {cor_report.decision}  ({cor_report.time_taken:.2f}s)")

        print(f"\n  Running usefulness check (V(P, {{q}}, p*)) ...")
        use_report = run_single_verification(
            verifier, usefulness_code, use_dir, label="usefulness"
        )
        print(f"    db = {use_report.decision}  ({use_report.time_taken:.2f}s)")

    # ------------------------------------------------------------------
    # 6. Apply decision calculus
    # ------------------------------------------------------------------
    section("6. DECISION PROCEDURE")
    da = cor_report.decision
    db = use_report.decision

    print(f"  da (correctness): {da}")
    print(f"  db (usefulness):  {db}\n")

    if db == "FALSE":
        final = "FALSE"
        rule = "DEC-FALSE: db = F => decide FALSE (short-circuit refutation)"
    elif da == "TRUE" and db in {"TRUE", "UNKNOWN", "TIMEOUT", "ERROR"}:
        final = "TRUE" if db == "TRUE" else "UNKNOWN"
        rule = f"DEC-PROP: da = T, db = {db} => decide {final}"
    elif da != "TRUE" and db != "FALSE":
        final = "UNKNOWN"
        rule = f"DEC-U: da != T, db != F => decide UNKNOWN"
    else:
        final = "UNKNOWN"
        rule = "No rule matched — decide UNKNOWN"

    print(f"  Rule applied:     {rule}")
    print(f"  Final decision:   {final}")
    verification_time = max(cor_report.time_taken, use_report.time_taken)
    print(f"  Verification time: {verification_time:.2f}s (max of parallel queries)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("SUMMARY")
    print(f"  Program:            {file_name}")
    print(f"  Target assertion:   {target_assert.content}")
    print(f"  Candidate inv:      {candidate.content} @ {candidate.marker_name}")
    print(f"  Baseline result:    {baseline_report.decision} ({baseline_report.time_taken:.2f}s)")
    print(f"  Decision procedure: {final} ({verification_time:.2f}s)")
    print(f"  Decision rule:      {rule}")
    if final == "TRUE" and baseline_report.time_taken > 0:
        speedup = baseline_report.time_taken / verification_time if verification_time > 0 else float("inf")
        print(f"  Speedup:            {speedup:.2f}x vs baseline")
    print(
        f"\n  Note: This demo uses candidate invariant '1' (trivially true).\n"
        f"  In the real pipeline, an LLM proposes a meaningful invariant\n"
        f"  that helps the verifier prove the target property faster.\n"
    )


if __name__ == "__main__":
    main()
