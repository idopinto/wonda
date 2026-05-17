"""
Measure UAutomizer baseline verification times on Code2Inv programs.

Runs the plain verifier (no invariants) on every .c/.i file in the Code2Inv
benchmark directory, records wall-clock times, and writes a JSON dataset
compatible with wonda's eval_data.load_local_evaluation_dataset format.

Usage (from project root):
    uv run python scripts/rebuttal/baseline_timing_code2inv.py --uautomizer-version 25
    uv run python scripts/rebuttal/baseline_timing_code2inv.py --uautomizer-version 25 --timeout 900
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import configs.global_config as GC
from wonda.verifiers.uautomizer import UAutomizerVerifier


def parse_args():
    parser = argparse.ArgumentParser(description="Measure baseline verification times on Code2Inv programs")
    parser.add_argument(
        "--programs-dir", type=str,
        default="data/eval/code2inv/orig_programs",
        help="Directory containing .c/.i programs (relative to project root or absolute)",
    )
    parser.add_argument(
        "--uautomizer-version", type=str, default="25",
        choices=list(GC.UAUTOMIZER_PATHS.keys()),
    )
    parser.add_argument("--arch", type=str, default="32bit", choices=["32bit", "64bit"])
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-program timeout in seconds")
    parser.add_argument("--num-trials", type=int, default=3, help="Number of trials per program for median timing")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/eval/code2inv/baseline_uauto<version>)",
    )
    parser.add_argument("--property", type=str, default="unreach-call.prp")
    return parser.parse_args()


def main():
    args = parse_args()

    programs_dir = Path(args.programs_dir)
    if not programs_dir.is_absolute():
        programs_dir = GC.ROOT_DIR / programs_dir

    src_files = sorted([*programs_dir.glob("*.c"), *programs_dir.glob("*.i")])
    if not src_files:
        print(f"No .c or .i files found in {programs_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (
        GC.ROOT_DIR / "data" / "eval" / "code2inv" / f"baseline_uauto{args.uautomizer_version}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "verifier_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[args.uautomizer_version],
        property_file_path=GC.PROPERTIES_DIR / args.property,
        arch=args.arch,
        timeout_seconds=args.timeout,
        version=args.uautomizer_version,
        memory_limit_mb=GC.MEMORY_LIMIT_MB,
    )

    print(f"{'=' * 70}")
    print(f"Code2Inv Baseline Timing — UAutomizer v{args.uautomizer_version}")
    print(f"{'=' * 70}")
    print(f"Programs dir : {programs_dir}")
    print(f"Programs     : {len(src_files)}")
    print(f"Trials/prog  : {args.num_trials}")
    print(f"Timeout      : {args.timeout}s")
    print(f"Output dir   : {output_dir}")
    print(f"{'=' * 70}\n")

    results = []
    t_start = time.time()

    for i, src_file in enumerate(src_files, 1):
        print(f"[{i}/{len(src_files)}] {src_file.name}")
        prog_reports = reports_dir / src_file.stem
        prog_reports.mkdir(parents=True, exist_ok=True)

        trial_times = []
        trial_decisions = []
        for trial in range(args.num_trials):
            trial_dir = prog_reports / f"trial_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            report = verifier.verify(
                program_path=src_file,
                reports_dir=trial_dir,
                timeout_seconds=args.timeout,
            )
            trial_times.append(report.time_taken)
            trial_decisions.append(report.decision)
            print(f"    trial {trial}: {report.decision} in {report.time_taken:.2f}s")

        median_time = statistics.median(trial_times)
        entry = {
            "file": src_file.name,
            "original_program": src_file.read_text().strip(),
            "median_timing": median_time,
            "baseline_details": {
                "uautomizer_version": args.uautomizer_version,
                "num_trials": args.num_trials,
                "timeout": args.timeout,
                "all_times": trial_times,
                "all_decisions": trial_decisions,
                "median_time": median_time,
                "mean_time": statistics.mean(trial_times),
            },
        }
        results.append(entry)
        print(f"    -> median={median_time:.2f}s  decisions={trial_decisions}\n")

    elapsed = time.time() - t_start

    dataset_path = output_dir / "baseline_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "uautomizer_version": args.uautomizer_version,
        "arch": args.arch,
        "timeout": args.timeout,
        "num_trials": args.num_trials,
        "num_programs": len(results),
        "total_elapsed_seconds": elapsed,
        "programs": [
            {
                "file": r["file"],
                "median_timing": r["median_timing"],
                "decisions": r["baseline_details"]["all_decisions"],
                "all_times": r["baseline_details"]["all_times"],
            }
            for r in results
        ],
    }
    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"BASELINE TIMING SUMMARY — UAutomizer v{args.uautomizer_version}")
    print(f"{'=' * 70}")
    print(f"{'Program':<65} {'Median(s)':>9} {'Decision':<10}")
    print("-" * 90)
    for r in results:
        d = r["baseline_details"]
        majority_decision = max(set(d["all_decisions"]), key=d["all_decisions"].count)
        print(f"{r['file']:<65} {r['median_timing']:>9.2f} {majority_decision:<10}")

    timings = [r["median_timing"] for r in results]
    print("-" * 90)
    print(f"{'Total programs:':<65} {len(results)}")
    print(f"{'Mean baseline time:':<65} {statistics.mean(timings):.2f}s")
    print(f"{'Median baseline time:':<65} {statistics.median(timings):.2f}s")
    print(f"{'Total wall-clock time:':<65} {elapsed:.1f}s")
    print(f"\nDataset saved to:  {dataset_path}")
    print(f"Summary saved to:  {summary_path}")


if __name__ == "__main__":
    main()
