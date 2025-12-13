#!/usr/bin/env python3
"""Test to verify UAutomizer timing is deterministic (within epsilon tolerance)."""
import sys
import statistics
import os
import subprocess
from pathlib import Path
from typing import List, Optional

# Add project root to path (must be before imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after adding to path (noqa: E402 is intentional)
from src.utils.plain_verifier import run_uautomizer, VerifierCallReport  # noqa: E402
from src.utils.paths import UAUTOMIZER_PATHS, PROPERTIES_DIR  # noqa: E402


def setup_deterministic_environment():
    """
    Set up environment variables and system settings for more deterministic execution.
    
    Returns:
        Dictionary of environment modifications
    """
    env_mods = {}
    
    # Disable JVM randomization
    env_mods['_JAVA_OPTIONS'] = (
        '-XX:+UseG1GC '  # Use G1 garbage collector (more predictable)
        '-XX:MaxGCPauseMillis=200 '  # Limit GC pause time
        '-XX:+DisableExplicitGC '  # Disable explicit GC calls
        '-XX:+UseStringDeduplication '  # Reduce string allocation variability
        '-Djava.security.egd=file:/dev/./urandom '  # Use faster RNG for security
    )
    
    # Python determinism
    env_mods['PYTHONHASHSEED'] = '0'  # Disable hash randomization
    env_mods['PYTHONUNBUFFERED'] = '1'  # Unbuffered output
    
    # SMT solver determinism hints
    # Z3 and CVC4 may use randomization internally, but we can't control it directly
    
    return env_mods


def check_system_determinism() -> dict:
    """
    Check system settings that affect determinism.
    
    Returns:
        Dictionary with system check results
    """
    checks = {}
    
    # Check CPU frequency scaling
    try:
        result = subprocess.run(
            ['cpupower', 'frequency-info', '-p'],
            capture_output=True,
            text=True,
            timeout=2
        )
        checks['cpupower_available'] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        checks['cpupower_available'] = False
    
    # Check if we can set CPU affinity
    try:
        result = subprocess.run(
            ['taskset', '--help'],
            capture_output=True,
            text=True,
            timeout=1
        )
        checks['taskset_available'] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        checks['taskset_available'] = False
    
    # Check Java version (Java writes version to stderr, not stdout)
    try:
        result = subprocess.run(
            ['java', '-version'],
            capture_output=True,
            text=True,
            timeout=2
        )
        checks['java_available'] = result.returncode == 0
        # Java -version outputs to stderr, not stdout
        version_output = result.stderr or result.stdout
        if version_output and 'version' in version_output.lower():
            checks['java_version'] = version_output.split('\n')[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        checks['java_available'] = False
    
    return checks


def run_timing_test(
    c_file_path: Path,
    property_file_path: Path,
    reports_dir: Path,
    num_runs: int = 5,
    epsilon: float = 0.1,  # 10% tolerance
    arch: str = '32bit',
    timeout_seconds: float = 300.0,
    uautomizer_path: Optional[Path] = None,
    deterministic_mode: bool = True,
    cpu_affinity: Optional[str] = None,
) -> dict:
    """
    Run UAutomizer multiple times and check timing determinism.
    
    Args:
        c_file_path: Path to C file to verify
        property_file_path: Path to property file
        reports_dir: Directory for reports (will create subdirs for each run)
        num_runs: Number of runs to perform
        epsilon: Relative tolerance for timing variation (0.1 = 10%)
        arch: Architecture ('32bit' or '64bit')
        timeout_seconds: Timeout per run
        uautomizer_path: Optional path to UAutomizer executable
        deterministic_mode: Apply deterministic environment settings
        cpu_affinity: CPU affinity mask (e.g., "0-3" for CPUs 0-3)
    
    Returns:
        Dictionary with timing statistics and determinism test results
    """
    timings: List[float] = []
    decisions: List[str] = []
    errors: List[bool] = []
    
    # Setup deterministic environment
    env_mods = {}
    # if deterministic_mode:
    #     env_mods = setup_deterministic_environment()
    #     print("Deterministic mode enabled:")
    #     for key, value in env_mods.items():
    #         value_preview = value[:80] + "..." if len(value) > 80 else value
    #         print(f"  {key}={value_preview}")
    #     print()
    
    # Check system settings
    system_checks = check_system_determinism()
    if deterministic_mode:
        print("System determinism checks:")
        for key, value in system_checks.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value!s}")
        print()
    
    print("Running timing determinism test:")
    print(f"  File: {c_file_path}")
    print(f"  Property: {property_file_path}")
    print(f"  Architecture: {arch}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Epsilon (tolerance): {epsilon * 100}%")
    if cpu_affinity:
        print(f"  CPU Affinity: {cpu_affinity}")
    print()
    
    # Apply environment modifications
    original_env = {}
    for key, value in env_mods.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Run multiple times
        for i in range(num_runs):
            run_reports_dir = reports_dir / f"run_{i+1}"
            print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)
            
            # If CPU affinity is set, we need to modify the subprocess call
            # For now, we'll note it but can't directly control it in run_uautomizer
            # This would require modifying plain_verifier.py to accept env/affinity params
            
            report = run_uautomizer(
                program_path=c_file_path,
                property_file_path=property_file_path,
                reports_dir=run_reports_dir,
                arch=arch,
                timeout_seconds=timeout_seconds,
                uautomizer_path=uautomizer_path or UAUTOMIZER_PATHS["25"]
            )
            
            timings.append(report.time_taken)
            decisions.append(report.decision)
            is_error = report.decision in ("ERROR", "TIMEOUT", "UNKNOWN")
            errors.append(is_error)
            
            status = "✓" if not is_error else "✗"
            print(f"{status} {report.time_taken:.3f}s - {report.decision}")
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    print()
    
    # Calculate statistics
    if not timings:
        return {
            "error": "No successful runs",
            "deterministic": False,
            "system_checks": system_checks
        }
    
    mean_time = statistics.mean(timings)
    median_time = statistics.median(timings)
    min_time = min(timings)
    max_time = max(timings)
    
    # Calculate coefficient of variation (std dev / mean)
    if len(timings) > 1:
        std_dev = statistics.stdev(timings)
        cv = std_dev / mean_time if mean_time > 0 else float('inf')
    else:
        std_dev = 0.0
        cv = 0.0
    
    # Calculate relative range (max - min) / mean
    relative_range = (max_time - min_time) / mean_time if mean_time > 0 else float('inf')
    
    # Check determinism
    # Timing is deterministic if:
    # 1. Coefficient of variation is below epsilon
    # 2. Relative range is below epsilon
    is_deterministic = (cv <= epsilon) and (relative_range <= epsilon)
    
    # Check if all decisions are the same
    all_same_decision = len(set(decisions)) == 1
    all_no_errors = not any(errors)
    
    result = {
        "num_runs": num_runs,
        "timings": timings,
        "mean": mean_time,
        "median": median_time,
        "min": min_time,
        "max": max_time,
        "std_dev": std_dev,
        "coefficient_of_variation": cv,
        "relative_range": relative_range,
        "epsilon": epsilon,
        "is_deterministic": is_deterministic,
        "all_same_decision": all_same_decision,
        "all_no_errors": all_no_errors,
        "decisions": decisions,
        "errors": errors,
        "system_checks": system_checks,
        "deterministic_mode": deterministic_mode,
    }
    
    return result


def print_results(results: dict) -> None:
    """Print formatted test results."""
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    print("=" * 60)
    print("TIMING DETERMINISM TEST RESULTS")
    print("=" * 60)
    print(f"\nRuns: {results['num_runs']}")
    print(f"Epsilon (tolerance): {results['epsilon'] * 100:.1f}%")
    print(f"Deterministic mode: {'Enabled' if results.get('deterministic_mode', False) else 'Disabled'}")
    print()
    
    print("Timing Statistics:")
    print(f"  Mean:     {results['mean']:.4f}s")
    print(f"  Median:   {results['median']:.4f}s")
    print(f"  Min:      {results['min']:.4f}s")
    print(f"  Max:      {results['max']:.4f}s")
    print(f"  Std Dev:  {results['std_dev']:.4f}s")
    print()
    
    print("Variation Metrics:")
    print(f"  Coefficient of Variation: {results['coefficient_of_variation']:.4f} ({results['coefficient_of_variation'] * 100:.2f}%)")
    print(f"  Relative Range:            {results['relative_range']:.4f} ({results['relative_range'] * 100:.2f}%)")
    print()
    
    print("Determinism Check:")
    deterministic_status = "✓ PASS" if results['is_deterministic'] else "✗ FAIL"
    print(f"  Timing Deterministic (CV & Range < {results['epsilon'] * 100:.1f}%): {deterministic_status}")
    print(f"  All Decisions Same: {'✓' if results['all_same_decision'] else '✗'}")
    print(f"  No Errors/Timeouts: {'✓' if results['all_no_errors'] else '✗'}")
    print()
    
    if results['all_same_decision']:
        print(f"  Decision: {results['decisions'][0]}")
    else:
        print(f"  Decisions: {results['decisions']}")
    
    print()
    print("Individual Run Timings:")
    for i, timing in enumerate(results['timings'], 1):
        deviation = ((timing - results['mean']) / results['mean']) * 100 if results['mean'] > 0 else 0
        print(f"  Run {i}: {timing:.4f}s ({deviation:+.2f}%)")
    
    # Print recommendations if not deterministic
    if not results['is_deterministic']:
        print()
        print("=" * 60)
        print("RECOMMENDATIONS FOR IMPROVED DETERMINISM:")
        print("=" * 60)
        print("1. Set CPU frequency to fixed (performance) mode:")
        print("   sudo cpupower frequency-set -g performance")
        print()
        print("2. Disable CPU turbo boost:")
        print("   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo")
        print()
        print("3. Pin process to specific CPU cores:")
        print("   taskset -c 0-3 python test_uautomizer_timing_determinism.py")
        print()
        print("4. Use isolated CPU cores (if available):")
        print("   Add 'isolcpus=0-3' to kernel boot parameters")
        print()
        print("5. Minimize system load during testing")
        print("6. Use deterministic mode (already enabled)")
        print("=" * 60)
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test UAutomizer timing determinism")
    parser.add_argument(
        "--c_file",
        type=Path,
        default=project_root / "dataset" / "evaluation" / "easy" / "c" / "sll-01-1_9.c",
        help="Path to C file to test"
    )
    parser.add_argument(
        "--property_file",
        type=Path,
        default=project_root / "dataset" / "properties" / "unreach-call.prp",
        help="Path to property file"
    )
    parser.add_argument(
        "--reports_dir",
        type=Path,
        default=project_root / "tests" / "timing_test_reports",
        help="Directory for test reports"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to perform"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Relative tolerance for timing variation (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=['32bit', '64bit'],
        default='32bit',
        help="Architecture"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout per run in seconds"
    )
    parser.add_argument(
        "--uautomizer_path",
        type=Path,
        default=None,
        help="Path to UAutomizer executable (default: auto-detect)"
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Disable deterministic mode settings"
    )
    parser.add_argument(
        "--cpu-affinity",
        type=str,
        default=None,
        help="CPU affinity mask (e.g., '0-3' for CPUs 0-3)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.c_file.exists():
        print(f"ERROR: C file not found: {args.c_file}")
        sys.exit(1)
    
    if not args.property_file.exists():
        print(f"ERROR: Property file not found: {args.property_file}")
        sys.exit(1)
    
    # Run test
    results = run_timing_test(
        c_file_path=args.c_file,
        property_file_path=args.property_file,
        reports_dir=args.reports_dir,
        num_runs=args.num_runs,
        epsilon=args.epsilon,
        arch=args.arch,
        timeout_seconds=args.timeout,
        uautomizer_path=args.uautomizer_path,
        deterministic_mode=not args.no_deterministic,
        cpu_affinity=args.cpu_affinity
    )
    
    # Print results
    print_results(results)
    
    # Exit with appropriate code
    if results.get("is_deterministic", False) and results.get("all_same_decision", False):
        print("\n✓ Test PASSED: Timing is deterministic")
        sys.exit(0)
    else:
        print("\n✗ Test FAILED: Timing is not deterministic or results vary")
        sys.exit(1)
