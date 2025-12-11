"""UAutomizer verifier interface for running verification and parsing results."""
import argparse
import json
import os
import subprocess
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from src.utils.paths import UAUTOMIZER_PATHS, EXPERIMENTS_DIR, PROPERTIES_DIR    
from src.utils.utils import write_file, parse_uautomizer_output

@dataclass
class VerifierCallReport:
    """
    Report containing verification results and metadata.
    Attributes:
        decision (str): The result of the verification by UAutomizer or from exception. 
        Possible values: "TRUE", "FALSE", "UNKNOWN", "ERROR", "TIMEOUT".
        decision_reason (str): for ERROR / UNKNOWN cases.
        time_taken (float): The total execution time for the verifier call, in seconds.
        reports_dir (str): Path to directory where verification outputs such as logs, error files,
            witness files (e.g., .yml, .graphml) are saved.
    """
    decision: str = "UNKNOWN"
    decision_reason: str = "default"
    time_taken: float = 0.0
    reports_dir: str = ""

    def to_dict(self) -> dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            'decision': self.decision,
            'time_taken': self.time_taken,
            'reports_dir': self.reports_dir,
            'decision_reason': self.decision_reason
        }
    
    def save_json(self, file_path: Path) -> None:
        """Save report as JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, file_path: Path) -> 'VerifierCallReport':
        """Load report from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(
            decision=data['decision'],
            time_taken=data['time_taken'],
            reports_dir=data['reports_dir'],
            decision_reason=data['decision_reason']
        )

def run_uautomizer(
    program_path: Path,
    property_file_path: Path,
    reports_dir: Path,
    arch: str = '32bit',
    timeout_seconds: float = 600.0,
    uautomizer_path: Path = None,
) -> VerifierCallReport:
    """
    Run UAutomizer verifier on a C file.
    
    Args:
        program_path: Path to the C program to verify.
        property_file_path: Path to the property specification file (.prp).
        reports_dir: Directory where log files will be saved.
        arch: Architecture ('32bit' or '64bit').
        timeout_seconds: Maximum execution time in seconds.
        uautomizer_path: Path to UAutomizer executable. If None, uses default.
    
    Returns:
        VerifierCallReport with verification results and metadata.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = reports_dir / f"{program_path.stem}.log"
    err_file_path = reports_dir / f"{program_path.stem}.err"
    
    # Validate required files exist
    for path in [uautomizer_path, program_path, property_file_path]:
        if not path.exists():
            err_msg = f"Required file not found: {path}"
            write_file(err_file_path, err_msg)
            return VerifierCallReport(reports_dir=str(reports_dir), decision="ERROR", decision_reason=err_msg)
    
    # Build command
    command = [
        'python3',
        str(uautomizer_path),
        '--spec', str(property_file_path),
        '--architecture', arch,
        '--file', str(program_path),
        '--full-output',
        '--witness-dir', str(reports_dir),
        '--witness-name', f"{program_path.stem}_witness",
    ]
    
    # Setup environment with uautomizer directory in PATH for SMT solvers
    env = os.environ.copy()
    env['PATH'] = str( uautomizer_path.parent) + os.pathsep + env.get('PATH', '')
    
    report = VerifierCallReport(reports_dir=str(reports_dir))
    temp_work_dir = Path(tempfile.mkdtemp(prefix="uautomizer_"))
    try:
        start_time = time.perf_counter()
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
            cwd=temp_work_dir
        )
        report.time_taken = time.perf_counter() - start_time
        
        write_file(log_file_path, completed_process.stdout)
        write_file(err_file_path, completed_process.stderr)
        
        report.decision, report.decision_reason = parse_uautomizer_output(completed_process.stdout)

    except subprocess.TimeoutExpired as e:
        report.decision = "TIMEOUT"
        report.time_taken = timeout_seconds
        stdout_content = e.stdout.decode('utf-8', errors='ignore') if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr_content = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else (e.stderr or "")
        write_file(log_file_path, stdout_content)
        write_file(err_file_path, stderr_content)
        
    except Exception as e:
        report.decision = "ERROR"
        report.decision_reason = str(e)
        write_file(err_file_path, report.decision_reason)
    
    finally:
        shutil.rmtree(temp_work_dir)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run UAutomizer verifier")
    parser.add_argument("--program_name", type=str, default='benchmark24_conjunctive_1')
    parser.add_argument("--property_name", type=str, default='unreach-call')
    parser.add_argument("--arch", type=str, default='32bit', choices=['32bit', '64bit'])
    parser.add_argument("--reports_dir", type=str, default='example_reports')
    parser.add_argument("--timeout_seconds", type=float, default=600.0)
    parser.add_argument("--uautomizer_version", type=str, default='25', choices=['23', '24', '25', '26'])
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    args.program_name = "freire"
    program_path = Path(f"/cs/labs/guykatz/idopinto12/projects/loop_invariant_generation/RLInv/examples/{args.program_name}.c")
    property_file_path = PROPERTIES_DIR / f"{args.property_name}.prp"
    reports_dir = EXPERIMENTS_DIR / args.reports_dir / args.program_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    uautomizer_path = UAUTOMIZER_PATHS[args.uautomizer_version]
    
    print("\n --- Running UAutomizer Verification ---")
    print(f"  Program: {program_path}")
    print(f"  Property: {property_file_path}")
    print(f"  Architecture: {args.arch}")
    print(f"  Timeout: {args.timeout_seconds}s")
    print(f"  UAutomizer path: {uautomizer_path}")
    
    result = run_uautomizer(
        program_path=program_path,
        property_file_path=property_file_path,
        reports_dir=reports_dir,
        arch=args.arch,
        timeout_seconds=args.timeout_seconds,
        uautomizer_path=uautomizer_path
    )
    
    print("\n--- Verification Complete ---")
    result_dict = result.to_dict()
    for key, value in result_dict.items():
        print(f"  {key}: {value}")

    print("--------------------------------")   
    print(f"WitnessGraphML file: {reports_dir / f'{args.program_name}_witness.graphml'}")
    print(f"Witness YAML file: {reports_dir / f'{args.program_name}_witness.yml'}")
    print(f"Error file: {reports_dir / f'{args.program_name}.err'}")
    print(f"Log file: {reports_dir / f'{args.program_name}.log'}")
    print("--------------------------------")


# uv run src/utils/plain_verifier.py --