"""UAutomizer verifier interface for running verification and parsing results."""

import argparse
import json
import os
import subprocess
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from configs import global_config as GC


@dataclass
class VerifierCallReport:
    """
    Report containing verification results and metadata.
    Attributes:
        decision (str): The result of the verification by UAutomizer or from exception.
        Possible values: "TRUE", "FALSE", "UNKNOWN", "ERROR", "TIMEOUT".
        decision_reason (str): for ERROR / UNKNOWN cases.
        time_taken (float): The total execution time for the verifier call, in seconds.
        runlim (dict): Runlim output.
        reports_dir (str): Path to directory where verification outputs such as logs, error files,
            witness files (e.g., .yml, .graphml) are saved.
    """

    decision: str = "UNKNOWN"
    decision_reason: str = ""
    time_taken: float = 0.0

    runlim: dict = field(default_factory=lambda: {
        "host": "",
        "status": "",
        "cpu_time_limit": 0,
        "real_time_limit": 0,
        "space_limit": 0,
        "cpu_time": 0.0,
        "real_time": 0.0,
        "space": 0,
    })
    reports_dir: str = ""

    def to_dict(self) -> dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "time_taken": self.time_taken,
            "runlim": self.runlim,
            "reports_dir": self.reports_dir, # path to directory where verification outputs such as logs, error files, witness files (e.g., .yml, .graphml) are saved.
        }

    def save_json(self, file_path: Path) -> None:
        """Save report as JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, file_path: Path) -> "VerifierCallReport":
        """Load report from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(
            decision=data["decision"],
            decision_reason=data["decision_reason"],
            time_taken=data["time_taken"],
            runlim=data["runlim"],
            reports_dir=data["reports_dir"],
        )


def write_file(file_path: Path, content: str) -> None:
    """Helper to write content to file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)

class UAutomizerVerifier:
    def __init__(
        self,
        uautomizer_path: Path,
        property_file_path: Path,
        arch: str = "32bit",
        timeout_seconds: float = 600.0,
        version: str = "25",
        memory_limit_mb: int = GC.MEMORY_LIMIT_MB,
    ):
        self.uautomizer_path = uautomizer_path
        self.property_file_path = property_file_path
        self.arch = arch
        self.timeout_seconds = timeout_seconds
        self.version = version
        self.memory_limit_mb = memory_limit_mb

    def verify(
        self, program_path: Path, reports_dir: Path, timeout_seconds: float = None,
        label: str = None,
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
            label: Optional label for logging (e.g., "inv_123/correctness").

        Returns:
            VerifierCallReport with verification results and metadata.
        """
        reports_dir.mkdir(parents=True, exist_ok=True)
        runlim_path = GC.TOOLS_DIR / "runlim" / "runlim"
        log_file_path = reports_dir / f"{program_path.stem}.log"
        err_file_path = reports_dir / f"{program_path.stem}.err"
        runlim_output_file = reports_dir / f"{program_path.stem}.runlim"

        # Validate required files exist
        for path in [self.uautomizer_path, program_path, self.property_file_path]:
            if not path.exists():
                err_msg = f"Required file not found: {path}"
                return VerifierCallReport(
                    reports_dir=str(reports_dir),
                    decision="ERROR",
                    decision_reason=err_msg,
                )

        # Build command
        # TODO: add memory limitation constraint 16gb
        # https://github.com/arminbiere/runlim
    
        runlim_command = [
            runlim_path,
            # "-t", str(int(self.timeout_seconds)),      # CPU time limit
            "-r", str(int(self.timeout_seconds)),      # Wall time limit
            "-s", str(self.memory_limit_mb),      # Memory limit in MB  
            "-o", str(runlim_output_file),   # Runlim output file
        ]
        command = [
            *runlim_command,
            "python3",
            str(self.uautomizer_path),
            "--spec",
            str(self.property_file_path),
            "--architecture",
            self.arch,
            "--file",
            str(program_path),
            "--full-output",
            "--witness-dir",
            str(reports_dir),
            "--witness-name",
            f"{program_path.stem}_witness",
        ]

        # Setup environment with uautomizer directory in PATH for SMT solvers
        env = os.environ.copy()
        env["PATH"] = (
            str(self.uautomizer_path.parent) + os.pathsep + env.get("PATH", "")
        )

        report = VerifierCallReport(reports_dir=str(reports_dir))
        temp_work_dir = Path(tempfile.mkdtemp(prefix="uautomizer_"))
        try:
            completed_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=env,
                cwd=temp_work_dir,
                start_new_session=True,
            )

            write_file(log_file_path, completed_process.stdout)
            write_file(err_file_path, completed_process.stderr)

            # Wait for runlim output file to be fully written (contains status line)
            # Under heavy parallelism, file I/O may lag behind subprocess completion
            runlim_text = ""
            max_wait_seconds = 5.0
            poll_interval = 0.1
            waited = 0.0
            while waited < max_wait_seconds:
                if runlim_output_file.exists():
                    runlim_text = runlim_output_file.read_text()
                    # Check if runlim has finished writing (status line present)
                    if "[runlim] status:" in runlim_text:
                        break
                time.sleep(poll_interval)
                waited += poll_interval
            
            # Final read after waiting
            if runlim_output_file.exists():
                runlim_text = runlim_output_file.read_text()
            
            if waited > 0 and "[runlim] status:" not in runlim_text:
                print(f"WARNING: Runlim output incomplete after {waited:.1f}s wait. "
                      f"Return code: {completed_process.returncode}, "
                      f"Stderr (last 200): {(completed_process.stderr or '')[-200:]!r}")
            report.runlim = self._parse_runlim_output(runlim_text)
            label_prefix = f"[{label}] " if label else ""
            # print(f"{label_prefix}runlim output: {report.runlim}")
            
            # Handle different runlim statuses
            if report.runlim["status"] == "ok":
                report.decision, report.decision_reason = self._parse_uautomizer_output(completed_process.stdout)
            elif report.runlim["status"] == "out of memory":
                report.decision = "ERROR"
                report.decision_reason = "Out of memory"
            elif report.runlim["status"] == "out of time":
                report.decision = "TIMEOUT"
                report.decision_reason = "Out of time"
            elif report.runlim["status"] is None:
                # Runlim didn't finish writing - child likely crashed immediately
                # Fall back to subprocess info for diagnosis
                report.decision = "ERROR"
                stderr_snippet = (completed_process.stderr or "")[:500]
                stdout_snippet = (completed_process.stdout or "")[:500]
                report.decision_reason = (
                    f"Runlim incomplete (child crashed?). "
                    f"returncode={completed_process.returncode}, "
                    f"stderr={stderr_snippet!r}, stdout_tail={stdout_snippet[-200:]!r}"
                )
            else:
                report.decision = "ERROR"
                report.decision_reason = f"Unknown runlim status: {report.runlim['status']}"
            
            report.time_taken = report.runlim.get("real_time") or 0.0

        except Exception as e:
            import traceback
            report.decision = "ERROR"
            report.decision_reason = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            write_file(err_file_path, report.decision_reason)
            print(f"[VERIFIER ERROR] {report.decision_reason}")

        finally:
            shutil.rmtree(temp_work_dir)
        return report

    def _parse_runlim_output(self, output: str) -> str:
        import re

        def get_value(pattern, output, cast=str, default=None):
            match = re.search(pattern, output, re.MULTILINE)
            if match:
                val = match.group(1)
                try:
                    return cast(val)
                except Exception:
                    return val if cast is str else default
            return default

        parsed = {}

        # Host
        parsed["host"] = get_value(r"\[runlim\] host:\s+([^\n]+)", output, str)

        # Time limit (seconds, int)
        parsed["cpu_time_limit"] = get_value(r"\[runlim\] time limit:\s+([0-9]+) seconds", output, int)

        # Real time limit (seconds, int)
        parsed["real_time_limit"] = get_value(r"\[runlim\] real time limit:\s+([0-9]+) seconds", output, int)

        # Space limit (MB, int)
        parsed["space_limit"] = get_value(r"\[runlim\] space limit:\s+([0-9]+) MB", output, int)

        # Status
        parsed["status"] = get_value(r"\[runlim\] status:\s+([^\n]+)", output, str)

        # Real time (seconds, float)
        parsed["real"] = get_value(r"\[runlim\] real:\s+([0-9.]+) seconds", output, float)

        # CPU time (seconds, float)
        parsed["time"] = get_value(r"\[runlim\] time:\s+([0-9.]+) seconds", output, float)

        # Space (MB, int)
        parsed["space"] = get_value(r"\[runlim\] space:\s+([0-9]+) MB", output, int)

        return {
            "host": parsed["host"],
            "status": parsed["status"],
            "cpu_time_limit": parsed["cpu_time_limit"],
            "real_time_limit": parsed["real_time_limit"],
            "space_limit": parsed["space_limit"],
            "cpu_time": parsed["time"],
            "real_time": parsed["real"],
            "space": parsed["space"],
        }

    def _parse_uautomizer_output(self, output: str) -> str:
        import re
        # print(f"UAutomizer output: {output}")
        result_pattern = r"Result:\s*\n(ERROR|TRUE|FALSE|UNKNOWN)(?::\s*(.+))?"
        match = re.search(result_pattern, output, re.MULTILINE)

        if match:
            decision = match.group(1)
            reason = match.group(2).strip() if match.group(2) else ""
            return decision, reason
        return "ERROR", "Unable to parse UAutomizer output."


def parse_args():
    parser = argparse.ArgumentParser(description="Run UAutomizer verifier")
    parser.add_argument("--program-dir", type=str, default="data/examples")
    parser.add_argument(
        "--program-name", type=str, default="paper_example_direct.c"
    )
    # parser.add_argument("--property_name", type=str, default='unreach-call.prp')
    parser.add_argument("--arch", type=str, default="32bit", choices=["32bit", "64bit"])
    parser.add_argument("--reports-dir", type=str, default="example_reports")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--uautomizer-version", type=str, default="25", choices=["23", "24", "25", "26"]
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    program_path = GC.ROOT_DIR / args.program_dir / args.program_name
    reports_dir = GC.EXPERIMENTS_DIR / args.reports_dir / args.program_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    uautomizer_path = GC.UAUTOMIZER_PATHS[args.uautomizer_version]
    property_file_path = GC.PROPERTIES_DIR / "unreach-call.prp"
    verifier = UAutomizerVerifier(
        uautomizer_path=uautomizer_path,
        property_file_path=property_file_path,
        memory_limit_mb=GC.MEMORY_LIMIT_MB,
        arch=args.arch,
        timeout_seconds=args.timeout_seconds,
        version=args.uautomizer_version,
    )
    print(f"==== Program ====\n{program_path.read_text()}")
    result = verifier.verify(
        program_path=program_path,
        reports_dir=reports_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print("\n --- Running UAutomizer Verification with Runlim ---")
    print(f"  Program: {program_path}")
    print(f"  Timeout: {args.timeout_seconds}s")
    print(f"  Memory limit: {GC.MEMORY_LIMIT_MB}MB")
    print(f"  Runlim path: {GC.TOOLS_DIR / 'runlim' / 'runlim'}")

    print("\n--- Verification Complete ---")
    result_dict = result.to_dict()
    for key, value in result_dict.items():
        print(f"  {key}: {value}")

    program_name = args.program_name.split(".")[0]
    print("--------------------------------")
    print(f"WitnessGraphML file: {reports_dir / f'{program_name}_witness.graphml'}")
    print(f"Witness YAML file: {reports_dir / f'{program_name}_witness.yml'}")
    print(f"Error file: {reports_dir / f'{program_name}.err'}")
    print(f"Log file: {reports_dir / f'{program_name}.log'}")
    print(f"Runlim file: {reports_dir / f'{program_name}.runlim'}")
    print("--------------------------------")



# uv run -m wonda.verifiers.uautomizer
# uv run -m wonda.verifiers.uautomizer --program-name test.c
