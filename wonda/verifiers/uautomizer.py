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
            "decision": self.decision,
            "time_taken": self.time_taken,
            "reports_dir": self.reports_dir,
            "decision_reason": self.decision_reason,
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
            time_taken=data["time_taken"],
            reports_dir=data["reports_dir"],
            decision_reason=data["decision_reason"],
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
    ):
        self.uautomizer_path = uautomizer_path
        self.property_file_path = property_file_path
        self.arch = arch
        self.timeout_seconds = timeout_seconds
        self.version = version

    def verify(
        self, program_path: Path, reports_dir: Path, timeout_seconds: float = None
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
        for path in [self.uautomizer_path, program_path, self.property_file_path]:
            if not path.exists():
                err_msg = f"Required file not found: {path}"
                return VerifierCallReport(
                    reports_dir=str(reports_dir),
                    decision="ERROR",
                    decision_reason=err_msg,
                )

        # Build command
    
        command = [
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
            start_time = time.perf_counter()
            completed_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
                if timeout_seconds is None
                else timeout_seconds,
                check=False,
                env=env,
                cwd=temp_work_dir,
            )
            report.time_taken = time.perf_counter() - start_time

            write_file(log_file_path, completed_process.stdout)
            write_file(err_file_path, completed_process.stderr)

            report.decision, report.decision_reason = self._parse_uautomizer_output(
                completed_process.stdout
            )

        except subprocess.TimeoutExpired as e:
            report.decision = "TIMEOUT"
            report.time_taken = (
                self.timeout_seconds if timeout_seconds is None else timeout_seconds
            )
            stdout_content = (
                e.stdout.decode("utf-8", errors="ignore")
                if isinstance(e.stdout, bytes)
                else (e.stdout or "")
            )
            stderr_content = (
                e.stderr.decode("utf-8", errors="ignore")
                if isinstance(e.stderr, bytes)
                else (e.stderr or "")
            )
            write_file(log_file_path, stdout_content)
            write_file(err_file_path, stderr_content)

        except Exception as e:
            import traceback
            report.decision = "ERROR"
            report.decision_reason = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            write_file(err_file_path, report.decision_reason)
            print(f"[VERIFIER ERROR] {report.decision_reason}")

        finally:
            shutil.rmtree(temp_work_dir)
        return report

    def _parse_uautomizer_output(self, output: str) -> str:
        import re

        result_pattern = r"Result:\s*\n(ERROR|TRUE|FALSE|UNKNOWN)(?::\s*(.+))?"
        match = re.search(result_pattern, output, re.MULTILINE)

        if match:
            decision = match.group(1)
            reason = match.group(2).strip() if match.group(2) else ""
            return decision, reason
        return "ERROR", "Unable to parse UAutomizer output."


def parse_args():
    parser = argparse.ArgumentParser(description="Run UAutomizer verifier")
    parser.add_argument("--program_dir", type=str, default="examples")
    parser.add_argument(
        "--program_name", type=str, default="benchmark24_conjunctive_1.c"
    )
    # parser.add_argument("--property_name", type=str, default='unreach-call.prp')
    parser.add_argument("--arch", type=str, default="32bit", choices=["32bit", "64bit"])
    parser.add_argument("--reports_dir", type=str, default="example_reports")
    parser.add_argument("--timeout_seconds", type=float, default=600.0)
    parser.add_argument(
        "--uautomizer_version", type=str, default="25", choices=["23", "24", "25", "26"]
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
    print("\n --- Running UAutomizer Verification ---")
    print(f"  UAutomizer: {uautomizer_path}")
    print(f"  Program: {program_path}")
    print(f"  Timeout: {args.timeout_seconds}s")

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
    print("--------------------------------")


# uv run -m wonda.verifiers.uautomizer
# uv run -m wonda.verifiers.uautomizer --program_name test.c
