import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import weave  # type: ignore[import-not-found]

from src.verifiers.uautomizer import UAutomizerVerifier, VerifierCallReport
from src.preprocess.predicate import Predicate
from src.preprocess.program import Program
from src.eval.validate import syntactic_validation

@dataclass
class DecisionProcedureReport:
    final_decision: str = "UNKNOWN"
    decision_rule: str = ""
    # program: Optional[Program] = None
    target: Optional[Predicate] = None
    # target_property_file_path: Optional[Path] = None
    candidate: Optional[Predicate] = None
    # Always keep the raw model output for debugging, even if parsing fails.
    raw_candidate: Optional[str] = None
    target_marker: Optional[str] = None
    is_valid: bool = False
    correctness_report: Optional[VerifierCallReport] = None
    usefulness_report: Optional[VerifierCallReport] = None
    verification_time: float = 0.0  # Only verification time (without model latency)
    model_latency: float = 0.0  # Model inference/token generation time
    total_time: float = 0.0  # verification_time + model_latency
    report_path: str = ""
    program_for_correctness: Optional[str] = None
    program_for_usefulness: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the report to a dictionary for JSON serialization."""
        return {
            "final_decision": self.final_decision,
            "decision_rule": self.decision_rule,
            "target": {
                "content": self.target.content,
                "marker_name": self.target.marker_name,
            }
            if self.target
            else None,
            "candidate": {
                "content": self.candidate.content,
                "marker_name": self.candidate.marker_name,
            }
            if self.candidate
            else None,
            "raw_candidate": self.raw_candidate,
            "is_valid": self.is_valid,
            "target_marker": self.target_marker,
            "correctness_report": self.correctness_report.to_dict()
            if self.correctness_report
            else None,
            "program_for_correctness": self.program_for_correctness,
            "program_for_usefulness": self.program_for_usefulness,
            "usefulness_report": self.usefulness_report.to_dict()
            if self.usefulness_report
            else None,
            "verification_time": self.verification_time,
            "model_latency": self.model_latency,
            "total_time": self.total_time,
            "report_path": self.report_path,
        }

    def save_json(self, report_path: str):
        with open(report_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, report_path: str):
        with open(report_path, "r") as f:
            data = json.load(f)
        return cls(**data)


class DecisionProcedure:
    def __init__(
        self,
        verifier: UAutomizerVerifier,
        program: Program,
        code_dir: Path,
        timeout: float,
    ):
        self.verifier = verifier
        self.program = program
        self.code_dir = code_dir
        self.timeout = timeout
        self.reports_dir = code_dir  # Use code_dir for reports
        # Ensure AST pipeline ran so `assertions` and marker ASTs exist.
        if not getattr(self.program, "marked_ast", None) or not getattr(
            self.program, "llm_code", None
        ):
            self.program.process(print_ast=False)

        # Target is the (single) target assertion recorded during process()
        self.target = (
            self.program.get_target_assert() if self.program.assertions else None
        )
        self.available_markers = self.program.get_available_markers()

    def run_verifier(self, program_str: str, kind: str) -> VerifierCallReport:
        program_path = self.code_dir / f"code_for_{kind}.c"
        with open(program_path, "w") as out_file:
            out_file.write(program_str)
        verifier_report: VerifierCallReport = self.verifier.verify(
            program_path=program_path,
            reports_dir=self.reports_dir,
            timeout_seconds=self.timeout,
        )
        return verifier_report

    def decide(
        self, candidate: Predicate, report: DecisionProcedureReport
    ) -> DecisionProcedureReport:
        program_for_correctness = self.program.get_program_with_assertion(
            predicate=candidate, assumptions=[], for_llm=False
        )

        program_for_usefullness = self.program.get_program_with_assertion(
            predicate=self.target, assumptions=[candidate], for_llm=False
        )
        report.program_for_correctness = program_for_correctness
        report.program_for_usefulness = program_for_usefullness
        # Parallel evaluation: run both verifier queries concurrently
        # da = V(P, Ø, q): Check if q is an invariant
        # db = V(P, {q}, p*): Check if target property holds assuming q is true
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both verifier tasks
            correctness_future = executor.submit(
                self.run_verifier,
                program_str=program_for_correctness,
                kind="correctness",
            )

            usefulness_future = executor.submit(
                self.run_verifier,
                program_str=program_for_usefullness,
                kind="usefulness",
            )

            # Track results as they complete
            invariant_correctness_report = None
            invariant_usefulness_report = None

            # Use as_completed to process results as they arrive (enables short-circuiting DEC-FALSE)
            # DEC-FALSE: If db = F, we can decide F without waiting for da to complete
            for future in as_completed([correctness_future, usefulness_future]):
                result = future.result()
                if future == correctness_future:
                    invariant_correctness_report = result
                elif future == usefulness_future:
                    invariant_usefulness_report = result
                    # Short-circuit DEC-FALSE: if usefulness is Falsified, we can skip waiting for correctness
                    # and decide F immediately (DEC-FALSE doesn't require da)
                    if invariant_usefulness_report.decision == "FALSE":
                        # Try to cancel correctness if it's still running
                        if not correctness_future.done():
                            correctness_future.cancel()
                        # We have what we need for DEC-FALSE, break early
                        break

            # After loop: if correctness completed but wasn't captured, get it now
            if invariant_correctness_report is None and correctness_future.done():
                try:
                    invariant_correctness_report = correctness_future.result()
                except Exception:
                    pass  # If it was cancelled or failed, leave it as None

        decision_rule = ""
        # Apply decision calculus
        final_decision = "UNKNOWN"

        # DEC-FALSE: If db = F, decide F (short-circuit refutation)
        # This is a "short-circuit" because da doesn't need to be T to decide F
        if (
            invariant_usefulness_report
            and invariant_usefulness_report.decision == "FALSE"
        ):
            final_decision = "FALSE"
            decision_rule = "DEC-FALSE"

        # DEC_PROP: If da = T and db ∈ {T, U}, decide db
        # Rule DEC-PROP implements the prove then-use strategy:
        # once the candidate invariant q is established, the outcome is exactly the verifier answer on the goal under the assumption q
        elif (
            invariant_correctness_report
            and invariant_correctness_report.decision == "TRUE"
            and invariant_usefulness_report.decision
            in {"TRUE", "UNKNOWN", "TIMEOUT", "ERROR"}
        ):
            if invariant_usefulness_report.decision == "TRUE":
                final_decision = "TRUE"
            else:
                final_decision = (
                    "UNKNOWN"  # TIMEOUTS AND ERRORS are considered as UNKNOWN
                )
            # final_decision = invariant_usefulness_report.decision
            decision_rule = "DEC-PROP"

        # DEC-U: If da ≠ T and db ≠ F, decide U
        # Rule DEC-U gives explicit conditions for inconclusiveness:
        # the goal is not refuted under q and q is not established as an invariant
        elif (
            invariant_correctness_report
            and invariant_correctness_report.decision != "TRUE"
            and invariant_usefulness_report.decision != "FALSE"
        ):
            final_decision = "UNKNOWN"
            decision_rule = "DEC-U"

        # Calculate verification time: max of both runs since they execute in parallel
        # Use 0 if correctness was cancelled (short-circuited)
        correctness_time = (
            invariant_correctness_report.time_taken
            if invariant_correctness_report
            else 0.0
        )
        usefulness_time = (
            invariant_usefulness_report.time_taken
            if invariant_usefulness_report
            else 0.0
        )
        verification_time = max(correctness_time, usefulness_time)

        # Update the report with decision results (keep all other fields from the initial report)
        report.final_decision = final_decision
        report.decision_rule = decision_rule
        report.correctness_report = invariant_correctness_report
        report.usefulness_report = invariant_usefulness_report
        report.verification_time = verification_time
        report.total_time = (
            verification_time + report.model_latency
        )  # assumes model_latency is set as input

        return report

    def parse_candidate(self, candidate: str) -> Predicate:
        """
        Parse candidate string to extract marker and content.
        
        First tries to extract JSON from ```json ... ``` code block,
        then falls back to parsing the string directly as JSON.
        """
        import re
        
        # Try to extract JSON from markdown code block first
        json_block_pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(json_block_pattern, candidate, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            # Fall back to parsing the candidate directly
            json_str = candidate
        
        json_candidate = json.loads(json_str)
        marker = json_candidate["marker"]
        content = json_candidate["content"]
        return Predicate(content=content, marker_name=marker)

    @weave.op()
    def run(self, candidate: str, model_latency: float, target_marker: Optional[str] = None) -> DecisionProcedureReport:
        final_report = DecisionProcedureReport(
            target=self.target,
            model_latency=model_latency,
            raw_candidate=candidate,
            target_marker=target_marker,
        )
        parsed_candidate: Optional[Predicate] = None
        is_valid = False
        try:
            parsed_candidate = self.parse_candidate(candidate)
            # print(f"Parsed candidate: {parsed_candidate}")
            # print(f"Target marker: {target_marker}")
            is_valid = syntactic_validation(parsed_candidate.content)
            if target_marker:
                is_valid = is_valid and parsed_candidate.marker_name == target_marker
        except Exception:
            # Any parsing/formatting failure => invalid candidate. Keep raw_candidate for inspection.
            is_valid = False
        final_report.is_valid = is_valid
        final_report.candidate = parsed_candidate
        if is_valid:
            assert parsed_candidate is not None
            final_report = self.decide(parsed_candidate, final_report)
        else:
            final_report.final_decision = "INVALID"
        # save the final report to a json file
        report_file_path = self.reports_dir / "decision_report.json"
        final_report.save_json(report_file_path)
        return final_report
