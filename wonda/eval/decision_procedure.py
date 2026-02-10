"""Decision procedure for invariant-based program verification.

Implements the decision calculus from:

    Quokka: Accelerating Program Verification with LLMs via Invariant Synthesis
    Anjiang Wei, Tarun Suresh, Tianran Sun, Haoze Wu, Ke Wang, Alex Aiken
    arXiv:2509.21629  —  https://arxiv.org/abs/2509.21629

Given a candidate invariant *q* and a target assertion *p**, the procedure
issues two parallel verifier queries:

    da = V(P, ∅, q)      — correctness: is q an invariant of program P?
    db = V(P, {q}, p*)    — usefulness:  does the target hold assuming q?

and combines the outcomes via three decision rules:

    DEC-FALSE  —  if db = F, decide FALSE  (short-circuit refutation)
    DEC-PROP   —  if da = T and db ∈ {T, U}, decide db
    DEC-U      —  if da ≠ T and db ≠ F, decide UNKNOWN
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import weave  # type: ignore[import-not-found]

from wonda.core.property import Property
from wonda.core.ast_program import AstProgram
from wonda.verifiers.uautomizer import UAutomizerVerifier, VerifierCallReport

logger = logging.getLogger(__name__)

@dataclass
class DecisionProcedureReport:
    final_decision: str = "UNKNOWN"
    decision_rule: str = ""
    target_assert: Optional[Property] = None
    candidate_invariant: Optional[Property] = None
    correctness_report: Optional[VerifierCallReport] = None
    usefulness_report: Optional[VerifierCallReport] = None
    verification_time: float = 0.0  # Only verification time (without model latency)
    report_path: str = ""
    program_for_correctness: Optional[str] = None
    program_for_usefulness: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert the report to a dictionary for JSON serialization."""
        # Convert target_assert and candidate_invariant to dictionaries if they exist
        target_assert_dict = {
            "content": self.target_assert.content if self.target_assert else None,
            "marker_name": self.target_assert.marker_name if self.target_assert else None,
        } if self.target_assert else None
        candidate_invariant_dict = {
            "content": self.candidate_invariant.content if self.candidate_invariant else None,
            "marker_name": self.candidate_invariant.marker_name if self.candidate_invariant else None,
        } if self.candidate_invariant else None
        return {
            "final_decision": self.final_decision,
            "decision_rule": self.decision_rule,
            "target_assert": target_assert_dict,
            "candidate_invariant": candidate_invariant_dict,
            "correctness_report": self.correctness_report.to_dict() if self.correctness_report else None,
            "usefulness_report": self.usefulness_report.to_dict() if self.usefulness_report else None,
            "program_for_correctness": self.program_for_correctness,
            "program_for_usefulness": self.program_for_usefulness,
            "verification_time": self.verification_time,
            "report_path": self.report_path,
        }

    def save_json(self, report_path: str):
        with open(report_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

class DecisionProcedure:
    def __init__(
        self,
        verifier: UAutomizerVerifier,
        program: AstProgram,
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
        self.target_assert = (
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

    @weave.op()
    def decide(
        self, candidate_invariant: Property
    ) -> DecisionProcedureReport:

        program_for_correctness = self.program.get_program_with_assertion(
            property_to_assert=candidate_invariant, assumptions=[], for_llm=False
        )

        program_for_usefulness = self.program.get_program_with_assertion(
            property_to_assert=self.target_assert, assumptions=[candidate_invariant], for_llm=False
        )

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
                program_str=program_for_usefulness,
                kind="usefulness",
            )

            # Track results as they complete
            invariant_correctness_report = None
            invariant_usefulness_report = None

            # Use as_completed to process results as they arrive.
            #
            # Note: even if we can *decide* FALSE based only on usefulness (DEC-FALSE),
            # ThreadPoolExecutor's context manager will still wait for running tasks on exit.
            # So we always attempt to collect both reports to keep timing/metrics consistent.
            for future in as_completed([correctness_future, usefulness_future]):
                try:
                    result = future.result()
                except Exception as e:
                    result = VerifierCallReport(
                        decision="ERROR",
                        decision_reason=f"future_exception: {type(e).__name__}: {e}",
                        time_taken=0.0,
                        reports_dir=str(self.reports_dir),
                    )

                if future == correctness_future:
                    invariant_correctness_report = result
                elif future == usefulness_future:
                    invariant_usefulness_report = result

            # After loop (and executor shutdown): ensure both reports are captured.
            # Because we used the executor as a context manager, both futures should now be done.
            if invariant_correctness_report is None:
                try:
                    invariant_correctness_report = correctness_future.result()
                except Exception as e:
                    invariant_correctness_report = VerifierCallReport(
                        decision="ERROR",
                        decision_reason=f"correctness_exception: {type(e).__name__}: {e}",
                        time_taken=0.0,
                        reports_dir=str(self.reports_dir),
                    )

            if invariant_usefulness_report is None:
                try:
                    invariant_usefulness_report = usefulness_future.result()
                except Exception as e:
                    invariant_usefulness_report = VerifierCallReport(
                        decision="ERROR",
                        decision_reason=f"usefulness_exception: {type(e).__name__}: {e}",
                        time_taken=0.0,
                        reports_dir=str(self.reports_dir),
                    )

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
            and invariant_usefulness_report
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
            and invariant_usefulness_report
            and invariant_usefulness_report.decision != "FALSE"
        ):
            final_decision = "UNKNOWN"
            decision_rule = "DEC-U"

        # Calculate verification time: max of both runs since they execute in parallel
        # Use 0 if correctness was cancelled (short-circuited) or time_taken is None
        correctness_time = (
            invariant_correctness_report.time_taken
            if invariant_correctness_report and invariant_correctness_report.time_taken is not None
            else 0.0
        )
        usefulness_time = (
            invariant_usefulness_report.time_taken
            if invariant_usefulness_report and invariant_usefulness_report.time_taken is not None
            else 0.0
        )
        verification_time = max(correctness_time, usefulness_time)

        report = DecisionProcedureReport(
            final_decision=final_decision,
            decision_rule=decision_rule,
            target_assert=self.target_assert,
            candidate_invariant=candidate_invariant,
            correctness_report=invariant_correctness_report,
            usefulness_report=invariant_usefulness_report,
            program_for_correctness=program_for_correctness,
            program_for_usefulness=program_for_usefulness,
            verification_time=verification_time,
        )

        logger.info(f"runlim correctness output: {invariant_correctness_report.runlim}")
        logger.info(f"runlim usefulness output: {invariant_usefulness_report.runlim}")
        return report