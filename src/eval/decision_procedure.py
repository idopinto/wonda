import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import weave

from src.eval.validate import syntactic_validation
from src.verifiers.uautomizer import UAutomizerVerifier, VerifierCallReport
from src.utils.predicate import Predicate
from src.utils.program import Program

@dataclass
class DecisionProcedureReport:
    final_decision: str = "UNKNOWN"
    decision_rule: str = ""
    # program: Optional[Program] = None
    target: Optional[Predicate] = None
    # target_property_file_path: Optional[Path] = None
    candidate: Optional[Predicate] = None
    is_valid: bool = False
    correctness_report: Optional[VerifierCallReport] = None
    usefulness_report: Optional[VerifierCallReport] = None
    verification_time: float = 0.0  # Only verification time (without model latency)
    model_latency: float = 0.0  # Model inference/token generation time
    total_time: float = 0.0  # verification_time + model_latency
    report_path: str = ""

    def to_dict(self) -> dict:
        """Convert the report to a dictionary for JSON serialization."""
        return {
            'final_decision': self.final_decision,
            'decision_rule': self.decision_rule,
            'target': {
                'content': self.target.content,
                'line_number': self.target.line_number
            } if self.target else None,
            'candidate': {
                'content': self.candidate.content,
                'line_number': self.candidate.line_number
            } if self.candidate else None,
            'is_valid': self.is_valid,
            'correctness_report': self.correctness_report.to_dict() if self.correctness_report else None,
            'usefulness_report': self.usefulness_report.to_dict() if self.usefulness_report else None,
            'verification_time': self.verification_time,
            'model_latency': self.model_latency,
            'total_time': self.total_time,
            'report_path': self.report_path
        }
    def save_json(self, report_path: str):
        with open(report_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def from_json(cls, report_path: str):
        with open(report_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class DecisionProcedure:
    def __init__(self, verifier: UAutomizerVerifier, program: Program, code_dir: Path):
        self.verifier = verifier
        self.program = program
        self.code_dir = code_dir

    def run_verifier(self, program_str: str, kind: str) -> VerifierCallReport:
        program_path = self.code_dir / f"code_for_{kind}.c"
        with open(program_path, 'w') as out_file:
            out_file.write(program_str)
        verifier_report: VerifierCallReport = self.verifier.verify(
            program_path=program_path,
            reports_dir=self.reports_dir
        )
        return verifier_report

    def decide(self, candidate: Predicate, report: DecisionProcedureReport) -> DecisionProcedureReport:
        program_for_correctness = self.program.get_program_with_assertion(predicate=candidate, 
                                                                          assumptions=[],
                                                                          assertion_points={},
                                                                          forGPT=False,
                                                                          dump=False)

        program_for_usefullness = self.program.get_program_with_assertion(predicate=self.target,
                                                                          assumptions=[candidate],
                                                                          assertion_points={},
                                                                          forGPT=False,
                                                                          dump=False)
        # Parallel evaluation: run both verifier queries concurrently
        # da = V(P, Ø, q): Check if q is an invariant
        # db = V(P, {q}, p*): Check if target property holds assuming q is true
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both verifier tasks
            correctness_future = executor.submit(
                self.run_verifier,
                program_str=program_for_correctness,
                kind="correctness"
            )
            
            usefulness_future = executor.submit(
                self.run_verifier,
                program_str=program_for_usefullness,
                kind="usefulness"
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
        if invariant_usefulness_report and invariant_usefulness_report.decision == "FALSE":
            final_decision = "FALSE"
            decision_rule = "DEC-FALSE"
        
        # DEC_PROP: If da = T and db ∈ {T, U}, decide db
        # Rule DEC-PROP implements the prove then-use strategy: 
        # once the candidate invariant q is established, the outcome is exactly the verifier answer on the goal under the assumption q
        elif (invariant_correctness_report and invariant_correctness_report.decision == "TRUE" and 
              invariant_usefulness_report.decision in {"TRUE", "UNKNOWN", "TIMEOUT", "ERROR"}):
              if invariant_usefulness_report.decision == "TRUE":
                final_decision = "TRUE"
              else:
                final_decision = "UNKNOWN" # TIMEOUTS AND ERRORS are considered as UNKNOWN
              # final_decision = invariant_usefulness_report.decision
              decision_rule = "DEC-PROP"
        
        # DEC-U: If da ≠ T and db ≠ F, decide U
        # Rule DEC-U gives explicit conditions for inconclusiveness:
        # the goal is not refuted under q and q is not established as an invariant
        elif (invariant_correctness_report and invariant_correctness_report.decision != "TRUE" and 
              invariant_usefulness_report.decision != "FALSE"):
            final_decision = "UNKNOWN"
            decision_rule = "DEC-U"
        
        # Calculate verification time: max of both runs since they execute in parallel
        # Use 0 if correctness was cancelled (short-circuited)
        correctness_time = invariant_correctness_report.time_taken if invariant_correctness_report else 0.0
        usefulness_time = invariant_usefulness_report.time_taken if invariant_usefulness_report else 0.0
        verification_time = max(correctness_time, usefulness_time)
        
        # Update the report with decision results (keep all other fields from the initial report)
        report.final_decision = final_decision
        report.decision_rule = decision_rule
        report.correctness_report = invariant_correctness_report
        report.usefulness_report = invariant_usefulness_report
        report.verification_time = verification_time
        report.total_time = verification_time + report.model_latency # assumes model_latency is set as input
        
        return report

    @weave.op()
    def run(self, candidate: Predicate, model_latency: float) -> DecisionProcedureReport:
        is_valid = syntactic_validation(candidate.content)
        final_report = DecisionProcedureReport(program=self.program,
                                               target=self.target, 
                                            #    target_property_file_path=self.target_property_file_path,
                                               candidate=candidate,
                                               is_valid=is_valid,
                                               model_latency=model_latency)
        if is_valid:
           final_report = self.decide(candidate, final_report)
           final_report.final_decision = "INVALID"
        
        # save the final report to a json file
        report_file_path = self.reports_dir / "decision_report.json"
        final_report.save_json(report_file_path)
        return final_report