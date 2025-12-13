from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.plain_verifier import run_uautomizer, VerifierCallReport
from src.utils.program import Program
from src.utils.predicate import Predicate
from src.eval.validate import syntactic_validation
import weave
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class DecisionProcedureReport:
    final_decision: str = "UNKNOWN"
    decision_rule: str = ""
    program: Optional[Program] = None
    target_assert: Optional[Predicate] = None
    target_property_file_path: Optional[Path] = None
    candidate_invariant: Optional[Predicate] = None
    syntactic_validation_result: bool = False
    invariant_correctness_report: Optional[VerifierCallReport] = None
    invariant_usefulness_report: Optional[VerifierCallReport] = None
    total_time_taken: float = 0.0  # Includes model  generation time
    verification_time_taken: float = 0.0  # Only verification time (without model generation)
    model_generation_time: float = 0.0  # Model inference/token generation time
    report_file_path: str = ""

    def to_dict(self) -> dict:
        """Convert the report to a dictionary for JSON serialization."""
        return {
            'final_decision': self.final_decision,
            'decision_rule': self.decision_rule,
            'target_assert': {
                'content': self.target_assert.content,
                'line_number': self.target_assert.line_number
            } if self.target_assert else None,
            'target_property_file_path': str(self.target_property_file_path) if self.target_property_file_path else None,
            'candidate_invariant': {
                'content': self.candidate_invariant.content,
                'line_number': self.candidate_invariant.line_number
            } if self.candidate_invariant else None,
            'syntactic_validation_result': self.syntactic_validation_result,
            'invariant_correctness_report': self.invariant_correctness_report.to_dict() if self.invariant_correctness_report else None,
            'invariant_usefulness_report': self.invariant_usefulness_report.to_dict() if self.invariant_usefulness_report else None,
            'total_time_taken': self.total_time_taken,
            'verification_time_taken': self.verification_time_taken,
            'model_generation_time': self.model_generation_time,
            'report_file_path': self.report_file_path
        }
    def save_json(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class DecisionProcedure:
    def __init__(self, program: Program, target_property_file_path: Path, arch: str, code_dir: Path, uautomizer_path: Path, timeout_seconds: float = 600.0):
        self.program = program
        self.target_property_file_path = target_property_file_path # "unreach-call.prp"
        if self.program.assertions and len(self.program.assertions) > 0:
            self.target_assert = program.assertions[0]  # TODO: Assuming first assert is the target
        else:
            self.target_assert = None   
        self.code_dir = code_dir
        self.arch = arch
        self.reports_dir = Path(code_dir).parent / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.uautomizer_path = uautomizer_path

    def run_verifier(self, program_str: str, kind: str):
        program_path = self.code_dir / f"code_for_{kind}.c"
        with open(program_path, 'w') as out_file:
            out_file.write(program_str)
        verifier_report: VerifierCallReport = run_uautomizer(
            program_path=program_path, 
            property_file_path=self.target_property_file_path,
            reports_dir=self.reports_dir,
            arch=self.arch,
            timeout_seconds=self.timeout_seconds,
            uautomizer_path=self.uautomizer_path
        )
        # print(f"Verifier report: {verifier_report}")
        return verifier_report
    
    def decide(self, candidate_invariant: Predicate, report: DecisionProcedureReport) -> DecisionProcedureReport:

        program_for_correctness = self.program.get_program_with_assertion(predicate=candidate_invariant, 
                                                     assumptions=[],
                                                     assertion_points={},
                                                     forGPT=False,
                                                     dump=False)

        program_for_usefullness = self.program.get_program_with_assertion(predicate=self.target_assert,
                                                                          assumptions=[candidate_invariant],
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
        verification_time_taken = max(correctness_time, usefulness_time)
        
        # Update the report with decision results (keep all other fields from the initial report)
        report.final_decision = final_decision
        report.decision_rule = decision_rule
        report.invariant_correctness_report = invariant_correctness_report
        report.invariant_usefulness_report = invariant_usefulness_report
        report.verification_time_taken = verification_time_taken
        report.total_time_taken = verification_time_taken + report.model_generation_time
        
        return report
    @weave.op()
    def run(self, candidate_invariant: Predicate, model_gen_time: float) -> DecisionProcedureReport:
        is_valid = syntactic_validation(candidate_invariant.content)
        final_report = DecisionProcedureReport(program=self.program,
                                               target_assert=self.target_assert, 
                                               target_property_file_path=self.target_property_file_path,
                                               candidate_invariant=candidate_invariant,
                                               syntactic_validation_result=is_valid,
                                               model_generation_time=model_gen_time)
        # is_logicaly_equivalent = check_semantic_equivalence(candidate_invariant.content, self.target_assert.content)
        # print(f"The candidate invariant is valid: {is_valid}")
        # print(f"The candidate invariant is logically equivalent to the target assert: {is_logicaly_equivalent}")
        if is_valid: # and not is_logicaly_equivalent:
           final_report = self.decide(candidate_invariant, final_report)
        
        # save the final report to a json file
        report_file_path = self.reports_dir / "decision_report.json"
        final_report.save_json(report_file_path)
        # print(f"Decision report saved to:\n\t {str(report_file_path)}")
        return final_report