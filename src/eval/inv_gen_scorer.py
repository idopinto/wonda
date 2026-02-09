"""
Scorer module for evaluation.

Contains the InvariantScorer class for evaluating model-generated invariants.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from src.eval.validate import validate_model_answer
import weave  # type: ignore[import-not-found]

from src.eval.decision_procedure import DecisionProcedure
from src.preprocess.program import Program
from src.verifiers.uautomizer_runlim import UAutomizerVerifier


class ResultCollector:
    """
    Collects evaluation results for plotting.

    This is a cleaner alternative to global mutable state - the collector
    is passed to the scorer and accumulates results.
    """

    def __init__(self):
        self.results: List[dict] = []

    def collect(self, result: dict) -> None:
        """Add a result to the collection."""
        self.results.append(result)

    def clear(self) -> None:
        """Clear all collected results."""
        self.results.clear()

    def get_results(self) -> List[dict]:
        """Get a copy of all collected results."""
        return self.results.copy()


class InvGenScorer(weave.Scorer):
    """
    Weave Scorer for evaluating model-generated loop invariants.

    Evaluates generated invariants for:
    1. validation (is the invariant syntactically valid?)
    2. correctness (is the invariant actually an invariant?)
    3. usefulness (when assuming the invariant is true, does the target property still hold?)
    4. vbs (virtual best solver): min(max(t_c, t_u), t_b))
        - The Virtual Best Solver (VBS) is an oracle that, for each instance in a benchmark, always selects the algorithm that performs best on that particular instance.
        - t_c: Time taken to verify the correctness of the invariant
        - t_u: Time taken to verify the target property assuming the invariant is true
        - t_b: Baseline verification time (median timing across k=3 runs, statically computed)
    5. vbs_e2e: min((max(t_c, t_u) + t_m), t_b)
        - t_m: Model latency (time taken to generate the invariant)
    6. speedup: t_b / max(t_c, t_u)
    7. speedup_e2e: t_b / (max(t_c, t_u )+ t_m)
    Attributes:
        config: Evaluation configuration.
    """

    verifier: Optional[UAutomizerVerifier] = None
    baseline_is_timeout: bool

    @weave.op()
    def score(self, output: Dict, original_program: str, median_timing: float,  target_marker: Optional[str] = None) -> Dict:
        """
        Score a model-generated invariant.

        Args:
            output: Model output containing 'reasoning', 'answer', and 'model_latency'.
            original_program: The C program source code from the dataset.
            median_timing: Baseline verification time from the dataset.
            target_marker: The target marker for the invariant. (INVARIANT_MARKER_k, Optional)
        Returns:
            Dict containing all evaluation metrics and artifacts.
        """
        # Create Program object from raw code
        program = Program().from_code(original_program)
        program.process(print_ast=False)

        raw_model_answer = output.get("answer", "")
        model_latency = output.get("model_latency", 0.0)
        parsed_model_answer, is_valid_dict = validate_model_answer(raw_model_answer, target_marker)
        validation_score = is_valid_dict["valid_json_format"] and is_valid_dict["valid_content"] and is_valid_dict["valid_marker"]
        
        # Initialize default values for when validation fails
        correctness_score = False
        usefulness_score = False
        vbs = median_timing
        vbs_e2e = median_timing
        speedup = 0.0
        speedup_e2e = 0.0
        verification_time = 0.0
        verification_time_e2e = model_latency
        decision_procedure_report = None
        
        if validation_score:
            timeout = median_timing if self.baseline_is_timeout else self.verifier.timeout_seconds
            with tempfile.TemporaryDirectory() as temp_dir:
                code_dir = Path(temp_dir)
                decision_procedure = DecisionProcedure(
                    verifier=self.verifier,
                    program=program,
                    code_dir=code_dir,
                    timeout=timeout,
                )
                decision_procedure_report = decision_procedure.decide(candidate_invariant=parsed_model_answer)
            
            verification_time = decision_procedure_report.verification_time # t_v
            verification_time_e2e = verification_time + model_latency # t_v + t_m
            correctness_score = decision_procedure_report.correctness_report and decision_procedure_report.correctness_report.decision == "TRUE"
            usefulness_score = correctness_score and decision_procedure_report.usefulness_report and decision_procedure_report.final_decision in {"TRUE", "FALSE"}
            # Calculate vbs and speedup (only meaningful if we have a correct invariant and conclusive final decision)
            vbs = min(verification_time, median_timing) if usefulness_score else median_timing
            vbs_e2e = min(verification_time_e2e, median_timing) if usefulness_score else median_timing
            speedup = median_timing / verification_time if verification_time > 0 and usefulness_score else 0.0
            speedup_e2e = median_timing / verification_time_e2e if verification_time_e2e > 0 and usefulness_score else 0.0
        
        result = {
            "validation_score": validation_score,
            "json_valid": is_valid_dict["valid_json_format"],
            "content_valid": is_valid_dict["valid_content"],
            "marker_valid": is_valid_dict["valid_marker"],
            "correctness_score": correctness_score,
            "usefulness_score": usefulness_score,
            "has_speedup": bool(speedup > 1),
            "has_speedup_e2e": bool(speedup_e2e > 1),
            "speedup": speedup,
            "speedup_e2e": speedup_e2e,
            "vbs": vbs,
            "vbs_e2e": vbs_e2e,
            "verification_time": verification_time,
            "verification_time_e2e": verification_time_e2e,
            "median_timing": median_timing,
        }

        return result

    @weave.op()
    def summarize(self, score_rows: List[Dict]) -> Dict:
        """
        Aggregate meaningful numeric scores across all examples.

        Only summarizes core metrics, not metadata like code snippets or reports.

        Args:
            score_rows: List of score dictionaries from each example.

        Returns:
            Dict containing aggregated metrics.
        """
        valid_rows = [r for r in score_rows if r is not None]
        if not valid_rows:
            return {}

        n = len(valid_rows)

        # Calculate counts and rates for boolean scores
        validation_count = sum(1 for r in valid_rows if r.get("validation_score", False))
        correctness_count = sum(1 for r in valid_rows if r.get("correctness_score", False))
        usefulness_count = sum(1 for r in valid_rows if r.get("usefulness_score", False))
        speedup_count = sum(1 for r in valid_rows if r.get("has_speedup", False))
        speedup_e2e_count = sum(1 for r in valid_rows if r.get("has_speedup_e2e", False))
        
        validation_rate = validation_count / n
        correctness_rate = correctness_count / n
        usefulness_rate = usefulness_count / n
        speedup_rate = speedup_count / n
        speedup_e2e_rate = speedup_e2e_count / n

        # Calculate Speedup>1: average speedup only from examples with speedup > 1, correct invariants with conclusive final decision
        speedups_gt1 = [r.get("speedup", 0.0) for r in valid_rows if r.get("has_speedup", False)]
        speedups_gt1_e2e = [r.get("speedup_e2e", 0.0) for r in valid_rows if r.get("has_speedup_e2e", False)]

        speedup_gt1 = sum(speedups_gt1) / len(speedups_gt1) if speedups_gt1 else 1.0
        speedup_gt1_e2e = sum(speedups_gt1_e2e) / len(speedups_gt1_e2e) if speedups_gt1_e2e else 1.0


        # Calculate Speedup_all: average speedup from all, but with speedup set to 1
        # for non-qualifying examples (matches _calculate_speedup_metrics)
        def get_speedup_or_one(r: Dict, e2e: bool) -> float:
            """Return speedup if qualifying, else 1.0."""
            if e2e:
                return (
                    r.get("speedup_e2e", 0)
                    if r.get("has_speedup_e2e", False) and r.get("speedup_e2e", 0) > 1
                    else 1.0
                )
            else:
                return (
                    r.get("speedup", 0)
                    if r.get("has_speedup", False)
                    and r.get("speedup", 0) > 1
                    else 1.0
                )

        speedup_all = sum(get_speedup_or_one(r, False) for r in valid_rows) / n
        speedup_all_e2e = sum(get_speedup_or_one(r, True) for r in valid_rows) / n

        # Virtual best performance (average of virtual best solver)
        # Use .get() with defaults in case some rows are missing keys due to errors
        vbp = sum(r.get("vbs", r.get("median_timing", 0.0)) for r in valid_rows) / n
        vbp_e2e = sum(r.get("vbs_e2e", r.get("median_timing", 0.0)) for r in valid_rows) / n
        avg_median_timing = sum(r.get("median_timing", 0.0) for r in valid_rows) / n
        return {
            # Counts
            "validation_count": validation_count,
            "correctness_count": correctness_count,
            "usefulness_count": usefulness_count,
            "speedup_count": speedup_count,
            "speedup_e2e_count": speedup_e2e_count,
                    
            # Rates
            "validation_rate": validation_rate,
            "correctness_rate": correctness_rate,
            "usefulness_rate": usefulness_rate,
            "speedup_rate": speedup_rate,
            "speedup_e2e_rate": speedup_e2e_rate,

            # Virtual best performance vs baseline timing
            "vbp": vbp,
            "vbp_e2e": vbp_e2e,
            "avg_median_timing": avg_median_timing,

            # Speedup metrics
            "speedup_gt1": speedup_gt1,
            "speedup_gt1_e2e": speedup_gt1_e2e,
            "speedup_all": speedup_all,
            "speedup_all_e2e": speedup_all_e2e,
        }
