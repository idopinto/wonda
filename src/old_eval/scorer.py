"""
Scorer module for evaluation.

Contains the InvariantScorer class for evaluating model-generated invariants.
"""
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import weave

import re

from src.eval.model import LOCATION_LABELS
from src.eval.decision_procedure import DecisionProcedure
from src.utils.predicate import Predicate
from src.utils.program import Program
from src.utils.program_normalizer import ProgramNormalizer
from src.verifiers.uautomizer import UAutomizerVerifier

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


class InvariantScorer(weave.Scorer):
    """
    Weave Scorer for evaluating model-generated loop invariants.
    
    Evaluates generated invariants for:
    1. Syntactic validity
    2. Correctness (is the invariant actually an invariant?)
    3. Speedup/usefulness (is verification faster with the invariant?)
    
    Attributes:
        config: Evaluation configuration.
        collector: Optional ResultCollector to accumulate results for plotting.
    """
    verifier: Optional[UAutomizerVerifier] = None
    collector: Optional[ResultCollector] = None
    baseline_is_timeout: bool

    # def model_post_init(self, __context):
    #     """Initialize verifier after Pydantic model creation."""
    #     object.__setattr__(self, 'baseline_is_timeout', self.baseline_is_timeout)

    def _parse_predicate(self, answer: str) -> tuple[str, str]:
        """
        Parse model answer to extract predicate content and line label.
        
        Args:
            answer: The model's final answer string.
            
        Returns:
            Tuple of (predicate_content, line_label).
        """
        if not answer:
            return "", ""
            
        assert_pos = answer.find("assert(")
        if assert_pos == -1:
            return "", ""
        
        # Track parentheses depth to find matching closing paren
        start_pos = assert_pos + 7  # after "assert("
        paren_count = 0
        end_pos = None
        
        for i in range(start_pos, len(answer)):
            if answer[i] == "(":
                paren_count += 1
            elif answer[i] == ")":
                if paren_count == 0:
                    end_pos = i
                    break
                paren_count -= 1
        
        if end_pos is None:
            return "", ""
        
        invariant = answer[start_pos:end_pos].strip()
        rest = answer[end_pos:]
        
        # Try multiple comment patterns (case-insensitive)
        patterns = [
            r"\);\s*//\s*[Ll]ine\s+([A-Za-z])",  # ); // Line A
            r"\);\s*//\s*([A-Za-z])",              # ); // A
            r"\);\s*/\*\s*[Ll]ine\s+([A-Za-z])\s*\*/",  # ); /* Line A */
        ]
        
        for pattern in patterns:
            match = re.search(pattern, rest)
            if match:
                return invariant, match.group(1).upper()
        
        return "", ""
    
    def _get_line_number_from_label(self, program: Program, line_label: str) -> int:
        """
        Map a line label (A, B, C...) to actual line number.
        
        Args:
            program: The program being analyzed.
            line_label: The label from model output (e.g., "A").
            
        Returns:
            The actual line number, or 0 if not found.
        """
        assertion_points = program.assertion_points
        for i, line_num in enumerate(sorted(assertion_points.keys())):
            if i < len(LOCATION_LABELS) and LOCATION_LABELS[i] == line_label:
                return line_num
        return 0
    
    @weave.op()
    def score(self, output: Dict, original_program: str, median_timing: float) -> Dict:
        """
        Score a model-generated invariant.
        
        Args:
            output: Model output containing 'reasoning', 'answer', and 'model_latency'.
            original_program: The C program source code from the dataset.
            median_timing: Baseline verification time from the dataset (may be Weave ref).
                   
        Returns:
            Dict containing all evaluation metrics and artifacts.
        """
        # Create Program object from raw code
        normalizer = ProgramNormalizer(code=original_program)
        program = Program(normalizer.lines_to_verify, normalizer.replacement)
        
        # Resolve baseline timing to float
        baseline_timing = median_timing
        
        # Parse the model's answer to extract predicate
        answer = output.get("answer", "")
        predicate_content, line_label = self._parse_predicate(answer)
        line_number = self._get_line_number_from_label(program, line_label)
        
        predicate = Predicate(
            content=predicate_content,
            line_number=line_number
        )
        
        # Determine timeout based on configuration
        timeout = baseline_timing if self.baseline_is_timeout else self.verifier.timeout_seconds

        with tempfile.TemporaryDirectory() as temp_dir:
            code_dir = Path(temp_dir)
            decision_procedure = DecisionProcedure(
                verifier=self.verifier,
                program=program,
                code_dir=code_dir,
                timeout=timeout
            )
            decision_procedure_report = decision_procedure.run(
                candidate=predicate,
                model_latency=output.get("model_latency", 0.0)
            )

        correctness_score = (
            decision_procedure_report.correctness_report is not None 
            and decision_procedure_report.correctness_report.decision == "TRUE"
        )
        speedup_no_gen = 0.0
        speedup_gen = 0.0
        # Calculate speedup (only meaningful if we have a conclusive result)
        if correctness_score and decision_procedure_report.final_decision in ["TRUE", "FALSE"]:
            speedup_no_gen = baseline_timing / decision_procedure_report.verification_time if decision_procedure_report.verification_time > 0 else 0.0
            speedup_gen = baseline_timing / decision_procedure_report.total_time if decision_procedure_report.total_time > 0 else 0.0

        
        result = {
            # Core scores (will be summarized)
            "validation_score": decision_procedure_report.is_valid,
            "correctness_score": correctness_score,
            "has_speedup_no_gen": bool(speedup_no_gen > 1),
            "has_speedup_gen": bool(speedup_gen > 1),
            "speedup_no_gen": speedup_no_gen,
            "speedup_gen": speedup_gen,
            "final_decision": decision_procedure_report.final_decision,
            "decision_rule": decision_procedure_report.decision_rule,
            "verification_time": decision_procedure_report.verification_time,
            "total_time": decision_procedure_report.total_time,
            # Include parsed fields for plotting
            "predicate_content": predicate_content,
            "baseline_timing": baseline_timing,
            # Code for click-to-view in plot
            "program_for_usefulness": getattr(decision_procedure_report, 'program_for_usefulness', ''),
            "usefulness_report": decision_procedure_report.usefulness_report.to_dict() if decision_procedure_report.usefulness_report else None,
            "program_for_correctness": getattr(decision_procedure_report, 'program_for_correctness', ''),
            "correctness_report": decision_procedure_report.correctness_report.to_dict() if decision_procedure_report.correctness_report else None,
        }
        
        # Collect result for plotting if collector is provided
        if self.collector is not None:
            self.collector.collect(result)
        
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
        speedup_count_no_gen = sum(1 for r in valid_rows if r.get("has_speedup_no_gen", False))
        speedup_count_gen = sum(1 for r in valid_rows if r.get("has_speedup_gen", False))
        
        validation_rate = validation_count / n
        correctness_rate = correctness_count / n
        speedup_rate_no_gen = speedup_count_no_gen / n
        speedup_rate_gen = speedup_count_gen / n
        
        # Calculate Speedup>1: average speedup only from examples with speedup > 1, correct invariants with conclusive final decision
        speedups_gt1_no_gen = [r["speedup_no_gen"] for r in valid_rows if r.get("has_speedup_no_gen", False)]
        speedups_gt1_gen = [r["speedup_gen"] for r in valid_rows if r.get("has_speedup_gen", False)]
        
        speedup_gt1_no_gen = sum(speedups_gt1_no_gen) / len(speedups_gt1_no_gen) if speedups_gt1_no_gen else 1.0
        speedup_gt1_gen = sum(speedups_gt1_gen) / len(speedups_gt1_gen) if speedups_gt1_gen else 1.0
        
        # Calculate Speedup_all: average speedup from all, but with speedup set to 1
        # for non-qualifying examples (matches _calculate_speedup_metrics)
        def get_speedup_or_one(r: Dict, with_gen: bool) -> float:
            """Return speedup if qualifying, else 1.0."""
            if with_gen:
                return r.get("speedup_gen", 0) if r.get("has_speedup_gen", False) and r.get("speedup_gen", 0) > 1 else 1.0
            else:
                return r.get("speedup_no_gen", 0) if r.get("has_speedup_no_gen", False) and r.get("speedup_no_gen", 0) > 1 else 1.0

        speedup_all_no_gen = sum(get_speedup_or_one(r, False) for r in valid_rows) / n
        speedup_all_gen = sum(get_speedup_or_one(r, True) for r in valid_rows) / n
        
        return {
            # Counts
            "validation_count": validation_count,
            "correctness_count": correctness_count,
            "speedup_count_no_gen": speedup_count_no_gen,
            "speedup_count_gen": speedup_count_gen,
            # Rates
            "validation_rate": validation_rate,
            "correctness_rate": correctness_rate,
            "speedup_rate_no_gen": speedup_rate_no_gen,
            "speedup_rate_gen": speedup_rate_gen,
            # Speedup metrics
            "speedup_gt1_no_gen": speedup_gt1_no_gen,
            "speedup_gt1_gen": speedup_gt1_gen,
            "speedup_all_no_gen": speedup_all_no_gen,
            "speedup_all_gen": speedup_all_gen,
            # Total
            "total_examples": n,
        }
