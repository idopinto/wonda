"""
Scorer module for evaluation.

Contains the InvariantScorer class for evaluating model-generated invariants.
"""
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import weave

import re

from src.eval.config import LOCATION_LABELS, EvalConfig
from src.eval.decision_procedure import DecisionProcedure
from src.utils.paths import PROPERTIES_DIR, UAUTOMIZER_PATHS
from src.utils.predicate import Predicate
from src.utils.program import Program
from src.utils.rewriter import Rewriter


target_property_file_path = PROPERTIES_DIR / "unreach-call.prp"


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
    config: EvalConfig
    collector: Optional[ResultCollector] = None
    
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
    
    def _create_program_from_code(self, code: str) -> Program:
        """
        Create a Program object from raw C code string.
        
        Args:
            code: The C program source code as a string.
            
        Returns:
            A Program object ready for verification.
        """
        rewriter = Rewriter.from_string(code, rewrite=True)
        return Program(rewriter.lines_to_verify, rewriter.replacement)
    
    # def _resolve_timing(self, value) -> float:
    #     """
    #     Resolve timing value to float, handling Weave references.
        
    #     Args:
    #         value: The timing value (could be float, int, string, or Weave ref).
            
    #     Returns:
    #         The timing value as a float.
    #     """
    #     # If it's already a numeric type, convert directly
    #     if isinstance(value, (int, float)):
    #         return float(value)
        
    #     # If it's a Weave ObjectRef object, try to resolve it
    #     if hasattr(value, 'get'):
    #         resolved = value.get()
    #         return float(resolved)
        
    #     # If it's a Weave reference string, resolve it using weave.ref()
    #     if isinstance(value, str) and value.startswith("weave://"):
    #         resolved = weave.ref(value).get()
    #         return float(resolved)
        
    #     # Try to convert string to float
    #     if isinstance(value, str):
    #         return float(value)
        
    #     raise ValueError(f"Cannot convert timing to float: {type(value)} = {value}")
    
    @weave.op()
    def score(self, output: Dict, original_program: str, median_timing) -> Dict:
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
        program = self._create_program_from_code(original_program)
        
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
        uautomizer_path = UAUTOMIZER_PATHS[self.config.uautomizer_version]
        model_latency = output.get("model_latency", 0.0)
        
        # Determine timeout based on configuration
        timeout = baseline_timing if self.config.baseline_is_timeout else self.config.timeout
        
        # Create temp directory for code files
        with tempfile.TemporaryDirectory() as temp_dir:
            code_dir = Path(temp_dir)
            decision_procedure = DecisionProcedure(
                program=program,
                target_property_file_path=target_property_file_path,
                arch=self.config.arch,
                code_dir=code_dir,
                uautomizer_path=uautomizer_path,
                timeout_seconds=timeout
            )
            decision_procedure_report = decision_procedure.run(
                predicate,
                model_gen_time=model_latency
            )
            
            # Read generated code files for Weave logging
            code_for_correctness = ""
            code_for_usefulness = ""
            correctness_path = code_dir / "code_for_correctness.c"
            usefulness_path = code_dir / "code_for_usefulness.c"
            
            if correctness_path.exists():
                code_for_correctness = correctness_path.read_text()
            if usefulness_path.exists():
                code_for_usefulness = usefulness_path.read_text()
        
        # Safe access to reports (could be None if short-circuited)
        correctness_report = decision_procedure_report.invariant_correctness_report
        usefulness_report = decision_procedure_report.invariant_usefulness_report
        
        invariant_correctness_score = (
            correctness_report.decision == "TRUE" if correctness_report else False
        )
        
        # Calculate speedup (only meaningful if we have a conclusive result)
        speedup_without_gen = 0.0
        speedup_with_gen = 0.0
        
        if (invariant_correctness_score and 
            decision_procedure_report.final_decision != "UNKNOWN"):
            if decision_procedure_report.verification_time_taken > 0:
                speedup_without_gen = (
                    baseline_timing / decision_procedure_report.verification_time_taken
                )
            if decision_procedure_report.total_time_taken > 0:
                speedup_with_gen = (
                    baseline_timing / decision_procedure_report.total_time_taken
                )
        
        result = {
            # Core scores (will be summarized)
            "syntactic_validation_score": decision_procedure_report.syntactic_validation_result,
            "invariant_correctness_score": invariant_correctness_score,
            "has_speedup_without_gen": speedup_without_gen > 1,
            "has_speedup_with_gen": speedup_with_gen > 1,
            "speedup_without_gen": speedup_without_gen,
            "speedup_with_gen": speedup_with_gen,
            "final_decision": decision_procedure_report.final_decision,
            "decision_rule": decision_procedure_report.decision_rule,
            # Metadata (logged but not summarized)
            "code_for_correctness": code_for_correctness,
            "invariant_correctness_report": (
                correctness_report.to_dict() if correctness_report else None
            ),
            "code_for_usefulness": code_for_usefulness,
            "invariant_usefulness_report": (
                usefulness_report.to_dict() if usefulness_report else None
            ),
            "verification_time": decision_procedure_report.verification_time_taken,
            "total_time": decision_procedure_report.total_time_taken,
            # Include parsed fields for plotting
            "predicate_content": predicate_content,
            "baseline_timing": baseline_timing,
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
        syntactic_validation_count = sum(
            1 for r in valid_rows if r.get("syntactic_validation_score", False)
        )
        invariant_correctness_count = sum(
            1 for r in valid_rows if r.get("invariant_correctness_score", False)
        )
        speedup_count_without_gen = sum(
            1 for r in valid_rows if r.get("has_speedup_without_gen", False)
        )
        speedup_count_with_gen = sum(
            1 for r in valid_rows if r.get("has_speedup_with_gen", False)
        )
        
        syntactic_validation_rate = syntactic_validation_count / n
        invariant_correctness_rate = invariant_correctness_count / n
        speedup_rate_without_gen = speedup_count_without_gen / n
        speedup_rate_with_gen = speedup_count_with_gen / n
        
        # Calculate Speedup>1: average speedup only from examples with speedup > 1,
        # correct invariant, and not UNKNOWN (matches _calculate_speedup_metrics)
        speedups_gt1_without_gen = [
            r["speedup_without_gen"] for r in valid_rows 
            if (r.get("invariant_correctness_score", False) 
                and r.get("final_decision") != "UNKNOWN"
                and r.get("speedup_without_gen", 0) > 1)
        ]
        speedups_gt1_with_gen = [
            r["speedup_with_gen"] for r in valid_rows 
            if (r.get("invariant_correctness_score", False) 
                and r.get("final_decision") != "UNKNOWN"
                and r.get("speedup_with_gen", 0) > 1)
        ]
        
        speedup_gt1_without_gen = (
            sum(speedups_gt1_without_gen) / len(speedups_gt1_without_gen) 
            if speedups_gt1_without_gen else 1.0
        )
        speedup_gt1_with_gen = (
            sum(speedups_gt1_with_gen) / len(speedups_gt1_with_gen) 
            if speedups_gt1_with_gen else 1.0
        )
        
        # Calculate Speedup_all: average speedup from all, but with speedup set to 1
        # for non-qualifying examples (matches _calculate_speedup_metrics)
        def get_speedup_or_one(r: Dict, speedup_key: str) -> float:
            """Return speedup if qualifying, else 1.0."""
            is_correct = r.get("invariant_correctness_score", False)
            is_not_unknown = r.get("final_decision") != "UNKNOWN"
            speedup = r.get(speedup_key, 0)
            if is_correct and is_not_unknown and speedup > 1:
                return speedup
            return 1.0
        
        speedup_all_without_gen = sum(
            get_speedup_or_one(r, "speedup_without_gen") for r in valid_rows
        ) / n
        speedup_all_with_gen = sum(
            get_speedup_or_one(r, "speedup_with_gen") for r in valid_rows
        ) / n
        
        return {
            # Counts
            "syntactic_validation_count": syntactic_validation_count,
            "invariant_correctness_count": invariant_correctness_count,
            "speedup_count_without_gen": speedup_count_without_gen,
            "speedup_count_with_gen": speedup_count_with_gen,
            # Rates
            "syntactic_validation_rate": syntactic_validation_rate,
            "invariant_correctness_rate": invariant_correctness_rate,
            "speedup_rate_without_gen": speedup_rate_without_gen,
            "speedup_rate_with_gen": speedup_rate_with_gen,
            # Speedup metrics
            "speedup_gt1_without_gen": speedup_gt1_without_gen,
            "speedup_gt1_with_gen": speedup_gt1_with_gen,
            "speedup_all_without_gen": speedup_all_without_gen,
            "speedup_all_with_gen": speedup_all_with_gen,
            # Total
            "total_examples": n,
        }
