"""
Per-location evaluation script for loop invariant generation using Weave.

This module evaluates invariant generation at each candidate location separately:
- Instead of letting the model choose the location, we iterate over all candidate locations
- For each location, we ask the model to generate an invariant specifically for that location
- Uses Weave's Evaluation framework for tracking and visualization

The metrics are the same as the main evaluation, but prompts are different.
"""
import asyncio
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import weave
from dotenv import load_dotenv
from weave import Evaluation
import hydra
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

import configs.global_config as GC
from src.preprocess.preprocess_eval_data import get_evaluation_dataset
from src.eval.model import LOCATION_LABELS
from src.eval.decision_procedure import DecisionProcedure
from src.eval.scorer import ResultCollector
from src.utils.predicate import Predicate
from src.utils.program import Program, AssertionPointAttributes
from src.utils.program_normalizer import ProgramNormalizer
from src.verifiers.uautomizer import UAutomizerVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class PerLocationInvariantModel(weave.Model):
    """
    Weave Model for generating invariants at specific locations.
    
    Unlike the main InvariantGeneratorModel which lets the model choose the location,
    this model is given a specific location and generates an invariant there.
    """
    client: str
    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    sampling_params: Dict
    reasoning_effort: str
    only_loop_beginnings: bool = True
    # Optional: only needed for HuggingFace client
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def model_post_init(self, __context):
        """Initialize model after Pydantic model creation."""
        if self.client == "hf":
            self.tokenizer, self.model = self._load_hf_model(self.model_cfg)
        elif self.client == "vllm":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["base_model_name"], token=True)
        else:
            raise ValueError(f"Invalid client: {self.client}")

    def _load_hf_model(self, model_cfg: dict) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """Load HuggingFace model."""
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_name"], token=True)
        model_kwargs = dict(
            attn_implementation="eager",
            dtype=torch.bfloat16,
            use_cache=True,
            device_map="auto"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model_name"],
            **model_kwargs,
            token=True
        )
        if model_cfg.get("eval_finetuned_model") and model_cfg.get("is_lora"):
            model = PeftModel.from_pretrained(model, model_cfg["finetuned_model_id"], token=True)
            model = model.merge_and_unload()
        model.eval()
        return tokenizer, model

    def _format_program_with_single_label(self, program: Program, target_line: int, target_label: str) -> str:
        """Format program with only the target location marked."""
        lines = program.lines.copy()
        
        # Apply GPT-friendly replacements
        for i, line in enumerate(lines):
            if line in program.replacement_for_GPT:
                lines[i] = program.replacement_for_GPT[line]
        
        # Add lemmas
        for lemma in program.lemmas:
            lines[lemma.line_number] += f"\nassume({lemma.content});"
        
        # Add only the target location label
        if target_line < len(lines):
            lines[target_line] += f"\n// Line {target_label}"
        
        # Add target assertion
        if program.assertions:
            target = program.assertions[0]
            if target.line_number < len(lines):
                lines[target.line_number] += f"\nassert({target.content}); // Target property"
        
        return "\n".join(lines)

    @weave.op
    def _parse_harmony_output(self, raw_output: str) -> Dict[str, Optional[str]]:
        """Parse Harmony format output."""
        final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)'
        final_match = re.search(final_pattern, raw_output, re.DOTALL)
        
        analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
        analysis_match = re.search(analysis_pattern, raw_output, re.DOTALL)
        
        analysis = analysis_match.group(1).strip() if analysis_match else ""
        final = final_match.group(1).strip() if final_match else None
        return {"analysis": analysis, "final": final}

    @weave.op
    def predict(self, program: Program, target_line: int, target_label: str) -> Dict:
        """
        Generate an invariant at a specific location.
        
        Args:
            program: The program to analyze.
            target_line: The line number to generate the invariant at.
            target_label: The label for this location (A, B, C, ...).
            
        Returns:
            Dict with generated invariant and metadata.
        """
        if self.client == "hf":
            return self.predict_hf(program, target_line, target_label)
        elif self.client == "vllm":
            return self.predict_vllm(program, target_line, target_label)
        else:
            raise ValueError(f"Invalid client: {self.client}")

    @weave.op
    def predict_hf(self, program: Program, target_line: int, target_label: str) -> Dict:
        """Generate using HuggingFace."""
        formatted_program = self._format_program_with_single_label(program, target_line, target_label)
        user_prompt = self.user_prompt_template.content.format(program=formatted_program, target_label=target_label)
        
        messages = [
            {"role": 'developer', "content": self.system_prompt.content},
            {"role": "user", "content": user_prompt},
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            reasoning_effort=self.reasoning_effort,
        ).to(self.model.device)
        
        inference_start_time = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, **self.sampling_params)
            model_latency = time.perf_counter() - inference_start_time
        
        raw_output = self.tokenizer.batch_decode(output_ids)[0]
        parsed_output = self._parse_harmony_output(raw_output)
        reasoning, answer = parsed_output["analysis"], parsed_output["final"]
        
        usage = {
            "prompt_tokens": input_ids.shape[1],
            "reasoning_tokens": len(self.tokenizer.encode(reasoning)) if reasoning else 0,
            "answer_tokens": len(self.tokenizer.encode(answer)) if answer else 0,
            "completion_tokens": output_ids.shape[1] - input_ids.shape[1],
            "total_tokens": input_ids.shape[1] + output_ids.shape[1] - input_ids.shape[1],
        }
        
        return {
            "raw_output": raw_output,
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
            "target_line": target_line,
            "target_label": target_label,
        }

    @weave.op
    def predict_vllm(self, program: Program, target_line: int, target_label: str) -> Dict:
        """Generate using vLLM."""
        import openai
        
        vllm_base_url = self.model_cfg.get("vllm_base_url", "http://localhost:8000/v1")
        vllm_model = self.model_cfg.get("vllm_model", "gen_inv_adapter")
        
        formatted_program = self._format_program_with_single_label(program, target_line, target_label)
        user_prompt = self.user_prompt_template.content.format(program=formatted_program, target_label=target_label)
        
        messages = [
            {"role": 'developer', "content": self.system_prompt.content},
            {"role": "user", "content": user_prompt},
        ]
        
        client = openai.OpenAI(base_url=vllm_base_url, api_key="dummy")
        
        inference_start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=vllm_model,
            messages=messages,
            max_tokens=self.sampling_params.get("max_new_tokens", 16384),
            temperature=self.sampling_params.get("temperature", 1.0),
            top_p=self.sampling_params.get("top_p", 1.0),
            extra_body={
                "include_reasoning": True,
                "reasoning_effort": self.reasoning_effort,
            },
        )
        model_latency = time.perf_counter() - inference_start_time
        
        message = response.choices[0].message
        reasoning = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', '') or ''
        answer = message.content or ''
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "reasoning_tokens": len(self.tokenizer.encode(reasoning)) if reasoning else 0,
            "answer_tokens": len(self.tokenizer.encode(answer)) if answer else 0,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        
        return {
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
            "target_line": target_line,
            "target_label": target_label,
        }


class PerLocationScorer(weave.Scorer):
    """
    Weave Scorer for evaluating invariants at specific locations.
    """
    verifier: Optional[UAutomizerVerifier] = None
    collector: Optional[ResultCollector] = None
    baseline_is_timeout: bool = False

    def _parse_predicate_at_location(self, answer: str) -> str:
        """
        Parse model answer to extract predicate content (no location needed).
        """
        if not answer:
            return ""
        
        assert_pos = answer.find("assert(")
        if assert_pos == -1:
            return ""
        
        start_pos = assert_pos + 7
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
            return ""
        
        return answer[start_pos:end_pos].strip()

    @weave.op()
    def score(self, output: Dict, original_program: str, median_timing: float, target_line: int, target_label: str) -> Dict:
        """
        Score a model-generated invariant at a specific location.
        
        Args:
            output: Model output containing 'reasoning', 'answer', and 'model_latency'.
            original_program: The C program source code from the dataset.
            median_timing: Baseline verification time.
            target_line: The line number where the invariant should be placed.
            target_label: The label for this location (A, B, C, ...).
            
        Returns:
            Dict containing all evaluation metrics.
        """
        # Create Program object
        normalizer = ProgramNormalizer(code=original_program)
        program = Program(normalizer.lines_to_verify, normalizer.replacement)
        
        baseline_timing = median_timing
        
        # Parse the invariant (no location parsing needed - we know where it goes)
        answer = output.get("answer", "")
        predicate_content = self._parse_predicate_at_location(answer)
        
        predicate = Predicate(content=predicate_content, line_number=target_line)
        
        timeout = baseline_timing if self.baseline_is_timeout else self.verifier.timeout_seconds
        
        with tempfile.TemporaryDirectory() as temp_dir:
            code_dir = Path(temp_dir)
            decision_procedure = DecisionProcedure(
                verifier=self.verifier,
                program=program,
                code_dir=code_dir,
                timeout=timeout
            )
            report = decision_procedure.run(
                candidate=predicate,
                model_latency=output.get("model_latency", 0.0)
            )
        
        correctness_score = (
            report.correctness_report is not None
            and report.correctness_report.decision == "TRUE"
        )
        
        speedup_no_gen = 0.0
        speedup_gen = 0.0
        if correctness_score and report.final_decision in ["TRUE", "FALSE"]:
            speedup_no_gen = baseline_timing / report.verification_time if report.verification_time > 0 else 0.0
            speedup_gen = baseline_timing / report.total_time if report.total_time > 0 else 0.0
        
        result = {
            # Core scores
            "validation_score": report.is_valid,
            "correctness_score": correctness_score,
            "has_speedup_no_gen": bool(speedup_no_gen > 1),
            "has_speedup_gen": bool(speedup_gen > 1),
            "speedup_no_gen": speedup_no_gen,
            "speedup_gen": speedup_gen,
            "final_decision": report.final_decision,
            "decision_rule": report.decision_rule,
            "verification_time": report.verification_time,
            "total_time": report.total_time,
            # Location info
            "target_line": target_line,
            "target_label": target_label,
            # Parsed fields
            "predicate_content": predicate_content,
            "baseline_timing": baseline_timing,
            # Reports
            "program_for_usefulness": getattr(report, 'program_for_usefulness', ''),
            "usefulness_report": report.usefulness_report.to_dict() if report.usefulness_report else None,
            "program_for_correctness": getattr(report, 'program_for_correctness', ''),
            "correctness_report": report.correctness_report.to_dict() if report.correctness_report else None,
        }
        
        if self.collector is not None:
            self.collector.collect(result)
        
        return result

    @weave.op()
    def summarize(self, score_rows: List[Dict]) -> Dict:
        """Aggregate scores across all examples and locations."""
        valid_rows = [r for r in score_rows if r is not None]
        if not valid_rows:
            return {}
        
        n = len(valid_rows)
        
        # Overall metrics
        validation_count = sum(1 for r in valid_rows if r.get("validation_score", False))
        correctness_count = sum(1 for r in valid_rows if r.get("correctness_score", False))
        speedup_count_no_gen = sum(1 for r in valid_rows if r.get("has_speedup_no_gen", False))
        speedup_count_gen = sum(1 for r in valid_rows if r.get("has_speedup_gen", False))
        
        validation_rate = validation_count / n
        correctness_rate = correctness_count / n
        speedup_rate_no_gen = speedup_count_no_gen / n
        speedup_rate_gen = speedup_count_gen / n
        
        # Speedup metrics
        speedups_gt1_no_gen = [r["speedup_no_gen"] for r in valid_rows if r.get("has_speedup_no_gen", False)]
        speedups_gt1_gen = [r["speedup_gen"] for r in valid_rows if r.get("has_speedup_gen", False)]
        
        speedup_gt1_no_gen = sum(speedups_gt1_no_gen) / len(speedups_gt1_no_gen) if speedups_gt1_no_gen else 1.0
        speedup_gt1_gen = sum(speedups_gt1_gen) / len(speedups_gt1_gen) if speedups_gt1_gen else 1.0
        
        # Per-location breakdown - aggregate to single average rate
        location_stats = {}
        for r in valid_rows:
            label = r.get("target_label", "?")
            if label not in location_stats:
                location_stats[label] = {"total": 0, "correct": 0, "speedup": 0}
            location_stats[label]["total"] += 1
            if r.get("correctness_score", False):
                location_stats[label]["correct"] += 1
            if r.get("has_speedup_no_gen", False):
                location_stats[label]["speedup"] += 1
    
        
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
            # Speedup
            "speedup_gt1_no_gen": speedup_gt1_no_gen,
            "speedup_gt1_gen": speedup_gt1_gen,
            # Total
            "total_examples": n,
            "num_locations": len(location_stats),
        }


def get_candidate_locations(program: Program, only_loop_beginnings: bool = True) -> Dict[int, str]:
    """
    Get all candidate locations for invariant generation.
    
    Returns:
        Dict mapping line numbers to labels (A, B, C, ...).
    """
    assertion_points = program.assertion_points
    
    if only_loop_beginnings:
        filtered_points = {
            line_num: attrs for line_num, attrs in assertion_points.items()
            if AssertionPointAttributes.BeginningOfLoop in attrs
        }
    else:
        filtered_points = assertion_points
    
    labeled = {}
    for i, line_num in enumerate(sorted(filtered_points.keys())):
        if i < len(LOCATION_LABELS):
            labeled[line_num] = LOCATION_LABELS[i]
    return labeled


def expand_dataset_per_location(dataset: List[dict], only_loop_beginnings: bool = True) -> List[dict]:
    """
    Expand dataset so each row represents a specific (program, location) pair.
    
    Args:
        dataset: Original dataset with one row per program.
        only_loop_beginnings: Whether to only include loop beginning locations.
        
    Returns:
        Expanded dataset with one row per (program, location) pair.
    """
    expanded = []
    
    for example in dataset:
        code = str(example["original_program"])
        normalizer = ProgramNormalizer(code=code)
        program = Program(normalizer.lines_to_verify, normalizer.replacement)
        
        locations = get_candidate_locations(program, only_loop_beginnings)
        
        for line_num, label in locations.items():
            expanded.append({
                **example,  # Keep all original fields
                "target_line": line_num,
                "target_label": label,
            })
    
    logger.info(f"Expanded {len(dataset)} examples to {len(expanded)} (program, location) pairs")
    return expanded


def preprocess_per_location_example(example: dict) -> dict:
    """
    Preprocess a dataset example for per-location model input.
    
    Args:
        example: Dataset example containing program, location info, and timing.
        
    Returns:
        Dict with program and location info for the model.
    """
    code = str(example["original_program"])
    normalizer = ProgramNormalizer(code=code)
    program = Program(normalizer.lines_to_verify, normalizer.replacement)
    
    return {
        "program": program,
        "target_line": example["target_line"],
        "target_label": example["target_label"],
    }


def get_run_name(cfg: DictConfig) -> str:
    """Get run name for per-location evaluation."""
    base = cfg.model.finetuned_model_id if cfg.model.eval_finetuned_model else cfg.model.base_model_name.split('/')[-1]
    return f"eval-per-location-{base}-{cfg.model.reasoning_effort}-{cfg.dataset.split}"


def get_model_display_name(cfg: DictConfig) -> str:
    """Get display name for the model."""
    if cfg.model.eval_finetuned_model:
        return f"{cfg.model.finetuned_model_id}-{cfg.model.reasoning_effort}"
    else:
        return f"{cfg.model.base_model_name.split('/')[-1]}-{cfg.model.reasoning_effort}"


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config")
def main(cfg: DictConfig):
    """Main per-location evaluation entry point."""
    print("=" * 100)
    print("PER-LOCATION EVALUATION (Weave)")
    print("=" * 100)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 100)
    
    run_name = get_run_name(cfg)
    model_display_name = get_model_display_name(cfg)
    logger.info(f"run_name: {run_name}")
    logger.info(f"model_display_name: {model_display_name}")
    
    # Initialize Weave
    weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")
    
    # Load dataset (limited for per-location evaluation)
    # limit = cfg.dataset.limit if cfg.dataset.limit > 0 else 10  # Default to 10
    raw_dataset = get_evaluation_dataset(
        dataset_name=cfg.dataset.name,
        limit=cfg.dataset.limit,
        prefix=cfg.dataset.prefix,
        split=cfg.dataset.split
    )
    logger.info(f"Loaded {len(raw_dataset)} base examples")
    
    # Expand dataset to have one row per (program, location) pair
    dataset = expand_dataset_per_location(raw_dataset, cfg.prompts.only_loop_beginnings)
    
    # Setup prompts
    system_prompt = weave.StringPrompt(cfg.prompts.default_system_prompt)
    user_prompt_template = weave.StringPrompt(cfg.prompts.per_location_user_prompt_template)
    
    # Create model
    invariant_generator = PerLocationInvariantModel(
        client=cfg.model.client,
        model_cfg=cfg.model,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        sampling_params=dict(cfg.model.sampling_params),
        reasoning_effort=cfg.model.reasoning_effort,
        only_loop_beginnings=cfg.prompts.only_loop_beginnings,
    )
    
    # Setup scorer
    result_collector = ResultCollector()
    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.scorer.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.scorer.verifier.property,
        arch=cfg.scorer.verifier.arch,
        timeout_seconds=cfg.scorer.verifier.timeout_seconds,
        version=cfg.scorer.verifier.version
    )
    scorer = PerLocationScorer(
        verifier=verifier,
        collector=result_collector,
        baseline_is_timeout=cfg.scorer.baseline_is_timeout
    )
    
    # Create evaluation
    evaluation = Evaluation(
        dataset=dataset,
        scorers=[scorer],
        preprocess_model_input=preprocess_per_location_example
    )
    
    # Run evaluation
    logger.info("Starting per-location evaluation...")
    asyncio.run(evaluation.evaluate(invariant_generator, __weave={"display_name": run_name}))
    
    # Log summary
    plot_results = result_collector.get_results()
    logger.info(f"Collected {len(plot_results)} results")
    
    if plot_results:
        # Per-location breakdown
        location_stats = {}
        for r in plot_results:
            label = r.get("target_label", "?")
            if label not in location_stats:
                location_stats[label] = {"total": 0, "correct": 0, "speedup": 0}
            location_stats[label]["total"] += 1
            if r.get("correctness_score", False):
                location_stats[label]["correct"] += 1
            if r.get("has_speedup_no_gen", False):
                location_stats[label]["speedup"] += 1
        
        logger.info("\nPer-location breakdown:")
        for label in sorted(location_stats.keys()):
            stats = location_stats[label]
            logger.info(f"  Line {label}: {stats['correct']}/{stats['total']} correct, {stats['speedup']}/{stats['total']} speedup")
    
    logger.info("Per-location evaluation complete!")


if __name__ == "__main__":
    main()
