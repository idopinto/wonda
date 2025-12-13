"""
Model module for invariant generation.

Contains the InvariantGeneratorModel class and model loading utilities.
"""
import re
import time
from typing import Dict, Optional

import torch
import weave
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.eval.config import LOCATION_LABELS, EvalConfig
from src.utils.program import Program


def load_model(config: EvalConfig) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load a trained model and tokenizer based on configuration.
    
    Args:
        config: Evaluation configuration containing model settings.
        
    Returns:
        Tuple of (tokenizer, model) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, token=True)
    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,
        use_cache=True,
        device_map="auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        **model_kwargs,
        token=True
    )

    if config.eval_finetuned_model:
        model = PeftModel.from_pretrained(model, config.peft_model_id, token=True)
        model = model.merge_and_unload()
    
    model.eval()
    return tokenizer, model


class InvariantGeneratorModel(weave.Model):
    """
    Weave Model for generating loop invariants.
    
    This model takes a C program with marked assertion points and generates
    candidate invariants that can accelerate traditional verifiers.
    """
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    sampling_params: Dict
    reasoning_effort: str

    @weave.op
    def predict(self, program: Program) -> Dict:
        """
        Generate a candidate invariant for a given program.
        
        Args:
            program: The C program to analyze.
        Returns:
            Dict containing the generated invariant, timing info, and usage stats.
        """
        assertion_points = program.assertion_points
        labeled_points, _ = self._label_assertion_points(assertion_points)
        formatted_program = self._format_program_with_labels(
            program=program,
            labeled_points=labeled_points
        )
        user_prompt = self.user_prompt_template.content.format(program=formatted_program)
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
            "usage": usage
        }

    @weave.op
    def _parse_harmony_output(self, raw_output: str) -> Dict[str, Optional[str]]:
        """
        Parse Harmony format output from GPT-OSS models.
        
        Harmony format structure:
        - Analysis/reasoning: <|channel|>analysis<|message|>...<|end|>
        - Final answer: <|channel|>final<|message|>...<|return|> or <|end|>
        
        Returns:
            Dict with 'reasoning' and 'raw_response' keys.
        """
        final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)'
        final_match = re.search(final_pattern, raw_output, re.DOTALL)
        
        analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
        analysis_match = re.search(analysis_pattern, raw_output, re.DOTALL)
        
        analysis = analysis_match.group(1).strip() if analysis_match else ""
        final = final_match.group(1).strip() if final_match else None
        return {"analysis": analysis, "final": final}

    def _label_assertion_points(self, assertion_points: dict) -> tuple[dict, dict]:
        """
        Label assertion points and create bidirectional mapping.
        
        Args:
            assertion_points: Dict mapping line numbers to assertion attributes.
            
        Returns:
            Tuple of (labeled_points, name_to_line) dicts.
        """
        labeled, name_to_line = {}, {}
        for i, line_num in enumerate(sorted(assertion_points.keys())):
            if i < len(LOCATION_LABELS):
                label = LOCATION_LABELS[i]
                labeled[line_num] = label
                name_to_line[label] = line_num
        return labeled, name_to_line

    def _format_program_with_labels(self, program: Program, labeled_points: dict) -> str:
        """
        Format program with location labels and target assertion.
        
        Args:
            program: The program to format.
            labeled_points: Dict mapping line numbers to labels.
            
        Returns:
            Formatted program string with labels.
        """
        lines = program.lines.copy()
        
        # Apply GPT-friendly replacements
        for i, line in enumerate(lines):
            if line in program.replacement_for_GPT:
                lines[i] = program.replacement_for_GPT[line]
        
        # Add lemmas
        for lemma in program.lemmas:
            lines[lemma.line_number] += f"\nassume({lemma.content});"
        
        # Add location labels
        for line_num, label in labeled_points.items():
            if line_num < len(lines):
                lines[line_num] += f"\n// Line {label}"
        
        # Add target assertion
        if program.assertions:
            target = program.assertions[0]
            if target.line_number < len(lines):
                lines[target.line_number] += f"\nassert({target.content}); // Target property"
        
        return "\n".join(lines)

