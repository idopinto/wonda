"""
Model module for invariant generation.

Contains the InvariantGeneratorModel class and model loading utilities.
"""

import re
import time
from typing import Any, Dict, Optional

import torch
import weave
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.utils.program import Program

LOCATION_LABELS = [chr(ord("A") + i) for i in range(26)]


class InvariantGeneratorModel(weave.Model):
    """
    Weave Model for generating loop invariants.

    This model takes a C program with marked assertion points and generates
    candidate invariants that can accelerate traditional verifiers.
    """

    client: str
    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    sampling_params: Dict
    reasoning_effort: str
    only_loop_beginnings: bool = (
        True  # If True, only mark lines at the beginning of loops
    )
    # Optional: only needed for HuggingFace client (use Any to avoid Pydantic validation issues)
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def model_post_init(self, __context):
        """Initialize model after Pydantic model creation."""
        if self.client == "hf":
            self.tokenizer, self.model = self._load_hf_model(self.model_cfg)
        elif self.client == "vllm":  # when h200 is available
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg["base_model_name"], token=True
            )
        else:
            raise ValueError(f"Invalid client: {self.client}")

    @weave.op
    def predict(self, program: Program) -> Dict:
        """
        Generate a candidate invariant for a given program.

        Args:
            program: The C program to analyze.
        Returns:
            Dict containing the generated invariant, timing info, and usage stats.
        """
        if self.client == "vllm":
            return self.predict_vllm(program)
        elif self.client == "hf":
            return self.predict_hf(program)
        else:
            raise ValueError(f"Invalid client: {self.client}")

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
        final_pattern = (
            r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)"
        )
        final_match = re.search(final_pattern, raw_output, re.DOTALL)

        analysis_pattern = r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"
        analysis_match = re.search(analysis_pattern, raw_output, re.DOTALL)

        analysis = analysis_match.group(1).strip() if analysis_match else ""
        final = final_match.group(1).strip() if final_match else None
        return {"analysis": analysis, "final": final}

    def _label_assertion_points(
        self, assertion_points: dict, only_loop_beginnings: bool = True
    ) -> tuple[dict, dict]:
        """
        Label assertion points and create bidirectional mapping.

        Args:
            assertion_points: Dict mapping line numbers to assertion attributes.
            only_loop_beginnings: If True, only label assertion points that are at the beginning of loops.

        Returns:
            Tuple of (labeled_points, name_to_line) dicts.
        """
        from src.utils.program import AssertionPointAttributes

        labeled, name_to_line = {}, {}

        # Filter assertion points if only_loop_beginnings is True
        if only_loop_beginnings:
            filtered_points = {
                line_num: attrs
                for line_num, attrs in assertion_points.items()
                if AssertionPointAttributes.BeginningOfLoop in attrs
            }
        else:
            filtered_points = assertion_points

        for i, line_num in enumerate(sorted(filtered_points.keys())):
            if i < len(LOCATION_LABELS):
                label = LOCATION_LABELS[i]
                labeled[line_num] = label
                name_to_line[label] = line_num
        return labeled, name_to_line

    def _format_program_with_labels(
        self, program: Program, labeled_points: dict
    ) -> str:
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
                lines[target.line_number] += (
                    f"\nassert({target.content}); // Target property"
                )

        return "\n".join(lines)

    @weave.op
    def predict_hf(self, program: Program) -> Dict:
        """
        Generate a candidate invariant for a given program using Hugging Face.

        Args:
            program: The C program to analyze.
        Returns:
            Dict containing the generated invariant, timing info, and usage stats.
        """
        assertion_points = program.assertion_points
        labeled_points, _ = self._label_assertion_points(
            assertion_points, only_loop_beginnings=self.only_loop_beginnings
        )
        formatted_program = self._format_program_with_labels(
            program=program, labeled_points=labeled_points
        )
        user_prompt = self.user_prompt_template.content.format(
            program=formatted_program
        )
        messages = [
            {"role": "developer", "content": self.system_prompt.content},
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
            "reasoning_tokens": len(self.tokenizer.encode(reasoning))
            if reasoning
            else 0,
            "answer_tokens": len(self.tokenizer.encode(answer)) if answer else 0,
            "completion_tokens": output_ids.shape[1] - input_ids.shape[1],
            "total_tokens": input_ids.shape[1]
            + output_ids.shape[1]
            - input_ids.shape[1],
        }

        return {
            "raw_output": raw_output,
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
        }

    @weave.op
    def predict_vllm(self, program: Program) -> Dict:
        """
        Generate a candidate invariant for a given program using VLLM.

        Args:
            program: The C program to analyze.
        Returns:
            Dict containing the generated invariant, timing info, and usage stats.
        """
        import openai

        # Get vLLM config from model_cfg
        vllm_base_url = self.model_cfg.get("vllm_base_url", "http://localhost:8000/v1")
        vllm_model = self.model_cfg.get("vllm_model", "gen_inv_adapter")

        assertion_points = program.assertion_points
        labeled_points, _ = self._label_assertion_points(
            assertion_points, only_loop_beginnings=self.only_loop_beginnings
        )
        formatted_program = self._format_program_with_labels(
            program=program, labeled_points=labeled_points
        )
        user_prompt = self.user_prompt_template.content.format(
            program=formatted_program
        )
        messages = [
            {"role": "developer", "content": self.system_prompt.content},
            {"role": "user", "content": user_prompt},
        ]

        # Create vLLM client
        client = openai.OpenAI(
            base_url=vllm_base_url,
            api_key="dummy",  # vLLM doesn't require API key by default
        )

        # Make request to vLLM server
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

        # Extract response components
        message = response.choices[0].message
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", "")
            or ""
        )
        answer = message.content or ""

        # Build raw output for compatibility (combine reasoning + answer)
        # raw_output = f"<|channel|>analysis<|message|>{reasoning}<|end|>\n<|channel|>final<|message|>{answer}<|return|>"

        # Build usage dict compatible with regular predict
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "reasoning_tokens": len(self.tokenizer.encode(reasoning))
            if reasoning
            else 0,
            "answer_tokens": len(self.tokenizer.encode(answer)) if answer else 0,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return {
            # "raw_output": raw_output,
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
        }

    def _load_hf_model(
        self, model_cfg: dict
    ) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """
        Load a trained model and tokenizer based on configuration.

        Args:
            model_cfg: Dictionary containing model settings.

        Returns:
            Tuple of (tokenizer, model) ready for inference.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["base_model_name"], token=True
        )
        model_kwargs = dict(
            attn_implementation="eager",
            dtype=torch.bfloat16,
            use_cache=True,
            device_map="auto",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model_name"], **model_kwargs, token=True
        )

        if model_cfg.get("eval_finetuned_model") and model_cfg.get("is_lora"):
            print(f"Loading finetuned model from {model_cfg['finetuned_model_id']}")
            model = PeftModel.from_pretrained(
                model, model_cfg["finetuned_model_id"], token=True
            )
            model = model.merge_and_unload()
            print("Merged and unloaded")
        model.eval()
        return tokenizer, model
