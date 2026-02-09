"""
GPT-OSS (Harmony) invariant generator model.

Implements local HuggingFace inference for GPT-OSS models with Harmony-format
output parsing (analysis/final channels) and configurable reasoning effort.
"""

import logging
import re
import time
from typing import Any, Dict, Optional

import torch
import weave

from src.eval.models.model_utils import load_hf_model
from src.preprocess.ast_program import AstProgram

logger = logging.getLogger(__name__)


class InvariantGeneratorOssModel(weave.Model):
    """
    Local HuggingFace GPT-OSS model for loop invariant generation.

    Given a C program with INVARIANT_MARKER annotations, generates candidate
    loop invariants. Uses Harmony output format with analysis and final channels.
    Supports base and LoRA fine-tuned checkpoints.
    """

    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    sampling_params: Dict
    reasoning_effort: str
    eval_ft_model: bool = False
    # Set in model_post_init from model_cfg
    base_model_id: Optional[str] = None
    ft_model_id: Optional[str] = None
    # Optional: only needed for local huggingface loading (use Any to avoid Pydantic validation issues)
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def model_post_init(self, __context):
        """Initialize model after Pydantic model creation."""
        self.base_model_id = self.model_cfg["base_model"]["id"]
        self.ft_model_id = self.model_cfg["ft_model"]["id"]
        self.tokenizer, self.model = load_hf_model(
            self.base_model_id, self.ft_model_id, self.eval_ft_model, self.model_cfg
        )

    @weave.op
    def predict(
        self, program: AstProgram, target_marker: Optional[str] = None, **_: Any
    ) -> Dict:
        """
        Generate a candidate invariant for a given program.

        Args:
            program: The C program to analyze.
        Returns:
            Dict containing the generated invariant, timing info, and usage stats.
        """
        user_prompt = self.user_prompt_template.content.format(
            program=program.llm_code, target_marker=target_marker
        )
        messages = [
            {"role": "developer", "content": self.system_prompt.content},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
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
            "total_tokens": output_ids.shape[1],
        }

        return {
            "raw_output": raw_output,
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
        }

    def _parse_harmony_output(self, raw_output: str) -> Dict[str, Optional[str]]:
        """
        Parse Harmony format output from GPT-OSS models.

        Harmony format structure:
        - Analysis/reasoning: <|channel|>analysis<|message|>...<|end|>
        - Final answer: <|channel|>final<|message|>...<|return|> or <|end|>

        Returns:
            Dict with 'analysis' and 'final' keys.
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

    def get_display_name(self) -> str:
        """Return the display name for this model configuration."""
        base_name = self.base_model_id.split("/")[-1]
        suffix = self.reasoning_effort if self.reasoning_effort else "no-reasoning"
        return f"{base_name}-{suffix}"

    def get_run_name(self) -> str:
        """Return the run name for evaluation tracking."""
        return f"eval-{self.get_display_name()}"
