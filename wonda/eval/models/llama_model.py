"""
Llama-based invariant generator model.

Implements local HuggingFace inference for Meta Llama models using the standard
chat template (no thinking/non-thinking toggle). Parses the assistant turn from
the model's <|start_header_id|>assistant<|end_header_id|> ... <|eot_id|> output.
"""

import logging
import re
import time
from typing import Any, Dict, Optional

import torch
import weave

from wonda.eval.models.model_utils import load_hf_model
from wonda.core.ast_program import AstProgram

logger = logging.getLogger(__name__)


class InvariantGeneratorLlamaModel(weave.Model):
    """
    Local HuggingFace Llama model for loop invariant generation.

    Given a C program with INVARIANT_MARKER annotations, generates candidate
    loop invariants. Supports base and LoRA fine-tuned checkpoints.
    Uses the standard Llama 3 chat template without thinking-mode toggles.
    """

    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    sampling_params: Dict
    eval_ft_model: bool = False
    base_model_id: Optional[str] = None
    ft_model_id: Optional[str] = None
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def model_post_init(self, __context):
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
            target_marker: The target loop marker to generate an invariant for.
        Returns:
            Dict containing the generated invariant, timing info, and usage stats.
        """
        user_prompt = self.user_prompt_template.content.format(
            program=program.llm_code, target_marker=target_marker
        )
        messages = [
            {"role": "system", "content": self.system_prompt.content},
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
        logger.info(f"Model latency: {model_latency}")

        raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        answer = self._parse_llama_output(raw_output)
        logger.info(f"Answer: {answer}")

        usage = {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": output_ids.shape[1] - input_ids.shape[1],
            "total_tokens": output_ids.shape[1],
        }

        return {
            "raw_output": raw_output,
            "reasoning": "",
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
        }

    def _parse_llama_output(self, raw_output: str) -> str:
        """
        Extract the assistant reply from Llama 3 chat-formatted output.

        Llama 3 format:
          <|start_header_id|>assistant<|end_header_id|>\n\n{reply}<|eot_id|>
        Falls back to returning everything after the last assistant header if
        the eot token is absent (e.g. truncated generation).
        """
        pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)(?:<\|eot_id\|>|$)"
        matches = re.findall(pattern, raw_output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return raw_output.strip()

    def get_display_name(self) -> str:
        model_id = self.ft_model_id if self.eval_ft_model else self.base_model_id
        return model_id.split("/")[-1]

    def get_run_name(self) -> str:
        return f"eval-{self.get_display_name()}"
