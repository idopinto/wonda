"""
Qwen-based invariant generator model.

Implements local HuggingFace inference for Qwen models (base and LoRA fine-tuned),
with support for thinking/non-thinking modes.
"""

import logging
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

from src.preprocess.program import Program

logger = logging.getLogger(__name__)


class InvariantGeneratorQwenModel(weave.Model):
    """
    Weave Model for generating loop invariants.

    This model takes a C program with marked assertion points and generates
    candidate invariants that can accelerate traditional verifiers.
    """

    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    sampling_params: Dict
    enable_thinking: bool
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
        self.tokenizer, self.model = self._load_hf_model()

    @weave.op
    def predict(
        self, program: Program, target_marker: Optional[str] = None, **_: Any
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
            {"role": "system", "content": self.system_prompt.content},
            {"role": "user", "content": user_prompt},
        ]
        inference_start_time = time.perf_counter()
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, **self.sampling_params)
        model_latency = time.perf_counter() - inference_start_time
        logger.info(f"Model latency: {model_latency}")

        raw_output = self.tokenizer.batch_decode(output_ids)[0]
        parsed_output = self._parse_qwen_output(raw_output)
        reasoning, answer = parsed_output["reasoning"], parsed_output["answer"]
        logger.info(f"Reasoning: {reasoning}")
        logger.info(f"Answer: {answer}")
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

    def _parse_qwen_output(self, raw_output: str) -> Dict[str, Optional[str]]:
        """Parse Qwen format output, extracting reasoning and answer."""
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, raw_output, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        answer_pattern = r"</think>\s*(.*?)(?:<\|im_end\|>|$)"
        answer_match = re.search(answer_pattern, raw_output, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        # Fallback: if no answer found, try parsing <|im_start|>assistant format (for qwen instruct models)
        if not answer:
            reasoning = ""
            assistant_pattern = r"<\|im_start\|>assistant\s*(.*?)(?:<\|im_end\|>|$)"
            assistant_match = re.search(assistant_pattern, raw_output, re.DOTALL)
            if assistant_match:
                answer = assistant_match.group(1).strip()

        return {"reasoning": reasoning, "answer": answer}

    def _load_hf_model(self) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """Load a trained model and tokenizer based on configuration.

        Returns:
            Tuple of (tokenizer, model) ready for inference.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, token=True
        )
        model_kwargs = dict(
            attn_implementation=self.model_cfg["base_model"].get("attn_implementation", "eager"),
            dtype=torch.bfloat16,
            use_cache=True,
            device_map="auto",
        )

        if self.eval_ft_model:
            if not self.model_cfg["ft_model"]["is_lora"]:
                logger.info(f"Loading non-LoRA-finetuned model from {self.ft_model_id}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.ft_model_id, **model_kwargs, token=True
                )
            else:
                logger.info(f"Loading base model from {self.base_model_id}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id, **model_kwargs, token=True
                )
                logger.info(f"Loading LoRA-finetuned model from {self.ft_model_id}")
                model = PeftModel.from_pretrained(model, self.ft_model_id, token=True)
                model = model.merge_and_unload()
                logger.info("Merged and unloaded")
        else:
            logger.info(f"Loading base model from {self.base_model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, **model_kwargs, token=True
            )

        model.eval()
        return tokenizer, model

    def get_display_name(self) -> str:
        """Return the display name for this model configuration."""
        model_id = self.ft_model_id if self.eval_ft_model else self.base_model_id
        base_name = model_id.split("/")[-1]
        suffix = "t" if self.enable_thinking else "nt" # t: thinking, nt: non-thinking
        return f"{base_name}-{suffix}"

    def get_run_name(self) -> str:
        """Return the run name for evaluation tracking."""
        return f"eval-{self.get_display_name()}"
