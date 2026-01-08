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

from src.preprocess.program import Program


class InvariantGeneratorOssModel(weave.Model):
    """
    Weave Model for generating loop invariants.

    This model takes a C program with marked assertion points and generates
    candidate invariants that can accelerate traditional verifiers.
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
        self.tokenizer, self.model = self._load_hf_model()

    @weave.op
    def predict(
        self, program: Program, target_label: Optional[str] = None, **_: Any
    ) -> Dict:
        """
        Generate a candidate invariant for a given program.

        Args:
            program: The C program to analyze.
        Returns:
            Dict containing the generated invariant, timing info, and usage stats.
        """
        
        return self.predict_hf(program, target_label=target_label)

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

    @weave.op
    def predict_hf(
        self, program: Program, target_label: Optional[str] = None, **_: Any
    ) -> Dict:
        """
        Generate a candidate invariant for a given program using Hugging Face.

        Args:
            program: The C program to analyze.
        Returns:
            Dict containing the generated invariant, timing info, and usage stats.
        """
        user_prompt = self.user_prompt_template.content.format(
            program=program.llm_code, target_label=target_label
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
            "prompt": "\n".join([message["content"] for message in messages]),
            "raw_output": raw_output,
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
        }

    @weave.op
    def predict_vllm(
        self, program: Program, target_label: Optional[str] = None, **_: Any
    ) -> Dict:
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
        user_prompt = self.user_prompt_template.content.format(
            program=program.llm_code, target_label=target_label
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
            "prompt": "\n".join([message["content"] for message in messages]),
            # "raw_output": raw_output,
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
        }

    def _load_hf_model(self) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """
        Load a trained model and tokenizer based on configuration.

        Returns:
            Tuple of (tokenizer, model) ready for inference.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, token=True
        )
        model_kwargs = dict(
            attn_implementation="eager",
            dtype=torch.bfloat16,
            use_cache=True,
            device_map="auto",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, **model_kwargs, token=True
        )

        if self.eval_ft_model and self.model_cfg.get("ft_model", {}).get("is_lora", False):
            print(f"Loading finetuned model from {self.ft_model_id}")
            model = PeftModel.from_pretrained(
                model, self.ft_model_id, token=True
            )
            model = model.merge_and_unload()
            print("Merged and unloaded")
        model.eval()
        return tokenizer, model

    def get_display_name(self) -> str:
        """Return the display name for this model configuration."""
        base_name = self.base_model_id.split("/")[-1]
        suffix = self.reasoning_effort if self.reasoning_effort else "no-reasoning"
        return f"{base_name}-{suffix}"

    def get_run_name(self) -> str:
        """Return the run name for evaluation tracking."""
        return f"eval-{self.get_display_name()}"
