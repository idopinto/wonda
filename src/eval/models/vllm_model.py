"""
Model module for invariant generation.

Contains the InvariantGeneratorModel class and model loading utilities.
"""

import re
import time
import logging
import json
from typing import Any, Dict, Optional

import requests
import torch
import weave
from src.preprocess.program import Program
from dotenv import load_dotenv
from openai import OpenAI
import os
from transformers import AutoTokenizer
from openai.types.chat import ChatCompletion

load_dotenv()
logger = logging.getLogger(__name__)
class InvariantGeneratorVllmModel(weave.Model):
    """
    Weave Model for generating loop invariants.

    This model takes a C program with marked assertion points and generates
    candidate invariants that can accelerate traditional verifiers.
    """
    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    client: Optional[OpenAI] = None
    tokenizer: Optional[AutoTokenizer] = None
    vllm_model: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, 'client', OpenAI(api_key="dummy", base_url=self.model_cfg.get("vllm", {}).get("base_url", "http://localhost:8000/v1")))
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(self.model_cfg["base_model"]["id"], trust_remote_code=True))
        if self.model_cfg["eval_ft_model"]:
            if self.model_cfg["ft_model"]["is_lora"]:
                object.__setattr__(self, 'vllm_model', self.model_cfg["vllm"]["model"])
            else:
                object.__setattr__(self, 'vllm_model', self.model_cfg["ft_model"]["id"])
        else:
            object.__setattr__(self, 'vllm_model', self.model_cfg["base_model"]["id"])

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
        if target_marker:
            user_prompt = self.user_prompt_template.content.format(program=program.llm_code, target_marker=target_marker)
        else:
            user_prompt = self.user_prompt_template.content.format(program=program.llm_code)
        
        messages = [
            {"role": "system", "content": self.system_prompt.content},
            {"role": "user", "content": user_prompt},
        ]

        # Make request to vLLM server
        inference_start_time = time.perf_counter()

        model_name = self.vllm_model or self.model_cfg["base_model"]["id"]
        extra_body = dict(self.model_cfg.get("extra_body", {}))
        
        # Get all sampling params and convert max_new_tokens to max_tokens
        raw_params = dict(self.model_cfg["base_model"]["sampling_params"])
        if "max_new_tokens" in raw_params:
            raw_params["max_tokens"] = raw_params.pop("max_new_tokens")
        
        # vLLM accepts all these parameters directly: max_tokens, temperature, top_p, top_k, min_p, etc.
        # Make direct HTTP request to bypass OpenAI client validation
        base_url = self.model_cfg.get("vllm", {}).get("base_url", "http://localhost:8000/v1")
        url = f"{base_url}/chat/completions"
        
        payload = {
            "model": model_name,
            "messages": messages,
            **raw_params,  # Include all params: max_tokens, temperature, top_p, top_k, min_p, etc.
        }
        if extra_body:
            payload.update(extra_body)
        
        http_response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        http_response.raise_for_status()
        
        # Convert response to OpenAI-compatible format
        response_data = http_response.json()
        response = ChatCompletion(**response_data)
        print(response)
        model_latency = time.perf_counter() - inference_start_time
        logger.info(f"Model latency: {model_latency}")
        # Extract response components
        message = response.choices[0].message
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", "")
            or ""
        )
        answer = message.content or ""
        logger.info(f"Answer: {answer}")
        # Build usage dict compatible with regular predict
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "reasoning_tokens": len(self.tokenizer.encode(reasoning)) if reasoning else 0,
            "answer_tokens": len(self.tokenizer.encode(answer)) if answer else 0,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        logger.info(f"Finished predicting with vLLM model {self.vllm_model}. now verifying answer...")

        return {
            "prompt": "\n".join([message["content"] for message in messages]),
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
        }
    def get_display_name(self) -> str:
        """Return the display name for this model configuration."""
        base_name = self.vllm_model.split("/")[-1]
        return f"{base_name}-vllm"

    def get_run_name(self) -> str:
        """Return the run name for evaluation tracking."""
        return f"eval-{self.get_display_name()}"