"""
Model module for invariant generation.

Contains the InvariantGeneratorModel class and model loading utilities.
"""

import time
from typing import Any, Dict, Optional

import weave
from src.preprocess.program import Program
from dotenv import load_dotenv
from openai import OpenAI
import os
import logging
from together import Together
logger = logging.getLogger(__name__)
load_dotenv()
class InvariantGeneratorTogetherModel(weave.Model):

    """
    Weave Model for generating loop invariants.

    This model takes a C program with marked assertion points and generates
    candidate invariants that can accelerate traditional verwifiers.
    """
    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    together_client: Optional[Together] = None

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, 'together_client', Together(api_key=os.getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1"))
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

        response = self.together_client.chat.completions.create(
            model=self.model_cfg["base_model"]["id"],
            messages=messages,
            **self.model_cfg["base_model"]["sampling_params"],
        )
        model_latency = time.perf_counter() - inference_start_time
        logger.info(f"Model latency: {model_latency}")
        
        # Extract response components
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", "") or ""
        answer = message.content or ""

        logger.info(f"Answer: {answer}")
        
        # Extract reasoning tokens from completion_tokens_details if available
        # completion_details = response.usage.completion_tokens_details
        # reasoning_tokens = 0
        # if completion_details:
        #     reasoning_tokens = getattr(completion_details, 'reasoning_tokens', 0) or 0
        
        # print(response.usage)
        # usage = {
        #     "prompt_tokens": response.usage.prompt_tokens,
        #     "reasoning_tokens": reasoning_tokens,
        #     "completion_tokens": response.usage.completion_tokens,
        #     "total_tokens": response.usage.total_tokens,
        # }

        # Extract cost details from OpenRouter response (tracked in Weave)
        # Cost details are in response.usage.cost_details based on actual response format
        # cost = {}
        # if hasattr(response.usage, 'cost_details') and response.usage.cost_details:
        #     cost_details = response.usage.cost_details
        #     cost = {
        #         "total_cost_usd": cost_details.get('upstream_inference_cost', 0) or 0,
        #         "prompt_cost_usd": cost_details.get('upstream_inference_prompt_cost', 0) or 0,
        #         "completion_cost_usd": cost_details.get('upstream_inference_completions_cost', 0) or 0,
        #     }
            # elif hasattr(response.usage, 'cost') and response.usage.cost:
            #     # Fallback to direct cost field if cost_details not available
            #     cost = {"total_cost_usd": response.usage.cost or 0}

        return {
            "prompt": "\n".join([message["content"] for message in messages]),
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": {},
            "cost": {},
        }
    def get_display_name(self) -> str:
        """Return the display name for this model configuration."""
        return f"{self.model_cfg['base_model']['id'].split('/')[-1]}"

    def get_run_name(self) -> str:
        """Return the run name for evaluation tracking."""
        return f"eval-{self.get_display_name()}"