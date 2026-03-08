"""
OpenRouter-based invariant generator model.

Implements inference via the OpenRouter API for cloud-hosted models,
with cost tracking and reasoning token extraction.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import weave
from dotenv import load_dotenv
from openai import OpenAI

from wonda.core.ast_program import AstProgram

load_dotenv()
logger = logging.getLogger(__name__)


class InvariantGeneratorOpenRouterModel(weave.Model):
    """
    OpenRouter API model for loop invariant generation.

    Given a C program with INVARIANT_MARKER annotations, generates candidate
    loop invariants using cloud-hosted models via the OpenRouter API.
    Tracks per-request cost from OpenRouter usage metadata.
    """

    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    openai_client: Optional[OpenAI] = None

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(
            self, 'openai_client',
            OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"),
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
        if target_marker:
            user_prompt = self.user_prompt_template.content.format(
                program=program.llm_code, target_marker=target_marker
            )
        else:
            user_prompt = self.user_prompt_template.content.format(
                program=program.llm_code
            )

        messages = [
            {"role": "system", "content": self.system_prompt.content},
            {"role": "user", "content": user_prompt},
        ]

        inference_start_time = time.perf_counter()
        response = self.openai_client.chat.completions.create(
            model=self.model_cfg["base_model"]["id"],
            messages=messages,
            **self.model_cfg["base_model"]["sampling_params"],
        )
        model_latency = time.perf_counter() - inference_start_time
        logger.debug("Model latency: %s", model_latency)

        # Extract response components
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", "") or ""
        answer = message.content or ""
        logger.debug("Answer: %s", answer[:200] + "..." if len(answer) > 200 else answer)

        # Extract reasoning tokens from completion_tokens_details if available
        completion_details = response.usage.completion_tokens_details
        reasoning_tokens = 0
        if completion_details:
            reasoning_tokens = getattr(completion_details, 'reasoning_tokens', 0) or 0

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "reasoning_tokens": reasoning_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        # Extract cost details from OpenRouter response
        cost = {}
        if hasattr(response.usage, 'cost_details') and response.usage.cost_details:
            cost_details = response.usage.cost_details
            cost = {
                "total_cost_usd": cost_details.get('upstream_inference_cost', 0) or 0,
                "prompt_cost_usd": cost_details.get('upstream_inference_prompt_cost', 0) or 0,
                "completion_cost_usd": cost_details.get('upstream_inference_completions_cost', 0) or 0,
            }
        elif hasattr(response.usage, 'cost') and response.usage.cost:
            cost = {"total_cost_usd": response.usage.cost or 0}

        return {
            "prompt": "\n".join([msg["content"] for msg in messages]),
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": usage,
            "cost": cost,
        }

    def get_display_name(self) -> str:
        """Return the display name for this model configuration."""
        return self.model_cfg['base_model']['id'].split('/')[-1]

    def get_run_name(self) -> str:
        """Return the run name for evaluation tracking."""
        return f"eval-{self.get_display_name()}"
