"""
Together AI-based invariant generator model.

Implements inference via the Together API for cloud-hosted models.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import weave
from dotenv import load_dotenv
from together import Together

from wonda.core.ast_program import AstProgram

load_dotenv()
logger = logging.getLogger(__name__)


class InvariantGeneratorTogetherModel(weave.Model):
    """
    Together API model for loop invariant generation.

    Given a C program with INVARIANT_MARKER annotations, generates candidate
    loop invariants using cloud-hosted models via the Together inference API.
    """

    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    together_client: Optional[Together] = None

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(
            self, 'together_client',
            Together(api_key=os.getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1"),
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
        print(response.__dict__)
        return {
            # "prompt": "\n".join([msg["content"] for msg in messages]),
            "reasoning": reasoning,
            "answer": answer,
            "model_latency": model_latency,
            "usage": response.usage,
            "cost": {},
        }

    def get_display_name(self) -> str:
        """Return the display name for this model configuration."""
        return self.model_cfg['base_model']['id'].split('/')[-1]

    def get_run_name(self) -> str:
        """Return the run name for evaluation tracking."""
        return f"eval-{self.get_display_name()}"
