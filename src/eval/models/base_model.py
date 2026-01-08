"""
Base interface for invariant generator models.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

import weave

from src.preprocess.program import Program


class InvariantGeneratorModelInterface(weave.Model):
    """
    Abstract base class defining the interface for invariant generator models.
    """

    client: str
    model_cfg: dict
    system_prompt: weave.StringPrompt
    user_prompt_template: weave.StringPrompt
    sampling_params: Dict
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    @abstractmethod
    def predict(
        self, program: Program, target_label: Optional[str] = None, **_: Any
    ) -> Dict:
        """Generate a candidate invariant for a given program."""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Return the display name for this model configuration."""
    pass

    @abstractmethod
    def get_run_name(self) -> str:
        """Return the run name for evaluation tracking."""
        pass
