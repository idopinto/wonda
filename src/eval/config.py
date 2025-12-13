"""
Configuration module for evaluation.

Centralizes all configuration, constants, and prompts for the evaluation pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
import configs.global_configurations as GC
# import os
import weave

# Constants
LOCATION_LABELS = [chr(ord('A') + i) for i in range(26)]

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are an expert C programmer and highly profiecient in generating strong loop invariants for C programs that accelerates traditional verifiers."""

DEFAULT_USER_PROMPT_TEMPLATE = """Generate a single candidate invariant for a given C program. you can place it at any of the marked lines with // Line <letter>, choose the most beneficial location:
```c
{program}
```
Answer only the invariant in the following format and nothing else:
assert(<invariant>); // Line <letter>

Example:
assert(a > 0 && a < 10); // Line A
"""
# TODO: add more examples
# os.environ["TRANSFORMERS_VERBOSITY"] = "info"


@dataclass
class SamplingParams:
    """Parameters for model sampling. """
    max_new_tokens: int = 16384
    # do_sample: bool = True
    temperature: float = 1.0
    top_p=1.0
    top_k=0.0# check k=100
    # maximum_context_length_window = 131,072

    def to_dict(self) -> Dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            # "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

@dataclass
class EvalConfig:
    """
    Configuration for the evaluation pipeline.
    
    Centralizes all configuration options including model settings,
    verification parameters, and runtime options.
    """

    # Dataset configuration
    dataset_name: str = "idopinto/invbench-evaluation-uautomizer25-k3"
    weave_team: str = "ip-ai"
    project_name: str = "eval-inv-gen"
    eval_finetuned_model: bool = False
    # split: str = "hard"
    # Model configuration
    base_model_name: str = "openai/gpt-oss-20b"
    finetuned_model_id: str = "gpt-oss-20b-rlinv-sft-sep"
    is_lora: bool = True
    reasoning_effort: str = "medium"
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    
    # Verification configuration
    
    verifier_config: GC.DefaultVerificationConfig = field(default_factory=lambda: GC.DefaultVerificationConfig(version='25', property_file_path=GC.PROPERTIES_DIR / "unreach-call.prp", arch='32bit', timeout_seconds=600.0))
    
    # When True, use baseline timing as timeout for verification
    # This cuts overall experiment time.
    baseline_is_timeout: bool = False
    
    # Prompts (initialized in __post_init__)
    system_prompt: weave.StringPrompt = field(default=None, repr=False)
    user_prompt_template: weave.StringPrompt = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize Weave prompts after dataclass creation."""
        if self.system_prompt is None:
            self.system_prompt = weave.StringPrompt(DEFAULT_SYSTEM_PROMPT)
        if self.user_prompt_template is None:
            self.user_prompt_template = weave.StringPrompt(DEFAULT_USER_PROMPT_TEMPLATE)
    
    def get_run_name(self, split: str) -> str:
        """Get run name for evaluation."""
        if self.eval_finetuned_model:
            return f"eval-{self.finetuned_model_id}-{self.reasoning_effort}-{split}"
        else:
            return f"eval-{self.base_model_name.split('/')[-1]}-{self.reasoning_effort}-{split}"
    
    def get_model_display_name(self) -> str:
        """Get display name for the model including reasoning effort."""
        if self.eval_finetuned_model:
            return f"{self.peft_model_id}-{self.reasoning_effort}"
        else:
            return f"{self.base_model_name.split('/')[-1]}-{self.reasoning_effort}"


def get_default_config() -> EvalConfig:
    """Get the default evaluation configuration."""
    return EvalConfig()

