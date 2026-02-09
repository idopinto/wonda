"""
Shared utilities for HuggingFace model loading.
"""

import logging

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def load_hf_model(
    base_model_id: str,
    ft_model_id: str,
    eval_ft_model: bool,
    model_cfg: dict,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a HuggingFace model and tokenizer based on configuration.

    Handles three scenarios:
    - Base model only (eval_ft_model=False)
    - Non-LoRA fine-tuned model (loads directly from ft_model_id)
    - LoRA fine-tuned model (loads base, applies adapter, merges)

    Returns:
        Tuple of (tokenizer, model) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=True)
    model_kwargs = dict(
        attn_implementation=model_cfg["base_model"].get("attn_implementation", "eager"),
        dtype=torch.bfloat16,
        use_cache=True,
        device_map="auto",
    )

    if eval_ft_model:
        if not model_cfg["ft_model"]["is_lora"]:
            logger.info(f"Loading non-LoRA-finetuned model from {ft_model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                ft_model_id, **model_kwargs, token=True
            )
        else:
            logger.info(f"Loading base model from {base_model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, **model_kwargs, token=True
            )
            logger.info(f"Loading LoRA-finetuned model from {ft_model_id}")
            model = PeftModel.from_pretrained(model, ft_model_id, token=True)
            model = model.merge_and_unload()
            logger.info("Merged and unloaded")
    else:
        logger.info(f"Loading base model from {base_model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, **model_kwargs, token=True
        )

    model.eval()
    return tokenizer, model
