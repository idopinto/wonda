"""Shared utilities for model training and inference."""
import logging

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from wonda.core.ast_program import AstProgram

logger = logging.getLogger(__name__)


def init_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a tokenizer and set pad_token if missing."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer {tokenizer.name_or_path}...")
    return tokenizer


def load_model(model_name: str, model_kwargs: dict) -> AutoModelForCausalLM:
    """Load a pretrained causal LM with the given kwargs."""
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info(f"Loaded model\n{model}\n")
    return model


def print_trainable_parameters(model) -> tuple[int, int]:
    """Print trainable vs total parameters to verify LoRA is working."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / all_params
    logger.info(
        f"Trainable params: {trainable_params:,} | "
        f"All params: {all_params:,} | Trainable %: {pct:.4f}%"
    )
    return trainable_params, all_params


def log_dataset_sample(dataset: Dataset, name: str) -> None:
    """Log the first sample of a dataset split for sanity-checking."""
    logger.info(f"{name} dataset sample:")
    for key, value in dataset[0].items():
        logger.info(f"  {key}: {value}")


def run_inference(
    sample: dict,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    model_kwargs: dict,
    prompts: dict,
) -> str:
    """Run a single inference pass on a sample for quick testing."""
    enable_thinking = model_kwargs["enable_thinking"]
    system_prompt = prompts["per_marker_system_prompt"]
    user_prompt_template = prompts["per_marker_user_prompt_template"]

    ast_program = AstProgram().from_code(sample["original_program"])
    ast_program.process(print_ast=False)
    user_prompt = user_prompt_template.format(
        program=ast_program.llm_code,
        target_marker=sample["invariants"][0]["marker"],
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=False,
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **model_kwargs["inference"]["sampling_params"])

    response = tokenizer.batch_decode(output_ids)[0]
    logger.info("=== Inference Test ===")
    logger.info(response)
    logger.info("=" * 50)
    return response
