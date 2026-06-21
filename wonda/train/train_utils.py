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


def derive_model_run_name(
    base_model_name: str,
    version: str,
    min_grade: int | None,
    use_peft: bool,
    num_train_epochs: int | None = None,
) -> str:
    """Build the canonical ``Wonda-<base>[-nt]-V<n>[-gt<N>][-LoRA][-e<N>]`` name.

    Conventions:
      - ``<base>``: HF repo name without the org prefix. ``Llama-3.X-`` is
        collapsed to ``Llama3.X-`` to match the user-facing naming style.
      - ``-nt``: appended only when the base is NOT an instruction-tuned model
        (case-insensitive substring check for ``instruct``).
      - ``-V<n>``: dataset version (V0/V1/V2), uppercased.
      - ``-gt<N>``: only for V2; ``N = min_grade`` (default 2). Denotes
        ``quality_grade >= N``.
      - ``-LoRA``: when ``use_peft=True``. Full fine-tune adds no suffix.
      - ``-e<N>``: training epochs.

    Trainers append ``-test`` externally for ``test_mode`` runs; this helper
    deliberately returns the production-style name.

    Examples:
        >>> derive_model_run_name("Qwen/Qwen3-0.6B", "v2", 2, False, 2)
        'Wonda-Qwen3-0.6B-nt-V2-gt2-e2'
        >>> derive_model_run_name("Qwen/Qwen3-8B", "v2", 2, True, 2)
        'Wonda-Qwen3-8B-nt-V2-gt2-LoRA-e2'
        >>> derive_model_run_name("meta-llama/Llama-3.1-8B-Instruct", "v2", 3, False, 3)
        'Wonda-Llama3.1-8B-Instruct-V2-gt3-e3'
    """
    base = base_model_name.split("/", 1)[-1]
    base = base.replace("Llama-3.1-", "Llama3.1-").replace("Llama-3.2-", "Llama3.2-")

    parts: list[str] = ["Wonda", base]
    if "instruct" not in base.lower():
        parts.append("nt")
    v = version.lower().lstrip("v")
    parts.append(f"V{v}")
    if version.lower() == "v2":
        eff_min_grade = min_grade if min_grade is not None else 2
        parts.append(f"gt{eff_min_grade}")
    if use_peft:
        parts.append("LoRA")
    if num_train_epochs is not None:
        parts.append(f"e{num_train_epochs}")
    return "-".join(parts)


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
