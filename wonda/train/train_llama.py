"""
SFT fine-tuning script for Llama 3.1 models on the loop-invariant generation task.

Loads a pre-trained Llama checkpoint, optionally applies LoRA, and trains it
via supervised fine-tuning on per loop marker invariant prediction data.
Automatically re-formats Qwen-templated SFT datasets to Llama 3.1 chat format.
Configuration is managed by Hydra (configs/train/).

Usage:
    uv run -m wonda.train.train_llama --config-name=<config> [overrides...]

Examples:
    uv run -m wonda.train.train_llama --config-name=llama3_8b dataset.version=v2 dataset.min_grade=3
    uv run -m wonda.train.train_llama --config-name=llama3_8b test_mode=true
"""
import re
import logging

import hydra
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from wonda.train.data_utils import load_sft_dataset, split_dataset
from wonda.train.train_utils import (
    load_model,
    log_dataset_sample,
    print_trainable_parameters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_QWEN_MSG_PATTERN = re.compile(
    r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL
)


def _reformat_qwen_to_llama(text: str, tokenizer: AutoTokenizer) -> str:
    """Re-format a Qwen chat-templated string into Llama 3.1 chat format.

    Parses <|im_start|>role ... <|im_end|> blocks, then applies the Llama
    tokenizer's chat_template to produce the correct special-token framing.
    Falls back to the original text if no Qwen markers are found.
    """
    matches = _QWEN_MSG_PATTERN.findall(text)
    if not matches:
        return text
    messages = [{"role": role, "content": content.strip()} for role, content in matches]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def reformat_dataset_for_llama(
    dataset: Dataset, tokenizer: AutoTokenizer, max_length: int
) -> Dataset:
    """Convert every sample's ``text`` field from Qwen to Llama chat format,
    dropping samples that exceed *max_length* tokens after conversion."""
    converted, skipped = [], 0
    for sample in dataset:
        new_text = _reformat_qwen_to_llama(sample["text"], tokenizer)
        if len(tokenizer.encode(new_text)) > max_length:
            skipped += 1
            continue
        new_sample = dict(sample)
        new_sample["text"] = new_text
        converted.append(new_sample)
    logger.info(
        f"Reformatted {len(converted)} samples to Llama chat format "
        f"(skipped {skipped} exceeding {max_length} tokens)"
    )
    return Dataset.from_list(converted)


def _model_name_to_size_slug(base_model_name: str) -> str:
    """Extract size slug, e.g. 'meta-llama/Llama-3.1-8B-Instruct' -> '8b'."""
    m = re.search(r"Llama[_-]3(?:\.\d+)?[_-](\d+)[Bb]", base_model_name)
    if not m:
        raise ValueError(f"Cannot derive size slug from: {base_model_name}")
    return m.group(1).lower() + "b"


def derive_run_names(cfg: DictConfig) -> tuple[str, str | None, str]:
    """Derive wandb name, hf_repo, and output_dir from config."""
    base = cfg.model.base_model_name
    version = cfg.dataset.get("version") or "v2"
    min_grade = cfg.dataset.get("min_grade")
    hf_org = cfg.dataset.get("hf_organization")
    tune_mode = "lora" if cfg.use_peft else "full"
    grade_suffix = ""
    if version == "v2":
        effective_min_grade = min_grade if min_grade is not None else 2
        grade_suffix = f"-g{effective_min_grade}"

    epochs = cfg.sft.get("num_train_epochs") or 2
    epoch_suffix = f"-e{epochs}"

    size_slug = _model_name_to_size_slug(base)
    wandb_name = f"llama3-{size_slug}-{tune_mode}-gen-inv-sft-{version}{grade_suffix}{epoch_suffix}"
    output_dir = f"trained_models/llama3-{size_slug}-{tune_mode}-gen-inv-sft-{version}{grade_suffix}{epoch_suffix}"
    if hf_org:
        if version == "v2":
            hf_repo = f"{hf_org}/wonda-qwen-nt-sft-{version}-g2"
        else:
            hf_repo = f"{hf_org}/wonda-qwen-nt-sft-{version}"
    else:
        hf_repo = None

    return wandb_name, hf_repo, output_dir


def init_llama_tokenizer(model_name: str) -> AutoTokenizer:
    """Load the Llama tokenizer with a dedicated padding token.

    Llama 3.1 ships ``<|finetune_right_pad_id|>`` (id 128004) which avoids
    conflating pad and eos during training.  Falls back to eos if unavailable.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    pad_token = "<|finetune_right_pad_id|>"
    if pad_token in tokenizer.get_vocab():
        tokenizer.pad_token = pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info(f"Loaded tokenizer {tokenizer.name_or_path} (pad={tokenizer.pad_token})")
    return tokenizer


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    training_args: SFTConfig,
    dataset: Dataset,
    validation_dataset: Dataset,
) -> None:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )
    trainer.train()


@hydra.main(version_base=None, config_path="../../configs/train", config_name="llama3_8b")
def main(cfg: DictConfig):
    logger.info("=" * 50)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 50)

    if cfg.wandb.get("name") is None or cfg.sft.get("output_dir") is None or cfg.dataset.get("hf_repo") is None:
        derived_wandb, derived_hf_repo, derived_output = derive_run_names(cfg)
        if cfg.wandb.get("name") is None:
            OmegaConf.update(cfg, "wandb.name", derived_wandb)
        if cfg.dataset.get("hf_repo") is None and cfg.dataset.get("json_path") is None and derived_hf_repo is not None:
            OmegaConf.update(cfg, "dataset.hf_repo", derived_hf_repo)
        if cfg.sft.get("output_dir") is None:
            OmegaConf.update(cfg, "sft.output_dir", derived_output)

    wandb_name = cfg.wandb.name
    output_dir = cfg.sft.output_dir
    hf_repo = cfg.dataset.get("hf_repo")

    if cfg.test_mode:
        logger.info("Training in test mode...")
        wandb_name = wandb_name + "-test"
        output_dir = output_dir + "-test"
        cfg.dataset.limit = 100

    if cfg.sft.push_to_hub and not hf_repo:
        raise ValueError(
            "push_to_hub is true but dataset.hf_repo is not set. "
            "Set dataset.hf_repo or dataset.hf_organization in your config."
        )
    if cfg.wandb.use_wandb:
        if not cfg.wandb.get("entity"):
            raise ValueError("use_wandb is true but wandb.entity is not set.")
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=wandb_name)
    else:
        logger.info("Wandb is disabled. Skipping wandb initialization...")

    model_name = cfg.model.base_model_name
    json_path = cfg.dataset.get("json_path")
    if not hf_repo and not json_path:
        raise ValueError("Provide dataset.hf_repo (pre-built SFT dataset) or dataset.json_path")
    if hf_repo and json_path:
        raise ValueError("Provide exactly one of dataset.hf_repo or dataset.json_path")

    tokenizer = init_llama_tokenizer(model_name)
    logger.info(f"model_name: {model_name} | dataset: {hf_repo or json_path} | output_dir: {output_dir}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.sft.num_train_epochs,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        optim=cfg.sft.optimizer,
        logging_steps=cfg.sft.logging_steps,
        save_strategy=cfg.sft.save_strategy,
        save_only_model=cfg.sft.save_only_model,
        learning_rate=cfg.sft.learning_rate,
        max_length=cfg.sft.max_length,
        max_grad_norm=cfg.sft.max_grad_norm,
        warmup_ratio=cfg.sft.warmup_ratio,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.sft.lr_scheduler_kwargs,
        dataset_text_field=cfg.sft.dataset_text_field,
        eval_strategy=cfg.sft.eval_strategy,
        eval_steps=cfg.sft.eval_steps,
        report_to=cfg.sft.report_to,
        push_to_hub=cfg.sft.push_to_hub,
        bf16=True,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        weight_decay=cfg.sft.get("weight_decay", 0.01),
    )

    model_init_kwargs = dict(
        attn_implementation=cfg.model.init_kwargs_train.attn_implementation,
        dtype=torch.bfloat16,
        use_cache=cfg.model.init_kwargs_train.use_cache,
        device_map=cfg.model.init_kwargs_train.device_map,
    )

    full_dataset = load_sft_dataset(
        hf_repo=hf_repo,
        json_path=json_path,
        limit=cfg.dataset.get("limit", -1),
        min_grade=cfg.dataset.get("min_grade"),
    )

    logger.info("Reformatting Qwen-templated dataset to Llama 3.1 chat format...")
    full_dataset = reformat_dataset_for_llama(
        full_dataset, tokenizer, max_length=cfg.sft.max_length
    )
    train_dataset, validation_dataset = split_dataset(full_dataset, split_ratio=cfg.dataset.split_ratio)

    log_dataset_sample(train_dataset, "Train")
    log_dataset_sample(validation_dataset, "Validation")

    model = load_model(model_name, model_init_kwargs)
    if cfg.use_peft:
        target_modules = list(cfg.lora.target_modules) if cfg.lora.target_modules else None
        lora_kwargs = {
            "r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "target_modules": target_modules,
            "lora_dropout": cfg.lora.lora_dropout,
            "bias": cfg.lora.bias,
            "task_type": cfg.lora.task_type,
        }
        lora_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
    train(model, tokenizer, training_args, train_dataset, validation_dataset)

    wandb.finish()
    logger.info("DONE.")


if __name__ == "__main__":
    main()
