"""
SFT fine-tuning script for Qwen models on the loop-invariant generation task.

Loads a pre-trained Qwen checkpoint, optionally applies LoRA, and trains it
via supervised fine-tuning on per loop marker invariant prediction data.
Configuration is managed by Hydra (configs/train/).

Usage:
    uv run -m wonda.train.train_qwen --config-name=<config> [overrides...]

Examples:
    uv run -m wonda.train.train_qwen --config-name=qwen3_0.6b dataset.version=v2 dataset.min_grade=3
    uv run -m wonda.train.train_qwen --config-name=qwen3_8b dataset.version=v1 test_mode=true
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
    init_tokenizer,
    load_model,
    log_dataset_sample,
    print_trainable_parameters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default HF org for pre-built SFT datasets
_DEFAULT_HF_ORG = "idopinto"


def _model_name_to_size_slug(base_model_name: str) -> str:
    """Extract size slug from base_model_name, e.g. 'Qwen/Qwen3-0.6B' -> '0.6b'."""
    m = re.search(r"Qwen3-(\d+\.?\d*)[Bb]", base_model_name)
    if not m:
        raise ValueError(f"Cannot derive size slug from base_model_name: {base_model_name}")
    return m.group(1).lower() + "b"


def derive_run_names(cfg: DictConfig) -> tuple[str, str | None, str]:
    """
    Derive wandb name, hf_repo, and output_dir from model + dataset.version + dataset.min_grade.
    Used when config has null for these (unified per-model-size configs).
    Returns (wandb_name, hf_repo, output_dir).
    """
    base = cfg.model.base_model_name
    version = cfg.dataset.get("version") or "v2"
    min_grade = cfg.dataset.get("min_grade")

    size_slug = _model_name_to_size_slug(base)
    wandb_name = f"qwen3-{size_slug}-nt-gen-inv-sft-{version}"
    output_dir = f"trained_models/qwen3-{size_slug}-nt-gen-inv-sft-{version}"

    if version == "v2":
        grade_suffix = f"-g{min_grade}" if min_grade is not None else "-g2"
        hf_repo = f"{_DEFAULT_HF_ORG}/wonda-qwen-nt-sft-{version}{grade_suffix}"
    else:
        hf_repo = f"{_DEFAULT_HF_ORG}/wonda-qwen-nt-sft-{version}"

    return wandb_name, hf_repo, output_dir


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


@hydra.main(version_base=None, config_path="../../configs/train", config_name="qwen3_0.6b")
def main(cfg: DictConfig):
    logger.info("=" * 50)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 50)

    # Auto-derive wandb name, hf_repo, output_dir when null 
    if cfg.wandb.get("name") is None or cfg.sft.get("output_dir") is None or cfg.dataset.get("hf_repo") is None:
        derived_wandb, derived_hf_repo, derived_output = derive_run_names(cfg)
        if cfg.wandb.get("name") is None:
            OmegaConf.update(cfg, "wandb.name", derived_wandb)
        if cfg.dataset.get("hf_repo") is None and cfg.dataset.get("json_path") is None:
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

    if cfg.wandb.use_wandb:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=wandb_name)
    else:
        logger.info("Wandb is disabled. Skipping wandb initialization...")

    model_name = cfg.model.base_model_name
    json_path = cfg.dataset.get("json_path")
    if not hf_repo and not json_path:
        raise ValueError("Provide dataset.hf_repo (pre-built SFT dataset) or dataset.json_path")
    if hf_repo and json_path:
        raise ValueError("Provide exactly one of dataset.hf_repo or dataset.json_path")
    tokenizer = init_tokenizer(model_name)
    logger.info(f"model_name: {model_name} | dataset: {hf_repo or json_path} | output_dir: {output_dir}")

    # Build SFT training args from config
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
    )

    # Build model kwargs from config
    model_init_kwargs = dict(
        attn_implementation=cfg.model.init_kwargs_train.attn_implementation,
        dtype=torch.bfloat16,
        use_cache=cfg.model.init_kwargs_train.use_cache,
        device_map=cfg.model.init_kwargs_train.device_map,
    )

    # Load pre-built SFT dataset and split (optional min_grade: keep only quality_grade >= min_grade)
    full_dataset = load_sft_dataset(
        hf_repo=hf_repo,
        json_path=json_path,
        limit=cfg.dataset.get("limit", -1),
        min_grade=cfg.dataset.get("min_grade"),
    )
    train_dataset, validation_dataset = split_dataset(full_dataset, split_ratio=cfg.dataset.split_ratio)

    log_dataset_sample(train_dataset, "Train")
    log_dataset_sample(validation_dataset, "Validation")

    # Load model and optionally apply LoRA
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
        if hasattr(cfg.lora, "target_parameters") and cfg.lora.target_parameters:
            lora_kwargs["target_parameters"] = list(cfg.lora.target_parameters)
        lora_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
    train(model, tokenizer, training_args, train_dataset, validation_dataset)

    wandb.finish()
    logger.info("DONE.")


if __name__ == "__main__":
    main()
