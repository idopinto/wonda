"""
SFT fine-tuning script for Qwen models on the loop-invariant generation task.

Loads a pre-trained Qwen checkpoint, optionally applies LoRA, and trains it
via supervised fine-tuning on per-marker invariant prediction data.
Configuration is managed by Hydra (configs/train/models/).

Usage:
    uv run -m src.train.train_qwen --config-name=<config> [overrides...]

Examples:
    uv run -m src.train.train_qwen --config-name=qwen3_0.6b_nt_v2 dataset.min_grade=3
    uv run -m src.train.train_qwen --config-name=qwen3_8b_nt_v1 test_mode=true
"""
import logging

import hydra
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.train.data_utils import load_training_data, split_dataset
from src.train.train_utils import (
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


@hydra.main(version_base=None, config_path="../../configs/train/models", config_name="qwen3_0.6b_nt_v0")
def main(cfg: DictConfig):
    logger.info("=" * 50)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 50)

    # Derive run-specific naming from min_grade
    min_grade_str = "." + str(cfg.dataset.min_grade) if cfg.dataset.get("min_grade") else ""
    wandb_name = cfg.wandb.name + min_grade_str
    output_dir = cfg.sft.output_dir + min_grade_str
    hf_upload_repo = (cfg.dataset.hf_upload_repo + min_grade_str) if cfg.dataset.get("hf_upload_repo") else None

    if cfg.test_mode:
        logger.info("Training in test mode...")
        wandb_name = wandb_name + "-test"
        hf_upload_repo = (hf_upload_repo + "-test") if hf_upload_repo else None
        output_dir = output_dir + "-test"
        cfg.dataset.limit = 100

    if cfg.wandb.use_wandb:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=wandb_name)
    else:
        logger.info("Wandb is disabled. Skipping wandb initialization...")

    model_name = cfg.model.base_model_name
    raw_repo_name = cfg.dataset.repo_name
    force_reprocess = cfg.dataset.get("force_reprocess", False)
    tokenizer = init_tokenizer(model_name)
    logger.info(
        f"model_name: {model_name} | raw_repo_name: {raw_repo_name} | "
        f"hf_upload_repo: {hf_upload_repo} | output_dir: {output_dir}"
    )
    logger.info(f"force_reprocess training data: {force_reprocess}")

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

    # Load and split data
    full_dataset = load_training_data(
        dataset_name=raw_repo_name,
        tokenizer=tokenizer,
        system_prompt=cfg.prompts.per_marker_system_prompt,
        user_prompt_template=cfg.prompts.per_marker_user_prompt_template,
        limit=cfg.dataset.limit,
        inv_mode=cfg.dataset.inv_mode,
        split=cfg.dataset.split,
        max_length=cfg.sft.max_length,
        json_path=cfg.dataset.get("json_path"),
        min_grade=cfg.dataset.get("min_grade") or 1,
        output_dir=cfg.sft.output_dir,
        hf_upload_repo=hf_upload_repo,
        force_reprocess=force_reprocess,
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
