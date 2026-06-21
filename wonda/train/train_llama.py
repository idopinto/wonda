"""
SFT fine-tuning script for Llama 3.1 models on the loop-invariant generation task.

Loads a pre-trained Llama checkpoint, optionally applies LoRA, and trains it
via supervised fine-tuning on per loop marker invariant prediction data.
The SFT cache for this family is rendered with Llama's chat template directly.
Configuration is managed by Hydra (configs/train/).

Usage:
    uv run -m wonda.train.train_llama --config-name=<config> [overrides...]

Examples:
    uv run -m wonda.train.train_llama --config-name=llama3_8b dataset.version=v2 dataset.min_grade=3
    uv run -m wonda.train.train_llama --config-name=llama3_8b test_mode=true
"""
import logging
from pathlib import Path

import hydra
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from wonda.preprocess.build_sft_dataset import sft_dataset_basename
from wonda.train.data_utils import (
    SFT_CACHE_ROOT,
    load_or_build_sft_dataset,
    split_dataset,
)
from wonda.train.train_utils import (
    derive_model_run_name,
    load_model,
    log_dataset_sample,
    print_trainable_parameters,
)

# Cache is built with the Llama-3.1 tokenizer so the dataset is rendered in
# Llama's chat format directly — no runtime reformat required. The Llama
# tokenizer is identical across 3.1 sizes; pinning the Instruct 8B keeps the
# build cheap while still producing a cache reusable by all Llama-3.1 trainees.
_SFT_CACHE_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
_PREPROCESS_CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "preprocess" / "build_sft_dataset.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def derive_run_names(cfg: DictConfig) -> tuple[str, str, str, str | None]:
    """Derive run names. Returns (wandb_name, dataset_cache_basename, output_dir, hub_model_id)."""
    base = cfg.model.base_model_name
    version = (cfg.dataset.get("version") or "V2").upper()
    min_grade = cfg.dataset.get("min_grade")
    hf_org = cfg.dataset.get("hf_organization")
    epochs = cfg.sft.get("num_train_epochs") or 2

    model_run_name = derive_model_run_name(
        base_model_name=base,
        version=version,
        min_grade=min_grade,
        use_peft=cfg.use_peft,
        num_train_epochs=epochs,
    )
    wandb_name = model_run_name
    output_dir = f"trained_models/{model_run_name}"
    dataset_cache_basename = sft_dataset_basename(_SFT_CACHE_TOKENIZER, version)
    hub_model_id = f"{hf_org}/{model_run_name}" if hf_org else None
    return wandb_name, dataset_cache_basename, output_dir, hub_model_id


def _compose_build_cfg(cfg: DictConfig) -> DictConfig:
    """Compose a ``build_sft_dataset`` config for the Llama cache.

    The Llama chat template has no ``enable_thinking`` toggle, so we strip the
    Qwen-flavoured ``chat_template_kwargs`` from the build config to avoid
    confusing :func:`transformers.PreTrainedTokenizer.apply_chat_template`.
    The raw input repo (full-raw vs full-v2) is picked by the builder from
    ``dataset.version``.
    """
    build_cfg = OmegaConf.load(_PREPROCESS_CFG_PATH)
    version = (cfg.dataset.get("version") or "V2").upper()
    OmegaConf.update(build_cfg, "dataset.version", version)
    OmegaConf.update(build_cfg, "tokenizer.name", _SFT_CACHE_TOKENIZER)
    OmegaConf.update(build_cfg, "tokenizer.chat_template_kwargs", {})
    return build_cfg


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

    derived_wandb, derived_cache_basename, derived_output, derived_hub_id = derive_run_names(cfg)
    if cfg.wandb.get("name") is None:
        OmegaConf.update(cfg, "wandb.name", derived_wandb)
    if cfg.sft.get("output_dir") is None:
        OmegaConf.update(cfg, "sft.output_dir", derived_output)
    if cfg.sft.get("hub_model_id") is None and derived_hub_id is not None:
        OmegaConf.update(cfg, "sft.hub_model_id", derived_hub_id)

    wandb_name = cfg.wandb.name
    output_dir = cfg.sft.output_dir
    hub_model_id = cfg.sft.get("hub_model_id")

    if cfg.test_mode:
        logger.info("Training in test mode...")
        wandb_name = wandb_name + "-test"
        output_dir = output_dir + "-test"
        if hub_model_id is not None:
            hub_model_id = hub_model_id + "-test"
        cfg.dataset.limit = 100

    if cfg.sft.push_to_hub and not hub_model_id:
        raise ValueError(
            "push_to_hub is true but sft.hub_model_id is not set. "
            "Set sft.hub_model_id, or dataset.hf_organization to auto-derive it."
        )
    if cfg.wandb.use:
        if not cfg.wandb.get("entity"):
            raise ValueError("use_wandb is true but wandb.entity is not set.")
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=wandb_name)
    else:
        logger.info("Wandb is disabled. Skipping wandb initialization...")

    model_name = cfg.model.base_model_name
    tokenizer = init_llama_tokenizer(model_name)
    logger.info(
        f"model_name: {model_name} | "
        f"dataset_cache: {derived_cache_basename} | "
        f"output_dir: {output_dir}"
    )

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
        hub_model_id=hub_model_id,
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

    # The cache is built with the Llama-3.1 tokenizer so rows are already in
    # Llama chat format — no runtime reformat needed.
    full_dataset = load_or_build_sft_dataset(
        base_name=derived_cache_basename,
        build_cfg=_compose_build_cfg(cfg),
        cache_root=SFT_CACHE_ROOT,
        limit=cfg.dataset.get("limit", -1),
        min_grade=cfg.dataset.get("min_grade"),
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
