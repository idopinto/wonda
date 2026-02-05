import logging
from pathlib import Path
import random
from typing import List

import hydra
import torch
import wandb
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.train.sft.data_utils import load_training_data, split_dataset, log_dataset_stats
from src.preprocess.program import Program as AstProgram
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_tokenizer(model_name: str)->AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer {tokenizer.name_or_path}...")
    return tokenizer

def load_model(model_name: str, model_kwargs: dict)->AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info(f"Loaded model\n{model}\n")
    return model

def run_inference(sample: dict, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, model_kwargs: dict, prompts: dict)->None:
    enable_thinking = model_kwargs["enable_thinking"]
    system_prompt = prompts["per_marker_system_prompt"]
    per_marker_user_prompt_template = prompts["per_marker_user_prompt_template"]
    ast_program = AstProgram().from_code(sample["original_program"])
    ast_program.process(print_ast=False)
    user_prompt = per_marker_user_prompt_template.format(
        program=ast_program.llm_code,
        target_marker=sample["invariants"][0]["marker"]
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages,add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=False)
    # print(f"Input text: {input_text}")
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    # print(f"Input ids: {input_ids}")
    # print(f"Sampling params: {model_kwargs['inference']['sampling_params']}")
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **model_kwargs["inference"]["sampling_params"])
    response = tokenizer.batch_decode(output_ids)[0]
    print("=== Inference Test ===")
    print(response)
    print("="*50)
    return response

def print_trainable_parameters(model):
    """Print trainable vs total parameters to verify LoRA is working."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / all_param
    logger.info(f"Trainable params: {trainable_params:,} | All params: {all_param:,} | Trainable %: {pct:.4f}%")
    return trainable_params, all_param

def train(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, training_args: SFTConfig, dataset: Dataset, validation_dataset: Dataset)->SFTTrainer:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

"""
configs:
- qwen3_0.6b_nt_v0.yaml - raw ground truth invariants for each marker   
 - uv run -m src.train.sft.train_qwen --config-name=qwen3_0.6b_nt_v0 test_mode=true
- qwen3_0.6b_nt_v1.yaml - pretty ground truth invariants for each location (no casting and minimal parenthesization)
 - uv run -m src.train.sft.train_qwen --config-name=qwen3_0.6b_nt_v1    
- qwen3_0.6b_nt_v2.yaml - simplified ground truth invariants for each marker
 - uv run -m src.train.sft.train_qwen --config-name=qwen3_0.6b_nt_v2 test_mode=true wandb.use_wandb=false
- qwen3_8b_nt_v0.yaml - raw ground truth invariants for each marker
 - uv run -m src.train.sft.train_qwen --config-name=qwen3_8b_nt_v0 test_mode=true
- qwen3_8b_nt_v1.yaml - pretty ground truth invariants for each marker (no casting and minimal parenthesization)
 - uv run -m src.train.sft.train_qwen --config-name=qwen3_8b_nt_v1
"""
@hydra.main(version_base=None, config_path="../../../configs/train/models", config_name="qwen3_0.6b_nt_v0")
def main(cfg: DictConfig):
    print("="*50)
    print(OmegaConf.to_yaml(cfg))
    print("="*50)

    min_grade_str = "."+str(cfg.dataset.min_grade) if cfg.dataset.min_grade else ""
    wandb_name = cfg.wandb.name + min_grade_str
    output_dir = cfg.sft.output_dir + min_grade_str
    hf_upload_repo = (cfg.dataset.hf_upload_repo + min_grade_str) if cfg.dataset.hf_upload_repo else None
    if cfg.test_mode:
        logger.info("Training in test mode...")
        wandb_name  = wandb_name + "-test"
        hf_upload_repo = (hf_upload_repo + "-test") if hf_upload_repo else None
        output_dir = output_dir + "-test"
        cfg.dataset.limit = 100
    if cfg.wandb.use_wandb:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=wandb_name)
    else:
        logger.info("Wandb is disabled. Skipping wandb initialization...")

    model_name = cfg.model.base_model_name
    model_init_kwargs = cfg.model.init_kwargs_train
    
    raw_repo_name = cfg.dataset.repo_name
    tokenizer = init_tokenizer(model_name)
    logger.info(f"model_name: {model_name} | raw_repo_name: {raw_repo_name} | hf_upload_repo: {hf_upload_repo} | output_dir: {output_dir}")
    force_reprocess = cfg.dataset.force_reprocess
    logger.info(f"force_reprocess training data: {force_reprocess}")

    # Build SFT training args from YAML
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
    # logger.info(f"Training args: {training_args}")
    # Build model kwargs from YAML
    model_init_kwargs = dict(
        attn_implementation=cfg.model.init_kwargs_train.attn_implementation,
        dtype=torch.bfloat16,
        use_cache=cfg.model.init_kwargs_train.use_cache,
        device_map=cfg.model.init_kwargs_train.device_map,
    )
    # logger.info(f"Model init kwargs: {model_init_kwargs}")
    full_dataset = load_training_data(dataset_name=raw_repo_name,
                        tokenizer=tokenizer,
                        system_prompt=cfg.prompts.per_marker_system_prompt,
                        user_prompt_template=cfg.prompts.per_marker_user_prompt_template,
                        limit=cfg.dataset.limit,
                        inv_mode=cfg.dataset.inv_mode, 
                        split=cfg.dataset.split,
                        max_length=cfg.sft.max_length,
                        json_path=cfg.dataset.json_path if cfg.dataset.json_path else None,
                        min_grade=cfg.dataset.min_grade if cfg.dataset.min_grade else 1,
                        output_dir=cfg.sft.output_dir,
                        hf_upload_repo=hf_upload_repo,
                        force_reprocess=force_reprocess)
    # Log stats for full dataset before split
    # log_dataset_stats(full_dataset, tokenizer, name="Full")
    
    train_dataset, validation_dataset = split_dataset(full_dataset, split_ratio=cfg.dataset.split_ratio)

    # Log stats for each split
    # log_dataset_stats(train_dataset, tokenizer, name="Train")
    # log_dataset_stats(validation_dataset, tokenizer, name="Validation")

    print("="*50)
    print("Train dataset sample:")
    for key, value in train_dataset[0].items():
        logger.info(f"{key}: {value}")
    print("="*50)
    print("Validation dataset sample:")
    for key, value in validation_dataset[0].items():
        logger.info(f"{key}: {value}")
    print("="*50)
    model = load_model(model_name, model_init_kwargs)

    if cfg.use_peft:
        # Convert OmegaConf ListConfig to regular Python list for JSON serialization
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

# Example usage:   
# uv run -m src.train.sft.train
# Config Location:
# configs/train/config.yaml
# sbatch command:
