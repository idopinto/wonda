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

from src.train.sft.data_utils import load_training_data, split_dataset
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
- qwen_0_6b_v0.yaml - raw ground truth invariants for each location
 - uv run -m src.train.sft.train_qwen --config-name=qwen_0_6b_v0 test_mode=true
- qwen_0_6b_v1.yaml - pretty ground truth invariants for each location (no casting and minimal parenthesization)
 - uv run -m src.train.sft.train_qwen --config-name=qwen_0_6b_v1    
- qwen_1_7b_v0.yaml - raw ground truth invariants for each location
 - uv run -m src.train.sft.train_qwen --config-name=qwen_1_7b_v0
- qwen_1_7b_v1.yaml - pretty ground truth invariants for each location (no casting and minimal parenthesization)
 - uv run -m src.train.sft.train_qwen --config-name=qwen_1_7b_v1
"""
@hydra.main(version_base=None, config_path="../../../configs/train/qwen", config_name="qwen_0_6b_v0")
def main(cfg: DictConfig):
    print("="*50)
    print(OmegaConf.to_yaml(cfg))
    print("="*50)
    if cfg.test_mode:
        logger.info("Training in test mode...")
        cfg.wandb.name = cfg.wandb.name + "-test"
        cfg.sft.output_dir = cfg.sft.output_dir + "-test"
        cfg.dataset.limit = 100

    if cfg.wandb.use_wandb:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.wandb.name)
    else:
        logger.info("Wandb is disabled. Skipping wandb initialization...")

    # Extract top-level config
    tokenizer = init_tokenizer(cfg.model.base_model_name)
    logger.info(f"model_name: { cfg.model.base_model_name} | repo_name: { cfg.dataset.repo_name} | output_dir: {cfg.sft.output_dir}")

    # Build SFT training args from YAML
    training_args = SFTConfig(
        output_dir=cfg.sft.output_dir,
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
    full_dataset = load_training_data(dataset_name=cfg.dataset.repo_name,
                        tokenizer=tokenizer,
                        system_prompt=cfg.prompts.per_marker_system_prompt,
                        user_prompt_template=cfg.prompts.per_marker_user_prompt_template,
                        limit=cfg.dataset.limit,
                        inv_mode=cfg.dataset.inv_mode, 
                        split=cfg.dataset.split,
                        max_length=cfg.sft.max_length)
    train_dataset, validation_dataset = split_dataset(full_dataset, split_ratio=cfg.dataset.split_ratio)
    model = load_model(cfg.model.base_model_name, model_init_kwargs)

    if cfg.use_peft:
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=cfg.lora.target_modules,
            target_parameters=list(cfg.lora.target_parameters),
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )
        # logger.info(f"LoRA config: {lora_config}")
        model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    # raw_dataset = load_dataset(cfg.dataset.repo_name, split=cfg.dataset.split)
    # random_index = random.randint(0, len(raw_dataset)-1)
    # sample = raw_dataset[random_index]
    # run_inference(sample, tokenizer, model, model_kwargs=cfg.model, prompts=cfg.prompts)
    train(model, tokenizer, training_args, train_dataset, validation_dataset)
    # run_inference(sample, tokenizer, model, model_kwargs=cfg.model, prompts=cfg.prompts)
    
    wandb.finish()
    logger.info("DONE.")

if __name__ == "__main__":
    main()

# Example usage:   
# uv run -m src.train.sft.train
# Config Location:
# configs/train/config.yaml
# sbatch command:
