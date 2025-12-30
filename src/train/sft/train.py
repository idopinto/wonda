
import os

import torch
from pathlib import Path

# Fix memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from typing import List
import logging
from transformers import Mxfp4Config
import wandb
from src.train.sft.data_collator_assistant_only import DataCollatorForAssistantOnlyLM
import hydra

from omegaconf import DictConfig, OmegaConf 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(dataset_name: str, limit: int = -1)->Dataset:
    dataset = load_dataset(dataset_name, split="train")
    if limit > 0:
        dataset = dataset.select(range(limit))
    logger.info(f"Loaded dataset {dataset_name}...")
    logger.info(f"Dataset Size: {len(dataset)}")

    return dataset

def init_tokenizer(model_name: str)->AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer {tokenizer.name_or_path}...")
    return tokenizer

def preview_conversation(tokenizer: AutoTokenizer, messages: List[dict])->None:
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    print(conversation)

def load_model(model_name: str, model_kwargs: dict)->AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info(f"Loaded model\n{model}\n...with kwargs\n{model_kwargs}\n")
    return model

def run_inference(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, messages: List[dict], max_new_tokens: int = 512)->None:
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    messages = [{
        "role": "system",
        "content": messages[0]["content"],
    },
    {
        "role": "user",
        "content": messages[1]["content"],
    }
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(output_ids)[0]
    print("=== Inference Test ===")
    print(response)
    print("="*50)
    return response

def apply_lora(model: AutoModelForCausalLM, lora_config: LoraConfig)->AutoModelForCausalLM:
    peft_model = get_peft_model(model, lora_config)
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    peft_model.print_trainable_parameters()
    return peft_model


def save_model(trainer: SFTTrainer, output_dir: str)->None:
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    trainer.save_model(output_dir)
    logger.info(f"Saved model to {output_dir}...")
    try:
        trainer.push_to_hub(f"idopinto/{output_dir}")
        logger.info("Pushed model to Hugging Face hub...")

    except Exception as e:
        logger.error(f"Error pushing model to Hugging Face hub: {e}")

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

def train(tokenizer: AutoTokenizer, peft_model: AutoModelForCausalLM, training_args: SFTConfig, dataset: Dataset, validation_dataset: Dataset)->SFTTrainer:
    # Verify LoRA is applied correctly
    print_trainable_parameters(peft_model)
    
    # Create custom data collator
    data_collator = DataCollatorForAssistantOnlyLM(tokenizer)
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    save_model(trainer, training_args.output_dir)


def evaluate(output_dir: str, model_name: str, messages: List[dict], model_eval_config: dict, base_kwargs: dict)->None:
    """ Load the merged model and evaluate (only on main process)"""
    # if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        # return
    logger.info(f"Evaluating model from {output_dir}...")
    try:
        tokenizer = init_tokenizer(model_name)
        model = load_model(model_name, base_kwargs)
        model = PeftModel.from_pretrained(model, output_dir)
        model = model.merge_and_unload()
        messages = [
            {
                "role": "system",
                "content": messages[0]["content"],
            },
            {
                "role": "user",
                "content": messages[1]["content"],
            },
        ]
        
        # Fix: Convert OmegaConf to dict and extract sampling_params
        model_eval_config = OmegaConf.to_container(model_eval_config, resolve=True)
        
        # Extract sampling params (nested under 'sampling_params' key)
        generate_kwargs = model_eval_config.get("sampling_params", {})
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
        
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            # from transformers import TextStreamer
            # output_ids = model.generate(input_ids, **generate_kwargs, streamer=TextStreamer(tokenizer))
            output_ids = model.generate(input_ids, **generate_kwargs)
            # response = tokenizer.batch_decode(output_ids)[0]
            response = tokenizer.batch_decode(output_ids)[0]
            print(response)
        # print("=== Evaluation Results ===")
        # print(f"{response}")
        print(f"Response length: {output_ids.shape[1]} tokens")
        # print("="*50)
        # return response
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")


def split_dataset(dataset: Dataset, split_ratio: float = 0.8)->(Dataset, Dataset):
    """Split dataset into train and validation sets."""
    train_size = int(len(dataset) * split_ratio)
    train_dataset = dataset.select(range(train_size))
    validation_dataset = dataset.select(range(train_size, len(dataset)))
    logger.info(f"Split dataset into train and validation sets. Train size: {len(train_dataset)}, Validation size: {len(validation_dataset)}")
    return train_dataset, validation_dataset

@hydra.main(version_base=None, config_path="../../../configs/train", config_name="config")
def main(cfg: DictConfig):
    print("="*50)
    print(OmegaConf.to_yaml(cfg))
    print("="*50)
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity)

    # Extract top-level config
    model_name = cfg.model.base_model_name
    hf_dataset = cfg.dataset.name
    output_dir = cfg.sft.output_dir
    logger.info(f"model_name: {model_name} | hf_dataset: {hf_dataset} | output_dir: {output_dir}")
    # Build LoRA config from YAML (convert ListConfig to list for JSON serialization)
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=cfg.lora.target_modules,
        target_parameters=list(cfg.lora.target_parameters),
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )
    logger.info(f"LoRA config: {lora_config}")
    # Build SFT training args from YAML
    training_args = SFTConfig(
        learning_rate=cfg.sft.learning_rate,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        num_train_epochs=cfg.sft.num_train_epochs,
        logging_steps=cfg.sft.logging_steps,
        eval_strategy=cfg.sft.eval_strategy,
        eval_steps=cfg.sft.eval_steps,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        max_length=cfg.sft.max_length,
        warmup_ratio=cfg.sft.warmup_ratio,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.sft.lr_scheduler_kwargs,
        output_dir=cfg.sft.output_dir,
        report_to=cfg.sft.report_to,
        push_to_hub=cfg.sft.push_to_hub,
    )
    logger.info(f"Training args: {training_args}")
    # Build model kwargs from YAML
    model_init_kwargs = dict(
        attn_implementation=cfg.model.init_kwargs_train.attn_implementation,
        dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=cfg.model.init_kwargs_train.use_cache,
        device_map=cfg.model.init_kwargs_train.device_map,
    )
    
    dataset = load_data(hf_dataset, limit=cfg.dataset.limit)
    train_dataset, validation_dataset = split_dataset(dataset, split_ratio=0.8)
    tokenizer = init_tokenizer(model_name)
    model = load_model(model_name, model_init_kwargs)
    peft_model = apply_lora(model, lora_config)
    train(tokenizer, peft_model, training_args, train_dataset, validation_dataset)
    evaluate(output_dir, model_name, validation_dataset[0]["messages"], cfg.model.inference, model_init_kwargs)
    
    wandb.finish()
    logger.info("DONE.")

if __name__ == "__main__":
    main()

# Example usage:   
# uv run -m src.train.sft.train
# Config Location:
# configs/train/config.yaml
# sbatch command:
# sbatch --job-name=sft-train --output=logs/sft-train-%j.log --error=logs/sft-train-%j.log --time=12:00:00 --partition=gpu --gres=gpu:1 --cpus-per-task=10 src/train/sft/train.py