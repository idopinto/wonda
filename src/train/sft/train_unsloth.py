"""Fine-tuning script using Unsloth for efficient LLM training."""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import wandb
from datasets import Dataset, load_dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt


@dataclass
class ModelConfig:
    """Configuration for model loading and LoRA setup."""
    model_name: str = "unsloth/gpt-oss-20b"
    max_seq_length: int = 1024
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    full_finetuning: bool = False
    hf_token: Optional[str] = None
    
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407


@dataclass
class TrainingConfig:
    """Configuration for SFT training."""
    dataset_name: str = "HuggingFaceH4/Multilingual-Thinking"
    dataset_split: str = "train"
    reasoning_effort: str = "medium"
    
    # Training hyperparameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 30
    num_train_epochs: Optional[int] = None
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    
    # Wandb logging
    report_to: str = "wandb"
    wandb_project: str = "gen-inv-unsloth"
    wandb_entity: str = "ip-ai"
    run_name: Optional[str] = None


def load_model(config: ModelConfig):
    """Load and configure the model with LoRA adapters."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        dtype=config.dtype,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=config.full_finetuning,
        token=config.hf_token,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.random_state,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer


def prepare_dataset(tokenizer, config: TrainingConfig) -> Dataset:
    """Load and format the training dataset."""
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    dataset = standardize_sharegpt(dataset)
    
    def format_prompts(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort=config.reasoning_effort,
            )
            for convo in examples["messages"]
        ]
        return {"text": texts}
    
    dataset = dataset.map(format_prompts, batched=True)
    return dataset


def create_trainer(model, tokenizer, dataset: Dataset, config: TrainingConfig) -> SFTTrainer:
    """Create and configure the SFT trainer."""
    # Get the actual EOS token from tokenizer
    eos_token = tokenizer.eos_token
    print(f"Using EOS token: {repr(eos_token)}")
    
    sft_config = SFTConfig(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=config.output_dir,
        report_to=config.report_to,
        run_name=config.run_name,
        dataset_text_field="text",
        eos_token=eos_token,
    )
    
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )


def generate_sample(model, tokenizer, messages: List[dict], reasoning_effort: str = "medium"):
    """Generate a sample response for testing."""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning_effort,
    ).to(model.device)
    
    streamer = TextStreamer(tokenizer)
    model.generate(**inputs, max_new_tokens=2048, streamer=streamer)


def push_to_hub(model, tokenizer, repo_name: str, token: Optional[str] = None):
    """Push the merged model to Hugging Face Hub."""
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    model.push_to_hub_merged(repo_name, tokenizer=tokenizer, token=token)


def main():
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Initialize wandb
    wandb.init(
        project=training_config.wandb_project,
        entity=training_config.wandb_entity,
        name=training_config.run_name,
        config={
            "model": model_config.__dict__,
            "training": training_config.__dict__,
        },
    )
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_config)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(tokenizer, training_config)
    print(f"Sample formatted text:\n{dataset[0]['text'][:500]}...")
    
    # Create trainer and train
    print("Starting training...")
    trainer = create_trainer(model, tokenizer, dataset, training_config)
    trainer.train()
    
    # # Test generation
    # print("\nTesting generation...")
    # test_messages = [
    #     {"role": "system", "content": "You are a helpful assistant that can solve mathematical problems."},
    #     {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
    # ]
    # generate_sample(model, tokenizer, test_messages)
    
    # Push to hub
    repo_name = "idopinto/gpt-oss-20b-unsloth-sft-test"
    print(f"\nPushing model to {repo_name}...")
    push_to_hub(model, tokenizer, repo_name)
    
    # Finish wandb run
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
