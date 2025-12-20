from src.train.rl.rewards import strategy_succeeds, function_works, no_cheating
import torch

from unsloth import FastLanguageModel
from src.train.rl.env import GameBoard
from transformers import TextStreamer
from datasets import Dataset
import wandb

wandb.init(project="gpt-oss-20b-rl-finetune", entity="ip-ai")
max_seq_length = 768  # Increase if your task needs longer outputs
lora_rank = 4  # Higher rank → better but more VRAM/compute
prompt = """
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "W", "A", "S", "D" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "W" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
""".strip()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",  # or unsloth/gpt-oss-20b-BF16 on H100
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for 16‑bit
    offload_embedding=True,  # saves ~1GB VRAM
)

# text = tokenizer.apply_chat_template(
#     [{"role": "user", "content": prompt}],
#     tokenize=False,
#     add_generation_prompt=True,
#     reasoning_effort="low",
# )

# _ = model.generate(
#     **tokenizer(text, return_tensors="pt").to("cuda"),
#     temperature=1.0,
#     max_new_tokens=512,
#     streamer=TextStreamer(tokenizer, skip_prompt=False),
# )

dataset = Dataset.from_list(
    [
        {
            "prompt": [{"role": "user", "content": prompt.strip()}],
            "answer": 0,
            "reasoning_effort": "low",
        }
    ]
    * 1000
)
maximum_length = len(
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}], add_generation_prompt=True
    )
)
print(maximum_length)
print(dataset[0])

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",  # big memory saver
    random_state=3407,
)
print(model)
print(model.print_trainable_parameters())

max_prompt_length = maximum_length + 1  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    temperature=1.0,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=2,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=1000,
    save_steps=100,
    report_to="wandb",  # Can use Weights & Biases, TrackIO
    output_dir="outputs",
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)
# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        function_works,
        no_cheating,
        strategy_succeeds,
    ],
    args=training_args,
    train_dataset=dataset,
    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)
trainer.train()
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
    reasoning_effort="low",
)

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature=1.0,
    max_new_tokens=1024,
    streamer=TextStreamer(tokenizer, skip_prompt=False),
)

# Merge and push to hub in mxfp4 4bit format
if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method="mxfp4")
if False:
    model.push_to_hub_merged(
        "idopinto/gpt-oss-20b-rl-finetune",
        tokenizer,
        token="hf...",
        save_method="mxfp4",
    )

# Merge and push to hub in 16bit
if False:
    model.save_pretrained_merged(
        "finetuned_model", tokenizer, save_method="merged_16bit"
    )
if False:  # Pushing to HF Hub
    model.push_to_hub_merged(
        "idopinto/gpt-oss-20b-rl-finetune",
        tokenizer,
        save_method="merged_16bit",
        token="",
    )
