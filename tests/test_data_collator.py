# test_data_collator.py
from transformers import AutoTokenizer
from src.train.data_collator_assistant_only import DataCollatorForAssistantOnlyLM

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create collator
collator = DataCollatorForAssistantOnlyLM(tokenizer)

# Test 1: Check response template tokenization
print("=== Test 1: Response Template ===")
print(f"Template: {collator.RESPONSE_TEMPLATE}")
print(f"Template IDs: {collator.response_template_ids}")
print(f"Decoded back: {tokenizer.decode(collator.response_template_ids)}")
print()

# Test 2: Create a sample conversation
print("=== Test 2: Masking Test ===")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "assert(2+2==4);", "thinking": 'I think the answer is 4.'},
]

# Tokenize
text = tokenizer.apply_chat_template(messages, tokenize=False)
print(f"Full text:\n{text}\n")

encoded = tokenizer(text, return_tensors="pt")
input_ids = encoded["input_ids"][0]

# Run collator
examples = [{"input_ids": input_ids.tolist()}]
batch = collator.torch_call(examples)

# Analyze results
labels = batch["labels"][0]
print(f"Input length: {len(input_ids)}")
print(f"Labels length: {len(labels)}")

# Show which tokens are masked vs. kept
print("\n=== Token-by-token analysis ===")
for i, (tok_id, label) in enumerate(zip(input_ids.tolist(), labels.tolist())):
    token_str = tokenizer.decode([tok_id])
    status = "MASKED (-100)" if label == -100 else f"KEPT (label={label})"
    print(f"{i:4d}: {repr(token_str):30s} | {status}")

# Summary
kept_count = (labels != -100).sum().item()
masked_count = (labels == -100).sum().item()
print("\n=== Summary ===")
print(f"Total tokens: {len(labels)}")
print(f"Kept (assistant): {kept_count}")
print(f"Masked (system/user): {masked_count}")

# Decode only the kept tokens
kept_tokens = [tok_id for tok_id, label in zip(input_ids.tolist(), labels.tolist()) if label != -100]
print(f"\nKept tokens decoded:\n{tokenizer.decode(kept_tokens)}")