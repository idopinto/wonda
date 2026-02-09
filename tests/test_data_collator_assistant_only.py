"""
Test script to validate DataCollatorForAssistantOnlyLM works correctly.

Run with: uv run python tests/test_data_collator_assistant_only.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from wonda.train.sft.data_collator_assistant_only import DataCollatorForAssistantOnlyLM


def clean_messages(messages):
    """Remove None 'thinking' fields that break the chat template."""
    cleaned = []
    for msg in messages:
        clean_msg = {"role": msg["role"], "content": msg["content"]}
        # Only include thinking if it's not None
        if msg.get("thinking") is not None:
            clean_msg["thinking"] = msg["thinking"]
        cleaned.append(clean_msg)
    return cleaned


def test_collator_with_real_dataset():
    """Test with actual training dataset format."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", token=True)
    collator = DataCollatorForAssistantOnlyLM(tokenizer)
    
    print("Loading dataset...")
    dataset = load_dataset("idopinto/invbench-training-uautomizer25-k1-gen-one", split="train")
    
    # Test on first 3 samples
    num_samples = min(3, len(dataset))
    
    for idx in range(num_samples):
        sample = dataset[idx]
        messages = sample["messages"]
        
        # Check for None values in messages
        print(f"\n--- Sample {idx} messages structure ---")
        for i, msg in enumerate(messages):
            thinking = msg.get('thinking')
            print(f"  Message {i}: role={msg.get('role')}, content_len={len(msg.get('content', '') or '')}, thinking={thinking}")
        
        # Clean messages (remove None thinking fields)
        messages = clean_messages(messages)
        
        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort="medium",
            )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            continue
        
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx + 1}")
        print(f"{'='*80}")
        
        # Show raw text (truncated)
        print(f"\n--- Raw text (first 800 chars) ---")
        print(text[:800] + "..." if len(text) > 800 else text)
        
        # Check if template exists in text
        template = collator.RESPONSE_TEMPLATE
        if template in text:
            print(f"\n✓ Template '{template}' found in text")
        else:
            print(f"\n✗ Template '{template}' NOT found in text!")
            print("Looking for similar patterns...")
            if "<|channel|>final" in text:
                print("  Found '<|channel|>final' - template might need adjustment")
            if "<|start|>assistant" in text:
                print("  Found '<|start|>assistant'")
            continue
        
        # Tokenize and run through collator
        tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        batch = collator([{"input_ids": tokenized["input_ids"][0]}])
        
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]
        
        total = len(labels)
        masked = (labels == -100).sum().item()
        unmasked = (labels != -100).sum().item()
        
        print(f"\n--- Masking Statistics ---")
        print(f"Total tokens: {total}")
        print(f"Masked (no loss): {masked} ({100*masked/total:.1f}%)")
        print(f"Unmasked (loss computed): {unmasked} ({100*unmasked/total:.1f}%)")
        
        if unmasked == 0:
            print("\n⚠️  WARNING: No tokens being trained on!")
        else:
            # Decode what's being trained on
            unmasked_ids = input_ids[labels != -100]
            trained_text = tokenizer.decode(unmasked_ids)
            
            print(f"\n--- Tokens being TRAINED ON (should be final answer only) ---")
            print(trained_text[:500] + "..." if len(trained_text) > 500 else trained_text)
            
            # Decode what's masked (first part)
            masked_ids = input_ids[labels == -100]
            masked_text = tokenizer.decode(masked_ids)
            
            print(f"\n--- Tokens MASKED (first 300 chars) ---")
            print(masked_text[:300] + "...")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Template used: {collator.RESPONSE_TEMPLATE}")
    print(f"Template token IDs: {collator.response_template_ids}")
    print("\nIf 'Unmasked' is 0%, the template doesn't match your data format.")
    print("If 'Unmasked' is ~100%, the template is not being found.")
    print("Expected: ~10-30% unmasked (final answer only), ~70-90% masked (reasoning + prompts)")


def test_template_matching():
    """Check how many samples contain the response template."""
    print("\n" + "="*80)
    print("TEMPLATE MATCHING TEST")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", token=True)
    collator = DataCollatorForAssistantOnlyLM(tokenizer)
    
    dataset = load_dataset("idopinto/invbench-training-uautomizer25-k1-gen-one", split="train")
    
    # First, check what fields are available
    print(f"Dataset fields: {dataset.column_names}")
    print(f"First sample keys: {dataset[0].keys()}")
    
    template = collator.RESPONSE_TEMPLATE
    found_count = 0
    check_count = min(20, len(dataset))
    
    for i in range(check_count):
        sample = dataset[i]
        messages = clean_messages(sample["messages"])
        
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort="medium",
            )
        except Exception:
            continue
                
        if template in text:
            found_count += 1
    
    print(f"Template: {template}")
    print(f"Found in {found_count}/{check_count} samples ({100*found_count/check_count:.0f}%)")
    
    if found_count == 0:
        print("\n⚠️  Template not found in any samples!")
        print("Checking alternative patterns in first sample...")
        
        sample = dataset[0]
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        
        patterns = [
            "<|channel|>final",
            "<|start|>assistant",
            "<|message|>assert",
            "final<|message|>",
        ]
        for pattern in patterns:
            if pattern in text:
                print(f"  ✓ Found: '{pattern}'")
            else:
                print(f"  ✗ Not found: '{pattern}'")
    else:
        print(f"\n✅ Template matching works!")


if __name__ == "__main__":
    test_template_matching()
    test_collator_with_real_dataset()
