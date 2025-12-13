'''
https://huggingface.co/openai/gpt-oss-20b/discussions/126
'''
from transformers import DataCollatorForLanguageModeling
import torch

class DataCollatorForAssistantOnlyLM(DataCollatorForLanguageModeling):
    """Custom data collator that only computes loss on assistant responses."""

    RESPONSE_TEMPLATE: str = "<|start|>assistant<|channel|>final<|message|>"
    response_template_ids: torch.Tensor

    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

        self.response_template_ids = tokenizer.encode(
            self.RESPONSE_TEMPLATE, add_special_tokens=False
        )

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # For each example in the batch, mask everything except assistant responses
        labels = batch["labels"].clone()

        for i, label in enumerate(labels):
            # Find where assistant responses start
            response_token_ids_start_idx = []

            # Search for the response template in the token sequence
            for idx in range(len(label) - len(self.response_template_ids) + 1):
                if (
                    label[idx : idx + len(self.response_template_ids)].tolist()
                    == self.response_template_ids
                ):
                    response_token_ids_start_idx.append(
                        idx + len(self.response_template_ids)
                    )

            # Mask all tokens except those after response templates
            if len(response_token_ids_start_idx) > 0:
                # Start with all tokens masked
                mask = torch.ones_like(label, dtype=torch.bool)

                # Unmask assistant response regions
                for start_idx in response_token_ids_start_idx:
                    # Find the next response template or end of sequence
                    next_idx = len(label)
                    for next_start in response_token_ids_start_idx:
                        if next_start > start_idx:
                            next_idx = next_start - len(self.response_template_ids)
                            break

                    mask[start_idx:next_idx] = False

                # Apply mask (set masked tokens to -100)
                labels[i] = torch.where(mask, -100, label)

        batch["labels"] = labels
        return batch
