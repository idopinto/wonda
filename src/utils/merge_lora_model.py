"""
Utility to merge a LoRA adapter with its base model and upload to HuggingFace Hub.

Usage:
    uv run -m src.utils.merge_lora_model \
        --base-model "Qwen/Qwen3-8B" \
        --adapter-path "idopinto/my-lora-adapter" \
        --repo-id "idopinto/my-merged-model"
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_and_upload(
    base_model_name: str,
    adapter_path: str,
    repo_id: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    save_locally: bool = False,
    local_dir: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload merged model",
) -> str:
    """
    Merge a LoRA adapter with its base model and upload to HuggingFace Hub.

    Args:
        base_model_name: HuggingFace model ID for the base model (e.g., "Qwen/Qwen3-8B")
        adapter_path: Path to the LoRA adapter (local path or HF repo ID)
        repo_id: Target HuggingFace repo ID for the merged model (e.g., "username/merged-model")
        torch_dtype: Data type for model weights (default: bfloat16)
        device_map: Device placement strategy (default: "auto")
        save_locally: If True, also save merged model to local directory
        local_dir: Local directory to save merged model (required if save_locally=True)
        private: If True, create a private repository
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded model on HuggingFace Hub
    """
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    # Verify merge
    total_params = sum(p.numel() for p in merged_model.parameters())
    logger.info(f"Merged model total parameters: {total_params:,}")

    # Save locally if requested
    if save_locally:
        if local_dir is None:
            local_dir = f"./merged-{Path(adapter_path).name}"
        logger.info(f"Saving merged model locally to: {local_dir}")
        merged_model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

    # Upload to HuggingFace Hub
    logger.info(f"Uploading merged model to: {repo_id}")

    # Create repo if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create repo (may already exist): {e}")

    # Push model and tokenizer
    merged_model.push_to_hub(
        repo_id,
        commit_message=commit_message,
        private=private,
    )
    tokenizer.push_to_hub(
        repo_id,
        commit_message=commit_message,
        private=private,
    )

    hub_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Successfully uploaded merged model to: {hub_url}")

    return hub_url


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model and upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HuggingFace model ID for the base model",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter (local or HF repo ID)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target HuggingFace repo ID for merged model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device placement strategy",
    )
    parser.add_argument(
        "--save-locally",
        action="store_true",
        help="Also save merged model locally",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory to save merged model",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload merged model",
        help="Commit message for the upload",
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    merge_and_upload(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        repo_id=args.repo_id,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        save_locally=args.save_locally,
        local_dir=args.local_dir,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
