"""
Count the total number of tokens in an SFT-ready dataset using the Qwen tokenizer.

Usage (from repo root):
    uv run scripts/count_dataset_tokens.py [--json_path PATH] [--hf_repo REPO]
                                           [--min_grade N] [--model MODEL_ID]

Examples:
    # V2 from local SFT cache
    uv run scripts/count_dataset_tokens.py \
        --json_path data/train/sft-ready/Wonda-Training-Dataset-Qwen3-V2/text.json \
        --min_grade 2 \
        --model Qwen/Qwen3-8B

    # Legacy v2-g2 from HuggingFace (pre-rename release)
    uv run scripts/count_dataset_tokens.py \
        --hf_repo idopinto/wonda-qwen-nt-sft-v2-g2 \
        --min_grade 2 \
        --model Qwen/Qwen3-8B
"""
import argparse
import logging

from transformers import AutoTokenizer
from wonda.train.data_utils import load_sft_dataset
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_JSON = "data/train/sft-ready/Wonda-Training-Dataset-Qwen3-V2/text.json"


def count_tokens(args: argparse.Namespace) -> None:
    json_path = args.json_path if not args.hf_repo else None
    hf_repo = args.hf_repo or None

    dataset = load_sft_dataset(
        hf_repo=hf_repo,
        json_path=json_path,
        limit=-1,
        min_grade=args.min_grade,
    )

    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    logger.info(f"Counting tokens for {len(dataset)} samples …")
    token_lengths = [len(tokenizer.encode(sample["text"])) for sample in tqdm(dataset)]

    total = sum(token_lengths)
    mean = total / len(token_lengths)
    print()
    print("=" * 50)
    print(f"Dataset:       {hf_repo or json_path}")
    print(f"min_grade:     {args.min_grade}")
    print(f"Tokenizer:     {args.model}")
    print(f"Samples:       {len(token_lengths):,}")
    print(f"Total tokens:  {total:,}")
    print(f"Mean tokens:   {mean:.1f}")
    print(f"Min tokens:    {min(token_lengths):,}")
    print(f"Max tokens:    {max(token_lengths):,}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count tokens in an SFT-ready dataset.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--json_path", default=DEFAULT_JSON, help="Local JSON dataset path")
    source.add_argument("--hf_repo", default=None, help="HuggingFace dataset repo id")
    parser.add_argument(
        "--min_grade", type=int, default=2,
        help="Minimum quality_grade to include (default: 2 = v2.2)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model id for tokenizer (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    count_tokens(args)


if __name__ == "__main__":
    main()
