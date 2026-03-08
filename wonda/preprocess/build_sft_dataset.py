#!/usr/bin/env python3
"""Build SFT-ready training dataset from raw or v2 datasets.

Converts per-marker invariant data into chat-formatted SFT samples. Run as a
preprocess step so training only loads pre-built datasets.

Usage:
    uv run -m wonda.preprocess.build_sft_dataset --config-name=build_sft_dataset
    uv run -m wonda.preprocess.build_sft_dataset dataset.version=v2 dataset.min_grade=3
    uv run -m wonda.preprocess.build_sft_dataset dataset.input_repo=idopinto/wonda-train-dataset-full-raw dataset.split=full dataset.version=v0
"""

import logging
from pathlib import Path

import hydra
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from wonda.core.ast_program import AstProgram
from wonda.preprocess.gt_invariant_normalization import normalize_invariant
from wonda.preprocess.gt_invariant_simplification import is_degenerate_invariant
from wonda.preprocess.sft_plots import (
    plot_invariant_token_distribution,
    plot_sft_dataset_stats,
    plot_token_distribution,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _apply_chat_template(tokenizer, conversation, chat_template_kwargs):
    """Apply chat template with model-agnostic extra kwargs (e.g. enable_thinking for Qwen3)."""
    kwargs = dict(chat_template_kwargs) if chat_template_kwargs else {}
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    )


def build_sft_dataset_v2(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt_template: str,
    chat_template_kwargs: dict,
    max_length: int = 1024,
    min_grade: int = 2,
    version: str = "v2",
    output_dir: str | None = None,
    hf_repo_name: str | None = None,
) -> Dataset:
    """Build SFT dataset from v2 format (gt_invariant per sample)."""
    logger.info(f"Creating SFT train data: build_sft_dataset_v2 for version={version}, min_grade={min_grade}")
    out_parts = []
    if output_dir:
        dataset_name = Path(output_dir).name
        out_parts.append(f"disk: {Path(output_dir) / f'{dataset_name}.json'} (folder + plots)")
    if hf_repo_name:
        out_parts.append(f"HF: {hf_repo_name}")
    if out_parts:
        logger.info(f"Dataset will be written at the end to: {' and '.join(out_parts)}")
    if output_dir:
        stats_plot_path = Path(output_dir) / "dataset_stats.png"
    else:
        stats_plot_path = None

    samples = []
    kept_grades = []
    invariant_token_lengths = []
    max_tokens = 0
    skipped_baseline = skipped_grade = skipped_no_invariant = skipped_length = skipped_degenerate = 0

    for sample in tqdm(raw_dataset):
        if sample.get("baseline_decision") != "TRUE":
            skipped_baseline += 1
            continue
        if not sample.get("gt_invariant"):
            skipped_no_invariant += 1
            continue
        invariant = sample["gt_invariant"]
        if invariant.get("quality_grade", -1) < min_grade:
            skipped_grade += 1
            continue
        if is_degenerate_invariant(invariant.get("target_invariant", "")):
            skipped_degenerate += 1
            continue
        program_str = sample["original_program"]
        ast_program = AstProgram().from_code(program_str)
        ast_program.process(print_ast=False)
        user_prompt = user_prompt_template.format(
            program=ast_program.llm_code,
            target_marker=invariant.get("marker", ""),
        )
        content = invariant.get("target_invariant", "")
            
        answer = f'```json\n{{"marker":"{invariant.get("marker", "")}","content":"{content}"}}\n```'
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ]
        text = _apply_chat_template(tokenizer, conversation, chat_template_kwargs)
        if len(tokenizer.encode(text)) > max_length - 1:
            skipped_length += 1
            logger.debug(
                f"Skipping sample {sample.get('file', '?')} token length > {max_length - 1}"
            )
            continue
        inv_token_len = len(tokenizer.encode(content))
        samples.append({
            "text": text,
            "quality_grade": invariant.get("quality_grade", -1),
            "speedup": invariant.get("speedup", 0.0),
        })
        kept_grades.append(invariant.get("quality_grade", -1))
        invariant_token_lengths.append(inv_token_len)
        max_tokens = max(max_tokens, len(tokenizer.encode(text)))

    logger.info(
        f"Filtering stats: skipped_baseline={skipped_baseline}, skipped_grade={skipped_grade}, "
        f"skipped_no_invariant={skipped_no_invariant}, skipped_length={skipped_length}, "
        f"skipped_degenerate={skipped_degenerate}, kept={len(samples)}"
    )
    logger.info(f"Max tokens: {max_tokens}")

    final_dataset = Dataset.from_list(samples)

    if len(samples) > 0 and stats_plot_path:
        plot_sft_dataset_stats(
            final_dataset=final_dataset,
            tokenizer=tokenizer,
            kept_grades=kept_grades,
            invariant_token_lengths=invariant_token_lengths,
            min_grade=min_grade,
            name=f"{version}.g{min_grade}",
            plot_path=str(stats_plot_path),
        )

    if hf_repo_name and len(samples) > 0:
        _upload_dataset_to_hf(final_dataset, hf_repo_name)

    return final_dataset


def build_sft_dataset_v0_v1(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt_template: str,
    chat_template_kwargs: dict,
    max_length: int = 1024,
    version: str = "v0",
    output_dir: str | None = None,
) -> Dataset:
    """Build SFT dataset from v0/v1 format (invariants list per sample). version: v0 (raw) or v1 (normalized)."""
    logger.info(f"Creating SFT train data: build_sft_dataset_v0_v1 for version={version}")
    if output_dir:
        token_plot_path = Path(output_dir) / f"token_length_distribution_{version}.png"
        inv_token_plot_path = Path(output_dir) / f"invariant_token_length_distribution_{version}.png"
    else:
        token_plot_path = inv_token_plot_path = None

    samples = []
    invariant_token_lengths = []
    max_tokens = 0
    skipped_baseline = skipped_no_invariant = skipped_length = 0

    for sample in tqdm(raw_dataset, desc="Processing samples"):
        if sample.get("baseline_decision") != "TRUE":
            skipped_baseline += 1
            continue
        if not sample.get("invariants"):
            skipped_no_invariant += 1
            continue
        program_str = sample["original_program"]
        ast_program = AstProgram().from_code(program_str)
        ast_program.process(print_ast=False)
        invariants = sample["invariants"]
        for invariant in invariants:
            if "invariant" not in invariant:
                skipped_no_invariant += 1
                continue
            user_prompt = user_prompt_template.format(
                program=ast_program.llm_code,
                target_marker=invariant["marker"],
            )
            if version == "v1":
                content = normalize_invariant(invariant["invariant"], pretty=True)
            elif version == "v0":
                content = invariant["invariant"]
            else:
                raise ValueError(f"Invalid version: {version}. Must be 'v0' or 'v1'")
            answer = f'```json\n{{"marker":"{invariant["marker"]}","content":"{content}"}}\n```'
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": answer},
            ]
            text = _apply_chat_template(tokenizer, conversation, chat_template_kwargs)
            if len(tokenizer.encode(text)) > max_length - 1:
                skipped_length += 1
                logger.debug(
                    f"Skipping sample {sample.get('file', '?')} token length > {max_length - 1}"
                )
                continue
            inv_token_len = len(tokenizer.encode(content))
            samples.append({"text": text})
            invariant_token_lengths.append(inv_token_len)
            max_tokens = max(max_tokens, len(tokenizer.encode(text)))

    logger.info(
        f"Filtering stats: skipped_baseline={skipped_baseline}, "
        f"skipped_no_invariant={skipped_no_invariant}, skipped_length={skipped_length}, kept={len(samples)}"
    )
    logger.info(f"Max tokens: {max_tokens}")

    if len(samples) > 0 and token_plot_path:
        plot_token_distribution(
            Dataset.from_list(samples),
            tokenizer,
            name=version,
            plot_path=str(token_plot_path),
        )
    if len(invariant_token_lengths) > 0 and inv_token_plot_path:
        plot_invariant_token_distribution(
            invariant_token_lengths,
            name=version,
            plot_path=str(inv_token_plot_path),
        )
    return Dataset.from_list(samples)


def _try_load_from_hf(repo_name: str) -> Dataset | None:
    """Try to load dataset from HuggingFace Hub. Returns None if not found."""
    try:
        logger.info(f"Attempting to load dataset from HuggingFace: {repo_name}")
        dataset = load_dataset(repo_name, split="train")
        logger.info(f"Loaded {repo_name} with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.info(f"Dataset not found on HuggingFace ({repo_name}): {e}")
        return None


def _upload_dataset_to_hf(dataset: Dataset, repo_name: str, private: bool = False) -> None:
    logger.info(f"Uploading dataset to HuggingFace Hub: {repo_name}")
    try:
        dataset.push_to_hub(repo_name, private=private)
        logger.info(f"Uploaded dataset to {repo_name}")
    except Exception as e:
        logger.error(f"Failed to upload dataset to HuggingFace: {e}")


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="build_sft_dataset",
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    chat_template_kwargs = dict(cfg.tokenizer.get("chat_template_kwargs", {}))

    if cfg.dataset.get("json_path"):
        raw_dataset = load_dataset("json", data_files=cfg.dataset.json_path, split="train")
    else:
        raw_dataset = load_dataset(cfg.dataset.input_repo, split=cfg.dataset.get("split", "train"))
    raw_dataset = raw_dataset.shuffle()
    if cfg.dataset.get("limit", -1) > 0:
        raw_dataset = raw_dataset.select(range(cfg.dataset.limit))

    version = cfg.dataset.get("version", "v2")
    if version not in ("v0", "v1", "v2"):
        raise ValueError(f"dataset.version must be v0, v1, or v2; got {version}")
    min_grade = cfg.dataset.get("min_grade", 2)
    max_length = cfg.dataset.get("max_length", 1024)
    base_output_dir = cfg.output.get("dir")
    # Build HF repo: {org}/wonda-{model_family}-{nt|t}-sft-{v0|v1|v2-g{min_grade}}
    org = cfg.output.get("hf_organization", "idopinto")
    model_family = cfg.tokenizer.name.split("/")[0].lower()
    enable_thinking = cfg.tokenizer.get("chat_template_kwargs", {}).get("enable_thinking", False)
    thinking_suffix = "t" if enable_thinking else "nt"
    repo_base = f"{org}/wonda-{model_family}-{thinking_suffix}-sft"
    if version == "v2":
        hf_repo_name = f"{repo_base}-v2-g{min_grade}"
    elif version == "v1":
        hf_repo_name = f"{repo_base}-v1"
    else:
        hf_repo_name = f"{repo_base}-v0"
    dataset_name = hf_repo_name.split("/")[-1]
    output_dir = str(Path(base_output_dir) / dataset_name) if base_output_dir else None

    if version == "v2":
        dataset = build_sft_dataset_v2(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            system_prompt=cfg.prompts.system_prompt,
            user_prompt_template=cfg.prompts.user_prompt_template,
            chat_template_kwargs=chat_template_kwargs,
            max_length=max_length,
            min_grade=min_grade,
            version=version,
            output_dir=output_dir,
            hf_repo_name=hf_repo_name,
        )
    else:
        dataset = build_sft_dataset_v0_v1(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            system_prompt=cfg.prompts.system_prompt,
            user_prompt_template=cfg.prompts.user_prompt_template,
            chat_template_kwargs=chat_template_kwargs,
            max_length=max_length,
            version=version,
            output_dir=output_dir,
        )
        if hf_repo_name and len(dataset) > 0:
            _upload_dataset_to_hf(dataset, hf_repo_name)

    if output_dir and len(dataset) > 0:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / f"{dataset_name}.json"
        dataset.to_json(str(json_path))
        logger.info(f"Saved SFT dataset to {json_path}")


if __name__ == "__main__":
    main()
