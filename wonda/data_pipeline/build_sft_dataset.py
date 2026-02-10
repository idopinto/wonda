#!/usr/bin/env python3
"""Build SFT-ready training dataset from raw or v2 datasets.

Converts per-marker invariant data into chat-formatted SFT samples. Run as a
preprocess step so training only loads pre-built datasets.

Usage:
    uv run -m wonda.data_pipeline.build_sft_dataset --config-name=build_sft_dataset
    uv run -m wonda.data_pipeline.build_sft_dataset dataset.inv_mode=simplified dataset.min_grade=3
"""

import logging
from pathlib import Path

import hydra
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from wonda.core.ast_program import AstProgram
from wonda.data_pipeline.gt_invariant_normalization import normalize_invariant
from wonda.data_pipeline.gt_invariant_simplification import is_degenerate_invariant

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
    inv_mode: str = "simplified",
    output_dir: str | None = None,
    hf_repo_name: str | None = None,
) -> Dataset:
    """Build SFT dataset from v2 format (gt_invariant per sample)."""
    logger.info(
        f"Creating SFT train data: build_sft_dataset_v2 for inv_mode={inv_mode}, min_grade={min_grade}"
    )
    if output_dir:
        token_plot_path = Path(output_dir) / f"token_length_distribution_per_marker_{inv_mode}_v2.{min_grade}.png"
        grade_plot_path = Path(output_dir) / f"grade_distribution_per_marker_{inv_mode}_v2.{min_grade}.png"
        inv_token_plot_path = Path(output_dir) / f"invariant_token_length_distribution_{inv_mode}_v2.{min_grade}.png"
    else:
        token_plot_path = grade_plot_path = inv_token_plot_path = None

    samples = []
    kept_grades = []
    invariant_token_lengths = []
    max_tokens = 0
    skipped_baseline = skipped_grade = skipped_no_invariant = skipped_length = skipped_trivial = 0

    for sample in tqdm(raw_dataset):
        if sample.get("baseline_decision") in {"UNKNOWN", "ERROR", "TIMEOUT"}:
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
            skipped_trivial += 1
            continue
        program_str = sample["original_program"]
        ast_program = AstProgram().from_code(program_str)
        ast_program.process(print_ast=False)
        user_prompt = user_prompt_template.format(
            program=ast_program.llm_code,
            target_marker=invariant.get("marker", ""),
        )
        if inv_mode == "simplified":
            content = invariant.get("target_invariant", "")
        elif inv_mode == "normalized":
            content = invariant.get("pretty_invariant", "")
        elif inv_mode == "raw":
            content = invariant.get("original_invariant", "")
        else:
            raise ValueError(
                f"Invalid inv_mode: {inv_mode}. Must be 'simplified', 'normalized', or 'raw'"
            )
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
        speedup = invariant.get("speedup", 0.0)
        samples.append({
            "text": text,
            "grade": invariant.get("quality_grade", -1),
            "invariant_token_length": inv_token_len,
            "speedup": speedup,
        })
        kept_grades.append(invariant.get("quality_grade", -1))
        invariant_token_lengths.append(inv_token_len)
        max_tokens = max(max_tokens, len(tokenizer.encode(text)))

    logger.info(
        f"Filtering stats: skipped_baseline={skipped_baseline}, skipped_grade={skipped_grade}, "
        f"skipped_no_invariant={skipped_no_invariant}, skipped_length={skipped_length}, "
        f"skipped_trivial={skipped_trivial}, kept={len(samples)}"
    )
    logger.info(f"Max tokens: {max_tokens}")

    final_dataset = Dataset.from_list(samples)

    if len(samples) > 0 and token_plot_path:
        _plot_token_distribution(
            final_dataset, tokenizer,
            name=f"per_marker_{inv_mode}_v2.{min_grade}",
            plot_path=str(token_plot_path),
        )
    if len(kept_grades) > 0 and grade_plot_path:
        _plot_grade_distribution(
            kept_grades,
            name=f"per_marker_{inv_mode}_v2.{min_grade}",
            plot_path=str(grade_plot_path),
        )
    if len(invariant_token_lengths) > 0 and inv_token_plot_path:
        _plot_invariant_token_distribution(
            invariant_token_lengths,
            name=f"per_marker_{inv_mode}_v2.{min_grade}",
            plot_path=str(inv_token_plot_path),
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
    inv_mode: str = "raw",
    output_dir: str | None = None,
) -> Dataset:
    """Build SFT dataset from v0/v1 format (invariants list per sample). inv_mode: 'raw' or 'normalized'."""
    logger.info(f"Creating SFT train data: build_sft_dataset_v0_v1 for inv_mode={inv_mode}")
    if output_dir:
        plot_path = Path(output_dir) / f"token_length_distribution_per_marker_{inv_mode}.png"
    else:
        plot_path = None

    samples = []
    max_tokens = 0
    skipped_baseline = skipped_no_invariant = skipped_length = 0

    for sample in tqdm(raw_dataset, desc="Processing samples"):
        if sample.get("baseline_decision") in {"UNKNOWN", "ERROR", "TIMEOUT"}:
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
            if inv_mode == "normalized":
                content = normalize_invariant(invariant["invariant"], pretty=True)
            elif inv_mode == "raw":
                content = invariant["invariant"]
            else:
                raise ValueError(
                    f"Invalid inv_mode: {inv_mode}. Must be 'raw' or 'normalized' (v0/v1)"
                )
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
            samples.append({"text": text})
            max_tokens = max(max_tokens, len(tokenizer.encode(text)))

    logger.info(
        f"Filtering stats: skipped_baseline={skipped_baseline}, "
        f"skipped_no_invariant={skipped_no_invariant}, skipped_length={skipped_length}, kept={len(samples)}"
    )
    logger.info(f"Max tokens: {max_tokens}")

    if len(samples) > 0 and plot_path:
        _plot_token_distribution(
            Dataset.from_list(samples),
            tokenizer,
            name=f"per_marker_{inv_mode}",
            plot_path=str(plot_path),
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


def _plot_token_distribution(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    name: str = "dataset",
    plot_path: str | None = None,
) -> None:
    from collections import Counter

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    token_lengths = [len(tokenizer.encode(s["text"])) for s in dataset]
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title(f"Token Length Distribution - {name}")
    total = len(token_lengths)
    mean_len = sum(token_lengths) / total if total > 0 else 0
    plt.axvline(mean_len, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_len:.1f}")
    min_len = min(token_lengths) if token_lengths else 0
    max_len = max(token_lengths) if token_lengths else 0
    plt.text(
        0.95, 0.95, f"Total: {total}\nMean: {mean_len:.1f}\nMin: {min_len}\nMax: {max_len}",
        transform=plt.gca().transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Token length distribution plot saved to {plot_path}")


def _plot_grade_distribution(
    grades: list,
    name: str = "dataset",
    plot_path: str | None = None,
) -> None:
    from collections import Counter

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    grade_counts = Counter(grades)
    sorted_grades = sorted(grade_counts.keys())
    counts = [grade_counts[g] for g in sorted_grades]
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(g) for g in sorted_grades], counts, color="steelblue", edgecolor="black")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", va="bottom", fontsize=10)
    plt.xlabel("Quality Grade")
    plt.ylabel("Frequency")
    plt.title(f"Grade Distribution (Kept Samples) - {name}")
    total = len(grades)
    mean_grade = sum(grades) / total if total > 0 else 0
    plt.text(0.95, 0.95, f"Total: {total}\nMean: {mean_grade:.2f}", transform=plt.gca().transAxes, ha="right", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Grade distribution plot saved to {plot_path}")


def _plot_invariant_token_distribution(
    token_lengths: list,
    name: str = "dataset",
    plot_path: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, color="coral", edgecolor="black", alpha=0.7)
    plt.xlabel("Invariant Token Length")
    plt.ylabel("Frequency")
    plt.title(f"Invariant Token Length Distribution - {name}")
    total = len(token_lengths)
    mean_len = sum(token_lengths) / total if total > 0 else 0
    min_len = min(token_lengths) if token_lengths else 0
    max_len = max(token_lengths) if token_lengths else 0
    plt.axvline(mean_len, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_len:.1f}")
    plt.text(0.95, 0.95, f"Total: {total}\nMean: {mean_len:.1f}\nMin: {min_len}\nMax: {max_len}", transform=plt.gca().transAxes, ha="right", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Invariant token length distribution plot saved to {plot_path}")


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

    inv_mode = cfg.dataset.inv_mode
    min_grade = cfg.dataset.get("min_grade", 2)
    max_length = cfg.dataset.get("max_length", 1024)
    output_dir = cfg.output.get("dir")
    hf_repo_base = cfg.output.get("hf_repo")
    # Only v2 (simplified) uses grade suffix; v0/v1 use base repo as-is
    hf_repo_name = (
        f"{hf_repo_base}-g{min_grade}" if inv_mode == "simplified" and hf_repo_base else hf_repo_base
    )

    if inv_mode == "simplified":
        dataset = build_sft_dataset_v2(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            system_prompt=cfg.prompts.system_prompt,
            user_prompt_template=cfg.prompts.user_prompt_template,
            chat_template_kwargs=chat_template_kwargs,
            max_length=max_length,
            min_grade=min_grade,
            inv_mode=inv_mode,
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
            inv_mode=inv_mode,
            output_dir=output_dir,
        )
        if hf_repo_name and len(dataset) > 0:
            _upload_dataset_to_hf(dataset, hf_repo_name)

    if output_dir and len(dataset) > 0:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / f"sft_dataset_{inv_mode}_g{min_grade}.json"
        dataset.to_json(str(json_path))
        logger.info(f"Saved SFT dataset to {json_path}")


if __name__ == "__main__":
    main()
