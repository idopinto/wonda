#!/usr/bin/env python3
"""Build SFT-ready training dataset from raw or V2 datasets.

Converts per-marker invariant data into chat-formatted SFT samples. Run as a
preprocess step so training only loads pre-built datasets.

Usage:
    uv run -m wonda.preprocess.build_sft_dataset --config-name=build_sft_dataset
    uv run -m wonda.preprocess.build_sft_dataset dataset.version=V2
    uv run -m wonda.preprocess.build_sft_dataset dataset.input_repo=idopinto/wonda-train-dataset-full-raw dataset.split=full dataset.version=V0
"""

import logging
import re
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

# V2 caches always include quality_grade >= MIN_GRADE_FLOOR; stricter
# thresholds are applied at load time by load_or_build_sft_dataset(min_grade=...).
MIN_GRADE_FLOOR = 1

# Raw HF sources for each dataset version. V0/V1 share the per-program-with-
# many-invariants format; V2 is the wonda-pipeline's one-best-invariant output.
_RAW_SOURCE_BY_VERSION: dict[str, tuple[str, str]] = {
    "V0": ("idopinto/Wonda-Training-Dataset-Full-Raw", "full"),
    "V1": ("idopinto/Wonda-Training-Dataset-Full-Raw", "full"),
    "V2": ("idopinto/Wonda-Training-Dataset-Full", "train"),
}
# Local fallback root: <RAW_LOCAL_ROOT>/<repo-basename>/<repo-basename>.json.
# Lets users pre-download the raw HF dataset once (e.g. for offline SLURM
# nodes) and have the build pick it up automatically.
RAW_LOCAL_ROOT = Path("data/train")


def _resolve_raw_dataset(version: str, json_path_override: str | None) -> Dataset:
    """Resolve the raw input dataset for ``version`` with sensible fallbacks.

    Precedence:
      1. Explicit ``json_path_override`` (``cfg.dataset.json_path``).
      2. Canonical local copy at
         ``data/train/<repo-basename>/<repo-basename>.json``.
      3. HF Hub ``load_dataset(repo, split=...)``.
    """
    repo, split = _RAW_SOURCE_BY_VERSION[version]
    if json_path_override:
        logger.info(f"Loading raw dataset from explicit json_path: {json_path_override}")
        return load_dataset("json", data_files=json_path_override, split="train")

    repo_basename = repo.split("/")[-1]
    local_path = RAW_LOCAL_ROOT / repo_basename / f"{repo_basename}.json"
    if local_path.exists():
        logger.info(f"Loading raw dataset from local cache: {local_path}")
        return load_dataset("json", data_files=str(local_path), split="train")

    logger.info(f"Loading raw dataset from HuggingFace: {repo} (split={split})")
    return load_dataset(repo, split=split)


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
    min_grade: int = MIN_GRADE_FLOOR,
    version: str = "V2",
    output_dir: str | None = None,
) -> Dataset:
    """Build SFT dataset from V2 format (gt_invariant per sample)."""
    logger.info(f"Creating SFT train data: build_sft_dataset_v2 for version={version}, min_grade={min_grade}")
    if output_dir:
        logger.info(f"Dataset will be written to disk under {output_dir} (variants + plots)")
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
            "messages": conversation,
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

    return final_dataset


def build_sft_dataset_v0_v1(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt_template: str,
    chat_template_kwargs: dict,
    max_length: int = 1024,
    version: str = "V0",
    output_dir: str | None = None,
) -> Dataset:
    """Build SFT dataset from V0/V1 format (invariants list per sample). version: V0 (raw) or V1 (normalized)."""
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
            if version == "V1":
                content = normalize_invariant(invariant["invariant"], pretty=True)
            elif version == "V0":
                content = invariant["invariant"]
            else:
                raise ValueError(f"Invalid version: {version!r}. Must be 'V0' or 'V1'")
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
            samples.append({"text": text, "messages": conversation})
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


# def _try_load_from_hf(repo_name: str) -> Dataset | None:
#     """Try to load dataset from HuggingFace Hub. Returns None if not found."""
#     try:
#         logger.info(f"Attempting to load dataset from HuggingFace: {repo_name}")
#         dataset = load_dataset(repo_name, split="train")
#         logger.info(f"Loaded {repo_name} with {len(dataset)} samples")
#         return dataset
#     except Exception as e:
#         logger.info(f"Dataset not found on HuggingFace ({repo_name}): {e}")
#         return None


def save_sft_variants(dataset: Dataset, output_dir: str | Path) -> tuple[Path, Path]:
    """Persist an SFT dataset to disk in both ``text`` and ``messages`` variants.

    Layout:
        ``<output_dir>/text.json``      — rows: {text, ...aux columns...}
        ``<output_dir>/messages.json``  — rows: {messages, ...aux columns...}

    Expects ``dataset`` to carry both a rendered ``text`` column and a
    template-free ``messages`` column (i.e. the conversation list passed to
    ``apply_chat_template``). Each variant drops the column it doesn't need
    so the on-disk files stay lean and unambiguous.

    Returns paths to (text_path, messages_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / "text.json"
    messages_path = out_dir / "messages.json"

    missing = {"text", "messages"} - set(dataset.column_names)
    if missing:
        raise RuntimeError(
            f"save_sft_variants: dataset is missing required columns {missing}. "
            f"Available: {dataset.column_names}"
        )

    text_ds = dataset.remove_columns(["messages"])
    text_ds.to_json(str(text_path))
    logger.info(f"Saved `text` variant -> {text_path} ({len(text_ds)} rows)")

    messages_ds = dataset.remove_columns(["text"])
    messages_ds.to_json(str(messages_path))
    logger.info(f"Saved `messages` variant -> {messages_path} ({len(messages_ds)} rows)")

    return text_path, messages_path


_QWEN_FAMILY_RE = re.compile(r"^(Qwen\d+(?:\.\d+)?)")
_LLAMA_FAMILY_RE = re.compile(r"^Llama-(\d+(?:\.\d+)?)")
_MISTRAL_FAMILY_RE = re.compile(r"^(Mistral|Ministral)\b", re.IGNORECASE)


def derive_model_family(tokenizer_name: str) -> str:
    """Derive a human-readable model-family slug from a tokenizer/model id.

    Examples:
        ``Qwen/Qwen3-0.6B``                  -> ``Qwen3``
        ``Qwen/Qwen2.5-7B-Instruct``         -> ``Qwen2.5``
        ``meta-llama/Llama-3.1-8B-Instruct`` -> ``Llama3.1``
        ``meta-llama/Llama-3.2-3B``          -> ``Llama3.2``
        ``mistralai/Mistral-7B-Instruct-v0.3`` -> ``Mistral``
        ``mistralai/Ministral-8B-Instruct-2410`` -> ``Ministral``

    Falls back to the first dash-separated segment of the model name if the
    family pattern is unrecognized — that keeps the cache key well-defined
    for new families without forcing a code edit, at the cost of losing the
    version digit.
    """
    name = tokenizer_name.split("/")[-1]
    if m := _QWEN_FAMILY_RE.match(name):
        return m.group(1)
    if m := _LLAMA_FAMILY_RE.match(name):
        return f"Llama{m.group(1)}"
    if m := _MISTRAL_FAMILY_RE.match(name):
        return m.group(1)
    logger.warning(
        f"derive_model_family: unrecognized tokenizer name {tokenizer_name!r}; "
        "falling back to the first dash-separated segment."
    )
    return name.split("-")[0]


def sft_dataset_basename(tokenizer_name: str, version: str) -> str:
    """Canonical SFT cache directory name.

    Pattern: ``Wonda-Training-Dataset-{ModelFamily}-V{0|1|2}``.

    Accepts ``version`` in either case (``V2`` or ``v2``) for robustness but the
    canonical form is uppercase.

    V2 caches always include the full ``quality_grade >= MIN_GRADE_FLOOR``
    population — no ``-g{N}`` suffix — so a single cache serves any stricter
    runtime ``min_grade`` filter.
    """
    family = derive_model_family(tokenizer_name)
    v_upper = version.upper()
    if v_upper not in {"V0", "V1", "V2"}:
        raise ValueError(f"version must be one of V0, V1, V2 (case-insensitive); got {version!r}")
    return f"Wonda-Training-Dataset-{family}-{v_upper}"


def build_and_save_sft_dataset(cfg: DictConfig) -> Dataset:
    """Run the full SFT-build pipeline driven by a Hydra config.

    Loads the raw HF dataset, builds the SFT-ready ``Dataset``, and writes the
    ``text`` + ``messages`` variants to disk under
    ``<output.dir>/Wonda-Training-Dataset-<family>-V{0|1|2}/``. Does NOT push
    to HF — the cache is local-only.

    For V2 the build always retains rows with ``quality_grade >= MIN_GRADE_FLOOR``
    (currently 1); stricter ``min_grade`` thresholds are applied at train time
    by :func:`wonda.train.data_utils.load_or_build_sft_dataset`.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    chat_template_kwargs = dict(cfg.tokenizer.get("chat_template_kwargs", {}))

    # Version is always uppercase end-to-end (V0 / V1 / V2). Accept either
    # case at the entry point for resilience but normalize immediately.
    version = cfg.dataset.get("version", "V2").upper()
    if version not in ("V0", "V1", "V2"):
        raise ValueError(f"dataset.version must be V0, V1, or V2 (got {cfg.dataset.get('version')!r})")

    raw_dataset = _resolve_raw_dataset(version, cfg.dataset.get("json_path"))
    raw_dataset = raw_dataset.shuffle()
    if cfg.dataset.get("limit", -1) > 0:
        raw_dataset = raw_dataset.select(range(cfg.dataset.limit))

    max_length = cfg.dataset.get("max_length", 1024)
    base_output_dir = cfg.output.get("dir")

    dataset_name = sft_dataset_basename(cfg.tokenizer.name, version)
    output_dir = str(Path(base_output_dir) / dataset_name) if base_output_dir else None

    if version == "V2":
        dataset = build_sft_dataset_v2(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            system_prompt=cfg.prompts.system_prompt,
            user_prompt_template=cfg.prompts.user_prompt_template,
            chat_template_kwargs=chat_template_kwargs,
            max_length=max_length,
            min_grade=MIN_GRADE_FLOOR,
            version=version,
            output_dir=output_dir,
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

    if output_dir and len(dataset) > 0:
        save_sft_variants(dataset, output_dir)

    return dataset


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="build_sft_dataset",
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    build_and_save_sft_dataset(cfg)


if __name__ == "__main__":
    main()
