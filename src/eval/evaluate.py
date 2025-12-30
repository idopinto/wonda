"""
Main evaluation script for loop invariant generation.

This module orchestrates the evaluation pipeline:
1. Load configuration via Hydra (../../configs/eval/config.yaml)
2. Initialize Weave for experiment tracking
3. Load and preprocess the evaluation dataset
4. Load the model (base or fine-tuned with LoRA)
5. Setup prompts and create the InvariantGeneratorModel
6. Run evaluation using Weave's Evaluation framework with UAutomizer verification
7. Collect results and generate visualization plots - only if there are results to plot
8. Optionally upload plots to Weights & Biases - only if the user wants to upload them
"""

import asyncio
import functools
import logging
import os
import re
import copy as py_copy
import time
import weave
from dotenv import load_dotenv
from weave import Evaluation
import hydra
from omegaconf import DictConfig, OmegaConf

import configs.global_config as GC
from src.preprocess.preprocess_eval_data import (
    get_evaluation_dataset,
    preprocess_example,
)
from src.eval.model import InvariantGeneratorModel
from src.eval.scorer import InvariantScorer, ResultCollector
from src.verifiers.uautomizer import UAutomizerVerifier
from src.eval.plot import plot_verification_vs_baseline
from src.eval.wandb_utils import upload_plot_to_wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables
load_dotenv()


def setup_prompts(
    prompts_config: dict,
) -> tuple[weave.StringPrompt, weave.StringPrompt]:
    """
    Setup prompts based on configuration.

    Args:
        prompts_config: Dictionary containing prompts settings.
    """
    eval_per_location = bool(prompts_config.get("eval_per_location", False))
    user_key = (
        "per_location_user_prompt_template"
        if eval_per_location
        else "default_user_prompt_template"
    )
    return weave.StringPrompt(
        prompts_config["default_system_prompt"]
    ), weave.StringPrompt(prompts_config[user_key])


def preprocess_for_model(example: dict, eval_per_location: bool) -> dict:
    """
    One preprocess function that supports both modes.

    - Always returns an AST-based Program under key `program`
    - Always returns `baseline_timing`
    - If eval_per_location=True, also returns `target_label` (required by per-location prompt)
    """
    out = preprocess_example(example)
    if eval_per_location:
        target_label = example["target_label"]
        out["target_label"] = str(target_label)
    return out


def _num_loop_markers(program_llm_code: str) -> int:
    """Return max k in INVARIANT_MARKER_k occurrences (0 if none)."""
    ks = [
        int(m.group(1))
        for m in re.finditer(r"INVARIANT_MARKER_(\d+)", program_llm_code)
    ]
    return max(ks) if ks else 0


def _get_num_loops(program: object) -> int:
    """Prefer Program.num_loops if present, else fall back to scanning llm_code."""
    n = getattr(program, "num_loops", None)
    if isinstance(n, int) and n >= 0:
        return n
    llm_code = getattr(program, "llm_code", "") or ""
    return _num_loop_markers(llm_code)


def get_run_name(cfg: DictConfig) -> str:
    """Get run name for evaluation."""
    mode = (
        "perloc" if bool(getattr(cfg.prompts, "eval_per_location", False)) else "anyloc"
    )
    if cfg.model.eval_finetuned_model:
        return f"eval-{cfg.model.finetuned_model_id.split('/')[-1]}-{cfg.model.reasoning_effort}-{cfg.dataset.split}-{mode}"
    else:
        return f"eval-{cfg.model.base_model_name.split('/')[-1]}-{cfg.model.reasoning_effort}-{cfg.dataset.split}-{mode}"


def get_model_display_name(cfg: DictConfig) -> str:
    """Get display name for the model including reasoning effort."""
    if cfg.model.eval_finetuned_model:
        return f"{cfg.model.finetuned_model_id.split('/')[-1]}-{cfg.model.reasoning_effort}"
    else:
        return (
            f"{cfg.model.base_model_name.split('/')[-1]}-{cfg.model.reasoning_effort}"
        )


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    print("=" * 100)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 100)
    run_name = get_run_name(cfg)
    model_display_name = get_model_display_name(cfg)
    logger.info(f"run_name: {run_name}")
    logger.info(f"model_display_name: {model_display_name}")
    weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")
    dataset = get_evaluation_dataset(
        dataset_name=cfg.dataset.name,
        limit=cfg.dataset.limit,
        prefix=cfg.dataset.prefix,
        split=cfg.dataset.split,
    )

    eval_per_location = bool(getattr(cfg.prompts, "eval_per_location", False))

    # If per-location mode is enabled but the dataset doesn't provide target_label,
    # expand each program into one example per loop marker (INVARIANT_MARKER_k).
    if eval_per_location and dataset and "target_label" not in dataset[0]:
        expanded = []
        for ex in dataset:
            # Preprocess once, then clone per marker k
            prog = preprocess_example(ex)["program"]
            k_max = _get_num_loops(prog)
            if k_max == 0:
                continue
            for k in range(1, k_max + 1):
                ex_k = dict(ex)
                # Avoid reprocessing in preprocess_for_model
                ex_k["original_program"] = py_copy.deepcopy(prog).code
                ex_k["target_label"] = str(k)
                expanded.append(ex_k)
        dataset = expanded
        logger.info(f"Expanded dataset to {len(dataset)} per-location examples")
    logger.info(f"Loaded {len(dataset)} examples")
    system_prompt, user_prompt_template = setup_prompts(cfg.prompts)
    invariant_generator = InvariantGeneratorModel(
        client=cfg.model.client,
        model_cfg=cfg.model,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        sampling_params=cfg.model.sampling_params,
        reasoning_effort=cfg.model.reasoning_effort,
    )
    result_collector = ResultCollector()
    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.scorer.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.scorer.verifier.property,
        arch=cfg.scorer.verifier.arch,
        timeout_seconds=cfg.scorer.verifier.timeout_seconds,
        version=cfg.scorer.verifier.version,
    )
    scorer = InvariantScorer(
        verifier=verifier,
        collector=result_collector,
        baseline_is_timeout=cfg.scorer.baseline_is_timeout,
    )

    preprocess_hook = functools.partial(
        preprocess_for_model, eval_per_location=eval_per_location
    )

    evaluation = Evaluation(
        dataset=dataset, scorers=[scorer], preprocess_model_input=preprocess_hook
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    asyncio.run(
        evaluation.evaluate(invariant_generator, __weave={"display_name": run_name})
    )

    # # Get collected results for plotting
    # plot_results = result_collector.get_results()
    # logger.info(f"Collected {len(plot_results)} results for plotting")

    # # # Generate and upload plot
    # if len(plot_results) > 0:
    #     logger.info("Generating evaluation plot...")
    #     plot_dir = GC.EXPERIMENTS_DIR / f"eval_plots_{run_name}"
    #     plot_dir.mkdir(parents=True, exist_ok=True)

    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     plot_path = (
    #         plot_dir / f"eval_plot_{cfg.weave.project_name}_{run_name}_{timestamp}.html"
    #     )

    #     plot_verification_vs_baseline(
    #         results=plot_results,
    #         model_name=model_display_name,
    #         baseline_name=f"UAutomizer{cfg.scorer.verifier.version}",
    #         split_name=cfg.dataset.split,
    #         plot_path=plot_path,
    #     )
    #     logger.info(f"Plot saved to: {plot_path}")

    #     # Upload to W&B (optional)
    #     if not cfg.weave.skip_wandb:
    #         upload_plot_to_wandb(
    #             plot_path=plot_path,
    #             project_name=cfg.weave.project_name,
    #             run_name=f"plot-{run_name}",
    #             artifact_name=f"eval-plot-{run_name}",
    #             entity=cfg.weave.entity,
    #         )
    # else:
    #     logger.warning("No results collected for plotting - skipping plot generation")

    # logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

# Example usage:
# uv run -m src.eval.evaluate 