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
import time
from src.eval.models.model_factory import ModelFactory
import weave
from dotenv import load_dotenv
from weave import Evaluation
import hydra
from omegaconf import DictConfig, OmegaConf

import configs.global_config as GC
from src.eval.preprocess_eval_data import (
    get_evaluation_dataset,
    preprocess_for_model,
)
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
    eval_per_marker: bool,
    prompts_config: dict,
) -> tuple[weave.StringPrompt, weave.StringPrompt]:
    """
    Setup prompts based on configuration.

    Args:
        prompts_config: Dictionary containing prompts settings.
    """
    system_key = (
        "per_marker_system_prompt"
        if eval_per_marker
        else "default_system_prompt"
    )
    user_key = (
        "per_marker_user_prompt_template"
        if eval_per_marker
        else "default_user_prompt_template"
    )
    return weave.StringPrompt(
        prompts_config[system_key]
    ), weave.StringPrompt(prompts_config[user_key])


'''
Experiments:
gpt5.2:
uv run -m src.eval.evaluate models=openai_config dataset.limit=1

qwen-0.6b-no-think:
uv run -m src.eval.evaluate models=qwen_no_think_config dataset.limit=1

qwen-0.6b-no-think-v0 (SFT v0 - raw gt invariants):
uv run -m src.eval.evaluate models=qwen_no_think_config models.eval_ft_model=true dataset.limit=1

qwen-0.6b-think:
uv run -m src.eval.evaluate models=qwen_config dataset.limit=1

gpt-oss-20b-low:
uv run -m src.eval.evaluate models=oss_config dataset.limit=1
'''
@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    print("=" * 100)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 100)
    weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")
    dataset = get_evaluation_dataset(
        dataset_name=cfg.dataset.name,
        limit=cfg.dataset.limit,
        prefix=cfg.dataset.prefix,
        split=cfg.dataset.split,
        re_split=cfg.dataset.re_split,
        difficulty_threshold=cfg.dataset.difficulty_threshold,
        eval_per_marker=cfg.eval_per_marker,
    )
    system_prompt, user_prompt_template = setup_prompts(cfg.eval_per_marker, cfg.prompts)
    eval_ft_model = cfg.models.get("eval_ft_model", False)
    invariant_generator = ModelFactory.create(cfg.models, system_prompt, user_prompt_template, eval_ft_model=eval_ft_model)
    logger.info(f"Run name: {invariant_generator.get_run_name()}")
    logger.info(f"Display name: {invariant_generator.get_display_name()}")
    logger.info(f"Evaluating fine-tuned model: {eval_ft_model}")
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
        preprocess_for_model, eval_per_marker=cfg.eval_per_marker
    )

    evaluation = Evaluation(
        dataset=dataset, scorers=[scorer], preprocess_model_input=preprocess_hook
    )

    # Run evaluation
    logger.info("Starting evaluation...")

    asyncio.run(
        evaluation.evaluate(invariant_generator, __weave={"display_name": invariant_generator.get_run_name()})
    )

    # Get collected results for plotting
    plot_results = result_collector.get_results()
    logger.info(f"Collected {len(plot_results)} results for plotting")

    # Generate and upload plot
    if len(plot_results) > 0:
        logger.info("Generating evaluation plot...")
        plot_dir = GC.EXPERIMENTS_DIR / f"eval_plots_{invariant_generator.get_run_name()}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = (
            plot_dir / f"eval_plot_{cfg.weave.project_name}_{invariant_generator.get_run_name()}_{timestamp}.html"
        )

        plot_verification_vs_baseline(
            results=plot_results,
            model_name=invariant_generator.get_display_name(),
            baseline_name=f"UAutomizer{cfg.scorer.verifier.version}",
            split_name=cfg.dataset.split,
            plot_path=plot_path,
        )
        logger.info(f"Plot saved to: {plot_path}")

        # Upload to W&B (optional)
        if not cfg.weave.skip_wandb:
            upload_plot_to_wandb(
                plot_path=plot_path,
                project_name=cfg.weave.project_name,
                run_name=f"plot-{invariant_generator.get_run_name()}",
                artifact_name=f"eval-plot-{invariant_generator.get_run_name()}",
                entity=cfg.weave.entity,
            )
    else:
        logger.warning("No results collected for plotting - skipping plot generation")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    logger.info("Starting evaluation...")
    main()

# Example usage:
# uv run -m src.eval.evaluate 
# uv run -m src.eval.evaluate models=qwen_config dataset.limit=1
# uv run -m src.eval.evaluate models=qwen_no_think_config dataset.limit=1
# uv run -m src.eval.evaluate models=oss_config dataset.limit=1
