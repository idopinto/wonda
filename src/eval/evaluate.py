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
import logging
import time

import weave
from dotenv import load_dotenv
from weave import Evaluation
import hydra
from omegaconf import DictConfig, OmegaConf

import configs.global_config as GC
from src.preprocess.preprocess_eval_data import get_evaluation_dataset, preprocess_example
from src.eval.model import InvariantGeneratorModel
from src.eval.plot import plot_verification_vs_baseline
from src.eval.scorer import InvariantScorer, ResultCollector
from src.eval.wandb_utils import upload_plot_to_wandb
from src.verifiers.uautomizer import UAutomizerVerifier
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure PyTorch memory allocation
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables
load_dotenv()

def setup_prompts(prompts_config: dict) -> tuple[weave.StringPrompt, weave.StringPrompt]:
    """
    Setup prompts based on configuration.
    
    Args:
        prompts_config: Dictionary containing prompts settings.
    """
    return weave.StringPrompt(prompts_config["default_system_prompt"]), weave.StringPrompt(prompts_config["default_user_prompt_template"])


def get_run_name(cfg: DictConfig) -> str:
    """Get run name for evaluation."""
    if cfg.model.eval_finetuned_model:
        return f"eval-{cfg.model.finetuned_model_id.split('/')[-1]}-{cfg.model.reasoning_effort}-{cfg.dataset.split}"
    else:
        return f"eval-{cfg.model.base_model_name.split('/')[-1]}-{cfg.model.reasoning_effort}-{cfg.dataset.split}"

def get_model_display_name(cfg: DictConfig) -> str:
    """Get display name for the model including reasoning effort."""
    if cfg.model.eval_finetuned_model:
        return f"{cfg.model.finetuned_model_id.split('/')[-1]}-{cfg.model.reasoning_effort}"
    else:
        return f"{cfg.model.base_model_name.split('/')[-1]}-{cfg.model.reasoning_effort}"

@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    print("="* 100)
    print(OmegaConf.to_yaml(cfg))
    print("="* 100)
    run_name = get_run_name(cfg)
    model_display_name = get_model_display_name(cfg)
    logger.info(f"run_name: {run_name}")
    logger.info(f"model_display_name: {model_display_name}")
    weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")
    dataset = get_evaluation_dataset(
        dataset_name=cfg.dataset.name,
        limit=cfg.dataset.limit,
        prefix=cfg.dataset.prefix,
        split=cfg.dataset.split
    )
    logger.info(f"Loaded {len(dataset)} examples")    
    system_prompt, user_prompt_template = setup_prompts(cfg.prompts)
    invariant_generator = InvariantGeneratorModel(
        client=cfg.model.client,
        model_cfg=cfg.model,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        sampling_params=cfg.model.sampling_params,
        reasoning_effort=cfg.model.reasoning_effort,
        only_loop_beginnings=cfg.prompts.only_loop_beginnings
    )
    result_collector = ResultCollector()    
    verifier = UAutomizerVerifier(uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.scorer.verifier.version],
                                  property_file_path=GC.PROPERTIES_DIR / cfg.scorer.verifier.property,
                                  arch=cfg.scorer.verifier.arch,
                                  timeout_seconds=cfg.scorer.verifier.timeout_seconds,
                                  version=cfg.scorer.verifier.version)
    scorer = InvariantScorer(verifier=verifier, collector=result_collector, baseline_is_timeout=cfg.scorer.baseline_is_timeout)
    evaluation = Evaluation(dataset=dataset,scorers=[scorer],preprocess_model_input=preprocess_example)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    asyncio.run(evaluation.evaluate(invariant_generator,__weave={"display_name": run_name}))
    
    # Get collected results for plotting
    plot_results = result_collector.get_results()
    logger.info(f"Collected {len(plot_results)} results for plotting")
    
    # # Generate and upload plot
    if len(plot_results) > 0:
        logger.info("Generating evaluation plot...")
        plot_dir = GC.EXPERIMENTS_DIR / f"eval_plots_{run_name}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        plot_path = plot_dir / f"eval_plot_{cfg.weave.project_name}_{run_name}_{timestamp}.html"
        
        plot_verification_vs_baseline(
            results=plot_results,
            model_name=model_display_name,
            baseline_name=f"UAutomizer{cfg.scorer.verifier.version}",
            split_name=cfg.dataset.split,
            plot_path=plot_path
        )
        logger.info(f"Plot saved to: {plot_path}")
        
        # Upload to W&B (optional)
        if not cfg.weave.skip_wandb:
            upload_plot_to_wandb(
                plot_path=plot_path,
                project_name=cfg.weave.project_name,
                run_name=f"plot-{run_name}",
                artifact_name=f"eval-plot-{run_name}",
                entity=cfg.weave.entity
            )
    else:
        logger.warning("No results collected for plotting - skipping plot generation")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
