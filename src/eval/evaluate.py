"""
Main evaluation script for loop invariant generation.

This module orchestrates the evaluation pipeline:
1. Load configuration and model
2. Load and preprocess dataset
3. Run evaluation with Weave
4. Generate and upload visualization
"""
import argparse
import asyncio
import logging
import os
import time

import weave
from dotenv import load_dotenv
from weave import Evaluation

from src.eval.config import get_default_config
from src.preprocess.preprocess_eval_data import get_evaluation_dataset, preprocess_example
from src.eval.model import create_invariant_generator, load_model
from src.eval.plot import plot_verification_vs_baseline
from src.eval.scorer import InvariantScorer, ResultCollector
from src.eval.wandb_utils import upload_plot_to_wandb
import configs.global_configurations as GC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure PyTorch memory allocation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables
load_dotenv()

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate models on InvBench dataset"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="inv-gen-eval",
        help="Weave project name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="hard",
        choices=["easy", "hard"],
        help="Data split"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds (default: 600.0) for the verifier"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit number of tasks (default: -1 for all)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for dataset files (default: empty)"
    )
    parser.add_argument(
        "--skip_wandb",
        action="store_true",
        help="Skip uploading plot to W&B"
    )
    parser.add_argument(
        "--eval-sft",
        action="store_true",
        help="Evaluate SFT model. If not provided, evaluate base model."
    )
    return parser.parse_args()

def main():
    """Main evaluation entry point."""
    args = parse_args()
    print(args)
    # Create configuration (testing=True by default in config)
    config = get_default_config()
    config.timeout = args.timeout
    
    # # Get derived names from config
    # dataset_name = config.get_dataset_name()
    # config.eval_sft = args.eval_sft
    # project_name = config.get_project_name()
    # run_name = config.get_run_name(args.split)
    
    # # Initialize Weave
    # weave.init(f"{config.weave_team}/{project_name}")
    
    # # Load dataset
    # logger.info(f"Loading dataset: {dataset_name}")
    # dataset = get_evaluation_dataset(
    #     dataset_name=dataset_name,
    #     limit=args.limit,
    #     prefix=args.prefix,
    #     split=args.split
    # )
    # logger.info(f"Loaded {len(dataset)} examples")
    
    # # Load model
    # logger.info(f"Loading model: {config.base_model_name}")
    # tokenizer, model = load_model(config)
    
    # # Create invariant generator
    # invariant_generator = create_invariant_generator(config, tokenizer, model)
    
    # # Create result collector for plotting
    # result_collector = ResultCollector()
    
    # # Create scorer with configuration and collector
    # scorer = InvariantScorer(config=config, collector=result_collector)
    
    # Create evaluation
    # evaluation = Evaluation(
    #     dataset=dataset,
    #     scorers=[scorer],
    #     preprocess_model_input=preprocess_example
    # )
    
    # # Run evaluation
    # logger.info("Starting evaluation...")
    # asyncio.run(
    #     evaluation.evaluate(
    #         invariant_generator,
    #         __weave={"display_name": run_name}
    #     )
    # )
    
    # # Get collected results for plotting
    # plot_results = result_collector.get_results()
    # logger.info(f"Collected {len(plot_results)} results for plotting")
    
    # # Generate and upload plot
    # if len(plot_results) > 0:
    #     logger.info("Generating evaluation plot...")
    #     plot_dir = GC.EXPERIMENTS_DIR / "plots"
    #     plot_dir.mkdir(parents=True, exist_ok=True)
        
    #     timestamp = time.strftime('%Y%m%d_%H%M%S')
    #     plot_path = plot_dir / f"eval_plot_{run_name}_{timestamp}.html"
        
    #     plot_verification_vs_baseline(
    #         results=plot_results,
    #         model_name=config.get_model_display_name(),
    #         baseline_name=f"UAutomizer{config.uautomizer_version}",
    #         split_name=args.split,
    #         plot_path=plot_path
    #     )
    #     logger.info(f"Plot saved to: {plot_path}")
        
    #     # Upload to W&B (optional)
    #     if not args.skip_wandb:
    #         upload_plot_to_wandb(
    #             plot_path=plot_path,
    #             project_name=project_name,
    #             run_name=f"plot-{run_name}",
    #             artifact_name=f"eval-plot-{run_name}",
    #             entity=config.weave_team
    #         )
    #     else:
    #         logger.info("Skipping W&B upload (--skip_wandb flag)")
    # else:
    #     logger.warning("No results collected for plotting - skipping plot generation")
    
    # logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
