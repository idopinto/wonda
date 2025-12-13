"""
Weights & Biases utilities for evaluation.

Contains functions for uploading artifacts and plots to W&B.
"""
import logging
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)


def upload_plot_to_wandb(
    plot_path: Path,
    project_name: str,
    run_name: str,
    artifact_name: str = "evaluation-plot",
    entity: str = None
) -> bool:
    """
    Upload an HTML plot to W&B as an artifact.
    
    Creates a new W&B run, uploads the plot as both an artifact and inline HTML,
    then closes the run.
    
    Args:
        plot_path: Path to the HTML file to upload.
        project_name: W&B project name (without entity/team prefix).
        run_name: Name for this W&B run.
        artifact_name: Name for the artifact.
        entity: W&B entity (team/username). If None, uses default.
    
    Returns:
        True if upload succeeded, False otherwise.
    """
    try:
        run = wandb.init(
            entity=entity,
            project=f"{project_name}-plots",
            name=run_name,
            job_type="visualization",
        )
        
        # Create and log artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="visualization",
            description="Interactive evaluation plot"
        )
        artifact.add_file(str(plot_path))
        run.log_artifact(artifact)
        
        # Also log the HTML directly for inline viewing
        wandb.log({"evaluation_plot": wandb.Html(str(plot_path))})
        
        run.finish()
        logger.info(f"Uploaded plot to W&B: {project_name}/{artifact_name}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to upload plot to W&B: {e}")
        logger.info(f"Plot is still available locally at: {plot_path}")
        return False

