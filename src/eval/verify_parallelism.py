"""
Quick script to verify WEAVE_PARALLELISM is working.

This script runs a small evaluation and checks:
1. That WEAVE_PARALLELISM is set correctly
2. That evaluation runs with timing that suggests parallelism
3. Logs any Weave internal parallelism info if available

Usage:
    WEAVE_PARALLELISM=4 uv run -m src.eval.verify_parallelism
    uv run -m src.eval.verify_parallelism  # Uses default of 4
"""

# IMPORTANT: Set parallelism BEFORE importing weave
import os
os.environ.setdefault("WEAVE_PARALLELISM", "4")

import asyncio
import time
import logging
from pathlib import Path

import weave
from weave import Evaluation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_parallelism_setting():
    """Verify WEAVE_PARALLELISM is set and log it."""
    parallelism = os.environ.get("WEAVE_PARALLELISM")
    if parallelism:
        logger.info(f"✓ WEAVE_PARALLELISM is set to: {parallelism}")
        return int(parallelism)
    else:
        logger.warning("✗ WEAVE_PARALLELISM is NOT set!")
        return None


async def test_parallel_evaluation():
    """Run a small test evaluation to verify parallelism."""
    parallelism = verify_parallelism_setting()
    
    # Create a simple test dataset
    test_dataset = [
        {"input": f"test_{i}", "output": f"result_{i}"}
        for i in range(10)
    ]
    
    # Simple scorer that takes some time
    @weave.op()
    def slow_scorer(output: str, input: str) -> dict:
        """A scorer that simulates work."""
        time.sleep(0.1)  # 100ms per item
        return {"score": len(output)}
    
    @weave.op()
    def summarize_scores(scores: list) -> dict:
        return {"total": len(scores)}
    
    class TestScorer(weave.Scorer):
        @weave.op()
        def score(self, output: str, input: str) -> dict:
            time.sleep(0.1)  # 100ms per item
            return {"score": len(output)}
        
        @weave.op()
        def summarize(self, score_rows: list) -> dict:
            return {"total": len(score_rows)}
    
    scorer = TestScorer()
    
    # Simple model
    @weave.op()
    def test_model(input: str) -> str:
        return f"result_{input}"
    
    evaluation = Evaluation(dataset=test_dataset, scorers=[scorer])
    
    logger.info(f"Running test evaluation with {len(test_dataset)} items...")
    logger.info(f"Expected sequential time: {len(test_dataset) * 0.1:.2f} seconds")
    logger.info(f"Expected parallel time (n={parallelism}): ~{len(test_dataset) * 0.1 / (parallelism or 1):.2f} seconds")
    
    start_time = time.time()
    summary = await evaluation.evaluate(test_model)
    duration = time.time() - start_time
    
    logger.info(f"Actual evaluation time: {duration:.2f} seconds")
    
    if parallelism:
        expected_sequential = len(test_dataset) * 0.1
        expected_parallel = expected_sequential / parallelism
        speedup = expected_sequential / duration
        
        logger.info(f"Expected sequential: {expected_sequential:.2f}s")
        logger.info(f"Expected parallel (n={parallelism}): {expected_parallel:.2f}s")
        logger.info(f"Actual: {duration:.2f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:  # Should see at least 1.5x speedup with parallelism
            logger.info("✓ Parallelism appears to be working!")
        else:
            logger.warning("⚠ Parallelism may not be working as expected")
    
    return summary


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Weave Parallelism Verification")
    logger.info("=" * 60)
    
    # Check environment before any weave operations
    verify_parallelism_setting()
    
    # Run test
    summary = asyncio.run(test_parallel_evaluation())
    
    logger.info("=" * 60)
    logger.info("Verification complete")
    logger.info("=" * 60)
