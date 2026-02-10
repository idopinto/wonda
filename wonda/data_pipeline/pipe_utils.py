from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, TypeVar

from datasets import load_dataset
from together import Together
from together.error import ServiceUnavailableError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =============================================================================
# Resume Support
# =============================================================================

def get_checkpoint_path(output_path: Path) -> Path:
    """Get the checkpoint file path based on the output path."""
    return output_path.parent / f"{output_path.stem}.checkpoint.json"


def build_processed_keys(processed: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Build a set of (file, marker, original_invariant) keys from processed entries.
    
    This is used to skip already-processed invariants when resuming.
    """
    keys = set()
    for entry in processed:
        gt_inv = entry.get("gt_invariant", {})
        key = (
            entry.get("file", ""),
            gt_inv.get("marker", ""),
            gt_inv.get("original_invariant", ""),
        )
        keys.add(key)
    return keys


def load_attempted_keys(checkpoint_path: Path) -> set[tuple[str, str, str]]:
    """Load set of all attempted (file, marker, original_invariant) keys from checkpoint.
    
    This includes both successful AND failed attempts, so we don't re-run failures.
    """
    # Check for temp file first (from atomic write that might have succeeded but not renamed)
    temp_file = checkpoint_path.with_suffix(".checkpoint.json.tmp")
    if temp_file.exists():
        try:
            with open(temp_file) as f:
                data = json.load(f)
            keys = {tuple(k) for k in data.get("attempted_keys", [])}
            logger.info(f"Loaded {len(keys)} attempted keys from temp checkpoint {temp_file}")
            # Try to rename it to the main checkpoint
            try:
                temp_file.rename(checkpoint_path)
            except Exception:
                pass  # Ignore rename errors
            return keys
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load temp checkpoint: {e}")
    
    if checkpoint_path.exists():
        # Check if file is empty or too small to be valid JSON
        try:
            file_size = checkpoint_path.stat().st_size
            if file_size < 10:  # Empty or nearly empty file
                logger.warning(f"Checkpoint file is empty or too small ({file_size} bytes). Will rebuild from output.")
                return set()
        except OSError:
            pass
        
        try:
            with open(checkpoint_path) as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"Checkpoint file is empty. Will rebuild from output.")
                    return set()
                data = json.loads(content)
            keys = {tuple(k) for k in data.get("attempted_keys", [])}
            logger.info(f"Loaded {len(keys)} attempted keys from checkpoint {checkpoint_path}")
            return keys
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load checkpoint: {e}. Will rebuild from output.")
    return set()


def save_checkpoint(checkpoint_path: Path, attempted_keys: set[tuple[str, str, str]]) -> None:
    """Save the set of attempted keys to checkpoint file (atomic write)."""
    data = {"attempted_keys": [list(k) for k in attempted_keys]}
    # Atomic write: write to temp file then rename (prevents corruption on crash)
    temp_file = checkpoint_path.with_suffix(".checkpoint.json.tmp")
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    temp_file.rename(checkpoint_path)


def load_existing_progress(output_path: Path) -> list[dict[str, Any]]:
    """Load existing processed entries from output file if it exists."""
    if output_path.exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
            logger.info(f"\033[34mLoaded {len(data)} existing processed entries from {output_path}\033[0m")
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"\033[34mCould not load existing progress: {e}. Starting fresh.\033[0m")
    else:
        logger.info(f"\033[34mNo existing progress file found at {output_path}. Starting fresh.\033[0m")
    return []


def load_data(path: Path, limit: int = -1) -> list[dict[str, Any]]:
    """Load training data JSON, optionally capped to *limit* entries."""
    with open(path) as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]
    logger.info(f"\033[34mLoaded {len(data)} entries from {path}\033[0m")
    return data

def load_from_hf(dataset_name: str, split: str = "full", limit: int = -1) -> list[dict[str, Any]]:
    """Load data from Hugging Face dataset."""
    dataset = load_dataset(dataset_name, split=split)
    if limit > 0:
        dataset = dataset.select(range(limit))
    logger.info(f"\033[34mLoaded {len(dataset)} entries from {dataset_name} split {split}\033[0m")
    return list(dataset)
# =============================================================================
# Retry Decorator
# =============================================================================

def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (ServiceUnavailableError, RateLimitError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exceptions to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed with {type(e).__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# LLM Candidate Generation
# =============================================================================

def parse_llm_response(content: str) -> tuple[str, str]:
    """Parse LLM response to extract simplified_invariant and rationale."""
    # Try JSON block first: ```json {...} ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    json_str = match.group(1) if match else content
    
    try:
        data = json.loads(json_str)
        return data.get("simplified_invariant", ""), data.get("rationale", "")
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM response: {content[:100]}...")
        return "", ""


# =============================================================================
# LLM Response Caching
# =============================================================================
# Use absolute path based on project root to avoid Hydra working directory issues
LLM_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "llm_response_cache"


def get_cache_key(model: str, messages: list[dict], max_tokens: int, temperature: float, n: int) -> str:
    """Create a deterministic hash for the API call parameters."""
    cache_data = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
    }, sort_keys=True)
    return hashlib.sha256(cache_data.encode()).hexdigest()


def save_to_cache(cache_key: str, response) -> None:
    """Serialize and save the API response to disk (atomic write)."""
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = LLM_CACHE_DIR / f"{cache_key}.json"
    
    data = {
        "id": response.id,
        "model": response.model,
        "choices": [
            {
                "index": c.index,
                "message": {
                    "role": c.message.role,
                    "content": c.message.content,
                },
                "finish_reason": c.finish_reason,
            }
            for c in response.choices
        ],
    }
    # Atomic write: write to temp file then rename (prevents corruption on crash)
    temp_file = cache_file.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    temp_file.rename(cache_file)


def load_from_cache(cache_key: str):
    """Load cached response if it exists. Returns None if not cached."""
    cache_file = LLM_CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file) as f:
            data = json.load(f)
        
        # Return a simple object that mimics the API response structure
        choices = []
        for c in data["choices"]:
            msg = SimpleNamespace(role=c["message"]["role"], content=c["message"]["content"])
            choice = SimpleNamespace(index=c["index"], message=msg, finish_reason=c["finish_reason"])
            choices.append(choice)
        
        return SimpleNamespace(id=data["id"], model=data["model"], choices=choices)
    except (json.JSONDecodeError, KeyError, IOError) as e:
        logger.warning(f"Failed to load cache file {cache_file}: {e}")
        return None


@retry_with_backoff(max_retries=5, base_delay=5.0, max_delay=120.0)
def call_together_api(
    client: Together,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    n: int,
):
    """Call Together API with retry logic, caching, and transient error handling."""
    cache_key = get_cache_key(model, messages, max_tokens, temperature, n)
    
    # Check cache first
    cached = load_from_cache(cache_key)
    if cached is not None:
        logger.info(f"Using cached LLM response: {cache_key[:12]}...")
        return cached
    
    # Make API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )
    
    # Save to cache
    save_to_cache(cache_key, response)
    logger.info(f"Cached LLM response: {cache_key[:12]}...")
    
    return response

