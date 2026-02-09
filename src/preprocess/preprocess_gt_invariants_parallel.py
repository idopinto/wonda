#!/usr/bin/env python3
"""
Preprocess ground truth invariants with parallel N-candidate generation and verification.

Pipeline:
1. Load training data with invariants
2. For each invariant:
   - If verbose: Generate N simplified candidates via LLM, verify in parallel
   - If non-verbose: Verify as-is
3. Filter by quality_grade > 0
4. Save qualifying entries

Usage:
    uv run -m src.preprocess.preprocess_gt_invariants_parallel dataset.limit=5
    sbatch scripts/preprocess/preprocess_gt_invariants_parallel.sbatch dataset.limit=-1
"""

import copy
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Optional, TypeVar

import hydra
import weave
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from together import Together
from together.error import ServiceUnavailableError, RateLimitError
from tqdm import tqdm

T = TypeVar("T")

from configs import global_config as GC
from src.preprocess.clean_invariants import clean_invariant
from src.core.ast_program import AstProgram
from src.core.property import Property
from src.eval.validate import syntactic_validation
from src.verifiers.uautomizer import UAutomizerVerifier, VerifierCallReport

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Retry Logic
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
# Data Classes
# =============================================================================
@dataclass
class CandidateQualityGrade:
    is_correct: bool = field(default=False)
    does_target_property_still_hold: bool = field(default=False)
    has_speedup: bool = field(default=False)
    speedup: Optional[float] = field(default=None)
    quality_grade: int = field(default=-1) # -1 means unknown (unchecked) quality.

@dataclass
class CandidateInvariant:
    """A candidate invariant with its rationale."""
    content: str = ""
    rationale: str = ""
    reasoning: str = ""
    
    @property
    def is_empty(self) -> bool:
        return not self.content.strip()


@dataclass
class InvariantContext:
    """Context for processing a single invariant."""
    entry: dict[str, Any]
    entry_idx: int
    inv_idx: int
    marker: str
    original_invariant: str
    pretty_invariant: str
    invariant_to_process: str
    baseline_decision: str
    baseline_timing: float
    program: AstProgram
    original_program: str

# =============================================================================
# Helper Functions
# =============================================================================
def is_verbose_invariant(inv_text: str, cfg: DictConfig) -> bool:
    """Check if an invariant matches the verbose criteria for simplification."""
    # disjunct_count = inv_text.count(" || ") + 1
    # char_length = len(inv_text)
    return cfg.filtering.min_char_length <= len(inv_text)
    # return cfg.filtering.min_disjuncts <= disjunct_count <= cfg.filtering.max_disjuncts and cfg.filtering.min_char_length <= char_length <= cfg.filtering.max_char_length

def compute_quality_grade(
    result: CandidateQualityGrade,
) -> int:
    """Compute the quality grade of a candidate."""
    if not result.is_correct:
        return 0
    elif not result.does_target_property_still_hold:
        return 1  
    elif not result.has_speedup:
        return 2
    else:
        return 3  

def run_verifier(verifier: UAutomizerVerifier, program_str: str, kind: str) -> VerifierCallReport:
    """Write program to temp file and run verifier."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_dir = Path(temp_dir)
        code_dir.mkdir(parents=True, exist_ok=True)
        program_path = code_dir / f"code_for_{kind}.c"
        with open(program_path, "w") as f:
            f.write(program_str)
        verifier_report: VerifierCallReport = verifier.verify(program_path=program_path, reports_dir=code_dir, timeout_seconds=verifier.timeout_seconds)
        logger.info(f"Verifier report for {kind}: {verifier_report.decision} ({verifier_report.decision_reason}) in {verifier_report.time_taken} seconds")
        return verifier_report

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


@weave.op()
def generate_candidates(
    client: Together,
    ctx: InvariantContext,
    cfg: DictConfig,
) -> list[CandidateInvariant]:
    """Generate N simplified invariant candidates using LLM."""
    program = AstProgram().from_code(ctx.original_program)
    program.process(print_ast=False)
    
    user_prompt = cfg.prompts.simplify_invariant_user_prompt.format(
        program=program.llm_code,
        invariant=ctx.invariant_to_process,
        marker=ctx.marker,
    )
    
    try:
        response = call_together_api(
            client=client,
            model=cfg.model.name,
            messages=[
                {"role": "system", "content": cfg.prompts.simplify_invariant_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=cfg.model.max_tokens,
            temperature=cfg.model.temperature,
            n=cfg.verifier.max_parallel_workers,
        )
    except (ServiceUnavailableError, RateLimitError) as e:
        logger.error(f"API call failed after all retries: {e}")
        return []
    
    candidates = []
    for choice in response.choices:
        msg = choice.message
        content, rationale = parse_llm_response(msg.content or "")
        candidates.append(CandidateInvariant(
            content=content,
            rationale=rationale,
            reasoning=getattr(msg, "reasoning", "") or "",
        ))
    
    logger.info(f"Generated {len(candidates)} candidates from LLM")
    return candidates


# =============================================================================
# Verification
# =============================================================================
@weave.op()
def verify_candidate(
    verifier: UAutomizerVerifier,
    candidate: CandidateInvariant,
    program: AstProgram,
    marker: str,
    baseline_decision: str,
    baseline_timing: float,
) -> CandidateQualityGrade:
    """Verify a single candidate (correctness + usefulness in parallel)."""
    result = CandidateQualityGrade()
    predicate = Property(content=candidate.content, marker_name=marker)
    
    # Syntactic validation
    result.is_valid = syntactic_validation(predicate.content)
    if not result.is_valid:
        result.quality_grade = 0
        logger.warning(f"Syntactically invalid: '{predicate.content[:50]}...'")
        return result
    
    # Filter trivial invariants (UAutomizer artifacts)
    normalized = predicate.content.replace('(', '').replace(')', '').replace(' ', '').lower()
    if normalized in ('0', '1', 'true', 'false'):
        result.quality_grade = 0
        logger.warning(f"Trivial invariant rejected: '{predicate.content}' (UAutomizer artifact)")
        return result
    
    # Prepare verification programs
    correctness_program = program.get_program_with_assertion(
        predicate=predicate, assumptions=[], for_llm=False
    )
    usefulness_program = program.get_program_with_assertion(
        predicate=program.get_target_assert(), assumptions=[predicate], for_llm=False
    )
    
    # Run both checks in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        correctness_future = executor.submit(run_verifier, verifier, correctness_program, "correctness")
        usefulness_future = executor.submit(run_verifier, verifier, usefulness_program, "usefulness")
        
        correctness_result = correctness_future.result()
        usefulness_result = usefulness_future.result()
    
    # Process results
    result.is_correct = correctness_result.decision == "TRUE"
    result.does_target_property_still_hold = (
        usefulness_result.decision in {"TRUE", "FALSE"} 
        and usefulness_result.decision == baseline_decision
    )
    # Total verification time is max of both (they run in parallel)
    total_verification_time = max(correctness_result.time_taken, usefulness_result.time_taken)
    result.has_speedup = total_verification_time < baseline_timing
    result.speedup = (baseline_timing / total_verification_time) if total_verification_time > 0 else 0.0
    result.quality_grade = compute_quality_grade(result)
    return result


def deduplicate_candidates(candidates: list[CandidateInvariant]) -> list[CandidateInvariant]:
    """Remove exact duplicates (by normalized content)."""
    seen = set()
    unique = []
    for c in candidates:
        if c.is_empty:
            continue
        key = " ".join(c.content.split())  # Normalize whitespace
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    if len(unique) < len(candidates):
        logger.info(f"Deduplicated: {len(candidates)} -> {len(unique)} candidates")
    return unique


def verify_candidates_parallel(
    verifier: UAutomizerVerifier,
    candidates: list[CandidateInvariant],
    program: AstProgram,
    marker: str,
    baseline_decision: str,
    baseline_timing: float,
    max_workers: int,
) -> list[tuple[CandidateInvariant, CandidateQualityGrade]]:
    """Verify all candidates in parallel."""
    candidates = deduplicate_candidates(candidates)
    if not candidates:
        logger.warning("No valid candidates to verify")
        return []
    
    logger.info(f"Verifying {len(candidates)} candidates (max_workers={max_workers})")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                verify_candidate, verifier, c, program, marker, baseline_decision, baseline_timing
            ): c for c in candidates
        }
        
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                quality = future.result()
                results.append((candidate, quality))
                logger.info(f"'{candidate.content[:40]}...' -> grade={quality.quality_grade}")
            except Exception as e:
                logger.error(f"Verification error: {e}")
                results.append((candidate, CandidateQualityGrade(quality_grade=0)))
    
    return results


# =============================================================================
# Entry Creation
# =============================================================================

def create_entry(
    ctx: InvariantContext,
    candidate: CandidateInvariant,
    quality: CandidateQualityGrade,
    is_simplified: bool,
) -> dict[str, Any]:
    """Create a processed entry with gt_invariant.
    
    Args:
        ctx: Invariant context with original/pretty invariants
        candidate: The candidate that was verified
        quality: Verification quality results
        is_simplified: True if candidate is LLM-simplified, False if verified as-is
    """
    entry = copy.deepcopy(ctx.entry)
    entry["gt_invariant"] = {
        "marker": ctx.marker,
        "original_invariant": ctx.original_invariant,
        "pretty_invariant": ctx.pretty_invariant,
        "simplified_invariant": candidate.content if is_simplified else None,
        "rationale": candidate.rationale,
        "target_invariant": candidate.content,  # The one that was verified
        "is_correct": quality.is_correct,
        "does_target_property_still_hold": quality.does_target_property_still_hold,
        "has_speedup": quality.has_speedup,
        "speedup": quality.speedup,
        "quality_grade": quality.quality_grade,
    }
    return entry


# =============================================================================
# Processing Functions
# =============================================================================

def process_verbose(
    ctx: InvariantContext,
    client: Together,
    verifier: UAutomizerVerifier,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Process verbose invariant: generate candidates, verify, filter."""
    candidates = generate_candidates(client, ctx, cfg)
    
    for i, c in enumerate(candidates):
        logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}, Candidate {i}: {c.content}")
    
    if not candidates:
        return []
    
    if not cfg.pipeline.verify:
        logger.warning("Verification disabled")
        return []
    
    verified = verify_candidates_parallel(
        verifier, candidates, ctx.program, ctx.marker,
        ctx.baseline_decision, ctx.baseline_timing,
        cfg.verifier.max_parallel_workers,
    )
    
    # Filter and create entries for qualifying candidates
    results = []
    for candidate, quality in verified:
        if quality.quality_grade > 0:
            results.append(create_entry(ctx, candidate, quality, is_simplified=True))
    
    # If all simplified candidates failed, fall back to verifying the prettified original
    if not results:
        logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: All {len(verified)} simplified candidates failed. Falling back to prettified invariant.")
        fallback_candidate = CandidateInvariant(
            content=ctx.invariant_to_process,  # This is the prettified invariant
            rationale=f"All {len(verified)} simplified candidates failed verification, using prettified original invariant",
        )
        fallback_quality = verify_candidate(
            verifier, fallback_candidate, ctx.program, ctx.marker,
            ctx.baseline_decision, ctx.baseline_timing,
        )
        if fallback_quality.quality_grade > 0:
            results.append(create_entry(ctx, fallback_candidate, fallback_quality, is_simplified=False))
            logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: Prettified invariant passed (grade={fallback_quality.quality_grade})")
        else:
            logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: Prettified invariant also failed (grade={fallback_quality.quality_grade})")
    
    logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: {len(results)}/{len(verified)} passed")
    return results


def process_non_verbose(
    ctx: InvariantContext,
    verifier: UAutomizerVerifier,
) -> list[dict[str, Any]]:
    """Process non-verbose invariant: verify as-is."""
    logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: Not verbose, verifying as-is")
    
    candidate = CandidateInvariant(
        content=ctx.invariant_to_process,
        rationale="Not simplified (not verbose enough)",
    )
    
    quality = verify_candidate(
        verifier, candidate, ctx.program, ctx.marker,
        ctx.baseline_decision, ctx.baseline_timing,
    )
    
    logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: grade={quality.quality_grade}")
    
    if quality.quality_grade > 0:
        return [create_entry(ctx, candidate, quality, is_simplified=False)]
    return []


def process_invariant(
    ctx: InvariantContext,
    client: Together,
    verifier: UAutomizerVerifier,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Process a single invariant (verbose or non-verbose)."""
    if not cfg.pipeline.simplify:
        return []
    
    if is_verbose_invariant(ctx.invariant_to_process, cfg):
        return process_verbose(ctx, client, verifier, cfg)
    else:
        return process_non_verbose(ctx, verifier)


# =============================================================================
# Main Pipeline
# =============================================================================

def preprocess_gt_invariants_parallel(
    data: list[dict[str, Any]],
    client: Together,
    cfg: DictConfig,
    verifier: UAutomizerVerifier,
) -> list[dict[str, Any]]:
    """Main preprocessing pipeline with resume support."""
    logger.info(f"Processing {len(data)} entries (N={cfg.verifier.max_parallel_workers} candidates)")
    
    output_path = Path(cfg.output.preprocessed_invariants_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = get_checkpoint_path(output_path)
    
    # Load existing progress for resume capability
    processed = load_existing_progress(output_path)
    
    # Load attempted keys from checkpoint (includes both successes AND failures)
    attempted_keys = load_attempted_keys(checkpoint_path)
    
    # If no checkpoint exists, rebuild from successful entries
    if not attempted_keys:
        attempted_keys = build_processed_keys(processed)
        if attempted_keys:
            logger.info(f"Rebuilt {len(attempted_keys)} attempted keys from output file")
    
    skipped_count = 0
    
    for i, entry in enumerate(tqdm(data, desc="Processing")):
        baseline_timing = entry.get("timings", {}).get("median", 0.0)
        baseline_decision = entry.get("baseline_decision", "UNKNOWN")
        original_program = entry.get("original_program", "")
        file_name = entry.get("file", "")
        
        # Skip invalid entries
        if baseline_decision not in {"TRUE", "FALSE"} or not entry.get("invariants"):
            logger.warning(f"Entry {i}: skipping (invalid baseline or no invariants)")
            continue
        
        # Parse program once per entry
        program = AstProgram().from_code(original_program)
        program.process(print_ast=False)
        
        for j, inv in enumerate(entry["invariants"]):
            original_inv = inv.get("invariant", "")
            marker = inv.get("marker", "")
            
            # Check if this invariant was already attempted (resume support)
            key = (file_name, marker, original_inv)
            if key in attempted_keys:
                skipped_count += 1
                if skipped_count % 100 == 0:
                    logger.info(f"Skipped {skipped_count} already-attempted invariants...")
                continue
            
            pretty_inv = ""
            if cfg.pipeline.prettify:
                pretty_inv = clean_invariant(original_inv, pretty=True)
            invariant_to_process = pretty_inv if cfg.pipeline.prettify else original_inv
            ctx = InvariantContext(
                entry=entry,
                entry_idx=i,
                inv_idx=j,
                marker=marker,
                original_invariant=original_inv,
                pretty_invariant=pretty_inv,
                invariant_to_process=invariant_to_process,
                baseline_decision=baseline_decision,
                baseline_timing=baseline_timing,
                program=program,
                original_program=original_program,
            )
            
            logger.info(f"Entry {i}, Inv {j}: marker={ctx.marker}, invariant_to_process={ctx.invariant_to_process[:80]}..., timing={baseline_timing:.2f}s")
            
            try:
                new_entries = process_invariant(ctx, client, verifier, cfg)
                processed.extend(new_entries)
                
                # Mark this invariant as attempted (success or failure)
                attempted_keys.add(key)
                
                # Save intermediate results after each invariant
                with open(output_path, "w") as f:
                    json.dump(processed, f, indent=2)
                
                # Save checkpoint with all attempted keys
                save_checkpoint(checkpoint_path, attempted_keys)
                
            except Exception as e:
                logger.error(f"Entry {i}, Inv {j}: Failed with error: {type(e).__name__}: {e}. Skipping...")
                # Still mark as attempted so we don't retry on crash
                attempted_keys.add(key)
                save_checkpoint(checkpoint_path, attempted_keys)
                continue
    
    if skipped_count > 0:
        logger.info(f"Resumed run: skipped {skipped_count} already-attempted invariants")
    logger.info(f"Total qualifying entries: {len(processed)}")
    
    # Push to hub
    if cfg.output.push_to_hub:
        dataset = Dataset.from_list(processed)
        hub_name = f"{cfg.output.hub_organization}/{cfg.output.hub_name}"
        logger.info(f"Pushing to: {hub_name}")
        DatasetDict({"train": dataset}).push_to_hub(hub_name)
    
    return processed


# =============================================================================
# Resume Support
# =============================================================================

def get_checkpoint_path(output_path: Path) -> Path:
    """Get the checkpoint file path based on the output path."""
    return output_path.parent / f".{output_path.stem}.checkpoint.json"


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
            logger.info(f"Loaded {len(data)} existing processed entries from {output_path}")
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load existing progress: {e}. Starting fresh.")
    return []


# =============================================================================
# Entry Point
# =============================================================================

def load_data(path: Path, limit: int = -1) -> list[dict[str, Any]]:
    """Load JSON data with optional limit."""
    with open(path) as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]
    logger.info(f"Loaded {len(data)} entries from {path}")
    return data


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="preprocess_gt_invariants_parallel",
)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.verifier.property,
        arch=cfg.verifier.arch,
        timeout_seconds=cfg.verifier.timeout,
        version=cfg.verifier.version,
    )
    
    data = load_data(Path(cfg.dataset.name), cfg.dataset.limit)
    
    if cfg.weave.get("use_weave", False):
        weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")
    
    client = Together()
    preprocess_gt_invariants_parallel(data, client, cfg, verifier)


if __name__ == "__main__":
    main()
