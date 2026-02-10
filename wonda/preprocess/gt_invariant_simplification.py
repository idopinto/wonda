#!/usr/bin/env python3
"""Ground-truth invariant simplification and verification.

For each invariant:
  - If verbose: generate N simplified candidates via LLM, verify in parallel.
  - If non-verbose: verify as-is.
Filter by quality_grade > 0 and return qualifying entries.
"""

import copy
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import weave
from omegaconf import DictConfig
from together import Together
from together.error import ServiceUnavailableError, RateLimitError

from wonda.core.ast_program import AstProgram
from wonda.core.property import Property
from wonda.preprocess.pipe_utils import call_together_api, parse_llm_response
from wonda.eval.validate import syntactic_validation
from wonda.verifiers.uautomizer_runlim import UAutomizerVerifier, VerifierCallReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class CandidateQualityGrade:
    is_correct: bool = field(default=False)
    is_sufficient: bool = field(default=False)
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
    elif not result.is_sufficient:
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
) -> tuple[CandidateQualityGrade, dict[str, float]]:
    """Verify a single candidate (correctness + sufficiency in parallel)."""
    grade = CandidateQualityGrade()
    prop = Property(content=candidate.content, marker_name=marker)
    
    # Syntactic validation
    if not syntactic_validation(prop.content):
        grade.quality_grade = 0
        logger.warning(f"Syntactically invalid: '{prop.content[:50]}...'")
        return grade, {}
    
    # Filter trivial invariants (UAutomizer artifacts)
    if is_degenerate_invariant(prop.content):
        grade.quality_grade = 0
        logger.warning(f"Degenerate invariant rejected: '{prop.content}'")
        return grade, {}
    
    # Prepare verification programs
    correctness_program = program.get_program_with_assertion(
        property_to_assert=prop, assumptions=[], for_llm=False
    )
    usefulness_program = program.get_program_with_assertion(
        property_to_assert=program.get_target_assert(), assumptions=[prop], for_llm=False
    )
    
    # Run both checks in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        correctness_future = executor.submit(run_verifier, verifier, correctness_program, "correctness")
        usefulness_future = executor.submit(run_verifier, verifier, usefulness_program, "usefulness")
        
        correctness_result = correctness_future.result()
        usefulness_result = usefulness_future.result()
    
    # Process results
    grade.is_correct = correctness_result.decision == "TRUE"
    grade.is_sufficient = usefulness_result.decision == "TRUE"
    # Total verification time is max of both (they run in parallel)
    total_verification_time = max(correctness_result.time_taken, usefulness_result.time_taken)

    timings = {
        "correctness": correctness_result.time_taken,
        "sufficiency": usefulness_result.time_taken,
        "total": total_verification_time,
    }
    grade.has_speedup = total_verification_time < baseline_timing
    grade.speedup = (baseline_timing / total_verification_time) if total_verification_time > 0 else 0.0
    grade.quality_grade = compute_quality_grade(grade)
    return grade, timings


def deduplicate_candidates(candidates: list[CandidateInvariant]) -> list[CandidateInvariant]:
    """Remove exact duplicates (by normalized content). Allows for semantically equivalent variants such as "a <= b" and "b >= a" (for data augmentation)."""
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
) -> list[tuple[CandidateInvariant, CandidateQualityGrade, dict[str, float]]]:
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
                quality, timings = future.result()
                results.append((candidate, quality, timings))
                logger.info(f"'{candidate.content[:80]}...' -> grade={quality.quality_grade}")
            except Exception as e:
                logger.error(f"Verification error: {e}")
                results.append((candidate, CandidateQualityGrade(quality_grade=0), {}))
    
    return results


# =============================================================================
# Entry Creation
# =============================================================================

def create_entry(
    ctx: InvariantContext,
    candidate: CandidateInvariant,
    quality: CandidateQualityGrade,
    timings: dict[str, float],
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
        "is_sufficient": quality.is_sufficient,
        "has_speedup": quality.has_speedup,
        "speedup": quality.speedup,
        "quality_grade": quality.quality_grade,
        "verification_timings": timings,
    }
    return entry


# =============================================================================
# LLM Candidate Generation
# =============================================================================

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
            n=cfg.model.n,
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
# Processing Functions
# =============================================================================

def process_verbose(
    ctx: InvariantContext,
    client: Together,
    verifier: UAutomizerVerifier,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Process verbose invariant: generate candidates, verify, filter."""
    logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: Verbose invariant (len={len(ctx.invariant_to_process)}), simplifying")
    candidates = generate_candidates(client, ctx, cfg)
    
    for i, c in enumerate(candidates):
        logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}, Candidate {i+1}/{len(candidates)}: {c.content}")
    
    if not candidates:
        return []
    
    if not cfg.pipe.verify:
        logger.warning("Verification disabled")
        return [create_entry(ctx, c, CandidateQualityGrade(), {}, is_simplified=True) for c in candidates]
    
    verified = verify_candidates_parallel(
        verifier, candidates, ctx.program, ctx.marker,
        ctx.baseline_decision, ctx.baseline_timing,
        cfg.model.n,
    )
    
    # Filter and create entries for qualifying candidates
    results = []
    for candidate, quality, timings in verified:
        if quality.quality_grade > 0:
            results.append(create_entry(ctx, candidate, quality,timings, is_simplified=True))
    
    # If all simplified candidates failed, fall back to verifying the normalized original   
    if not results:
        logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: All {len(verified)} simplified candidates failed. Falling back to normalized invariant.")
        fallback_candidate = CandidateInvariant(
            content=ctx.invariant_to_process,  # This is the normalized invariant
            rationale=f"All {len(verified)} simplified candidates failed verification, using normalized original invariant",
        )
        fallback_quality, timings = verify_candidate(
            verifier, fallback_candidate, ctx.program, ctx.marker,
            ctx.baseline_decision, ctx.baseline_timing,
        )
        if fallback_quality.quality_grade > 0:
            results.append(create_entry(ctx, fallback_candidate, fallback_quality, timings, is_simplified=False))
            logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: Normalized invariant passed (grade={fallback_quality.quality_grade})")
        else:
            logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: Normalized invariant also failed (grade={fallback_quality.quality_grade})")
    
    logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: {len(results)}/{len(verified)} passed")
    return results


def process_non_verbose(
    ctx: InvariantContext,
    verifier: UAutomizerVerifier,
) -> list[dict[str, Any]]:
    """Process non-verbose invariant: verify as-is."""
    if is_degenerate_invariant(ctx.invariant_to_process):
        logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: Degenerate invariant, skipping")
        return []
    logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: Not verbose, not degenerate, verifying as-is")
    
    candidate = CandidateInvariant(
        content=ctx.invariant_to_process,
        rationale="Not simplified (not verbose enough)",
    )
    
    quality, timings = verify_candidate(
        verifier, candidate, ctx.program, ctx.marker,
        ctx.baseline_decision, ctx.baseline_timing,
    )
    
    logger.info(f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: grade={quality.quality_grade}")
    
    if quality.quality_grade > 0:
        return [create_entry(ctx, candidate, quality, timings, is_simplified=False)]
    return []


def is_degenerate_invariant(content: str) -> bool:
    """Check if an invariant is degenerate (0, 1, true, false).

    These are UAutomizer artifacts that carry no useful information and
    should be skipped before spending LLM / verifier budget on them.
    """
    if not content:
        return True
    normalized = content.strip().lower().replace("(", "").replace(")", "").replace(" ", "").replace("!", "")
    return normalized in {"0", "1", "true", "false", ""}


def process_invariant(
    ctx: InvariantContext,
    client: Together,
    verifier: UAutomizerVerifier,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Process a single invariant (verbose or non-verbose)."""
    if not cfg.pipe.simplify or not is_verbose_invariant(ctx.invariant_to_process, cfg):
        results = process_non_verbose(ctx, verifier)
    else:
        results = process_verbose(ctx, client, verifier, cfg)
    return results