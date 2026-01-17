#!/usr/bin/env python3
"""
Analyze training data to count how many invariants fall into verbose vs non-verbose paths.
"""

import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.preprocess.clean_invariants import clean_invariant

def is_verbose_invariant(inv_text: str, cfg: DictConfig) -> bool:
    """Check if an invariant matches the verbose criteria for simplification."""
    return cfg.filtering.min_char_length <= len(inv_text)

def analyze_data(data_path: str, config_path: str):
    """Analyze the data and count verbose vs non-verbose invariants."""
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Load data
    print(f"Loading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    # Statistics
    total_invariants = 0
    verbose_count = 0
    non_verbose_count = 0
    skipped_entries = 0
    
    # Detailed stats
    verbose_stats = {
        'char_lengths': []
    }
    non_verbose_stats = {
        'char_lengths': []
    }
    
    # Process each entry
    for i, entry in enumerate(data):
        baseline_decision = entry.get("baseline_decision", "UNKNOWN")
        invariants = entry.get("invariants", [])
        
        # Skip invalid entries (same logic as preprocessing)
        if baseline_decision not in {"TRUE", "FALSE"} or not invariants:
            skipped_entries += 1
            continue
        
        # Process each invariant
        for j, inv in enumerate(invariants):
            original_inv = inv.get("invariant", "")
            if not original_inv:
                continue
            
            total_invariants += 1
            
            # Apply prettify if configured
            if cfg.pipeline.prettify:
                try:
                    invariant_to_process = clean_invariant(original_inv, pretty=True)
                except Exception as e:
                    # If prettification fails, use original
                    print(f"Warning: Prettification failed for entry {i}, inv {j}: {e}")
                    invariant_to_process = original_inv
            else:
                invariant_to_process = original_inv
            
            # Count char length
            char_length = len(invariant_to_process)
            
            # Classify
            is_verbose = is_verbose_invariant(invariant_to_process, cfg)
            
            if is_verbose:
                verbose_count += 1
                verbose_stats['char_lengths'].append(char_length)
            else:
                non_verbose_count += 1
                non_verbose_stats['char_lengths'].append(char_length)
    
    # Print results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print(f"\nTotal entries processed: {len(data) - skipped_entries}")
    print(f"Skipped entries (invalid baseline/no invariants): {skipped_entries}")
    print(f"Total invariants analyzed: {total_invariants}")
    print(f"\nVERBOSE path: {verbose_count} ({verbose_count/total_invariants*100:.2f}%)")
    print(f"NON-VERBOSE path: {non_verbose_count} ({non_verbose_count/total_invariants*100:.2f}%)")
    
    # Print filtering criteria
    print(f"\nFiltering criteria:")
    print(f"  min_char_length: {cfg.filtering.min_char_length}")
    print(f"  prettify: {cfg.pipeline.prettify}")
    
    # Print statistics
    if verbose_stats['char_lengths']:
        print(f"\nVERBOSE path statistics:")
        print(f"  Char length: min={min(verbose_stats['char_lengths'])}, max={max(verbose_stats['char_lengths'])}, avg={sum(verbose_stats['char_lengths'])/len(verbose_stats['char_lengths']):.2f}")
    
    if non_verbose_stats['char_lengths']:
        print(f"\nNON-VERBOSE path statistics:")
        print(f"  Char length: min={min(non_verbose_stats['char_lengths'])}, max={max(non_verbose_stats['char_lengths'])}, avg={sum(non_verbose_stats['char_lengths'])/len(non_verbose_stats['char_lengths']):.2f}")
    
    # Breakdown of why non-verbose
    if non_verbose_stats['char_lengths']:
        print(f"\nNON-VERBOSE breakdown:")
        print(f"  Too short (< {cfg.filtering.min_char_length} chars): {sum(1 for l in non_verbose_stats['char_lengths'] if l < cfg.filtering.min_char_length)}")

if __name__ == "__main__":
    data_path = "data/train/invbench-train-uautomizer25-k1-full-raw/invbench-train-uautomizer25-k1-full-raw.json"
    config_path = "configs/preprocess/preprocess_gt_invariants_parallel.yaml"
    
    analyze_data(data_path, config_path)
