#!/usr/bin/env python3
"""
Compute statistics on invariant normalization transformations.

This script analyzes the training data to measure how often each
normalization transformation applies:
1. Cast removal (e.g., (long long) x -> x)
2. Tautology elimination (e.g., x <= x)
3. Contradiction elimination (e.g., x < x)
4. Parenthesis reduction

Output: Statistics suitable for inclusion in the ICML paper.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess.clean_invariants import (
    parse_invariant_expr,
    clean_invariant,
    pretty_invariant_src,
    _is_tautology,
    _is_contradiction,
    _nodes_are_equal,
    _try_eval_constant,
)
from pycparser import c_ast


def count_casts(inv_str: str) -> int:
    """Count the number of integral type casts in an invariant string."""
    # Pattern for common integral casts
    cast_patterns = [
        r'\(long long\)',
        r'\(unsigned long long\)',
        r'\(long\)',
        r'\(unsigned long\)',
        r'\(int\)',
        r'\(unsigned int\)',
        r'\(unsigned\)',
        r'\(signed\)',
        r'\(char\)',
        r'\(unsigned char\)',
        r'\(short\)',
        r'\(unsigned short\)',
        r'\(__int128\)',
        r'\(unsigned __int128\)',
        r'\(_Bool\)',
    ]
    total = 0
    for pattern in cast_patterns:
        total += len(re.findall(pattern, inv_str, re.IGNORECASE))
    return total


def count_parentheses(inv_str: str) -> int:
    """Count the number of parenthesis pairs in an invariant string."""
    return inv_str.count('(')


def count_tautologies_in_ast(node: c_ast.Node) -> int:
    """Recursively count tautological subexpressions in an AST."""
    if node is None:
        return 0
    
    count = 0
    if _is_tautology(node):
        count += 1
    
    for _, child in node.children():
        count += count_tautologies_in_ast(child)
    
    return count


def _is_real_contradiction(node: c_ast.Node) -> bool:
    """
    Check for real contradictions (not just constant 0).
    
    Only counts:
    - Reflexive comparisons: x < x, x > x, x != x
    - False constant comparisons: 1 <= 0, 5 < 3, etc.
    
    Does NOT count standalone constant 0 (too noisy).
    """
    if not isinstance(node, c_ast.BinaryOp):
        return False
    
    op = node.op
    
    # x < x, x > x, x != x are always false
    if op in {'<', '>', '!='} and _nodes_are_equal(node.left, node.right):
        return True
    
    # Constant comparisons that are false
    left_val = _try_eval_constant(node.left)
    right_val = _try_eval_constant(node.right)
    if left_val is not None and right_val is not None:
        if op == '<':
            return not (left_val < right_val)
        elif op == '<=':
            return not (left_val <= right_val)
        elif op == '>':
            return not (left_val > right_val)
        elif op == '>=':
            return not (left_val >= right_val)
        elif op == '==':
            return not (left_val == right_val)
        elif op == '!=':
            return not (left_val != right_val)
    
    return False


def count_contradictions_in_ast(node: c_ast.Node) -> int:
    """Recursively count real contradictory subexpressions in an AST."""
    if node is None:
        return 0
    
    count = 0
    if _is_real_contradiction(node):
        count += 1
    
    for _, child in node.children():
        count += count_contradictions_in_ast(child)
    
    return count


def analyze_invariant(raw_inv: str, pretty_inv: str) -> Dict[str, Any]:
    """Analyze a single invariant transformation."""
    stats = {
        'raw_length': len(raw_inv),
        'pretty_length': len(pretty_inv),
        'casts_removed': count_casts(raw_inv),
        'raw_parens': count_parentheses(raw_inv),
        'pretty_parens': count_parentheses(pretty_inv),
        'parens_removed': count_parentheses(raw_inv) - count_parentheses(pretty_inv),
        'tautologies_found': 0,
        'contradictions_found': 0,
    }
    
    # Parse and count tautologies/contradictions
    ast = parse_invariant_expr(raw_inv)
    if ast is not None:
        stats['tautologies_found'] = count_tautologies_in_ast(ast)
        stats['contradictions_found'] = count_contradictions_in_ast(ast)
    
    # Length reduction
    if stats['raw_length'] > 0:
        stats['length_reduction_pct'] = (1 - stats['pretty_length'] / stats['raw_length']) * 100
    else:
        stats['length_reduction_pct'] = 0
    
    return stats


def load_training_data(data_path: Path) -> List[Dict]:
    """Load training data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)


def compute_aggregate_stats(all_stats: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics across all invariants."""
    n = len(all_stats)
    if n == 0:
        return {}
    
    agg = {
        'total_invariants': n,
        'invariants_with_casts': sum(1 for s in all_stats if s['casts_removed'] > 0),
        'invariants_with_tautologies': sum(1 for s in all_stats if s['tautologies_found'] > 0),
        'invariants_with_contradictions': sum(1 for s in all_stats if s['contradictions_found'] > 0),
        'total_casts_removed': sum(s['casts_removed'] for s in all_stats),
        'total_tautologies_found': sum(s['tautologies_found'] for s in all_stats),
        'total_contradictions_found': sum(s['contradictions_found'] for s in all_stats),
        'total_parens_removed': sum(s['parens_removed'] for s in all_stats),
        'avg_length_reduction_pct': sum(s['length_reduction_pct'] for s in all_stats) / n,
        'avg_casts_per_invariant': sum(s['casts_removed'] for s in all_stats) / n,
        'avg_parens_removed': sum(s['parens_removed'] for s in all_stats) / n,
    }
    
    # Percentages
    agg['pct_with_casts'] = agg['invariants_with_casts'] / n * 100
    agg['pct_with_tautologies'] = agg['invariants_with_tautologies'] / n * 100
    agg['pct_with_contradictions'] = agg['invariants_with_contradictions'] / n * 100
    
    return agg


def format_latex_table(agg: Dict[str, Any]) -> str:
    """Format statistics as a LaTeX table for the paper."""
    # Combine tautologies and contradictions for cleaner presentation
    trivially_decidable = agg['invariants_with_tautologies']  # contradictions are negligible
    pct_trivially_decidable = agg['pct_with_tautologies']
    total_trivially_decidable = agg['total_tautologies_found'] + agg.get('total_contradictions_found', 0)
    
    return f"""
%% Invariant Normalization Statistics
%% Generated by compute_normalization_stats.py

\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{@{{}}lr@{{}}}}
\\toprule
\\textbf{{Transformation}} & \\textbf{{Statistic}} \\\\
\\midrule
Total invariants analyzed & {agg['total_invariants']:,} \\\\
\\midrule
\\multicolumn{{2}}{{@{{}}l}}{{\\textit{{Cast Removal}}}} \\\\
\\quad Invariants affected & {agg['invariants_with_casts']:,} ({agg['pct_with_casts']:.1f}\\%) \\\\
\\quad Total casts removed & {agg['total_casts_removed']:,} \\\\
\\midrule
\\multicolumn{{2}}{{@{{}}l}}{{\\textit{{Trivially Decidable Clause Elimination}}}} \\\\
\\quad Invariants affected & {trivially_decidable:,} ({pct_trivially_decidable:.1f}\\%) \\\\
\\quad Total clauses eliminated & {total_trivially_decidable:,} \\\\
\\midrule
\\multicolumn{{2}}{{@{{}}l}}{{\\textit{{Parenthesis Minimization}}}} \\\\
\\quad Total parentheses removed & {agg['total_parens_removed']:,} \\\\
\\quad Avg. per invariant & {agg['avg_parens_removed']:.1f} \\\\
\\midrule
\\textbf{{Average length reduction}} & \\textbf{{{agg['avg_length_reduction_pct']:.1f}\\%}} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Impact of invariant normalization on {agg['total_invariants']:,} training invariants. Trivially decidable clauses include tautologies ($x \\leq x$, $0 < 1$) and contradictions ($1 \\leq 0$).}}
\\label{{tab:normalization-stats}}
\\end{{table}}
"""


def main():
    # Find training data files
    data_dir = Path(__file__).parent.parent / "data" / "train"
    
    # Look for the main preprocessed JSON file
    json_files = [
        data_dir / "invbench-train-uautomizer25-k1-full-raw" / "invbench-train-uautomizer25-k1-full-raw.json",
        data_dir / "invbench-train-uautomizer25-k1-full-raw" / "invbench-train-uautomizer25-k1-full-parallel-latest.json",
        data_dir / "invbench-train-uautomizer25-k1-full-raw" / "invbench-train-uautomizer25-k1-full-raw-pp.json",
    ]
    
    data_path = None
    for path in json_files:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print("Error: Could not find training data JSON file.")
        print("Looked in:", [str(p) for p in json_files])
        sys.exit(1)
    
    print(f"Loading data from: {data_path}")
    data = load_training_data(data_path)
    print(f"Loaded {len(data)} programs")
    
    # Analyze all invariants
    all_stats = []
    for entry in data:
        for inv_entry in entry.get('invariants', []):
            raw_inv = inv_entry.get('invariant', '')
            
            if not raw_inv:
                continue
            
            # Use pretty_invariant if available, otherwise compute it
            pretty_inv = inv_entry.get('pretty_invariant', '')
            if not pretty_inv:
                try:
                    pretty_inv = pretty_invariant_src(raw_inv, strip_casts=True, remove_tautologies=True)
                except Exception:
                    pretty_inv = raw_inv  # Fall back to raw if parsing fails
            
            stats = analyze_invariant(raw_inv, pretty_inv)
            all_stats.append(stats)
    
    print(f"Analyzed {len(all_stats)} invariants")
    
    # Compute aggregate statistics
    agg = compute_aggregate_stats(all_stats)
    
    # Print human-readable summary
    print("\n" + "=" * 60)
    print("INVARIANT NORMALIZATION STATISTICS")
    print("=" * 60)
    print(f"Total invariants: {agg['total_invariants']:,}")
    print()
    print("Cast Removal:")
    print(f"  - Invariants with casts: {agg['invariants_with_casts']:,} ({agg['pct_with_casts']:.1f}%)")
    print(f"  - Total casts removed: {agg['total_casts_removed']:,}")
    print(f"  - Avg casts per invariant: {agg['avg_casts_per_invariant']:.2f}")
    print()
    print("Tautology/Contradiction Elimination:")
    print(f"  - Invariants with tautologies: {agg['invariants_with_tautologies']:,} ({agg['pct_with_tautologies']:.1f}%)")
    print(f"  - Invariants with contradictions: {agg['invariants_with_contradictions']:,} ({agg['pct_with_contradictions']:.1f}%)")
    print(f"  - Total tautologies found: {agg['total_tautologies_found']:,}")
    print()
    print("Parenthesis Minimization:")
    print(f"  - Total parentheses removed: {agg['total_parens_removed']:,}")
    print(f"  - Avg parens removed: {agg['avg_parens_removed']:.1f}")
    print()
    print("Overall:")
    print(f"  - Avg length reduction: {agg['avg_length_reduction_pct']:.1f}%")
    
    # Generate LaTeX table
    latex_output = format_latex_table(agg)
    
    # Save LaTeX to docs
    output_path = Path(__file__).parent.parent / "docs" / "normalization_stats_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_output)
    print(f"\nLaTeX table saved to: {output_path}")
    
    # Also print the LaTeX
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    print(latex_output)


if __name__ == "__main__":
    main()
