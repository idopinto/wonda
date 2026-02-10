import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from wonda.verifiers.uautomizer_runlim import UAutomizerVerifier, VerifierCallReport, write_file
from wonda.core.ast_program import AstProgram
from wonda.core.property import Property
import tempfile
from configs import global_config as GC
from logging import getLogger
from tqdm import tqdm
logger = getLogger(__name__)



def verify_correctness_with_temp_dir(
    verifier: UAutomizerVerifier, program_to_verify: str, program_name: str, predicate: Property
) -> VerifierCallReport:
    """Verify if the invariant is correct."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reports_dir = Path(temp_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        to_verify_path = reports_dir / f"check_{program_name}_{predicate.marker_name}.c"
        write_file(to_verify_path, program_to_verify)
        report = verifier.verify(
            program_path=to_verify_path,
            reports_dir=reports_dir,
            timeout_seconds=verifier.timeout_seconds,
        )
    print(
        f"Report: {report.decision} ({report.decision_reason}) in {report.time_taken} seconds"
    )
    return report

def test_gt_invariants(
    data: list[dict[str, Any]], 
    verifier: UAutomizerVerifier, 
    limit: int = -1,
    save_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Test correctness of original, pretty, and simplified invariants.
    
    Args:
        data: List of items with invariants to test
        verifier: UAutomizer verifier instance
        limit: Maximum number of items to process (-1 for all)
        save_path: Path to save results JSON (if None, auto-generates timestamped path)
    
    Returns a detailed report with statistics and failure lists for each invariant type.
    """
    data = data[:limit] if limit != -1 else data
    
    # Statistics tracking
    stats = {
        'original': {'total': 0, 'correct': 0, 'failures': [], 'times': []},
        'pretty': {'total': 0, 'correct': 0, 'failures': [], 'skipped': 0, 'times': []},
        'simplified': {'total': 0, 'correct': 0, 'failures': [], 'skipped': 0, 'same_as_pretty': 0, 'times': []},
    }
    
    # Length reduction tracking
    length_stats = {
        'original_to_pretty': [],      # (original_len, pretty_len, reduction_pct)
        'pretty_to_simplified': [],    # (pretty_len, simplified_len, reduction_pct)
        'original_to_simplified': [],  # (original_len, simplified_len, reduction_pct)
    }
    
    # Time comparison tracking (when both verified successfully)
    time_comparisons = []  # List of {file, marker, original_time, pretty_time, simplified_time, speedup}
    
    for item in tqdm(data, desc="Processing items"):
        program_name = item['file']
        print(f"\n{'='*60}")
        print(f"Processing program {program_name} | {len(item['invariants'])} invariants")
        print(f"{'='*60}")
        
        program = AstProgram().from_code(item['original_program'])
        program.process(print_ast=False)
        
        for invariant in item['invariants']:
            marker = invariant['marker']
            
            # Get all invariant versions
            original_inv = invariant.get('invariant')
            pretty_inv = invariant.get('pretty_invariant')
            simplified_inv = invariant.get('simplified_invariant') or invariant.get('generalized_invariant')
            
            print(f"\n--- Marker: {marker} ---")
            
            # Calculate and display length reductions
            if original_inv and pretty_inv:
                orig_len = len(original_inv)
                pretty_len = len(pretty_inv)
                reduction_pct = ((orig_len - pretty_len) / orig_len * 100) if orig_len > 0 else 0
                length_stats['original_to_pretty'].append((orig_len, pretty_len, reduction_pct))
                print(f"  📏 Original→Pretty: {orig_len} → {pretty_len} chars ({reduction_pct:+.1f}% reduction)")
            
            if pretty_inv and simplified_inv and simplified_inv != pretty_inv:
                pretty_len = len(pretty_inv)
                simp_len = len(simplified_inv)
                reduction_pct = ((pretty_len - simp_len) / pretty_len * 100) if pretty_len > 0 else 0
                length_stats['pretty_to_simplified'].append((pretty_len, simp_len, reduction_pct))
                print(f"  📏 Pretty→Simplified: {pretty_len} → {simp_len} chars ({reduction_pct:+.1f}% reduction)")
            
            if original_inv and simplified_inv and simplified_inv != pretty_inv:
                orig_len = len(original_inv)
                simp_len = len(simplified_inv)
                reduction_pct = ((orig_len - simp_len) / orig_len * 100) if orig_len > 0 else 0
                length_stats['original_to_simplified'].append((orig_len, simp_len, reduction_pct))
                print(f"  📏 Original→Simplified: {orig_len} → {simp_len} chars ({reduction_pct:+.1f}% reduction)")
            
            # Track times for this invariant
            inv_times = {'original': None, 'pretty': None, 'simplified': None}
            inv_verified = {'original': False, 'pretty': False, 'simplified': False}
            
            # # Test original invariant
            # if original_inv:
            #     print(f"[ORIGINAL] {original_inv[:100]}{'...' if len(original_inv) > 100 else ''}")
            #     predicate = Property(content=original_inv, marker_name=marker)
            #     program_to_verify = program.get_program_with_assertion(
            #         predicate=predicate, assumptions=[], for_llm=False
            #     )
            #     print(f"  Program to verify:\n{program_to_verify}")
            #     report = verify_correctness_with_temp_dir(verifier, program_to_verify, program_name, predicate)
            #     stats['original']['total'] += 1
            #     stats['original']['times'].append(report.time_taken)
            #     inv_times['original'] = report.time_taken
            #     if report.decision == "TRUE":
            #         stats['original']['correct'] += 1
            #         inv_verified['original'] = True
            #         print(f"  ✅ CORRECT ({report.time_taken:.2f}s)")
            #     else:
            #         stats['original']['failures'].append({
            #             'file': program_name,
            #             'marker': marker,
            #             'invariant': original_inv,
            #             'reason': report.decision_reason,
            #             'time': report.time_taken
            #         })
            #         print(f"  ❌ FAILED: {report.decision} - {report.decision_reason}")
            
            # Test pretty invariant
            # if pretty_inv:
            #     print(f"[PRETTY] {pretty_inv[:100]}{'...' if len(pretty_inv) > 100 else ''}")
            #     predicate = Property(content=pretty_inv, marker_name=marker)
            #     program_to_verify = program.get_program_with_assertion(
            #         predicate=predicate, assumptions=[], for_llm=False
            #     )
            #     print(f"  Program to verify:\n{program_to_verify}")
            #     report = verify_correctness_with_temp_dir(verifier, program_to_verify, program_name, predicate)
            #     stats['pretty']['total'] += 1
            #     stats['pretty']['times'].append(report.time_taken)
            #     inv_times['pretty'] = report.time_taken
            #     if report.decision == "TRUE":
            #         stats['pretty']['correct'] += 1
            #         inv_verified['pretty'] = True
            #         print(f"  ✅ CORRECT ({report.time_taken:.2f}s)")
            #     else:
            #         stats['pretty']['failures'].append({
            #             'file': program_name,
            #             'marker': marker,
            #             'invariant': pretty_inv,
            #             'original': original_inv,
            #             'reason': report.decision_reason,
            #             'time': report.time_taken
            #         })
            #         print(f"  ❌ FAILED: {report.decision} - {report.decision_reason}")
            # else:
            #     stats['pretty']['skipped'] += 1
            
            # # Test simplified invariant (skip if same as pretty)
            # if simplified_inv:
            #     if simplified_inv == pretty_inv:
            #         stats['simplified']['same_as_pretty'] += 1
            #         print(f"[SIMPLIFIED] ⏭️ Skipped (same as pretty)")
            #     else:
            #         print(f"[SIMPLIFIED] {simplified_inv[:100]}{'...' if len(simplified_inv) > 100 else ''}")
            #         predicate = Property(content=simplified_inv, marker_name=marker)
            #         program_to_verify = program.get_program_with_assertion(
            #             predicate=predicate, assumptions=[], for_llm=False
            #         )
            #         print(f"  Program to verify:\n{program_to_verify}")
            #         report = verify_correctness_with_temp_dir(verifier, program_to_verify, program_name, predicate)
            #         stats['simplified']['total'] += 1
            #         stats['simplified']['times'].append(report.time_taken)
            #         inv_times['simplified'] = report.time_taken
            #         if report.decision == "TRUE":
            #             stats['simplified']['correct'] += 1
            #             inv_verified['simplified'] = True
            #             print(f"  ✅ CORRECT ({report.time_taken:.2f}s)")
            #         else:
            #             stats['simplified']['failures'].append({
            #                 'file': program_name,
            #                 'marker': marker,
            #                 'invariant': simplified_inv,
            #                 'original': original_inv,
            #                 'reason': report.decision_reason,
            #                 'time': report.time_taken
            #             })
            #             print(f"  ❌ FAILED: {report.decision} - {report.decision_reason}")
            # else:
            #     stats['simplified']['skipped'] += 1
            
            # Compare times if simplified was verified
            if inv_verified['simplified'] and inv_times['simplified'] is not None:
                comparison = {
                    'file': program_name,
                    'marker': marker,
                    'original_time': inv_times['original'],
                    'pretty_time': inv_times['pretty'],
                    'simplified_time': inv_times['simplified'],
                    'original_verified': inv_verified['original'],
                    'pretty_verified': inv_verified['pretty'],
                }
                # Calculate speedup vs original (if original verified)
                if inv_verified['original'] and inv_times['original']:
                    comparison['speedup_vs_original'] = inv_times['original'] / inv_times['simplified']
                    comparison['time_saved_vs_original'] = inv_times['original'] - inv_times['simplified']
                # Calculate speedup vs pretty (if pretty verified)
                if inv_verified['pretty'] and inv_times['pretty']:
                    comparison['speedup_vs_pretty'] = inv_times['pretty'] / inv_times['simplified']
                    comparison['time_saved_vs_pretty'] = inv_times['pretty'] - inv_times['simplified']
                
                time_comparisons.append(comparison)
                
                # Print time comparison
                print(f"\n  ⏱️ TIME COMPARISON:")
                if inv_verified['original'] and inv_times['original']:
                    speedup = comparison.get('speedup_vs_original', 0)
                    saved = comparison.get('time_saved_vs_original', 0)
                    faster = "🚀 FASTER" if speedup > 1 else "🐢 SLOWER"
                    print(f"     vs Original: {inv_times['original']:.2f}s → {inv_times['simplified']:.2f}s ({speedup:.2f}x {faster}, {saved:+.2f}s)")
                if inv_verified['pretty'] and inv_times['pretty']:
                    speedup = comparison.get('speedup_vs_pretty', 0)
                    saved = comparison.get('time_saved_vs_pretty', 0)
                    faster = "🚀 FASTER" if speedup > 1 else "🐢 SLOWER"
                    print(f"     vs Pretty: {inv_times['pretty']:.2f}s → {inv_times['simplified']:.2f}s ({speedup:.2f}x {faster}, {saved:+.2f}s)")
    
    # Print summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    # Correctness stats
    for inv_type in ['original', 'pretty', 'simplified']:
        s = stats[inv_type]
        total = s['total']
        correct = s['correct']
        rate = (correct / total * 100) if total > 0 else 0
        skipped = s.get('skipped', 0)
        same_as_pretty = s.get('same_as_pretty', 0)
        
        print(f"\n{inv_type.upper()} INVARIANTS:")
        print(f"  Total: {total}, Correct: {correct}, Failed: {total - correct}")
        print(f"  Correctness Rate: {rate:.2f}%")
        if skipped > 0:
            print(f"  Skipped (not available): {skipped}")
        if same_as_pretty > 0:
            print(f"  Skipped (same as pretty): {same_as_pretty}")
        
        if s['failures']:
            print(f"  Failed files:")
            for f in s['failures'][:10]:  # Show first 10 failures
                print(f"    - {f['file']} ({f['marker']}): {f['reason']}")
            if len(s['failures']) > 10:
                print(f"    ... and {len(s['failures']) - 10} more")
    
    # Length reduction stats
    print("\n" + "-"*60)
    print("LENGTH REDUCTION STATISTICS")
    print("-"*60)
    
    for reduction_type, label in [
        ('original_to_pretty', 'Original → Pretty'),
        ('pretty_to_simplified', 'Pretty → Simplified'),
        ('original_to_simplified', 'Original → Simplified'),
    ]:
        data_points = length_stats[reduction_type]
        if data_points:
            avg_orig = sum(d[0] for d in data_points) / len(data_points)
            avg_new = sum(d[1] for d in data_points) / len(data_points)
            avg_reduction = sum(d[2] for d in data_points) / len(data_points)
            total_orig = sum(d[0] for d in data_points)
            total_new = sum(d[1] for d in data_points)
            total_reduction = ((total_orig - total_new) / total_orig * 100) if total_orig > 0 else 0
            
            print(f"\n{label}:")
            print(f"  Count: {len(data_points)} invariants")
            print(f"  Avg length: {avg_orig:.0f} → {avg_new:.0f} chars")
            print(f"  Avg reduction: {avg_reduction:.1f}%")
            print(f"  Total reduction: {total_reduction:.1f}% ({total_orig} → {total_new} chars)")
        else:
            print(f"\n{label}: No data")
    
    # Verification time stats
    print("\n" + "-"*60)
    print("VERIFICATION TIME STATISTICS")
    print("-"*60)
    
    for inv_type in ['original', 'pretty', 'simplified']:
        times = stats[inv_type]['times']
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)
            print(f"\n{inv_type.upper()}:")
            print(f"  Count: {len(times)} verifications")
            print(f"  Avg time: {avg_time:.2f}s")
            print(f"  Min/Max: {min_time:.2f}s / {max_time:.2f}s")
            print(f"  Total time: {total_time:.2f}s")
    
    # Speedup comparison (simplified vs original/pretty)
    if time_comparisons:
        print("\n" + "-"*60)
        print("SPEEDUP ANALYSIS (Simplified vs Original/Pretty)")
        print("-"*60)
        
        # vs Original
        speedups_orig = [c['speedup_vs_original'] for c in time_comparisons if 'speedup_vs_original' in c]
        if speedups_orig:
            avg_speedup = sum(speedups_orig) / len(speedups_orig)
            faster_count = sum(1 for s in speedups_orig if s > 1)
            slower_count = sum(1 for s in speedups_orig if s <= 1)
            time_saved = sum(c.get('time_saved_vs_original', 0) for c in time_comparisons if 'time_saved_vs_original' in c)
            print(f"\nSimplified vs Original:")
            print(f"  Comparisons: {len(speedups_orig)}")
            print(f"  Avg speedup: {avg_speedup:.2f}x")
            print(f"  Faster: {faster_count} ({faster_count/len(speedups_orig)*100:.1f}%)")
            print(f"  Slower: {slower_count} ({slower_count/len(speedups_orig)*100:.1f}%)")
            print(f"  Total time saved: {time_saved:.2f}s")
        
        # vs Pretty
        speedups_pretty = [c['speedup_vs_pretty'] for c in time_comparisons if 'speedup_vs_pretty' in c]
        if speedups_pretty:
            avg_speedup = sum(speedups_pretty) / len(speedups_pretty)
            faster_count = sum(1 for s in speedups_pretty if s > 1)
            slower_count = sum(1 for s in speedups_pretty if s <= 1)
            time_saved = sum(c.get('time_saved_vs_pretty', 0) for c in time_comparisons if 'time_saved_vs_pretty' in c)
            print(f"\nSimplified vs Pretty:")
            print(f"  Comparisons: {len(speedups_pretty)}")
            print(f"  Avg speedup: {avg_speedup:.2f}x")
            print(f"  Faster: {faster_count} ({faster_count/len(speedups_pretty)*100:.1f}%)")
            print(f"  Slower: {slower_count} ({slower_count/len(speedups_pretty)*100:.1f}%)")
            print(f"  Total time saved: {time_saved:.2f}s")
    
    # Add stats to return value
    stats['length_stats'] = length_stats
    stats['time_comparisons'] = time_comparisons
    
    # Save results to JSON file
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(GC.ROOT_DIR / "tests" / "results" / f"gt_invariants_test_{timestamp}.json")
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON serialization
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'limit': limit,
            'verifier_timeout': verifier.timeout_seconds,
        },
        'summary': {
            'original': {
                'total': stats['original']['total'],
                'correct': stats['original']['correct'],
                'failed': stats['original']['total'] - stats['original']['correct'],
                'rate': (stats['original']['correct'] / stats['original']['total'] * 100) if stats['original']['total'] > 0 else 0,
            },
            'pretty': {
                'total': stats['pretty']['total'],
                'correct': stats['pretty']['correct'],
                'failed': stats['pretty']['total'] - stats['pretty']['correct'],
                'rate': (stats['pretty']['correct'] / stats['pretty']['total'] * 100) if stats['pretty']['total'] > 0 else 0,
                'skipped': stats['pretty']['skipped'],
            },
            'simplified': {
                'total': stats['simplified']['total'],
                'correct': stats['simplified']['correct'],
                'failed': stats['simplified']['total'] - stats['simplified']['correct'],
                'rate': (stats['simplified']['correct'] / stats['simplified']['total'] * 100) if stats['simplified']['total'] > 0 else 0,
                'skipped': stats['simplified']['skipped'],
                'same_as_pretty': stats['simplified']['same_as_pretty'],
            },
        },
        'length_reduction': {},
        'failures': {
            'original': stats['original']['failures'],
            'pretty': stats['pretty']['failures'],
            'simplified': stats['simplified']['failures'],
        },
    }
    
    # Add length reduction summary
    for reduction_type in ['original_to_pretty', 'pretty_to_simplified', 'original_to_simplified']:
        data_points = length_stats[reduction_type]
        if data_points:
            results['length_reduction'][reduction_type] = {
                'count': len(data_points),
                'avg_original_len': sum(d[0] for d in data_points) / len(data_points),
                'avg_new_len': sum(d[1] for d in data_points) / len(data_points),
                'avg_reduction_pct': sum(d[2] for d in data_points) / len(data_points),
                'total_original_len': sum(d[0] for d in data_points),
                'total_new_len': sum(d[1] for d in data_points),
            }
    
    # Add timing statistics
    results['timing'] = {}
    for inv_type in ['original', 'pretty', 'simplified']:
        times = stats[inv_type]['times']
        if times:
            results['timing'][inv_type] = {
                'count': len(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times),
            }
    
    # Add speedup analysis
    if time_comparisons:
        speedups_orig = [c['speedup_vs_original'] for c in time_comparisons if 'speedup_vs_original' in c]
        speedups_pretty = [c['speedup_vs_pretty'] for c in time_comparisons if 'speedup_vs_pretty' in c]
        
        results['speedup_analysis'] = {
            'vs_original': {
                'count': len(speedups_orig),
                'avg_speedup': sum(speedups_orig) / len(speedups_orig) if speedups_orig else 0,
                'faster_count': sum(1 for s in speedups_orig if s > 1),
                'slower_count': sum(1 for s in speedups_orig if s <= 1),
                'total_time_saved': sum(c.get('time_saved_vs_original', 0) for c in time_comparisons if 'time_saved_vs_original' in c),
            } if speedups_orig else None,
            'vs_pretty': {
                'count': len(speedups_pretty),
                'avg_speedup': sum(speedups_pretty) / len(speedups_pretty) if speedups_pretty else 0,
                'faster_count': sum(1 for s in speedups_pretty if s > 1),
                'slower_count': sum(1 for s in speedups_pretty if s <= 1),
                'total_time_saved': sum(c.get('time_saved_vs_pretty', 0) for c in time_comparisons if 'time_saved_vs_pretty' in c),
            } if speedups_pretty else None,
        }
        results['time_comparisons_detail'] = time_comparisons
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {save_path}")
    
    return stats

if __name__ == "__main__":
    json_path = Path(GC.ROOT_DIR / "data" / "train" / "invbench-train-uautomizer25-k1-full-raw" / "invbench-train-uautomizer25-k1-full-cleaned.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {json_path}")
    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS["25"],
        property_file_path=GC.PROPERTIES_DIR / "unreach-call.prp",
        arch="32bit",
        timeout_seconds=600.0,
        version="25",
        memory_limit_mb=GC.MEMORY_LIMIT_MB,
    )
    test_gt_invariants(data, verifier, limit=-1)