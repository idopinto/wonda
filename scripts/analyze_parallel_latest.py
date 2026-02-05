#!/usr/bin/env python3
"""
Generate statistics and plots from the parallel-latest processed data.

This script loads the existing processed JSON file and generates:
- Grade distribution plot and CSV
- Speedup distribution plot and CSV
- Virtual best performance statistics and plot

Usage:
    uv run scripts/analyze_parallel_latest.py
"""

import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_data(path: Path) -> list[dict]:
    """Load JSON data."""
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {path}")
    return data


def save_grade_distribution(data: list[dict], output_dir: Path, label: str) -> None:
    """Save grade distribution statistics and plot."""
    grades = []
    for entry in data:
        gt_inv = entry.get("gt_invariant", {})
        grade = gt_inv.get("quality_grade")
        if grade is not None:
            grades.append(grade)
    
    if not grades:
        print("No grade data available")
        return
    
    # Count grades
    grade_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for g in grades:
        if g in grade_counts:
            grade_counts[g] += 1
    
    total = len(grades)
    
    print(f"\nGrade distribution ({label}):")
    for grade, count in sorted(grade_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  Grade {grade}: {count} ({pct:.1f}%)")
    
    # Save CSV
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = stats_dir / f"grade_distribution_{label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grade", "count", "percentage"])
        for grade, count in sorted(grade_counts.items()):
            pct = (count / total * 100) if total > 0 else 0
            writer.writerow([grade, count, f"{pct:.2f}"])
        writer.writerow(["total", total, "100.00"])
    print(f"Grade distribution data saved to: {csv_path}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grades_list = list(grade_counts.keys())
    counts = list(grade_counts.values())
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    
    bars = ax.bar(grades_list, counts, color=colors, edgecolor='black', width=0.6)
    
    # Add labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (count / total * 100) if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Quality Grade', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Quality Grade Distribution ({label})\n'
                 f'Total: {total} | Grade 2+: {grade_counts[2] + grade_counts[3]} '
                 f'({(grade_counts[2] + grade_counts[3]) / total * 100:.1f}%)',
                 fontsize=14)
    ax.set_xticks(grades_list)
    ax.set_xticklabels(['0\n(Invalid/\nIncorrect)', '1\n(Correct,\nTarget Fails)', 
                       '2\n(Usable,\nNo Speedup)', '3\n(Usable,\nSpeedup)'])
    
    plt.tight_layout()
    
    plot_path = stats_dir / f"grade_distribution_{label}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grade distribution plot saved to: {plot_path}")


def save_speedup_distribution(data: list[dict], output_dir: Path, label: str) -> None:
    """Save speedup distribution statistics and plot."""
    speedups = []
    for entry in data:
        gt_inv = entry.get("gt_invariant", {})
        speedup = gt_inv.get("speedup")
        grade = gt_inv.get("quality_grade", 0)
        # Only include grade >= 2 (usable invariants)
        if speedup is not None and speedup > 0 and grade >= 2:
            speedups.append(speedup)
    
    if not speedups:
        print("No speedup data available")
        return
    
    speedups = np.array(speedups)
    
    stats = {
        "count": len(speedups),
        "mean": float(np.mean(speedups)),
        "median": float(np.median(speedups)),
        "std": float(np.std(speedups)),
        "min": float(np.min(speedups)),
        "max": float(np.max(speedups)),
        "p25": float(np.percentile(speedups, 25)),
        "p75": float(np.percentile(speedups, 75)),
        "p90": float(np.percentile(speedups, 90)),
        "p95": float(np.percentile(speedups, 95)),
        "speedup_gt_1": int(np.sum(speedups > 1)),
        "speedup_gt_1_pct": float(np.sum(speedups > 1) / len(speedups) * 100),
    }
    
    print(f"\nSpeedup statistics ({label}, grade >= 2 only):")
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.2f}x")
    print(f"  Median: {stats['median']:.2f}x")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']:.2f}x")
    print(f"  Max: {stats['max']:.2f}x")
    print(f"  P25: {stats['p25']:.2f}x")
    print(f"  P75: {stats['p75']:.2f}x")
    print(f"  Speedup > 1: {stats['speedup_gt_1']} ({stats['speedup_gt_1_pct']:.1f}%)")
    
    # Save stats CSV
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_csv_path = stats_dir / f"speedup_stats_{label}.csv"
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])
    print(f"Speedup statistics saved to: {stats_csv_path}")
    
    # Save raw data
    raw_csv_path = stats_dir / f"speedups_{label}.csv"
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speedup"])
        for s in speedups:
            writer.writerow([f"{s:.4f}"])
    print(f"Raw speedup data saved to: {raw_csv_path}")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    bins = np.linspace(0, min(stats['max'], 10), 50)
    ax1.hist(speedups, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=1, color='black', linestyle='--', linewidth=2, label='No speedup (1x)')
    ax1.axvline(x=stats['mean'], color='blue', linestyle='-', linewidth=2, label=f'Mean: {stats["mean"]:.2f}x')
    ax1.axvline(x=stats['median'], color='orange', linestyle='-', linewidth=2, label=f'Median: {stats["median"]:.2f}x')
    
    ax1.set_xlabel('Speedup (x)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Speedup Distribution ({label}, grade >= 2)\n'
                  f'n={stats["count"]} | Speedup > 1x: {stats["speedup_gt_1"]} ({stats["speedup_gt_1_pct"]:.1f}%)',
                  fontsize=12)
    ax1.legend(loc='upper right')
    
    # Box plot
    bp = ax2.boxplot([speedups], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    
    textstr = f'Mean: {stats["mean"]:.2f}x\nMedian: {stats["median"]:.2f}x\nMin: {stats["min"]:.2f}x\nMax: {stats["max"]:.2f}x\nP25: {stats["p25"]:.2f}x\nP75: {stats["p75"]:.2f}x'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No speedup (1x)')
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title(f'Speedup Box Plot ({label})', fontsize=14)
    ax2.set_xticklabels([label])
    
    plt.tight_layout()
    
    plot_path = stats_dir / f"speedup_distribution_{label}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Speedup distribution plot saved to: {plot_path}")


def save_virtual_best_statistics(data: list[dict], output_dir: Path, label: str) -> None:
    """Save virtual best performance statistics and plot.
    
    Virtual best = min(baseline_time, verification_time) if grade >= 2, else baseline_time
    """
    baseline_times = []
    verification_times = []
    virtual_best_times = []
    grades = []
    
    for entry in data:
        gt_inv = entry.get("gt_invariant", {})
        timings = entry.get("timings", {})
        
        baseline_time = timings.get("median")
        speedup = gt_inv.get("speedup")
        grade = gt_inv.get("quality_grade", 0)
        
        if baseline_time is None or baseline_time <= 0:
            continue
        
        # Calculate verification time from speedup
        if speedup is not None and speedup > 0:
            verification_time = baseline_time / speedup
        else:
            verification_time = None
        
        baseline_times.append(baseline_time)
        grades.append(grade)
        
        if verification_time is not None:
            verification_times.append(verification_time)
        
        # Virtual best: use invariant if grade >= 2 and faster, else baseline
        if grade >= 2 and verification_time is not None:
            virtual_best_times.append(min(baseline_time, verification_time))
        else:
            virtual_best_times.append(baseline_time)
    
    if not baseline_times or not virtual_best_times:
        print("No timing data available for virtual best statistics")
        return
    
    baseline_times = np.array(baseline_times)
    virtual_best_times = np.array(virtual_best_times)
    verification_times = np.array(verification_times) if verification_times else None
    
    # Compute aggregate statistics
    total_baseline = float(np.sum(baseline_times))
    total_virtual_best = float(np.sum(virtual_best_times))
    total_verification = float(np.sum(verification_times)) if verification_times is not None else None
    
    overall_speedup = total_baseline / total_virtual_best if total_virtual_best > 0 else 0
    time_saved = total_baseline - total_virtual_best
    time_saved_pct = (time_saved / total_baseline * 100) if total_baseline > 0 else 0
    
    # Count how many times virtual best chose invariant vs baseline
    chose_invariant = sum(1 for b, v in zip(baseline_times, virtual_best_times) if v < b)
    chose_baseline = len(virtual_best_times) - chose_invariant
    
    stats = {
        "count": len(baseline_times),
        "total_baseline_time": total_baseline,
        "total_virtual_best_time": total_virtual_best,
        "total_verification_time": total_verification,
        "overall_speedup": overall_speedup,
        "time_saved_seconds": time_saved,
        "time_saved_percent": time_saved_pct,
        "chose_invariant_count": chose_invariant,
        "chose_baseline_count": chose_baseline,
        "chose_invariant_pct": (chose_invariant / len(virtual_best_times) * 100) if len(virtual_best_times) > 0 else 0,
        "mean_baseline_time": float(np.mean(baseline_times)),
        "mean_virtual_best_time": float(np.mean(virtual_best_times)),
        "median_baseline_time": float(np.median(baseline_times)),
        "median_virtual_best_time": float(np.median(virtual_best_times)),
    }
    
    print(f"\nVirtual Best Performance Statistics ({label}):")
    print(f"  Total instances: {stats['count']}")
    print(f"  Total baseline time: {stats['total_baseline_time']:.2f}s ({stats['total_baseline_time']/3600:.2f}h)")
    print(f"  Total virtual best time: {stats['total_virtual_best_time']:.2f}s ({stats['total_virtual_best_time']/3600:.2f}h)")
    print(f"  Overall speedup: {stats['overall_speedup']:.2f}x")
    print(f"  Time saved: {stats['time_saved_seconds']:.2f}s ({stats['time_saved_percent']:.1f}%)")
    print(f"  Chose invariant: {stats['chose_invariant_count']} ({stats['chose_invariant_pct']:.1f}%)")
    print(f"  Chose baseline: {stats['chose_baseline_count']}")
    
    # Save statistics as CSV
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_csv_path = stats_dir / f"virtual_best_stats_{label}.csv"
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])
    print(f"Virtual best statistics saved to: {stats_csv_path}")
    
    # Save raw data
    raw_csv_path = stats_dir / f"virtual_best_times_{label}.csv"
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline_time", "virtual_best_time", "grade", "chose"])
        for b, v, g in zip(baseline_times, virtual_best_times, grades):
            chose = "invariant" if v < b else "baseline"
            writer.writerow([f"{b:.4f}", f"{v:.4f}", g, chose])
    print(f"Virtual best raw data saved to: {raw_csv_path}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart comparing totals
    ax1 = axes[0]
    categories = ['Baseline', 'Virtual Best']
    times = [total_baseline / 3600, total_virtual_best / 3600]  # Convert to hours
    colors = ['#e74c3c', '#2ecc71']
    bars = ax1.bar(categories, times, color=colors, edgecolor='black')
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}h',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Total Time (hours)', fontsize=12)
    ax1.set_title(f'Total Time Comparison ({label})\n'
                  f'Speedup: {overall_speedup:.2f}x | Time Saved: {time_saved/3600:.1f}h ({time_saved_pct:.1f}%)',
                  fontsize=12)
    
    # Plot 2: Pie chart showing choice distribution
    ax2 = axes[1]
    sizes = [chose_invariant, chose_baseline]
    labels = [f'Invariant\n({chose_invariant}, {stats["chose_invariant_pct"]:.1f}%)',
              f'Baseline\n({chose_baseline}, {100-stats["chose_invariant_pct"]:.1f}%)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'Virtual Best Choice Distribution\n(n={stats["count"]})', fontsize=12)
    
    plt.tight_layout()
    
    plot_path = stats_dir / f"virtual_best_comparison_{label}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Virtual best comparison plot saved to: {plot_path}")


def save_grade3_statistics(data: list[dict], output_dir: Path, label: str) -> None:
    """Save detailed statistics for grade 3 invariants only."""
    speedups = []
    baseline_times = []
    verification_times = []
    
    for entry in data:
        gt_inv = entry.get("gt_invariant", {})
        timings = entry.get("timings", {})
        grade = gt_inv.get("quality_grade", 0)
        speedup = gt_inv.get("speedup")
        baseline = timings.get("median")
        
        if grade == 3 and speedup and speedup > 0 and baseline and baseline > 0:
            speedups.append(speedup)
            baseline_times.append(baseline)
            verification_times.append(baseline / speedup)
    
    if not speedups:
        print("No grade 3 data available")
        return
    
    speedups = np.array(speedups)
    baseline_times = np.array(baseline_times)
    verification_times = np.array(verification_times)
    
    total_baseline = float(np.sum(baseline_times))
    total_verification = float(np.sum(verification_times))
    time_saved = total_baseline - total_verification
    
    stats = {
        "count": len(speedups),
        "mean_speedup": float(np.mean(speedups)),
        "median_speedup": float(np.median(speedups)),
        "std_speedup": float(np.std(speedups)),
        "min_speedup": float(np.min(speedups)),
        "max_speedup": float(np.max(speedups)),
        "p25_speedup": float(np.percentile(speedups, 25)),
        "p75_speedup": float(np.percentile(speedups, 75)),
        "p90_speedup": float(np.percentile(speedups, 90)),
        "p95_speedup": float(np.percentile(speedups, 95)),
        "count_gt_2x": int(np.sum(speedups > 2)),
        "pct_gt_2x": float(np.sum(speedups > 2) / len(speedups) * 100),
        "count_gt_5x": int(np.sum(speedups > 5)),
        "pct_gt_5x": float(np.sum(speedups > 5) / len(speedups) * 100),
        "count_gt_10x": int(np.sum(speedups > 10)),
        "pct_gt_10x": float(np.sum(speedups > 10) / len(speedups) * 100),
        "total_baseline_seconds": total_baseline,
        "total_baseline_hours": total_baseline / 3600,
        "total_verification_seconds": total_verification,
        "total_verification_hours": total_verification / 3600,
        "time_saved_seconds": time_saved,
        "time_saved_hours": time_saved / 3600,
        "time_saved_percent": (time_saved / total_baseline * 100) if total_baseline > 0 else 0,
        "aggregate_speedup": total_baseline / total_verification if total_verification > 0 else 0,
    }
    
    print(f"\n=== Grade 3 Only Statistics ({label}) ===")
    print(f"Count: {stats['count']}")
    print(f"\nSpeedup Statistics:")
    print(f"  Mean: {stats['mean_speedup']:.2f}x")
    print(f"  Median: {stats['median_speedup']:.2f}x")
    print(f"  Min: {stats['min_speedup']:.2f}x")
    print(f"  Max: {stats['max_speedup']:.2f}x")
    print(f"  P25: {stats['p25_speedup']:.2f}x")
    print(f"  P75: {stats['p75_speedup']:.2f}x")
    print(f"  P90: {stats['p90_speedup']:.2f}x")
    print(f"  >2x: {stats['count_gt_2x']} ({stats['pct_gt_2x']:.1f}%)")
    print(f"  >5x: {stats['count_gt_5x']} ({stats['pct_gt_5x']:.1f}%)")
    print(f"  >10x: {stats['count_gt_10x']} ({stats['pct_gt_10x']:.1f}%)")
    print(f"\nTime Statistics:")
    print(f"  Total baseline: {stats['total_baseline_hours']:.2f}h")
    print(f"  Total verification: {stats['total_verification_hours']:.2f}h")
    print(f"  Time saved: {stats['time_saved_hours']:.2f}h ({stats['time_saved_percent']:.1f}%)")
    print(f"  Aggregate speedup: {stats['aggregate_speedup']:.2f}x")
    
    # Save stats CSV
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_csv_path = stats_dir / f"grade3_stats_{label}.csv"
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])
    print(f"Grade 3 statistics saved to: {stats_csv_path}")
    
    # Save raw speedups
    raw_csv_path = stats_dir / f"grade3_speedups_{label}.csv"
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speedup", "baseline_time", "verification_time"])
        for s, b, v in zip(speedups, baseline_times, verification_times):
            writer.writerow([f"{s:.4f}", f"{b:.4f}", f"{v:.4f}"])
    print(f"Grade 3 raw data saved to: {raw_csv_path}")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    bins = np.linspace(1, min(stats['max_speedup'], 10), 40)
    ax1.hist(speedups, bins=bins, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax1.axvline(x=stats['mean_speedup'], color='blue', linestyle='-', linewidth=2, 
                label=f'Mean: {stats["mean_speedup"]:.2f}x')
    ax1.axvline(x=stats['median_speedup'], color='orange', linestyle='-', linewidth=2, 
                label=f'Median: {stats["median_speedup"]:.2f}x')
    ax1.axvline(x=2, color='red', linestyle='--', linewidth=1.5, label='2x threshold')
    
    ax1.set_xlabel('Speedup (x)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Grade 3 Speedup Distribution ({label})\n'
                  f'n={stats["count"]} | >2x: {stats["count_gt_2x"]} ({stats["pct_gt_2x"]:.1f}%)',
                  fontsize=12)
    ax1.legend(loc='upper right')
    
    # Bar chart comparing times
    categories = ['Baseline', 'With Invariant']
    times = [stats['total_baseline_hours'], stats['total_verification_hours']]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax2.bar(categories, times, color=colors, edgecolor='black')
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}h',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Total Time (hours)', fontsize=12)
    ax2.set_title(f'Grade 3: Time Comparison\n'
                  f'Speedup: {stats["aggregate_speedup"]:.2f}x | Saved: {stats["time_saved_hours"]:.1f}h ({stats["time_saved_percent"]:.1f}%)',
                  fontsize=12)
    
    plt.tight_layout()
    
    plot_path = stats_dir / f"grade3_distribution_{label}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grade 3 distribution plot saved to: {plot_path}")


def save_invariant_length_statistics(data: list[dict], output_dir: Path, label: str) -> None:
    """Save invariant character length statistics and plot by grade."""
    # Collect lengths by grade and invariant type
    lengths_by_grade = {0: [], 1: [], 2: [], 3: []}
    original_lengths = []
    target_lengths = []
    
    for entry in data:
        gt_inv = entry.get("gt_invariant", {})
        grade = gt_inv.get("quality_grade")
        
        # Get different invariant versions
        original = gt_inv.get("original_invariant", "")
        target = gt_inv.get("target_invariant", "")  # simplified version
        
        if original:
            original_lengths.append(len(original))
        if target:
            target_lengths.append(len(target))
            if grade is not None and grade in lengths_by_grade:
                lengths_by_grade[grade].append(len(target))
    
    if not target_lengths:
        print("No invariant length data available")
        return
    
    original_lengths = np.array(original_lengths) if original_lengths else None
    target_lengths = np.array(target_lengths)
    
    # Compute statistics
    stats = {
        "count": len(target_lengths),
        "mean_target_length": float(np.mean(target_lengths)),
        "median_target_length": float(np.median(target_lengths)),
        "std_target_length": float(np.std(target_lengths)),
        "min_target_length": float(np.min(target_lengths)),
        "max_target_length": float(np.max(target_lengths)),
        "p25_target_length": float(np.percentile(target_lengths, 25)),
        "p75_target_length": float(np.percentile(target_lengths, 75)),
        "p90_target_length": float(np.percentile(target_lengths, 90)),
    }
    
    if original_lengths is not None:
        stats["mean_original_length"] = float(np.mean(original_lengths))
        stats["median_original_length"] = float(np.median(original_lengths))
        stats["compression_ratio"] = float(np.mean(original_lengths) / np.mean(target_lengths)) if np.mean(target_lengths) > 0 else 0
    
    # Stats by grade
    for grade in [0, 1, 2, 3]:
        if lengths_by_grade[grade]:
            stats[f"mean_length_grade_{grade}"] = float(np.mean(lengths_by_grade[grade]))
            stats[f"median_length_grade_{grade}"] = float(np.median(lengths_by_grade[grade]))
            stats[f"count_grade_{grade}"] = len(lengths_by_grade[grade])
    
    print(f"\n=== Invariant Length Statistics ({label}) ===")
    print(f"Count: {stats['count']}")
    if original_lengths is not None:
        print(f"\nOriginal invariant:")
        print(f"  Mean: {stats['mean_original_length']:.1f} chars")
        print(f"  Median: {stats['median_original_length']:.1f} chars")
    print(f"\nSimplified invariant (target):")
    print(f"  Mean: {stats['mean_target_length']:.1f} chars")
    print(f"  Median: {stats['median_target_length']:.1f} chars")
    print(f"  Min: {stats['min_target_length']:.0f} chars")
    print(f"  Max: {stats['max_target_length']:.0f} chars")
    if original_lengths is not None:
        print(f"\nCompression ratio: {stats['compression_ratio']:.2f}x")
    print(f"\nBy grade (simplified):")
    for grade in [0, 1, 2, 3]:
        if f"mean_length_grade_{grade}" in stats:
            print(f"  Grade {grade}: mean={stats[f'mean_length_grade_{grade}']:.1f}, "
                  f"median={stats[f'median_length_grade_{grade}']:.1f}, n={stats[f'count_grade_{grade}']}")
    
    # Save stats CSV
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_csv_path = stats_dir / f"invariant_length_stats_{label}.csv"
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])
    print(f"Invariant length statistics saved to: {stats_csv_path}")
    
    # Save raw data
    raw_csv_path = stats_dir / f"invariant_lengths_{label}.csv"
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_length", "target_length", "grade"])
        for entry in data:
            gt_inv = entry.get("gt_invariant", {})
            grade = gt_inv.get("quality_grade", "")
            original = gt_inv.get("original_invariant", "")
            target = gt_inv.get("target_invariant", "")
            if target:
                writer.writerow([len(original) if original else "", len(target), grade])
    print(f"Invariant length raw data saved to: {raw_csv_path}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram of target lengths
    ax1 = axes[0, 0]
    bins = np.linspace(0, min(stats['p90_target_length'] * 1.5, stats['max_target_length']), 50)
    ax1.hist(target_lengths, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=stats['mean_target_length'], color='blue', linestyle='-', linewidth=2, 
                label=f'Mean: {stats["mean_target_length"]:.0f}')
    ax1.axvline(x=stats['median_target_length'], color='orange', linestyle='-', linewidth=2, 
                label=f'Median: {stats["median_target_length"]:.0f}')
    ax1.set_xlabel('Character Length', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Simplified Invariant Length Distribution\n(n={stats["count"]})', fontsize=12)
    ax1.legend(loc='upper right')
    
    # Plot 2: Box plot by grade
    ax2 = axes[0, 1]
    grade_data = [lengths_by_grade[g] for g in [1, 2, 3] if lengths_by_grade[g]]
    grade_labels = [f'Grade {g}\n(n={len(lengths_by_grade[g])})' for g in [1, 2, 3] if lengths_by_grade[g]]
    if grade_data:
        bp = ax2.boxplot(grade_data, patch_artist=True, labels=grade_labels)
        colors = ['#f39c12', '#3498db', '#2ecc71']
        for patch, color in zip(bp['boxes'], colors[:len(grade_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax2.set_ylabel('Character Length', fontsize=12)
    ax2.set_title('Invariant Length by Quality Grade', fontsize=12)
    
    # Plot 3: Original vs Simplified comparison (if available)
    ax3 = axes[1, 0]
    if original_lengths is not None:
        ax3.scatter(original_lengths, target_lengths, alpha=0.3, s=10, c='steelblue')
        max_val = max(np.max(original_lengths), np.max(target_lengths))
        ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='y=x (no change)')
        ax3.set_xlabel('Original Length (chars)', fontsize=12)
        ax3.set_ylabel('Simplified Length (chars)', fontsize=12)
        ax3.set_title(f'Original vs Simplified Length\nCompression: {stats["compression_ratio"]:.2f}x', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.set_xlim(0, np.percentile(original_lengths, 99))
        ax3.set_ylim(0, np.percentile(target_lengths, 99))
    else:
        ax3.text(0.5, 0.5, 'No original invariant data', ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Mean length by grade bar chart
    ax4 = axes[1, 1]
    grades_with_data = [g for g in [1, 2, 3] if f"mean_length_grade_{g}" in stats]
    if grades_with_data:
        means = [stats[f"mean_length_grade_{g}"] for g in grades_with_data]
        medians = [stats[f"median_length_grade_{g}"] for g in grades_with_data]
        x = np.arange(len(grades_with_data))
        width = 0.35
        bars1 = ax4.bar(x - width/2, means, width, label='Mean', color='steelblue', edgecolor='black')
        bars2 = ax4.bar(x + width/2, medians, width, label='Median', color='coral', edgecolor='black')
        ax4.set_xlabel('Quality Grade', fontsize=12)
        ax4.set_ylabel('Character Length', fontsize=12)
        ax4.set_title('Mean/Median Length by Grade', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Grade {g}' for g in grades_with_data])
        ax4.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = stats_dir / f"invariant_length_distribution_{label}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Invariant length distribution plot saved to: {plot_path}")


def main():
    # Input file
    input_path = Path("data/train/invbench-train-uautomizer25-k1-full-raw/invbench-train-uautomizer25-k1-full-parallel-latest.json")
    
    # Output directory (same parent as input, in "v2" subfolder)
    output_dir = input_path.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label = "v2"
    
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    data = load_data(input_path)
    
    # Generate plots
    save_grade_distribution(data, output_dir, label)
    save_speedup_distribution(data, output_dir, label)
    save_virtual_best_statistics(data, output_dir, label)
    save_grade3_statistics(data, output_dir, label)
    save_invariant_length_statistics(data, output_dir, label)
    
    print(f"\nDone! All statistics and plots saved to {output_dir}/stats/")


if __name__ == "__main__":
    main()
