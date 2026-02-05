#!/usr/bin/env python3
"""
Plot the distribution of type casts in the training data.
"""

import json
import re
from collections import Counter
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_casts(inv_str: str) -> list[str]:
    """Extract all integral type casts from an invariant string."""
    # Pattern for integral type casts
    # Matches: (long long), (__int128), (int), (unsigned long long), etc.
    cast_pattern = r'\(((?:long\s+)?long|__int128|unsigned\s+(?:long\s+)?long?|int|char|short|_Bool|bool|signed)\s*\)'
    
    matches = re.findall(cast_pattern, inv_str, re.IGNORECASE)
    # Normalize whitespace and case
    normalized = [' '.join(m.split()).lower() for m in matches]
    return normalized

def main():
    data_file = Path(__file__).parent.parent / "data" / "train" / "invbench-train-uautomizer25-k1-full-raw" / "invbench-train-uautomizer25-k1-full-raw.json"
    
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return
    
    print(f"Loading data from: {data_file}")
    with open(data_file) as f:
        data = json.load(f)
    
    cast_types = Counter()
    total_invariants = 0
    invariants_with_casts = 0
    
    for entry in data:
        for inv_entry in entry.get('invariants', []):
            raw = inv_entry.get('invariant', '')
            if not raw:
                continue
            
            total_invariants += 1
            casts = extract_casts(raw)
            if casts:
                invariants_with_casts += 1
                cast_types.update(casts)
    
    print(f"\nAnalyzed {total_invariants} invariants")
    print(f"Invariants with casts: {invariants_with_casts} ({100*invariants_with_casts/total_invariants:.1f}%)")
    print(f"Total casts found: {sum(cast_types.values())}")
    
    # Sort by count
    sorted_pairs = sorted(cast_types.items(), key=lambda x: x[1], reverse=True)
    cast_names, cast_counts = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution of Type Casts', 'Proportion of Cast Types'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Plot 1: Bar chart
    fig.add_trace(
        go.Bar(
            x=[f'({name})' for name in cast_names],
            y=cast_counts,
            text=[f'{count:,}' for count in cast_counts],
            textposition='outside',
            marker_color='steelblue',
            marker_line_color='black',
            marker_line_width=1,
        ),
        row=1, col=1
    )
    
    fig.update_xaxes(title_text="Cast Type", row=1, col=1, tickangle=-45)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    # Plot 2: Pie chart (top 4 + other)
    top_n = 4
    if len(cast_names) > top_n:
        top_names = list(cast_names[:top_n])
        top_counts = list(cast_counts[:top_n])
        other_count = sum(cast_counts[top_n:])
        if other_count > 0:
            top_names.append('other')
            top_counts.append(other_count)
    else:
        top_names = list(cast_names)
        top_counts = list(cast_counts)
    
    fig.add_trace(
        go.Pie(
            labels=[f'({name})' for name in top_names],
            values=top_counts,
            textinfo='label+percent',
            textfont_size=12,
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Type Cast Distribution in Training Data",
        title_x=0.5,
        height=600,
        showlegend=False,
    )
    
    # Save plot
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "cast_type_distribution.html"
    fig.write_html(str(output_path))
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PNG
    try:
        output_path_png = output_dir / "cast_type_distribution.png"
        fig.write_image(str(output_path_png), width=1400, height=600, scale=2)
        print(f"Plot saved to: {output_path_png}")
    except Exception as e:
        print(f"Note: Could not save PNG (requires kaleido): {e}")
        print("HTML file saved successfully.")
    
    # Print summary
    print("\n" + "=" * 70)
    print("CAST TYPE SUMMARY")
    print("=" * 70)
    for name, count in sorted_pairs:
        pct = 100 * count / sum(cast_types.values())
        print(f"  ({name:20s}): {count:6,} ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
