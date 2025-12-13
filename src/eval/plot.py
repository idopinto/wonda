"""
Interactive evaluation plot generation for InvBench results.
Generates Plotly HTML plots with toggleable UNKNOWN points and generation time inclusion.
"""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def reindent_c_code(code: str, indent_size: int = 4) -> str:
    """
    Re-indent C code based on brace counting.
    
    This is useful for displaying code that was processed with clang-format
    using IndentWidth: 0, which removes all indentation.
    
    Args:
        code: C code string to re-indent.
        indent_size: Number of spaces per indentation level.
        
    Returns:
        Re-indented C code string.
    """
    if not code or not code.strip():
        return code
    
    lines = code.split('\n')
    result = []
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append('')
            continue
        
        # Count braces in this line
        open_count = stripped.count('{')
        close_count = stripped.count('}')
        
        # Determine indent for THIS line:
        # If line starts with }, decrease indent first
        line_indent = indent_level
        if stripped.startswith('}'):
            line_indent = max(0, indent_level - 1)
        
        # Apply indent and add to result
        indented_line = ' ' * (line_indent * indent_size) + stripped
        result.append(indented_line)
        
        # Calculate indent for NEXT line based on net brace change
        # But handle the case where line starts with } specially
        if stripped.startswith('}'):
            # We already decreased for this line, now handle any { on same line
            # Example: "} else {" - net change from indent_level should be 0
            # We decreased by 1 for the }, now increase by open_count
            indent_level = max(0, indent_level - 1 + open_count)
        else:
            # Normal case: add opens, subtract closes
            net_change = open_count - close_count
            indent_level = max(0, indent_level + net_change)
    
    return '\n'.join(result)


def convert_weave_summary_to_plot_metrics(summary: dict) -> dict:
    """
    Convert Weave evaluation summary to plot metrics format.
    
    Args:
        summary: Weave summary dict from InvariantScorer.summarize()
        
    Returns:
        dict: Metrics in plot-compatible format
    """
    return {
        "metrics_without_gen": {
            "% Correct Invariant": float(np.round(summary.get("invariant_correctness_rate", 0) * 100, 2)),
            "% Speedup": float(np.round(summary.get("speedup_rate_without_gen", 0) * 100, 2)),
            "Speedup>1": float(np.round(summary.get("speedup_gt1_without_gen", 1), 2)),
            "Speedup_all": float(np.round(summary.get("speedup_all_without_gen", 1), 2)),
        },
        "metrics_with_gen": {
            "% Correct Invariant": float(np.round(summary.get("invariant_correctness_rate", 0) * 100, 2)),
            "% Speedup": float(np.round(summary.get("speedup_rate_with_gen", 0) * 100, 2)),
            "Speedup>1": float(np.round(summary.get("speedup_gt1_with_gen", 1), 2)),
            "Speedup_all": float(np.round(summary.get("speedup_all_with_gen", 1), 2)),
        }
    }


def calculate_metrics_from_results(results: List[dict]) -> dict:
    """
    Calculate metrics directly from evaluation results list.
    
    This computes the same metrics as InvariantScorer.summarize() but from
    raw results, useful when Weave summary is not available.
    
    Args:
        results: List of evaluation result dicts
        
    Returns:
        dict: Metrics in plot-compatible format
    """
    n = len(results)
    if n == 0:
        return {
            "metrics_without_gen": {
                "% Correct Invariant": 0.0,
                "% Speedup": 0.0,
                "Speedup>1": 1.0,
                "Speedup_all": 1.0,
            },
            "metrics_with_gen": {
                "% Correct Invariant": 0.0,
                "% Speedup": 0.0,
                "Speedup>1": 1.0,
                "Speedup_all": 1.0,
            }
        }
    
    # Calculate invariant correctness rate
    invariant_correctness_rate = sum(
        1 for r in results if r.get("invariant_correctness_score", False)
    ) / n
    
    # Calculate speedup rates
    speedup_rate_without_gen = sum(
        1 for r in results if r.get("has_speedup_without_gen", False)
    ) / n
    speedup_rate_with_gen = sum(
        1 for r in results if r.get("has_speedup_with_gen", False)
    ) / n
    
    # Helper to check if result qualifies for speedup
    def qualifies(r: dict) -> bool:
        return (r.get("invariant_correctness_score", False) 
                and r.get("final_decision") != "UNKNOWN")
    
    # Calculate Speedup>1 (average of only qualifying with speedup > 1)
    speedups_gt1_without = [r["speedup_without_gen"] for r in results 
                           if qualifies(r) and r.get("speedup_without_gen", 0) > 1]
    speedups_gt1_with = [r["speedup_with_gen"] for r in results 
                        if qualifies(r) and r.get("speedup_with_gen", 0) > 1]
    
    speedup_gt1_without_gen = sum(speedups_gt1_without) / len(speedups_gt1_without) if speedups_gt1_without else 1.0
    speedup_gt1_with_gen = sum(speedups_gt1_with) / len(speedups_gt1_with) if speedups_gt1_with else 1.0
    
    # Calculate Speedup_all (non-qualifying counted as 1)
    def get_speedup_or_one(r: dict, key: str) -> float:
        if qualifies(r) and r.get(key, 0) > 1:
            return r[key]
        return 1.0
    
    speedup_all_without_gen = sum(get_speedup_or_one(r, "speedup_without_gen") for r in results) / n
    speedup_all_with_gen = sum(get_speedup_or_one(r, "speedup_with_gen") for r in results) / n
    
    return {
        "metrics_without_gen": {
            "% Correct Invariant": float(np.round(invariant_correctness_rate * 100, 2)),
            "% Speedup": float(np.round(speedup_rate_without_gen * 100, 2)),
            "Speedup>1": float(np.round(speedup_gt1_without_gen, 2)),
            "Speedup_all": float(np.round(speedup_all_without_gen, 2)),
        },
        "metrics_with_gen": {
            "% Correct Invariant": float(np.round(invariant_correctness_rate * 100, 2)),
            "% Speedup": float(np.round(speedup_rate_with_gen * 100, 2)),
            "Speedup>1": float(np.round(speedup_gt1_with_gen, 2)),
            "Speedup_all": float(np.round(speedup_all_with_gen, 2)),
        }
    }


def plot_verification_vs_baseline(
    results: List[dict],
    model_name: str = "gpt-oss-20b",
    baseline_name: str = "UAutomizer25",
    split_name: str = "easy",
    fig_size: tuple = (800, 800),
    plot_path: Path = Path("plot.html"),
    metrics: Optional[dict] = None
) -> Path:
    """
    Plot LLM verification vs. baseline time with interactive controls.
    
    Features:
    - Color-coded by decision (TRUE=green, FALSE=red, UNKNOWN=blue)
    - Interactive checkboxes to toggle UNKNOWN visibility
    - Interactive checkbox to include/exclude generation time
    - Metrics annotation that updates with generation time toggle
    
    Args:
        results: List of evaluation result dicts
        model_name: Name of the model being evaluated
        baseline_name: Name of the baseline verifier
        split_name: Dataset split name (easy/hard)
        fig_size: Tuple of (width, height) for the figure
        plot_path: Path to save the HTML file
        metrics: Optional pre-calculated metrics dict
        
    Returns:
        Path to the generated HTML file
    """
    df = pd.DataFrame(results)
    n_total = len(df)
    
    if n_total == 0:
        raise ValueError("No results to plot")
    
    # Prepare columns with safe defaults
    for col in ["verification_time", "total_time", "baseline_timing", "speedup_without_gen", "speedup_with_gen"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(2)
    
    # Ensure required columns exist
    if "final_decision" not in df.columns:
        df["final_decision"] = "UNKNOWN"
    if "predicate_content" not in df.columns:
        df["predicate_content"] = ""
    if "invariant_correctness_score" not in df.columns:
        df["invariant_correctness_score"] = False
    if "decision_rule" not in df.columns:
        df["decision_rule"] = ""
    if "syntactic_validation_score" not in df.columns:
        df["syntactic_validation_score"] = False
    if "has_speedup_without_gen" not in df.columns:
        df["has_speedup_without_gen"] = False
    if "has_speedup_with_gen" not in df.columns:
        df["has_speedup_with_gen"] = False
    
    # Extract usefulness decision from invariant_usefulness_report
    def extract_usefulness_decision(row):
        report = row.get("invariant_usefulness_report")
        if isinstance(report, dict):
            return report.get("decision", "N/A")
        return "N/A"
    
    df["usefulness_decision"] = df.apply(extract_usefulness_decision, axis=1)
    
    # Ensure code_for_usefulness is available for click-to-view
    if "code_for_usefulness" not in df.columns:
        df["code_for_usefulness"] = ""
    
    # Calculate speedup if not present
    df["speedup_without_gen"] = np.where(
        df["verification_time"] > 0,
        (df["baseline_timing"] / df["verification_time"]).round(2),
        0
    )
    df["speedup_with_gen"] = np.where(
        df["total_time"] > 0,
        (df["baseline_timing"] / df["total_time"]).round(2),
        0
    )
    
    # Calculate metrics if not provided
    if metrics is None:
        metrics = calculate_metrics_from_results(results)
    
    # Count decisions
    decision_counts = df["final_decision"].value_counts().to_dict()
    count_true = decision_counts.get("TRUE", 0)
    count_false = decision_counts.get("FALSE", 0)
    count_unknown = decision_counts.get("UNKNOWN", 0)
    counts_str = f"Model results: TRUE ({count_true}) | FALSE ({count_false}) | UNKNOWN ({count_unknown})"

    # Color and symbol maps
    color_map = {"TRUE": "green", "FALSE": "red", "UNKNOWN": "blue"}
    symbol_map = {"TRUE": "circle", "FALSE": "diamond", "UNKNOWN": "triangle-up"}

    # Prepare hover data columns
    df["task_index"] = range(len(df))
    df["hover_validation"] = df["syntactic_validation_score"].map({True: "VALID", False: "INVALID"})
    df["hover_correct"] = df["invariant_correctness_score"].map({True: "TRUE", False: "FALSE"})
    df["hover_useful_no_gen"] = df["has_speedup_without_gen"].map({True: "YES", False: "NO"})
    df["hover_useful_with_gen"] = df["has_speedup_with_gen"].map({True: "YES", False: "NO"})
    df["hover_usefulness_decision"] = df["usefulness_decision"]  # Actual verification result
    df["hover_rule"] = df["decision_rule"]

    # Create figure
    fig = go.Figure()

    no_gen_indices_always = []
    no_gen_indices_unknown = []
    gen_indices_always = []
    gen_indices_unknown = []

    # Add traces for WITHOUT gen time (visible by default)
    for decision in ["TRUE", "FALSE", "UNKNOWN"]:
        df_subset = df[df["final_decision"] == decision]
        if len(df_subset) > 0:
            hover_without = (
                "<b>%{customdata[0]}</b><br>" +
                "Final Decision=%{customdata[1]}<br>" +
                "LLM-assisted Verification Time (s)=%{x}<br>" +
                "Baseline Timing (s)=%{y}<br>" +
                "Task Index=%{customdata[2]}<br>" +
                "Speedup=%{customdata[3]}<br>" +
                "Validation=%{customdata[4]}<br>" +
                "Correctness=%{customdata[5]}<br>" +
                "Usefulness Decision=%{customdata[6]}<br>" +
                "Useful (speedup>1)=%{customdata[7]}<br>" +
                "Rule=%{customdata[8]}<br>"
            )
            fig.add_trace(go.Scatter(
                x=df_subset["verification_time"],
                y=df_subset["baseline_timing"],
                mode='markers',
                marker=dict(size=10, color=color_map[decision], symbol=symbol_map[decision]),
                name=decision,
                customdata=df_subset[["predicate_content", "final_decision", "task_index", "speedup_without_gen", "hover_validation", "hover_correct", "hover_usefulness_decision", "hover_useful_no_gen", "hover_rule"]].values,
                hovertemplate=hover_without,
                visible=True,
                legendgroup=decision,
                showlegend=True
            ))
            current_idx = len(fig.data) - 1
            if decision == "UNKNOWN":
                no_gen_indices_unknown.append(current_idx)
            else:
                no_gen_indices_always.append(current_idx)

    # Add traces for WITH gen time (hidden by default)
    for decision in ["TRUE", "FALSE", "UNKNOWN"]:
        df_subset = df[df["final_decision"] == decision]
        if len(df_subset) > 0:
            hover_with = (
                "<b>%{customdata[0]}</b><br>" +
                "Final Decision=%{customdata[1]}<br>" +
                "LLM-assisted Verification+Gen Time (s)=%{x}<br>" +
                "Baseline Timing (s)=%{y}<br>" +
                "Task Index=%{customdata[2]}<br>" +
                "Speedup=%{customdata[3]}<br>" +
                "Validation=%{customdata[4]}<br>" +
                "Correctness=%{customdata[5]}<br>" +
                "Usefulness Decision=%{customdata[6]}<br>" +
                "Useful (speedup>1)=%{customdata[7]}<br>" +
                "Rule=%{customdata[8]}<br>"
            )
            fig.add_trace(go.Scatter(
                x=df_subset["total_time"],
                y=df_subset["baseline_timing"],
                mode='markers',
                marker=dict(size=10, color=color_map[decision], symbol=symbol_map[decision]),
                name=decision,
                customdata=df_subset[["predicate_content", "final_decision", "task_index", "speedup_with_gen", "hover_validation", "hover_correct", "hover_usefulness_decision", "hover_useful_with_gen", "hover_rule"]].values,
                hovertemplate=hover_with,
                visible=False,
                legendgroup=decision,
                showlegend=False
            ))
            current_idx = len(fig.data) - 1
            if decision == "UNKNOWN":
                gen_indices_unknown.append(current_idx)
            else:
                gen_indices_always.append(current_idx)

    # Calculate axis limits
    min_val = min(
        df["verification_time"].min() if not df["verification_time"].isna().all() else 0.1,
        df["total_time"].min() if not df["total_time"].isna().all() else 0.1,
        df["baseline_timing"].min() if not df["baseline_timing"].isna().all() else 0.1
    )
    min_val = max(0.1, min_val)  # Ensure positive for potential log scale
    
    max_val = max(
        df["verification_time"].max() if not df["verification_time"].isna().all() else 600,
        df["total_time"].max() if not df["total_time"].isna().all() else 600,
        df["baseline_timing"].max() if not df["baseline_timing"].isna().all() else 600,
        600
    )
    lims = [min_val, max_val]

    # Draw diagonal line (orange)
    fig.add_shape(type="line",
        x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1],
        line=dict(color="orange")
    )
    
    # Add dashed lines for timeout at 600s
    fig.add_shape(type="line",
        x0=600, y0=lims[0], x1=600, y1=lims[1],
        line=dict(color="gray", dash="dash")
    )
    fig.add_shape(type="line",
        x0=lims[0], y0=600, x1=lims[1], y1=600,
        line=dict(color="gray", dash="dash")
    )

    # Update layout
    fig.update_layout(
        width=fig_size[0], height=fig_size[1],
        showlegend=True,
        xaxis=dict(
            range=[lims[0]*0.95, lims[1]*1.05],
            title="LLM-assisted Verification Time (s)",
            autorange=False,
            fixedrange=False
        ),
        yaxis=dict(
            range=[lims[0]*0.95, lims[1]*1.05],
            title="Baseline Time (s)",
            autorange=False,
            fixedrange=False
        ),
        margin=dict(t=140, b=120),
        title={
            "text": (
                f"<b>LLM-assisted Verification vs Baseline</b><br>"
                f"<span style='font-size:1.1em; color: #666;'>"
                f"Model: {model_name} | Baseline: {baseline_name} | Split: {split_name} [{n_total}]"
                f"</span><br>"
                f"<span style='font-size:1.1em; color: #666;'>"
                f"Timeout: set to baseline time."
                f"</span><br>"
                f"<span style='font-size:1.0em; color: #333;'>"
                f"{counts_str}"
                f"</span>"
            ),
            "x": 0.5
        }
    )

    # Prepare metrics strings
    m_without = metrics["metrics_without_gen"]
    m_with = metrics["metrics_with_gen"]
    metric_str_without = " | ".join([f"<b>{k}: {v}</b>" for k, v in m_without.items()])
    metric_str_with = " | ".join([f"<b>{k}: {v}</b>" for k, v in m_with.items()])
    
    # Add metrics annotation
    fig.add_annotation(
        text=f"{metric_str_without}",
        xref="paper", yref="paper",
        x=0.5, y=-0.18,
        showarrow=False,
        align="center",
        font=dict(size=15, color="#111", family="Arial"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.2)",
        borderpad=4,
        name="metricsAnnotation"
    )

    # Generate HTML with custom checkboxes
    html_string = fig.to_html(include_plotlyjs='cdn')
    
    # Serialize indices for JS
    ng_always_js = json.dumps(no_gen_indices_always)
    ng_unk_js = json.dumps(no_gen_indices_unknown)
    g_always_js = json.dumps(gen_indices_always)
    g_unk_js = json.dumps(gen_indices_unknown)
    
    # Build code data for click-to-view (indexed by task_index)
    # Re-indent code for better readability (original may have IndentWidth: 0 from clang-format)
    code_data = []
    for idx, row in df.iterrows():
        raw_code = str(row.get("code_for_usefulness", ""))
        formatted_code = reindent_c_code(raw_code) if raw_code else ""
        code_data.append({
            "task_index": int(row["task_index"]),
            "predicate": str(row.get("predicate_content", "")),
            "code_for_usefulness": formatted_code,
            "final_decision": str(row.get("final_decision", "")),
            "usefulness_decision": str(row.get("usefulness_decision", "")),
        })
    code_data_js = json.dumps(code_data)

    # Inject custom checkbox controls
    checkbox_html = f"""
    <!-- Prism.js for C syntax highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-c.min.js"></script>
    
    <style>
    body {{
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        padding: 20px;
        box-sizing: border-box;
    }}
    .plotly-graph-div {{
        margin: 0 auto;
    }}
    /* Modal styles */
    .code-modal {{
        display: none;
        position: fixed;
        z-index: 2000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
    }}
    .code-modal-content {{
        background-color: #1e1e1e;
        margin: 3% auto;
        padding: 20px;
        border-radius: 8px;
        width: 85%;
        max-width: 1000px;
        max-height: 85vh;
        overflow: auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .code-modal-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #444;
    }}
    .code-modal-title {{
        color: #fff;
        font-family: Arial, sans-serif;
        font-size: 16px;
        margin: 0;
    }}
    .code-modal-close {{
        color: #aaa;
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
        background: none;
        border: none;
        padding: 0;
        line-height: 1;
    }}
    .code-modal-close:hover {{
        color: #fff;
    }}
    /* Override Prism styles for our modal */
    .code-modal pre[class*="language-"] {{
        margin: 0;
        padding: 15px;
        border-radius: 5px;
        font-size: 13px;
        line-height: 1.6;
        tab-size: 4;
        background: #2d2d2d;
        overflow-x: auto;
    }}
    .code-modal code[class*="language-"] {{
        font-family: 'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
        tab-size: 4;
    }}
    .code-modal-info {{
        color: #888;
        font-family: Arial, sans-serif;
        font-size: 12px;
        margin-bottom: 10px;
    }}
    .code-modal-info span {{
        margin-right: 20px;
    }}
    .code-modal-info .label {{
        color: #aaa;
    }}
    .code-modal-info .value {{
        color: #4fc3f7;
    }}
    .code-modal-info .value.true {{
        color: #81c784;
    }}
    .code-modal-info .value.false {{
        color: #e57373;
    }}
    </style>
    
    <!-- Code Modal -->
    <div id="codeModal" class="code-modal">
        <div class="code-modal-content">
            <div class="code-modal-header">
                <h3 class="code-modal-title">Code for Usefulness Check - Task <span id="modalTaskIndex"></span></h3>
                <button class="code-modal-close" onclick="closeCodeModal()">&times;</button>
            </div>
            <div class="code-modal-info">
                <span><span class="label">Predicate:</span> <span class="value" id="modalPredicate"></span></span>
                <span><span class="label">Final Decision:</span> <span class="value" id="modalFinalDecision"></span></span>
                <span><span class="label">Usefulness Decision:</span> <span class="value" id="modalUsefulnessDecision"></span></span>
            </div>
            <pre class="language-c"><code id="modalCode" class="language-c"></code></pre>
        </div>
    </div>
    
    <div style="position: fixed; top: 10px; right: 10px; background: white; padding: 12px; border: 2px solid #ddd; border-radius: 5px; z-index: 1000; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="margin-bottom: 8px;">
            <label style="cursor: pointer; font-family: Arial; font-size: 14px;">
                <input type="checkbox" id="toggleUnknown" checked style="margin-right: 5px; cursor: pointer;">
                Show UNKNOWN points
            </label>
        </div>
        <div style="margin-bottom: 8px;">
            <label style="cursor: pointer; font-family: Arial; font-size: 14px;">
                <input type="checkbox" id="toggleGenTime" style="margin-right: 5px; cursor: pointer;">
                Include Generation Time
            </label>
        </div>
        <div style="font-size: 11px; color: #666; font-family: Arial;">
            💡 Click a point to view code
        </div>
    </div>
    <script>
    var metricWithoutGen = "{metric_str_without}";
    var metricWithGen = "{metric_str_with}";

    var noGenIndicesAlways = {ng_always_js};
    var noGenIndicesUnknown = {ng_unk_js};
    var genIndicesAlways = {g_always_js};
    var genIndicesUnknown = {g_unk_js};
    
    // Code data for click-to-view
    var codeData = {code_data_js};

    function setVisible(plotDiv, indices, isVisible) {{
        if (indices.length > 0) {{
            Plotly.restyle(plotDiv, {{
                'visible': isVisible,
                'showlegend': isVisible
            }}, indices);
        }}
    }}

    function updatePlot() {{
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        var showUnknown = document.getElementById('toggleUnknown').checked;
        var includeGenTime = document.getElementById('toggleGenTime').checked;

        var currentXRange = plotDiv.layout.xaxis.range.slice();
        var currentYRange = plotDiv.layout.yaxis.range.slice();

        if (includeGenTime) {{
            setVisible(plotDiv, noGenIndicesAlways, false);
            setVisible(plotDiv, noGenIndicesUnknown, false);
            setVisible(plotDiv, genIndicesAlways, true);
            setVisible(plotDiv, genIndicesUnknown, showUnknown);

            var annotations = plotDiv.layout.annotations;
            if (annotations && annotations.length > 0) {{
                annotations[annotations.length - 1].text = metricWithGen;
            }}
            Plotly.relayout(plotDiv, {{
                'xaxis.title.text': 'LLM Verification+Gen Time (s)',
                'xaxis.range': currentXRange,
                'xaxis.autorange': false,
                'yaxis.range': currentYRange,
                'yaxis.autorange': false,
                'annotations': annotations
            }});
        }} else {{
            setVisible(plotDiv, genIndicesAlways, false);
            setVisible(plotDiv, genIndicesUnknown, false);
            setVisible(plotDiv, noGenIndicesAlways, true);
            setVisible(plotDiv, noGenIndicesUnknown, showUnknown);

            var annotations = plotDiv.layout.annotations;
            if (annotations && annotations.length > 0) {{
                annotations[annotations.length - 1].text = metricWithoutGen;
            }}
            Plotly.relayout(plotDiv, {{
                'xaxis.title.text': 'LLM Verification Time (s)',
                'xaxis.range': currentXRange,
                'xaxis.autorange': false,
                'yaxis.range': currentYRange,
                'yaxis.autorange': false,
                'annotations': annotations
            }});
        }}
    }}

    // Modal functions
    function openCodeModal(taskIndex) {{
        var data = codeData.find(function(d) {{ return d.task_index === taskIndex; }});
        if (!data) {{
            alert('No code data found for task ' + taskIndex);
            return;
        }}
        
        document.getElementById('modalTaskIndex').textContent = taskIndex;
        document.getElementById('modalPredicate').textContent = data.predicate || 'N/A';
        
        var finalDecisionEl = document.getElementById('modalFinalDecision');
        finalDecisionEl.textContent = data.final_decision || 'N/A';
        finalDecisionEl.className = 'value ' + (data.final_decision === 'TRUE' ? 'true' : 'false');
        
        var usefulnessDecisionEl = document.getElementById('modalUsefulnessDecision');
        usefulnessDecisionEl.textContent = data.usefulness_decision || 'N/A';
        usefulnessDecisionEl.className = 'value ' + (data.usefulness_decision === 'TRUE' ? 'true' : 'false');
        
        var code = data.code_for_usefulness || 'No code available';
        var codeEl = document.getElementById('modalCode');
        codeEl.textContent = code;
        
        // Apply Prism syntax highlighting
        if (typeof Prism !== 'undefined') {{
            Prism.highlightElement(codeEl);
        }}
        
        document.getElementById('codeModal').style.display = 'block';
    }}
    
    function closeCodeModal() {{
        document.getElementById('codeModal').style.display = 'none';
    }}
    
    // Close modal when clicking outside
    window.onclick = function(event) {{
        var modal = document.getElementById('codeModal');
        if (event.target === modal) {{
            closeCodeModal();
        }}
    }}
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(event) {{
        if (event.key === 'Escape') {{
            closeCodeModal();
        }}
    }});

    document.addEventListener('DOMContentLoaded', function() {{
        document.getElementById('toggleUnknown').addEventListener('change', updatePlot);
        document.getElementById('toggleGenTime').addEventListener('change', updatePlot);
        
        // Add click handler for plot points
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        plotDiv.on('plotly_click', function(data) {{
            if (data.points && data.points.length > 0) {{
                var point = data.points[0];
                // Task index is in customdata[2]
                var taskIndex = point.customdata ? point.customdata[2] : null;
                if (taskIndex !== null && taskIndex !== undefined) {{
                    openCodeModal(parseInt(taskIndex));
                }}
            }}
        }});
    }});
    </script>
    """

    # Insert checkbox HTML after opening body tag
    html_string = html_string.replace('<body>', '<body>\n' + checkbox_html)
    
    # Save to file
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plot_path, 'w') as f:
        f.write(html_string)
    
    return plot_path


def plot_from_json(
    results_path: Path,
    model_name: str = "gpt-oss-20b",
    baseline_name: str = "UAutomizer25",
    split_name: str = "easy",
    plot_path: Optional[Path] = None
) -> Path:
    """
    Generate plot from a JSON results file.
    
    Args:
        results_path: Path to JSON file with results
        model_name: Model name for title
        baseline_name: Baseline name for title
        split_name: Split name for title
        plot_path: Output path (defaults to same dir as results)
        
    Returns:
        Path to generated HTML file
    """
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError(f"Unexpected JSON structure in {results_path}")
    
    if plot_path is None:
        plot_path = results_path.parent / f"{results_path.stem}_plot.html"
    
    return plot_verification_vs_baseline(
        results=results,
        model_name=model_name,
        baseline_name=baseline_name,
        split_name=split_name,
        plot_path=plot_path
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation plot from results")
    parser.add_argument("--results", type=Path, required=True, help="Path to results JSON file")
    parser.add_argument("--model", type=str, default="gpt-oss-20b", help="Model name")
    parser.add_argument("--baseline", type=str, default="UAutomizer25", help="Baseline name")
    parser.add_argument("--split", type=str, default="easy", help="Split name")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    
    args = parser.parse_args()
    
    plot_path = plot_from_json(
        results_path=args.results,
        model_name=args.model,
        baseline_name=args.baseline,
        split_name=args.split,
        plot_path=args.output
    )
    print(f"Plot saved to: {plot_path}")

