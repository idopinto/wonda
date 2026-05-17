import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
})

models = {
    'GPT-5.2':           {'r_correct': 72.4, 'vbp_e2e': 163.4, 'group': 'proprietary'},
    'GPT-OSS-120B':      {'r_correct': 58.0, 'vbp_e2e': 167.6, 'group': 'oss'},
    'Qwen3-8B':          {'r_correct': 23.9, 'vbp_e2e': 181.7, 'group': 'base'},
    'Qwen3-8B-V2':       {'r_correct': 42.8, 'vbp_e2e': 166.7, 'group': 'ours'},
    'Qwen3-4B':          {'r_correct': 22.8, 'vbp_e2e': 185.7, 'group': 'base'},
    'Qwen3-4B-V2':       {'r_correct': 44.4, 'vbp_e2e': 165.7, 'group': 'ours'},
}

group_styles = {
    'proprietary': {'color': '#2c3e50', 'marker': 'D', 'size': 120, 'label': 'Proprietary LLM'},
    'oss':         {'color': '#8e44ad', 'marker': 'D', 'size': 120, 'label': 'Open-Weights LLM'},
    'base':        {'color': '#c0392b', 'marker': 'o', 'size': 90,  'label': 'Qwen3 Base'},
    'ours':        {'color': '#27ae60', 'marker': 's', 'size': 120, 'label': 'Qwen3-V2 (Ours)'},
}

arrows = [
    ('Qwen3-8B', 'Qwen3-8B-V2'),
    ('Qwen3-4B', 'Qwen3-4B-V2'),
]

fig, ax = plt.subplots(figsize=(9, 6))

for base_name, v2_name in arrows:
    b, v = models[base_name], models[v2_name]
    ax.annotate('', xy=(v['r_correct'], v['vbp_e2e']),
                xytext=(b['r_correct'], b['vbp_e2e']),
                arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.8,
                                connectionstyle='arc3,rad=0.12'))

plotted_labels = set()
for name, d in models.items():
    s = group_styles[d['group']]
    label = s['label'] if s['label'] not in plotted_labels else None
    plotted_labels.add(s['label'])
    ax.scatter(d['r_correct'], d['vbp_e2e'], c=s['color'], marker=s['marker'],
               s=s['size'], zorder=5, label=label, edgecolors='white', linewidths=0.8)

offsets = {
    'GPT-5.2':       ( 8,  -14),
    'GPT-OSS-120B':  ( 8,   9),
    'Qwen3-8B':      (-65,   0),
    'Qwen3-8B-V2':   ( 8,   14),
    'Qwen3-4B':      (-65,  -8),
    'Qwen3-4B-V2':   ( 8,  -16),
}

use_connector = {'Qwen3-8B', 'Qwen3-4B', 'Qwen3-8B-V2', 'Qwen3-4B-V2'}

for name, d in models.items():
    ox, oy = offsets[name]
    fontweight = 'bold' if d['group'] == 'ours' else 'normal'
    arrow_props = dict(arrowstyle='-', color='#bbbbbb', lw=0.8) if name in use_connector else None
    ax.annotate(name, (d['r_correct'], d['vbp_e2e']),
                textcoords='offset points', xytext=(ox, oy),
                fontsize=11, fontweight=fontweight,
                color=group_styles[d['group']]['color'],
                arrowprops=arrow_props)

ax.set_xlabel(r'$R_{\mathrm{correct}}$ (%)    $\longrightarrow$ better', fontsize=13)
ax.set_ylabel(r'$\mathrm{VBP_{E2E}}$ (s)    $\longleftarrow$ better', fontsize=13)
ax.set_title('Correctness vs. End-to-End Verification Speed (Hard Instances)', fontsize=14, pad=12)

ax.invert_yaxis()
ax.set_xlim(14, 82)
ax.set_ylim(190, 155)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2)

fig.tight_layout()
out = '/cs/labs/guykatz/idopinto12/projects/wonda/scripts/rebuttal'
fig.savefig(f'{out}/pareto_frontier_e2e.png', dpi=200, bbox_inches='tight')
fig.savefig(f'{out}/pareto_frontier_e2e.pdf', bbox_inches='tight')
print("Saved pareto_frontier_e2e.png and pareto_frontier_e2e.pdf")
