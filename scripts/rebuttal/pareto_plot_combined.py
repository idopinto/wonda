import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
})

models_vbp = {
    'GPT-5.2':      {'r_correct': 72.4, 'y': 155.6, 'group': 'proprietary'},
    'GPT-OSS-120B': {'r_correct': 58.0, 'y': 165.8, 'group': 'oss'},
    'Qwen3-8B':     {'r_correct': 23.9, 'y': 181.6, 'group': 'base'},
    'Qwen3-8B-V2':  {'r_correct': 42.8, 'y': 166.5, 'group': 'ours'},
    'Qwen3-4B':     {'r_correct': 22.8, 'y': 185.6, 'group': 'base'},
    'Qwen3-4B-V2':  {'r_correct': 44.4, 'y': 165.5, 'group': 'ours'},
}

models_e2e = {
    'GPT-5.2':      {'r_correct': 72.4, 'y': 163.4, 'group': 'proprietary'},
    'GPT-OSS-120B': {'r_correct': 58.0, 'y': 167.6, 'group': 'oss'},
    'Qwen3-8B':     {'r_correct': 23.9, 'y': 181.7, 'group': 'base'},
    'Qwen3-8B-V2':  {'r_correct': 42.8, 'y': 166.7, 'group': 'ours'},
    'Qwen3-4B':     {'r_correct': 22.8, 'y': 185.7, 'group': 'base'},
    'Qwen3-4B-V2':  {'r_correct': 44.4, 'y': 165.7, 'group': 'ours'},
}

group_styles = {
    'proprietary': {'color': '#2c3e50', 'marker': 'D', 'size': 110, 'label': 'Proprietary LLM'},
    'oss':         {'color': '#8e44ad', 'marker': 'D', 'size': 110, 'label': 'Open-Weights LLM'},
    'base':        {'color': '#c0392b', 'marker': 'o', 'size': 80,  'label': 'Qwen3 Base'},
    'ours':        {'color': '#27ae60', 'marker': 's', 'size': 110, 'label': 'Qwen3-V2 (Ours)'},
}

arrows = [
    ('Qwen3-8B', 'Qwen3-8B-V2'),
    ('Qwen3-4B', 'Qwen3-4B-V2'),
]

offsets = {
    'GPT-5.2':       ( 8,  -14),
    'GPT-OSS-120B':  ( 8,   9),
    'Qwen3-8B':      (-62,   0),
    'Qwen3-8B-V2':   ( 8,   14),
    'Qwen3-4B':      (-62,  -8),
    'Qwen3-4B-V2':   ( 8,  -16),
}

use_connector = {'Qwen3-8B', 'Qwen3-4B', 'Qwen3-8B-V2', 'Qwen3-4B-V2'}


def plot_panel(ax, models, ylabel, title):
    for base_name, v2_name in arrows:
        b, v = models[base_name], models[v2_name]
        ax.annotate('', xy=(v['r_correct'], v['y']),
                    xytext=(b['r_correct'], b['y']),
                    arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.8,
                                    connectionstyle='arc3,rad=0.12'))

    plotted_labels = set()
    for name, d in models.items():
        s = group_styles[d['group']]
        label = s['label'] if s['label'] not in plotted_labels else None
        plotted_labels.add(s['label'])
        ax.scatter(d['r_correct'], d['y'], c=s['color'], marker=s['marker'],
                   s=s['size'], zorder=5, label=label, edgecolors='white', linewidths=0.8)

    for name, d in models.items():
        ox, oy = offsets[name]
        fontweight = 'bold' if d['group'] == 'ours' else 'normal'
        arrow_props = dict(arrowstyle='-', color='#bbbbbb', lw=0.8) if name in use_connector else None
        ax.annotate(name, (d['r_correct'], d['y']),
                    textcoords='offset points', xytext=(ox, oy),
                    fontsize=10, fontweight=fontweight,
                    color=group_styles[d['group']]['color'],
                    arrowprops=arrow_props)

    ax.set_xlabel(r'$R_{\mathrm{correct}}$ (%)    $\longrightarrow$ better', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.invert_yaxis()
    ax.set_xlim(14, 82)
    ax.set_ylim(190, 148)
    ax.grid(True, alpha=0.2)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

plot_panel(ax1, models_vbp,
           r'VBP (s)    $\longleftarrow$ better',
           '(a) Correctness vs. VBP')

plot_panel(ax2, models_e2e,
           r'$\mathrm{VBP_{E2E}}$ (s)    $\longleftarrow$ better',
           r'(b) Correctness vs. $\mathrm{VBP_{E2E}}$')

ax1.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

fig.tight_layout(w_pad=3)
out = '/cs/labs/guykatz/idopinto12/projects/wonda/scripts/rebuttal'
fig.savefig(f'{out}/pareto_combined.png', dpi=200, bbox_inches='tight')
fig.savefig(f'{out}/pareto_combined.pdf', bbox_inches='tight')
print("Saved pareto_combined.png and pareto_combined.pdf")
