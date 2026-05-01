import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(__file__)

stages = ['QK', 'SM', 'AV', 'ACC']
colors = {'QK': '#4C72B0', 'SM': '#DD8452', 'AV': '#55A868', 'ACC': '#C44E52'}
bar_w = 0.18
x = np.arange(4)
sweep_labels = ['25%', '50%', '100%', '200%']

# ── data ──────────────────────────────────────────────────────────────────────

glb_cycles = {
    'QK':  [0.00011017840006388724] * 4,
    'SM':  [1.5603809515596367e-05] * 4,
    'AV':  [0.00011017840006388724] * 4,
    'ACC': [1.2190476184059662e-07] * 4,
}
glb_total = [0.00044083550501738955] * 4

bw_cycles = {
    'QK':  [0.00044071360025554895, 0.00022035680012777448, 0.00011017840006388724, 5.508920003194362e-05],
    'SM':  [1.5603809515596367e-05] * 4,
    'AV':  [0.00044071360025554895, 0.00022035680012777448, 0.00011017840006388724, 5.508920003194362e-05],
    'ACC': [3.202084712938813e-07, 1.6010423564694065e-07, 1.2190476184059662e-07, 1.2190476184059662e-07],
}
bw_total = [0.0017631746094934897, 0.0008815873047467448, 0.00044083550501738955, 0.00022047870488961507]

ct_cycles = {
    'QK':  [0.00011017840006388724] * 4,
    'SM':  [6.241523806238547e-05, 3.1207619031192735e-05, 1.5603809515596367e-05, 7.801904757798184e-06],
    'AV':  [0.00011017840006388724] * 4,
    'ACC': [4.876190473623865e-07, 2.4380952368119324e-07, 1.2190476184059662e-07, 8.005211782347033e-08],
}
ct_total = [0.00044120121930291134, 0.00044095740977923015, 0.00044083550501738955, 0.0004407936523733724]

energy_vals = {
    'QK':  0.00440040966177228,
    'SM':  0.00013484449258408302,
    'AV':  0.004392215046662048,
    'ACC': 2.8405923653402333e-06,
}


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')


# ── Fig 1: per-stage latency — GLB capacity sweep ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for i, s in enumerate(stages):
    ax.bar(x + i * bar_w, [v * 1e6 for v in glb_cycles[s]], bar_w, label=s, color=colors[s])
ax.set_xticks(x + bar_w * 1.5)
ax.set_xticklabels(sweep_labels)
ax.set_xlabel('GLB Capacity (% of baseline 1 GB)')
ax.set_ylabel('Latency (µs)')
ax.set_title('Per-Stage Latency vs GLB Capacity')
ax.legend()
ax.set_ylim(bottom=0)
save(fig, 'fig1_glb_cycles.png')

# ── Fig 2: total pipeline latency — all sweeps ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sweep_labels, [v * 1e6 for v in glb_total], 'o-', label='GLB Capacity', color='#4C72B0')
ax.plot(sweep_labels, [v * 1e6 for v in bw_total],  's-', label='Memory BW',    color='#DD8452')
ax.plot(sweep_labels, [v * 1e6 for v in ct_total],  '^-', label='Compute TP',   color='#55A868')
ax.set_xlabel('Sweep Setting (% of baseline)')
ax.set_ylabel('Total Pipeline Latency (µs)')
ax.set_title('Total Pipeline Latency Across All Sweeps')
ax.legend()
ax.set_ylim(bottom=0)
save(fig, 'fig2_total_pipeline_cycles.png')

# ── Fig 3: per-stage latency — memory BW sweep ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for i, s in enumerate(stages):
    ax.bar(x + i * bar_w, [v * 1e6 for v in bw_cycles[s]], bar_w, label=s, color=colors[s])
ax.set_xticks(x + bar_w * 1.5)
ax.set_xticklabels(sweep_labels)
ax.set_xlabel('Memory Bandwidth (% of baseline 614 GB/s)')
ax.set_ylabel('Latency (µs)')
ax.set_title('Per-Stage Latency vs Memory Bandwidth')
ax.legend()
ax.set_ylim(bottom=0)
save(fig, 'fig3_bw_cycles.png')

# ── Fig 4: per-stage latency — compute throughput sweep ──────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for i, s in enumerate(stages):
    ax.bar(x + i * bar_w, [v * 1e6 for v in ct_cycles[s]], bar_w, label=s, color=colors[s])
ax.set_xticks(x + bar_w * 1.5)
ax.set_xticklabels(sweep_labels)
ax.set_xlabel('Compute Throughput (% of baseline 1.05 GHz)')
ax.set_ylabel('Latency (µs)')
ax.set_title('Per-Stage Latency vs Compute Throughput')
ax.legend()
ax.set_ylim(bottom=0)
save(fig, 'fig4_compute_cycles.png')

# ── Fig 5: energy breakdown (flat across all sweeps) ─────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
e_vals = [energy_vals[s] * 1e3 for s in stages]
bars = ax.bar(stages, e_vals, color=[colors[s] for s in stages])
ax.set_ylabel('Energy (mJ)')
ax.set_title('Per-Stage Energy (identical across all sweeps)')
for bar, v in zip(bars, e_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f'{v:.2f}', ha='center', va='bottom', fontsize=9)
save(fig, 'fig5_energy_breakdown.png')

# ── Fig 6: BW speedup vs ideal linear ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
bw_factors = [0.25, 0.5, 1.0, 2.0]
base_qk = bw_cycles['QK'][2]
actual_speedup = [base_qk / v for v in bw_cycles['QK']]
ax.plot(sweep_labels, bw_factors,      'k--', label='Ideal (linear)')
ax.plot(sweep_labels, actual_speedup,  'o-',  label='QK/AV actual', color='#4C72B0')
ax.set_xlabel('Memory Bandwidth (% of baseline)')
ax.set_ylabel('Speedup vs baseline')
ax.set_title('QK/AV Speedup vs Memory BW')
ax.legend()
ax.set_ylim(bottom=0)
save(fig, 'fig6_bw_speedup.png')

print('\nDone.')
