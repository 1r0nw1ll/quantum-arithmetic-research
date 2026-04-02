#!/usr/bin/env python3
"""Generate figures for the observer-coherence paper."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figure 1: Cross-domain surrogate validation summary
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

domains = ['Finance', 'ERA5', 'Audio', 'EEG', 'Teleconn.', 'Seismology']
real_vals = [0.31, 0.46, 0.75, 0.19, 95.0, 0.21]  # |real metric|
surr_means = [0.05, 0.01, 0.00, 0.02, 16.7, 0.13]  # surrogate mean
surr_stds = [0.05, 0.14, 0.40, 0.007, 13.0, 0.13]  # surrogate std
pass_counts = [3, 4, 3, 2, 4, 0]
total_tests = [4, 4, 4, 3, 4, 4]
metrics = ['|r|', '|r|', '|partial r|', 'mean ΔR²', 'χ²', '|r|']

# Normalize to z-scores for comparable display
z_scores = [(r - m) / s if s > 0 else 0 for r, m, s in zip(real_vals, surr_means, surr_stds)]

colors = ['#2ecc71' if p >= 2 else '#e74c3c' for p in pass_counts]
bars = ax.barh(range(len(domains)), z_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

ax.axvline(x=1.96, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='z = 1.96 (p < 0.05)')
ax.axvline(x=0, color='black', linewidth=0.8)

for i, (z, p, t, m) in enumerate(zip(z_scores, pass_counts, total_tests, metrics)):
    label = f'{p}/{t}'
    ax.text(max(z + 0.3, 1.0), i, label, va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(domains)))
ax.set_yticklabels(domains, fontsize=11)
ax.set_xlabel('z-score (real vs. surrogate distribution)', fontsize=11)
ax.set_title('Process-Level Surrogate Validation Across Six Domains', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(-1, max(z_scores) + 3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('fig1_surrogate_validation.pdf', bbox_inches='tight')
plt.savefig('fig1_surrogate_validation.png', dpi=200, bbox_inches='tight')
print("Saved fig1_surrogate_validation.pdf/png")

# Figure 2: Pipeline architecture diagram (text-based)
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3))
ax2.axis('off')

steps = [
    ('Multi-channel\nsignal x_t ∈ R^d', '#3498db'),
    ('k-means\nclustering', '#2ecc71'),
    ('QA state\nmapping φ', '#e67e22'),
    ('T-operator\nprediction', '#9b59b6'),
    ('Rolling QCI\n(window W)', '#e74c3c'),
]

for i, (label, color) in enumerate(steps):
    x = 0.1 + i * 0.18
    rect = plt.Rectangle((x, 0.3), 0.14, 0.4, facecolor=color, alpha=0.3,
                           edgecolor=color, linewidth=2, transform=ax2.transAxes)
    ax2.add_patch(rect)
    ax2.text(x + 0.07, 0.5, label, ha='center', va='center',
             fontsize=9, fontweight='bold', transform=ax2.transAxes)
    if i < len(steps) - 1:
        ax2.annotate('', xy=(x + 0.17, 0.5), xytext=(x + 0.14, 0.5),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                     transform=ax2.transAxes)

# Firewall markers
ax2.text(0.28, 0.2, '← Observer boundary →', ha='center', fontsize=8,
         style='italic', color='gray', transform=ax2.transAxes)
ax2.text(0.64, 0.2, '← QA layer (integer only) →', ha='center', fontsize=8,
         style='italic', color='gray', transform=ax2.transAxes)
ax2.text(0.92, 0.2, '← Projection →', ha='center', fontsize=8,
         style='italic', color='gray', transform=ax2.transAxes)

ax2.set_title('Topographic Observer Pipeline', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fig2_pipeline.pdf', bbox_inches='tight')
plt.savefig('fig2_pipeline.png', dpi=200, bbox_inches='tight')
print("Saved fig2_pipeline.pdf/png")

print("Done — 2 figures generated")
