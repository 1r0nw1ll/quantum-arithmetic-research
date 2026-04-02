#!/usr/bin/env python3
"""Generate Figure 1 for the Fibonacci resonance paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from the analysis
orders = [0, 1, 2, 3, 4, 5]
fib_counts = [1, 33, 6, 4, 1, 0]
nonfib_counts = [0, 10, 0, 2, 1, 2]
expected_fib_pct = [100, 22.2, 50.0, 40.0, 33.3, 25.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Stacked bar chart by order
x = np.arange(len(orders))
width = 0.6
bars_fib = ax1.bar(x, fib_counts, width, label='Fibonacci', color='#2166ac', alpha=0.85)
bars_nonfib = ax1.bar(x, nonfib_counts, width, bottom=fib_counts, label='Non-Fibonacci', color='#b2182b', alpha=0.85)

ax1.set_xlabel('Resonance order $|p-q|$', fontsize=11)
ax1.set_ylabel('Number of resonances', fontsize=11)
ax1.set_title('(a) Resonance count by order', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(orders)
ax1.legend(fontsize=9, loc='upper right')

# Add percentage labels
for i, (f, nf) in enumerate(zip(fib_counts, nonfib_counts)):
    total = f + nf
    if total > 0:
        pct = f / total * 100
        ax1.text(i, total + 0.5, f'{pct:.0f}%', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='#2166ac')

# Panel B: Observed vs expected Fibonacci rate
orders_nonzero = [1, 2, 3, 4, 5]
obs_pct = [77, 100, 67, 50, 0]
exp_pct = [22.2, 50.0, 40.0, 33.3, 25.0]

x2 = np.arange(len(orders_nonzero))
width2 = 0.35
bars_obs = ax2.bar(x2 - width2/2, obs_pct, width2, label='Observed', color='#2166ac', alpha=0.85)
bars_exp = ax2.bar(x2 + width2/2, exp_pct, width2, label='Expected (uniform)', color='#999999', alpha=0.6)

ax2.set_xlabel('Resonance order $|p-q|$', fontsize=11)
ax2.set_ylabel('Fibonacci fraction (%)', fontsize=11)
ax2.set_title('(b) Observed vs expected Fibonacci rate', fontsize=12, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(orders_nonzero)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 115)
ax2.axhline(y=31.2, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.text(len(orders_nonzero)-0.5, 33, 'overall expected\n(31%)', fontsize=7, color='gray', ha='right')

# Add significance markers
for i, (o, e) in enumerate(zip(obs_pct, exp_pct)):
    if o > e * 1.5 and o > 0:  # substantially above expected
        ax2.text(i - width2/2, o + 2, '*', ha='center', fontsize=14, fontweight='bold', color='#2166ac')

plt.tight_layout()
plt.savefig('figure1_fibonacci_resonance.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figure1_fibonacci_resonance.png', bbox_inches='tight', dpi=150)
print('Saved figure1_fibonacci_resonance.pdf and .png')
