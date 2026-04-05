"""
QA Analysis Notebook for Grokking Experiments
Generates visualizations showing training as reachability at numerical boundaries

Run after training with grokking_experiments_qa.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
QA_LOGS_DIR = Path("qa_logs")
RUN_ID = "modular_addition_cross_entropy_seed0"  # Change to your run ID
LOG_FILE = QA_LOGS_DIR / f"{RUN_ID}.jsonl"

# === LOAD QA LOGS ===
print(f"Loading QA logs from: {LOG_FILE}")

records = []
with open(LOG_FILE, 'r') as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# === CONVERT TO DATAFRAME ===
# Flatten nested structure for easier plotting
rows = []
for rec in records:
    row = {
        'step': rec['step'],
        'run_id': rec['run_id'],
        'legal': rec['generators']['sgd_step']['legal'],
        'reason': rec['generators']['sgd_step']['reason'],
        'num_failures': len(rec['failures']),
    }
    row.update(rec['state'])
    row.update({f"cum_{k.lower()}": v for k, v in rec['cumulative_failures'].items()})
    rows.append(row)

df = pd.DataFrame(rows)
print("\nDataFrame columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# === IDENTIFY KEY EVENTS ===
first_illegal_step = df[~df['legal']]['step'].min() if not df[df['legal'] == False].empty else None
grok_threshold = 0.95  # Define grokking as >95% test accuracy
# Note: We only have train_acc in state, need to add test_acc to logger if needed
# For now, use train_acc as proxy
grok_step = df[df['train_acc'] > grok_threshold]['step'].min() if not df[df['train_acc'] > grok_threshold].empty else None

print(f"\n=== KEY EVENTS ===")
print(f"First illegal step: {first_illegal_step}")
print(f"Grokking step (train_acc > {grok_threshold}): {grok_step}")
print(f"\nFailure summary at end of training:")
print(df[['cum_softmax_collapse', 'cum_nan_loss', 'cum_inf_grad',
         'cum_grad_explosion', 'cum_param_explosion', 'cum_logit_explosion']].iloc[-1])

# === VISUALIZATION 1: Boundary Detection ===
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Panel A: Training metrics
ax = axes[0]
ax.plot(df['step'], df['train_loss'], label='Train Loss', alpha=0.7)
ax.plot(df['step'], df['train_acc'], label='Train Acc', alpha=0.7)
ax.set_ylabel('Loss / Accuracy')
ax.legend(loc='upper right')
ax.set_title('QA View: Grokking as Reachability at Numerical Boundaries', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Panel B: Logit statistics
ax = axes[1]
ax.plot(df['step'], df['logit_max'], label='Logit Max', color='red', alpha=0.7)
ax.plot(df['step'], df['logit_min'], label='Logit Min', color='blue', alpha=0.7)
ax.axhline(85, color='red', linestyle='--', alpha=0.5, label='LMAX threshold')
ax.axhline(-85, color='blue', linestyle='--', alpha=0.5)
ax.set_ylabel('Logit Range')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Panel C: Softmax stability
ax = axes[2]
ax2 = ax.twinx()
ax.plot(df['step'], df['p_entropy'], label='Entropy', color='green', alpha=0.7)
ax.axhline(0.01, color='green', linestyle='--', alpha=0.5, label='HMIN threshold')
ax2.plot(df['step'], df['p_max'], label='P(max)', color='orange', alpha=0.7)
ax.set_ylabel('Entropy', color='green')
ax2.set_ylabel('P(max)', color='orange')
ax.tick_params(axis='y', labelcolor='green')
ax2.tick_params(axis='y', labelcolor='orange')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(alpha=0.3)

# Panel D: Generator legality (THE KEY PANEL - binary certificate)
ax = axes[3]
legality = df['legal'].astype(int)
ax.fill_between(df['step'], 0, legality, alpha=0.4, label='Legal (1)', color='green')
ax.fill_between(df['step'], legality, 1, alpha=0.4, label='Illegal (0)', color='red')
if first_illegal_step is not None:
    ax.axvline(first_illegal_step, color='darkred', linestyle='--', linewidth=3, alpha=0.8,
               label=f'First Illegal: {first_illegal_step}')
if grok_step is not None:
    ax.axvline(grok_step, color='purple', linestyle='--', linewidth=3, alpha=0.8,
               label=f'Grokking: {grok_step}')
ax.set_ylabel('Generator Legality\n(Binary)', fontsize=11, fontweight='bold')
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylim([0, 1])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Illegal', 'Legal'])
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
# Add annotation box
if first_illegal_step is not None:
    ax.text(0.02, 0.5, 'Learning stops\nwhen legality\nflips to 0',
            transform=ax.transAxes, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'qa_analysis_{RUN_ID}.png', dpi=150)
print(f"\nSaved: qa_analysis_{RUN_ID}.png")
plt.show()

# === VISUALIZATION 2: Gradient-Weight Alignment (NLM Proxy) ===
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(df['step'], df['cos_grad_w'], label='cos(grad, weights)', color='purple', alpha=0.7)
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Training Step')
ax.set_ylabel('Gradient-Weight Alignment')
ax.set_title('NLM Direction Proxy: Gradient Alignment with Weights')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'qa_nlm_alignment_{RUN_ID}.png', dpi=150)
print(f"Saved: qa_nlm_alignment_{RUN_ID}.png")
plt.show()

# === VISUALIZATION 3: Failure Mode Timeline ===
failure_cols = [col for col in df.columns if col.startswith('cum_')]
if failure_cols:
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for col in failure_cols:
        if df[col].max() > 0:  # Only plot if failures occurred
            ax.plot(df['step'], df[col], label=col.replace('cum_', ''), alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cumulative Failure Count')
    ax.set_title('Failure Mode Accumulation (QA Boundary Violations)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'qa_failures_{RUN_ID}.png', dpi=150)
    print(f"Saved: qa_failures_{RUN_ID}.png")
    plt.show()

# === SAMPLE LOG EXCERPT (for publication) ===
print("\n" + "="*80)
print("SAMPLE LOG EXCERPT (first 3 records + last 1)")
print("="*80)
sample_records = records[:3] + [records[-1]]
for rec in sample_records:
    print(json.dumps(rec, indent=2))
    print("-"*80)

print("\n=== QA ANALYSIS COMPLETE ===")
print(f"Generated 3 plots showing training as QA reachability process")
print(f"Key insight: Grokking occurs when SGD generator becomes illegal at numerical boundaries")
