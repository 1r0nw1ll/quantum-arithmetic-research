"""
QA Analysis Plotting - FIXED VERSION
Shows test accuracy, uses correct logit metrics, highlights grokking vs saturation
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
QA_LOGS_DIR = Path("qa_logs")
RUN_ID = "modular_addition_cross_entropy_seed0"
LOG_FILE = QA_LOGS_DIR / f"{RUN_ID}.jsonl"

# === LOAD QA LOGS ===
print(f"Loading: {LOG_FILE}")

records = []
with open(LOG_FILE, 'r') as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# === CONVERT TO DATAFRAME ===
rows = []
for rec in records:
    row = {
        'step': rec['step'],
        'legal': rec['generators']['sgd_step']['legal'],
        'reason': rec['generators']['sgd_step']['reason'],
    }
    row.update(rec['state'])
    row.update({f"cum_{k.lower()}": v for k, v in rec['cumulative_failures'].items()})
    rows.append(row)

df = pd.DataFrame(rows)

# === IDENTIFY KEY EVENTS ===
first_illegal = df[~df['legal']]['step'].min() if not df[df['legal'] == False].empty else None
grok_threshold = 0.95

# Check if we have test_acc
has_test_acc = 'test_acc' in df.columns

if has_test_acc:
    grok_step = df[df['test_acc'] > grok_threshold]['step'].min() if not df[df['test_acc'] > grok_threshold].empty else None
else:
    grok_step = None
    print("⚠ No test accuracy logged - can't detect grokking")

print(f"\n=== KEY EVENTS ===")
print(f"First illegal step: {first_illegal}")
if has_test_acc:
    print(f"Grokking step (test_acc > {grok_threshold}): {grok_step}")
print(f"\nFinal train accuracy: {df['train_acc'].iloc[-1]:.3f}")
if has_test_acc:
    print(f"Final test accuracy: {df['test_acc'].iloc[-1]:.3f}")

# === VISUALIZATION ===
fig, axes = plt.subplots(5 if has_test_acc else 4, 1, figsize=(12, 12 if has_test_acc else 10), sharex=True)
ax_idx = 0

# Panel A: Training metrics
ax = axes[ax_idx]
ax_idx += 1
ax.plot(df['step'], df['train_loss'], label='Train Loss', alpha=0.7, color='blue')
ax.set_ylabel('Train Loss', color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax2 = ax.twinx()
ax2.plot(df['step'], df['train_acc'], label='Train Acc', alpha=0.7, color='green')
if has_test_acc:
    ax2.plot(df['step'], df['test_acc'], label='Test Acc', alpha=0.7, color='orange', linewidth=2)
ax2.set_ylabel('Accuracy', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim([0, 1.05])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.set_title('QA View: Training Dynamics at Numerical Boundaries', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Mark grokking if detected
if grok_step is not None:
    ax.axvline(grok_step, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='Grokking')

# Panel B: Logit magnitude (FIXED)
ax = axes[ax_idx]
ax_idx += 1
ax.plot(df['step'], df['logit_max_abs'], label='Max |Logit|', color='red', alpha=0.7)
ax.plot(df['step'], df['logit_norm'] / 100, label='Logit Norm / 100', color='darkred', alpha=0.7, linestyle='--')
ax.axhline(85, color='red', linestyle=':', alpha=0.5, label='Overflow threshold (fp32)')
ax.set_ylabel('Logit Magnitude')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Panel C: Softmax stability
ax = axes[ax_idx]
ax_idx += 1
ax2 = ax.twinx()
ax.plot(df['step'], df['p_entropy'], label='Entropy', color='green', alpha=0.7)
ax.axhline(0.01, color='green', linestyle='--', alpha=0.5, label='Collapse threshold')
ax2.plot(df['step'], df['p_max'], label='P(max)', color='orange', alpha=0.7)
ax.set_ylabel('Entropy', color='green')
ax2.set_ylabel('P(max)', color='orange')
ax.set_yscale('log')
ax.tick_params(axis='y', labelcolor='green')
ax2.tick_params(axis='y', labelcolor='orange')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(alpha=0.3)

# Panel D: Gradient health
ax = axes[ax_idx]
ax_idx += 1
ax.plot(df['step'], df['grad_norm'], label='Gradient Norm', color='purple', alpha=0.7)
ax.axhline(1e-6, color='purple', linestyle='--', alpha=0.5, label='Effectively dead')
ax.set_ylabel('Gradient Norm')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Panel E: Generator legality (THE KEY PANEL)
ax = axes[ax_idx]
legality = df['legal'].astype(int)
ax.fill_between(df['step'], 0, legality, alpha=0.4, label='Legal (1)', color='green')
ax.fill_between(df['step'], legality, 1, alpha=0.4, label='Illegal (0)', color='red')

if first_illegal is not None:
    ax.axvline(first_illegal, color='darkred', linestyle='--', linewidth=3, alpha=0.8,
               label=f'First Illegal: {first_illegal}')
if grok_step is not None:
    ax.axvline(grok_step, color='purple', linestyle='--', linewidth=3, alpha=0.8,
               label=f'Grokking: {grok_step}')

ax.set_ylabel('Generator\nLegality', fontsize=11, fontweight='bold')
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylim([0, 1])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Illegal', 'Legal'])
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Annotation
if first_illegal is not None:
    if grok_step is None or first_illegal < grok_step:
        annotation = 'Legality lost\nbefore grokking\n(saturation)'
    else:
        annotation = 'Learning stops\nwhen legality\nflips to 0'

    ax.text(0.02, 0.5, annotation,
            transform=ax.transAxes, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'qa_analysis_fixed_{RUN_ID}.png', dpi=150)
print(f"\n✓ Saved: qa_analysis_fixed_{RUN_ID}.png")
plt.show()

# === DIAGNOSIS ===
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

final_train_acc = df['train_acc'].iloc[-1]
final_test_acc = df['test_acc'].iloc[-1] if has_test_acc else None

if has_test_acc:
    if final_train_acc > 0.99 and final_test_acc < 0.5:
        print("❌ MEMORIZATION (no grokking)")
        print("   Train acc ~1.0, test acc low")
        print("   Need longer training or different hyperparams")
    elif final_train_acc > 0.99 and final_test_acc > 0.95:
        if grok_step is not None and grok_step > 1000:
            print("✓ GROKKING DETECTED")
            print(f"   Test acc jumped at step {grok_step}")
        else:
            print("⚠ FAST LEARNING (not classic grokking)")
            print("   Both train/test high from early on")
    else:
        print("⚠ STILL TRAINING")
        print("   Neither train nor test fully converged")
else:
    print("⚠ NO TEST DATA")
    print("   Can't distinguish grokking from memorization")

if first_illegal is not None and first_illegal < 1000:
    print(f"\n⚠ EARLY SATURATION (step {first_illegal})")
    print("   Softmax collapsed + gradients died early")
    print("   This may prevent grokking from occurring")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

if not has_test_acc:
    print("1. Add test set evaluation to logger")
    print("2. Rerun with log_step(..., test_logits, test_targets)")
elif grok_step is None:
    print("1. Run longer (50k+ epochs) or use GPU")
    print("2. Try different LR or regularization")
    print("3. Compare with StableMax to extend legality window")
else:
    print("✓ Ready to publish!")
    print("  You have grokking + legality flip correlation")

print("="*60)
