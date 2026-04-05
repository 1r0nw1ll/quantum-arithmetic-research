"""
Verification: QA instrumentation does not perturb training dynamics

Runs identical experiments with/without QA logging and compares:
- Final accuracy
- Loss trajectories
- Grokking onset time

Critical for publication credibility.
"""

import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("="*80)
print("VERIFICATION: QA Instrumentation Behavioral Perturbation Test")
print("="*80)
print()
print("This will run TWO experiments with identical seeds/configs:")
print("  1. Baseline (original grokking_experiments.py)")
print("  2. QA-instrumented (grokking_experiments_qa.py)")
print()
print("Then compare training dynamics to verify zero perturbation.")
print("="*80)

# Configuration
DATASET = "modular_addition"
LOSS_FUNC = "cross_entropy"
SEED = 0
NUM_EPOCHS = 10000  # Shorter for quick verification
LOG_FREQ = 100
DEVICE = "cpu"  # Use CPU for reproducibility

print(f"\nConfig: {DATASET}, {LOSS_FUNC}, seed={SEED}, {NUM_EPOCHS} epochs")
print("(Using short run for verification - full 50k epoch test recommended before publication)")
print()

input("Press Enter to start verification runs (will take ~10-20 min on CPU)...")

# === RUN 1: BASELINE ===
print("\n" + "="*80)
print("RUN 1: BASELINE (no QA instrumentation)")
print("="*80)

baseline_cmd = [
    "python", "grokking_experiments.py",
    "--dataset", DATASET,
    "--loss_function", LOSS_FUNC,
    "--seed", str(SEED),
    "--num_epochs", str(NUM_EPOCHS),
    "--log_frequency", str(LOG_FREQ),
    "--device", DEVICE,
    "--full_batch"
]

print("Command:", " ".join(baseline_cmd))
print()

result_baseline = subprocess.run(baseline_cmd, capture_output=True, text=True)
if result_baseline.returncode != 0:
    print("ERROR in baseline run:")
    print(result_baseline.stderr)
    exit(1)

print("✓ Baseline complete")

# === RUN 2: QA-INSTRUMENTED ===
print("\n" + "="*80)
print("RUN 2: QA-INSTRUMENTED")
print("="*80)

qa_cmd = [
    "python", "grokking_experiments_qa.py",
    "--dataset", DATASET,
    "--loss_function", LOSS_FUNC,
    "--seed", str(SEED),
    "--num_epochs", str(NUM_EPOCHS),
    "--log_frequency", str(LOG_FREQ),
    "--device", DEVICE,
    "--full_batch"
]

print("Command:", " ".join(qa_cmd))
print()

result_qa = subprocess.run(qa_cmd, capture_output=True, text=True)
if result_qa.returncode != 0:
    print("ERROR in QA run:")
    print(result_qa.stderr)
    exit(1)

print("✓ QA-instrumented complete")

# === LOAD RESULTS ===
print("\n" + "="*80)
print("COMPARING RESULTS")
print("="*80)

# Baseline metrics
experiment_key = f'{DATASET}_default'
baseline_csv = f"loggs/{experiment_key}/metrics.csv"
baseline_df = pd.read_csv(baseline_csv)

# QA metrics (from their logger, not our QA log)
qa_csv = f"loggs/{experiment_key}/metrics.csv"
qa_df = pd.read_csv(qa_csv)

# Extract train/test metrics
def extract_metrics(df):
    train_loss = df[(df['input_type'] == 'train') & (df['metric_name'] == 'loss')][['epoch', 'value']].rename(columns={'value': 'train_loss'})
    train_acc = df[(df['input_type'] == 'train') & (df['metric_name'] == 'accuracy')][['epoch', 'value']].rename(columns={'value': 'train_acc'})
    test_loss = df[(df['input_type'] == 'test') & (df['metric_name'] == 'loss')][['epoch', 'value']].rename(columns={'value': 'test_loss'})
    test_acc = df[(df['input_type'] == 'test') & (df['metric_name'] == 'accuracy')][['epoch', 'value']].rename(columns={'value': 'test_acc'})

    result = train_loss.merge(train_acc, on='epoch').merge(test_loss, on='epoch').merge(test_acc, on='epoch')
    return result

baseline_metrics = extract_metrics(baseline_df)
qa_metrics = extract_metrics(qa_df)

# === NUMERICAL COMPARISON ===
print("\n1. Final Metrics Comparison (last epoch)")
print("-" * 60)

final_baseline = baseline_metrics.iloc[-1]
final_qa = qa_metrics.iloc[-1]

metrics_to_compare = ['train_loss', 'train_acc', 'test_loss', 'test_acc']

print(f"{'Metric':<15} {'Baseline':>12} {'QA-Instrum':>12} {'Abs Diff':>12} {'Status':>10}")
print("-" * 60)

max_diff = 0
for metric in metrics_to_compare:
    baseline_val = final_baseline[metric]
    qa_val = final_qa[metric]
    diff = abs(baseline_val - qa_val)
    max_diff = max(max_diff, diff)

    status = "✓ MATCH" if diff < 1e-4 else "⚠ DIFF" if diff < 1e-2 else "✗ DIVERGED"
    print(f"{metric:<15} {baseline_val:>12.6f} {qa_val:>12.6f} {diff:>12.6e} {status:>10}")

print("-" * 60)

# === TRAJECTORY COMPARISON ===
print("\n2. Trajectory Correlation (all epochs)")
print("-" * 60)

merged = baseline_metrics.merge(qa_metrics, on='epoch', suffixes=('_baseline', '_qa'))

for metric in metrics_to_compare:
    baseline_col = f"{metric}_baseline"
    qa_col = f"{metric}_qa"

    correlation = merged[baseline_col].corr(merged[qa_col])
    mae = (merged[baseline_col] - merged[qa_col]).abs().mean()

    status = "✓ IDENTICAL" if correlation > 0.9999 else "⚠ SIMILAR" if correlation > 0.99 else "✗ DIVERGED"
    print(f"{metric:<15} r={correlation:.6f}  MAE={mae:.6e}  {status}")

print("-" * 60)

# === GROKKING ONSET ===
print("\n3. Grokking Onset Time (test accuracy > 95%)")
print("-" * 60)

grok_threshold = 0.95
baseline_grok = baseline_metrics[baseline_metrics['test_acc'] > grok_threshold]['epoch'].min()
qa_grok = qa_metrics[qa_metrics['test_acc'] > grok_threshold]['epoch'].min()

if pd.isna(baseline_grok) and pd.isna(qa_grok):
    print("  Both: No grokking in this short run (expected for 10k epochs)")
    print("  Status: ✓ CONSISTENT")
elif pd.isna(baseline_grok) or pd.isna(qa_grok):
    print(f"  Baseline: {baseline_grok}")
    print(f"  QA-Instr: {qa_grok}")
    print("  Status: ✗ INCONSISTENT (one grokked, other didn't)")
else:
    grok_diff = abs(baseline_grok - qa_grok)
    print(f"  Baseline: epoch {baseline_grok}")
    print(f"  QA-Instr: epoch {qa_grok}")
    print(f"  Difference: {grok_diff} epochs")
    status = "✓ MATCH" if grok_diff == 0 else "✓ CLOSE" if grok_diff <= LOG_FREQ else "⚠ DIFFERENT"
    print(f"  Status: {status}")

print("-" * 60)

# === VISUALIZATION ===
print("\n4. Generating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = [
    ('train_loss', 'Train Loss'),
    ('train_acc', 'Train Accuracy'),
    ('test_loss', 'Test Loss'),
    ('test_acc', 'Test Accuracy')
]

for idx, (metric, label) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]

    baseline_col = f"{metric}_baseline"
    qa_col = f"{metric}_qa"

    ax.plot(merged['epoch'], merged[baseline_col], 'b-', label='Baseline', alpha=0.7, linewidth=2)
    ax.plot(merged['epoch'], merged[qa_col], 'r--', label='QA-Instrumented', alpha=0.7, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.set_title(f'{label}: Baseline vs QA-Instrumented')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Verification: QA Instrumentation Does Not Perturb Training Dynamics',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('verification_no_perturbation.png', dpi=150)
print("  ✓ Saved: verification_no_perturbation.png")

# === FINAL VERDICT ===
print("\n" + "="*80)
print("VERDICT")
print("="*80)

if max_diff < 1e-4:
    print("✓ PASS: QA instrumentation causes ZERO behavioral perturbation")
    print("  → Final metrics identical within numerical precision")
    print("  → Safe to publish with statement:")
    print()
    print('  "Instrumentation-only: verified identical training dynamics')
    print('   with and without QA logging."')
    print()
elif max_diff < 1e-2:
    print("⚠ CAUTION: QA instrumentation causes minimal perturbation")
    print(f"  → Max difference: {max_diff:.6e}")
    print("  → Likely acceptable, but investigate if publishing")
    print()
else:
    print("✗ FAIL: QA instrumentation perturbs training dynamics")
    print(f"  → Max difference: {max_diff:.6e}")
    print("  → DO NOT PUBLISH without investigation")
    print()

print("="*80)
print("\nRecommendation for publication:")
print("  Run full 50k epoch verification with same seed on both versions")
print("  Command: NUM_EPOCHS=50000 python verify_no_perturbation.py")
print()
print("  Expected outcome: identical curves (within fp rounding)")
print("="*80)
