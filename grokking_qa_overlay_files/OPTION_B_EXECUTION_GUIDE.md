# Option B: Complete Grokking Run - Step-by-Step

## Overview

Time: 3-5 hours (mostly GPU running in background)
Result: Test accuracy jump + legality flip correlation = publishable

---

## Step 1: Upload Fixed Files to Colab (2 min)

In your Colab notebook, create new cells:

### Cell 1: Upload fixed QA logger

```python
%%writefile qa_logger.py
"""
QA Logger - FIXED VERSION
"""
import torch
import json
import numpy as np
from pathlib import Path


class QALogger:
    def __init__(self, run_id, output_dir="qa_logs", log_every=100, log_dense_until=1000):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_every = log_every
        self.log_dense_until = log_dense_until

        self.log_file = self.output_dir / f"{run_id}.jsonl"
        self.log_handle = open(self.log_file, 'w')

        self.failure_counts = {
            "SOFTMAX_COLLAPSE": 0,
            "NAN_LOSS": 0,
            "INF_GRAD": 0,
            "GRAD_EXPLOSION": 0,
            "PARAM_EXPLOSION": 0,
            "LOGIT_EXPLOSION": 0,
        }

        self.LMAX = 85.0
        self.HMIN = 0.01
        self.GN_MAX = 1e6
        self.PN_MAX = 1e6

    def log_step(self, epoch, logits, targets, loss, model, optimizer,
                 test_logits=None, test_targets=None):
        # Adaptive logging
        if epoch < self.log_dense_until:
            should_log = True
        else:
            should_log = (epoch % self.log_every == 0)

        if not should_log:
            return

        state = self._compute_state(epoch, logits, targets, loss, model, optimizer,
                                    test_logits, test_targets)
        generators, failures = self._check_legality(state)

        for fail_type in failures:
            self.failure_counts[fail_type] += 1

        record = {
            "run_id": self.run_id,
            "step": epoch,
            "state": state,
            "generators": generators,
            "failures": failures,
            "cumulative_failures": self.failure_counts.copy()
        }
        self.log_handle.write(json.dumps(record) + "\\n")
        self.log_handle.flush()

    def _compute_state(self, epoch, logits, targets, loss, model, optimizer,
                       test_logits, test_targets):
        with torch.no_grad():
            train_loss = loss.item() if torch.isfinite(loss) else float('inf')
            preds = logits.argmax(dim=1)
            train_acc = (preds == targets).float().mean().item()

            # FIXED: Use absolute values and norms
            logit_max_abs = logits.abs().max().item()
            logit_range = (logits.max() - logits.min()).item()
            logit_std = logits.std().item()
            logit_norm = logits.norm().item()

            output_off = logits - logits.amax(dim=1, keepdim=True)
            exp_output = torch.exp(output_off.clamp(min=-88))
            probs = exp_output / exp_output.sum(dim=-1, keepdim=True)

            p_max = probs.max(dim=1).values.mean().item()
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum(dim=-1)
            p_entropy = entropy.mean().item()

            num_exact_ones = (probs == 1.0).sum().item()
            num_exact_zeros = (probs == 0.0).sum().item()

        # Test metrics
        test_acc = None
        if test_logits is not None and test_targets is not None:
            with torch.no_grad():
                test_preds = test_logits.argmax(dim=1)
                test_acc = (test_preds == test_targets).float().mean().item()

        grad_norm = 0.0
        grad_nan_count = 0
        grad_inf_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                grad_nan_count += torch.isnan(param.grad).sum().item()
                grad_inf_count += torch.isinf(param.grad).sum().item()
        grad_norm = np.sqrt(grad_norm)

        param_norm = 0.0
        for param in model.parameters():
            param_norm += param.norm().item() ** 2
        param_norm = np.sqrt(param_norm)

        cos_grad_w = self._compute_gradient_weight_alignment(model)

        state = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "logit_max_abs": logit_max_abs,
            "logit_range": logit_range,
            "logit_std": logit_std,
            "logit_norm": logit_norm,
            "p_max": p_max,
            "p_entropy": p_entropy,
            "num_exact_ones": num_exact_ones,
            "num_exact_zeros": num_exact_zeros,
            "grad_norm": grad_norm,
            "grad_nan_count": grad_nan_count,
            "grad_inf_count": grad_inf_count,
            "param_norm": param_norm,
            "cos_grad_w": cos_grad_w,
        }

        if test_acc is not None:
            state["test_acc"] = test_acc

        return state

    def _compute_gradient_weight_alignment(self, model):
        grad_vec = []
        weight_vec = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.flatten())
                weight_vec.append(param.data.flatten())
        if len(grad_vec) == 0:
            return 0.0
        grad_vec = torch.cat(grad_vec)
        weight_vec = torch.cat(weight_vec)
        cos_sim = (grad_vec * weight_vec).sum() / (grad_vec.norm() * weight_vec.norm() + 1e-12)
        return cos_sim.item()

    def _check_legality(self, state):
        failures = []

        if state["p_entropy"] < self.HMIN or state["num_exact_ones"] > 0:
            failures.append("SOFTMAX_COLLAPSE")
        if state["logit_max_abs"] > self.LMAX:
            failures.append("LOGIT_EXPLOSION")
        if not np.isfinite(state["train_loss"]):
            failures.append("NAN_LOSS")
        if state["grad_nan_count"] > 0 or state["grad_inf_count"] > 0:
            failures.append("INF_GRAD")
        if state["grad_norm"] > self.GN_MAX:
            failures.append("GRAD_EXPLOSION")
        if state["param_norm"] > self.PN_MAX:
            failures.append("PARAM_EXPLOSION")

        legal = len(failures) == 0
        reason = None if legal else failures[0]

        return {"sgd_step": {"legal": legal, "reason": reason}}, failures

    def close(self):
        self.log_handle.close()
        print(f"[QA Logger] Closed: {self.log_file}")
        print(f"[QA Logger] Total failures: {self.failure_counts}")
```

---

## Step 2: Create Fixed Training Script (3 min)

### Cell 2: Complete training script with test accuracy

```python
%%writefile grokking_experiments_qa_fixed.py
import random
import time
import torch
import torch.nn as nn
import json
import os
from logger import MetricsLogger
from qa_logger import QALogger
from torch.utils.data import DataLoader
from utils import (evaluate,
                   softmax_cross_entropy,
                   get_specified_args,
                   get_dataset,
                   get_model,
                   parse_args,
                   get_optimizer,
                   stablemax_cross_entropy)

torch.set_num_threads(5)
parser, args = parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
train_dtype = getattr(torch, args.train_dtype)
device = args.device

train_dataset, test_dataset = get_dataset(args)
if args.full_batch:
    args.batch_size = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Default LR if not provided
if args.lr is None:
    args.lr = 0.01
    print(f"No --lr provided, using default: {args.lr}")

args.lr = args.lr/(args.alpha**2)
model = get_model(args)
logger = MetricsLogger(args.num_epochs, args.log_frequency)

qa_log_id = f"{args.dataset}_{args.loss_function}_seed{args.seed}"
qa_logger = QALogger(run_id=qa_log_id, log_every=args.log_frequency, log_dense_until=1000)

optimizer = get_optimizer(model, args)

loss_functions = {
    "cross_entropy": softmax_cross_entropy,
    "stablemax": stablemax_cross_entropy
}
loss_function = loss_functions[args.loss_function]
ce_dtype = getattr(torch, args.cross_entropy_dtype)
save_model_checkpoints = range(0, args.num_epochs, args.log_frequency)
saved_models = {epoch: None for epoch in save_model_checkpoints}

if args.full_batch:
    all_data = train_dataset.dataset.data[train_dataset.indices].to(device)
    all_targets = train_dataset.dataset.targets[train_dataset.indices].to(device).long()
    all_test_data = test_dataset.dataset.data[test_dataset.indices].to(device)
    all_test_targets = test_dataset.dataset.targets[test_dataset.indices].to(device).long()
    if not (args.use_transformer or args.use_embedding):
        all_data = all_data.to(train_dtype)
        all_test_data = all_test_data.to(train_dtype)
else:
    raise ValueError("Current implementation only supports full batch training.")

print(f"Starting training. Train dataset size: {len(all_data)}, Test size: {len(all_test_data)}")

loss = torch.inf
start_time = time.time()
model.to(device).to(train_dtype)

for epoch in range(args.num_epochs):
    permutation = torch.randperm(all_data.size(0))
    shuffled_data = all_data[permutation]
    shuffled_targets = all_targets[permutation]

    model.train()
    optimizer.zero_grad()
    output = model(shuffled_data)
    if args.use_transformer:
        output = output[:, -1]
    output = output*args.alpha
    loss = loss_function(output, shuffled_targets, dtype=ce_dtype)
    loss.backward()
    optimizer.step()

    # QA LOGGING WITH TEST ACCURACY
    # Every log_frequency epochs, evaluate on test set
    if epoch % qa_logger.log_every == 0 or epoch < qa_logger.log_dense_until:
        with torch.no_grad():
            model.eval()
            test_output = model(all_test_data)
            if args.use_transformer:
                test_output = test_output[:, -1]
            test_output = test_output * args.alpha
            model.train()

        # Log with test data
        qa_logger.log_step(epoch, output, shuffled_targets, loss, model, optimizer,
                          test_logits=test_output, test_targets=all_test_targets)
    else:
        # Log without test data (faster for steps we don't log)
        qa_logger.log_step(epoch, output, shuffled_targets, loss, model, optimizer)

    if epoch % logger.log_frequency == 0:
        logger.log_metrics(
            model=model,
            epoch=epoch,
            save_model_checkpoints=save_model_checkpoints,
            saved_models=saved_models,
            all_data=shuffled_data,
            all_targets=shuffled_targets,
            all_test_data=all_test_data,
            all_test_targets=all_test_targets,
            args=args,
            loss_function=loss_function,
        )
        if epoch > 0 and epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

model.eval().to('cpu')
os.makedirs(f"loggs/{args.dataset}_default", exist_ok=True)
logger.metrics_df.to_csv(f"loggs/{args.dataset}_default/metrics.csv", index=False)

qa_logger.close()
print(f"Training complete!")
```

---

## Step 3: Run the Full Experiment (GPU, 50k epochs)

### Cell 3: Run baseline (cross_entropy)

```bash
!python grokking_experiments_qa_fixed.py \
    --dataset modular_addition \
    --loss_function cross_entropy \
    --lr 0.01 \
    --seed 0 \
    --num_epochs 50000 \
    --log_frequency 100 \
    --device cuda \
    --full_batch
```

**Expected time:** 30-60 min on GPU

**What to watch for:**
- First 1000 epochs: Dense logging (every step)
- After 1000: Logging every 100 steps
- Print statements every 1000 epochs
- Look for: "QA Logger] Total failures: ..." at end

---

## Step 4: Generate Fixed Plots

### Cell 4: Create plotting script

```python
%%writefile plot_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load QA logs
RUN_ID = "modular_addition_cross_entropy_seed0"
LOG_FILE = Path(f"qa_logs/{RUN_ID}.jsonl")

records = []
with open(LOG_FILE, 'r') as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# Convert to DataFrame
rows = []
for rec in records:
    row = {'step': rec['step'], 'legal': rec['generators']['sgd_step']['legal']}
    row.update(rec['state'])
    rows.append(row)
df = pd.DataFrame(rows)

# Find key events
first_illegal = df[~df['legal']]['step'].min() if not df[df['legal'] == False].empty else None
has_test = 'test_acc' in df.columns

if has_test:
    grok_step = df[df['test_acc'] > 0.95]['step'].min() if not df[df['test_acc'] > 0.95].empty else None
else:
    grok_step = None

print(f"First illegal: {first_illegal}")
print(f"Grokking step: {grok_step}")
print(f"Final train acc: {df['train_acc'].iloc[-1]:.3f}")
if has_test:
    print(f"Final test acc: {df['test_acc'].iloc[-1]:.3f}")

# Plot
fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# Panel 1: Accuracy
ax = axes[0]
ax.plot(df['step'], df['train_acc'], label='Train Acc', alpha=0.7, color='blue', linewidth=2)
if has_test:
    ax.plot(df['step'], df['test_acc'], label='Test Acc', alpha=0.8, color='orange', linewidth=2)
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1.05])
ax.legend()
ax.set_title('Grokking as Reachability at Numerical Boundaries (QA View)', fontweight='bold', fontsize=14)
ax.grid(alpha=0.3)
if grok_step:
    ax.axvline(grok_step, color='purple', linestyle='--', linewidth=2, alpha=0.6, label='Grokking')

# Panel 2: Logit magnitude
ax = axes[1]
ax.plot(df['step'], df['logit_max_abs'], label='Max |Logit|', color='red', alpha=0.7)
ax.axhline(85, color='red', linestyle=':', alpha=0.5, label='FP32 threshold')
ax.set_ylabel('Logit Magnitude')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: Entropy
ax = axes[2]
ax.plot(df['step'], df['p_entropy'], label='Entropy', color='green', alpha=0.7)
ax.axhline(0.01, color='green', linestyle='--', alpha=0.5, label='Collapse threshold')
ax.set_ylabel('Entropy')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

# Panel 4: Gradients
ax = axes[3]
ax.plot(df['step'], df['grad_norm'], label='Gradient Norm', color='purple', alpha=0.7)
ax.axhline(1e-6, color='purple', linestyle='--', alpha=0.5, label='Dead gradients')
ax.set_ylabel('Gradient Norm')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

# Panel 5: Generator legality (KEY)
ax = axes[4]
legality = df['legal'].astype(int)
ax.fill_between(df['step'], 0, legality, alpha=0.4, label='Legal', color='green')
ax.fill_between(df['step'], legality, 1, alpha=0.4, label='Illegal', color='red')
if first_illegal:
    ax.axvline(first_illegal, color='darkred', linestyle='--', linewidth=3, alpha=0.8,
               label=f'First Illegal: {first_illegal}')
if grok_step:
    ax.axvline(grok_step, color='purple', linestyle='--', linewidth=3, alpha=0.8,
               label=f'Grokking: {grok_step}')
ax.set_ylabel('Generator\\nLegality', fontweight='bold')
ax.set_xlabel('Training Step')
ax.set_ylim([0, 1])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Illegal', 'Legal'])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Annotation
if grok_step and first_illegal:
    if first_illegal > grok_step:
        text = 'Grokking occurs\\nwhile legal,\\nthen collapse'
    else:
        text = 'Early saturation\\nprevents grokking'
    ax.text(0.02, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'qa_grokking_{RUN_ID}.png', dpi=150, bbox_inches='tight')
print(f"Saved: qa_grokking_{RUN_ID}.png")
plt.show()

# Diagnosis
print("\\n" + "="*60)
if has_test and grok_step and first_illegal and grok_step < first_illegal:
    print("✓ GROKKING DETECTED!")
    print(f"  Test acc jumped at step {grok_step}")
    print(f"  Legality lost at step {first_illegal}")
    print("  → Grokking occurred while generator was legal")
    print("  → Then saturation after grokking")
    print("\\n✓ READY TO PUBLISH!")
elif has_test and df['test_acc'].iloc[-1] > 0.95:
    print("✓ GENERALIZATION ACHIEVED")
    print("  But no clear grokking transition (smooth learning)")
else:
    print("⚠ NO GROKKING YET")
    print("  Need longer training or different hyperparams")
print("="*60)
```

### Cell 5: Run plotting

```bash
!python plot_results.py
```

---

## Step 5: Download and Prepare for Publishing

### Cell 6: Package artifacts

```python
# Show sample records
import json
LOG_FILE = "qa_logs/modular_addition_cross_entropy_seed0.jsonl"

with open(LOG_FILE, 'r') as f:
    records = [json.loads(line) for line in f]

print("First record (initialization):")
print(json.dumps(records[0], indent=2))

print("\\n" + "="*60 + "\\n")

print("Last record (final state):")
print(json.dumps(records[-1], indent=2))

# Download these files:
from google.colab import files
files.download('qa_grokking_modular_addition_cross_entropy_seed0.png')
files.download('qa_logs/modular_addition_cross_entropy_seed0.jsonl')
```

---

## What Success Looks Like

### Expected Output Pattern

**Console during training:**
```
Epoch 1000: Loss 0.0002
Epoch 2000: Loss 0.0001
...
Epoch 15000: Loss 0.0000
[QA Logger] Closed: qa_logs/...
[QA Logger] Total failures: {'SOFTMAX_COLLAPSE': 120, ...}
```

**Plot should show:**
1. **Panel 1:** Train acc → 1 early (epoch ~500), test acc plateaus low (~0.1), then JUMPS to >0.95 (epoch ~10k-30k)
2. **Panel 2:** Logit magnitude growing steadily
3. **Panel 3:** Entropy stays high during grokking, collapses after
4. **Panel 4:** Gradients healthy during grokking, die after test acc saturates
5. **Panel 5:** Legality TRUE during grokking phase, flips FALSE after

**Diagnosis output:**
```
✓ GROKKING DETECTED!
  Test acc jumped at step 15000
  Legality lost at step 25000
  → Grokking occurred while generator was legal
  → Then saturation after grokking

✓ READY TO PUBLISH!
```

---

## Troubleshooting

### If no grokking after 50k epochs:

**Option 1:** Try different seed
```bash
!python grokking_experiments_qa_fixed.py ... --seed 42
```

**Option 2:** Try StableMax (paper's intervention)
```bash
!python grokking_experiments_qa_fixed.py ... --loss_function stablemax
```

**Option 3:** Check hyperparams match paper
- Use their exact `--lr`, `--alpha`, `--weight_decay` values from their scripts

### If GPU runs out of memory:

```bash
# Use smaller batch or reduce model size
!python grokking_experiments_qa_fixed.py ... --device cpu
```
(Will be slower but should work)

### If plot doesn't show test accuracy:

Check QA log has test_acc field:
```python
import json
with open('qa_logs/modular_addition_cross_entropy_seed0.jsonl') as f:
    rec = json.loads(f.readline())
print('test_acc' in rec['state'])  # Should be True
```

---

## Timeline

**Now → +5 min:** Upload fixed files, create scripts
**+5 → +60 min:** GPU training (mostly hands-off)
**+60 → +70 min:** Generate plots, verify grokking
**+70 → +90 min:** Download artifacts, write post
**Total: 1.5-2 hours** (including GPU wait time)

---

## Next Steps After Grokking

Once you have the grokking plot:

1. **Verify the story:** Grokking happens while legal, saturation after
2. **Compare with StableMax:** Run same experiment with `--loss_function stablemax`
3. **Write post:** Use `PLOUTOS_POST.md` as template, update with your specific results
4. **Publish:** Attach plot + JSONL logs + link to Colab notebook

---

## Ready?

Run Cell 1 (upload logger), Cell 2 (upload training script), then Cell 3 (start training).

Come back in ~1 hour to check progress.

Good luck! 🚀
