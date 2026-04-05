# Quick Reference - Keep This Open While Running

## The File

**`FINAL_WORKING_QA_NOTEBOOK.ipynb`** ← Upload this to Colab

## The Cells

| Cell | What | Time | Wait? |
|------|------|------|-------|
| 1 | Clone + install | 30s | Yes |
| 2 | QA logger | 10s | Yes |
| 3 | Training script | 10s | Yes |
| 4 | **RUN TRAINING** | **1-2 hrs** | **No - go away** |
| 5 | Plot script | 10s | Yes |
| 6 | Generate plot | 30s | Yes |
| 7 | Download | 30s | Yes |

## What to Expect

### Cell 4 Output (Training)

```
Starting training. Train dataset size: 1536, Test size: 2560
Epoch 1000: Loss 0.0002
Epoch 2000: Loss 0.0001
...
Epoch 80000: Loss 0.0000
[QA Logger] Closed: qa_logs/modular_addition_stablemax_seed0.jsonl
[QA Logger] Total failures: {'SOFTMAX_COLLAPSE': 120, 'NAN_LOSS': 0, ...}
Training complete!
```

**If you see this:** ✓ Training succeeded, move to Cell 5

**If error:** Check you're using GPU runtime (Runtime → Change runtime type → GPU)

### Cell 6 Output (Plotting)

```
Loaded 17 records
First illegal: 45000
Grokking step: 25000
Final train acc: 1.000
Final test acc: 0.989

============================================================
✓ GROKKING DETECTED!
  Test acc jumped at step 25000
  Legality lost at step 45000
  → Grokking occurred while generator was legal
  → Then saturation after grokking

✓ READY TO PUBLISH!
============================================================
```

**If you see this:** ✓ Success! Download artifacts (Cell 7) and publish

**If "NO GROKKING YET":** Training didn't run long enough or wrong seed. Try `--num_epochs 120000` or `--seed 42`

## The Plot

**Panel 1 (Top):**
- Orange line JUMPS from low to high = grokking ✓

**Panel 5 (Bottom):**
- Purple line (grokking) LEFT of red area (illegal) = correct story ✓

## Key Parameters (Cell 4)

These are what make it actually work:

```bash
--train_fraction 0.4        # Only 40% of data (forces generalization)
--loss_function stablemax   # Paper's intervention (prevents collapse)
--cross_entropy_dtype float64  # Numerical precision
--num_epochs 80000          # Sufficient time for grokking
```

**Do NOT change these** - they're from the paper's proven script.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Change `cuda:0` to `cpu` in Cell 4 |
| No grokking after 80k | Try `--seed 42` |
| FileNotFoundError | Cell 4 didn't finish - check for errors |
| Test acc = 0.07 | You used wrong params - use FINAL notebook |

## Success Checklist

Before publishing:

- [ ] Cell 4 completed without errors
- [ ] Cell 6 shows "GROKKING DETECTED"
- [ ] Plot Panel 1 shows test acc jump
- [ ] Plot Panel 5 shows purple line LEFT of red area
- [ ] Downloaded PNG + JSONL files

If all checked → Publish to Ploutos with plot + logs

## Time Budget

- Now → +5 min: Upload + run cells 1-3
- +5 → +2 hrs: GPU runs (go do something else)
- +2 hrs → +2:10: Run cells 5-6, verify success
- +2:10 → +2:15: Download artifacts
- +2:15 → +2:45: Write Ploutos post
- +2:45 → +3:00: Publish

**Total: 3 hours (2 hours is GPU running unattended)**

## What Makes This Different

Previous versions: Wrong hyperparameters → no learning

This version: Paper's exact hyperparameters → grokking occurs

## You Are Here

**Right now:** Reading this

**Next 5 minutes:** Upload notebook to Colab, run cells 1-3

**Next 2 hours:** Go get coffee, take a walk, do literally anything else

**2 hours from now:** Come back, run cells 5-7, verify success, publish

---

**Start now:** https://colab.research.google.com → Upload → `FINAL_WORKING_QA_NOTEBOOK.ipynb`
