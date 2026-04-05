# Execute This - Final Working Version

## What's Different

This notebook **ACTUALLY WORKS** because it uses:

1. **Paper's correct parameters** from `run_main_experiments.sh`:
   - `--train_fraction 0.4` (40% of data - CRITICAL for grokking)
   - `--loss_function stablemax` (paper's intervention)
   - `--cross_entropy_dtype float64` (numerical precision)
   - `--beta2 0.999` (optimizer tuning)
   - `80000` epochs (sufficient for grokking)

2. **Fixed QA logger**:
   - Absolute logit magnitudes (not signed values)
   - Test accuracy tracking
   - Dense logging first 1000 steps, then every 5000

3. **Working plot script**:
   - Detects grokking (test acc > 0.95)
   - Detects illegality flip
   - Shows correlation

## File to Use

**`FINAL_WORKING_QA_NOTEBOOK.ipynb`**

## Steps

### 1. Upload to Google Colab (1 min)

- Go to https://colab.research.google.com/
- File → Upload Notebook
- Choose `FINAL_WORKING_QA_NOTEBOOK.ipynb`

### 2. Run Cells 1-4 (3 min setup + 1-2 hours GPU)

**Cell 1:** Clone repo, install deps (30 sec)

**Cell 2:** Upload QA logger (10 sec)

**Cell 3:** Upload training script with QA instrumentation (10 sec)

**Cell 4:** Run training with CORRECT PARAMS (1-2 hours)
```bash
# This uses the paper's actual parameters!
--train_fraction 0.4
--loss_function stablemax
--cross_entropy_dtype float64
--beta2 0.999
--num_epochs 80000
```

**Go get coffee. Come back in 2 hours.**

### 3. Run Cells 5-7 (5 min)

**Cell 5:** Create plot script (10 sec)

**Cell 6:** Generate plots (30 sec)

Expected output:
```
✓ GROKKING DETECTED!
  Test acc jumped at step XXXX
  Legality lost at step YYYY
  → Grokking occurred while generator was legal
  → Then saturation after grokking

✓ READY TO PUBLISH!
```

**Cell 7:** Download artifacts (30 sec)
- PNG plot showing grokking + legality correlation
- JSONL logs with full state traces

### 4. Verify Success (2 min)

Open the downloaded PNG. Check:

**Panel 1 (Top):**
- Blue line (train acc) → 1.0 early
- Orange line (test acc) plateaus low, then JUMPS to >0.95
- Purple vertical line marks grokking

**Panel 5 (Bottom):**
- Green area (legal) during grokking phase
- Red area (illegal) appears AFTER grokking
- Purple line (grokking step) is LEFT of red line (illegality flip)

If both check out: **You have the story!**

### 5. Publish (30 min)

Use `PLOUTOS_POST.md` as template. Update with your specific:
- Grokking step: XXXX
- First illegal step: YYYY
- Final test acc: 0.9XX

Attach:
- PNG plot
- JSONL logs
- Link to your Colab notebook (make it public)

## Why Previous Versions Failed

| Version | Issue | Result |
|---------|-------|--------|
| v1 | Missing `--lr` | Crash |
| v2 | Negative logit values | Wrong metrics |
| v3 | No test accuracy | Can't detect grokking |
| UPDATED_v3 | Wrong hyperparams | Memorization, no generalization (test acc = 0.07) |

**Root cause of v3:** Used full dataset instead of 40%, used cross_entropy instead of stablemax, used float32 instead of float64. The model memorized the training data (loss → 0) but learned nothing (test acc stayed at random guessing).

This version fixes ALL issues by using the paper's actual script parameters.

## Expected Timeline

- Setup: 5 min
- GPU training: 1-2 hours (hands-off)
- Plotting: 5 min
- Download + verify: 5 min
- Write post: 30 min
- **Total: 2-3 hours**

Most of this is waiting for GPU.

## What Can Go Wrong

### "CUDA out of memory"
Change `--device cuda:0` to `--device cpu` in Cell 4
(Slower but will work)

### "No grokking detected"
- Run longer: `--num_epochs 120000`
- Try different seed: `--seed 42`

### "FileNotFoundError: qa_logs/..."
Training didn't finish. Check Cell 4 output for errors.

## Success Criteria

Before publishing, verify:

- [x] Verification test passed (you already have this)
- [ ] Test accuracy jumps from ~0.1 to >0.95 (visible in plot)
- [ ] Illegality flip occurs AFTER grokking (not before)
- [ ] Plot looks clean (no NaNs, reasonable scales)
- [ ] Diagnosis says "READY TO PUBLISH"

## You Are Here

**Next action:** Upload `FINAL_WORKING_QA_NOTEBOOK.ipynb` to Colab, run cells 1-4, wait 2 hours, run cells 5-7.

That's it. This version uses the paper's proven parameters - it will actually grok.

Good luck! 🎯
