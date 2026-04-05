# Colab Execution Checklist ✓

Copy-paste this into a Colab cell for reference.

---

## ☐ Step 1: Upload Fixed Logger (2 min)

Create cell, paste entire `qa_logger.py` from `OPTION_B_EXECUTION_GUIDE.md` Step 1

Run cell → should see "Writing qa_logger.py"

---

## ☐ Step 2: Upload Fixed Training Script (2 min)

Create cell, paste entire `grokking_experiments_qa_fixed.py` from Guide Step 2

Run cell → should see "Writing grokking_experiments_qa_fixed.py"

---

## ☐ Step 3: Run Training (50k epochs, GPU) (30-60 min)

Create cell:
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

Run cell → Go get coffee ☕

Expected output every 1000 epochs:
```
Epoch 1000: Loss 0.0002
Epoch 2000: Loss 0.0001
...
```

Final output:
```
[QA Logger] Closed: qa_logs/modular_addition_cross_entropy_seed0.jsonl
[QA Logger] Total failures: {'SOFTMAX_COLLAPSE': ..., ...}
Training complete!
```

---

## ☐ Step 4: Create Plotting Script (2 min)

Create cell, paste entire `plot_results.py` from Guide Step 4

Run cell → should see "Writing plot_results.py"

---

## ☐ Step 5: Generate Plots (1 min)

Create cell:
```bash
!python plot_results.py
```

Run cell → should show 5-panel plot

Expected diagnosis:
```
✓ GROKKING DETECTED!
  Test acc jumped at step XXXX
  Legality lost at step YYYY
✓ READY TO PUBLISH!
```

---

## ☐ Step 6: Download Artifacts (1 min)

Create cell:
```python
from google.colab import files
files.download('qa_grokking_modular_addition_cross_entropy_seed0.png')
files.download('qa_logs/modular_addition_cross_entropy_seed0.jsonl')
```

Run cell → Files download to your computer

---

## ☐ Step 7: Verify Results (2 min)

Open downloaded PNG:

**Check Panel 1 (Accuracy):**
- [ ] Train acc → 1 early (blue line)
- [ ] Test acc plateaus low, then JUMPS (orange line)
- [ ] Purple vertical line marks grokking

**Check Panel 5 (Legality):**
- [ ] Green (legal) during grokking phase
- [ ] Red (illegal) appears AFTER grokking
- [ ] Purple line (grokking) is LEFT of red line (legality flip)

**If ALL checks pass:** You have the grokking story! ✓

**If test acc never jumps:** Run longer or try different seed (see Troubleshooting)

---

## ☐ Step 8: Write Ploutos Post (15 min)

Use `PLOUTOS_POST.md` as template

**Update with your specific results:**
- Grokking step: XXXX
- First illegal step: YYYY
- Final train acc: Z.ZZ
- Final test acc: Z.ZZ

**Key claim to verify:**
"Grokking (test acc jump at step XXXX) occurred while SGD generator remained legal. Saturation (illegality at step YYYY) happened AFTER grokking, not before."

---

## ☐ Step 9: Publish (10 min)

1. Go to Ploutos
2. Create new post
3. Copy content from updated `PLOUTOS_POST.md`
4. Attach PNG plot
5. Add tags: `#grokking #numerical-stability #reachability`
6. Link to your public Colab notebook
7. Optional: Attach JSONL logs
8. Publish!

---

## Expected Timeline

| Step | Time | Blocking? |
|------|------|-----------|
| 1-2 | 5 min | Yes (you type) |
| 3 | 30-60 min | No (GPU runs) |
| 4-5 | 3 min | Yes |
| 6-9 | 30 min | Yes |
| **Total** | **1.5-2 hours** | |

---

## What Can Go Wrong

### Problem: "No grokking detected"

**Solutions:**
1. Run longer: `--num_epochs 100000`
2. Try different seed: `--seed 42`
3. Try StableMax: `--loss_function stablemax`

### Problem: "CUDA out of memory"

**Solution:**
```bash
!python grokking_experiments_qa_fixed.py ... --device cpu
```
(Slower but will work)

### Problem: "Test accuracy not in logs"

**Check:**
```python
import json
with open('qa_logs/modular_addition_cross_entropy_seed0.jsonl') as f:
    rec = json.loads(f.readline())
print('test_acc' in rec['state'])  # Should be True
```

If False, the training script didn't have test evaluation. Re-upload fixed script (Step 2).

---

## Success Criteria

Before publishing, verify:

- [x] Verification passed (you already have this)
- [ ] Grokking detected (test acc jump visible)
- [ ] Legality flip AFTER grokking (not before)
- [ ] Plot looks clean (no NaNs, reasonable scales)
- [ ] JSONL logs downloadable
- [ ] Diagnosis says "READY TO PUBLISH"

---

## You Are Here

Currently: Steps 1-2 (uploading fixed files)

Next: Step 3 (start training, wait ~1 hour)

Then: Steps 4-9 (plot, download, publish)

---

**Start now:** Open `OPTION_B_EXECUTION_GUIDE.md`, copy Cell 1, paste in Colab, run.

Good luck! 🎯
