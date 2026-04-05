# Colab Run - What to Fix

## What You Have Now ✓

- ✅ Verification passed (zero perturbation)
- ✅ QA logger working
- ✅ Plot generated
- ✅ JSONL trace captured

## What's Wrong (And How to Fix)

### Issue 1: Not Actually Grokking

**Problem:** Your run shows memorization + saturation, not grokking.

**Evidence:**
- Train acc → 1.0 very early
- No test accuracy tracked
- Softmax collapsed at step 200
- Gradients dead (norm ~1e-7)

**What grokking looks like:**
```
Steps 0-1000:    Train acc → 1.0, Test acc ~0.1  (memorize)
Steps 1000-5000: Train acc = 1.0, Test acc ~0.1  (plateau)
Steps 5000-10k:  Train acc = 1.0, Test acc → 0.95 (GROKKING!)
```

**You're stuck in phase 1** (memorize + saturate).

---

### Issue 2: Missing Test Accuracy

**Problem:** You can't claim "grokking" without showing test accuracy jump.

**Fix:** Modify the QA logger call to include test data:

```python
# In training loop, after optimizer.step():

# Evaluate on test set periodically
if epoch % logger.log_frequency == 0:
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
    # Log without test data (faster)
    qa_logger.log_step(epoch, output, shuffled_targets, loss, model, optimizer)
```

---

### Issue 3: Logit Plot Misleading

**Problem:** Showing `logit_max = -90` as if it's approaching +88 overflow.

**Fix:** Use the fixed logger (`qa_logger_fixed.py`) which tracks:
- `logit_max_abs` - absolute magnitude (correct)
- `logit_norm` - Frobenius norm (shows true explosion)
- `logit_range` - max - min (another useful metric)

Replace your current `qa_logger.py` with `qa_logger_fixed.py`.

---

### Issue 4: Coarse Logging

**Problem:** Only 100 samples over 10k steps = every 100 steps.

**Result:** "First illegal step: 200" is very coarse.

**Fix:** Use adaptive logging (already in fixed logger):
```python
qa_logger = QALogger(run_id="...", log_every=100, log_dense_until=1000)
```

This logs **every step** for first 1000 epochs, then every 100 after.

---

## What to Do Next (Two Options)

### Option A: Post the Honest "Saturation" Story

**Frame it as:**
> "QA overlay reveals early numerical saturation in modular arithmetic training. Softmax collapses to one-hot at step ~200, gradients drop to ~1e-7, and learning effectively stops. This is a certificate-style trace of boundary-induced training paralysis. Next: reproduce full grokking regime (longer training + test accuracy tracking) and show how stabilization extends legality window."

**Pros:**
- Honest, defensible
- Still demonstrates the QA overlay concept
- Shows verification works
- Invites collaboration ("help me run longer!")

**Cons:**
- Not the sexy "grokking" story
- Less impactful

---

### Option B: Run Until You Get Real Grokking

**What you need:**

1. **Add test accuracy logging** (see Issue 2 fix above)

2. **Run longer** - Grokking on modular addition typically requires:
   - CPU: 50k-200k epochs (hours)
   - GPU: 10k-50k epochs (minutes-hours)

3. **Use paper's exact hyperparams** (from their script):
   ```bash
   !python grokking_experiments_qa.py \
       --dataset modular_addition \
       --loss_function cross_entropy \
       --lr 0.01 \
       --alpha 1.0 \
       --seed 0 \
       --num_epochs 50000 \
       --log_frequency 100 \
       --device cuda \
       --full_batch
   ```

4. **Compare baseline vs StableMax**:
   ```bash
   # Baseline
   !python grokking_experiments_qa.py --loss_function cross_entropy ...

   # Intervention
   !python grokking_experiments_qa.py --loss_function stablemax ...
   ```

**Expected outcome:**
- Baseline: Legality lost early → no grokking
- StableMax: Legality window extended → grokking appears

**That's the money shot.**

---

## Immediate Next Steps (Recommended)

### 1. Replace the logger (5 min)

```bash
# In Colab
!cp qa_logger_fixed.py qa_logger.py
```

### 2. Add test accuracy tracking (10 min)

Edit `grokking_experiments_qa.py` training loop (see Issue 2 fix above).

### 3. Run one of these:

**Quick test (1-2 hours, CPU):**
```bash
!python grokking_experiments_qa.py \
    --dataset modular_addition \
    --loss_function cross_entropy \
    --lr 0.01 \
    --seed 0 \
    --num_epochs 20000 \
    --log_frequency 100 \
    --device cpu \
    --full_batch
```

**Proper run (GPU, if available):**
```bash
!python grokking_experiments_qa.py \
    --dataset modular_addition \
    --loss_function cross_entropy \
    --lr 0.01 \
    --seed 0 \
    --num_epochs 50000 \
    --log_frequency 100 \
    --device cuda \
    --full_batch
```

### 4. Use the fixed plotting script

```bash
!python qa_plot_fixed.py
```

This will:
- Show test accuracy (if logged)
- Use correct logit metrics
- Diagnose whether you got grokking or saturation
- Give you a recommendation

---

## Expected Outcome (After Fixes)

**If you run long enough with test accuracy:**

Your plot will show:
- Panel 1: Train acc → 1 early, test acc stays low, then JUMPS
- Panel 2: Logit magnitude growing (correctly plotted)
- Panel 3: Entropy stable initially, then collapses
- Panel 4: Gradients stay healthy during grokking, die after
- Panel 5: **Legality flips AFTER test acc jumps** (the key claim)

**That's publishable.**

---

## Summary

**Current state:** Working instrumentation, but wrong narrative (saturation ≠ grokking)

**Fixes needed:**
1. Replace logger → `qa_logger_fixed.py`
2. Add test accuracy tracking
3. Run longer (20k-50k epochs)
4. Use fixed plotting script

**Time required:** 2-3 hours total (mostly waiting for training)

**Payoff:** Actual grokking + legality flip correlation → strong Ploutos post

---

## Questions?

**Q: Can I just post the saturation story now?**
A: Yes, if you frame it honestly (Option A above). It's still valuable.

**Q: Do I need GPU?**
A: No, but it's 5-10x faster. CPU works, just takes longer.

**Q: What if I still don't see grokking after 50k epochs?**
A: Try StableMax or check hyperparams match the paper exactly. Or post as "saturation study" (still valid).

**Q: Should I wait for perfect results before posting?**
A: No. Post the verification + saturation story, then follow up with grokking results later. Iterative publishing is fine.

---

**Next command:** Replace the logger and rerun with test accuracy:

```bash
# In Colab
!wget https://your-repo/qa_logger_fixed.py -O qa_logger.py
# Or just copy-paste the content
```
