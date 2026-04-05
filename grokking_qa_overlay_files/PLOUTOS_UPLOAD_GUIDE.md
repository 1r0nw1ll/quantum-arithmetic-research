# Running QA Overlay on Ploutos (Quick Guide)

## Upload & Run in 3 Steps

### Step 1: Upload Notebook to Ploutos

1. Go to Ploutos
2. Click "New Notebook" or similar
3. Upload `ploutos_qa_overlay_demo.ipynb`

**OR**

Create new notebook and copy-paste the content from `ploutos_qa_overlay_demo.ipynb`

---

### Step 2: Run All Cells

Click "Run All" or execute cells sequentially.

**Expected runtime:**
- CPU: ~20-30 minutes total
- GPU: ~5-10 minutes total

**What it does:**
1. Clones original paper repo
2. Installs dependencies
3. Runs quick verification (1000 epochs)
4. Runs full experiment (10k epochs)
5. Generates plots
6. Exports artifacts

---

### Step 3: Download & Publish

After notebook completes:

**Download these files:**
- `qa_analysis_modular_addition_cross_entropy_seed0.png` (main plot)
- `qa_logs/modular_addition_cross_entropy_seed0.jsonl` (optional, for verification)

**Then create Ploutos post:**
1. Copy content from `PLOUTOS_POST.md`
2. Attach the PNG plot
3. Add tags: `#grokking #numerical-stability #reachability`
4. Link to your notebook (make it public on Ploutos)
5. Publish!

---

## Troubleshooting

### "Out of memory"
- Ploutos compute might be resource-limited
- Reduce `FULL_EPOCHS` from 10000 to 5000
- Use CPU instead of GPU (set `DEVICE = 'cpu'`)

### "Dependencies not found"
- Ploutos should have PyTorch pre-installed
- If not, the notebook installs it in cell 2
- Wait for installation to complete before proceeding

### "Plots look wrong"
- Check if verification passed (cell output should say "PASS")
- Inspect `first_illegal` value - should be a reasonable epoch number
- Try running with different seed

---

## Expected Output

**Verification cell should show:**
```
VERIFICATION RESULTS
Max difference: < 1e-6
Correlation: > 0.9999
✓ PASS: Zero behavioral perturbation
```

**Final plot should show:**
- Panel A: Loss decreasing, accuracy increasing
- Panel B: Logits approaching threshold (~85)
- Panel C: Entropy collapsing toward 0
- Panel D: **Legality flipping from 1 (green) to 0 (red)**

**Key observation:**
Panel D flip should coincide with Panel A accuracy plateau.

---

## Alternative: Run Locally with PyTorch

If Ploutos doesn't work, run locally:

```bash
# On machine with PyTorch
cd grokking_qa_overlay
jupyter notebook ploutos_qa_overlay_demo.ipynb

# Then upload results to Ploutos as a post
```

---

## What to Post

**Title:**
"Grokking as Reachability at Numerical Boundaries (QA View)"

**Content:**
Copy from `PLOUTOS_POST.md`

**Attachments:**
1. Main plot (required)
2. Notebook link (required - make public)
3. JSONL logs (optional)
4. GitHub repo link (when ready)

**Tags:**
`#grokking #numerical-stability #discrete-mathematics #reachability #phase-transitions`

---

## Timeline

**On Ploutos:**
- Upload notebook: 2 min
- Run all cells: 10-30 min (automatic)
- Download artifacts: 1 min
- Create post: 15 min
- **Total: ~30-50 min**

**Immediate next step:** Upload `ploutos_qa_overlay_demo.ipynb` to Ploutos and click "Run All"
