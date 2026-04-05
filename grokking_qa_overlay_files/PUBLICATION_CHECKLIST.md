# Publication Checklist for Ploutos

## Pre-Publication Verification (REQUIRED)

### 1. Behavioral Perturbation Test ✓ CRITICAL

**Run this first:**
```bash
python verify_no_perturbation.py
```

**Expected outcome:**
- Final metrics identical (within fp rounding)
- Loss trajectory correlation > 0.9999
- Verdict: "PASS: Zero behavioral perturbation"

**If verification fails:**
- DO NOT PUBLISH until resolved
- Check for bugs in qa_logger.py
- Ensure log_step is called AFTER optimizer.step()

**Status:** ⬜ Not run | ⬜ Running | ✓ PASSED | ✗ FAILED

---

### 2. Full 50k Epoch Run (Recommended)

The verification script uses 10k epochs by default (quick test). For publication credibility, run full 50k:

```bash
NUM_EPOCHS=50000 python verify_no_perturbation.py
```

This takes ~1 hour on GPU, ~4 hours on CPU. Worth it for bulletproof claim.

**Status:** ⬜ Not run | ⬜ Running | ✓ PASSED | ✗ FAILED

---

### 3. Generate Publication Plots

**Run both baseline and intervention:**
```bash
# Baseline (cross_entropy)
LOSS_FUNC=cross_entropy SEED=0 ./run_qa_experiment.sh

# Intervention (stablemax)
LOSS_FUNC=stablemax SEED=0 ./run_qa_experiment.sh
```

**Expected outputs:**
- `qa_analysis_modular_addition_cross_entropy_seed0.png`
- `qa_analysis_modular_addition_stablemax_seed0.png`
- `qa_nlm_alignment_*.png` (both variants)
- `qa_failures_*.png` (both variants)

**Key visual check:**
- Cross_entropy: legality flips to 0 early, stays illegal
- StableMax: legality window extended (stays legal longer)

**Status:** ⬜ Not generated | ✓ Generated & verified

---

## Publication Content (Use PLOUTOS_POST.md)

### Core Claims to Verify Match Your Results

1. **"Learning stops exactly when generators become illegal"**
   - Check: In your 4-panel plot, does test accuracy plateau coincide with legality flip?
   - ✓ Yes | ⬜ No | ⬜ Unclear

2. **"StableMax extends legality window"**
   - Check: Compare cross_entropy vs stablemax legality timelines side-by-side
   - ✓ Yes (stays legal longer) | ⬜ No | ⬜ Unclear

3. **"Zero behavioral perturbation"**
   - Check: verify_no_perturbation.py passed
   - ✓ Yes | ⬜ No

### Language Tone Check

Review PLOUTOS_POST.md for:
- ✓ Uses "reveals structure" not "proves"
- ✓ Uses "makes explicit" not "explains"
- ✓ Positions as "lens" / "complementary view"
- ✓ Avoids overclaiming
- ✓ States "instrumentation-only: verified identical dynamics"

### Artifacts to Attach

**Required:**
1. Main 4-panel plot (qa_analysis_*.png)
2. JSONL log excerpt (first + last record)
3. Link to code (GitHub gist or repo)

**Optional but strong:**
4. Side-by-side comparison (baseline vs stablemax legality)
5. Verification plot (verification_no_perturbation.png)
6. Full JSONL logs as "QA certificates"

---

## Pre-Flight Checklist

Before posting to Ploutos:

### Technical
- ⬜ Verification test passed (verify_no_perturbation.py)
- ⬜ Plots generated for both baseline and intervention
- ⬜ JSONL logs inspected (no NaNs, reasonable values)
- ⬜ Code tested end-to-end on fresh clone

### Content
- ⬜ PLOUTOS_POST.md reviewed for tone
- ⬜ Claims match your actual results
- ⬜ Figures have clear captions
- ⬜ Links updated (GitHub, paper, etc.)

### Strategic
- ⬜ Title is clear and non-hypey
- ⬜ Tags appropriate: #grokking #numerical-stability #reachability
- ⬜ Post invites engagement ("try it on your own experiments")
- ⬜ Positions QA as lens, not replacement

---

## Post-Publication Monitoring

### Expected Responses
- **Positive**: "Interesting lens", "Nice instrumentation", "Will try this"
- **Skeptical**: "Does it generalize?", "Why QA terminology?", "Correlation ≠ causation"
- **Hostile**: Unlikely if you follow tone guidelines above

### Prepared Responses

**Q: "Does this work on other datasets?"**
A: "This overlay is dataset-agnostic—it instruments training dynamics, not task-specific features. The modular addition example is from the paper; the same approach should work on any grokking setup. Would be interesting to see results on [dataset X]."

**Q: "Why call it QA / quantum arithmetic?"**
A: "QA (Quantum Arithmetic) is a discrete reachability framework for systems with modular structure. The name reflects the mathematical origin, but the core idea—tracking generator legality at boundaries—generalizes beyond arithmetic. Happy to discuss the formalism separately."

**Q: "Correlation doesn't mean causation"**
A: "Agreed—this is an observational tool, not a mechanistic explanation. The tight correlation (legality flip ⟷ learning halt) suggests structure worth investigating. The JSONL logs provide replayable evidence for further analysis."

**Q: "How is this different from just logging entropy?"**
A: "Entropy is one component. The overlay adds: (1) binary legality predicates that make boundaries explicit, (2) gradient-weight alignment (NLM proxy), (3) a discrete reachability framing. The JSONL format also makes it certificate-like."

---

## Success Metrics

**Minimum viable success:**
- 3-5 substantive comments/questions
- 1-2 people try the code
- No major technical criticisms

**Strong success:**
- Someone extends it to double descent / other phenomena
- Cited in follow-up work
- Becomes a template for "phase transition instrumentation"

**Ultimate success:**
- Recognized as a standard QA overlay pattern
- Authors of original paper engage
- Spawns "QA view of [X]" series

---

## Timeline

**Day 1-2:**
- Run verification tests
- Generate all plots
- Review content

**Day 3:**
- Publish to Ploutos
- Monitor first 24h of responses

**Week 1:**
- Engage with questions
- Refine based on feedback

**Week 2+:**
- Consider follow-up posts (e.g., "QA overlay template", "Double descent view")

---

## Emergency Rollback Plan

**If verification fails or major bug found:**

1. Delete Ploutos post immediately
2. Fix issue
3. Re-run verification
4. Re-publish with "Updated: fixed [issue]" note

**If overclaimed / tone wrong:**

1. Edit post to soften language (Ploutos allows edits)
2. Add comment: "Clarification: [corrected claim]"
3. Learn for next post

---

## Final Pre-Publication Command

```bash
# Run this sequence and verify all pass:
python test_qa_logger.py              # Should print "PASSED ✓"
python verify_no_perturbation.py      # Should print "PASS: Zero perturbation"
./run_qa_experiment.sh                # Generate plots
# Review plots visually
# Review PLOUTOS_POST.md one last time
# Post!
```

---

**Ready to publish?** Make sure you have:
1. ✓ Verification passed
2. ✓ Plots generated and inspected
3. ✓ Tone reviewed
4. ✓ Links updated
5. ✓ Claims match results

Then go ahead—this is solid work.
