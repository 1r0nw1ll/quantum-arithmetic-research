# Phase 2 Paper - HI 2.0 Experimental Results Integration Plan

**Date**: December 10, 2025
**Status**: ⏳ Experiments running, paper update draft ready

---

## Experimental Validation Status

### Seismic Classification (Agent 1)
**Config**: Radial_family (w_ang=0.0, w_rad=0.6, w_fam=0.4)
**Task**: Earthquake vs Explosion discrimination
**Hypothesis**: Primitive (gcd=1) earthquakes vs Female/Composite (gcd≥2) explosions
**Status**: 🔄 Running

**Expected Results**:
- HI 2.0 accuracy > HI 1.0 by 5-15% (radial component provides gcd discrimination)
- Primitive earthquakes: H_radial ≈ 1.0
- Composite explosions: H_radial ≤ 0.5
- Separation metric: ~0.30 (from ablation study)

### EEG Seizure Detection (Agent 2)
**Config**: Angular_radial (w_ang=0.5, w_rad=0.5, w_fam=0.0)
**Task**: Pre-ictal / Ictal / Post-ictal state classification
**Hypothesis**: Pisano transitions + gcd patterns discriminate seizure states
**Status**: 🔄 Running

**Expected Results**:
- HI 2.0 F1-score > HI 1.0 by 3-10% (angular+radial captures temporal structure)
- Pre-ictal: Mixed angular patterns (transition)
- Ictal: Female patterns (gcd=2, octave harmonics)
- Post-ictal: Primitive patterns (gcd=1, recovery)

---

## Paper Sections to Update

### Section 5.1: Seismic Classification
**Current**: Results with HI 1.0 (E8-only)
**Add**: Comparison table and discussion

```markdown
#### 5.1.3 HI 2.0 Enhancement: Radial Harmonicity for Gcd Discrimination

The Radial_family configuration (w_ang=0.0, w_rad=0.6, w_fam=0.4)
emphasizes primitivity discrimination via the gcd criterion. This aligns
with the geological hypothesis that deep tectonic earthquakes generate
primitive QA tuples (gcd=1), while shallow cavity explosions produce
composite patterns (gcd≥2).

**Results**: [INSERT TABLE]

| Metric            | HI 1.0 (E8-only) | HI 2.0 (Radial_family) | Improvement |
|-------------------|------------------|------------------------|-------------|
| Accuracy          | XX.X%            | YY.Y%                  | +Z.Z%       |
| Precision (Eqk)   | XX.X%            | YY.Y%                  | +Z.Z%       |
| Recall (Eqk)      | XX.X%            | YY.Y%                  | +Z.Z%       |
| F1-Score          | XX.X%            | YY.Y%                  | +Z.Z%       |

**Discussion**: The radial harmonicity component (1/gcd) provides a
physically interpretable feature: primitive earthquakes map to E8 root
shell (H_radial=1.0), while composite explosions map to higher weight
shells (H_radial<0.5). This separation validates the gcd criterion as
a robust seismic discriminator.
```

### Section 5.2: EEG Seizure Detection
**Current**: Results with HI 1.0 (E8-only)
**Add**: Multi-class comparison and temporal analysis

```markdown
#### 5.2.3 HI 2.0 Enhancement: Angular Harmonicity for Temporal Transitions

The Angular_radial configuration (w_ang=0.5, w_rad=0.5, w_fam=0.0)
combines Pisano period structure (angular) with primitivity (radial)
to capture seizure state transitions. Pre-ictal states exhibit Pisano
transitions as brain networks reorganize before seizure onset.

**Results**: [INSERT TABLE]

| Metric            | HI 1.0 (E8-only) | HI 2.0 (Angular_radial) | Improvement |
|-------------------|------------------|-------------------------|-------------|
| Accuracy (3-class)| XX.X%            | YY.Y%                   | +Z.Z%       |
| F1 (Pre-ictal)    | XX.X%            | YY.Y%                   | +Z.Z%       |
| F1 (Ictal)        | XX.X%            | YY.Y%                   | +Z.Z%       |
| F1 (Post-ictal)   | XX.X%            | YY.Y%                   | +Z.Z%       |
| Macro F1          | XX.X%            | YY.Y%                   | +Z.Z%       |

**Discussion**: The angular component captures transitional geometry as
brain networks move through Pisano cycles. Ictal states show female
patterns (H_radial=0.5, gcd=2), consistent with octave harmonic hypothesis
for synchronized seizure activity.
```

### Section 6.2.3: HI 1.0 vs HI 2.0 Comparison (Already Complete)
**Status**: ✅ Theoretical framework in place
**Add**: Reference to experimental validation

**Update**:
```markdown
These theoretical predictions are validated experimentally in Sections
5.1.3 (seismic) and 5.2.3 (EEG), where domain-specific HI 2.0 configurations
achieve improvements of X-Y% over the HI 1.0 baseline.
```

---

## Figures to Add

### Figure: Seismic HI 2.0 Results
**Panel A**: Radial harmonicity distribution (earthquakes vs explosions)
**Panel B**: ROC curves (HI 1.0 vs HI 2.0)
**Panel C**: Confusion matrices (side-by-side comparison)
**Panel D**: H_radial vs accuracy scatter plot

### Figure: EEG HI 2.0 Results
**Panel A**: Angular+Radial trajectories for seizure states
**Panel B**: 3-class F1 scores (bar chart comparison)
**Panel C**: Temporal evolution of H_angular and H_radial
**Panel D**: State transition probabilities in component space

---

## Statistical Tests to Include

1. **Paired t-test**: HI 1.0 vs HI 2.0 accuracy (per-sample comparison)
2. **McNemar's test**: Binary classification improvement significance
3. **Cohen's kappa**: Inter-configuration agreement
4. **Effect size**: Cohen's d for improvement magnitude

---

## Integration Checklist

When experimental results arrive:

### 1. Results Processing
- [ ] Load seismic_hi2_0_results.json
- [ ] Load eeg_hi2_0_results.json
- [ ] Run hi2_0_comparison_template.py
- [ ] Generate LaTeX comparison table

### 2. Visualization
- [ ] Create seismic HI 2.0 figure (4-panel)
- [ ] Create EEG HI 2.0 figure (4-panel)
- [ ] Add to paper_figures/ directory

### 3. Paper Updates
- [ ] Insert comparison tables in Sections 5.1.3 and 5.2.3
- [ ] Update discussion with experimental findings
- [ ] Add figure references
- [ ] Update Section 6.2.3 with validation statement

### 4. Statistical Validation
- [ ] Run significance tests
- [ ] Report p-values and effect sizes
- [ ] Add statistical test descriptions to Methods section

### 5. Final Review
- [ ] Verify all numbers match JSON results
- [ ] Check figure quality (300 DPI)
- [ ] Proofread new sections
- [ ] Update abstract with HI 2.0 improvements

---

## Time Estimate

- Experiments (agents): 1-2 hours (in progress)
- Results processing: 15 minutes
- Visualization: 30 minutes
- Paper integration: 45 minutes
- **Total**: ~2.5-3 hours (on track for 2-3 hour estimate)

---

## Success Criteria

**Minimum**: HI 2.0 shows statistically significant improvement (p<0.05) in at least one domain
**Target**: HI 2.0 improves both seismic and EEG by 5%+ with p<0.01
**Stretch**: HI 2.0 improvements match or exceed ablation study predictions

---

**Status**: ⏳ Awaiting experimental results from parallel agents
**Next**: Process results → Generate figures → Update paper
