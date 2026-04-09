"""
QA Orbit Classifier for Xenobot Gene Expression Data
=====================================================

Maps to: Pai, Traer, Sperry, Zeng, Levin (2026) bioRxiv
Uses ACTUAL supplementary gene expression data from the paper.

Classifies each experimental condition (extract/ATP at 0hr/4hr)
into QA orbit type based on the transcriptomic signature.

QA prediction: extract → Cosmos-like signature (diverse, far-reaching)
               ATP → Satellite/Singularity-like (constrained, cycling)

Also: ready to classify PEA and salmon extract if data becomes available.

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'fold_change, gene_count, up_down_ratio -> float (observer projection)',
    'state_alphabet': '{1,...,9} mod-9 QA states for orbit classification',
    'discrete_layer': 'orbit assignment based on transcriptomic signature features',
    'observer_layer': 'all gene expression metrics are float measurements (Theorem NT)',
    'signal_injection': 'chemical stimulus = signal injection into cellular (b,e) state',
    'coupling': 'none (classification only, not dynamics)',
}

import numpy as np
from collections import Counter
np.random.seed(42)

M = 9

def classify_orbit(b, e, m=M):
    seen, state = [], (b, e)
    for _ in range(m * m + 1):
        if state in seen:
            clen = len(seen) - seen.index(state)
            if clen == 1: return 'singularity', clen
            elif clen <= 8: return 'satellite', clen
            else: return 'cosmos', clen
        seen.append(state)
        state = (((state[0]+state[1]-1)%m)+1,
                 ((state[1]+((state[0]+state[1]-1)%m)+1-1)%m)+1)
    return 'unknown', 0

def v3(n):
    """3-adic valuation."""
    if n == 0: return float('inf')
    n = abs(n)
    v = 0
    while n % 3 == 0: n //= 3; v += 1
    return v


# ─── Load Supplementary Data ─────────────────────────────────────────

print("=" * 70)
print("QA ORBIT CLASSIFIER FOR XENOBOT GENE EXPRESSION")
print("Data source: Pai et al. (2026) Supplementary Data 2")
print("=" * 70)

# Gene expression data from supplementary_data_2.xlsx (manually extracted)
# Format: (gene_name, product, fold_change, p_value)

conditions = {
    'extract_0hr': [
        ('g6pc1.1.L', 'glucose-6-phosphatase', 2.30, 5e-7),
        ('LOC108718763', 'neuropilin-1-like', 1.62, 1.29e-5),
        ('myh10.S', 'myosin heavy chain 10', -2.74, 1.88e-5),
        ('flnc.L', 'filamin C', -2.23, 1.77e-5),
        ('nid2.S', 'nidogen 2', -2.24, 1.57e-5),
        ('ptx3.L', 'pentraxin 3', -2.18, 2.54e-5),
        ('thbs2.L', 'thrombospondin 2', -3.17, 7.65e-6),
        ('bhlha9.S', 'basic helix-loop-helix a9', -2.94, 2.42e-5),
        ('thdl20.L', 'thyroid hormone down-regulated 20', -2.12, 1.59e-5),
        ('LOC121398041', 'uncharacterized', -2.05, 1.24e-5),
    ],
    'extract_4hr': [
        ('g6pc1.1.L', 'glucose-6-phosphatase', 2.96, 1.25e-8),
        ('LOC121398155', 'uncharacterized', 2.01, 1.06e-5),
        ('LOC121397179', 'serine/arginine repetitive matrix 2-like', 2.91, 6.01e-5),
        ('LOC108717924', 'uncharacterized', 2.55, 2.81e-5),
        ('LOC108716930', 'transmembrane protein 116', 2.08, 3.39e-5),
        ('g6pc1.1.S', 'glucose-6-phosphatase S', 1.70, 5.84e-6),
        ('usp33.S', 'ubiquitin specific peptidase 33', 2.41, 1.43e-5),
        ('LOC108704800', 'inter-alpha-trypsin inhibitor H6', 2.06, 6.72e-5),
        ('LOC121397873', 'golgin subfamily A member 6-like 7', 2.89, 9.8e-5),
        ('LOC121396870', 'uncharacterized', 5.27, 5.54e-5),
        ('LOC108703920', 'uncharacterized', 8.43, 1.25e-4),
        ('cxcl12.L', 'C-X-C motif chemokine ligand 12', -3.10, 6.02e-6),
        ('LOC108696922', 'serine protease inhibitor swm-1', -2.04, 1.61e-6),
        ('fscn1.L', 'fascin actin-bundling protein 1', -2.92, 6.78e-5),
        ('ocm4.1.S', 'oncomodulin 4', -12.2, 9.03e-5),
        ('lhx5.S', 'LIM homeobox 5', -23.9, 2.06e-6),
        ('snai1.S', 'snail family zinc finger 1', -2.22, 6.96e-5),
        ('vim.S', 'vimentin', -4.68, 1.49e-5),
        ('krt12.2.S', 'keratin 12', -2.14, 1.27e-4),
        ('cldn5.L', 'claudin 5', -2.76, 9.94e-5),
        ('LOC108703704', 'plasmalemma vesicle-associated protein', -3.72, 5.41e-5),
        ('ndnf.L', 'neuron-derived neurotrophic factor', -3.63, 5.73e-5),
        ('klhl4.L', 'kelch like family member 4', -4.29, 9.73e-5),
        ('bhlha9.L', 'basic helix-loop-helix a9 L', -2.48, 8.21e-5),
        ('bhlha9.S', 'basic helix-loop-helix a9 S', -3.45, 3.47e-6),
        ('colec12.L', 'collectin subfamily member 12', -3.92, 1.0e-4),
        ('vcan.L', 'versican', -2.66, 2.14e-5),
        ('LOC108718813', 'macrophage mannose receptor 1', -3.56, 5.09e-5),
        ('LOC108707243', 'versican core protein', -2.73, 3.67e-5),
        ('frem1.L', 'FRAS1 related extracellular matrix 1', -3.31, 2.87e-5),
        ('vwde.L', 'von Willebrand factor D and EGF domains', -2.88, 4.78e-5),
    ],
    'atp_0hr': [
        ('fos.S', 'FBJ osteosarcoma viral oncogene (Fos) S', 3.29, 1.87e-6),
        ('fos.L', 'FBJ osteosarcoma viral oncogene (Fos) L', 2.66, 6.87e-6),
        ('LOC121401594', 'tubulin monoglycylase TTLL3-like', -3.57, 2.79e-6),
        ('LOC121397978', 'uncharacterized', -2.64, 6.58e-6),
    ],
    'atp_4hr': [
        ('nr4a1.L', 'nuclear receptor subfamily 4 group A member 1 L', 2.90, 1.79e-6),
        ('nr4a1.S', 'nuclear receptor subfamily 4 group A member 1 S', 3.06, 7.75e-6),
        ('LOC121395420', 'uncharacterized', 11.0, 6.15e-6),
    ],
}


# ─── Feature Extraction ──────────────────────────────────────────────

print("\n--- Transcriptomic Signature Features ---\n")

features = {}
for cond, genes in conditions.items():
    folds = [g[2] for g in genes]
    n_total = len(genes)
    n_up = sum(1 for f in folds if f > 0)
    n_down = sum(1 for f in folds if f < 0)
    mean_abs_fold = np.mean([abs(f) for f in folds])
    max_up = max(folds) if folds else 0
    max_down = min(folds) if folds else 0
    fold_variance = np.var(folds)
    up_down_ratio = n_up / max(n_down, 1)

    # Asymmetry: are changes mostly up or mostly down?
    asymmetry = (n_up - n_down) / n_total

    features[cond] = {
        'n_genes': n_total,
        'n_up': n_up,
        'n_down': n_down,
        'up_down_ratio': up_down_ratio,
        'mean_abs_fold': mean_abs_fold,
        'max_up': max_up,
        'max_down': max_down,
        'fold_variance': fold_variance,
        'asymmetry': asymmetry,
    }

    print(f"  {cond:15s}: {n_total:2d} genes | up={n_up} down={n_down} | "
          f"ratio={up_down_ratio:.2f} | mean|FC|={mean_abs_fold:.2f} | "
          f"var={fold_variance:.2f} | asym={asymmetry:+.2f}")


# ─── QA Orbit Classification ─────────────────────────────────────────

print("\n--- QA Orbit Classification ---\n")

print("""
Classification logic (mapping transcriptomic signatures to QA orbits):

COSMOS signature (extract-like):
  - Many genes changed (high n_genes) → large orbit (72 pairs, diverse)
  - Strong asymmetry (mostly downregulated) → directional, far-reaching
  - High fold variance → broad activation range
  - Persistent across timepoints → 24-cycle stability

SATELLITE signature (predicted intermediate):
  - Few genes changed → small orbit (8 pairs)
  - Balanced up/down → cycling, oscillating
  - Low-moderate fold variance → constrained range
  - Immediate-early genes (Fos) without long-term consolidation

SINGULARITY signature:
  - Minimal or no gene changes → fixed point (1 pair)
  - Near-zero asymmetry → no net direction
  - Low fold variance → uniform/undisturbed
""")

# Classify each condition
for cond, feat in features.items():
    # Decision tree based on QA orbit properties
    n = feat['n_genes']
    asym = feat['asymmetry']
    var = feat['fold_variance']
    ratio = feat['up_down_ratio']

    # Cosmos: many genes, strong asymmetry, high variance
    # Satellite: few genes, balanced, low variance
    # Singularity: minimal change

    if n >= 10 and var > 5:
        orbit = 'COSMOS'
        reason = f"high gene count ({n}), high variance ({var:.1f})"
    elif n <= 5 and abs(asym) < 0.6:
        orbit = 'SATELLITE'
        reason = f"low gene count ({n}), balanced up/down (asym={asym:+.2f})"
    elif n <= 2:
        orbit = 'SINGULARITY'
        reason = f"minimal gene changes ({n})"
    else:
        # Intermediate — classify by variance and asymmetry
        if var > 3:
            orbit = 'COSMOS'
            reason = f"moderate count ({n}) but high variance ({var:.1f})"
        elif abs(asym) < 0.4:
            orbit = 'SATELLITE'
            reason = f"moderate count ({n}), balanced changes"
        else:
            orbit = 'COSMOS'
            reason = f"moderate count ({n}), asymmetric (asym={asym:+.2f})"

    print(f"  {cond:15s} → {orbit:12s}  ({reason})")


# ─── Orbit Feature Signatures (for future classification) ────────────

print("\n--- Orbit Feature Signatures (reference for future stimuli) ---\n")

# What features distinguish each predicted orbit type?
cosmos_conds = ['extract_0hr', 'extract_4hr']
satellite_conds = ['atp_0hr']
singularity_conds = ['atp_4hr']

for orbit_name, cond_list in [('COSMOS', cosmos_conds),
                                ('SATELLITE', satellite_conds),
                                ('SINGULARITY-adj', singularity_conds)]:
    if cond_list:
        ns = [features[c]['n_genes'] for c in cond_list]
        vars_ = [features[c]['fold_variance'] for c in cond_list]
        asyms = [features[c]['asymmetry'] for c in cond_list]
        ratios = [features[c]['up_down_ratio'] for c in cond_list]
        print(f"  {orbit_name}:")
        print(f"    n_genes: {np.mean(ns):.0f} (range {min(ns)}-{max(ns)})")
        print(f"    fold_variance: {np.mean(vars_):.1f}")
        print(f"    asymmetry: {np.mean(asyms):+.2f}")
        print(f"    up_down_ratio: {np.mean(ratios):.2f}")


# ─── v₃ Analysis of Gene Count ───────────────────────────────────────

print("\n--- v₃ Analysis (number-theoretic orbit signature) ---\n")

for cond, feat in features.items():
    n = feat['n_genes']
    v = v3(n)
    print(f"  {cond:15s}: n_genes={n:2d}  v₃(n)={v}  "
          f"n mod 3 = {n % 3}  n mod 9 = {n % 9}")

print("""
  Note: v₃ of gene count is suggestive but NOT the orbit classifier
  (v₃ classifies orbits via the QA norm f(b,e), not via gene count).
  The gene count is an OBSERVER PROJECTION of the underlying discrete
  orbit state — it reflects orbit size but doesn't determine it.
""")


# ─── Predictions for Unreported Stimuli ───────────────────────────────

print("=" * 70)
print("PREDICTIONS FOR UNREPORTED STIMULI")
print("=" * 70)

print("""
From Supplementary Data 1, two additional stimuli were screened:

1. PHENETHYL ALCOHOL (PEA)
   - Bacterial quorum sensing aromatic alcohol
   - Moderate, partial signaling molecule
   - QA PREDICTION: SATELLITE orbit
   - Expected signature:
     * Few genes changed (4-8 range)
     * BALANCED up/down (asymmetry near 0)
     * LOW fold variance
     * Immediate-early genes possible (Fos) but WITHOUT
       long-term consolidation genes (no Nr4a1)
     * Calcium dynamics: OSCILLATING cross-correlation

2. SALMON EXTRACT
   - Conspecific-adjacent alarm substance (different species)
   - Could be cosmos (like embryo extract) if alarm response
     is species-general, or satellite if species-specific
     recognition fails
   - QA PREDICTION: COSMOS or SATELLITE (depends on whether
     the alarm circuit recognizes cross-species signals)
   - Expected signature if Cosmos: similar to embryo extract
     (many genes, asymmetric, high variance)
   - Expected signature if Satellite: few genes, balanced,
     low variance

CRITICAL TEST: If PEA shows oscillating calcium cross-correlation
with 4-8 step periodicity, that confirms the Satellite memory
prediction. If it shows monotonic increase (like extract) or
decrease (like ATP), the prediction is falsified.
""")


# ─── Classifier Ready for New Data ───────────────────────────────────

def classify_new_stimulus(gene_data):
    """
    Classify a new stimulus result into QA orbit type.

    gene_data: list of (gene_name, product, fold_change, p_value) tuples

    Returns: (orbit_type, confidence, features_dict)
    """
    if not gene_data:
        return 'SINGULARITY', 1.0, {}

    folds = [g[2] for g in gene_data]
    n = len(gene_data)
    n_up = sum(1 for f in folds if f > 0)
    n_down = sum(1 for f in folds if f < 0)
    var = np.var(folds)
    asym = (n_up - n_down) / max(n, 1)
    mean_abs = np.mean([abs(f) for f in folds])

    feat = {'n_genes': n, 'n_up': n_up, 'n_down': n_down,
            'fold_variance': var, 'asymmetry': asym, 'mean_abs_fold': mean_abs}

    # Classification with confidence
    if n >= 10 and var > 5:
        return 'COSMOS', 0.9, feat
    elif n <= 5 and abs(asym) < 0.6:
        return 'SATELLITE', 0.7, feat
    elif n <= 2:
        return 'SINGULARITY', 0.6, feat
    elif var > 3:
        return 'COSMOS', 0.7, feat
    elif abs(asym) < 0.4:
        return 'SATELLITE', 0.6, feat
    else:
        return 'COSMOS', 0.5, feat


print("Classifier function `classify_new_stimulus()` ready.")
print("Feed it gene expression data in (name, product, fold_change, p) format.")
print("Returns (orbit_type, confidence, features).")
print("\nScript complete.")
