"""
QA Representational Geometry Analysis
======================================

Connection to: Kirsanov & Chung (NAACL 2025) "The Geometry of Prompting"

Question: Does QA tuple algebra produce distinct representational geometry
WITHOUT any optimization? If the orbit structure creates clean geometric
clusters under standard metrics, this is evidence that discrete modular
arithmetic is a source of representational structure.

Metrics (matching standard representational geometry literature):
  1. Effective dimensionality (participation ratio of eigenvalues)
  2. Inter-class distance (centroid separation between orbit types)
  3. Intra-class variance (spread within each orbit type)
  4. Fisher discriminant ratio (inter/intra — higher = cleaner clusters)
  5. Cosine similarity structure (within vs between orbits)
  6. E8 alignment by orbit type (do orbits align differently?)
  7. Singular value spectrum (rank structure of tuple matrix)

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'dimensionality, distances, cosine_sim, SVD -> float (observer projections)',
    'state_alphabet': '{1,...,9} and {1,...,24} (mod-9 and mod-24)',
    'discrete_layer': '(b,e,d,a) integer tuples; orbit classification by cycle length',
    'observer_layer': 'all geometric metrics are float measurements (Theorem NT)',
    'signal_injection': 'none (static algebraic analysis)',
    'coupling': 'resonance = tuple inner product (analyzed, not used for dynamics)',
}

import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)


# ─── QA Core ──────────────────────────────────────────────────────────

def qa_tuple(b, e, m):
    """A2-compliant: d and a derived from b+e, b+2e."""
    d = ((b + e - 1) % m) + 1
    a = ((b + 2*e - 1) % m) + 1
    return [b, e, d, a]

def classify_orbit(b, e, m):
    """Classify by cycle length."""
    seen, state = [], (b, e)
    for _ in range(m * m + 1):
        if state in seen:
            clen = len(seen) - seen.index(state)
            if clen == 1: return 'singularity'
            elif clen <= 8: return 'satellite'
            else: return 'cosmos'
        seen.append(state)
        state = (((state[0]+state[1]-1)%m)+1,
                 ((state[1]+((state[0]+state[1]-1)%m)+1-1)%m)+1)
    return 'unknown'

def resonance(t1, t2):
    """Tuple inner product (QA-native attention score)."""
    return sum(a*b for a, b in zip(t1, t2))


# ─── Generate All Tuples ─────────────────────────────────────────────

def generate_all_tuples(m):
    """Generate all (b,e) pairs and their 4D tuples + orbit labels."""
    tuples = []
    labels = []
    pairs = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            t = qa_tuple(b, e, m)
            tuples.append(t)
            labels.append(classify_orbit(b, e, m))
            pairs.append((b, e))
    return np.array(tuples, dtype=float), labels, pairs


# ─── Representational Geometry Metrics ────────────────────────────────

def participation_ratio(matrix):
    """
    Effective dimensionality via participation ratio of singular values.
    PR = (sum(s_i))^2 / sum(s_i^2). Ranges from 1 (rank-1) to min(n,p) (uniform).
    Standard metric in representational geometry (Abbott et al. 2011).
    """
    # S1 compliant: s*s not s**2
    s = np.linalg.svd(matrix - np.mean(matrix, axis=0), compute_uv=False)
    s_sq = s * s
    if np.sum(s_sq) == 0:
        return 0
    pr = (np.sum(s) * np.sum(s)) / np.sum(s_sq)
    return pr

def fisher_discriminant(tuples, labels, class_names):
    """
    Fisher discriminant ratio: inter-class variance / intra-class variance.
    Higher = more separable clusters. Classic metric for representational quality.
    """
    class_means = {}
    class_vars = {}
    global_mean = np.mean(tuples, axis=0)

    for cls in class_names:
        mask = [i for i, l in enumerate(labels) if l == cls]
        if not mask:
            continue
        pts = tuples[mask]
        class_means[cls] = np.mean(pts, axis=0)
        # Intra-class variance: average squared distance to class mean
        diffs = pts - class_means[cls]
        class_vars[cls] = np.mean(np.sum(diffs * diffs, axis=1))

    if len(class_means) < 2:
        return 0, 0, 0

    # Inter-class: variance of class means around global mean
    mean_diffs = np.array([class_means[c] - global_mean for c in class_means])
    inter = np.mean(np.sum(mean_diffs * mean_diffs, axis=1))

    # Intra-class: average of within-class variances
    intra = np.mean([class_vars[c] for c in class_vars])

    ratio = inter / max(intra, 1e-10)
    return ratio, inter, intra

def cosine_similarity_matrix(tuples):
    """Pairwise cosine similarity matrix (observer projection)."""
    norms = np.sqrt(np.sum(tuples * tuples, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-10)
    normalized = tuples / norms
    return normalized @ normalized.T

def within_between_cosine(cos_matrix, labels, class_names):
    """Average cosine similarity within vs between orbit classes."""
    n = len(labels)
    within_sims = []
    between_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within_sims.append(cos_matrix[i, j])
            else:
                between_sims.append(cos_matrix[i, j])

    within_mean = np.mean(within_sims) if within_sims else 0
    between_mean = np.mean(between_sims) if between_sims else 0
    return within_mean, between_mean


# ─── E8 Alignment ────────────────────────────────────────────────────

def generate_e8_roots():
    """Generate all 240 E8 root vectors."""
    roots = []
    # Type 1: all permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0) — 112 roots
    from itertools import combinations
    for positions in combinations(range(8), 2):
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                v = [0] * 8
                v[positions[0]] = s1
                v[positions[1]] = s2
                roots.append(v)

    # Type 2: (+-1/2, +-1/2, ..., +-1/2) with even number of minus signs — 128 roots
    for bits in range(256):
        v = []
        neg_count = 0
        for j in range(8):
            if bits & (1 << j):
                v.append(-0.5)
                neg_count += 1
            else:
                v.append(0.5)
        if neg_count % 2 == 0:
            roots.append(v)

    return np.array(roots)

def e8_alignment(tuple_4d, e8_roots):
    """
    Max cosine similarity of a 4D QA tuple projected to 8D against E8 roots.
    Projection: (b,e,d,a) → (b,e,d,a, b+e,b+d,e+a,d+a) normalized.
    """
    b, e, d, a = tuple_4d
    vec_8d = np.array([b, e, d, a,
                       b + e, b + d, e + a, d + a], dtype=float)
    norm = np.sqrt(np.sum(vec_8d * vec_8d))
    if norm < 1e-10:
        return 0.0
    vec_8d /= norm

    # Max cosine sim to any E8 root
    e8_norms = np.sqrt(np.sum(e8_roots * e8_roots, axis=1, keepdims=True))
    e8_normed = e8_roots / np.maximum(e8_norms, 1e-10)
    cos_sims = vec_8d @ e8_normed.T
    return np.max(cos_sims)


# ─── ANALYSIS ─────────────────────────────────────────────────────────

print("=" * 70)
print("QA REPRESENTATIONAL GEOMETRY ANALYSIS")
print("Does discrete modular arithmetic produce clean geometric structure?")
print("=" * 70)

e8_roots = generate_e8_roots()
print(f"E8 root system: {len(e8_roots)} vectors in 8D")

for m in [9, 24]:
    print(f"\n{'=' * 70}")
    print(f"MODULUS {m}: {m*m} tuples in 4D")
    print(f"{'=' * 70}")

    tuples, labels, pairs = generate_all_tuples(m)
    class_names = sorted(set(labels))
    census = Counter(labels)

    print(f"\nOrbit census: {dict(census)}")

    # 1. Participation ratio (effective dimensionality)
    pr = participation_ratio(tuples)
    print(f"\n1. EFFECTIVE DIMENSIONALITY")
    print(f"   Participation ratio (full): {pr:.2f} / 4.00")
    print(f"   (4.00 = fully distributed; 1.00 = rank-1)")

    for cls in class_names:
        mask = [i for i, l in enumerate(labels) if l == cls]
        if len(mask) > 1:
            pr_cls = participation_ratio(tuples[mask])
            print(f"   {cls:12s} ({len(mask):3d} pts): PR = {pr_cls:.2f}")

    # 2. Fisher discriminant ratio
    fisher, inter, intra = fisher_discriminant(tuples, labels, class_names)
    print(f"\n2. FISHER DISCRIMINANT RATIO")
    print(f"   Inter-class variance: {inter:.2f}")
    print(f"   Intra-class variance: {intra:.2f}")
    print(f"   Fisher ratio: {fisher:.2f}")
    print(f"   (>1 = classes separable; >>1 = clean clusters)")

    # 3. Centroid distances
    print(f"\n3. CENTROID DISTANCES")
    centroids = {}
    for cls in class_names:
        mask = [i for i, l in enumerate(labels) if l == cls]
        centroids[cls] = np.mean(tuples[mask], axis=0)
        print(f"   {cls:12s} centroid: [{', '.join(f'{x:.2f}' for x in centroids[cls])}]")

    for i, c1 in enumerate(class_names):
        for c2 in class_names[i+1:]:
            dist = np.sqrt(np.sum((centroids[c1] - centroids[c2]) *
                                  (centroids[c1] - centroids[c2])))
            print(f"   dist({c1}, {c2}) = {dist:.2f}")

    # 4. Cosine similarity structure
    cos_mat = cosine_similarity_matrix(tuples)
    within, between = within_between_cosine(cos_mat, labels, class_names)
    print(f"\n4. COSINE SIMILARITY STRUCTURE")
    print(f"   Within-orbit mean:  {within:.4f}")
    print(f"   Between-orbit mean: {between:.4f}")
    print(f"   Gap (within - between): {within - between:.4f}")
    print(f"   (Positive gap = orbits are cosine-distinguishable)")

    # Per-orbit-pair cosine analysis
    for i, c1 in enumerate(class_names):
        for c2 in class_names[i:]:
            sims = []
            mask1 = [k for k, l in enumerate(labels) if l == c1]
            mask2 = [k for k, l in enumerate(labels) if l == c2]
            for ii in mask1:
                for jj in mask2:
                    if ii != jj:
                        sims.append(cos_mat[ii, jj])
            if sims:
                print(f"   cos({c1:4s}, {c2:4s}): mean={np.mean(sims):.4f} "
                      f"std={np.std(sims):.4f}")

    # 5. Singular value spectrum
    centered = tuples - np.mean(tuples, axis=0)
    svd_vals = np.linalg.svd(centered, compute_uv=False)
    svd_vals_norm = svd_vals / np.sum(svd_vals) if np.sum(svd_vals) > 0 else svd_vals
    print(f"\n5. SINGULAR VALUE SPECTRUM")
    print(f"   Singular values: [{', '.join(f'{s:.2f}' for s in svd_vals[:4])}]")
    print(f"   Normalized:      [{', '.join(f'{s:.3f}' for s in svd_vals_norm[:4])}]")
    print(f"   Top-1 captures:  {svd_vals_norm[0]:.1%}")
    print(f"   Top-2 captures:  {sum(svd_vals_norm[:2]):.1%}")

    # 6. E8 alignment by orbit type
    print(f"\n6. E8 ALIGNMENT BY ORBIT TYPE")
    e8_by_orbit = defaultdict(list)
    for i, (t, l) in enumerate(zip(tuples, labels)):
        alignment = e8_alignment(t, e8_roots)
        e8_by_orbit[l].append(alignment)

    for cls in class_names:
        vals = e8_by_orbit[cls]
        print(f"   {cls:12s}: mean={np.mean(vals):.4f} "
              f"std={np.std(vals):.4f} "
              f"min={np.min(vals):.4f} max={np.max(vals):.4f}")

    # E8 alignment ANOVA
    if len(class_names) >= 2:
        groups = [e8_by_orbit[c] for c in class_names if len(e8_by_orbit[c]) > 1]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"   ANOVA F={f_stat:.2f}, p={p_val:.2e}")
            print(f"   Orbits differ in E8 alignment: {'YES' if p_val < 0.05 else 'NO'}")

    # 7. Resonance structure
    print(f"\n7. RESONANCE STRUCTURE (tuple inner products)")
    res_within = defaultdict(list)
    res_between = []
    for i in range(len(tuples)):
        for j in range(i + 1, min(i + 50, len(tuples))):  # sample for speed
            r = resonance(tuples[i], tuples[j])
            if labels[i] == labels[j]:
                res_within[labels[i]].append(r)
            else:
                res_between.append(r)

    for cls in class_names:
        if res_within[cls]:
            print(f"   Within-{cls:10s}: mean={np.mean(res_within[cls]):.1f} "
                  f"std={np.std(res_within[cls]):.1f}")
    if res_between:
        print(f"   Between-orbit:     mean={np.mean(res_between):.1f} "
              f"std={np.std(res_between):.1f}")
    within_all = [v for vals in res_within.values() for v in vals]
    if within_all and res_between:
        t_res, p_res = stats.mannwhitneyu(within_all, res_between, alternative='greater')
        print(f"   Within > Between: p={p_res:.4e}")


# ─── SUMMARY ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: QA Representational Geometry")
print("=" * 70)

print("""
Key question: Does QA produce clean geometric structure from pure
arithmetic — no optimization, no gradients, no learning?

Metrics to evaluate:
  - Participation ratio: how distributed is the representation?
  - Fisher ratio: how separable are the orbit clusters?
  - Cosine gap: do within-orbit pairs look more similar than between?
  - E8 alignment: do orbits align differently to exceptional geometry?
  - Resonance structure: does algebraic "attention" distinguish orbits?

If these metrics show clean structure, it demonstrates that discrete
modular arithmetic is a GENERATIVE SOURCE of representational geometry —
the kind of structure Kirsanov & Chung (2025) analyze in LLM embeddings,
but arising here without any learned parameters.

Connection to "Geometry of Prompting": their paper shows prompting
method determines representational geometry. QA shows modular arithmetic
determines representational geometry. The question: are these the same
geometry, viewed from different substrates?
""")

print("Script complete.")
