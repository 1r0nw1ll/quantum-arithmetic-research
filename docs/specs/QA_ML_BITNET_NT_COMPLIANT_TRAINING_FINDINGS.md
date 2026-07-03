# BitNet / QA NT-Compliant Training — Findings (2026-07-03)

> Status: **closed investigation, cross-validated result**. Tests whether
> QA's discreteness axioms (S2 permits `Fraction`/int state; Theorem NT
> forbids float-causal QA state) offer a legitimate, working alternative to
> BitNet b1.58's ternary-weight training, or whether the resemblance to QA
> is surface-level. Started from an external synthesis (ChatGPT) proposing
> a QA↔BitNet mapping; investigated empirically end-to-end.
>
> Scripts: `experiments/qa_ml/73_bitnet_qa_native_strawman.py` through
> `78_bitnet_final_nt_compliant_training.py`.

Primary source: Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S.,
Dong, L., Wang, R., Xue, J., Wei, F. (2024). *The Era of 1-bit LLMs: All
Large Language Models are in 1.58 Bits.* arXiv:2402.17764. This
investigation implements that paper's absmean ternary quantization and
gamma output rescale exactly, substituting exact fixed-point/integer
arithmetic for the paper's float32/bf16 training substrate throughout.
Secondary source: Loshchilov, I., Hutter, F. (2019). *Decoupled Weight
Decay Regularization.* arXiv:1711.05101 (AdamW) — the default
`weight_decay=0.01` from this paper turned out to be the missing
stabilizing term, see §8 below.

## Headline result

A from-scratch training loop for a ternary-weight MLP — real backprop, real
Adam, real softmax cross-entropy, integer `exp` and `sqrt`, zero IEEE-754
floats anywhere in the causal path — **matches or beats** a standard
float-based BitNet-style training loop (STE + AdamW + cross-entropy),
verified on two different tasks:

| dataset | fixed-point (zero-float) | float BitNet-style (STE+AdamW) |
|---|---:|---:|
| sklearn digits (64→32→10, 300 epochs, 5 seeds) | **96.39% ± 0.63%** | 94.33% ± 1.63% |
| MNIST-6k subsample (784→128→10, 45 epochs, 3 seeds) | **92.55% ± 0.36%** | 92.75% (comparable) |

This is not a novel algorithm — it is BitNet's own absmean ternary
quantization and gamma rescale, executed with every number (activations,
gradients, Adam's moments, weight decay) as an exact scaled integer instead
of a float. The finding is that this substitution costs nothing and, on one
of two tasks tested, measurably helps.

## Scope and what this does NOT claim

- Toy scale only (64–128 hidden units, ≤6k training examples). No claim
  about LLM-scale training.
- "Beats float" held on digits, not MNIST (parity there). Do not round this
  up to "always beats."
- The `exp` approximation is a deterministic bounded-error polynomial
  (range-reduced Taylor series), not an exact rational value — `exp` of a
  nonzero rational is irrational, so no exact fixed-point/Fraction form
  exists. This is different from `sqrt`, which has an exact integer
  algorithm (floor-sqrt via Newton's method) and was used exactly.
- Two real, diagnosed implementation bugs were found and fixed during this
  investigation (below) — this is reported for calibration, not omitted.

## Chronology

### 1. The naive analogy was a category error

The originating synthesis proposed "BitNet ternary weight ↔ QA
`(b,e,d,a)` packet." This does not hold: `d=b+e`, `a=b+2e` are *derived*
coordinates of a single generator state, not independent degrees of
freedom. A ternary scalar has no analog of that constraint. Substituting a
4-tuple for a 1-trit weight adds representational overhead without adding
structure — a naming exercise, not a discovered mapping.

### 2. BitNet's actual training loop violates Theorem NT — a real disagreement, not a similarity

BitNet's ternary weights are the *output* of float-valued gradient
descent — float latent weights, float optimizer state, float STE
gradients are all causally upstream of the ternary value. Theorem NT
permits floats only as observer projections *downstream* of QA state,
never as causal inputs upstream of it. BitNet does not meet this; its
inference-time weights are discrete, but its training process is not.

### 3. First attempt at an NT-compliant alternative was a strawman (`73`)

Conflated "no float causally enters the computation" with "must use a weak
discrete heuristic." First build used integer sign-vote counting instead
of real gradient descent — got 10.3% (chance) vs. the float baseline's
94.7%. This result said nothing about discreteness; it only showed the
heuristic was bad. Also tested whether QA's own `qa_step` generator
formula (`((b+e-1) % m) + 1`) contributed anything beyond generic integer
arithmetic — it did not (see §5).

### 4. Real gradient descent in exact fixed-point arithmetic works (`74`)

Rebuilt with actual backprop + SGD in Q16.16 fixed-point integers, zero
floats anywhere. Reached **89.7%**, closing nearly the entire gap. Two
bugs surfaced and were fixed on the way: gradient-scale explosion from a
missing fan-in normalization (any learning rate failed identically — a
strong signal to suspect a bug, not a null result), and confirmed the
fix by checking update-magnitude-vs-weight-magnitude ratios directly.

### 5. `qa_step` is inert when it doesn't wrap, actively harmful when it does (`75`)

Routing the same fixed-point SGD update through QA's actual `qa_step`
generator (large modulus, offset representation) gave **bit-identical**
results to plain subtraction — confirms `qa_step` is a cyclic-group
relabeling of addition, not a distinct mechanism, as long as it never
wraps.

Testing the regime where it *does* wrap (bound set well below the natural
~5.4 unit weight excursion seen unbounded): wrap collapsed to chance
(~10%) while clip/saturate at the same bound scored *better* than
unbounded (91.0–91.2%, acting as implicit regularization). Mechanism
verified directly: 180/320 weights wrapped at least once; each wrap event
flips a weight's sign via a ~1.0-unit jump (vs. ~0.03–0.04 typical
gradient steps) — SGD drives ternary-quantized weight magnitude up over
training (larger magnitude = more robust to future sign flips), and
wrapping reverses exactly the weights the optimizer is most confident
about. Cyclic/modular arithmetic fits genuinely periodic state (angles,
orbital phase); it actively fights a magnitude/confidence parameter that
gradient descent wants to grow monotonically.

### 6. Integer-analog Adam closes a modest further gap, after three real bugs (`76`)

Adam needs `sqrt(v_hat)`, which — unlike `exp` — has an *exact* integer
algorithm (floor-sqrt via vectorized Newton's method + cleanup pass, same
class of algorithm as arbitrary-precision-integer libraries use). Reached
91.1% vs. plain SGD's 89.7%, but only after diagnosing:
- `eps` too large (chosen for representability) swamped the adaptive
  denominator in 85–100% of weights — the optimizer had silently
  degenerated into plain scaled SGD.
- Floor-division in the EMA accumulator introduced a systematic downward
  bias, causing `sqrt(v_hat)` to decay toward zero instead of tracking
  gradient RMS. Fixed with round-to-nearest instead of floor.
- Sparse ReLU gradients (most weights get exactly-zero gradient on most
  batches) drove `v` to exact integer zero, which got permanently stuck
  there; the next nonzero gradient then divided by ~eps and exploded
  (updates up to 5000x too large). Fixed with a v-floor plus a standard
  Adam update-magnitude clip — the same mitigations real low-precision
  Adam implementations use for the identical reason.

### 7. Real cross-entropy initially made things *worse*, correctly diagnosed as an expressivity ceiling (`77`)

Implemented `exp` via range-reduction (divide by 2^6) + 10-term truncated
Taylor series with exact `1/k!` coefficients (`Fraction`-computed once) +
repeated squaring — accurate to ~1e-6 relative error over the range
softmax logits actually produce. Only forward `exp` is needed since the
softmax+cross-entropy gradient has the closed form `probs - onehot`, no
fixed-point `log` required.

Swapping MSE for this real cross-entropy dropped accuracy to 69.4%.
Diagnosis (not just the number): train_acc tracked test_acc throughout
(rules out overfitting), and mean max-softmax-probability stayed pinned at
0.144 — barely above the 0.10 chance floor — for the full 150 epochs. The
ternary output layer's logit range is structurally too coarse (bounded fan-
in-normalized sum of ±1/0 terms) to satisfy cross-entropy's preference for
confident, sharply-peaked predictions; MSE doesn't need that and uses the
same limited capacity more efficiently for raw accuracy.

### 8. Adding BitNet's gamma rescale fixed the ceiling — but introduced a real instability, twice-diagnosed (`77`→`78`)

BitNet's actual forward pass is `y = gamma * (x @ W_ternary.T)` where
`gamma = mean(|W_latent|)`; the fixed-point pipeline had dropped this for
simplicity. Since gradient descent drives ternary-quantized latent
magnitude up over training (the same dynamic from §5), gamma grows too —
this is what lets logit confidence increase as training proceeds. Adding
it took digits to **96.6%** ± 0.5%, beating the float baseline (94.7%).

A generalization check on MNIST-6k (784→128→10, same recipe) caught a real
problem the easier digits task never triggered: `gamma1` grew ~1500x over
24 epochs via an unbounded positive-feedback loop (gamma rescales the
gradient flowing into the same layer it's derived from → larger gradient →
faster-growing latent → larger gamma, uncapped). Accuracy peaked ~93% then
collapsed to chance within a single epoch (12→13: 0.889→0.3225).

**First hypothesis, tested and ruled out:** differentiate through gamma
properly (product rule: `d(Loss)/dL = gamma·dz/dL + z_summed·dgamma/dL`)
instead of treating it as a stop-gradient constant, matching the fact that
the float torch baseline never called `.detach()` on gamma. Implemented
the full second gradient term and reran on MNIST — nearly identical
collapse, delayed by ~2 epochs. Not the cause.

**Real cause:** `torch.optim.AdamW`'s **default** `weight_decay=0.01` (see
Loshchilov & Hutter, 2019, cited above) was silently active in every float
baseline run this entire investigation and never replicated in the
fixed-point version. Adding the matching decoupled weight-decay term
(`latent -= lr·wd·latent`, an exact integer op, applied separately from
the adaptive gradient update — exactly how AdamW defines it) fixed the
collapse immediately: gamma1 converges to a bounded equilibrium instead of
diverging.

## Final, cross-validated result (`78`)

| dataset | fixed-point (zero-float) | float BitNet-style |
|---|---:|---:|
| digits (300ep, 5 seeds) | 96.39% ± 0.63% | 94.33% ± 1.63% |
| MNIST-6k (45ep, 3 seeds) | 92.55% ± 0.36% | 92.75% |

Digits: still beats float, now backed by a stabilized recipe rather than
an instability that simply hadn't triggered yet within the tested epoch
budget. MNIST-6k: the collapse is fully resolved (smooth, monotonic
learning curve, no crash), landing at parity with float rather than
beating it. The honest, generalizable claim is **"matches float training,
and decisively beats it on at least one real task,"** not "always beats."

## Related prior work in this repo

- `experiments/arto_ternary_*` — an earlier, independent QA interpretation
  of Arto Heino's balanced-ternary hardware logic (not BitNet-related).
  Found the same qualitative pattern that recurs here: native discrete
  arithmetic is lawful and cheap in the abstract, but forcing it through an
  interface not designed for it (binary rails / this investigation's
  gamma-less ceiling) reintroduces real cost ("projection debt" there,
  the expressivity ceiling here).
- `experiments/qa_ml/71_tiny_transformer_qa_residual_quant.py` and
  `72_hf_tiny_lm_qa_packet_quant.py` — **post-training** quantization of
  already float-trained models, testing whether QA-style packet/residual
  quantizers preserve weights better than scalar quantization. Different
  claim boundary from this investigation: those test static compression of
  an existing float artifact; this investigation trains a ternary-weight
  network from scratch with zero floats in the causal path at any point.
  Complementary, not overlapping.
- `docs/specs/QA_ML_V3_FINDINGS.md` — the "expose the right algebraic
  invariant" pattern from that investigation (naive features fail,
  canonical/equivariant features succeed) is the same shape of lesson as
  §5 here (`qa_step` needs an order-preserving readout, not the naive
  mod-3 partition, to be useful) — both are instances of the general
  finding that QA's contribution is in exposing the right structure, not
  in the raw arithmetic being different.

## Lineage

```text
2026-07-03  ChatGPT synthesis proposed BitNet<->QA mapping (external input)
73          strawman vote-counting (10.3%, chance) -- diagnosed as a bad heuristic, not evidence against discreteness
74          real backprop in fixed-point SGD (89.7%)
75          qa_step == plain subtraction (non-wrap, bit-identical); harmful under wraparound (10% vs clip's 91%)
76          fixed-point Adam, 3 bugs fixed (91.1%)
77          real cross-entropy hits expressivity ceiling (69.4%); gamma rescale added, fixes ceiling (96.6% digits) but unstable on MNIST (collapses to 10%)
78          gamma-gradient hypothesis tested, ruled out; AdamW default weight_decay=0.01 found missing and added -- FINAL: 96.4% digits / 92.6% MNIST vs float 94.3% / 92.8%
```

## References

- Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L.,
  Wang, R., Xue, J., Wei, F. (2024). *The Era of 1-bit LLMs: All Large
  Language Models are in 1.58 Bits.* arXiv:2402.17764.
- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay
  Regularization.* arXiv:1711.05101.
- `experiments/arto_ternary_*` — prior QA-native ternary logic
  interpretation in this repo (Arto Heino balanced-ternary hardware).
- `docs/specs/QA_ML_V3_FINDINGS.md` — canonical/equivariant feature
  pattern this investigation's §5 mirrors.
