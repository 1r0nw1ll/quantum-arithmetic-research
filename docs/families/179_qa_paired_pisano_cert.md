# Family [179] QA_PAIRED_PISANO_CERT.v1

## One-line summary

Fibonacci coprime pairs have 2.25x higher paired Pisano divisibility (both p|pi(m) AND q|pi(m)) than non-Fibonacci pairs (p=0.0017, 499 moduli).

## Mathematical content

### Paired divisibility

For coprime pair (p,q) and modulus m, "both-divide" means both p and q divide the Pisano period pi(m). Across m=2..500:

- Fibonacci pairs: mean both-divide rate 0.526
- Non-Fibonacci pairs: mean both-divide rate 0.234
- Ratio: 2.25x, Mann-Whitney U p=0.0017

### Order-1 deep dive

Among consecutive ratios (|p-q|=1): Fibonacci mean 0.857 vs non-Fibonacci mean 0.227, ratio 3.77x, Mann-Whitney p=0.028.

### LCM mechanism

For coprime (p,q), both-divide requires lcm(p,q)=p*q | pi(m). Fibonacci pairs have smaller products AND Fibonacci numbers individually divide pi(m) 3.70x more often (from [219]).

### Product-matched control

5/9 (56%) Fibonacci pairs beat their closest product-matched non-Fibonacci pair. Advantage is real but modest after controlling for product size.

### 4:1 exception

4:1 has lcm=4 (very small), giving it higher both-divide rate than some Fibonacci pairs at order-3. This explains why Kirkwood gap 4:1 is as deep as Fibonacci resonances.

## Tier

Tier 3 (p<0.01 with honest controls).

## Checks

PP_1, PP_PAIRS, PP_STAT, PP_RATIO, PP_ORDER, PP_MECH, PP_HONEST, PP_W, PP_F

## Fixtures

1 PASS + 1 FAIL; self-test ok.

## Source

`qa_pisano_paired_divisibility.py`

## Validator checks

| Check | Description |
|-------|-------------|
| PP_1 | schema_version == 'QA_PAIRED_PISANO_CERT.v1' |
| PP_PAIRS | ≥4 fibonacci_pairs with valid both_divide_rate |
| PP_RECOMPUTE | every declared both_divide_rate genuinely recomputed from real Pisano periods m=2..500 (added 2026-07-06) |
| PP_STAT | overall Mann-Whitney p < 0.05 |
| PP_RATIO | overall ratio ≥ 1.5 |
| PP_ORDER | order_analysis present, order labels genuinely match \|p-q\|, and fib_mean/nonfib_mean/ratio genuinely recomputed (hardened 2026-07-06) |
| PP_MECH | mechanism.lcm_principle present |
| PP_HONEST | caveats section present |
| PP_W | ≥2 sources |
| PP_F | fail detection |

## Verification Note (2026-07-06)

Independently re-ran `qa_pisano_paired_divisibility.py` (which is fully
deterministic — no randomness affects the Pisano-period/divisibility
computation) and confirmed the cert's headline claims exactly: overall
Fibonacci both-divide mean 0.5264≈0.526, non-Fibonacci mean 0.234,
ratio 2.25x, Mann-Whitney U=145.5 p=0.0017; order-1 Fibonacci mean
0.8567≈0.857, non-Fibonacci 0.2273≈0.227, ratio 3.77x, p=0.028;
product-matched wins 5/9 (56%). All match the cert's declared
top-level `statistics` block and the original authoring commit message
(`29dc7aa1`) exactly — **the core scientific claim is genuine and
reproducible**, not fabricated.

**However, found that every supporting detail table was wrong**:

1. **All 9 `fibonacci_pairs[].both_divide_rate` values were incorrect**
   — none matched genuine recomputation for their own (p,q) label (e.g.
   declared (2,1)=0.8577 vs actual 0.9980; declared (8,5)=0.3146 vs
   actual 0.2124). The mean of the 9 wrong values happens to be within
   0.0007 of the mean of the 9 correct values (0.5257 vs 0.5264) — an
   unexplained coincidence that let the aggregate `overall_fib_mean`
   pass unnoticed despite every individual entry being wrong.

2. **The `order_analysis` table had wrong pair-to-order groupings**:
   `(8,3)` was listed under `order: 4` alongside `(5,1)`, but
   `|8-3|=5 ≠ 4` — it belongs in its own order-5 row (which was
   entirely missing from the fixture).

3. **Every order's fib_mean/nonfib_mean/ratio was wrong**, and at
   **order 3 the error reversed the scientific conclusion**: the
   fixture declared a 1.77x Fibonacci advantage, but genuine
   recomputation gives ratio=0.83 — Fibonacci pairs actually
   *underperform* non-Fibonacci pairs at order 3. This is not a rounding
   nitpick: a ratio <1 versus >1 is the difference between "Fibonacci
   wins" and "Fibonacci loses" at that order. Notably, the corrected
   result **resolves an internal contradiction** the old fixture didn't
   catch — its own `mechanism.exception` text already said "4:1 beats
   Fibonacci at order-3" (a non-Fibonacci pair outperforming), which is
   logically incompatible with the same fixture's order_analysis
   claiming a 1.77x *Fibonacci* advantage at that same order. The
   corrected numbers make the cert internally consistent for the first
   time: non-Fibonacci order-3 mean (0.362, pulled up by 4:1's high
   rate) genuinely exceeds the Fibonacci order-3 mean (0.300).

**Checked whether this affected the paper**: `papers/in-progress/
fibonacci-resonance/paper.tex` only states the general "Pisano period
divisibility" mechanism concept in its conclusion — it does not cite
any of the specific broken numbers (no matches for "2.25", "0.526",
"3.77", "Mann-Whitney", "both-divide" in the paper source). The
paper's actual empirical claims (341-system OOS replication, Fisher
p=3.21e-46) are a separate, already-audited analysis unaffected by
this cert's detail-table bugs.

**Fixes applied**: hardened the validator with `PP_RECOMPUTE` (genuine
per-pair Pisano-period recomputation) and a hardened `PP_ORDER` (checks
order labels against real `|p-q|`, and recomputes fib_mean/nonfib_mean/
ratio per order); regenerated the fixture's `fibonacci_pairs` and
`order_analysis` sections with genuinely correct values, including
adding the missing order-5 row and correcting the order-3 sign
reversal. Verified both hardened checks reject planted regressions
(old per-pair value, old order-3 ratio).
