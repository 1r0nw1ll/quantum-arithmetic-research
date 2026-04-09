# Pre-registration — Bateson Tiered Reachability × MITRE ATT&CK

**Locked:** 2026-04-05 (before any results file exists)
**Script:** `qa_bateson_attack_reachability.py` (repo root)
**Cert family targeted for extension:** [191] QA_BATESON_LEARNING_LEVELS_CERT.v1
**Runner:** Claude (per 2026-03-09 autonomy override)

## Why this experiment

External landscape survey (2026-04-05, 3 parallel research subagents, OB capture `018fbb4d`) identified **a concrete gap**: no prior work does reachability/unreachability proofs on the MITRE ATT&CK graph. The entire ecosystem treats ATT&CK as a taxonomy/coverage tool. [191]'s Tiered Reachability Theorem — shipped 2026-04-04, exhaustively verified on S_9 with exact counts (81+1712+3456+1312=6561) — is a structurally novel primitive that has never been applied to a real threat-intelligence graph.

This is a pilot to test whether the ATT&CK graph inherits QA orbit structure under a principled encoding, as a first step toward a "provable denial zone" cert extension.

## Hypotheses (sign-locked)

**H1 (primary):** Real attacker technique chains — extracted from MITRE intrusion-set `uses` relationships and ordered by kill-chain tactic sequence — show a tier distribution that differs from a uniform random-chain baseline. Specifically, **low-tier transitions (tier 0 and tier 1) are over-represented** in real data.

> Test: chi-square(observed tier counts, baseline tier counts), p < 0.001.
> Direction: `(n_tier_0 + n_tier_1) / n_total` in real data > in shuffled data, locked positive.

**H2 (secondary):** The orbit-family distribution over all ATT&CK techniques differs from uniform.

> Test: chi-square against uniform over {cosmos, satellite, singularity}, p < 0.001.

**H3 (descriptive):** The tier distribution across all intrusion-set transition pairs matches the structural prediction of [191] qualitatively — i.e., Level-I is the largest reachable tier, Level-2a dominates the non-reachable mass, Level-2b and Level-0 are smaller. No numerical commitment since the S_9 counts are specific to the uniform-over-S_9 setting, not the attacker-conditional setting.

## Data

- **Source:** `enterprise-attack.json` from `https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json`
- **License:** [ATT&CK Terms of Use](https://attack.mitre.org/resources/legal-and-branding/terms-of-use/) — permissive for research
- **Cache:** `qa_alphageometry_ptolemy/external_validation_data/mitre_attck_stix/enterprise-attack.json` + `.MANIFEST.json` with SHA256
- **Expected:** ~800 `attack-pattern` objects (techniques + sub-techniques), ~140 `intrusion-set` objects (groups), thousands of `relationship` objects
- **No modification** to any other cert or data file

## Observer projection (Theorem-NT boundary crossing)

The STIX bundle is a JSON document of discrete identifiers. There is **no continuous-to-discrete crossing** in this experiment. All inputs (technique IDs, tactic names, relationships) are natively discrete symbols. The "projection" is just a deterministic lookup table: technique → (tactic_rank, within_tactic_rank) → (b, e). No T2-b risk. No floats.

## Encoding

**Canonical tactic ordering** (14 tactics, kill-chain order):
1. reconnaissance
2. resource-development
3. initial-access
4. execution
5. persistence
6. privilege-escalation
7. defense-evasion
8. credential-access
9. discovery
10. lateral-movement
11. collection
12. command-and-control
13. exfiltration
14. impact

**Mapping:**
- `tactic_rank(T)` = position of T's primary tactic in the canonical list (1..14). Primary tactic = first `kill_chain_phase` entry in the STIX object (deterministic).
- `within_tactic_rank(T)` = rank of T among all techniques sharing its primary tactic, sorted alphabetically by `external_id` (T1234, T1234.001, etc.).
- `b = qa_mod(tactic_rank(T), 9)` — in {1..9}
- `e = qa_mod(within_tactic_rank(T), 9)` — in {1..9}
- `d = qa_mod(b + e, 9)`, `a = qa_mod(b + 2*e, 9)`

Collisions into S_9: ~800 techniques into 81 states ≈ 10 per state average. Collisions are **identical under the null**, so the chi-square test is properly controlled: both real and baseline chains pigeonhole into the same state alphabet at the same rate.

**Orbit family** (at m=9, per [191]):
- `singularity`: (b, e) == (9, 9)
- `satellite`: 3|b AND 3|e, excluding singularity
- `cosmos`: everything else

**Tier classification** for a transition (s₀ → s₁):
- tier 0: s₀ == s₁
- tier 1: same T-orbit (reachable by powers of `qa_step`)
- tier 2a: different orbit, same family
- tier 2b: different family, same modulus

## Technique-chain construction

For each `intrusion-set` object G:
1. Collect all techniques T where a `relationship(G uses T)` exists.
2. Sort those techniques by `(tactic_rank, within_tactic_rank)` to produce a canonical kill-chain ordering.
3. Take consecutive pairs → transitions.
4. Classify each transition's tier.

Aggregate across all intrusion-sets → observed tier histogram.

## Baseline

For each intrusion-set G with k techniques, generate 1000 random chains by sampling k techniques uniformly with replacement from the full ATT&CK technique set, keeping the tactic-sorted ordering. Compute tier histograms for each, average.

**Why this baseline:** it controls for chain length and tactic ordering (so the effect we measure is orbit structure beyond tactic-ordering), but does NOT control for group-specific technique preference (that's H1's entire point — groups cluster in specific orbits).

## Statistical test

Primary: **chi-square(observed_tier_counts, expected_tier_counts_from_baseline)** with 3 df (tiers 0, 1, 2a, 2b). Report statistic, p, and effect size (Cohen's w).

Secondary: permutation test — shuffle group→technique assignments 1000 times, recompute tier histograms, compare observed chi-square to the permutation null. Reports empirical p.

## Decision criteria (locked)

| Outcome | Conditions | Action |
|---|---|---|
| **STRONG** | H1 chi² p < 0.001 (perm p < 0.001) AND (low_tier_frac_real > low_tier_frac_baseline) AND H2 p < 0.001 | Propose cert amendment: add "attack graph witness" to [191]; write new fixture; OB + memory capture; paper candidate |
| **WEAK** | H1 OR H2 p < 0.05, but not STRONG | Pilot positive; no cert amendment; flag for follow-up (expand to ATT&CK mobile/ICS or add sub-technique resolution) |
| **NULL** | neither H1 nor H2 significant, or direction wrong | Honest negative; OB capture; the m=9 encoding does not carry ATT&CK signal; consider m=24 encoding as fallback (separate pre-reg) |

Sign flip at significance counts as NULL (the whole point of [191] is that low-tier reachability is the *dominant* structure; an inversion would be a different, unrelated phenomenon).

## Deliverables

1. `qa_bateson_attack_reachability.py` — standalone script, repo root, linter-clean
2. `external_validation_data/mitre_attck_stix/enterprise-attack.json` + `.MANIFEST.json` (SHA256 locked)
3. `results/bateson_attack/`:
   - `per_group.csv` (intrusion-set-level tier counts)
   - `per_transition.csv` (all observed transitions with tier labels)
   - `orbit_family_histogram.png`
   - `tier_distribution.png`
   - `summary.json`
4. OB captures at: data pull, first stats, final outcome
5. If STRONG: new fixture `qa_bateson_learning_levels_cert_v1/fixtures/bll_attck_witness_pass.json`

## Guardrails checklist

- [x] Theorem-NT: no continuous→discrete crossing (STIX is natively discrete)
- [x] A1: all states in {1..9}, never 0 (qa_mod enforces)
- [x] S1: no `**2`
- [x] S2: states are `int`
- [x] T1: path length is integer transition count
- [x] No secrets committed (public STIX bundle)
- [x] Sign locked before first run
- [x] Baseline controls chain length AND tactic ordering AND collision rate
- [x] Permutation test preserves observed data structure

## What would make me wrong

- **NULL result**: ATT&CK does not inherit QA structure under the (tactic, within-tactic) mod-9 encoding. This would be important — it would mean the Bateson filtration is a feature of the specific algebraic structure of (Z/9Z)^2, not a general discrete-graph property. It would narrow [191]'s scope to "algebraic state spaces only" and suggest a different encoding (e.g., technique-similarity-graph-based) is needed for attack-graph applications.
- **Direction flip**: real chains show *more* tier-2 transitions than baseline. This would suggest attackers deliberately cross orbit boundaries — potentially a sign of sophistication, but it would invalidate the "QA structure = natural attacker behavior" framing.
- **STRONG but driven by H2 alone**: orbit-family distribution is non-uniform but transitions look random. This would mean ATT&CK has structure in its *states* but not its *dynamics* — a weaker finding than full-filtration inheritance.
