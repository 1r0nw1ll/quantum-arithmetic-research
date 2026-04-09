#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=none, state_alphabet=mod9_discrete_from_symbolic_STIX"
"""
qa_bateson_attack_reachability.py — Bateson Tiered Reachability × MITRE ATT&CK.

Tests whether the MITRE ATT&CK technique-chain graph inherits QA orbit structure
under the canonical (tactic_rank, within_tactic_rank) mod-9 encoding, aligned
with [191]'s exhaustively-verified S_9 filtration.

Pre-registration: results/bateson_attack/PRE_REGISTRATION.md (locked 2026-04-05).

Hypothesis (sign-locked):
    H1: real attacker technique chains (from MITRE intrusion-set `uses` relations)
        show over-represented low-tier (0 + 1) transitions vs a tactic-ordered
        random baseline. chi-square p < 0.001.
    H2: orbit-family distribution over all techniques differs from uniform.

Theorem-NT: no continuous->discrete crossing. STIX is natively symbolic.

Run:
    python qa_bateson_attack_reachability.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------ paths
REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "qa_alphageometry_ptolemy" / "external_validation_data" / "mitre_attck_stix"
DATA_FILE = DATA_DIR / "enterprise-attack.json"
MANIFEST_FILE = DATA_DIR / "MANIFEST.json"
RESULTS_DIR = REPO / "results" / "bateson_attack"

STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"

# ------------------------------------------------------------------ QA primitives (local, no imports)
MODULUS = 9
SEED = 42
N_PERM = 1000


def qa_mod(x: int, m: int = MODULUS) -> int:
    """A1-compliant: result in {1..m}, never 0."""
    return ((int(x) - 1) % m) + 1


def qa_step(b: int, e: int, m: int = MODULUS) -> tuple[int, int]:
    """Fibonacci map: (b,e) -> (e, b+e mod m). A1-compliant."""
    return (e, qa_mod(b + e, m))


def orbit_family_s9(b: int, e: int) -> str:
    """Per [191] cert: singularity=(9,9), satellite=both div by 3, else cosmos."""
    if b == 9 and e == 9:
        return "singularity"
    if (b % 3 == 0) and (e % 3 == 0):
        return "satellite"
    return "cosmos"


def enumerate_orbits() -> tuple[list[tuple[tuple[int, int], ...]], dict[tuple[int, int], int]]:
    """Decompose S_9 into T-orbits. Returns (orbit_list, index_map)."""
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for b in range(1, MODULUS + 1):
        for e in range(1, MODULUS + 1):
            if (b, e) in seen:
                continue
            orbit: list[tuple[int, int]] = []
            cur = (b, e)
            while cur not in seen:
                seen.add(cur)
                orbit.append(cur)
                cur = qa_step(cur[0], cur[1])
            orbits.append(tuple(orbit))
    idx = {pt: i for i, o in enumerate(orbits) for pt in o}
    return orbits, idx


def classify_tier(s0: tuple[int, int], s1: tuple[int, int], idx: dict[tuple[int, int], int]) -> str:
    """Per [191] Tiered Reachability Theorem."""
    if s0 == s1:
        return "0"
    if idx[s0] == idx[s1]:
        return "1"
    if orbit_family_s9(*s0) == orbit_family_s9(*s1):
        return "2a"
    return "2b"


# ------------------------------------------------------------------ canonical tactic order
TACTIC_ORDER = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]
TACTIC_RANK = {t: i + 1 for i, t in enumerate(TACTIC_ORDER)}


# ------------------------------------------------------------------ data pull
def pull_stix() -> dict:
    """Download enterprise-attack.json from MITRE's github mirror."""
    if DATA_FILE.exists() and MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)
        with open(DATA_FILE, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual == manifest.get("sha256"):
            print(f"[data] cache hit, sha256={actual[:12]}")
            with open(DATA_FILE) as f:
                return json.load(f)
        print("[data] cache sha256 mismatch — refetching")

    print(f"[data] fetching {STIX_URL} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(STIX_URL, headers={"User-Agent": "qa-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    with open(DATA_FILE, "wb") as f:
        f.write(raw)
    sha = hashlib.sha256(raw).hexdigest()
    manifest = {
        "source": "mitre-attack/attack-stix-data",
        "source_url": STIX_URL,
        "license": "MITRE ATT&CK Terms of Use",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": sha,
        "bytes": len(raw),
        "script": "qa_bateson_attack_reachability.py",
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[data] wrote {len(raw)/1e6:.1f} MB, sha256={sha[:12]}")
    return json.loads(raw)


# ------------------------------------------------------------------ STIX parsing
def parse_bundle(bundle: dict) -> dict:
    """Extract techniques, intrusion-sets, and 'uses' relationships."""
    techniques: dict[str, dict] = {}   # stix_id -> {external_id, primary_tactic, name, revoked}
    intrusion_sets: dict[str, dict] = {}
    uses_rels: list[tuple[str, str]] = []  # (source_ref, target_ref)

    for obj in bundle.get("objects", []):
        otype = obj.get("type")
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        if otype == "attack-pattern":
            # Find external_id
            ext_id = None
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    ext_id = ref.get("external_id")
                    break
            if not ext_id:
                continue
            # Primary tactic = first kill_chain_phase with mitre-attack source
            primary_tactic = None
            for kcp in obj.get("kill_chain_phases", []):
                if kcp.get("kill_chain_name") == "mitre-attack":
                    primary_tactic = kcp.get("phase_name")
                    break
            if not primary_tactic:
                continue
            techniques[obj["id"]] = {
                "external_id": ext_id,
                "primary_tactic": primary_tactic,
                "name": obj.get("name", ""),
            }

        elif otype == "intrusion-set":
            intrusion_sets[obj["id"]] = {
                "name": obj.get("name", ""),
            }

        elif otype == "relationship" and obj.get("relationship_type") == "uses":
            uses_rels.append((obj.get("source_ref"), obj.get("target_ref")))

    return {
        "techniques": techniques,
        "intrusion_sets": intrusion_sets,
        "uses_rels": uses_rels,
    }


# ------------------------------------------------------------------ QA encoding
def encode_techniques(techniques: dict) -> dict[str, tuple[int, int]]:
    """Map each technique stix_id -> (b, e) via (tactic_rank, within_tactic_rank)."""
    # Group by primary tactic
    by_tactic: dict[str, list[tuple[str, str]]] = defaultdict(list)  # tactic -> [(ext_id, stix_id)]
    for stix_id, info in techniques.items():
        by_tactic[info["primary_tactic"]].append((info["external_id"], stix_id))

    encoding: dict[str, tuple[int, int]] = {}
    for tactic, items in by_tactic.items():
        items.sort()  # alphabetical by external_id
        trank = TACTIC_RANK.get(tactic)
        if trank is None:
            continue
        b = qa_mod(trank)
        for within_rank, (ext_id, stix_id) in enumerate(items, start=1):
            e = qa_mod(within_rank)
            encoding[stix_id] = (b, e)
    return encoding


def group_technique_chains(parsed: dict) -> dict[str, list[str]]:
    """For each intrusion-set, return its technique chain sorted by (tactic_rank, within_tactic_rank)."""
    techniques = parsed["techniques"]
    intrusion_sets = parsed["intrusion_sets"]
    uses_rels = parsed["uses_rels"]

    # Build group -> set of technique stix_ids
    group_techs: dict[str, set[str]] = defaultdict(set)
    for src, tgt in uses_rels:
        if src in intrusion_sets and tgt in techniques:
            group_techs[src].add(tgt)

    # Sort each group's techniques by (tactic_rank, within_tactic_rank)
    # Need within_tactic_rank — recompute from encoding keys
    chains: dict[str, list[str]] = {}
    for gid, tech_ids in group_techs.items():
        items = []
        for tid in tech_ids:
            info = techniques[tid]
            trank = TACTIC_RANK.get(info["primary_tactic"], 99)
            items.append((trank, info["external_id"], tid))
        items.sort()
        chains[gid] = [tid for _, _, tid in items]
    return chains


# ------------------------------------------------------------------ stats
def tier_histogram(chains: dict[str, list[str]], encoding: dict[str, tuple[int, int]],
                   orbit_idx: dict[tuple[int, int], int]) -> Counter:
    """Classify each consecutive transition in each chain by tier."""
    counter: Counter = Counter()
    for gid, chain in chains.items():
        for i in range(len(chain) - 1):
            s0 = encoding[chain[i]]
            s1 = encoding[chain[i + 1]]
            counter[classify_tier(s0, s1, orbit_idx)] += 1
    return counter


def baseline_tier_histogram(chains: dict[str, list[str]], all_tech_ids: list[str],
                            encoding: dict[str, tuple[int, int]],
                            orbit_idx: dict[tuple[int, int], int],
                            n_iter: int, seed: int) -> Counter:
    """Baseline: for each group, sample k random techniques, sort by (tactic, within),
    compute transitions. Average over n_iter."""
    rng = np.random.default_rng(seed)
    techniques_meta = [(tid, encoding[tid]) for tid in all_tech_ids]
    counter: Counter = Counter()
    for _ in range(n_iter):
        for gid, chain in chains.items():
            k = len(chain)
            if k < 2:
                continue
            idxs = rng.choice(len(techniques_meta), size=k, replace=False)
            picks = [techniques_meta[i] for i in idxs]
            # Sort by (b, e) to mimic tactic-ordering
            picks.sort(key=lambda x: (x[1][0], x[1][1]))
            for i in range(k - 1):
                s0 = picks[i][1]
                s1 = picks[i + 1][1]
                counter[classify_tier(s0, s1, orbit_idx)] += 1
    # Scale down to per-iteration average
    return Counter({k: v / n_iter for k, v in counter.items()})


def permutation_test(chains: dict[str, list[str]], all_tech_ids: list[str],
                     encoding: dict[str, tuple[int, int]],
                     orbit_idx: dict[tuple[int, int], int],
                     observed_chi2: float, n_perm: int, seed: int) -> tuple[float, np.ndarray]:
    """Permutation null: reshuffle group->technique assignments, recompute chi2."""
    rng = np.random.default_rng(seed)
    chain_sizes = [(gid, len(c)) for gid, c in chains.items()]
    tech_pool = list(all_tech_ids)
    nulls = np.zeros(n_perm)

    # Baseline histogram (expected) - compute once
    baseline = baseline_tier_histogram(chains, all_tech_ids, encoding, orbit_idx, 50, seed + 1)
    baseline_arr = np.array([baseline.get(t, 0.0) for t in ["0", "1", "2a", "2b"]])
    if baseline_arr.sum() == 0:
        return 1.0, nulls
    baseline_arr = baseline_arr / baseline_arr.sum()

    for p in range(n_perm):
        perm_counts: Counter = Counter()
        for gid, k in chain_sizes:
            if k < 2:
                continue
            idxs = rng.choice(len(tech_pool), size=k, replace=False)
            picks = [(tech_pool[i], encoding[tech_pool[i]]) for i in idxs]
            picks.sort(key=lambda x: (x[1][0], x[1][1]))
            for i in range(k - 1):
                s0 = picks[i][1]
                s1 = picks[i + 1][1]
                perm_counts[classify_tier(s0, s1, orbit_idx)] += 1
        obs_arr = np.array([perm_counts.get(t, 0) for t in ["0", "1", "2a", "2b"]], dtype=float)
        n = obs_arr.sum()
        if n == 0:
            nulls[p] = 0
            continue
        exp_arr = baseline_arr * n
        exp_arr = np.where(exp_arr < 1e-9, 1e-9, exp_arr)
        chi2 = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
        nulls[p] = chi2

    perm_p = float((nulls >= observed_chi2).sum() + 1) / (n_perm + 1)
    return perm_p, nulls


def chi2_tiers(observed: Counter, expected: Counter) -> tuple[float, float, float]:
    """Chi-square on tier distribution. Returns (chi2, p, cohen_w)."""
    tiers = ["0", "1", "2a", "2b"]
    obs = np.array([observed.get(t, 0) for t in tiers], dtype=float)
    exp = np.array([expected.get(t, 0) for t in tiers], dtype=float)
    n = obs.sum()
    if n == 0 or exp.sum() == 0:
        return (0.0, 1.0, 0.0)
    exp = exp * (n / exp.sum())
    exp = np.where(exp < 1e-9, 1e-9, exp)
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    df = len(tiers) - 1
    p = float(1.0 - stats.chi2.cdf(chi2, df=df))
    w = float(np.sqrt(chi2 / n))
    return chi2, p, w


def orbit_family_chi2(encoding: dict[str, tuple[int, int]]) -> tuple[float, float, dict]:
    """H2: chi-square of orbit-family distribution vs uniform."""
    fams = Counter(orbit_family_s9(*be) for be in encoding.values())
    n = sum(fams.values())
    obs = np.array([fams.get("cosmos", 0), fams.get("satellite", 0), fams.get("singularity", 0)], dtype=float)
    # Uniform expected
    exp = np.array([n / 3, n / 3, n / 3])
    exp = np.where(exp < 1e-9, 1e-9, exp)
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    p = float(1.0 - stats.chi2.cdf(chi2, df=2))
    return chi2, p, dict(fams)


# ------------------------------------------------------------------ decision
def classify_outcome(h1_chi2_p: float, h1_perm_p: float, h1_direction_ok: bool,
                     h2_p: float) -> str:
    # Pre-reg rule: "Sign flip at significance counts as NULL. Direction wrong -> NULL."
    # This dominates any H2 signal because the pre-registered hypothesis is
    # about dynamics (H1), not just static distribution (H2).
    if not h1_direction_ok and h1_chi2_p < 0.05:
        return "NULL"
    strong = (h1_chi2_p < 0.001 and h1_perm_p < 0.001 and h1_direction_ok and h2_p < 0.001)
    if strong:
        return "STRONG"
    weak = (h1_chi2_p < 0.05 and h1_direction_ok) or h2_p < 0.05
    if weak:
        return "WEAK"
    return "NULL"


# ------------------------------------------------------------------ main
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 72)
    print("QA Bateson Tiered Reachability × MITRE ATT&CK")
    print("=" * 72)

    # --- 1. data
    bundle = pull_stix()
    parsed = parse_bundle(bundle)
    n_tech = len(parsed["techniques"])
    n_grp = len(parsed["intrusion_sets"])
    n_rel = len(parsed["uses_rels"])
    print(f"[stix] techniques={n_tech}  intrusion-sets={n_grp}  uses-rels={n_rel}")

    # --- 2. encoding
    encoding = encode_techniques(parsed["techniques"])
    print(f"[enc]  encoded {len(encoding)}/{n_tech} techniques into S_9")
    state_usage = Counter(encoding.values())
    print(f"[enc]  occupied states: {len(state_usage)}/{MODULUS*MODULUS}  "
          f"max-collision={max(state_usage.values())}  mean={np.mean(list(state_usage.values())):.2f}")

    # --- 3. orbit infrastructure
    orbits, orbit_idx = enumerate_orbits()
    print(f"[orb]  S_9 decomposes into {len(orbits)} orbits; sizes: "
          f"{sorted(Counter(len(o) for o in orbits).items())}")

    # --- 4. H2: orbit-family distribution
    h2_chi2, h2_p, fam_counts = orbit_family_chi2(encoding)
    print(f"[H2]  orbit-family counts: {fam_counts}")
    print(f"[H2]  chi2={h2_chi2:.2f}  p={h2_p:.4g}")

    # --- 5. build chains
    chains = group_technique_chains(parsed)
    chain_lens = [len(c) for c in chains.values()]
    chains_ge2 = {g: c for g, c in chains.items() if len(c) >= 2}
    print(f"[chain] {len(chains)} groups; {len(chains_ge2)} with len>=2; "
          f"mean_len={np.mean(chain_lens):.1f}  median={np.median(chain_lens):.0f}  max={max(chain_lens)}")

    # --- 6. observed tier histogram
    observed = tier_histogram(chains_ge2, encoding, orbit_idx)
    n_obs = sum(observed.values())
    print(f"[obs]  observed transitions: {n_obs}")
    for t in ["0", "1", "2a", "2b"]:
        print(f"       tier {t}: {observed.get(t, 0)} ({observed.get(t, 0)/max(n_obs,1)*100:.1f}%)")

    # --- 7. baseline
    print(f"[base] computing baseline (200 iters) ...")
    all_tech_ids = [tid for tid in parsed["techniques"] if tid in encoding]
    baseline = baseline_tier_histogram(chains_ge2, all_tech_ids, encoding, orbit_idx, 200, SEED)
    n_base = sum(baseline.values())
    print(f"[base] baseline transitions (avg/iter): {n_base:.1f}")
    for t in ["0", "1", "2a", "2b"]:
        print(f"       tier {t}: {baseline.get(t, 0):.1f} ({baseline.get(t, 0)/max(n_base,1)*100:.1f}%)")

    # --- 8. H1 chi-square
    chi2, p_chi2, w = chi2_tiers(observed, baseline)
    print(f"[H1]  chi2={chi2:.2f}  p={p_chi2:.4g}  Cohen_w={w:.3f}")

    # --- 9. direction check
    low_tier_real = (observed.get("0", 0) + observed.get("1", 0)) / max(n_obs, 1)
    low_tier_base = (baseline.get("0", 0) + baseline.get("1", 0)) / max(n_base, 1)
    direction_ok = low_tier_real > low_tier_base
    print(f"[dir] low-tier (0+1) real={low_tier_real:.4f}  baseline={low_tier_base:.4f}  "
          f"direction_ok={direction_ok}")

    # --- 10. permutation
    print(f"[perm] {N_PERM} label permutations (seed={SEED}) ...")
    perm_p, nulls = permutation_test(chains_ge2, all_tech_ids, encoding, orbit_idx,
                                      chi2, N_PERM, SEED + 2)
    print(f"[perm] permutation p = {perm_p:.4g}  null mean chi2={nulls.mean():.2f}  sd={nulls.std():.2f}")

    # --- 11. decision
    outcome = classify_outcome(p_chi2, perm_p, direction_ok, h2_p)
    print("=" * 72)
    print(f"OUTCOME: {outcome}")
    print("=" * 72)

    # --- 12. per-group csv
    rows = []
    for gid, chain in chains_ge2.items():
        name = parsed["intrusion_sets"][gid]["name"]
        tiers_c: Counter = Counter()
        for i in range(len(chain) - 1):
            tiers_c[classify_tier(encoding[chain[i]], encoding[chain[i + 1]], orbit_idx)] += 1
        rows.append({
            "group_id": gid,
            "group_name": name,
            "n_techniques": len(chain),
            "n_transitions": len(chain) - 1,
            "tier_0": tiers_c.get("0", 0),
            "tier_1": tiers_c.get("1", 0),
            "tier_2a": tiers_c.get("2a", 0),
            "tier_2b": tiers_c.get("2b", 0),
        })
    df = pd.DataFrame(rows).sort_values("n_transitions", ascending=False)
    df.to_csv(RESULTS_DIR / "per_group.csv", index=False)

    # --- 13. summary
    summary = {
        "schema": "qa_bateson_attack_reachability.v1",
        "pre_registration": "results/bateson_attack/PRE_REGISTRATION.md",
        "data_manifest_sha256": json.load(open(MANIFEST_FILE))["sha256"],
        "n_techniques_encoded": len(encoding),
        "n_intrusion_sets": n_grp,
        "n_chains_ge2": len(chains_ge2),
        "n_transitions_observed": n_obs,
        "observed_tier_counts": {t: int(observed.get(t, 0)) for t in ["0", "1", "2a", "2b"]},
        "baseline_tier_counts_per_iter": {t: float(baseline.get(t, 0)) for t in ["0", "1", "2a", "2b"]},
        "orbit_family_counts": {k: int(v) for k, v in fam_counts.items()},
        "H1_chi2": chi2,
        "H1_chi2_p": p_chi2,
        "H1_cohen_w": w,
        "H1_perm_p": perm_p,
        "H1_low_tier_real_frac": low_tier_real,
        "H1_low_tier_baseline_frac": low_tier_base,
        "H1_direction_ok": direction_ok,
        "H2_chi2": h2_chi2,
        "H2_p": h2_p,
        "n_permutations": N_PERM,
        "seed": SEED,
        "thresholds": {
            "STRONG": "H1 p<0.001 perm p<0.001 direction positive AND H2 p<0.001",
            "WEAK": "H1 p<0.05 direction positive OR H2 p<0.05",
            "NULL": "otherwise",
        },
        "outcome": outcome,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # --- 14. plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # tier distribution
        fig, ax = plt.subplots(figsize=(9, 5))
        tiers = ["0", "1", "2a", "2b"]
        x = np.arange(len(tiers))
        real = np.array([observed.get(t, 0) / max(n_obs, 1) for t in tiers])
        base = np.array([baseline.get(t, 0) / max(n_base, 1) for t in tiers])
        ax.bar(x - 0.2, real, 0.4, label="observed (real groups)")
        ax.bar(x + 0.2, base, 0.4, label="baseline (random chains)")
        ax.set_xticks(x)
        ax.set_xticklabels(tiers)
        ax.set_xlabel("tier")
        ax.set_ylabel("fraction of transitions")
        ax.set_title(f"ATT&CK tier distribution — {outcome} (chi2 p={p_chi2:.2g}, perm p={perm_p:.2g})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "tier_distribution.png", dpi=120)
        plt.close(fig)

        # orbit family
        fig, ax = plt.subplots(figsize=(7, 5))
        fams = ["cosmos", "satellite", "singularity"]
        counts = [fam_counts.get(f, 0) for f in fams]
        ax.bar(fams, counts)
        ax.set_ylabel("# techniques")
        ax.set_title(f"ATT&CK orbit family distribution (H2 chi2 p={h2_p:.2g})")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "orbit_family_histogram.png", dpi=120)
        plt.close(fig)
        print(f"[plot] wrote tier_distribution.png + orbit_family_histogram.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")

    print(f"[done] results in {RESULTS_DIR}")
    return outcome


if __name__ == "__main__":
    sys.exit(0 if main() in ("STRONG", "WEAK", "NULL") else 1)
