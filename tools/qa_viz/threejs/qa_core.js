// qa_core.js — shared QA primitives for Three.js viz, mirroring qa_orbit_rules.py.
// Single source of truth; both qa_torus.html and qa_nucleus.html import from here.

export const MOD = 9;
export const SAT_DIV = MOD / 3;

// A1-compliant mod in {1..MOD}. Matches qa_orbit_rules.py:qa_step.
export function qa_mod(n) {
  return ((n - 1) % MOD + MOD) % MOD + 1;
}

// Digital root in {1..9}.
export function dr(n) {
  n = Math.abs(n);
  return n === 0 ? MOD : (n % MOD || MOD);
}

// Canonical generators (per docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md).
//   σ (sigma, "shift"):   (b,e) → (e, qa_mod(b+e))      — the qa_step Fibonacci dynamic
//   μ (mu,    "swap"):    (b,e) → (e, b)
//   ν (nu,    "halve"):   (b,e) → (qa_mod(5b), qa_mod(5e))  (5 = 2⁻¹ mod 9)
// σ is THE orbit generator. μ, ν are structural operators for reachability experiments.
export function sigma(b, e) { return [e, qa_mod(b + e)]; }
export function mu(b, e)    { return [e, b]; }
export function nu(b, e)    { return [qa_mod(5 * b), qa_mod(5 * e)]; }

export const qa_step = sigma;  // alias for back-compat

// norm_f(b,e) = b² + be − e² — the Z[φ] quadratic norm; see qa_orbit_rules.py:norm_f.
// Structural identity: (3|b ∧ 3|e) ⇔ 9|norm_f, so v3(norm_f) ∈ {0} ∪ {≥2}, never 1.
// This is why Satellite is defined by 3∣b ∧ 3∣e (not by b=e).
export function norm_f(b, e) { return b * b + b * e - e * e; }

// Third-coordinate completion (b,e) → D₉³. Two parameterizations live on the same base:
//   qa_coord_d(b,e)   = qa_mod(b + e)        — QA Fibonacci forward (our canonical d)
//   drth_coord_A(b,e) = qa_mod(2e − b)       — DRTH superplane-A affine continuation
// See docs/theory/QA_DRTH_MAPPING.md §2 for the parameterization derivation i=b, q=e⊖b.
// Linear identity: qa_coord_d + drth_coord_A ≡ 3e (mod 9) — linear in e, not a conservation law.
export function qa_coord_d(b, e)   { return qa_mod(b + e); }
export function drth_coord_A(b, e) { return qa_mod(2 * e - b); }

// Archetype map per DRTH §2.3: archetype(x) labels Chase's Γ_G={1,4,7}=1, Γ_R={2,5,8}=2, Γ_B={3,6,9}=3.
// These are the three cosets of the 3-ideal in (D₉, ⊕) = Z₉/3Z.
export function archetype(x) { return ((x - 1) % 3) + 1; }

// 1D rotor/stator split (DRTH §2.2). Satellite ⇔ is_stator(b) ∧ is_stator(e) (for our 2D classifier).
export function is_stator(x) { return x % SAT_DIV === 0; }
export function is_rotor(x)  { return x % SAT_DIV !== 0; }

export function v3(n) {
  if (n === 0) return Infinity;
  n = Math.abs(n);
  let v = 0;
  while (n % 3 === 0) { n = n / 3 | 0; v++; }
  return v;
}

// Orbit classification (canonical).
export function classify(b, e) {
  if (b === MOD && e === MOD) return 'singularity';
  if (b % SAT_DIV === 0 && e % SAT_DIV === 0) return 'satellite';
  return 'cosmos';
}

// Orbit period from seed (b0,e0) under σ.
export function orbit_period(b0, e0, cap = MOD * MOD + 1) {
  let b = b0, e = e0;
  const seen = new Set();
  for (let i = 0; i < cap; i++) {
    const key = b * 100 + e;
    if (seen.has(key)) return seen.size;
    seen.add(key);
    [b, e] = sigma(b, e);
  }
  return seen.size;
}

// σ-only orbit length = length of the cycle (b,e) → σ(b,e) → σ²(b,e) → ….
// On mod 9 (inert Case A of QA Orbit Theorem Part II, since (5|3) = −1), orbit
// length is uniform per class: 24 on Cosmos (= π(9)), 8 on Satellite (= π(3)),
// 1 on Singularity. See docs/theory/QA_ORBIT_THEOREM_SYNTHESIS.md.
export function sigma_orbit_length(b0, e0, cap = MOD * MOD + 1) {
  let b = b0, e = e0;
  for (let k = 1; k <= cap; k++) {
    [b, e] = sigma(b, e);
    if (b === b0 && e === e0) return k;
  }
  return -1;
}

// Partition the 72 Cosmos states into their 3 period-24 orbits under σ.
// Returns { orbitId: Map('b,e' → 0|1|2), orbits: Array<Array<[b,e]>> }.
function _partitionCosmos() {
  const orbitId = new Map();
  const orbits = [];
  for (let b = 1; b <= MOD; b++) {
    for (let e = 1; e <= MOD; e++) {
      if (classify(b, e) !== 'cosmos') continue;
      const key = b + ',' + e;
      if (orbitId.has(key)) continue;
      const idx = orbits.length;
      const orbit = [];
      let cb = b, ce = e;
      while (!orbitId.has(cb + ',' + ce)) {
        orbitId.set(cb + ',' + ce, idx);
        orbit.push([cb, ce]);
        [cb, ce] = sigma(cb, ce);
      }
      orbits.push(orbit);
    }
  }
  return { orbitId, orbits };
}
export const COSMOS_PARTITION = _partitionCosmos();

export function cosmos_orbit_id(b, e) {
  return COSMOS_PARTITION.orbitId.get(b + ',' + e);  // 0, 1, 2, or undefined
}

// Palettes
export const PALETTE_CLASS = {
  cosmos:      0x4a90e2,
  satellite:   0xe25b5b,
  singularity: 0xf2c14e,
};

// 3 distinct cosmos-orbit colors (distinguishable on dark bg).
export const PALETTE_COSMOS_ORBITS = [
  0x5fa8ff,  // orbit 0: bright blue
  0xc077ff,  // orbit 1: purple
  0x50e0b8,  // orbit 2: teal
];

// Archetype palette (DRTH §2.3 cosets Γ_G / Γ_R / Γ_B).
// Distinct from PALETTE_CLASS to avoid visual collision.
export const PALETTE_ARCHETYPE = {
  1: 0x9ee27a,  // Γ_G (1,4,7): soft green
  2: 0xffa94d,  // Γ_R (2,5,8): amber
  3: 0xb78dff,  // Γ_B (3,6,9): lavender
};

// Choose color for a (b,e) given active color mode.
//   'class'     — one color per orbit class (default)
//   'orbit'     — color Cosmos by which of the 3 period-24 orbits it lives on
//   'archetype' — color by archetype(d) where d = qa_mod(b+e); 3 bands per DRTH cosets
export function pickColor(b, e, mode = 'class') {
  const cls = classify(b, e);
  if (mode === 'orbit' && cls === 'cosmos') {
    return PALETTE_COSMOS_ORBITS[cosmos_orbit_id(b, e)];
  }
  if (mode === 'archetype') {
    const d = qa_mod(b + e);
    return PALETTE_ARCHETYPE[archetype(d)];
  }
  return PALETTE_CLASS[cls];
}
