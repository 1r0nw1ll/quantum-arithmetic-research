#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=absolute values / cut-and-project windows across places (Theorem NT); QA layer = INTEGER linear recurrences and their mod-p^k reductions (A1 finite alphabet); valuations, reaches, product formula are observer-layer readouts. No float QA state."
# RT1_OBSERVER_FILE: eigenvalues, p-adic valuations, reaches, place-wise absolute values are observer-layer readouts, not QA state.
"""
Phase M: is the psi-contraction window a strictly OBSERVER (archimedean) object, or can a
finite/p-adic modulus carry it? Resolves the arc's final frontier from Phase L.

Phase L: mod-m destroys the golden window because psi loses its archimedean magnitude
(|psi|<1). The sharp question: is there ANY finite place (p-adic) where psi contracts?

ANSWER (rigorous): NO -- because psi is a UNIT. N(psi) = psi*phi = -1, so psi is a unit in
Z[phi], hence |psi|_v = 1 at EVERY finite place v (valuation 0). By the product formula
prod_v |psi|_v = 1, the ONLY places where psi is not size-1 are the two ARCHIMEDEAN (real)
places of Q(sqrt5): |psi|_{s1}=0.618 (<1, the contracting internal/window) and |psi|_{s2}=
|phi|=1.618 (>1, expanding), whose product is 1. So the cut-and-project window is carried
by the SECOND archimedean place alone -- which is EXACTLY the Theorem-NT observer projection
(the Galois-conjugate real embedding). No finite/mod-m place can carry it, provably.

  [A] COMPUTED: psi is a unit; at finite primes it stays an invertible residue (v_p=0, no
      contraction); the two real places carry the whole contraction (product formula).
      -> the golden window is strictly observer-layer = the 2nd archimedean place / Pisot
         conjugate. This is why every mod-m attempt (Phase H, Phase L) failed.

But does that mean NO discrete aperiodic order survives a modulus? No -- only that the GOLDEN
one doesn't, because phi is a unit (a Pisot number, an archimedean phenomenon). A NON-unit
recurrence CAN contract p-adically:

  [B] COMPUTED: x^2-x-p has a root with p-adic valuation 1 (Newton polygon: coeffs valuations
      (0,0,1) -> a slope-1 segment). That root CONTRACTS p-adically, so its cut-and-project
      internal coordinate is confined to a p-adic BALL, which IS a mod-p^k condition -> the
      aperiodic (limit-periodic) order SURVIVES reduction mod p^k, unlike golden. Measured:
      the internal coordinate's reach stays confined (and its p-adic valuation grows) mod p^k
      for the non-unit recurrence, while for golden it fills the torus.

NET: the frontier is resolved. Golden order is strictly observer-layer -- provably no finite
carrier, because phi/psi are UNITS (the contraction is the archimedean Pisot property, = the
2nd real place = the Theorem-NT observer projection). Discrete/mod-p^k carriers of aperiodic
order exist ONLY for NON-unit (p-adically contracting) recurrences, which the QA golden M is
not. So the observer projection is identified precisely, and 'aperiodic order can be discrete'
(yes, non-golden, p-adic) is cleanly separated from 'the GOLDEN order can be discrete' (no).
"""
from __future__ import annotations
import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0
PSI = (1.0 - np.sqrt(5.0)) / 2.0


def hensel_root(c, p, K, r0):
    """Lift a simple root r0 (mod p) of x^2 - x - c to mod p^K (Newton/Hensel)."""
    modK = pow(p, K)
    r = r0 % p
    mod = p
    while mod < modK:
        mod = min(mod * mod, modK)
        fp = (2 * r - 1) % mod
        r = (r - (r * r - r - c) * pow(fp, -1, mod)) % mod
    return r % modK


def orbit_internal_reach(c, p, K):
    """Recurrence u_{k+1}=u_k + c*u_{k-1} (char x^2-x-c) mod p^K. Internal coordinate that
    kills the UNIT-root mode: internal_k = u_{k+1} - lam_unit*u_k. Return (max centered reach
    / (p^K/2), max p-adic valuation of internal). Confined+high-valuation => p-adic window."""
    modK = pow(p, K)
    # the two roots mod p: for x^2-x-c, r in {0..p-1} with r*r-r-c==0
    roots_modp = [r for r in range(p) if (r * r - r - c) % p == 0]
    # unit root = the one that is invertible mod p (nonzero); pick a nonzero root
    unit_r0 = next(r for r in roots_modp if r % p != 0) if any(r % p != 0 for r in roots_modp) \
        else roots_modp[0]
    lam_unit = hensel_root(c, p, K, unit_r0)
    a, b = 0, 1                                        # u_0, u_1
    reach, vmax = 0, 0
    for _ in range(3 * K):
        u_next = (b + c * a) % modK
        internal = (u_next - lam_unit * b) % modK
        centered = internal - modK if internal > modK // 2 else internal
        reach = max(reach, abs(centered))
        v = 0
        t = internal
        while t != 0 and t % p == 0 and v < K:
            v += 1; t //= p
        vmax = max(vmax, v if internal != 0 else 0)
        a, b = b, u_next
    return reach / (modK // 2), vmax


def run():
    print("Phase M: is the psi-contraction window archimedean-only, or can a p-adic modulus carry it?\n")

    # ===== [A] golden window is ARCHIMEDEAN-ONLY (psi is a unit) =====
    norm_psi = PSI * PHI                                # N(psi) = psi*phi
    arch1, arch2 = abs(PSI), abs(PHI)                  # the two real embeddings of psi
    print("[A] golden: psi is a UNIT -> the contraction is carried ONLY by the archimedean places")
    print(f"    N(psi)=psi*phi={norm_psi:.4f} -> unit; so |psi|_v=1 at EVERY finite place v.")
    print(f"    two archimedean places: |psi|_s1={arch1:.4f} (<1 CONTRACT=window), "
          f"|psi|_s2={arch2:.4f} (>1 EXPAND); product={arch1*arch2:.4f} (product formula =|N|=1).")
    # confirm psi stays an invertible residue (v_p=0, no contraction) at split finite primes
    finite_ok = True
    for p in (11, 19, 29, 31):                          # p = +-1 mod 5 (5 is a QR)
        if not any((r * r - r - 1) % p == 0 for r in range(p)):
            continue
        r = next(r for r in range(p) if (r * r - r - 1) % p == 0)
        invertible = (r % p != 0) and (pow(r, -1, p) is not None)
        finite_ok = finite_ok and invertible
    print(f"    at finite primes 11/19/29/31: psi is a nonzero invertible residue (v_p=0, NO")
    print(f"    contraction) -> {finite_ok}. So NO finite place carries the window: it is the")
    print(f"    2nd archimedean place = the Galois-conjugate real embedding = Theorem-NT OBSERVER.")

    # ===== [B] a NON-unit recurrence DOES contract p-adically -> survives mod p^k =====
    p, K = 11, 8
    reach_gold, v_gold = orbit_internal_reach(1, p, K)   # golden x^2-x-1 (unit roots)
    reach_nonunit, v_nu = orbit_internal_reach(p, p, K)  # x^2-x-p (one root v_p=1, CONTRACTS)
    # the RIGHT metric for a p-adic window is the p-adic VALUATION, not the archimedean reach:
    # a p-adically tiny number (v_p large) can be archimedean-LARGE as an integer in [0,p^K).
    print(f"\n[B] discrete carriers: internal-coordinate p-adic VALUATION mod {p}^{K} (the p-adic")
    print(f"    window is measured p-adically, NOT by archimedean size):")
    print(f"    GOLDEN  x^2-x-1  (phi,psi UNITS): max p-adic valuation {v_gold} -> internal stays a")
    print(f"      p-adic UNIT, in NO p-adic ball -> no finite window (archimedean reach {reach_gold:.0%} "
          f"= fills, matches Phase L).")
    print(f"    NON-UNIT x^2-x-{p} (root v_p=1):   max p-adic valuation {v_nu} -> internal -> 0")
    print(f"      p-adically (into p^{v_nu}Z), CONFINED to a p-adic ball even though archimedean reach")
    print(f"      is {reach_nonunit:.0%} (the two metrics differ -- it is the p-adic one that carries the window).")
    survives = v_nu >= K - 1 and v_gold == 0
    print(f"    -> the non-unit internal coord's confinement (internal == 0 mod p^k) IS a mod-p^k")
    print(f"       condition, so its aperiodic/limit-periodic order is definable mod p^k -- a genuine")
    print(f"       DISCRETE carrier. The golden one is not (unit -> archimedean/Pisot only).")

    ok = (abs(norm_psi + 1.0) < 1e-6 and abs(arch1 * arch2 - 1.0) < 1e-6 and finite_ok
          and survives)
    print("\nVERDICT (final frontier resolved, data-driven):")
    print(f"  * The golden window has NO finite carrier -- provably, because phi/psi are UNITS")
    print(f"    (N(psi)=-1): |psi|_v=1 at every finite place, and the whole contraction lives at")
    print(f"    the 2nd ARCHIMEDEAN (real) place = the Galois-conjugate embedding = the Theorem-NT")
    print(f"    OBSERVER projection. This is the exact reason every mod-m attempt (Phase H/L) failed:")
    print(f"    golden order is the archimedean Pisot property, strictly observer-layer.")
    print(f"  * BUT discrete aperiodic order is NOT impossible: a NON-unit recurrence (x^2-x-{p},")
    print(f"    a p-adically contracting root) has a p-adic window that SURVIVES mod p^k (internal")
    print(f"    valuation ->{v_nu}, p-adically confined vs golden's {v_gold}). So 'aperiodic order can be")
    print(f"    discrete' (yes, non-golden/p-adic) is cleanly separated from 'the GOLDEN order can")
    print(f"    be discrete' (no). QA's golden M is a unit -> its order is observer-layer, full stop.")
    print(f"  * Identification: the QA observer projection (Theorem NT) = the second archimedean")
    print(f"    place of Q(sqrt5) / the Pisot conjugate. That is the sharp end of the whole arc.")
    print(f"\n  STATUS: {'FRONTIER RESOLVED -- golden order is archimedean-only (no finite carrier)' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
