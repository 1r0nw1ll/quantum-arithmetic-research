#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=continuous-torus + statistical readouts (Theorem NT); the QA layer is the integer golden element M on (Z/mZ)^2 (A1). The chaos lives in the observer-layer continuous limit; randomness diagnostics are observer readouts, never QA state."
# RT1_OBSERVER_FILE: FFT / entropy / Lyapunov / compression are observer-layer statistical readouts, not QA state.
"""
Phase A of the "apparent randomness from deterministic law" program.

QA's golden element M = [[0,1],[1,1]] (the T-operator (b,e)->(e,b+e); M^2 = M + I,
eigenvalues phi, psi) is a HYPERBOLIC toral automorphism -- the Fibonacci cat map.
Two faces of the SAME deterministic generator:

  * CONTINUOUS (observer-layer, on the real 2-torus [0,1)^2): a textbook deterministic-
    CHAOS system -- mixing, positive Lyapunov = ln(phi) ~ 0.481/step, measure-theoretically
    a BERNOULLI shift. Its coarse-grained bit stream is indistinguishable from i.i.d.
    randomness by standard tests, yet fully deterministic, exactly reconstructible, and
    exponentially sensitive to initial conditions.

  * DISCRETE (the QA layer, on (Z/mZ)^2, A1 {1..m}): finite -> eventually periodic
    (Pisano period). The single-coordinate readout is Fibonacci mod m -- a LINEAR
    recurrence with tiny linear complexity, hence PREDICTABLE and NOT random-looking.
    Apparent randomness needs the full mixing / a nonlinear readout.

The discriminator between "predictable-linear determinism" and "random-looking
deterministic chaos" is LINEAR COMPLEXITY (Berlekamp-Massey): the Fibonacci stream has
LC ~ 2 (a degree-2 LFSR); the cat-map bit stream has LC ~ n/2 (like true random).

Point: the golden deterministic law MANUFACTURES apparent randomness (in the chaotic
limit) -- a concrete, in-QA counterexample to the inference "looks random => is
ontically random", the same inference the Stosszahlansatz / Gibbs foundations assume.
No claim that QA is random; the claim is that determinism can be statistically
indistinguishable from randomness, so randomness must be argued, not assumed.
"""
from __future__ import annotations
import zlib
import numpy as np

PHI = (1 + 5 ** 0.5) / 2
M = np.array([[0, 1], [1, 1]], dtype=np.float64)   # golden element / Fibonacci cat map


# ---- continuous Fibonacci cat map on the 2-torus (observer-layer) ----
def catmap_bits(x0, y0, n, drop=100):
    """Deterministic bit stream from the continuous cat map: x_{k+1},y_{k+1} =
    (y, x+y) mod 1; emit bit = [x >= 0.5]. Chaotic, Bernoulli in the limit."""
    x, y = x0, y0
    bits = np.empty(n, dtype=np.int8)
    for _ in range(drop):
        x, y = y % 1.0, (x + y) % 1.0
    for k in range(n):
        x, y = y % 1.0, (x + y) % 1.0
        bits[k] = 1 if x >= 0.5 else 0
    return bits


# ---- discrete QA golden orbit on (Z/mZ)^2 (the QA layer) ----
def fib_mod_m(seed_b, seed_e, m, n):
    """QA golden iteration (b,e)->(e, qa_mod(b+e)) on the A1 alphabet {1..m}
    (qa_mod(x)=((x-1)%m)+1, never 0); return the b-coordinate stream -- the A1
    Fibonacci recurrence, deterministic and LINEAR/affine."""
    def qa_mod(x):
        return ((x - 1) % m) + 1
    b, e = qa_mod(seed_b), qa_mod(seed_e)
    out = np.empty(n, dtype=np.int64)
    for k in range(n):
        out[k] = b
        b, e = e, qa_mod(b + e)
    return out


# ---- randomness / determinism diagnostics (observer-layer) ----
def shannon_bits(bits):
    p1 = bits.mean()
    return 0.0 if p1 in (0.0, 1.0) else float(-p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1))


def spectral_flatness(bits):
    x = bits.astype(float) - bits.mean()
    ps = np.abs(np.fft.rfft(x)) ** 2
    ps = ps[1:]
    ps = ps[ps > 0]
    return float(np.exp(np.mean(np.log(ps))) / np.mean(ps)) if ps.size else 0.0


def max_autocorr(bits, kmax=64):
    x = bits.astype(float) - bits.mean()
    denom = np.dot(x, x) + 1e-12
    return float(max(abs(np.dot(x[:-k], x[k:]) / denom) for k in range(1, kmax)))


def compress_ratio(bits):
    packed = np.packbits(bits.astype(np.uint8))
    return len(zlib.compress(packed.tobytes(), 9)) / max(len(packed.tobytes()), 1)


def linear_complexity(bits):
    """Berlekamp-Massey linear complexity of a GF(2) sequence (textbook, pure ints)."""
    s = [int(x) & 1 for x in bits]
    n = len(s)
    b = [0] * n; c = [0] * n
    b[0] = c[0] = 1
    L, m = 0, -1
    for N in range(n):
        d = s[N]
        for i in range(1, L + 1):
            d ^= c[i] & s[N - i]
        if d:
            t = c[:]
            for i in range(n - (N - m)):
                c[N - m + i] ^= b[i]
            if L <= N // 2:
                L, m, b = N + 1 - L, N, t
    return L


def divergence_lyapunov(x0, y0, n=300, eps0=1e-9):
    """Benettin estimate of the Lyapunov exponent: grow a tiny separation eps0, add the
    per-step log growth log(d/eps0), then RENORMALIZE the perturbed point back to eps0
    along the separation direction each step (so it never saturates/wraps)."""
    x, y = x0, y0
    xp, yp = x0 + eps0, y0
    total, steps = 0.0, 0
    for _ in range(n):
        x, y = y % 1.0, (x + y) % 1.0
        xp, yp = yp % 1.0, (xp + yp) % 1.0
        dx, dy = xp - x, yp - y
        d = (dx * dx + dy * dy) ** 0.5
        if d == 0:
            continue
        total += np.log(d / eps0)
        xp, yp = x + dx * (eps0 / d), y + dy * (eps0 / d)   # renormalize to eps0
        steps += 1
    return total / steps if steps else 0.0


def report(name, bits):
    print(f"{name:22} H={shannon_bits(bits):.3f}  flatness={spectral_flatness(bits):.3f}  "
          f"|autocorr|max={max_autocorr(bits):.3f}  zlib={compress_ratio(bits):.3f}  "
          f"LC={linear_complexity(bits[:512])}/512")


def run():
    n = 20000
    eigs = np.linalg.eigvals(M)
    print("QA golden element M = [[0,1],[1,1]]  (M^2 = M + I)")
    print(f"  eigenvalues {eigs[0]:+.4f}, {eigs[1]:+.4f}  (phi={PHI:.4f}, psi={1-PHI:+.4f}); "
          f"hyperbolic -> Fibonacci cat map")
    lyap = np.log(PHI)
    print(f"  Lyapunov (continuous chaos) = ln(phi) = {lyap:.4f}/step; "
          f"empirical estimate = {divergence_lyapunov(0.1234, 0.5678):.3f}\n")

    print("Randomness diagnostics -- compare each row to the i.i.d. reference below")
    print("(H->1, raw-periodogram flatness-> exp(-gamma)~0.56 [the white value, NOT 1],")
    print(" |autocorr|->0, zlib->1, LC->n/2):")
    cat = catmap_bits(0.31415926, 0.27182818, n)
    report("cat map (continuous)", cat)
    # discrete QA golden orbit, LINEAR coordinate readout (Fibonacci mod 2):
    fib_bits = (fib_mod_m(1, 1, 2**16, n) & 1).astype(np.int8)
    report("QA Fibonacci mod 2", fib_bits)
    # an i.i.d. reference:
    report("true random (ref)", np.random.default_rng(0).integers(0, 2, n).astype(np.int8))

    print("\nDeterminism (both streams):")
    c2 = catmap_bits(0.31415926, 0.27182818, n)
    print(f"  exact reconstruction (same seed -> identical stream): {np.array_equal(cat, c2)}")
    near = catmap_bits(0.31415926 + 1e-9, 0.27182818, n)
    hd = np.mean(cat[:2000] != near[:2000])
    print(f"  sensitive dependence: seed +1e-9 -> bit-disagreement after mixing = {hd:.3f} "
          f"(~0.5 = fully decorrelated)")

    print("\nCONCLUSION:")
    print("  The continuous golden cat map is DETERMINISTIC yet statistically indistinguishable")
    print("  from the i.i.d. REFERENCE on every diagnostic (H~1, spectral flatness at the white")
    print("  value ~0.56=exp(-gamma), no autocorrelation, incompressible, LC~n/2) AND exactly")
    print("  reconstructible + exponentially sensitive (Lyapunov ln phi).")
    print("  The DISCRETE QA linear readout (Fibonacci mod m) is deterministic AND predictable")
    print("  (LC~2, a degree-2 LFSR) -- so 'looks random' is not automatic: linear complexity")
    print("  separates predictable-linear determinism from chaos. Either way, randomness is a")
    print("  property one must DEMONSTRATE, not an axiom -- the golden generator manufactures")
    print("  apparent randomness from deterministic law. (Phase B: does the cat-map spectrum")
    print("  match SED's zero-point-field omega^3 density?)")


if __name__ == "__main__":
    run()
