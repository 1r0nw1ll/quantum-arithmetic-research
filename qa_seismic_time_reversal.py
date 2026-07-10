#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=geometry_and_velocity_to_QA_phase; qa_neg = time reversal on integer phase mod m; coherent-sum focal field is observer-layer (Theorem NT)"
"""
QA Seismic Time-Reversal Focusing — the [522] operator on realistic wave physics.

The fair strong-source test of cert [522]: a receiver array records a monochromatic
source through a HETEROGENEOUS (scattering) medium; TIME-REVERSE the records
(qa_neg = phase conjugation) and back-propagate through the SAME medium -> the field
refocuses on the source. Green's function per path = amplitude/r * exp(2pi i * QA
phase / m) with phase = quantize(k * travel-length); the medium = M point
scatterers adding single-scatter paths (source->scatterer->receiver).

Three falsifiable claims (Fink's time-reversal-mirror physics):
  (A) REFOCUS       the back-propagated field peaks at the true source location;
  (B) SPECIFICITY   re-emitting through a DIFFERENT medium (moved scatterers)
                    destroys the focus (the [518]/[522] same-medium fingerprint);
  (C) SUPER-RESOLUTION  a SCATTERING medium focuses TIGHTER than a homogeneous one
                    (scatterers enlarge the effective aperture) -- Fink-style
                    aperture enhancement (Fink 2000). This is the STRONG
                    source+known-medium regime EEG lacked (shown on a fixed-seed
                    toy geometry, not claimed beyond it).

Theorem NT: geometry/velocity/scatterers are observer inputs (crossed once, in);
qa_neg time reversal is integer phase arithmetic mod m; the coherent-sum focal
field magnitude is an observer readout (crossed once, out).

FINDINGS (32-receiver array, aperture 16, k=6, source at (0,8)) -- honest:
  (A) REFOCUS works: back-propagation localizes the source -- the scattering-medium
      peak is at x=-0.03 and the homogeneous-medium peak is exactly x=+0.000 (both
      within the grid step of the source). The [522] refocus survives realistic
      geometry.
  (C) SUPER-RESOLUTION visible in this fixed-seed demo: n~8 scatterers focus TIGHTER
      than homogeneous (FWHM 2.69 vs 3.22, ratio ~0.84 -- Fink-style aperture
      enhancement); heavy scattering (n>=20) decoheres into noise and degrades it
      (not a claim beyond this toy geometry/seed).
  (B) SPECIFICITY is WEAK in this light-scattering regime: the direct path refocuses
      on its own, so a mismatched medium still focuses near the source. Strong
      same-medium specificity would need heavy MULTIPLE scattering -- which the
      coarse QA mod-24 phase decoheres. Honest tension: specificity wants strong
      scattering, QA-24 resolution wants weak scattering.
  LIMIT: the QA mod-24 observer projection caps spatial resolution -- the focal
      FWHM (~3.2) is ~6x the diffraction limit (~0.52 for this geometry); the
      refocus LOCATION is correct but the coarse phase quantization broadens the
      spot. STRONGER than EEG (no clean source/medium there): here refocus and
      super-resolution work, with honest quantization and specificity caveats.
"""
from __future__ import annotations
import numpy as np

M = 24


def qa_mod(x):
    return ((np.asarray(x, np.int64) - 1) % M) + 1


def qa_neg(a):
    return qa_mod(-np.asarray(a, np.int64))


def _phase(dist, k):
    """Observer projection: travel-length -> integer QA phase in {1..m}."""
    return qa_mod(np.rint(k * dist).astype(np.int64))


def green(a, b, scatterers, k, scat_amp=0.5):
    """Monochromatic Green's function a->b: direct path + single-scatter paths.
    Returns a complex phasor with QA-quantized phases (amplitude ~ 1/r observer)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    r = np.hypot(*(a - b)) + 1e-9
    g = (1.0 / r) * np.exp(2j * np.pi * (int(_phase(r, k)) - 1) / M)          # direct
    for s in scatterers:
        s = np.asarray(s, float)
        r1 = np.hypot(*(a - s)) + 1e-9
        r2 = np.hypot(*(s - b)) + 1e-9
        ph = int(_phase(r1 + r2, k))
        g += scat_amp / (r1 * r2) * np.exp(2j * np.pi * (ph - 1) / M)          # scattered
    return g


def focal_line(source, receivers, scatterers_rec, scatterers_reemit, k, xs, y):
    """Record source at the array through scatterers_rec, time-reverse (qa_neg the
    phase = complex conjugate), re-emit through scatterers_reemit; return |field|
    along the line y=const over xs (normalized)."""
    R = np.array([green(source, rx, scatterers_rec, k) for rx in receivers])
    # time reversal = phase conjugation: qa_neg on the QA phase == complex conjugate
    R_tr = np.conj(R)
    field = []
    for x in xs:
        loc = (x, y)
        gvec = np.array([green(rx, loc, scatterers_reemit, k) for rx in receivers])
        field.append(abs(np.sum(R_tr * gvec)))
    f = np.array(field)
    return f / f.max()


def fwhm(xs, f):
    """Full-width at half-maximum of the focal peak (in x units)."""
    half = 0.5
    above = np.where(f >= half)[0]
    if len(above) < 2:
        return np.nan
    return xs[above[-1]] - xs[above[0]]


def run():
    rng = np.random.default_rng(42)
    k = 6.0                                   # wavenumber (carrier); wavelength ~ 1.05
    src = (0.0, 8.0)                          # source, in front of the array
    N = 32
    receivers = [(x, 0.0) for x in np.linspace(-8, 8, N)]      # line array, aperture 16
    xs = np.linspace(-4, 4, 801)

    # scattering medium: a MODERATE number of point scatterers (super-resolution
    # regime; heavy scattering decoheres into noise -- an honest limit)
    scat = [(float(rng.uniform(-10, 10)), float(rng.uniform(1, 7))) for _ in range(8)]
    scat_alt = [(float(rng.uniform(-10, 10)), float(rng.uniform(1, 7))) for _ in range(8)]

    print(f"QA SEISMIC TIME-REVERSAL FOCUSING  (m={M}, {N}-receiver array, source at {src})\n")

    # [A] REFOCUS through the scattering medium
    f_scat = focal_line(src, receivers, scat, scat, k, xs, src[1])
    peak_x = xs[int(np.argmax(f_scat))]
    print("[A] Refocus through the scattering medium:")
    print(f"  focal peak at x={peak_x:+.2f} (source x={src[0]:+.2f}) -> localizes source: {abs(peak_x-src[0])<0.2}")

    # [B] SAME-MEDIUM SPECIFICITY: re-emit through a DIFFERENT medium
    f_mis = focal_line(src, receivers, scat, scat_alt, k, xs, src[1])
    print("\n[B] Same-medium specificity (re-emit through matched vs moved scatterers):")
    print(f"  matched-medium peak {f_scat.max():.3f} (=1) at x={peak_x:+.2f}; "
          f"mismatched-medium peak-to-side ratio {f_mis.max()/np.median(f_mis):.2f} vs "
          f"matched {f_scat.max()/np.median(f_scat):.2f}")
    print(f"  mismatched focus at x={xs[int(np.argmax(f_mis))]:+.2f} "
          f"(scattered/degraded: {abs(xs[int(np.argmax(f_mis))]-src[0])>0.2 or f_mis.max()/np.median(f_mis) < 0.6*f_scat.max()/np.median(f_scat)})")

    # [C] SUPER-RESOLUTION: scattering medium focuses tighter than homogeneous
    f_homo = focal_line(src, receivers, [], [], k, xs, src[1])
    w_scat, w_homo = fwhm(xs, f_scat), fwhm(xs, f_homo)
    print("\n[C] Super-resolution (focal width, scattering vs homogeneous medium):")
    print(f"  homogeneous-medium FWHM: {w_homo:.3f}")
    print(f"  scattering-medium  FWHM: {w_scat:.3f}")
    if not (np.isnan(w_scat) or np.isnan(w_homo)):
        print(f"  scattering focuses {'TIGHTER' if w_scat < w_homo else 'wider'} "
              f"(ratio {w_scat/w_homo:.2f})  -- Fink-style aperture enhancement "
              f"(this fixed-seed demo): {w_scat < w_homo}")
    print("\nTime reversal (qa_neg) refocuses ON THE SOURCE and (this fixed-seed demo)")
    print("scattering improves resolution -- Fink-style aperture enhancement. Same-medium")
    print("specificity is weak in this light-scattering regime (direct path dominates);")
    print("QA mod-24 caps focal width (~6x the diffraction limit).")


if __name__ == "__main__":
    run()
