# [456] QA Witt Tower Eisenstein Form Real-Data Certificate

## Claim

Tests the Eisenstein form f(b,e) = b²+be−e² (the Z[φ] norm from cert [214]) on real ^GSPC monthly rank-bin state pairs (N=299, same QA mapping as certs [453]–[455]). Documents four structural properties of the Eisenstein form on real market data, including an honest null for the predictive signal.

## Algebraic Foundation

From cert [214]: f(e, b+e) = −f(b,e) for all integers b, e. This is the Z[φ] norm sign-flip under the T-operator (b,e)→(e,b+e). Along any QA T-trajectory the form alternates: +, −, +, −, ...

**Verified on all 299 real bin pairs**: f(e,b+e)+f(b,e)=0 for every actual market bin pair — algebraic, trivially true, confirmed on integer data.

## Distribution Asymmetry

| Sign | Count | Fraction | Interpretation |
|---|---|---|---|
| f > 0 | 207 | 69.2% | b dominates: "past" return was strong relative to "present" |
| f < 0 | 89 | 29.8% | e dominates: "present" return was strong |
| f = 0 | **3** | **1.0%** | Only at b=e=0 (both consecutive months hit absolute bottom bin) |

The positive bias (69% f>0) reflects the bull-market drift: in a 25-year window dominated by uptrends, the current return bin e tends to sit near or below the previous bin b in relative rank, producing f>0 more often.

The 3 f=0 cases are crash-clustering signatures — months where two consecutive monthly returns both rank at the absolute bottom of the 25-year distribution (bin 0). These are extreme co-crash events.

## Sign-Flip Rate

**Sign changes per transition: 148/298 = 49.7%**

Under the T-step, f always flips. In real market data, the sign changes approximately every other month — consistent with the T-step alternation pattern, though the market doesn't follow T-trajectories exactly.

## f<0 Transience

**P(f_{t+1}<0 | f_t<0) = 13/88 = 14.8%**

f<0 states revert to f>0 with 85% probability within one month. This mirrors the S-orbit transience from cert [455]: both the Singularity orbit and the f<0 Eisenstein state are unstable, transient configurations that the market quickly exits.

The f-sign Markov structure:

| From \ To | f>0 | f<0 |
|---|---|---|
| **f>0** | 131/204 = 64% | 73/204 = 36% |
| **f<0** | 75/88 = 85% | 13/88 = **15%** |

f>0 is the dominant attractor state (64% self-transition), just as Cosmos orbit is the dominant attractor in cert [455].

## Predictive Null (Key Result)

**perm_p(two-tail) = 0.752** — f sign does NOT predict next-month returns.

| f-sign at t | Next-return positive | Mean next return |
|---|---|---|
| f > 0 | 63.3% | +0.58% |
| f < 0 | 67.0% | +0.74% |

The difference (0.74%−0.58%=0.16%) is not statistically significant (perm_p=0.752). The Eisenstein form is a **contemporaneous state descriptor**, not a predictor. It classifies where the market currently is in Z[φ] norm space — it does not tell you where it is going.

## S-Orbit Eisenstein Values

S-orbit pairs (b%9=0 and e%9=0) appear with Eisenstein values: {−81, 0, 81, 324, 405}

| Pair (b,e) | f(b,e) |
|---|---|
| (0,0) | 0 |
| (9,18) | −81 |
| (9,0) or (9,9) | 81 |
| (18,18) | 324 |
| (18,9) or (9,9 next) | 405 or 81 |

S-orbit class does not determine the Eisenstein sign. The orbit class (divisibility by 9) and the Eisenstein form (golden-ratio norm) are independent properties of the state pair.

## Connection to Prior Certs

- **[214]**: Algebraic identity f(e,b+e)=−f(b,e) — the foundation this cert verifies on real data
- **[455]**: f<0 transience parallels S-orbit transience; both are unstable states with high reversion probability
- **[453]**: S-orbit recession concentration — the recession signal is in orbit class, not Eisenstein sign
- **[110]**: Witt tower structural parent

## What This Cert Establishes

The Eisenstein form is part of the QA mathematical structure (from cert [214]) but its SIGN does not carry a market prediction signal on monthly S&P 500 data. The form captures a different aspect of state-pair geometry — the relative balance between "past" and "present" returns — that is structurally interesting (asymmetric distribution, reversion dynamics) but orthogonal to the recession prediction captured in cert [453].

This is an honest null: the Eisenstein form does not predict next-month returns. The orbit class (which cert [453] uses) is the relevant predictive feature.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: T-step identity | f(e,b+e)+f(b,e)=0 for all N=299 pairs | PASS |
| C2: Distribution asymmetry | n_pos/N=69.2% ∈ [0.55,0.80]; n_zero=3 ≤ 5 | PASS |
| C3: Sign-flip rate | 49.7% ∈ [0.40,0.60] (T-step alternation pattern) | PASS |
| C4: f<0 transience | P(persist f<0)=14.8% ≤ 0.30 | PASS |
| C5: Predictive null | perm_p=0.752 ≥ 0.10 | PASS |
| C6: S-orbit coverage | S-orbit spans both signs {−81, 0, 81, 324, 405} | PASS |

## Primary Sources

- Wall, H. S. (1960). doi:10.1080/00029890.1960.11989541
- Cert [214]: QA Eisenstein form f(e,b+e)=−f(b,e) (algebraic identity)

## Related Certs

- [214] QA Eisenstein Form (algebraic parent)
- [110] QA Witt Tower Framework (structural parent)
- [453] QA Witt Tower Orbit Recession Predictor (orbit class IS the signal; Eisenstein sign is not)
- [455] QA Witt Tower Orbit Transition Markov Chain (f<0 and S-orbit share transience structure)
