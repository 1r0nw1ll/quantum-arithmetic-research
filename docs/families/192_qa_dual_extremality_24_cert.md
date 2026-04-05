# Family [192] QA_DUAL_EXTREMALITY_24_CERT.v1

## One-line summary

$m = 24$ is simultaneously the minimum non-trivial Pisano-period fixed point AND the maximum Carmichael-$\lambda=2$ modulus. $\pi(9) = 24$ bridges QA's theoretical modulus to its applied modulus in one Pisano step. This closes the Level-III self-improvement fixed point question from cert [191].

## Mathematical content

### The Pisano operator as a natural Level-III operator

Define $\pi: \mathbb{N} \to \mathbb{N}$ by $\pi(m) = $ length of the Fibonacci sequence modulo $m$ before returning to $(0, 1)$. This is literally the orbit length of the QA dynamic $T(b, e) = (e, b + e \bmod m)$ restricted to the Fibonacci seed. Under $\pi$, a modulus is "improved" to the length of its own Fibonacci orbit — a natural self-referential operation on moduli.

### The three facts

**1. Pisano fixed points** (OEIS [A235702](https://oeis.org/A235702)):
$$\{m > 1 : \pi(m) = m\} = \{24, 120, 600, 3000, 15000, \ldots\} = \{24 \cdot 5^k : k \geq 0\}$$
The minimum non-trivial Pisano fixed point is $\boxed{m = 24}$.

**2. Carmichael $\lambda = 2$ set**:
$$\{m : \lambda(m) = 2\} = \{3, 4, 6, 8, 12, 24\}$$
The maximum is $\boxed{m = 24}$. Structural proof: if $\lambda(m) = 2$ then every prime factor of $m$ is in $\{2, 3\}$ (otherwise $p - 1 \geq 4 \mid \lambda(m)$), $v_2(m) \leq 3$ (otherwise $\lambda(2^k) = 2^{k-2} \geq 4$), and $v_3(m) \leq 1$ (otherwise $\lambda(9) = 6$). Hence $m \mid 24$, and the set is exactly the divisors of 24 with $m \geq 3$.

**3. The QA bridge**: $\pi(9) = 24$. Our theoretical modulus maps to our applied modulus in exactly one Pisano step. The basin of attraction of 24 under $\pi$ in $[1, 30]$ is
$$\pi^{-1}(24) \cap [1, 30] = \{6, 9, 12, 16, 18, 24\}$$
Six distinct moduli funnel into 24; $m = 9$ is one of the six natural pre-images.

### Joint extremality

| Property | Operator | Extremal at 24 | Direction |
|----------|----------|----------------|-----------|
| Pisano fixed point | $\pi$ | $24 = \min\{m > 1 : \pi(m) = m\}$ | minimum (non-trivial) |
| Carmichael $\lambda = 2$ | $\lambda$ | $24 = \max\{m : \lambda(m) = 2\}$ | maximum |

**Original contribution**: Both halves are classical. The joint statement — that $m = 24$ is simultaneously minimum under $\pi$ and maximum under $\lambda=2$, and that $m = 9$ is a canonical pre-image in the $\pi$-basin — is (to our knowledge) original to the QA project and provides a principled number-theoretic answer to "why $m = 24$?"

### Supporting classical results

- **Cannonball identity** (Watson 1918): $\sum_{k=1}^{24} k^2 = 4900 = 70^2$, the unique nontrivial solution. This is the construction step that builds the Leech lattice $\Lambda_{24}$ from the Lorentzian lattice $II_{25,1}$ via the Weyl vector $(0, 1, \ldots, 24 \mid 70)$.
- **24-theorem**: for every prime $p \geq 5$, $p^2 - 1 \equiv 0 \pmod{24}$. Proof: $p$ odd $\Rightarrow 8 \mid (p-1)(p+1)$; $p \not\equiv 0 \pmod 3 \Rightarrow 3 \mid p^2 - 1$; hence $24 \mid p^2 - 1$.
- **Leech lattice / bosonic string / monstrous moonshine**: $24$ is the dimension of the unique even unimodular lattice with no roots, the critical dimension $-2$ of bosonic string theory, and the exponent in $\eta^{24} = \Delta$ (Dedekind eta / Ramanujan tau). See Baez, "My Favorite Numbers: 24" (Rankin Lecture, Glasgow 2008).

### Witnesses

| $m$ | $\pi(m)$ | $\lambda(m)$ | Role |
|-----|----------|--------------|------|
| 24 | **24** | **2** | Joint extremum — Pisano FP AND max $\lambda=2$ |
| 9 | **24** | 6 | QA theoretical modulus; maps to 24 in one step |
| 12 | 24 | **2** | In both $\lambda=2$ set AND Pisano-24 basin, but $\pi(12) \neq 12$ |
| 8 | 12 | **2** | $\lambda=2$ member; $(\mathbb{Z}/8\mathbb{Z})^\times$ all square to 1 |
| 6 | 24 | **2** | Both $\lambda=2$ AND Pisano-24 basin; minimal such element |
| 120 | **120** | 4 | Second Pisano FP (confirms sequence $24 \cdot 5^k$) |
| 1 | 1 | 1 | Trivial case (excluded from non-trivial extremality counts) |

## Checks

| ID | Description |
|----|-------------|
| DE_1         | schema_version matches |
| DE_PISANO    | $\pi(m)$ verified for declared witnesses |
| DE_MIN_FP    | exhaustive $[2, 200]$ sweep: non-trivial Pisano FPs = $\{24, 120\}$, min = 24 |
| DE_CARMICHAEL | $\lambda(m)$ verified; $\{m \in [1,100] : \lambda(m) = 2\} = \{3,4,6,8,12,24\}$ |
| DE_MAX_LAM   | max of $\lambda=2$ set = 24 |
| DE_JOINT     | $\pi(24) = 24$ AND $\lambda(24) = 2$ |
| DE_BRIDGE    | $\pi(9) = 24$ |
| DE_BASIN     | $\pi^{-1}(24) \cap [1, 30] = \{6, 9, 12, 16, 18, 24\}$ |
| DE_CANNON    | cannonball identity $\sum_{k=1}^{24} k^2 = 70^2$ |
| DE_24THM     | $p^2 - 1 \equiv 0 \pmod{24}$ for primes $p \in [5, 50]$ |
| DE_SRC       | source attribution (Wall / OEIS / Carmichael / Watson / Baez) |
| DE_WITNESS   | $\geq 5$ witnesses with verified values |
| DE_F         | fail_ledger well-formed |

## Source grounding

- **Wall, D.D.** "Fibonacci series modulo m", *American Mathematical Monthly* 67 (1960), 525–532
- **OEIS** [A001175](https://oeis.org/A001175) (Pisano periods), [A235702](https://oeis.org/A235702) (Pisano period fixed points)
- **Carmichael, R.D.** "Note on a new number theory function", *Bull. AMS* 16 (1910), 232–238
- **Watson, G.N.** "The problem of the square pyramid", *Messenger of Mathematics* 48 (1918), 1–22
- **Baez, J.** "My Favorite Numbers: 24", Rankin Lecture, University of Glasgow, 2008. [math.ucr.edu/home/baez/numbers/24.pdf](https://math.ucr.edu/home/baez/numbers/24.pdf)
- **Conway, J.H. & Sloane, N.J.A.** *Sphere Packings, Lattices and Groups*, Springer (1988), chapters on the Leech lattice
- Verification: `tools/verify_dual_extremality_24.py`

## Connection to [191] Bateson Learning Levels

Cert [191] formalized Level-III operators as modulus-changing maps between state spaces $S_m \to S_n$. Item 5 of the [191] sketch asked: **does a Level-III operator on QA state spaces have $m = 9$ or $m = 24$ as a fixed point? This would be the stability criterion for safe kernel self-improvement.**

This cert answers the question. **The Pisano period operator $\pi$ is the canonical Level-III operator on moduli** (it takes a modulus to the length of its Fibonacci orbit, which is literally the orbit length of the QA dynamic). Under $\pi$:
- $m = 24$ is the minimum non-trivial fixed point.
- $m = 9$ is a pre-image reaching the fixed point in one step.

QA's modulus choices sit at the two canonical positions in the minimal attractor basin. **Level-III self-improvement in QA is not an open problem — it has a classical answer grounded in Wall's 1960 theorem and OEIS A235702.**

## Connection to other families

- **[130] QA Origin of 24** — provides a DUAL (geometric) derivation of 24 via $H^2 - G^2 = 2CF$ always divisible by 24. This cert adds a THIRD derivation via joint Pisano/Carmichael extremality.
- **[150] QA Septenary Unit Group** — $(\mathbb{Z}/9\mathbb{Z})^\times \cong \mathbb{Z}/6\mathbb{Z}$ is the unit group of our theoretical modulus; $\pi(9) = 24$ shows it maps to the Pisano FP.
- **[191] QA Bateson Learning Levels** — closes item 5 of the sketch; the Pisano operator is the natural Level-III operator.

## Fixture files

- `fixtures/de_pass_extremality.json` — PASS: 7 witnesses (24, 9, 12, 8, 6, 120, 1) with verified Pisano and Carmichael values; full theorem statement; original-contribution annotation
- `fixtures/de_fail_bad_pisano.json` — FAIL fixture for testing validator
