QA Structural Hypothesis

Monoid identification

Forward‑only generators {σ, R} generate the free monoid of 2×2 determinant‑1 matrices with non‑negative integer entries (often denoted \operatorname{SL}_2(\mathbb{N}_0)).  In the QA kernel we fix an explicit matrix representation by mapping the operations to elementary matrices:
	•	σ ↦ L = \begin{bmatrix}1 & 0\\1 & 1\end{bmatrix}.  Applying σ to (b,e) corresponds to left‑multiplication by this matrix, yielding (b,\,e+b).
	•	R ↦ \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}.  By definition R := μ σ μ, so it sends (b,e) to (b+e,\,e).

These two matrices freely generate the submonoid of \operatorname{SL}_2(\mathbb{Z}) consisting of integer matrices with determinant 1 and non‑negative entries.  Every product of these matrices again has determinant 1 and non‑negative integer entries.

Uniqueness / normal form

Encoding convention
	•	In implementation-level words, token "L" means apply σ.
	•	Token "R" means apply the generator R := μ σ μ.
	•	So word strings are over {"L","R"}, while algebraic statements may refer to {σ, R}.

For any coprime pair (b,e)\in\mathbb{N}^2 reachable from the seed (1,1) by forward applications of σ and R, there is a unique word in the alphabet {σ, R} that produces it.  This uniqueness can be proven by reversing the process via a simple parent rule derived from the Euclidean algorithm.  Given a target state (b,e), its unique predecessor and the last move taken are determined as follows:
	•	If e > b, then the last move was σ, and the parent state is (b, e-b).
	•	If b > e, then the last move was R, and the parent state is (b-e, e).
	•	The process stops at the base seed (1,1).

Iteratively applying this rule yields a reverse path to (1,1); reading the moves in reverse order gives the unique forward word.  If at any step the subtraction would produce a non‑positive entry, or the pair ceases to be coprime, the reverse procedure fails (see failure modes below).

Invariants
	•	Greatest common divisor.  The quantity \gcd(b,e) is invariant under σ and R (and under the swap μ as well).  Thus starting from (1,1) (which has gcd 1) and applying only σ and R, all reachable states remain coprime.  When including other operations, the gcd evolves in a controlled way: applying the scalar multiplication \lambda_k multiplies \gcd(b,e) by k (for integer k\ge 1), and applying the halving map \nu divides \gcd(b,e) by 2 whenever both coordinates are even.

Failure modes

All QA operations use a unified failure algebra.  Each failure is reported with a fail_type drawn from the canonical list below and always includes an invariant_diff and a details field.  The allowed fail_type values are:
	•	NOT_IN_NATURAL_DOMAIN.  An argument is outside the natural‑number domain.  This covers non‑positive integers, non‑integers and invalid scaling factors k.
	•	NOT_COPRIME.  The target pair does not have gcd 1 when required for seed‑reachability.
	•	REVERSE_STEP_INVALID.  A reverse subtraction in the parent rule would produce a non‑positive coordinate; the reverse computation cannot proceed.
	•	ODD_BLOCK.  The halving operation \nu requires both coordinates to be even; when either coordinate is odd the operation fails with this type.
	•	NOT_UNIQUE.  Used by reachability enumerators when two distinct (\text{word},\text{scale}) descriptions generate the same state.  This signals a logical inconsistency and is useful for audits.

These QA‑specific failure modes provide structured feedback when an operation cannot be performed within the positive unimodular monoid or its scaling extensions.
