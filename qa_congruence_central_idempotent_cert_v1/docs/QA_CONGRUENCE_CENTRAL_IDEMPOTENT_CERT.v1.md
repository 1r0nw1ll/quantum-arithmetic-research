# QA_CONGRUENCE_CENTRAL_IDEMPOTENT_CERT.v1

This family certifies whether a chosen **central element** of the finite quotient
\(G = PSL_2(\mathbb{Z}/72\mathbb{Z})\) is an **idempotent** in the group algebra
center \(Z(\mathbb{Q}[G])\).

## Why this exists

`QA_CONGRUENCE_SECTOR_LABEL_CERT.v1` certifies that a label is a **class function**
(conjugacy invariant) on the full quotient group. That makes it eligible for
congruence-sectoring.

However, “class function” is weaker than “spectral projector”. A projector-like
object corresponds to a **central idempotent** (or sum of primitive central
idempotents) in the center of the group algebra.

This family provides the next QA gate:

- PASS: a specific central element \(e\) satisfies \(e^2=e\) exactly in
  \(Z(\mathbb{Q}[G])\).
- FAIL: a witness shows \(e^2-e\neq 0\).

## Central elements (class-average basis)

Let \(C_0,\dots,C_{r-1}\) be the conjugacy classes of \(G\). Define
\[
A_i := \frac{1}{|C_i|}\sum_{g\in C_i} g \in \mathbb{Q}[G].
\]
Any
\[
e = \sum_i \alpha_i A_i,\quad \alpha_i\in\mathbb{Q}
\]
is central.

## v1 scope

v1 is an **exact obstruction gate**, not a full center-algebra engine:

- It provides exact PASS only for the trivial idempotents `0` and `1`.
- For general bucket constructions it provides exact FAIL witnesses by checking
  only the **identity-coordinate** of \(e^2-e\) in the class-average basis.

This is sufficient to certify the key negative statement:
“a class-function bucket is not automatically a projector”, while avoiding
the full class-multiplication table \(A_iA_j=\sum_k m_{ij}^k A_k\).
