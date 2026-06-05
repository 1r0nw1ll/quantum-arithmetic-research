# [346] QA Fibonacci-Lucas Bridge

**Family**: `qa_fibonacci_lucas_bridge_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XV pp.99-107

> "The series will run: 1, 3, 4, 7, 11, 18,......"  
> "This same series will also result from the addition of two standard Fibonacci  
>  series, with one series moved two integers to the right from the other."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Lucas sequence 1,3,4,7,11,18,... satisfies L(n)=L(n-1)+L(n-2); 20 terms | PASS |
| C2 | L(n+1) = F(n) + F(n+2) for n=1..19 (Iverson's shifted Fibonacci sums) | PASS |
| C3 | Two consecutive Fibonacci quadruples as integer proxies for phi-bead quadruples; shift by one | PASS |
| C4 | gcd(L(n), F(n+1)) ∈ {1,2} for n=1..20 (Lucas-Fibonacci coprimeness) | PASS |

## Structure

### Lucas Sequence (C1)

$$L(1)=1,\; L(2)=3,\; L(n)=L(n-1)+L(n-2)$$

$$1, 3, 4, 7, 11, 18, 29, 47, 76, 123, \ldots$$

Iverson derives this from the powers of the Golden Section numbers via:
- Odd positions: $\phi'^{(2n-1)} - \phi^{(2n-1)}$
- Even positions: $\phi'^{(2n)} + \phi^{(2n)}$

where $\phi = (\sqrt{5}-1)/2$ and $\phi' = (\sqrt{5}+1)/2$.

### Shifted Fibonacci Sum (C2)

The Lucas sequence arises from two overlapping Fibonacci series shifted by two positions:

$$L(n+1) = F(n) + F(n+2)$$

| n | F(n) | F(n+2) | Sum | L(n+1) |
|---|------|---------|-----|--------|
| 1 | 1 | 2 | 3 | L(2)=3 ✓ |
| 2 | 1 | 3 | 4 | L(3)=4 ✓ |
| 3 | 2 | 5 | 7 | L(4)=7 ✓ |
| 4 | 3 | 8 | 11 | L(5)=11 ✓ |
| 5 | 5 | 13 | 18 | L(6)=18 ✓ |

### Golden Section Bead Quadruples (C3)

Iverson identifies two Fibonacci-type quadruples from the Golden Section:

- **Set A**: $(\phi^2, \phi, 1, \phi') \approx (0.38197, 0.61803, 1, 1.61803)$
- **Set B**: $(\phi, 1, \phi', \phi'^2) \approx (0.61803, 1, 1.61803, 2.61803)$

Both satisfy the Fibonacci-type property (each term = sum of two previous). In integer form (consecutive Fibonacci quadruples, n=11):
- Set A ≡ $(89, 144, 233, 377)$
- Set B ≡ $(144, 233, 377, 610)$

Set B is Set A shifted left by one position — sharing three elements. This is the same overlapping-quadruple structure as the BABTHE dual chain [345].

### Lucas-Fibonacci Coprimeness (C4)

For all 20 tested pairs: $\gcd(L(n), F(n+1)) \in \{1, 2\}$. In practice all 20 are coprime ($\gcd = 1$).

## Observer Projection Note (Theorem NT)

"Golden Section," "noble sections," "phi," "phi-prime" are observer labels for the irrational limit of Fibonacci ratios. The causal integer structure: F(n)+F(n+2)=L(n+1) (integer addition), L(n)=L(n-1)+L(n-2) (integer recurrence). No continuous irrationals enter the QA causal layer.

**Depends on**: [343] Fibonacci Bead Number Quadruple; [345] BABTHE Dual Bead Chain
