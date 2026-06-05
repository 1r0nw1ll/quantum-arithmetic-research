# [348] QA Fibonacci Divisibility Period Laws

**Family**: `qa_fibonacci_divisibility_period_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XV pp.107-110

> "One can see that every third number is an even number. Of these, every other one  
>  is a 2-par number and the ones in between are all 4-par numbers. So every sixth  
>  number of the extended Fibonacci series is a 4-par number. Beginning with any  
>  number which is divisible by 3, every fourth integer is a 3-tri number.  
>  In the same way, every fifth integer is a 5-pent number."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Every 3rd Fibonacci number is even (2-par or 4-par): F(3k) ≡ 0 (mod 2) | PASS |
| C2 | Every 4th Fibonacci number divisible by 3 (3-tri): F(4k) ≡ 0 (mod 3) | PASS |
| C3 | Every 5th Fibonacci number divisible by 5 (5-pent): F(5k) ≡ 0 (mod 5) | PASS |
| C4 | Every 6th Fibonacci number divisible by 4 (4-par); odd-3rd = 2-par only | PASS |
| C5 | Coprime pair counts: 15 (b-odd,≤6), 13 (b-odd,≤5), 35 (b-any,≤7), 18 (e,d pairs,d<8) | PASS |

## Structure

### Fibonacci Divisibility Periods (C1-C4)

The standard Fibonacci sequence F(1),F(2),F(3),... = 1,1,2,3,5,8,13,21,34,55,89,144,...:

| Divisor | Period | First occurrence | QA classification |
|---------|--------|-----------------|-------------------|
| 2 | 3 | F(3)=2 | 2-par or 4-par |
| 3 | 4 | F(4)=3 | 3-tri |
| 4 | 6 | F(6)=8 | 4-par |
| 5 | 5 | F(5)=5 | 5-pent |

**C4 refinement**: F(3k) is always even. Among these:
- F(6k) = F(3·2k): divisible by 4 (4-par)
- F(3·(2k-1)): divisible by 2 only (2-par)

This is the "par-type alternation" within the even Fibonacci numbers.

### Coprime Pair Counts (C5)

From Iverson's Chapter XV Q&A:

**Q7**: "How many pairs (b,e) with b odd, gcd(b,e)=1, are there with both numbers less than 7? Less than 6? Less than 8 with b odd or even?"

| Constraint | Count |
|-----------|-------|
| b odd, both ≤ 6 | 15 |
| b odd, both ≤ 5 | 13 |
| b any, both ≤ 7 | 35 |

**Q8**: "How many (e,d) pairs with d<8, gcd(e,d)=1?"

| Constraint | Count |
|-----------|-------|
| 1 ≤ e ≤ d < 8, gcd(e,d)=1 | 18 |

The 18 includes the diagonal pair (e=1,d=1) with gcd(1,1)=1 (corresponding to b=d-e=0, the degenerate "singularity" case).

### General Divisibility Law

The Fibonacci divisibility periods follow Iverson's pattern: every $p$-th Fibonacci number (for prime $p$) is divisible by $p$ when $p \in \{2,3,5\}$. This connects to the Pisano period $\pi(m)$, the period of the Fibonacci sequence mod $m$: $\pi(2)=3$, $\pi(3)=4$, $\pi(4)=6$, $\pi(5)=5$.

## Observer Projection Note (Theorem NT)

"Par-number," "tri-number," "pent-number" are observer classification labels on modular arithmetic residues. The causal structure: Fibonacci recurrence $F(n)=F(n-1)+F(n-2)$ over integers, modular arithmetic $F(n) \bmod m$, periodicity of residues. No continuous functions enter the QA causal layer.

**Depends on**: [343] Fibonacci Bead Number Quadruple; [344] Prime Residue Symmetry; [346] Fibonacci-Lucas Bridge
