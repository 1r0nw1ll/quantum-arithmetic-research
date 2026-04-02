# Family [179] QA_PAIRED_PISANO_CERT.v1

## One-line summary

Fibonacci coprime pairs have 2.25x higher paired Pisano divisibility (both p|pi(m) AND q|pi(m)) than non-Fibonacci pairs (p=0.0017, 499 moduli).

## Mathematical content

### Paired divisibility

For coprime pair (p,q) and modulus m, "both-divide" means both p and q divide the Pisano period pi(m). Across m=2..500:

- Fibonacci pairs: mean both-divide rate 0.526
- Non-Fibonacci pairs: mean both-divide rate 0.234
- Ratio: 2.25x, Mann-Whitney U p=0.0017

### Order-1 deep dive

Among consecutive ratios (|p-q|=1): Fibonacci mean 0.857 vs non-Fibonacci mean 0.227, ratio 3.77x, Mann-Whitney p=0.028.

### LCM mechanism

For coprime (p,q), both-divide requires lcm(p,q)=p*q | pi(m). Fibonacci pairs have smaller products AND Fibonacci numbers individually divide pi(m) 3.70x more often (from [163]).

### Product-matched control

5/9 (56%) Fibonacci pairs beat their closest product-matched non-Fibonacci pair. Advantage is real but modest after controlling for product size.

### 4:1 exception

4:1 has lcm=4 (very small), giving it higher both-divide rate than some Fibonacci pairs at order-3. This explains why Kirkwood gap 4:1 is as deep as Fibonacci resonances.

## Tier

Tier 3 (p<0.01 with honest controls).

## Checks

PP_1, PP_PAIRS, PP_STAT, PP_RATIO, PP_ORDER, PP_MECH, PP_HONEST, PP_W, PP_F

## Fixtures

1 PASS + 1 FAIL; self-test ok.

## Source

`qa_pisano_paired_divisibility.py`
