Handle PolynomialError in Mod gcd simplification

Catch `PolynomialError` from `gcd(p, q)` in `Mod.eval` so modulo expressions
with `Piecewise` arguments stay unevaluated instead of raising during
substitution. Add a regression test for the reported `subs({1: 1.0})` case.
