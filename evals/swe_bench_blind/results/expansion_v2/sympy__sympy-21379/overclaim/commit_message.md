Handle Piecewise `Mod` gcd failures during substitution

Catch `PolynomialError` in `Mod.eval()` when gcd extraction encounters
`Piecewise` expressions, and add a regression test covering
`expr.subs({1: 1.0})` for the reported `exp(sinh(Piecewise(...) / z))`
case with real symbols.
