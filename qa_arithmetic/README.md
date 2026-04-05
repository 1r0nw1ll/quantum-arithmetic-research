# qa-arithmetic

Core primitives for [Quantum Arithmetic](https://github.com/1r0nw1ll/quantum-arithmetic-research). Pure Python, zero dependencies.

## Install

```bash
pip install qa-arithmetic
```

## Usage

```python
from qa_arithmetic import qa_step, qa_mod, orbit_family, qa_tuple, identities

# A1-compliant Fibonacci shift
qa_step(1, 1, 24)        # (1, 2)
qa_step(1, 2, 24)        # (2, 3)

# Modular reduction: result in {1,...,m}, never 0
qa_mod(25, 24)            # 1
qa_mod(0, 24)             # 24

# Orbit classification
orbit_family(1, 1, 24)   # "cosmos"
orbit_family(8, 8, 24)   # "satellite"
orbit_family(24, 24, 24) # "singularity"

# Full QA tuple (b, e, d, a)
qa_tuple(1, 1, 24)       # (1, 1, 2, 3)

# All 16 identities
ids = identities(2, 1)
ids["C"]  # 4  (green quadrance = 2de)
ids["F"]  # 3  (red quadrance = d*d - e*e = ab)
ids["G"]  # 5  (blue quadrance = d*d + e*e)
# C*C + F*F == G*G always (Wildberger Theorem 6)
```

## License

MIT
