# QA_ENERGY_CAPABILITY_SEPARATION_CERT.v1

Machine-tract certificate family for the theorem-level hygiene:

> `Energy ≠ Capability`  
> `Capability = Reachability(S, G, I)`  
> Energy is only an ordering heuristic over legal successors.

This family proves a **constructive separation witness** in a finite Caps(N,N)
universe with canonical QA generators (subset `{sigma, mu}`):

- target is reachable under the generator set (explicit witness path)
- deterministic min-energy-legal policy fails to reach target within budget

It is intentionally **exact** (no floats) and **replayable** (deterministic).

## Run

```bash
python qa_energy_capability_separation_cert/validator.py --self-test
python qa_energy_capability_separation_cert/validator.py qa_energy_capability_separation_cert/fixtures/valid_min.json
```

