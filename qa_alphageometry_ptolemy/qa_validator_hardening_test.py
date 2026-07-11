#!/usr/bin/env python3
# QA_COMPLIANCE = "test harness — no QA state; adversarial validation regression test"
"""
Adversarial regression test for the graph-structure + topology-resonance cert
validators (the 2026-07-11 hardening audit).

Each validator's reference SUCCESS example must validate with zero failures; every
malformed mutation of a boolean / enum / status field must be REJECTED (>= 1 failure).
Guards against regression of the hardened fields:
  - boolean fields must be real booleans (not bool()-coerced: "false"/1 rejected)
  - success must be a boolean
  - phase_lock is a hard invariant (must be true on success certs)
  - generator_grounding.status must be a valid enum and not falsely claim
    concrete_implementation (no concrete generator implementation exists)

Run: python qa_validator_hardening_test.py   (exit 0 = all guards hold)
"""
from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _failures(validator_cls, cert):
    """Failure count across all three validation levels. A crash in any level counts
    as a failure -- a validator that raises is NOT a successful validation (so the
    valid example is not spuriously reported as 0 failures on a regression, and a
    malformed cert that crashes recompute still counts as rejected)."""
    v = validator_cls()
    crashes = 0
    for level in (v.validate_schema, v.validate_consistency, v.validate_recompute):
        try:
            level(cert)
        except Exception:
            crashes += 1
    return sum(1 for r in v.results if r.status.name == "FAILED") + crashes


def _set(path, value):
    """Return a mutator that sets a dotted path in the cert to value."""
    keys = path.split(".")

    def mut(cert):
        o = cert
        for k in keys[:-1]:
            o = o.setdefault(k, {})
        o[keys[-1]] = value
    return mut


def _del(path):
    keys = path.split(".")

    def mut(cert):
        o = cert
        for k in keys[:-1]:
            o = o.get(k, {})
        o.pop(keys[-1], None)
    return mut


CASES = {
    "topology": {
        "cls": "qa_topology_resonance_validator_v1.TopologyResonanceValidator",
        "example": "examples/topology/topology_resonance_success.json",
        "mutations": {
            "success=0": _set("success", 0),
            "success='true'": _set("success", "true"),
            "phase_lock=false": _set("invariants.phase_lock", False),
            "phase_lock='true'": _set("invariants.phase_lock", "true"),
            "scc_monotone='false'": _set("invariants.scc_monotone_non_decreasing", "false"),
            "resonance_certified=1": _set("topology_witness.resonance_certified", 1),
            "phase_preserved='false'": _set("phase_witness.phase_preserved", "false"),
            "grounding.status='bogus'": _set("generator_grounding.status", "bogus"),
            "grounding.status='concrete_implementation'": _set("generator_grounding.status", "concrete_implementation"),
            "grounding.status missing": _del("generator_grounding.status"),
        },
    },
    "graph": {
        "cls": "qa_graph_structure_validator_v1.GraphStructureValidator",
        "example": "examples/graph_structure/graph_structure_success.json",
        "mutations": {
            "success=0": _set("success", 0),
            "success='true'": _set("success", "true"),
            "phase_preserved='false'": _set("phase_witness.phase_preserved", "false"),
        },
    },
}


def main() -> int:
    import importlib
    ok = True
    for name, spec in CASES.items():
        mod_name, cls_name = spec["cls"].rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        base = json.load(open(os.path.join(HERE, spec["example"]), encoding="utf-8"))

        valid_fail = _failures(cls, copy.deepcopy(base))
        good = valid_fail == 0
        ok &= good
        print(f"[{name}] VALID example -> {valid_fail} failures  {'OK' if good else 'REGRESSION'}")

        for mname, mut in spec["mutations"].items():
            c = copy.deepcopy(base)
            mut(c)
            f = _failures(cls, c)
            rejected = f > 0
            ok &= rejected
            print(f"    {mname:44} -> {f} failures  {'rejected' if rejected else 'HOLE (passed!)'}")

    print(f"\nvalidator hardening: {'ALL GUARDS HOLD' if ok else 'REGRESSION DETECTED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
