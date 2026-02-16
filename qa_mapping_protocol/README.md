# QA Mapping Protocol (Machine Tract)

This folder defines **QA_MAPPING_PROTOCOL.v1**: a minimal, enforceable contract for claiming that an external object/system has been “mapped into QA”.

## What it is

A mapping is considered valid (v1) iff it produces a concrete object:

`M = (S, G, I, F, R, C)` where

- `S`: state manifold
- `G`: generator moves
- `I`: invariants
- `F`: failure taxonomy
- `R`: reachability relation
- `C`: determinism contract (v1: unique successor or typed FAIL)

## Files

- `schema.json`: JSON Schema for `QA_MAPPING_PROTOCOL.v1`
- `validator.py`: deterministic gate validator (Gate 1–5)
- `canonical_mapping_protocol.json`: canonical shared mapping object (usable by refs)
- `fixtures/valid_min.json`: minimal passing fixture
- `fixtures/invalid_missing_determinism.json`: intentionally failing fixture

## How to run

```bash
python qa_mapping_protocol/validator.py --self-test
python qa_mapping_protocol/validator.py qa_mapping_protocol/fixtures/valid_min.json
```

