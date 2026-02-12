# QA Math Compiler — Demo Pack v1

This directory contains a curated set of fully certified mathematical examples.

Each example demonstrates the complete QA Math Compiler pipeline:

1. Human-readable mathematical claim
2. Formal Lean task definition
3. Step-by-step proof trace
4. Deterministic replay record
5. Human <-> formal pairing certificate

Read files in each example in this order:

1. `claim.txt`
2. `task.json`
3. `trace.json`
4. `replay.json`
5. `pair.json`
6. `status.json`

Validation entry points:

```bash
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py demo_pack qa_alphageometry_ptolemy/qa_math_compiler/demo_pack_v1
python qa_alphageometry_ptolemy/qa_meta_validator.py
```

No proof without replay.
No replay without determinism.
No determinism without artifacts.
