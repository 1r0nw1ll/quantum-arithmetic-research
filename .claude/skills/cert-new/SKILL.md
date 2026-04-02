---
description: Scaffold a new QA certificate family with all required artifacts
---

The user will provide a family number and name, e.g. `/cert-new 155 QA_BEARDEN_PHASE_CONJUGATE_CERT`.

Create the full family scaffold under `qa_alphageometry_ptolemy/`:

1. **Directory**: `qa_{lowercase_slug}_v1/`
2. **mapping_protocol_ref.json** with fields: `protocol_version`, `ref_path`, `ref_sha256`
3. **schema.json** — minimal valid JSON Schema for the cert
4. **validator.py** — stdlib-only Python validator stub that loads schema, validates fixtures, exits 0/1
5. **fixtures/pass_default.json** — one valid fixture
6. **fixtures/fail_missing_field.json** — one invalid fixture (missing required field)
7. **docs/families/[NN]_slug.md** — human tract stub with family number, title, status, and "TODO: fill" sections

Follow existing family patterns exactly. Check a recent family (e.g. family [154]) for current conventions.

After scaffolding, register the new family in `qa_meta_validator.py`'s FAMILY_SWEEPS and run the meta-validator to confirm it passes.
