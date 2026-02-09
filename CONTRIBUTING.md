# Contributing

## Quick start (local)

All validators are deterministic and require only Python 3.10+ stdlib. No GPU, no pip install, no external downloads.

Run the full suite from the repo root:

```bash
cd qa_alphageometry_ptolemy
python qa_meta_validator.py           # families [1]-[24] + doc gate [25]
python qa_conjecture_core.py          # 5 checks: factories, ledger, guards
python qa_fst/qa_fst_validate.py      # 8 checks: spine, certs, manifest
python qa_kayser/qa_kayser_validate.py --all  # 28 checks: C1-C6 correspondence suite (merkle-rooted)
```

Fast mode (manifest integrity only, skips deep semantic checks):
```bash
python qa_meta_validator.py --fast
```

## The two-tract rule

Every certificate family must ship with **both** tracts:

1. **Machine tract**: schema, validator, cert bundle, counterexamples, meta-validator hook
2. **Human tract**: `docs/families/[NN]_<slug>.md` + index entry in `docs/families/README.md`

Meta-validator test **[25]** enforces this. CI fails if docs are missing for any registered family.

## Adding a new family

### 1. Write the machine tract

- Create schema(s) in `qa_alphageometry_ptolemy/schemas/`
- Create validator in `qa_alphageometry_ptolemy/` (must expose a function with signature `(base_dir: str) -> Optional[str]`)
- Create semantics cert, witness pack, and counterexamples pack in `qa_alphageometry_ptolemy/certs/`

### 2. Register in FAMILY_SWEEPS

In `qa_alphageometry_ptolemy/qa_meta_validator.py`, add **one tuple** to `FAMILY_SWEEPS`:

```python
FAMILY_SWEEPS = [
    # ... existing families ...
    (25, "My new family",
     _validate_my_new_family_if_present,
     "semantics + witness + counterexamples", "25_my_new_family"),
]
```

The five fields:
- Family ID (next sequential integer)
- Label (shown in CI output)
- Validator function (`base_dir -> Optional[str]`: `None` = pass, string = skip reason)
- Pass description (shown on success)
- Doc slug (must match `docs/families/{slug}.md`)

### 3. Write the human tract

Copy the template:
```bash
cp docs/families/TEMPLATE.md docs/families/25_my_new_family.md
```

Fill in all sections (what/artifacts/how-to-run/semantics/failures/examples/changelog).

Add a row to `docs/families/README.md`:
```markdown
| [25] | [My New Family](25_my_new_family.md) | Triplet | PASS |
```

### 4. Validate locally

```bash
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

All tests including doc gate [25] must pass before pushing.

## Rehashing

When you change a source path, payload, or any field that feeds into a hash chain:

```bash
python qa_svp_cmc_validator.py --rehash certs/witness/QA_SVP_CMC_WITNESS_PACK.v1.json
python qa_ingest_validator.py --rehash certs/witness/QA_INGEST_WITNESS_PACK.v1.json
```

Hash chain rules:
- **Domain-separated hashing**: `ds_sha256(domain, payload) = sha256(domain.encode() + b'\x00' + payload)`
- **Canonical JSON**: `json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`
- **Manifest placeholder**: use HEX64_ZERO (64 `'0'` characters), not the string `"placeholder"`
- Changing any layer cascades through all downstream hashes

## Gitignored sources

The `ingestion candidates/` directory is gitignored. Source files referenced by witness/counterexample packs must live in `qa_ingestion_sources/` (committed to git). Never point `source_ref` at a gitignored path.

## Reporting a failure (as an obstruction)

Open a GitHub Issue and include:

1. Command run + OS + Python version
2. Exact error output
3. If applicable: the certificate or input file that triggered it
4. Any invariant diffs reported by the validator

Tag with one of:

- `obstruction:invariant` -- invariant mismatch or violation
- `obstruction:non-reachability` -- generator reachability failure
- `obstruction:drift` -- numeric or behavior drift between environments
- `obstruction:schema` -- structural or schema-version violation

## CI

All validators run automatically on push and PR via GitHub Actions. See the badge at the top of the README for current status.
