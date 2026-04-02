---
name: validate
description: Run the full QA validation suite — axiom linter + meta-validator
user_invocable: true
---

Run the full QA smoke test. Execute BOTH commands and report results:

1. `python tools/qa_axiom_linter.py --all` — scan all Python files for axiom violations
2. `cd qa_alphageometry_ptolemy && python qa_meta_validator.py` — validate all certificate families

Report: total files scanned, violations found (with file:line), families passed/failed.
If everything passes, just say "All clean — [N] files, [M] families, 0 violations."
