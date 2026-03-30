TESTING (Certificate Harness)
-----------------------------

This repo includes a minimal, scoped pytest harness for the Generator-Relative Barrier Certificate.

Run:
  pytest

Expected:
  3 passed

Notes:
- pytest.ini scopes collection to the three certificate tests only.
- pythonpath="." is set so bfs_verify.py imports cleanly.
- Any additional tests elsewhere in the repo are out of scope for this certificate harness by design.
