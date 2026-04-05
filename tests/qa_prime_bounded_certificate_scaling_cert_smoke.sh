#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 qa_prime_bounded_certificate_scaling_cert_v1/validator.py --self-test --json >/tmp/qa_prime_bounded_certificate_scaling_cert_self_test.json
python3 qa_prime_bounded_certificate_scaling_cert_v1/validator.py --file qa_prime_bounded_certificate_scaling_cert_v1/fixtures/pass_scaling_100_1000.json --json >/tmp/qa_prime_bounded_certificate_scaling_cert_pass.json

python3 - <<'PY'
import json
from pathlib import Path

self_test = json.loads(Path("/tmp/qa_prime_bounded_certificate_scaling_cert_self_test.json").read_text())
assert self_test["ok"] is True
validated = json.loads(Path("/tmp/qa_prime_bounded_certificate_scaling_cert_pass.json").read_text())
assert validated["ok"] is True
print("qa_prime_bounded_certificate_scaling_cert_smoke: ok")
PY
