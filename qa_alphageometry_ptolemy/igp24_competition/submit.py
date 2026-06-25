"""
IGP24 Competition Submission Client
SAIR Inverse Galois Problem, deadline 2026-08-15.

API: POST https://api.sair.foundation/api/public/v1/competitions/igp24/submissions
     Authorization: Bearer $SAIR_API_KEY
     Body: {"payload": {"polynomials": ["a0,a1,...,a24", ...]}}

Polynomial format: 25 comma-separated integers, ASCENDING powers (a0 first).
Rate: 5 submissions/day initially (increases after 5 scoreable pairs credited).
Max: 100 polynomial lines per submission, 100,000 bytes.

SETUP:
    export SAIR_API_KEY="sair_..."
    python3 submit.py --dry-run   # preview what will be sent
    python3 submit.py --submit    # actually submit
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

API_URL   = "https://api.sair.foundation/api/public/v1/competitions/igp24/submissions"
TOKEN_KEY = "SAIR_API_KEY"
HERE      = Path(__file__).parent


def load_submission_txt(path: Path = HERE / "submission.txt") -> list[str]:
    """Parse submission.txt → list of coefficient strings (comment lines stripped)."""
    polys = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            polys.append(line)
    return polys


def submit(polys: list[str], token: str) -> dict:
    payload = json.dumps({"payload": {"polynomials": polys}}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "QA-IGP24/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
            return {"status": resp.status, "body": json.loads(body)}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "reason": e.reason, "body": e.read().decode(errors="replace")}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    dry_run = "--submit" not in sys.argv

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else HERE / "submission.txt"
    polys = load_submission_txt(path)
    print(f"submission.txt: {len(polys)} polynomial(s)")
    for i, p in enumerate(polys):
        coeffs = p.split(",")
        print(f"  [{i}] deg {len(coeffs)-1}: {p[:60]}...")
    print()

    if dry_run:
        print("[DRY RUN] — pass --submit to send")
        print()
        print("Equivalent curl:")
        body = json.dumps({"payload": {"polynomials": polys}}, indent=2)
        print(f'curl -X POST "{API_URL}" \\')
        print(f'  -H "Authorization: Bearer $SAIR_API_KEY" \\')
        print(f'  -H "Content-Type: application/json" \\')
        print(f"  -d '{body}'")
        return

    token = os.environ.get(TOKEN_KEY, "").strip()
    if not token:
        print(f"Error: set {TOKEN_KEY} env var first.", file=sys.stderr)
        sys.exit(1)

    print(f"Submitting {len(polys)} polynomial(s) to SAIR IGP24...")
    result = submit(polys, token)
    print(f"Response: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
