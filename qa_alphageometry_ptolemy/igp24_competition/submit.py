"""
IGP24 Competition Submission Client
SAIR Inverse Galois Problem, Stage 1 closes 2026-08-15.
Platform: competition.sair.foundation/competitions/igp24/

SETUP (one time):
  1. Register at competition.sair.foundation
  2. Go to your profile → API tokens → create token
  3. export IGP24_TOKEN="your-token-here"
     OR set TOKEN variable below
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
import time
from polynomials import (
    C24_T1_COEFFS, C12C2_T2_COEFFS, C2C2C6_T3_COEFFS,
    to_magma_poly, C24_T1_DISC_FIELD
)

BASE_URL = "https://competition.sair.foundation/competitions/igp24"
TOKEN    = os.environ.get("IGP24_TOKEN", "")   # set before calling submit()

# T-number → (group name, coefficients, notes)
SUBMISSIONS: dict[str, tuple[str, list[int], str]] = {
    "24T1": ("C24",      C24_T1_COEFFS,    f"disc={C24_T1_DISC_FIELD} optimal CRT K3⊗K8"),
    "24T2": ("C12xC2",   C12C2_T2_COEFFS,  "cyclotomic Phi_39, conductor 39"),
    "24T3": ("C2^2xC6",  C2C2C6_T3_COEFFS, "cyclotomic Phi_56, conductor 56"),
}


def _post(endpoint: str, payload: dict) -> dict:
    url  = f"{BASE_URL}/{endpoint.lstrip('/')}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
            "User-Agent": "QA-IGP24-client/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"error": e.code, "reason": e.reason, "body": body}
    except Exception as e:
        return {"error": str(e)}


def verify_poly(t_number: str, coeffs: list[int]) -> dict:
    """Call competition's Magma API to verify Gal(f/Q) == t_number group."""
    return _post("api/verify", {
        "t_number": t_number,
        "polynomial": coeffs,          # high-to-low degree
        "format": "coeffs_high_to_low",
    })


def submit_poly(t_number: str, coeffs: list[int]) -> dict:
    """Submit polynomial for scoring."""
    return _post("api/submit", {
        "t_number": t_number,
        "polynomial": coeffs,
        "format": "coeffs_high_to_low",
    })


def check_scoreboard() -> dict:
    """Fetch current scoreboard / solved groups."""
    req = urllib.request.Request(
        f"{BASE_URL}/api/scoreboard",
        headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": "QA-IGP24-client/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def run_campaign(dry_run: bool = True) -> None:
    if not TOKEN and not dry_run:
        raise RuntimeError(
            "Set IGP24_TOKEN env var first.\n"
            "Register at competition.sair.foundation, then:\n"
            "  export IGP24_TOKEN='your-token'\n"
        )

    print(f"=== IGP24 Submission Campaign {'(DRY RUN)' if dry_run else ''} ===")
    print(f"Platform: {BASE_URL}")
    print()

    for t_num, (name, coeffs, notes) in SUBMISSIONS.items():
        print(f"── {t_num} ({name}): {notes}")
        print(f"   Polynomial degree: {len(coeffs)-1}")
        print(f"   Leading coeff: {coeffs[0]}, constant: {coeffs[-1]}")
        print(f"   Magma: f := {to_magma_poly(coeffs)[:80]}...")

        if dry_run:
            print("   [DRY RUN — no API call]")
        else:
            # First verify, then submit if verified
            print("   Verifying with competition API...")
            result = verify_poly(t_num, coeffs)
            print(f"   Verify response: {result}")

            if result.get("galois_group") == t_num or result.get("verified"):
                print(f"   ✓ Verified. Submitting...")
                sub = submit_poly(t_num, coeffs)
                print(f"   Submit response: {sub}")
            else:
                print(f"   ✗ Verification failed or unexpected response, skipping submit.")

            time.sleep(2)   # respect rate limits
        print()


def magma_script() -> str:
    """Generate self-contained Magma script for manual submission at competition portal."""
    lines = [
        "// IGP24 QA-derived polynomials — paste into competition Magma console",
        "// QA source: qa_inverse_galois_degree24_cert_v1 [506]",
        "",
    ]
    for t_num, (name, coeffs, notes) in SUBMISSIONS.items():
        lines += [
            f"// {t_num} ({name}): {notes}",
            f"f{t_num.replace('24T','')} := {to_magma_poly(coeffs)};",
            f"G{t_num.replace('24T','')} := GaloisGroup(NumberField(f{t_num.replace('24T','')}));",
            f"print \"Gal = \", G{t_num.replace('24T','')};",
            f"print \"disc = \", Discriminant(NumberField(f{t_num.replace('24T','')}));",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--magma" in sys.argv:
        print(magma_script())
    elif "--submit" in sys.argv:
        run_campaign(dry_run=False)
    else:
        run_campaign(dry_run=True)
        print()
        print("── Magma script for manual portal submission:")
        print(magma_script())
