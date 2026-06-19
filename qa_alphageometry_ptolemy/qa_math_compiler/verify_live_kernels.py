#!/usr/bin/env python3
"""Run and certify minimal proofs through Lean, Coq/Rocq, and Isabelle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "toolchains.json"
DEFAULT_REPORT = ROOT / "artifacts" / "live_kernel_certificate.v1.json"
CERT_DOMAIN = "qa.math_compiler.live_kernel_certificate.v1"

LEAN_SOURCE = """theorem qa_math_compiler_live_lean (a b : Nat) :
  a + b = b + a := Nat.add_comm a b
"""

COQ_SOURCE = """Theorem qa_math_compiler_live_coq :
  forall n : nat, n + 0 = n.
Proof.
  induction n.
  - reflexivity.
  - simpl. rewrite IHn. reflexivity.
Qed.
"""

ISABELLE_SOURCE = """theory QA_Math_Compiler_Live
  imports Main
begin

theorem qa_math_compiler_live_isabelle:
  fixes a b :: nat
  shows "a + b = b + a"
  by simp

end
"""

ISABELLE_ROOT = """session QA_Math_Compiler_Live = HOL +
  theories
    QA_Math_Compiler_Live
"""


def canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def domain_hash(domain: str, payload: object) -> str:
    return sha256_bytes(domain.encode("utf-8") + b"\x00" + canonical_bytes(payload))


def source_hash(source: str) -> str:
    return sha256_bytes(source.encode("utf-8"))


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def find_executable(names: List[str], extra_paths: List[str] | None = None) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for path in extra_paths or []:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def run(
    command: List[str],
    cwd: Path,
    timeout: int = 180,
    env: Dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def first_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def version_ok(version: str, expected_pattern: str) -> bool:
    return re.search(expected_pattern, version) is not None


def result(
    assistant_id: str,
    observed_version: str,
    pinned_version: str,
    expected_pattern: str,
    source_sha256: str,
    proof: subprocess.CompletedProcess[str],
) -> Dict[str, object]:
    matched = version_ok(observed_version, expected_pattern)
    return {
        "assistant_id": assistant_id,
        "ok": proof.returncode == 0 and matched,
        "version": pinned_version,
        "version_pattern": expected_pattern,
        "version_match": matched,
        "source_sha256": source_sha256,
        "returncode": proof.returncode,
    }


def missing_result(assistant_id: str, expected_pattern: str) -> Dict[str, object]:
    return {
        "assistant_id": assistant_id,
        "ok": False,
        "fail_type": "EXECUTABLE_MISSING",
        "version": "",
        "version_pattern": expected_pattern,
        "version_match": False,
        "source_sha256": "",
        "returncode": 127,
    }


def verify_lean(
    root: Path,
    pinned_version: str,
    expected_pattern: str,
) -> Dict[str, object]:
    executable = find_executable(
        ["lean"],
        [str(Path.home() / ".elan" / "bin" / "lean")],
    )
    if executable is None:
        return missing_result("lean4", expected_pattern)
    source_path = root / "LiveLean.lean"
    source_path.write_text(LEAN_SOURCE, encoding="utf-8")
    version_proc = run([executable, "--version"], root)
    version = first_line(version_proc.stdout or version_proc.stderr)
    proof = run([executable, str(source_path)], root)
    return result(
        "lean4",
        version,
        pinned_version,
        expected_pattern,
        source_hash(LEAN_SOURCE),
        proof,
    )


def verify_coq(
    root: Path,
    pinned_version: str,
    expected_pattern: str,
) -> Dict[str, object]:
    executable = find_executable(["coqc"])
    command_prefix: List[str]
    if executable is not None:
        command_prefix = [executable]
    else:
        rocq = find_executable(["rocq"])
        if rocq is None:
            return missing_result("rocq", expected_pattern)
        executable = rocq
        command_prefix = [rocq, "compile"]
    source_path = root / "LiveCoq.v"
    source_path.write_text(COQ_SOURCE, encoding="utf-8")
    version_proc = run([executable, "--version"], root)
    version = first_line(version_proc.stdout or version_proc.stderr)
    proof = run(command_prefix + [str(source_path)], root)
    return result(
        "rocq",
        version,
        pinned_version,
        expected_pattern,
        source_hash(COQ_SOURCE),
        proof,
    )


def isabelle_environment(root: Path, release: str) -> Dict[str, str]:
    user_home = root / "isabelle_user"
    user_settings = user_home / ".isabelle" / release / "etc"
    user_settings.mkdir(parents=True)
    (user_settings / "settings").write_text(
        'ISABELLE_TOOL_JAVA_OPTIONS="-Djava.awt.headless=true -Xms128m -Xmx1g -Xss8m"\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["USER_HOME"] = str(user_home)
    if sys.platform == "darwin":
        shim_dir = root / "isabelle_shims"
        shim_dir.mkdir()
        sysctl_shim = shim_dir / "sysctl"
        sysctl_shim.write_text(
            '#!/bin/sh\n'
            'if [ "$1" = "-n" ] && [ "$2" = "hw.physicalcpu" ]; then\n'
            '  echo 4\n'
            '  exit 0\n'
            'fi\n'
            'exec /usr/sbin/sysctl "$@"\n',
            encoding="utf-8",
        )
        sysctl_shim.chmod(0o755)
        environment["PATH"] = str(shim_dir) + os.pathsep + environment.get("PATH", "")
    return environment


def verify_isabelle(
    root: Path,
    pinned_version: str,
    expected_pattern: str,
    release: str,
) -> Dict[str, object]:
    executable = find_executable(
        ["isabelle"],
        [
            "/Applications/Isabelle.app/Isabelle/bin/isabelle",
            f"/Applications/{release}.app/bin/isabelle",
        ],
    )
    if executable is None:
        return missing_result("isabelle", expected_pattern)
    session_dir = root / "isabelle"
    session_dir.mkdir()
    (session_dir / "ROOT").write_text(ISABELLE_ROOT, encoding="utf-8")
    (session_dir / "QA_Math_Compiler_Live.thy").write_text(
        ISABELLE_SOURCE,
        encoding="utf-8",
    )
    environment = isabelle_environment(root, release)
    version_proc = run([executable, "version"], root, env=environment)
    version = first_line(version_proc.stdout or version_proc.stderr)
    proof = run(
        [executable, "build", "-D", str(session_dir), "QA_Math_Compiler_Live"],
        root,
        timeout=600,
        env=environment,
    )
    return result(
        "isabelle",
        version,
        pinned_version,
        expected_pattern,
        source_hash(ISABELLE_SOURCE + "\n" + ISABELLE_ROOT),
        proof,
    )


def build_certificate(manifest_path: Path) -> Dict[str, object]:
    manifest = load_json(manifest_path)
    assistants = manifest.get("assistants")
    if not isinstance(assistants, dict):
        raise ValueError("toolchain manifest assistants must be an object")
    with tempfile.TemporaryDirectory(prefix="qa_math_compiler_kernels_") as tmp:
        root = Path(tmp)
        results = [
            verify_lean(
                root,
                str(assistants["lean4"]["version"]),
                str(assistants["lean4"]["version_pattern"]),
            ),
            verify_coq(
                root,
                str(assistants["rocq"]["version"]),
                str(assistants["rocq"]["version_pattern"]),
            ),
            verify_isabelle(
                root,
                str(assistants["isabelle"]["version"]),
                str(assistants["isabelle"]["version_pattern"]),
                str(assistants["isabelle"]["release"]),
            ),
        ]
    body: Dict[str, object] = {
        "schema_id": "QA_MATH_COMPILER_LIVE_KERNEL_CERTIFICATE.v1",
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
        "results": results,
        "status": "PASS" if all(item.get("ok") is True for item in results) else "FAIL",
    }
    body["certificate_sha256"] = domain_hash(CERT_DOMAIN, body)
    return body


def validate_certificate(
    certificate: Dict[str, object],
    manifest_path: Path,
) -> List[str]:
    errors: List[str] = []
    manifest = load_json(manifest_path)
    expected_manifest_hash = sha256_bytes(canonical_bytes(manifest))
    if certificate.get("schema_id") != "QA_MATH_COMPILER_LIVE_KERNEL_CERTIFICATE.v1":
        errors.append("SCHEMA_ID_MISMATCH")
    if certificate.get("manifest_sha256") != expected_manifest_hash:
        errors.append("MANIFEST_HASH_MISMATCH")
    supplied_hash = certificate.get("certificate_sha256")
    body = dict(certificate)
    body.pop("certificate_sha256", None)
    if supplied_hash != domain_hash(CERT_DOMAIN, body):
        errors.append("CERTIFICATE_HASH_MISMATCH")
    results = certificate.get("results")
    if not isinstance(results, list) or len(results) != 3:
        errors.append("RESULT_SET_INVALID")
        return errors
    expected_ids = ["lean4", "rocq", "isabelle"]
    if [item.get("assistant_id") for item in results if isinstance(item, dict)] != expected_ids:
        errors.append("ASSISTANT_ORDER_INVALID")
    if any(not isinstance(item, dict) or item.get("ok") is not True for item in results):
        errors.append("KERNEL_RESULT_FAILED")
    if certificate.get("status") != "PASS":
        errors.append("CERTIFICATE_STATUS_FAILED")
    return errors


def semantic_projection(certificate: Dict[str, object]) -> Dict[str, object]:
    return {
        "schema_id": certificate.get("schema_id"),
        "manifest_sha256": certificate.get("manifest_sha256"),
        "results": certificate.get("results"),
        "status": certificate.get("status"),
    }


def self_test(manifest_path: Path, report_path: Path) -> bool:
    valid = load_json(report_path)
    checks = [
        ("valid certificate accepted", not validate_certificate(valid, manifest_path)),
    ]
    bad_hash = dict(valid)
    bad_hash["certificate_sha256"] = "0" * 64
    checks.append(
        (
            "tampered certificate hash rejected",
            "CERTIFICATE_HASH_MISMATCH" in validate_certificate(bad_hash, manifest_path),
        )
    )
    bad_status = dict(valid)
    bad_status["status"] = "FAIL"
    checks.append(
        (
            "failed status rejected",
            "CERTIFICATE_STATUS_FAILED" in validate_certificate(bad_status, manifest_path),
        )
    )
    ok = all(passed for _, passed in checks)
    payload = {
        "ok": ok,
        "checks": [{"name": name, "ok": passed} for name, passed in checks],
    }
    print(canonical_bytes(payload).decode("utf-8"))
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--check-report", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return 0 if self_test(args.manifest, args.report) else 1
    if args.check_report:
        certificate = load_json(args.report)
        errors = validate_certificate(certificate, args.manifest)
        payload = {
            "ok": not errors,
            "errors": errors,
            "report": str(args.report),
        }
        print(canonical_bytes(payload).decode("utf-8"))
        return 0 if not errors else 1

    certificate = build_certificate(args.manifest)
    errors = validate_certificate(certificate, args.manifest)
    if args.write_report and not errors:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(canonical_bytes(certificate) + b"\n")
    if not errors and args.report.exists() and not args.write_report:
        committed = load_json(args.report)
        errors.extend(validate_certificate(committed, args.manifest))
        if semantic_projection(committed) != semantic_projection(certificate):
            errors.append("LIVE_REPORT_SEMANTIC_MISMATCH")
    payload = {
        "ok": not errors,
        "errors": errors,
        "certificate": certificate,
    }
    print(canonical_bytes(payload).decode("utf-8"))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
