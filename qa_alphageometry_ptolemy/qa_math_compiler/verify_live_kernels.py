#!/usr/bin/env python3
"""Run minimal proofs through installed Lean, Coq/Rocq, and Isabelle kernels."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


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


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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


def verify_lean(root: Path) -> Dict[str, object]:
    executable = find_executable(
        ["lean"],
        [str(Path.home() / ".elan" / "bin" / "lean")],
    )
    if executable is None:
        return {"ok": False, "fail_type": "EXECUTABLE_MISSING", "assistant_id": "lean4"}
    source_path = root / "LiveLean.lean"
    source_path.write_text(LEAN_SOURCE, encoding="utf-8")
    version = run([executable, "--version"], root)
    proof = run([executable, str(source_path)], root)
    return {
        "ok": proof.returncode == 0,
        "assistant_id": "lean4",
        "executable": executable,
        "version": first_line(version.stdout or version.stderr),
        "source_sha256": source_hash(LEAN_SOURCE),
        "returncode": proof.returncode,
        "stdout": proof.stdout.strip(),
        "stderr": proof.stderr.strip(),
    }


def verify_coq(root: Path) -> Dict[str, object]:
    executable = find_executable(["coqc"])
    command_prefix: List[str]
    if executable is not None:
        command_prefix = [executable]
    else:
        rocq = find_executable(["rocq"])
        if rocq is None:
            return {"ok": False, "fail_type": "EXECUTABLE_MISSING", "assistant_id": "coq"}
        executable = rocq
        command_prefix = [rocq, "compile"]
    source_path = root / "LiveCoq.v"
    source_path.write_text(COQ_SOURCE, encoding="utf-8")
    version = run([executable, "--version"], root)
    proof = run(command_prefix + [str(source_path)], root)
    return {
        "ok": proof.returncode == 0,
        "assistant_id": "coq",
        "executable": executable,
        "version": first_line(version.stdout or version.stderr),
        "source_sha256": source_hash(COQ_SOURCE),
        "returncode": proof.returncode,
        "stderr": proof.stderr.strip(),
    }


def verify_isabelle(root: Path) -> Dict[str, object]:
    executable = find_executable(
        ["isabelle"],
        [
            "/Applications/Isabelle.app/Isabelle/bin/isabelle",
            "/Applications/Isabelle2025.app/Isabelle/bin/isabelle",
            "/Applications/Isabelle2025-2.app/bin/isabelle",
        ],
    )
    if executable is None:
        return {"ok": False, "fail_type": "EXECUTABLE_MISSING", "assistant_id": "isabelle"}
    session_dir = root / "isabelle"
    session_dir.mkdir()
    user_home = root / "isabelle_user"
    user_settings = user_home / ".isabelle" / "Isabelle2025-2" / "etc"
    user_settings.mkdir(parents=True)
    (user_settings / "settings").write_text(
        'ISABELLE_TOOL_JAVA_OPTIONS="-Djava.awt.headless=true -Xms128m -Xmx1g -Xss8m"\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["USER_HOME"] = str(user_home)
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
    (session_dir / "ROOT").write_text(ISABELLE_ROOT, encoding="utf-8")
    (session_dir / "QA_Math_Compiler_Live.thy").write_text(
        ISABELLE_SOURCE,
        encoding="utf-8",
    )
    version = run([executable, "version"], root, env=environment)
    proof = run(
        [executable, "build", "-D", str(session_dir), "QA_Math_Compiler_Live"],
        root,
        timeout=600,
        env=environment,
    )
    return {
        "ok": proof.returncode == 0,
        "assistant_id": "isabelle",
        "executable": executable,
        "version": first_line(version.stdout or version.stderr),
        "source_sha256": source_hash(ISABELLE_SOURCE + "\n" + ISABELLE_ROOT),
        "returncode": proof.returncode,
        "stdout": proof.stdout.strip(),
        "stderr": proof.stderr.strip(),
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qa_math_compiler_kernels_") as tmp:
        root = Path(tmp)
        results = [verify_lean(root), verify_coq(root), verify_isabelle(root)]
    payload = {
        "ok": all(result.get("ok") is True for result in results),
        "schema_id": "QA_MATH_COMPILER_LIVE_KERNEL_REPORT.v1",
        "results": results,
    }
    print(canonical(payload))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
