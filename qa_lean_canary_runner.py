#!/usr/bin/env python3
"""
qa_lean_canary_runner.py

Sidecar-only Lean canaries so conjecture failures are interpretable.
Writes a small .lean file per canary and runs Lean on it.
Emits proofs/canary_results.json with stdout/stderr + pass/fail.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class CanaryResult:
    name: str
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    lean_file: str


def _run(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_toolchain_canary() -> str:
    return """-- Toolchain Canary (no imports)

theorem canary_true : True := by trivial

theorem canary_rfl (n : Nat) : n = n := by rfl
"""


def build_import_canary(import_module: str) -> str:
    return f"""import {import_module}

-- Import Canary (QA module loads)

theorem canary_import_true : True := by trivial

theorem canary_import_rfl (n : Nat) : n = n := by rfl
"""


def build_invariant_canary(import_module: str, tuple_expr: str) -> str:
    return f"""import {import_module}

-- Invariant Canary (definitional equalities only)
-- tuple_expr must be a definitional constructor expression in your library.

def t := {tuple_expr}

-- Uncomment and adjust identifiers once you confirm them in your Lean module.
-- example : t.B = 1 := by rfl
-- example : t.D = 4 := by rfl
-- example : t.A = 9 := by rfl
-- example : t.X = 2 := by rfl
-- example : t.C = 4 := by rfl
-- example : t.F = 3 := by rfl
-- example : t.W = 8 := by rfl
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Lean canary runner (sidecar)")
    ap.add_argument("--workspace", required=True, help="Workspace dir")
    ap.add_argument("--lean-cmd", default="lean", help="Lean command, e.g. 'lean' or 'lake env lean'")
    ap.add_argument("--lean-cwd", default=None, help="Working directory for lean")
    ap.add_argument("--import-module", default=None, help="Lean module to import for QA canaries")
    ap.add_argument("--skip-import", action="store_true", help="Skip import canary")
    ap.add_argument("--skip-invariant", action="store_true", help="Skip invariant canary")
    ap.add_argument(
        "--tuple-expr",
        default="{ b := 1, e := 1, d := 2, a := 3 }",
        help="Lean constructor expression for invariant canary",
    )
    ap.add_argument(
        "--canary-lean-file",
        default=None,
        help="Path to a generated Lean file to compile as a canary",
    )
    ap.add_argument("--soft-fail", action="store_true", help="Do not fail run on canary failures")
    args = ap.parse_args()

    ws = Path(args.workspace)
    proofs_dir = ws / "proofs"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    results_path = proofs_dir / "canary_results.json"

    lean_cwd = args.lean_cwd or str(ws)
    lean_cmd_tokens = args.lean_cmd.split()

    results: List[CanaryResult] = []
    any_fail = False
    generated_compile_ok: Optional[bool] = None
    generated_compile_lean_file: Optional[str] = None

    def run_canary(name: str, content: str) -> None:
        nonlocal any_fail
        lean_file = proofs_dir / f"canary_{name}.lean"
        _write_text(lean_file, content)
        proc = _run(lean_cmd_tokens + [str(lean_file.resolve())], cwd=lean_cwd)
        ok = proc.returncode == 0
        if not ok:
            any_fail = True
        results.append(
            CanaryResult(
                name=name,
                ok=ok,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                lean_file=str(lean_file),
            )
        )

    run_canary("toolchain", build_toolchain_canary())

    if args.canary_lean_file:
        lean_path = Path(args.canary_lean_file)
        generated_compile_lean_file = str(lean_path)
        if not lean_path.exists():
            results.append(
                CanaryResult(
                    name="generated_compile",
                    ok=False,
                    returncode=2,
                    stdout="",
                    stderr=f"canary lean file not found: {lean_path}",
                    lean_file=str(lean_path),
                )
            )
            any_fail = True
            generated_compile_ok = False
        else:
            proc = _run(lean_cmd_tokens + [str(lean_path.resolve())], cwd=lean_cwd)
            ok = proc.returncode == 0
            if not ok:
                any_fail = True
            generated_compile_ok = ok
            results.append(
                CanaryResult(
                    name="generated_compile",
                    ok=ok,
                    returncode=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    lean_file=str(lean_path),
                )
            )

    if not args.skip_import:
        if not args.import_module:
            results.append(
                CanaryResult(
                    name="import",
                    ok=False,
                    returncode=2,
                    stdout="",
                    stderr="--import-module not provided; import canary skipped",
                    lean_file="",
                )
            )
            any_fail = True
        else:
            run_canary("import", build_import_canary(args.import_module))

    if not args.skip_invariant:
        if not args.import_module:
            results.append(
                CanaryResult(
                    name="invariant",
                    ok=False,
                    returncode=2,
                    stdout="",
                    stderr="--import-module not provided; invariant canary skipped",
                    lean_file="",
                )
            )
            any_fail = True
        else:
            run_canary("invariant", build_invariant_canary(args.import_module, args.tuple_expr))

    payload: dict[str, Any] = {
        "schema_version": "qa_lean_canaries@1",
        "workspace": str(ws),
        "lean_cmd": args.lean_cmd,
        "lean_cwd": lean_cwd,
        "import_module": args.import_module,
        "tuple_expr": args.tuple_expr,
        "generated_compile_ok": generated_compile_ok,
        "generated_compile_lean_file": generated_compile_lean_file,
        "results": [asdict(r) for r in results],
        "ok": not any_fail,
    }
    _write_text(results_path, json.dumps(payload, indent=2))

    if any_fail and not args.soft_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
