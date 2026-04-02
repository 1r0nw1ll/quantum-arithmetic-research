#!/usr/bin/env python3
"""
qa_theorem_discovery_orchestrator_rust.py

Torch-free QA theorem discovery pipeline using Rust backend for invariants.

Entry points:
- qa_graph_builder_rust.py
- qa_rust_embedding_miner.py
- qa_lean_verifier_rust.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RustOrchestrator")

SCHEMA_VERSION = "qa_rust_pipeline@1"
CANONICAL_VALIDATOR = Path(
    "Formalizing tuple drift in quantum-native learning/files/files(1)/validate_canonical_v2.py"
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


class QARustOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.workspace = Path(self.config["workspace"])
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.inputs_dir = self.workspace / "inputs"
        self.edges_dir = self.workspace / "edges"
        self.embeddings_dir = self.workspace / "embeddings"
        self.conjectures_dir = self.workspace / "conjectures"
        self.proofs_dir = self.workspace / "proofs"
        self.obstructions_dir = self.workspace / "obstructions"
        self.logs_dir = self.workspace / "logs"
        for d in (
            self.inputs_dir,
            self.edges_dir,
            self.embeddings_dir,
            self.conjectures_dir,
            self.proofs_dir,
            self.obstructions_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self._attach_file_logger()
        self.results = {
            "start_time": None,
            "end_time": None,
            "stages": {},
            "errors": [],
            "summary": {},
        }
        self.lean_available = None
        self.lean_check_output = ""

    def _attach_file_logger(self) -> None:
        log_path = self.logs_dir / "run.log"
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path):
                return
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(file_handler)

    def _run_stage(self, name: str, command: list, description: str) -> bool:
        logger.info("=" * 70)
        logger.info("STAGE: %s", name)
        logger.info(description)
        logger.info("=" * 70)

        stage_start = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        stage_time = time.time() - stage_start
        output = (result.stdout or "") + (result.stderr or "")
        failed = result.returncode != 0

        self.results["stages"][name] = {
            "status": "failed" if failed else "success",
            "duration": stage_time,
            "output": output[-1000:],
            "command": " ".join(command),
        }

        if failed:
            logger.error("\n✗ Stage failed after %.2fs", stage_time)
            logger.error(output[-1000:])
            self.results["errors"].append({
                "stage": name,
                "error": "command failed",
                "stderr": result.stderr,
            })
            return False

        logger.info("\n✓ Stage completed successfully in %.2fs", stage_time)
        return True

    def _check_lean_available(self) -> bool:
        if self.lean_available is not None:
            return self.lean_available

        lean_cmd = str(self.config.get("lean_canary_cmd", "lean")).split()
        command = lean_cmd + ["--version"]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            output = (result.stdout or "") + (result.stderr or "")
            self.lean_check_output = output[-500:]
            self.lean_available = result.returncode == 0
        except FileNotFoundError:
            self.lean_check_output = "Lean command not found"
            self.lean_available = False
        return self.lean_available

    @staticmethod
    def _classify_lean_failure(stderr: str) -> str:
        message = (stderr or "").lower()
        proof_markers = (
            "tactic failed",
            "unsolved goals",
            "unsolved goal",
            "failed to prove",
        )
        compile_markers = (
            "unknown constant",
            "unknown identifier",
            "unknown module prefix",
            "invalid syntax",
            "unexpected token",
            "parse error",
            "type mismatch",
            "application type mismatch",
            "failed to synthesize",
            "failed to compile",
            "no such file or directory",
            "failed to open file",
            "could not resolve import",
            "unknown package",
            "error during download",
            "failed to download",
        )
        if any(marker in message for marker in proof_markers):
            return "PROOF_FAILED"
        if any(marker in message for marker in compile_markers):
            return "COMPILE_ERROR"
        if "error:" in message:
            return "COMPILE_ERROR"
        return "PROOF_FAILED"

    def stage_0_preflight(self) -> bool:
        validator = Path(
            CANONICAL_VALIDATOR
        )
        command = ["python", str(validator)]

        logger.info("=" * 70)
        logger.info("STAGE: Stage 0: Canonical Preflight")
        logger.info("Validating canonical QA invariants and checksums")
        logger.info("=" * 70)

        stage_start = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        stage_time = time.time() - stage_start
        output = (result.stdout or "") + (result.stderr or "")
        failed = result.returncode != 0 or "VALIDATION FAILED" in output

        self.results["stages"]["Stage 0: Canonical Preflight"] = {
            "status": "failed" if failed else "success",
            "duration": stage_time,
            "output": output[-1000:],
            "command": " ".join(command),
        }

        if failed:
            logger.error("\n✗ Stage failed after %.2fs", stage_time)
            logger.error(output[-1000:])
            self.results["errors"].append({
                "stage": "Stage 0: Canonical Preflight",
                "error": "canonical validation failed",
                "stderr": result.stderr,
            })
            return False

        logger.info("\n✓ Stage completed successfully in %.2fs", stage_time)
        return True

    def stage_1_build_graph(self) -> bool:
        command = [
            "python", "qa_graph_builder_rust.py",
            "--input", str(self.config["dataset"]),
            "--output", str(self.config["graph_edges_output"]),
            "--stats", str(self.config["graph_stats_output"]),
        ]
        return self._run_stage(
            "Stage 1: Graph Construction",
            command,
            "Building canonical QA edge list (sigma, mu, lambda2, nu)",
        )

    def stage_2_embeddings(self) -> bool:
        command = [
            "python", "qa_rust_embedding_miner.py",
            "--dataset", str(self.config["dataset"]),
            "--embeddings-out", str(self.config["embeddings_output"]),
            "--clusters-out", str(self.config["clusters_output"]),
            "--conjectures-out", str(self.config["conjectures_output"]),
            "--conjectures-list-out", str(self.config["conjectures_list_output"]),
            "--k", str(self.config["cluster_k"]),
            "--max-iters", str(self.config["cluster_iters"]),
            "--seed", str(self.config["cluster_seed"]),
            "--min-cluster", str(self.config["min_cluster_size"]),
            "--qa-lab-root", str(self.config["qa_lab_root"]),
        ]
        if self.config.get("skip_spot_check"):
            command.append("--skip-spot-check")
        if self.config.get("spot_check_n") is not None:
            command.extend(["--spot-check-n", str(self.config["spot_check_n"])])
        return self._run_stage(
            "Stage 2: Rust Embeddings + Conjectures",
            command,
            "Computing invariants in Rust, building embeddings, clustering, mining conjectures",
        )

    def stage_2b_lean_canaries(self) -> bool:
        if self.config.get("skip_lean_canary"):
            logger.info("Skipping Lean canaries (--skip-lean-canary).")
            self.results["stages"]["Stage 2b: Lean Canaries"] = {
                "status": "skipped",
                "duration": 0.0,
            }
            return True

        if not self._check_lean_available():
            logger.warning("Lean not available; skipping canaries.")
            self.results["stages"]["Stage 2b: Lean Canaries"] = {
                "status": "skipped",
                "duration": 0.0,
                "output": self.lean_check_output,
                "command": " ".join(str(self.config.get("lean_canary_cmd", "lean")).split() + ["--version"]),
            }
            return True

        known_good_path = self.proofs_dir / "canary_known_good.lean"
        known_good_path.write_text(
            "theorem known_good : True := by trivial\n",
            encoding="utf-8",
        )

        command = [
            "python", "qa_lean_canary_runner.py",
            "--workspace", str(self.workspace),
            "--lean-cmd", str(self.config["lean_canary_cmd"]),
            "--canary-lean-file", str(known_good_path),
        ]
        import_module = self.config.get("lean_canary_import_module")
        if import_module:
            command.extend(["--import-module", str(import_module)])
        else:
            command.append("--skip-import")
            command.append("--skip-invariant")
        if self.config.get("skip_lean_canary_import"):
            command.append("--skip-import")
        if self.config.get("skip_lean_canary_invariant"):
            command.append("--skip-invariant")
        if self.config.get("lean_canary_tuple_expr"):
            command.extend(["--tuple-expr", str(self.config["lean_canary_tuple_expr"])])
        if self.config.get("lean_canary_soft_fail"):
            command.append("--soft-fail")

        return self._run_stage(
            "Stage 2b: Lean Canaries",
            command,
            "Running Lean toolchain/import/invariant canaries",
        )

    def stage_3_verify_lean(self) -> bool:
        if self.config.get("skip_lean_verify"):
            logger.info("Skipping Lean verification (--skip-lean-verify).")
            self.results["stages"]["Stage 3: Formal Verification"] = {
                "status": "skipped",
                "duration": 0.0,
            }
            return True

        if not self._check_lean_available():
            logger.warning("Lean not available; skipping verification.")
            self.results["stages"]["Stage 3: Formal Verification"] = {
                "status": "skipped",
                "duration": 0.0,
                "output": self.lean_check_output,
                "command": " ".join(str(self.config.get("lean_canary_cmd", "lean")).split() + ["--version"]),
            }
            return True

        command = [
            "python", "qa_lean_verifier_rust.py",
            "--conjectures", str(self.config["conjectures_list_output"]),
            "--output-dir", str(self.config["lean_proofs_dir"]),
            "--max-conjectures", str(self.config["max_conjectures_to_verify"]),
        ]
        if self.config.get("tier2_lemmas"):
            command.extend(["--tier2-lemmas", str(self.config["tier2_lemmas"])])
        return self._run_stage(
            "Stage 3: Formal Verification",
            command,
            "Verifying conjectures with Lean 4",
        )

    def stage_4_export_results(self) -> bool:
        logger.info("=" * 70)
        logger.info("STAGE: Stage 4: Results Export")
        logger.info("Generating summary report")
        logger.info("=" * 70)

        stage_start = time.time()
        try:
            summary = self.generate_summary_report(
                self.config["conjectures_output"],
                self.config["lean_proofs_dir"] / "proof_records.json",
            )
            summary_path = self.workspace / "discovery_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            obstruction_path = self.generate_tier2_obstructions(
                self.config["lean_proofs_dir"] / "proof_records.json"
            )
            generator_ledger_path = None
            if obstruction_path:
                generator_ledger_path = self.generate_generator_ledger(summary, obstruction_path)
            self.generate_obstructions_ledger(summary, obstruction_path, generator_ledger_path)
            self.write_manifest()
            self.generate_text_report(summary)
            stage_time = time.time() - stage_start
            self.results["stages"]["Stage 4: Export"] = {
                "status": "success",
                "duration": stage_time,
            }
            logger.info("✓ Summary report saved to %s", summary_path)
            return True
        except Exception as exc:
            logger.error("✗ Export failed: %s", exc)
            return False

    def generate_summary_report(self, conjectures_path: Path, proofs_path: Path) -> dict:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {k: str(v) for k, v in self.config.items()},
            "statistics": {},
        }

        if Path(conjectures_path).exists():
            with open(conjectures_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            conjectures = payload.get("conjectures") if isinstance(payload, dict) else payload
            summary["statistics"]["total_conjectures"] = len(conjectures)
            summary["statistics"]["top_conjecture_score"] = (
                conjectures[0]["rank_score"] if conjectures else 0
            )

        proof_failures = 0
        compile_errors = 0
        rfl_total = 0
        simp_total = 0
        other_total = 0
        verified_conjectures = 0
        verified_aux = 0
        failed_conjectures = 0
        failed_aux = 0
        tier2_attempted = 0
        tier2_verified = 0
        tier2_failed = 0
        tier2_compile_errors = 0
        tier2_failed_by_reason = {}
        if Path(proofs_path).exists():
            with open(proofs_path, "r", encoding="utf-8") as f:
                proofs = json.load(f)
            summary["statistics"]["verified_proofs"] = len(proofs.get("verified", []))
            for record in proofs.get("verified", []):
                rfl_total += int(record.get("rfl_count", 0) or 0)
                simp_total += int(record.get("simp_count", 0) or 0)
                other_total += int(record.get("lemma_count", 0) or 0) - (
                    int(record.get("rfl_count", 0) or 0)
                    + int(record.get("simp_count", 0) or 0)
                )
                verified_conjectures += int(record.get("conjecture_count", 0) or 0)
                verified_aux += int(record.get("aux_count", 0) or 0)
                tier2_attempted += int(record.get("tier2_attempted", 0) or 0)
                if record.get("tier2_verified") is True:
                    tier2_verified += int(record.get("tier2_attempted", 0) or 0)
                elif record.get("tier2_verified") is False:
                    combined = "\n".join([
                        record.get("tier2_output", ""),
                        record.get("tier2_errors", ""),
                    ])
                    if self._classify_lean_failure(combined) == "COMPILE_ERROR":
                        tier2_compile_errors += int(record.get("tier2_attempted", 0) or 0)
                    else:
                        tier2_failed += int(record.get("tier2_attempted", 0) or 0)
                    reason = record.get("tier2_reason") or "UNKNOWN"
                    tier2_failed_by_reason[reason] = tier2_failed_by_reason.get(reason, 0) + int(
                        record.get("tier2_attempted", 0) or 0
                    )
            for record in proofs.get("failed", []):
                combined = "\n".join([
                    record.get("lean_output", ""),
                    record.get("lean_errors", ""),
                ])
                classification = self._classify_lean_failure(combined)
                if classification == "COMPILE_ERROR":
                    compile_errors += 1
                else:
                    proof_failures += 1
                rfl_total += int(record.get("rfl_count", 0) or 0)
                simp_total += int(record.get("simp_count", 0) or 0)
                other_total += int(record.get("lemma_count", 0) or 0) - (
                    int(record.get("rfl_count", 0) or 0)
                    + int(record.get("simp_count", 0) or 0)
                )
                failed_conjectures += int(record.get("conjecture_count", 0) or 0)
                failed_aux += int(record.get("aux_count", 0) or 0)
                tier2_attempted += int(record.get("tier2_attempted", 0) or 0)
                if record.get("tier2_verified") is True:
                    tier2_verified += int(record.get("tier2_attempted", 0) or 0)
                elif record.get("tier2_verified") is False:
                    combined = "\n".join([
                        record.get("tier2_output", ""),
                        record.get("tier2_errors", ""),
                    ])
                    if self._classify_lean_failure(combined) == "COMPILE_ERROR":
                        tier2_compile_errors += int(record.get("tier2_attempted", 0) or 0)
                    else:
                        tier2_failed += int(record.get("tier2_attempted", 0) or 0)
                    reason = record.get("tier2_reason") or "UNKNOWN"
                    tier2_failed_by_reason[reason] = tier2_failed_by_reason.get(reason, 0) + int(
                        record.get("tier2_attempted", 0) or 0
                    )
            summary["statistics"]["failed_proofs"] = proof_failures
            summary["statistics"]["compile_errors"] = compile_errors
            summary["statistics"]["rfl_proofs"] = rfl_total
            summary["statistics"]["simp_proofs"] = simp_total
            summary["statistics"]["other_proofs"] = other_total
            summary["statistics"]["verified_conjectures"] = verified_conjectures
            summary["statistics"]["verified_aux"] = verified_aux
            summary["statistics"]["failed_conjectures"] = failed_conjectures
            summary["statistics"]["failed_aux"] = failed_aux
            summary["statistics"]["tier2_attempted"] = tier2_attempted
            summary["statistics"]["tier2_verified"] = tier2_verified
            summary["statistics"]["tier2_failed"] = tier2_failed
            summary["statistics"]["tier2_compile_errors"] = tier2_compile_errors
            summary["statistics"]["tier2_failed_by_reason"] = tier2_failed_by_reason
            total_lemmas = rfl_total + simp_total + other_total
            summary["statistics"]["rfl_fraction"] = (rfl_total / total_lemmas) if total_lemmas else 0.0

        canary_path = self.workspace / "proofs" / "canary_results.json"
        generated_compile_ok = None
        if canary_path.exists():
            with open(canary_path, "r", encoding="utf-8") as f:
                canary = json.load(f)
            summary["statistics"]["lean_canary_ok"] = bool(canary.get("ok"))
            generated_compile_ok = canary.get("generated_compile_ok")
            summary["statistics"]["generated_compile_ok"] = generated_compile_ok

        summary["statistics"]["lean_available"] = self.lean_available
        if self.config.get("skip_lean_verify"):
            summary["statistics"]["lean_verify_status"] = "SKIPPED"
        elif self.lean_available is False:
            summary["statistics"]["lean_verify_status"] = "TOOLCHAIN_MISSING"
        else:
            if generated_compile_ok is False:
                summary["statistics"]["lean_verify_status"] = "COMPILE_ERROR"
            elif compile_errors > 0:
                summary["statistics"]["lean_verify_status"] = "COMPILE_ERROR"
            elif proof_failures > 0:
                summary["statistics"]["lean_verify_status"] = "PROOF_FAILED"
            else:
                summary["statistics"]["lean_verify_status"] = "OK"

        return summary

    def generate_tier2_obstructions(self, proofs_path: Path) -> Path | None:
        if not Path(proofs_path).exists():
            return None

        with open(proofs_path, "r", encoding="utf-8") as f:
            proofs = json.load(f)

        records = []
        for bucket in ("verified", "failed"):
            for record in proofs.get(bucket, []):
                if record.get("tier2_verified") is not False:
                    continue
                tier2_file = record.get("tier2_file")
                tier2_output = record.get("tier2_output", "")
                tier2_errors = record.get("tier2_errors", "")
                reason = record.get("tier2_reason") or "UNKNOWN"
                generator_extensions = record.get("tier2_lemmas") or []
                cluster_id = record.get("cluster_id")
                representative = record.get("conjecture", {}).get("representative", {})
                tuple_hash = representative.get("tuple_hash")
                packet_hash = representative.get("packet_hash")
                for lemma in record.get("lemmas", []):
                    if lemma.get("tier") != "2" or lemma.get("role") != "conjecture":
                        continue
                    lemma_name = lemma.get("name") or ""
                    formula = lemma.get("formula") or ""
                    goal_fingerprint = _sha256_text(f"{lemma_name}:{formula}")
                    if packet_hash:
                        state_id = f"{packet_hash}:{lemma_name}"
                    elif tuple_hash:
                        state_id = f"{tuple_hash}:{lemma_name}"
                    else:
                        state_id = f"cluster_{cluster_id}:{lemma_name}"
                    records.append({
                        "state_id": state_id,
                        "cluster_id": cluster_id,
                        "lemma": lemma,
                        "proof_goal_fingerprint": goal_fingerprint,
                        "tuple_hash": tuple_hash,
                        "packet_hash": packet_hash,
                        "failure_class": reason,
                        "generator_set": ["definitional", "simp"],
                        "generator_extensions": generator_extensions,
                        "witness": {
                            "tier2_file": tier2_file,
                            "output": tier2_output,
                            "errors": tier2_errors,
                        },
                    })

        obstruction_path = self.obstructions_dir / "tier2_obstructions.json"
        payload = {
            "schema_version": "qa_tier2_obstructions@1",
            "workspace": str(self.workspace),
            "dataset": str(self.config.get("dataset")),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "records": records,
        }
        obstruction_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return obstruction_path

    def _load_static_obstructions(self) -> list[str]:
        index_path = Path("obstructions/static/index.json")
        if not index_path.exists():
            return []
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        ledgers = data.get("ledgers", [])
        if isinstance(ledgers, list):
            return [str(item) for item in ledgers]
        return []

    def generate_generator_ledger(self, summary: dict, obstruction_path: Path) -> Path:
        ledger_path = self.obstructions_dir / "generator_ledger.json"
        stats = summary.get("statistics", {})
        payload = {
            "schema_version": "qa_generator_ledger@1",
            "workspace": str(self.workspace),
            "dataset": str(self.config.get("dataset")),
            "generator_set_base": ["definitional", "simp"],
            "generator_extensions": [
                item.strip()
                for item in str(self.config.get("tier2_lemmas", "")).split(",")
                if item.strip()
            ],
            "summary": {
                "tier2_attempted": stats.get("tier2_attempted", 0),
                "tier2_verified": stats.get("tier2_verified", 0),
                "tier2_failed": stats.get("tier2_failed", 0),
                "tier2_failed_by_reason": stats.get("tier2_failed_by_reason", {}),
            },
            "obstructions": {
                "path": str(obstruction_path),
                "sha256": _sha256_file(obstruction_path),
            },
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        ledger_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ledger_path

    def generate_obstructions_ledger(
        self,
        summary: dict,
        obstruction_path: Path | None,
        generator_ledger_path: Path | None,
    ) -> Path:
        ledger_path = self.obstructions_dir / "obstructions_ledger.json"
        static_ledgers = self._load_static_obstructions()
        payload = {
            "schema_version": "qa_obstructions_ledger@1",
            "workspace": str(self.workspace),
            "dataset": str(self.config.get("dataset")),
            "static_ledgers": static_ledgers,
            "computed_ledgers": (
                [str(obstruction_path)] if obstruction_path else []
            ),
            "generator_ledger": str(generator_ledger_path) if generator_ledger_path else None,
            "summary": {
                "tier2_attempted": summary.get("statistics", {}).get("tier2_attempted", 0),
                "tier2_verified": summary.get("statistics", {}).get("tier2_verified", 0),
                "tier2_failed": summary.get("statistics", {}).get("tier2_failed", 0),
                "tier2_failed_by_reason": summary.get("statistics", {}).get("tier2_failed_by_reason", {}),
            },
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        ledger_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ledger_path

    def generate_text_report(self, summary: dict) -> None:
        report_path = self.workspace / "DISCOVERY_REPORT.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("QA RUST PIPELINE - FINAL REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Workspace: {self.workspace}\n\n")
            f.write("RESULTS SUMMARY:\n")
            f.write("-" * 70 + "\n")
            stats = summary.get("statistics", {})
            f.write(f"Total Conjectures Discovered: {stats.get('total_conjectures', 0)}\n")
            f.write(f"Verified Proofs: {stats.get('verified_proofs', 0)}\n")
            f.write(f"Proof Failures: {stats.get('failed_proofs', 0)}\n")
            if "compile_errors" in stats:
                f.write(f"Compile Errors: {stats.get('compile_errors', 0)}\n")
            if "rfl_proofs" in stats:
                f.write(f"RFL Proofs: {stats.get('rfl_proofs', 0)}\n")
            if "simp_proofs" in stats:
                f.write(f"Simp Proofs: {stats.get('simp_proofs', 0)}\n")
            if "other_proofs" in stats:
                f.write(f"Other Proofs: {stats.get('other_proofs', 0)}\n")
            if "verified_conjectures" in stats:
                f.write(f"Verified Conjectures: {stats.get('verified_conjectures', 0)}\n")
            if "verified_aux" in stats:
                f.write(f"Verified Aux: {stats.get('verified_aux', 0)}\n")
            if "failed_conjectures" in stats:
                f.write(f"Failed Conjectures: {stats.get('failed_conjectures', 0)}\n")
            if "failed_aux" in stats:
                f.write(f"Failed Aux: {stats.get('failed_aux', 0)}\n")
            if "tier2_attempted" in stats:
                f.write(f"Tier2 Attempted: {stats.get('tier2_attempted', 0)}\n")
            if "tier2_verified" in stats:
                f.write(f"Tier2 Verified: {stats.get('tier2_verified', 0)}\n")
            if "tier2_failed" in stats:
                f.write(f"Tier2 Failed: {stats.get('tier2_failed', 0)}\n")
            if "tier2_compile_errors" in stats:
                f.write(f"Tier2 Compile Errors: {stats.get('tier2_compile_errors', 0)}\n")
            if "tier2_failed_by_reason" in stats:
                f.write(f"Tier2 Failed By Reason: {stats.get('tier2_failed_by_reason')}\n")
            if "rfl_fraction" in stats:
                f.write(f"RFL Fraction: {stats.get('rfl_fraction', 0):.3f}\n")
            f.write(f"Top Conjecture Score: {stats.get('top_conjecture_score', 0):.2f}\n\n")
            if "lean_available" in stats:
                f.write(f"Lean Available: {stats.get('lean_available')}\n")
            if "lean_verify_status" in stats:
                f.write(f"Lean Verify Status: {stats.get('lean_verify_status')}\n")
            if "lean_canary_ok" in stats:
                f.write(f"Lean Canary OK: {stats.get('lean_canary_ok')}\n")
            if "generated_compile_ok" in stats:
                f.write(f"Generated Compile OK: {stats.get('generated_compile_ok')}\n")
            f.write("\n")
            f.write("STAGE EXECUTION:\n")
            f.write("-" * 70 + "\n")
            for stage_name, stage_data in self.results["stages"].items():
                status = stage_data["status"]
                duration = stage_data.get("duration", 0)
                f.write(f"{stage_name}: {status.upper()} ({duration:.2f}s)\n")
            total_time = self.results.get("end_time", 0) - self.results.get("start_time", 0)
            f.write(f"\nTotal Pipeline Duration: {total_time:.2f}s ({total_time/60:.2f}m)\n")
            f.write("\n" + "=" * 70 + "\n")
        logger.info("✓ Text report saved to %s", report_path)

    def write_manifest(self) -> None:
        files = [
            self.config["dataset"],
            self.config["graph_edges_output"],
            self.config["graph_stats_output"],
            self.config["embeddings_output"],
            self.config["clusters_output"],
            self.config["conjectures_output"],
            self.config["conjectures_list_output"],
            self.config["lean_proofs_dir"] / "proof_records.json",
            self.workspace / "proofs" / "canary_results.json",
            self.workspace / "obstructions" / "tier2_obstructions.json",
            self.workspace / "obstructions" / "generator_ledger.json",
            self.workspace / "obstructions" / "obstructions_ledger.json",
            Path("obstructions/static/index.json"),
            self.workspace / "discovery_summary.json",
            self.workspace / "DISCOVERY_REPORT.txt",
            self.logs_dir / "run.log",
        ]
        for ledger_path in self._load_static_obstructions():
            files.append(Path(ledger_path))
        manifest_files = []
        for path in files:
            path = Path(path)
            if not path.exists():
                continue
            manifest_files.append({
                "path": str(path),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            })

        qa_lab_rs = Path(self.config["qa_lab_root"]) / "qa_lab_rs.so"
        qa_lab_rs_meta = None
        if qa_lab_rs.exists():
            qa_lab_rs_meta = {
                "path": str(qa_lab_rs),
                "sha256": _sha256_file(qa_lab_rs),
                "mtime": qa_lab_rs.stat().st_mtime,
                "size": qa_lab_rs.stat().st_size,
            }

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "git_commit": _git_commit(),
            "dataset_sha256": _sha256_file(Path(self.config["dataset"])),
            "dataset_rows": int(self.config["dataset_rows"]),
            "qa_lab_rs": qa_lab_rs_meta,
            "files": manifest_files,
        }
        manifest_path = self.workspace / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("✓ Manifest saved to %s", manifest_path)

    def prepare_inputs(self) -> None:
        dataset_src = Path(self.config["dataset"]).resolve()
        dataset_dst = self.inputs_dir / "dataset.csv"
        if not dataset_dst.exists():
            shutil.copy2(dataset_src, dataset_dst)
        self.config["dataset"] = str(dataset_dst)
        self.config["dataset_rows"] = sum(1 for _ in dataset_dst.open("r", encoding="utf-8")) - 1

    def run(self) -> bool:
        logger.info("╔" + "=" * 68 + "╗")
        logger.info("║" + " " * 68 + "║")
        logger.info("║" + "  QA RUST THEOREM DISCOVERY PIPELINE".center(68) + "║")
        logger.info("║" + " " * 68 + "║")
        logger.info("╚" + "=" * 68 + "╝")

        self.results["start_time"] = time.time()

        if self.config.get("run_preflight", True):
            if not self.stage_0_preflight():
                logger.error("Pipeline aborted: Stage 0 failed")
                return False

        self.prepare_inputs()

        if not self.stage_1_build_graph():
            logger.error("Pipeline aborted: Stage 1 failed")
            return False

        if not self.stage_2_embeddings():
            logger.error("Pipeline aborted: Stage 2 failed")
            return False

        if not self.stage_2b_lean_canaries():
            logger.error("Pipeline aborted: Stage 2b failed")
            return False

        if not self.stage_3_verify_lean():
            logger.warning("Stage 3 had issues, but continuing...")

        self.results["end_time"] = time.time()
        self.stage_4_export_results()
        total_time = self.results["end_time"] - self.results["start_time"]

        logger.info("\n" + "╔" + "=" * 68 + "╗")
        logger.info("║" + " " * 68 + "║")
        logger.info("║" + "  PIPELINE COMPLETE!".center(68) + "║")
        logger.info("║" + " " * 68 + "║")
        logger.info("╚" + "=" * 68 + "╝")
        logger.info("\nTotal Duration: %.2fs (%.2fm)", total_time, total_time / 60)
        logger.info("Workspace: %s", self.workspace)
        logger.info("\nCheck %s for full results", self.workspace / "DISCOVERY_REPORT.txt")

        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rust-backed QA theorem discovery pipeline")
    parser.add_argument("--dataset", default="qa_10000_balanced_tuples.csv", help="Input dataset")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory (default: workspaces/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--workspace-root", default="workspaces", help="Workspace root directory")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    parser.add_argument("--k", type=int, default=32, help="K-means clusters")
    parser.add_argument("--cluster-iters", type=int, default=50, help="K-means max iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--min-cluster", type=int, default=3, help="Minimum cluster size")
    parser.add_argument("--max-conjectures", type=int, default=20, help="Max conjectures to verify")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip canonical validator preflight")
    parser.add_argument("--qa-lab-root", default="qa_lab", help="Path to qa_lab root")
    parser.add_argument("--skip-spot-check", action="store_true", help="Skip canonical invariant check")
    parser.add_argument("--spot-check-n", type=int, default=50, help="Spot-check sample size")
    parser.add_argument("--skip-lean-canary", action="store_true", help="Skip Lean canaries")
    parser.add_argument("--lean-canary-cmd", default="lean", help="Lean command for canaries")
    parser.add_argument("--lean-canary-import-module", default=None, help="Lean module for import canary")
    parser.add_argument("--skip-lean-canary-import", action="store_true", help="Skip import canary")
    parser.add_argument("--skip-lean-canary-invariant", action="store_true", help="Skip invariant canary")
    parser.add_argument("--lean-canary-tuple-expr", default=None, help="Tuple expression for invariant canary")
    parser.add_argument("--lean-canary-soft-fail", action="store_true", help="Do not fail on canary failure")
    parser.add_argument("--skip-lean-verify", action="store_true", help="Skip Lean verification stage")
    parser.add_argument("--tier2-lemmas", default="", help="Comma-separated lemmas to add to Tier-2 simp set")

    args = parser.parse_args()

    if args.workspace:
        workspace = Path(args.workspace)
    else:
        run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        workspace = Path(args.workspace_root) / f"run_{run_id}"
    config = {
        "dataset": args.dataset,
        "workspace": workspace,
        "graph_edges_output": workspace / "edges" / "qa_graph_edges.csv",
        "graph_stats_output": workspace / "edges" / "edge_stats.json",
        "embeddings_output": workspace / "embeddings" / "embeddings.npy",
        "clusters_output": workspace / "embeddings" / "cluster_labels.npy",
        "conjectures_output": workspace / "conjectures" / "conjectures.json",
        "conjectures_list_output": workspace / "conjectures" / "conjectures_list.json",
        "lean_proofs_dir": workspace / "proofs",
        "qa_lab_root": args.qa_lab_root,
        "cluster_k": args.k,
        "cluster_iters": args.cluster_iters,
        "cluster_seed": args.seed,
        "min_cluster_size": args.min_cluster,
        "max_conjectures_to_verify": args.max_conjectures,
        "run_preflight": not args.skip_preflight,
        "skip_spot_check": args.skip_spot_check,
        "spot_check_n": args.spot_check_n,
        "skip_lean_canary": args.skip_lean_canary,
        "lean_canary_cmd": args.lean_canary_cmd,
        "lean_canary_import_module": args.lean_canary_import_module,
        "skip_lean_canary_import": args.skip_lean_canary_import,
        "skip_lean_canary_invariant": args.skip_lean_canary_invariant,
        "lean_canary_tuple_expr": args.lean_canary_tuple_expr,
        "lean_canary_soft_fail": args.lean_canary_soft_fail,
        "skip_lean_verify": args.skip_lean_verify,
        "tier2_lemmas": args.tier2_lemmas,
    }

    try:
        orchestrator = QARustOrchestrator(config)
        ok = orchestrator.run()
        return 0 if ok else 1
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
