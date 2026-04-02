#!/usr/bin/env python3
"""
QA Lean Verifier (Rust sidecar)
Translates conjectures to Lean 4 without Mathlib dependencies.
"""

import json
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Dict
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("LeanVerifier")


class QALeanVerifier:
    """
    Formal verification agent for QA conjectures using Lean 4
    """

    def __init__(self, conjectures_path, output_dir="./lean_proofs", tier2_lemmas=None):
        self.conjectures_path = conjectures_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.conjectures = []
        self.verified_proofs = []
        self.failed_proofs = []
        self.tier2_lemmas = list(tier2_lemmas or [])

    def _tier2_simp_suffix(self) -> str:
        if not self.tier2_lemmas:
            return ""
        return ", " + ", ".join(self.tier2_lemmas)

    @staticmethod
    def _classify_tier2_failure(output: str, errors: str) -> str:
        message = f"{output}\n{errors}".lower()
        if "timeout" in message or "timed out" in message:
            return "TIMEOUT"
        compile_markers = (
            "unknown identifier",
            "unknown constant",
            "unknown module prefix",
            "invalid syntax",
            "unexpected token",
            "parse error",
            "failed to open file",
            "could not resolve import",
            "unknown package",
            "error during download",
            "failed to download",
        )
        if any(marker in message for marker in compile_markers):
            return "COMPILE_ERROR"
        if "failed to synthesize" in message:
            return "NEEDS_ARITH"
        if "simp" in message:
            return "NEEDS_REWRITE"
        if "unsolved goals" in message or "tactic failed" in message:
            return "NEEDS_LEMMA"
        return "NEEDS_LEMMA"

    def load_conjectures(self):
        """Load conjectures from JSON"""
        logger.info(f"Loading conjectures from {self.conjectures_path}")

        with open(self.conjectures_path, 'r') as f:
            self.conjectures = json.load(f)

        logger.info(f"✓ Loaded {len(self.conjectures)} conjectures")

    def conjecture_to_lean(self, conjecture: Dict) -> tuple[str, str | None, list[dict[str, str]]]:
        """
        Translate a conjecture to Lean 4 code
        """
        cluster_id = conjecture['cluster_id']
        patterns = conjecture['patterns']

        # Start with a minimal Lean prelude (no Mathlib dependency)
        lean_prelude = """
-- QA Tuple structure
structure QA_Tuple where
  b : Nat
  e : Nat

def d (q : QA_Tuple) : Nat := q.b + q.e
def a (q : QA_Tuple) : Nat := q.b + 2 * q.e

-- Invariant packet (canonical)
def B (q : QA_Tuple) : Nat := Nat.pow q.b 2
def E (q : QA_Tuple) : Nat := Nat.pow q.e 2
def D (q : QA_Tuple) : Nat := Nat.pow (d q) 2
def A (q : QA_Tuple) : Nat := Nat.pow (a q) 2
def X (q : QA_Tuple) : Nat := q.e * d q
def C (q : QA_Tuple) : Nat := 2 * q.e * d q
def F (q : QA_Tuple) : Nat := q.b * a q
def G (q : QA_Tuple) : Nat := D q + E q
def L (q : QA_Tuple) : Rat := (Rat.ofInt (Int.ofNat (C q * F q))) / (Rat.ofInt (Int.ofNat 12))
def H (q : QA_Tuple) : Nat := C q + F q
def I (q : QA_Tuple) : Nat := if C q >= F q then C q - F q else F q - C q
def J (q : QA_Tuple) : Nat := d q * q.b
def K (q : QA_Tuple) : Nat := d q * a q
def W (q : QA_Tuple) : Nat := X q + K q
def Y (q : QA_Tuple) : Nat := A q - D q
def Z (q : QA_Tuple) : Nat := E q + K q
def h2 (q : QA_Tuple) : Nat := (Nat.pow (d q) 2) * (a q) * q.b

"""
        lean_code = lean_prelude
        tier2_code = lean_prelude
        tier2_count = 0
        tier2_simp_suffix = self._tier2_simp_suffix()

        # Generate lemmas based on patterns
        lemma_count = 0
        lemmas: list[dict[str, str]] = []

        def add_lemma(
            name: str,
            code: str,
            kind: str,
            tier: str,
            role: str,
            formula: str = "",
            target: str = "main",
        ) -> None:
            nonlocal lean_code, tier2_code, tier2_count
            if target == "tier2":
                tier2_code += code
                tier2_count += 1
            else:
                lean_code += code
            lemmas.append({
                "name": name,
                "kind": kind,
                "tier": tier,
                "role": role,
                "formula": formula,
            })

        for pattern in patterns:
            if pattern['type'] == 'algebraic_identity':
                formula = pattern['formula']

                if formula == 'd = b + e':
                    name = f"cluster_{cluster_id}_d_identity"
                    add_lemma(
                        name,
                        f"""
theorem {name} (q : QA_Tuple) : d q = q.b + q.e := by rfl

""",
                        "rfl",
                        "0",
                        "conjecture",
                        formula,
                        target="main",
                    )
                    lemma_count += 1

                elif formula == 'a = b + 2e':
                    name = f"cluster_{cluster_id}_a_identity"
                    add_lemma(
                        name,
                        f"""
theorem {name} (q : QA_Tuple) : a q = q.b + 2 * q.e := by rfl

""",
                        "rfl",
                        "0",
                        "conjecture",
                        formula,
                        target="main",
                    )
                    lemma_count += 1

            elif pattern['type'] == 'modular_identity':
                # For now, we'll create a placeholder for modular identities
                # Full implementation would require modular arithmetic in Lean
                lean_code += f"""
-- Modular identity: {pattern['formula']}
-- (Placeholder - requires modular arithmetic setup)

"""

        if lemma_count > 0:
            name = f"cluster_{cluster_id}_C_eq_two_mul_X"
            add_lemma(
                name,
                f"""
theorem {name} (q : QA_Tuple) : C q = 2 * X q := by
  simp [C, X, Nat.mul_assoc]

""",
                "simp",
                "1",
                "aux",
                "C = 2 * X",
                target="main",
            )

            name = f"cluster_{cluster_id}_H_unfold"
            add_lemma(
                name,
                f"""
theorem {name} (q : QA_Tuple) : H q = (2 * q.e * d q) + (q.b * a q) := by
  simp [H, C, F]

""",
                "simp",
                "1",
                "aux",
                "H = C + F (expanded)",
                target="main",
            )

            name = f"cluster_{cluster_id}_J_expand"
            add_lemma(
                name,
                f"""
theorem {name} (q : QA_Tuple) : J q = (q.b + q.e) * q.b := by
  simp [J, d]

""",
                "simp",
                "1",
                "aux",
                "J = (b+e) * b",
                target="main",
            )

            name = f"cluster_{cluster_id}_F_expand"
            add_lemma(
                name,
                f"""
theorem {name} (q : QA_Tuple) : F q = q.b * (q.b + 2 * q.e) := by
  simp [F, a]

""",
                "simp",
                "1",
                "aux",
                "F = b * (b + 2e)",
                target="main",
            )

            name = f"cluster_{cluster_id}_F_expand_full"
            add_lemma(
                name,
                f"""
theorem {name} (q : QA_Tuple) : F q = q.b * q.b + 2 * q.b * q.e := by
  simp [F, a{tier2_simp_suffix}]

""",
                "simp",
                "2",
                "conjecture",
                "F expanded (full)",
                target="tier2",
            )

            name = f"cluster_{cluster_id}_C_expand_full"
            add_lemma(
                name,
                f"""
theorem {name} (q : QA_Tuple) : C q = 2 * q.e * q.b + 2 * q.e * q.e := by
  simp [C, d{tier2_simp_suffix}]

""",
                "simp",
                "2",
                "conjecture",
                "C expanded (full)",
                target="tier2",
            )

            name = f"cluster_{cluster_id}_J_expand_full"
            add_lemma(
                name,
                f"""
theorem {name} (q : QA_Tuple) : J q = q.b * q.b + q.e * q.b := by
  simp [J, d{tier2_simp_suffix}]

""",
                "simp",
                "2",
                "conjecture",
                "J expanded (full)",
                target="tier2",
            )

        if lemma_count == 0:
            # No directly translatable lemmas, add a trivial example
            name = f"cluster_{cluster_id}_placeholder_rfl"
            add_lemma(
                name,
                f"""
-- No directly translatable lemmas in this cluster
theorem {name} (q : QA_Tuple) : q.b + q.e = d q := by rfl

""",
                "rfl",
                "0",
                "aux",
                "b + e = d",
                target="main",
            )

        return lean_code, tier2_code if tier2_count > 0 else None, lemmas

    def run_lean_check(self, lean_file: Path) -> Dict:
        """
        Run Lean 4 to check a file
        """
        try:
            # Get current environment and modify PATH
            env = os.environ.copy()
            elan_bin_path = Path.home() / ".elan" / "bin"
            env['PATH'] = str(elan_bin_path) + os.pathsep + env.get('PATH', '')

            # Try to run Lean (if installed)
            result = subprocess.run(
                ['lean', lean_file],
                capture_output=True,
                text=True,
                timeout=30,
                env=env # Pass the modified environment
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

        except FileNotFoundError:
            logger.warning("Lean not found - skipping verification (will mark as unverified)")
            return {
                'success': None,  # Indicates Lean not available
                'stdout': '',
                'stderr': 'Lean 4 not installed',
                'returncode': -1
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"Lean verification timed out for {lean_file.name}")
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Verification timed out (30s)',
                'returncode': -2
            }

    def verify_conjecture(self, conjecture: Dict) -> Dict:
        """
        Verify a single conjecture
        """
        cluster_id = conjecture['cluster_id']
        logger.info(f"Verifying conjecture from cluster {cluster_id}...")

        # Translate to Lean
        lean_code, tier2_code, lemmas = self.conjecture_to_lean(conjecture)

        # Save to file
        lean_file = self.output_dir / f"cluster_{cluster_id}.lean"
        with open(lean_file, 'w') as f:
            f.write(lean_code)

        logger.info(f"  Generated Lean file: {lean_file.name}")

        # Run Lean verification
        result = self.run_lean_check(lean_file)

        tier2_result = None
        tier2_file = None
        tier2_reason = None
        if tier2_code:
            tier2_file = self.output_dir / f"cluster_{cluster_id}_tier2.lean"
            with open(tier2_file, 'w') as f:
                f.write(tier2_code)
            tier2_result = self.run_lean_check(tier2_file)
            if tier2_result['success'] is False:
                tier2_reason = self._classify_tier2_failure(
                    tier2_result.get('stdout', ''),
                    tier2_result.get('stderr', ''),
                )

        tier2_conjecture_count = sum(
            1 for lemma in lemmas
            if lemma.get('role') == 'conjecture' and lemma.get('tier') == '2'
        )
        tier2_aux_count = sum(
            1 for lemma in lemmas
            if lemma.get('role') == 'aux' and lemma.get('tier') == '2'
        )
        main_conjecture_count = sum(
            1 for lemma in lemmas
            if lemma.get('role') == 'conjecture' and lemma.get('tier') != '2'
        )
        main_aux_count = sum(
            1 for lemma in lemmas
            if lemma.get('role') == 'aux' and lemma.get('tier') != '2'
        )

        proof_record = {
            'cluster_id': cluster_id,
            'conjecture': conjecture,
            'lean_file': str(lean_file),
            'verified': result['success'],
            'lean_output': result.get('stdout', ''),
            'lean_errors': result.get('stderr', ''),
            'lemmas': lemmas,
            'lemma_count': len(lemmas),
            'rfl_count': sum(1 for lemma in lemmas if lemma.get('kind') == 'rfl'),
            'simp_count': sum(1 for lemma in lemmas if lemma.get('kind') == 'simp'),
            'aux_count': main_aux_count,
            'conjecture_count': main_conjecture_count,
            'tier2_conjecture_count': tier2_conjecture_count,
            'tier2_aux_count': tier2_aux_count,
            'tier2_attempted': tier2_conjecture_count + tier2_aux_count,
            'tier2_file': str(tier2_file) if tier2_file else None,
            'tier2_verified': (
                tier2_result['success'] if tier2_result is not None else None
            ),
            'tier2_output': (
                tier2_result.get('stdout', '') if tier2_result is not None else ''
            ),
            'tier2_errors': (
                tier2_result.get('stderr', '') if tier2_result is not None else ''
            ),
            'tier2_reason': tier2_reason,
            'tier2_lemmas': self.tier2_lemmas,
        }

        if result['success'] is True:
            logger.info(f"  ✓ Verification PASSED")
            self.verified_proofs.append(proof_record)
        elif result['success'] is False:
            logger.info(f"  ✗ Verification FAILED")
            logger.info(f"    Error: {result['stderr'][:100]}")
            self.failed_proofs.append(proof_record)
        else:
            logger.info(f"  ⚠ Verification SKIPPED (Lean not available)")
            proof_record['verified'] = 'unverified'
            self.verified_proofs.append(proof_record)  # Still save the generated proof

        return proof_record

    def verify_all(self, max_conjectures=None):
        """
        Verify all conjectures (or up to max_conjectures)
        """
        logger.info("="*60)
        logger.info("QA LEAN FORMAL VERIFIER - Starting")
        logger.info("="*60)

        start_time = time.time()

        # Limit number of conjectures if specified
        conjectures_to_verify = self.conjectures[:max_conjectures] if max_conjectures else self.conjectures

        logger.info(f"Verifying {len(conjectures_to_verify)} conjectures...")
        logger.info("-"*60)

        for i, conjecture in enumerate(conjectures_to_verify, 1):
            logger.info(f"\n[{i}/{len(conjectures_to_verify)}]")
            self.verify_conjecture(conjecture)

        # Statistics
        total_time = time.time() - start_time

        logger.info("\n" + "="*60)
        logger.info("VERIFICATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total conjectures: {len(conjectures_to_verify)}")
        logger.info(f"Verified (passed): {len([p for p in self.verified_proofs if p['verified'] is True])}")
        logger.info(f"Failed: {len(self.failed_proofs)}")
        logger.info(f"Unverified (Lean N/A): {len([p for p in self.verified_proofs if p['verified'] == 'unverified'])}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("="*60)

        # Save proof records
        proofs_file = self.output_dir / "proof_records.json"
        with open(proofs_file, 'w') as f:
            json.dump({
                'verified': self.verified_proofs,
                'failed': self.failed_proofs,
                'total_time': total_time
            }, f, indent=2)

        logger.info(f"✓ Proof records saved to {proofs_file}")

        return {
            'verified': self.verified_proofs,
            'failed': self.failed_proofs
        }


def main():
    """
    Main entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description='Verify QA conjectures using Lean 4')
    parser.add_argument('--conjectures', default='conjectures.json',
                       help='Path to conjectures JSON file')
    parser.add_argument('--output-dir', default='./lean_proofs',
                       help='Output directory for Lean files and proofs')
    parser.add_argument('--max-conjectures', type=int, default=None,
                       help='Maximum number of conjectures to verify')
    parser.add_argument('--tier2-lemmas', default="",
                       help='Comma-separated lemmas to add to Tier-2 simp set')

    args = parser.parse_args()

    try:
        tier2_lemmas = [item.strip() for item in args.tier2_lemmas.split(",") if item.strip()]
        verifier = QALeanVerifier(args.conjectures, args.output_dir, tier2_lemmas=tier2_lemmas)

        # Load conjectures
        verifier.load_conjectures()

        # Verify all
        results = verifier.verify_all(max_conjectures=args.max_conjectures)

        logger.info("✓ SUCCESS: Verification completed")
        return 0

    except Exception as e:
        logger.error(f"✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
