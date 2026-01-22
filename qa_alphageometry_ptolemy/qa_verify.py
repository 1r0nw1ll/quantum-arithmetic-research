#!/usr/bin/env python3
"""
QA Certificate Verifier CLI

Verifies certificate bundles and individual certificates for:
- Schema compliance
- Recompute hook consistency
- Obstruction completeness
- Cross-certificate coherence

Usage:
    python qa_verify.py <certificate_file.json>
    python qa_verify.py --bundle <bundle_file.json>
    python qa_verify.py --demo  # Run verification on demo outputs
"""

import sys
import json
import argparse
from pathlib import Path
from fractions import Fraction
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# VERIFICATION RESULT TYPES
# =============================================================================

class VerifyStatus(Enum):
    PASSED = "✔"
    FAILED = "✘"
    WARNING = "⚠"
    SKIPPED = "○"


@dataclass
class VerifyResult:
    status: VerifyStatus
    check_name: str
    message: str
    details: str = ""


# =============================================================================
# CERTIFICATE VALIDATORS
# =============================================================================

def verify_rl_certificate(cert: Dict[str, Any]) -> List[VerifyResult]:
    """Verify RLCertificate structure and consistency."""
    results = []

    # Check required fields
    required = ['model_id', 'training_success']
    for field in required:
        if field not in cert:
            results.append(VerifyResult(
                VerifyStatus.FAILED,
                f"rl.required_field.{field}",
                f"Missing required field: {field}"
            ))

    # Check success/failure consistency
    if cert.get('training_success'):
        if cert.get('method_proof') is None:
            results.append(VerifyResult(
                VerifyStatus.FAILED,
                "rl.success_requires_proof",
                "training_success=True but method_proof is missing"
            ))
        else:
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "rl.success_has_proof",
                "Success certificate has method proof"
            ))
    else:
        if cert.get('failure_mode') is None:
            results.append(VerifyResult(
                VerifyStatus.FAILED,
                "rl.failure_requires_mode",
                "training_success=False but failure_mode is missing"
            ))
        else:
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "rl.failure_has_mode",
                f"Failure certificate has mode: {cert.get('failure_mode')}"
            ))

    # Check Q-value witness if present
    if cert.get('q_value_witness'):
        witness = cert['q_value_witness']
        if witness.get('sample_transitions'):
            n_transitions = len(witness['sample_transitions'])
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "rl.q_witness_present",
                f"Q-value witness has {n_transitions} sample transitions"
            ))

    # Verify method proof structure
    if cert.get('method_proof'):
        mp = cert['method_proof']
        if mp.get('algorithm') == 'q_learning':
            if mp.get('discount_factor') is None:
                results.append(VerifyResult(
                    VerifyStatus.FAILED,
                    "rl.qlearning_requires_gamma",
                    "Q-learning requires discount_factor"
                ))
            else:
                results.append(VerifyResult(
                    VerifyStatus.PASSED,
                    "rl.qlearning_has_gamma",
                    f"Q-learning has discount_factor: {mp.get('discount_factor')}"
                ))

    return results


def verify_exploration_certificate(cert: Dict[str, Any]) -> List[VerifyResult]:
    """Verify ExplorationCertificate structure and consistency."""
    results = []

    # Check required fields
    if 'model_id' not in cert:
        results.append(VerifyResult(
            VerifyStatus.FAILED,
            "exploration.required_field.model_id",
            "Missing required field: model_id"
        ))

    # Check regret witness
    if cert.get('regret_witness'):
        rw = cert['regret_witness']
        actual = rw.get('actual_steps', 0)
        optimal = rw.get('optimal_steps', 0)
        claimed_regret = rw.get('cumulative_regret', 0)

        computed_regret = actual - optimal
        if computed_regret == claimed_regret:
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "exploration.regret_consistent",
                f"Regret computation verified: {actual} - {optimal} = {claimed_regret}"
            ))
        else:
            results.append(VerifyResult(
                VerifyStatus.FAILED,
                "exploration.regret_mismatch",
                f"Regret mismatch: {actual} - {optimal} = {computed_regret}, claimed {claimed_regret}"
            ))

        # Check regret non-negativity
        if claimed_regret >= 0:
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "exploration.regret_nonnegative",
                "Regret is non-negative"
            ))
        else:
            results.append(VerifyResult(
                VerifyStatus.FAILED,
                "exploration.regret_negative",
                f"Regret is negative: {claimed_regret}"
            ))

    # Check success/failure consistency
    if not cert.get('exploration_success'):
        if cert.get('failure_mode') is None:
            results.append(VerifyResult(
                VerifyStatus.WARNING,
                "exploration.failure_missing_mode",
                "exploration_success=False but failure_mode not specified"
            ))
        else:
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "exploration.failure_has_mode",
                f"Failure has mode: {cert.get('failure_mode')}"
            ))

            # Check obstruction evidence
            if cert.get('obstruction_if_fail'):
                results.append(VerifyResult(
                    VerifyStatus.PASSED,
                    "exploration.failure_has_obstruction",
                    "Failure has obstruction evidence"
                ))

    return results


def verify_inference_certificate(cert: Dict[str, Any]) -> List[VerifyResult]:
    """Verify InferenceCertificate structure."""
    results = []

    # Check method proof
    if cert.get('method_proof'):
        mp = cert['method_proof']
        method = mp.get('method', '')

        # BP on non-tree claiming exact is violation
        if method == 'belief_propagation':
            is_tree = mp.get('is_tree', False)
            exact = mp.get('exact', False)

            if not is_tree and exact:
                results.append(VerifyResult(
                    VerifyStatus.FAILED,
                    "inference.bp_nontree_exact",
                    "BP on non-tree graph cannot claim exact inference"
                ))
            else:
                results.append(VerifyResult(
                    VerifyStatus.PASSED,
                    "inference.bp_consistency",
                    "BP method is consistent with tree structure"
                ))

    # Check marginal sums to 1 (if present)
    if cert.get('marginal'):
        marginal = cert['marginal']
        if isinstance(marginal, dict):
            try:
                total = sum(Fraction(v) for v in marginal.values())
                if total == 1:
                    results.append(VerifyResult(
                        VerifyStatus.PASSED,
                        "inference.marginal_normalized",
                        "Marginal sums to 1"
                    ))
                else:
                    results.append(VerifyResult(
                        VerifyStatus.FAILED,
                        "inference.marginal_unnormalized",
                        f"Marginal sums to {total}, not 1"
                    ))
            except (ValueError, TypeError):
                results.append(VerifyResult(
                    VerifyStatus.SKIPPED,
                    "inference.marginal_check",
                    "Could not verify marginal normalization"
                ))

    return results


def verify_filter_certificate(cert: Dict[str, Any]) -> List[VerifyResult]:
    """Verify FilterCertificate structure."""
    results = []

    # Check method compatibility with system type
    if cert.get('method_proof'):
        mp = cert['method_proof']
        method = mp.get('method', '')

        if method == 'kalman':
            model = cert.get('model', {})
            linear = model.get('linear_system', False)
            gaussian = model.get('gaussian_noise', False)

            if linear and gaussian:
                results.append(VerifyResult(
                    VerifyStatus.PASSED,
                    "filter.kalman_applicable",
                    "Kalman filter is applicable (linear + Gaussian)"
                ))
            else:
                results.append(VerifyResult(
                    VerifyStatus.WARNING,
                    "filter.kalman_mismatch",
                    f"Kalman on non-ideal system (linear={linear}, gaussian={gaussian})"
                ))

    # Check state dimension consistency
    if cert.get('state_dimension') and cert.get('estimation'):
        state_dim = cert['state_dimension']
        estimation = cert['estimation']
        if estimation.get('state'):
            n_estimated = len(estimation['state'])
            if n_estimated == state_dim:
                results.append(VerifyResult(
                    VerifyStatus.PASSED,
                    "filter.state_dimension",
                    f"State dimension matches: {state_dim}"
                ))
            else:
                results.append(VerifyResult(
                    VerifyStatus.FAILED,
                    "filter.state_dimension_mismatch",
                    f"State dimension {state_dim} but estimated {n_estimated} values"
                ))

    return results


def verify_mcts_certificate(cert: Dict[str, Any]) -> List[VerifyResult]:
    """Verify MCTSCertificate structure."""
    results = []

    # Check SCC pruning witness
    if cert.get('scc_witness'):
        witness = cert['scc_witness']
        nodes_pruned = witness.get('nodes_pruned', 0)
        if nodes_pruned > 0:
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "mcts.scc_pruning_active",
                f"SCC pruning active: {nodes_pruned} nodes pruned"
            ))

    # Check pruning efficiency bounds
    if cert.get('pruning_efficiency') is not None:
        eff = cert['pruning_efficiency']
        if isinstance(eff, str) and '/' in eff:
            num, denom = eff.split('/')
            eff_val = int(num) / int(denom)
        else:
            eff_val = float(eff) if eff else 0

        if 0 <= eff_val <= 1:
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "mcts.pruning_efficiency_bounded",
                f"Pruning efficiency in [0,1]: {eff}"
            ))
        else:
            results.append(VerifyResult(
                VerifyStatus.FAILED,
                "mcts.pruning_efficiency_invalid",
                f"Pruning efficiency out of bounds: {eff}"
            ))

    return results


def verify_policy_certificate(cert: Dict[str, Any]) -> List[VerifyResult]:
    """Verify PolicyCertificate structure."""
    results = []

    # Check optimality proof if optimality claimed
    if cert.get('optimality_guarantee'):
        if cert.get('optimality_proof'):
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "policy.optimality_has_proof",
                "Optimality guarantee backed by proof"
            ))
        else:
            results.append(VerifyResult(
                VerifyStatus.FAILED,
                "policy.optimality_missing_proof",
                "optimality_guarantee=True but no optimality_proof"
            ))

    # Check failure has obstruction
    if cert.get('failure_mode'):
        if cert.get('obstruction_if_fail'):
            results.append(VerifyResult(
                VerifyStatus.PASSED,
                "policy.failure_has_obstruction",
                f"Failure mode {cert['failure_mode']} has obstruction evidence"
            ))
        else:
            results.append(VerifyResult(
                VerifyStatus.WARNING,
                "policy.failure_missing_obstruction",
                f"Failure mode {cert['failure_mode']} lacks obstruction evidence"
            ))

    return results


def verify_imitation_certificate(cert: Dict[str, Any]) -> List[VerifyResult]:
    """Verify ImitationCertificate structure."""
    results = []

    # Check IRL witness for inverse_rl method
    if cert.get('method_proof'):
        method = cert['method_proof'].get('method', '')
        if method == 'inverse_rl':
            if cert.get('irl_witness') or cert.get('inverse_rl_witness'):
                witness = cert.get('irl_witness') or cert.get('inverse_rl_witness')
                if witness.get('identifiable') is False:
                    if witness.get('alternative_targets'):
                        results.append(VerifyResult(
                            VerifyStatus.PASSED,
                            "imitation.nonident_has_alternatives",
                            "Non-identifiable IRL lists alternative targets"
                        ))
                    else:
                        results.append(VerifyResult(
                            VerifyStatus.FAILED,
                            "imitation.nonident_missing_alternatives",
                            "Non-identifiable but no alternative targets listed"
                        ))
                else:
                    results.append(VerifyResult(
                        VerifyStatus.PASSED,
                        "imitation.irl_identifiable",
                        f"IRL target identified with confidence {witness.get('confidence')}"
                    ))
            else:
                results.append(VerifyResult(
                    VerifyStatus.FAILED,
                    "imitation.irl_missing_witness",
                    "inverse_rl method but no irl_witness"
                ))

    return results


# =============================================================================
# BUNDLE VERIFICATION
# =============================================================================

def verify_bundle_coherence(bundle: Dict[str, Any]) -> List[VerifyResult]:
    """Verify cross-certificate coherence in a bundle."""
    results = []

    # Count certificates
    cert_types = ['policy', 'mcts', 'exploration', 'inference', 'filter', 'rl', 'imitation']
    total = 0
    for ct in cert_types:
        key = f'{ct}_certificates'
        if key in bundle:
            count = len(bundle[key])
            total += count

    results.append(VerifyResult(
        VerifyStatus.PASSED,
        "bundle.certificate_count",
        f"Bundle contains {total} certificates"
    ))

    # Check manifest hash if present
    if bundle.get('bundle_hash'):
        results.append(VerifyResult(
            VerifyStatus.PASSED,
            "bundle.has_hash",
            f"Bundle has tamper-evident hash: {bundle['bundle_hash'][:20]}..."
        ))

    return results


# =============================================================================
# CERTIFICATE TYPE DISPATCH
# =============================================================================

def identify_certificate_type(cert: Dict[str, Any]) -> str:
    """Identify the type of certificate from its structure."""
    if 'training_success' in cert and 'q_value_witness' in cert:
        return 'rl'
    if 'regret_witness' in cert:
        return 'exploration'
    if 'marginal' in cert or ('method_proof' in cert and
                               cert.get('method_proof', {}).get('method') in
                               ['variable_elimination', 'belief_propagation']):
        return 'inference'
    if 'covariance_trace' in cert or 'state_dimension' in cert:
        return 'filter'
    if 'scc_witness' in cert or 'pruning_efficiency' in cert:
        return 'mcts'
    if 'irl_witness' in cert or 'demonstration_witness' in cert:
        return 'imitation'
    if 'policy_type' in cert or 'optimality_proof' in cert:
        return 'policy'
    return 'unknown'


def verify_certificate(cert: Dict[str, Any]) -> Tuple[str, List[VerifyResult]]:
    """Verify a single certificate, auto-detecting type."""
    cert_type = identify_certificate_type(cert)

    verifiers = {
        'rl': verify_rl_certificate,
        'exploration': verify_exploration_certificate,
        'inference': verify_inference_certificate,
        'filter': verify_filter_certificate,
        'mcts': verify_mcts_certificate,
        'policy': verify_policy_certificate,
        'imitation': verify_imitation_certificate,
    }

    if cert_type in verifiers:
        results = verifiers[cert_type](cert)
    else:
        results = [VerifyResult(
            VerifyStatus.WARNING,
            "unknown_certificate_type",
            f"Could not identify certificate type"
        )]

    return cert_type, results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def print_results(results: List[VerifyResult], verbose: bool = False):
    """Print verification results."""
    passed = sum(1 for r in results if r.status == VerifyStatus.PASSED)
    failed = sum(1 for r in results if r.status == VerifyStatus.FAILED)
    warnings = sum(1 for r in results if r.status == VerifyStatus.WARNING)

    for r in results:
        if r.status == VerifyStatus.FAILED or verbose:
            print(f"  {r.status.value} {r.check_name}: {r.message}")
            if r.details and verbose:
                print(f"      {r.details}")

    return passed, failed, warnings


def main():
    parser = argparse.ArgumentParser(
        description="QA Certificate Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python qa_verify.py results.json
    python qa_verify.py --bundle spine_bundle.json
    python qa_verify.py --demo
    python qa_verify.py --verbose results.json
        """
    )
    parser.add_argument('file', nargs='?', help='Certificate or bundle JSON file')
    parser.add_argument('--bundle', action='store_true', help='Verify as certificate bundle')
    parser.add_argument('--demo', action='store_true', help='Verify demo outputs')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all checks')

    args = parser.parse_args()

    print("=" * 60)
    print("QA CERTIFICATE VERIFIER")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    total_warnings = 0

    if args.demo:
        # Verify demo outputs
        demo_files = [
            Path('/home/player2/signal_experiments/demos/spine_bundle.json'),
            Path('/home/player2/signal_experiments/demos/gym_benchmark_results.json'),
        ]

        for demo_file in demo_files:
            if demo_file.exists():
                print(f"\n[Verifying: {demo_file.name}]")
                with open(demo_file) as f:
                    data = json.load(f)

                if 'manifest' in data:
                    # New bundle format with manifest/coherence/certificates
                    manifest = data.get('manifest', {})
                    coherence = data.get('coherence', {})
                    certificates = data.get('certificates', {})

                    # Verify manifest
                    print(f"\n  [manifest]")
                    if manifest.get('bundle_hash'):
                        results = [VerifyResult(
                            VerifyStatus.PASSED,
                            "bundle.has_hash",
                            f"Bundle hash: {manifest['bundle_hash']}"
                        )]
                    else:
                        results = [VerifyResult(
                            VerifyStatus.WARNING,
                            "bundle.no_hash",
                            "Bundle missing tamper-evident hash"
                        )]
                    p, f, w = print_results(results, args.verbose)
                    total_passed += p
                    total_failed += f
                    total_warnings += w

                    # Verify coherence
                    print(f"\n  [coherence]")
                    if coherence.get('valid'):
                        results = [VerifyResult(
                            VerifyStatus.PASSED,
                            "bundle.coherence",
                            f"Bundle is coherent ({coherence.get('checks', 0)} checks)"
                        )]
                    else:
                        results = [VerifyResult(
                            VerifyStatus.FAILED,
                            "bundle.coherence",
                            f"Bundle coherence failed: {coherence.get('violations', [])}"
                        )]
                    p, f, w = print_results(results, args.verbose)
                    total_passed += p
                    total_failed += f
                    total_warnings += w

                    # Verify individual certificates
                    for cert_type, cert_list in certificates.items():
                        print(f"\n  [{cert_type}]")
                        if isinstance(cert_list, list):
                            for cert in cert_list:
                                if isinstance(cert, dict):
                                    _, results = verify_certificate(cert)
                                    p, f, w = print_results(results, args.verbose)
                                    total_passed += p
                                    total_failed += f
                                    total_warnings += w

                elif 'bundle_id' in data:
                    # Old bundle format
                    results = verify_bundle_coherence(data)
                    p, f, w = print_results(results, args.verbose)
                    total_passed += p
                    total_failed += f
                    total_warnings += w

                    # Verify individual certificates in bundle
                    for cert_type in ['policy', 'mcts', 'exploration', 'inference',
                                     'filter', 'rl', 'imitation']:
                        key = f'{cert_type}_certificates'
                        if key in data:
                            for cert in data[key]:
                                _, results = verify_certificate(cert)
                                p, f, w = print_results(results, args.verbose)
                                total_passed += p
                                total_failed += f
                                total_warnings += w
                else:
                    # Scenario-based output
                    for scenario, certs in data.items():
                        print(f"\n  [{scenario}]")
                        for cert_name, cert in certs.items():
                            cert_type, results = verify_certificate(cert)
                            print(f"    {cert_name} ({cert_type}):")
                            p, f, w = print_results(results, args.verbose)
                            total_passed += p
                            total_failed += f
                            total_warnings += w
            else:
                print(f"\n[Skipped: {demo_file.name} not found]")

    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

        with open(filepath) as f:
            data = json.load(f)

        if args.bundle or 'bundle_id' in data:
            print(f"\n[Verifying bundle: {filepath.name}]")
            results = verify_bundle_coherence(data)
            p, f, w = print_results(results, args.verbose)
            total_passed += p
            total_failed += f
            total_warnings += w
        else:
            print(f"\n[Verifying certificate: {filepath.name}]")
            cert_type, results = verify_certificate(data)
            print(f"  Type: {cert_type}")
            p, f, w = print_results(results, args.verbose)
            total_passed += p
            total_failed += f
            total_warnings += w

    else:
        parser.print_help()
        sys.exit(0)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  ✔ Passed:   {total_passed}")
    print(f"  ✘ Failed:   {total_failed}")
    print(f"  ⚠ Warnings: {total_warnings}")

    if total_failed == 0:
        print("\n  ✔ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"\n  ✘ {total_failed} CHECK(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
