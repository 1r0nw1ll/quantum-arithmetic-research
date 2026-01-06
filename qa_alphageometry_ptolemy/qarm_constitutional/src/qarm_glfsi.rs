// QARM v0.2 GLFSI Theorem Verification
// Generator-Local Failure Signature Invariance
// Mirrors TLC state exploration and verifies theorem computationally

use crate::qarm_invariants::*;
use crate::qarm_moves::*;
use crate::qarm_state::*;
use std::collections::{HashMap, HashSet};

/// Failure signature for a generator
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureSignature {
    pub out_of_bounds: usize,
    pub fixed_q_violation: usize,
}

/// Explore reachable states under a given generator set
/// Returns (reachable_states, failure_crosstab)
pub fn explore_reachable(
    use_sigma: bool,
    use_mu: bool,
    use_lambda: bool,
) -> (HashSet<State>, HashMap<Move, HashMap<FailType, usize>>) {
    let mut visited: HashSet<State> = HashSet::new();
    let mut queue: Vec<State> = Vec::new();
    let mut crosstab: HashMap<Move, HashMap<FailType, usize>> = HashMap::new();

    // Initialize with all initial states
    let init_states = initial_states();
    for s in &init_states {
        visited.insert(s.clone());
        queue.push(s.clone());
    }

    // BFS exploration
    while let Some(current) = queue.pop() {
        // Skip if already failed (absorbing)
        if current.fail != FailType::Ok {
            continue;
        }

        // Try sigma
        if use_sigma {
            let next = step_sigma(&current);
            // Add to visited set regardless of success/failure (failures are distinct states)
            if !visited.contains(&next) {
                visited.insert(next.clone());
                record_failure(&mut crosstab, &next);  // Only record new states
                // Only queue OK states for further exploration
                if next.fail == FailType::Ok {
                    queue.push(next);
                }
            }
        }

        // Try mu
        if use_mu {
            let next = step_mu(&current);
            if !visited.contains(&next) {
                visited.insert(next.clone());
                record_failure(&mut crosstab, &next);  // Only record new states
                if next.fail == FailType::Ok {
                    queue.push(next);
                }
            }
        }

        // Try lambda for each k in KSET
        if use_lambda {
            for &k in KSET {
                let next = step_lambda(&current, k);
                if !visited.contains(&next) {
                    visited.insert(next.clone());
                    record_failure(&mut crosstab, &next);  // Only record new states
                    if next.fail == FailType::Ok {
                        queue.push(next);
                    }
                }
            }
        }
    }

    (visited, crosstab)
}

/// Record failure in crosstab (only for failed states)
fn record_failure(crosstab: &mut HashMap<Move, HashMap<FailType, usize>>, state: &State) {
    // Only record if this is a failure state
    if state.fail != FailType::Ok {
        let move_entry = crosstab.entry(state.last_move).or_insert_with(HashMap::new);
        *move_entry.entry(state.fail).or_insert(0) += 1;
    }
}

/// Extract failure signature for a specific generator
pub fn get_signature(crosstab: &HashMap<Move, HashMap<FailType, usize>>, gen: Move) -> FailureSignature {
    let move_failures = crosstab.get(&gen).cloned().unwrap_or_default();
    FailureSignature {
        out_of_bounds: *move_failures.get(&FailType::OutOfBounds).unwrap_or(&0),
        fixed_q_violation: *move_failures.get(&FailType::FixedQViolation).unwrap_or(&0),
    }
}

/// Verify GLFSI theorem
pub fn verify_glfsi() -> Result<(), String> {
    println!("======================================================================");
    println!("GLFSI Theorem Verification (Rust)");
    println!("======================================================================");
    println!();

    // Explore with full generator set
    println!("Run 1: Exploring with full generator set {{σ, μ, λ}}...");
    let (states_full, crosstab_full) = explore_reachable(true, true, true);
    println!("  Distinct states: {}", states_full.len());

    // Verify invariants hold
    let states_vec: Vec<State> = states_full.iter().cloned().collect();
    verify_invariants(&states_vec).map_err(|e| format!("Full set invariant violation: {}", e))?;
    println!("  ✅ All invariants hold");
    println!();

    // Explore without mu
    println!("Run 2: Exploring with reduced generator set {{σ, λ}} (no μ)...");
    let (states_nomu, crosstab_nomu) = explore_reachable(true, false, true);
    println!("  Distinct states: {}", states_nomu.len());

    // Verify invariants hold
    let states_vec_nomu: Vec<State> = states_nomu.iter().cloned().collect();
    verify_invariants(&states_vec_nomu).map_err(|e| format!("No-mu set invariant violation: {}", e))?;
    println!("  ✅ All invariants hold");
    println!();

    // Extract signatures
    let sig_sigma_full = get_signature(&crosstab_full, Move::Sigma);
    let sig_mu_full = get_signature(&crosstab_full, Move::Mu);
    let sig_lambda_full = get_signature(&crosstab_full, Move::Lambda);

    let sig_sigma_nomu = get_signature(&crosstab_nomu, Move::Sigma);
    let sig_lambda_nomu = get_signature(&crosstab_nomu, Move::Lambda);

    println!("======================================================================");
    println!("Per-Generator Failure Signatures");
    println!("======================================================================");
    println!();

    // Check sigma invariance
    println!("Generator σ:");
    println!("  OUT_OF_BOUNDS:     {:3} (full) vs {:3} (no-μ)  {}",
        sig_sigma_full.out_of_bounds,
        sig_sigma_nomu.out_of_bounds,
        if sig_sigma_full.out_of_bounds == sig_sigma_nomu.out_of_bounds { "✅ INVARIANT" } else { "❌ CHANGED" }
    );
    println!("  FIXED_Q_VIOLATION: {:3} (full) vs {:3} (no-μ)  {}",
        sig_sigma_full.fixed_q_violation,
        sig_sigma_nomu.fixed_q_violation,
        if sig_sigma_full.fixed_q_violation == sig_sigma_nomu.fixed_q_violation { "✅ INVARIANT" } else { "❌ CHANGED" }
    );
    println!();

    // Check lambda invariance
    println!("Generator λ:");
    println!("  OUT_OF_BOUNDS:     {:3} (full) vs {:3} (no-μ)  {}",
        sig_lambda_full.out_of_bounds,
        sig_lambda_nomu.out_of_bounds,
        if sig_lambda_full.out_of_bounds == sig_lambda_nomu.out_of_bounds { "✅ INVARIANT" } else { "❌ CHANGED" }
    );
    println!("  FIXED_Q_VIOLATION: {:3} (full) vs {:3} (no-μ)  {}",
        sig_lambda_full.fixed_q_violation,
        sig_lambda_nomu.fixed_q_violation,
        if sig_lambda_full.fixed_q_violation == sig_lambda_nomu.fixed_q_violation { "✅ INVARIANT" } else { "❌ CHANGED" }
    );
    println!();

    // Check mu (should be absent in no-mu run)
    println!("Generator μ:");
    println!("  OUT_OF_BOUNDS:     {:3} (full) vs --- (absent)",
        sig_mu_full.out_of_bounds
    );
    println!("  FIXED_Q_VIOLATION: {:3} (full) vs --- (absent)",
        sig_mu_full.fixed_q_violation
    );
    println!("  Status: Generator removed from set (expected)");
    println!();

    // Verify against TLC results
    println!("======================================================================");
    println!("TLC Verification");
    println!("======================================================================");
    println!();

    // Expected values from TLC
    const EXPECTED_STATES_FULL: usize = 504;
    const EXPECTED_STATES_NOMU: usize = 383;
    const EXPECTED_SIGMA_OOB: usize = 21;
    const EXPECTED_SIGMA_FQ: usize = 100;
    const EXPECTED_MU_OOB: usize = 40;
    const EXPECTED_MU_FQ: usize = 74;
    const EXPECTED_LAMBDA_OOB: usize = 105;
    const EXPECTED_LAMBDA_FQ: usize = 35;

    let mut errors = Vec::new();

    if states_full.len() != EXPECTED_STATES_FULL {
        errors.push(format!("State count mismatch (full): expected {}, got {}",
            EXPECTED_STATES_FULL, states_full.len()));
    }

    if states_nomu.len() != EXPECTED_STATES_NOMU {
        errors.push(format!("State count mismatch (no-μ): expected {}, got {}",
            EXPECTED_STATES_NOMU, states_nomu.len()));
    }

    if sig_sigma_full.out_of_bounds != EXPECTED_SIGMA_OOB {
        errors.push(format!("σ OUT_OF_BOUNDS mismatch: expected {}, got {}",
            EXPECTED_SIGMA_OOB, sig_sigma_full.out_of_bounds));
    }

    if sig_sigma_full.fixed_q_violation != EXPECTED_SIGMA_FQ {
        errors.push(format!("σ FIXED_Q_VIOLATION mismatch: expected {}, got {}",
            EXPECTED_SIGMA_FQ, sig_sigma_full.fixed_q_violation));
    }

    if sig_mu_full.out_of_bounds != EXPECTED_MU_OOB {
        errors.push(format!("μ OUT_OF_BOUNDS mismatch: expected {}, got {}",
            EXPECTED_MU_OOB, sig_mu_full.out_of_bounds));
    }

    if sig_mu_full.fixed_q_violation != EXPECTED_MU_FQ {
        errors.push(format!("μ FIXED_Q_VIOLATION mismatch: expected {}, got {}",
            EXPECTED_MU_FQ, sig_mu_full.fixed_q_violation));
    }

    if sig_lambda_full.out_of_bounds != EXPECTED_LAMBDA_OOB {
        errors.push(format!("λ OUT_OF_BOUNDS mismatch: expected {}, got {}",
            EXPECTED_LAMBDA_OOB, sig_lambda_full.out_of_bounds));
    }

    if sig_lambda_full.fixed_q_violation != EXPECTED_LAMBDA_FQ {
        errors.push(format!("λ FIXED_Q_VIOLATION mismatch: expected {}, got {}",
            EXPECTED_LAMBDA_FQ, sig_lambda_full.fixed_q_violation));
    }

    if !errors.is_empty() {
        println!("❌ TLC verification FAILED:");
        for err in &errors {
            println!("  - {}", err);
        }
        println!();
        return Err(format!("TLC verification failed: {} errors", errors.len()));
    }

    println!("✅ All values match TLC exactly:");
    println!("  Full set: {} states", EXPECTED_STATES_FULL);
    println!("  No-μ set: {} states", EXPECTED_STATES_NOMU);
    println!("  σ: OOB={}, FQ={}", EXPECTED_SIGMA_OOB, EXPECTED_SIGMA_FQ);
    println!("  μ: OOB={}, FQ={}", EXPECTED_MU_OOB, EXPECTED_MU_FQ);
    println!("  λ: OOB={}, FQ={}", EXPECTED_LAMBDA_OOB, EXPECTED_LAMBDA_FQ);
    println!();

    // Final GLFSI check
    let glfsi_holds =
        sig_sigma_full == sig_sigma_nomu &&
        sig_lambda_full == sig_lambda_nomu;

    println!("======================================================================");
    if glfsi_holds {
        println!("✅ GLFSI THEOREM CONFIRMED");
        println!();
        println!("Per-generator failure signatures are INVARIANT under generator set changes.");
        println!("Failure modes are intrinsic to (state, generator) pairs.");
    } else {
        println!("❌ GLFSI THEOREM REJECTED");
        println!();
        println!("Per-generator failure signatures changed unexpectedly.");
    }
    println!("======================================================================");

    if !glfsi_holds {
        return Err("GLFSI theorem violated".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glfsi_theorem() {
        // This is the critical test that verifies the theorem
        verify_glfsi().expect("GLFSI theorem must hold");
    }

    #[test]
    fn test_initial_state_count() {
        let init = initial_states();
        assert_eq!(init.len(), 121, "Expected 121 initial states (from TLC)");
    }

    #[test]
    fn test_exploration_produces_states() {
        let (states, _) = explore_reachable(true, true, true);
        assert!(states.len() > 0, "Should produce reachable states");
        assert!(states.len() >= 121, "Should include at least initial states");
    }

    #[test]
    fn test_signature_extraction() {
        let mut crosstab: HashMap<Move, HashMap<FailType, usize>> = HashMap::new();

        // Simulate some failures
        let mut sigma_fails = HashMap::new();
        sigma_fails.insert(FailType::OutOfBounds, 10);
        sigma_fails.insert(FailType::FixedQViolation, 5);
        crosstab.insert(Move::Sigma, sigma_fails);

        let sig = get_signature(&crosstab, Move::Sigma);
        assert_eq!(sig.out_of_bounds, 10);
        assert_eq!(sig.fixed_q_violation, 5);
    }
}
