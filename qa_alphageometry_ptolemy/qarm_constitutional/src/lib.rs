// QARM v0.2 Constitutional Implementation
// TLA+ Verified Rust Mirror
//
// This library is a mechanical translation of QARM_v02_Failures.tla
// DO NOT modify without updating the TLA+ spec first and re-running TLC
//
// Constitutional Authority: QARM_v02_Failures.tla
// TLC Verification: 504 states (full), 383 states (no-μ), 0 violations
// GLFSI Theorem: Confirmed by model checking

pub mod qarm_state;
pub mod qarm_moves;
pub mod qarm_invariants;
pub mod qarm_glfsi;

// Re-export core types for convenience
pub use qarm_state::{CAP, KSET, State, FailType, Move};
pub use qarm_moves::{step_sigma, step_mu, step_lambda};
pub use qarm_invariants::{check_all_invariants, verify_invariants};
pub use qarm_glfsi::verify_glfsi;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_constitutional_parameters() {
        // Verify constitutional constants match TLC config
        assert_eq!(CAP, 20, "CAP must match TLC config");
        assert_eq!(KSET, &[2, 3], "KSET must match TLC config");
    }

    #[test]
    fn test_initial_state_count_matches_tlc() {
        let states = qarm_state::initial_states();
        assert_eq!(states.len(), 121, "Must match TLC initial state count");
    }

    #[test]
    fn test_all_initial_states_satisfy_invariants() {
        let states = qarm_state::initial_states();
        for s in &states {
            assert!(check_all_invariants(s),
                "Initial state {:?} violates invariants", s);
        }
    }
}
