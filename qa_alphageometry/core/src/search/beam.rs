//! Beam search implementation
//!
//! Minimum viable solver: expand states, score with QA+geometric, keep best k

use crate::ir::{GeoState, ProofTrace, ProofStep, ProofStepId, Fact};
use crate::rules::all_rules;
use crate::search::scoring::{score_state, StateScore, ScoringConfig};
use std::collections::VecDeque;

/// Beam search configuration
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Beam width (number of states to keep)
    pub beam_width: usize,

    /// Maximum search depth
    pub max_depth: usize,

    /// Maximum total states explored
    pub max_states: usize,

    /// Scoring configuration
    pub scoring: ScoringConfig,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 10,
            max_depth: 50,
            max_states: 1000,
            scoring: ScoringConfig::default(),
        }
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Whether a proof was found
    pub solved: bool,

    /// Final proof trace (if solved)
    pub proof: Option<ProofTrace>,

    /// Number of states explored
    pub states_explored: usize,

    /// Final depth reached
    pub depth_reached: usize,

    /// Best score achieved
    pub best_score: f64,
}

/// Beam search solver
pub struct BeamSolver {
    config: BeamConfig,
}

impl BeamSolver {
    /// Create a new beam search solver
    pub fn new(config: BeamConfig) -> Self {
        Self { config }
    }

    /// Solve a geometry problem
    ///
    /// Returns a proof trace if a solution is found within budget
    pub fn solve(&self, initial_state: GeoState) -> SearchResult {
        // Check if already solved
        if initial_state.is_goal_satisfied() {
            return SearchResult {
                solved: true,
                proof: Some(ProofTrace::new()),
                states_explored: 0,
                depth_reached: 0,
                best_score: 10.0,
            };
        }

        // Initialize beam with initial state
        let mut beam: Vec<(GeoState, StateScore, ProofTrace)> = vec![];
        let initial_score = score_state(&initial_state, &self.config.scoring, 0);
        beam.push((initial_state, initial_score, ProofTrace::new()));

        let mut states_explored = 0;
        let mut best_score = initial_score.total;

        // Beam search loop
        for depth in 0..self.config.max_depth {
            if states_explored >= self.config.max_states {
                break;
            }

            let mut next_beam = Vec::new();

            // Expand each state in current beam
            for (state, _score, trace) in beam.iter() {
                // Try to expand state (would use rules here)
                // For now, we'll create a stub that just returns empty
                let successors = self.expand_state(state, trace, depth);

                for (successor_state, successor_trace) in successors {
                    states_explored += 1;

                    // Check if goal satisfied
                    if successor_state.is_goal_satisfied() {
                        return SearchResult {
                            solved: true,
                            proof: Some(successor_trace),
                            states_explored,
                            depth_reached: depth + 1,
                            best_score: 10.0,
                        };
                    }

                    // Score successor
                    let successor_score = score_state(
                        &successor_state,
                        &self.config.scoring,
                        depth + 1,
                    );

                    best_score = best_score.max(successor_score.total);

                    next_beam.push((successor_state, successor_score, successor_trace));

                    if states_explored >= self.config.max_states {
                        break;
                    }
                }

                if states_explored >= self.config.max_states {
                    break;
                }
            }

            if next_beam.is_empty() {
                // No more successors - search exhausted
                return SearchResult {
                    solved: false,
                    proof: None,
                    states_explored,
                    depth_reached: depth,
                    best_score,
                };
            }

            // Keep top-k by score
            next_beam.sort_by(|a, b| {
                b.1.total.partial_cmp(&a.1.total).unwrap_or(std::cmp::Ordering::Equal)
            });
            next_beam.truncate(self.config.beam_width);

            beam = next_beam;
        }

        // Max depth reached without solution
        SearchResult {
            solved: false,
            proof: None,
            states_explored,
            depth_reached: self.config.max_depth,
            best_score,
        }
    }

    /// Expand a state by applying deduction rules
    fn expand_state(
        &self,
        state: &GeoState,
        trace: &ProofTrace,
        _depth: usize,
    ) -> Vec<(GeoState, ProofTrace)> {
        let rules = all_rules();
        let mut successors = Vec::new();

        for rule in rules {
            let new_facts = rule.apply(state);

            for fact in new_facts {
                // Create new state with this fact added
                let mut new_state = state.clone();
                new_state.facts.insert(fact.clone());

                // Create new proof trace
                let mut new_trace = trace.clone();
                let step_id = ProofStepId(new_trace.steps.len() as u32);

                new_trace.add_step(ProofStep {
                    id: step_id,
                    rule_id: rule.id().to_string(),
                    premises: vec![], // Could track premises for better proof readability
                    conclusions: vec![fact],
                    score: rule.cost(),
                    explanation: None,
                });

                successors.push((new_state, new_trace));
            }
        }

        successors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Goal, FactStore, Fact, LineId};

    #[test]
    fn test_solver_creation() {
        let config = BeamConfig::default();
        let _solver = BeamSolver::new(config);
    }

    #[test]
    fn test_already_solved() {
        let target = Fact::Parallel(LineId(1), LineId(2));

        let mut facts = FactStore::new();
        facts.insert(target.clone());

        let goal = Goal::new(vec![target]);
        let state = GeoState::new(Default::default(), facts, goal);

        let solver = BeamSolver::new(BeamConfig::default());
        let result = solver.solve(state);

        assert!(result.solved);
        assert_eq!(result.depth_reached, 0);
        assert_eq!(result.states_explored, 0);
    }

    #[test]
    fn test_unsolvable_exhausts_beam() {
        let target = Fact::Parallel(LineId(1), LineId(2));
        let goal = Goal::new(vec![target]);

        // Empty facts - can't solve
        let state = GeoState::new(Default::default(), Default::default(), goal);

        let config = BeamConfig {
            beam_width: 5,
            max_depth: 10,
            max_states: 50,
            scoring: ScoringConfig::default(),
        };

        let solver = BeamSolver::new(config);
        let result = solver.solve(state);

        assert!(!result.solved);
        assert!(result.states_explored <= 50);
    }

    #[test]
    fn test_search_respects_max_depth() {
        let target = Fact::Parallel(LineId(1), LineId(2));
        let goal = Goal::new(vec![target]);

        let state = GeoState::new(Default::default(), Default::default(), goal);

        let config = BeamConfig {
            beam_width: 5,
            max_depth: 3,
            max_states: 1000,
            scoring: ScoringConfig::default(),
        };

        let solver = BeamSolver::new(config);
        let result = solver.solve(state);

        assert!(result.depth_reached <= 3);
    }

    #[test]
    fn test_search_result_fields() {
        let state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let solver = BeamSolver::new(BeamConfig::default());
        let result = solver.solve(state);

        // Should have valid fields
        assert!(result.states_explored >= 0);
        assert!(result.depth_reached >= 0);
        assert!(result.best_score >= 0.0);
    }

    #[test]
    fn test_parallel_transitivity_proof() {
        // TOY PROBLEM: Prove L1∥L3 given L1∥L2 and L2∥L3
        let mut facts = FactStore::new();
        facts.insert(Fact::Parallel(LineId(1), LineId(2)));
        facts.insert(Fact::Parallel(LineId(2), LineId(3)));

        let target = Fact::Parallel(LineId(1), LineId(3));
        let goal = Goal::new(vec![target.clone()]);

        let state = GeoState::new(Default::default(), facts, goal);

        let config = BeamConfig {
            beam_width: 10,
            max_depth: 5,
            max_states: 100,
            scoring: ScoringConfig::default(),
        };

        let solver = BeamSolver::new(config);
        let result = solver.solve(state);

        // Should solve using ParallelTransitivity rule
        assert!(result.solved, "Failed to solve parallel transitivity problem");
        assert!(result.proof.is_some(), "No proof trace generated");

        let proof = result.proof.unwrap();
        assert!(proof.steps.len() >= 1, "Proof should have at least 1 step");

        // Verify the proof uses parallel_transitivity rule
        assert!(
            proof.steps.iter().any(|s| s.rule_id == "parallel_transitivity"),
            "Proof should use parallel_transitivity rule"
        );

        // Verify the conclusion is in the proof
        assert!(
            proof.steps.iter().any(|s| s.conclusions.contains(&target)),
            "Proof should contain target fact L1∥L3"
        );

        println!("✅ Parallel transitivity proof found!");
        println!("   Steps: {}", proof.steps.len());
        println!("   States explored: {}", result.states_explored);
        println!("   Depth: {}", result.depth_reached);
    }

    #[test]
    fn test_qa_guidance_comparison() {
        // BENCHMARK: Compare QA on vs QA off
        // Problem: Prove L1⊥L3 from L1∥L2, L2∥L3, L4⊥L1
        let mut facts = FactStore::new();
        facts.insert(Fact::Parallel(LineId(1), LineId(2)));
        facts.insert(Fact::Parallel(LineId(2), LineId(3)));
        facts.insert(Fact::Perpendicular(LineId(4), LineId(1)));

        let target = Fact::Perpendicular(LineId(4), LineId(3));
        let goal = Goal::new(vec![target]);

        // TEST 1: QA OFF (qa_weight = 0.0)
        let state1 = GeoState::new(Default::default(), facts.clone(), goal.clone());
        let config_qa_off = BeamConfig {
            beam_width: 10,
            max_depth: 10,
            max_states: 200,
            scoring: ScoringConfig {
                geometric_weight: 1.0,
                qa_weight: 0.0,  // QA OFF
                step_penalty: 0.1,
            },
        };

        let solver_qa_off = BeamSolver::new(config_qa_off);
        let result_qa_off = solver_qa_off.solve(state1);

        // TEST 2: QA ON (qa_weight = 0.3)
        let state2 = GeoState::new(Default::default(), facts, goal);
        let config_qa_on = BeamConfig {
            beam_width: 10,
            max_depth: 10,
            max_states: 200,
            scoring: ScoringConfig {
                geometric_weight: 0.7,
                qa_weight: 0.3,  // QA ON
                step_penalty: 0.1,
            },
        };

        let solver_qa_on = BeamSolver::new(config_qa_on);
        let result_qa_on = solver_qa_on.solve(state2);

        println!("\n📊 QA Guidance Comparison:");
        println!("   QA OFF - Solved: {}, States: {}, Depth: {}",
                 result_qa_off.solved, result_qa_off.states_explored, result_qa_off.depth_reached);
        println!("   QA ON  - Solved: {}, States: {}, Depth: {}",
                 result_qa_on.solved, result_qa_on.states_explored, result_qa_on.depth_reached);

        // Both should solve (or both fail) for this simple problem
        assert_eq!(result_qa_off.solved, result_qa_on.solved,
                   "QA guidance should not prevent solving");

        if result_qa_off.solved && result_qa_on.solved {
            println!("   ✅ Both configurations solved the problem!");
            println!("   QA guidance changes search order but preserves correctness.");
        }
    }
}
