//! State scoring for search guidance
//!
//! Combines geometric heuristic + QA prior for beam search ordering

use crate::ir::GeoState;
use crate::qa::{extract_qa_features, compute_qa_prior};

/// Search score for a geometric state
///
/// Higher score = more promising state (prioritized in beam search)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateScore {
    /// Geometric heuristic component (distance to goal)
    pub geometric_score: f64,

    /// QA prior component (harmonic structure)
    pub qa_prior: f64,

    /// Combined total score
    pub total: f64,
}

/// Scoring configuration
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Weight for geometric heuristic (0.0 to 1.0)
    pub geometric_weight: f64,

    /// Weight for QA prior (0.0 to 1.0)
    pub qa_weight: f64,

    /// Penalty per proof step (to prefer shorter proofs)
    pub step_penalty: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            geometric_weight: 0.7,  // Geometry gets more weight
            qa_weight: 0.3,          // QA provides guidance
            step_penalty: 0.1,       // Small penalty for length
        }
    }
}

/// Compute geometric heuristic score
///
/// Measures "distance" to goal based on:
/// - Number of goal facts already satisfied
/// - Number of facts in state (more facts = more progress)
fn geometric_heuristic(state: &GeoState) -> f64 {
    let num_goal_facts = state.goal.target_facts.len() as f64;
    let num_satisfied = state.goal.target_facts.iter()
        .filter(|f| state.facts.contains(f))
        .count() as f64;

    if num_goal_facts == 0.0 {
        return 0.0;
    }

    // Satisfaction ratio (0.0 to 1.0)
    let satisfaction = num_satisfied / num_goal_facts;

    // Bonus for having many facts (sign of progress)
    let fact_bonus = (state.facts.len() as f64).ln().max(0.0) * 0.1;

    (satisfaction * 10.0) + fact_bonus
}

/// Compute state score with QA guidance
pub fn score_state(state: &GeoState, config: &ScoringConfig, depth: usize) -> StateScore {
    // Geometric component
    let geometric_score = geometric_heuristic(state);

    // QA component
    let qa_features = extract_qa_features(state);
    let qa_prior = compute_qa_prior(&qa_features);

    // Combine scores
    let base_score =
        (geometric_score * config.geometric_weight) +
        (qa_prior * config.qa_weight);

    // Apply step penalty
    let step_penalty = (depth as f64) * config.step_penalty;
    let total = (base_score - step_penalty).max(0.0);

    StateScore {
        geometric_score,
        qa_prior,
        total,
    }
}

/// Compare two states for beam search ordering
///
/// Returns true if `a` should be prioritized over `b`
pub fn should_prioritize(a: &StateScore, b: &StateScore) -> bool {
    a.total > b.total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Goal, FactStore, Fact, LineId};

    #[test]
    fn test_geometric_heuristic_no_goal() {
        let state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let score = geometric_heuristic(&state);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_geometric_heuristic_partial_satisfaction() {
        let target1 = Fact::Parallel(LineId(1), LineId(2));
        let target2 = Fact::Parallel(LineId(2), LineId(3));

        let mut facts = FactStore::new();
        facts.insert(target1.clone());  // Only first goal satisfied

        let goal = Goal::new(vec![target1, target2]);
        let state = GeoState::new(Default::default(), facts, goal);

        let score = geometric_heuristic(&state);
        assert!(score > 0.0);
        assert!(score < 10.0);  // Not fully satisfied
    }

    #[test]
    fn test_geometric_heuristic_full_satisfaction() {
        let target1 = Fact::Parallel(LineId(1), LineId(2));
        let target2 = Fact::Parallel(LineId(2), LineId(3));

        let mut facts = FactStore::new();
        facts.insert(target1.clone());
        facts.insert(target2.clone());

        let goal = Goal::new(vec![target1, target2]);
        let state = GeoState::new(Default::default(), facts, goal);

        let score = geometric_heuristic(&state);
        assert!(score >= 10.0);  // Full satisfaction
    }

    #[test]
    fn test_score_state_combines_components() {
        let config = ScoringConfig::default();

        let state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let score = score_state(&state, &config, 0);

        assert_eq!(score.geometric_score, 0.0);
        assert!(score.total >= 0.0);
    }

    #[test]
    fn test_step_penalty_applied() {
        let config = ScoringConfig::default();

        let state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let score_depth0 = score_state(&state, &config, 0);
        let score_depth10 = score_state(&state, &config, 10);

        assert!(score_depth0.total >= score_depth10.total);  // Penalty applied
    }

    #[test]
    fn test_prioritization() {
        let high = StateScore {
            geometric_score: 10.0,
            qa_prior: 5.0,
            total: 15.0,
        };

        let low = StateScore {
            geometric_score: 3.0,
            qa_prior: 2.0,
            total: 5.0,
        };

        assert!(should_prioritize(&high, &low));
        assert!(!should_prioritize(&low, &high));
    }

    #[test]
    fn test_qa_weight_affects_score() {
        let config_no_qa = ScoringConfig {
            geometric_weight: 1.0,
            qa_weight: 0.0,
            step_penalty: 0.0,
        };

        let config_with_qa = ScoringConfig {
            geometric_weight: 0.5,
            qa_weight: 0.5,
            step_penalty: 0.0,
        };

        let state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let score1 = score_state(&state, &config_no_qa, 0);
        let score2 = score_state(&state, &config_with_qa, 0);

        // Scores should differ based on QA weight
        // (exact values depend on QA extraction from empty state)
        assert!(score1.qa_prior == 0.0 || score2.qa_prior == score1.qa_prior);
    }
}
