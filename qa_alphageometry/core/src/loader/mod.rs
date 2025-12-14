//! Problem loader for Geometry3K and similar datasets
//!
//! Parses geometric problems from JSON format and converts them to IR

pub mod geometry3k;

use crate::ir::{GeoState, Goal, Fact};
use serde::{Deserialize, Serialize};

/// A geometric problem with givens and goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryProblem {
    /// Problem ID
    pub id: String,

    /// Problem description (natural language)
    pub description: String,

    /// Given facts (premises)
    pub givens: Vec<Fact>,

    /// Goal fact(s) to prove
    pub goals: Vec<Fact>,

    /// Optional: expected answer (for validation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,

    /// Optional: difficulty level
    #[serde(skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<u8>,
}

impl GeometryProblem {
    /// Convert problem to initial GeoState
    pub fn to_state(&self) -> GeoState {
        let mut facts = crate::ir::FactStore::new();

        for fact in &self.givens {
            facts.insert(fact.clone());
        }

        let goal = Goal::new(self.goals.clone());

        GeoState::new(Default::default(), facts, goal)
    }
}

/// Problem loader result
pub type LoadResult<T> = Result<T, LoadError>;

/// Problem loading errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// File not found
    FileNotFound(String),

    /// JSON parsing error
    ParseError(String),

    /// Invalid problem format
    InvalidFormat(String),

    /// Unsupported fact type
    UnsupportedFact(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::FileNotFound(path) => write!(f, "File not found: {}", path),
            LoadError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LoadError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            LoadError::UnsupportedFact(msg) => write!(f, "Unsupported fact: {}", msg),
        }
    }
}

impl std::error::Error for LoadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{LineId};

    #[test]
    fn test_problem_to_state() {
        let problem = GeometryProblem {
            id: "test_001".to_string(),
            description: "Test parallel transitivity".to_string(),
            givens: vec![
                Fact::Parallel(LineId(1), LineId(2)),
                Fact::Parallel(LineId(2), LineId(3)),
            ],
            goals: vec![Fact::Parallel(LineId(1), LineId(3))],
            answer: None,
            difficulty: Some(1),
        };

        let state = problem.to_state();

        assert_eq!(state.facts.len(), 2);
        assert_eq!(state.goal.target_facts.len(), 1);
    }
}
