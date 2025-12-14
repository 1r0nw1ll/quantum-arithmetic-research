//! Proof steps and traces
//!
//! This module defines structures for representing proof steps and complete proof traces,
//! including serialization support for proof verification and export.

use super::facts::{Fact, ProofStepId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during proof trace operations
#[derive(Error, Debug)]
pub enum ProofError {
    #[error("JSON serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Invalid proof step: {0}")]
    InvalidStep(String),

    #[error("Proof step not found: {0}")]
    StepNotFound(ProofStepId),
}

/// Result type for proof operations
pub type ProofResult<T> = Result<T, ProofError>;

/// A single step in a geometric proof
///
/// Each step applies a reasoning rule to premise facts to derive conclusion facts.
/// Steps are scored based on their usefulness or likelihood.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofStep {
    /// Unique identifier for this step
    pub id: ProofStepId,

    /// Name of the inference rule applied
    pub rule_id: String,

    /// Input facts required by this rule
    pub premises: Vec<Fact>,

    /// Output facts derived by this rule
    pub conclusions: Vec<Fact>,

    /// Score indicating quality/likelihood of this step (higher is better)
    pub score: f64,

    /// Optional human-readable explanation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}

impl ProofStep {
    /// Create a new proof step
    pub fn new(
        id: ProofStepId,
        rule_id: String,
        premises: Vec<Fact>,
        conclusions: Vec<Fact>,
        score: f64,
    ) -> Self {
        Self {
            id,
            rule_id,
            premises,
            conclusions,
            score,
            explanation: None,
        }
    }

    /// Create a proof step with an explanation
    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = Some(explanation);
        self
    }

    /// Check if this step is valid (has at least one conclusion)
    pub fn is_valid(&self) -> bool {
        !self.conclusions.is_empty()
    }

    /// Get the number of premises
    pub fn num_premises(&self) -> usize {
        self.premises.len()
    }

    /// Get the number of conclusions
    pub fn num_conclusions(&self) -> usize {
        self.conclusions.len()
    }
}

/// Complete trace of a proof attempt
///
/// Contains the sequence of proof steps, outcome, and metadata about the proof process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofTrace {
    /// Ordered sequence of proof steps
    pub steps: Vec<ProofStep>,

    /// Whether the proof successfully reached the goal
    pub solved: bool,

    /// Hash of the final proof state (for deduplication)
    pub final_state_hash: u64,

    /// Additional metadata about the proof
    pub metadata: HashMap<String, String>,
}

impl Default for ProofTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofTrace {
    /// Create a new empty proof trace
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            solved: false,
            final_state_hash: 0,
            metadata: HashMap::new(),
        }
    }

    /// Create a proof trace with initial metadata
    pub fn with_metadata(metadata: HashMap<String, String>) -> Self {
        Self {
            steps: Vec::new(),
            solved: false,
            final_state_hash: 0,
            metadata,
        }
    }

    /// Add a proof step to the trace
    pub fn add_step(&mut self, step: ProofStep) {
        self.steps.push(step);
    }

    /// Mark the proof as solved and set final state hash
    pub fn mark_solved(&mut self, final_hash: u64) {
        self.solved = true;
        self.final_state_hash = final_hash;
    }

    /// Mark the proof as unsolved
    pub fn mark_unsolved(&mut self, final_hash: u64) {
        self.solved = false;
        self.final_state_hash = final_hash;
    }

    /// Get the number of steps in the trace
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the trace is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Add metadata entry
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Get a proof step by ID
    pub fn get_step(&self, id: ProofStepId) -> Option<&ProofStep> {
        self.steps.iter().find(|s| s.id == id)
    }

    /// Get all conclusions derived in the proof
    pub fn all_conclusions(&self) -> Vec<&Fact> {
        self.steps
            .iter()
            .flat_map(|step| &step.conclusions)
            .collect()
    }

    /// Get the total score of the proof (sum of step scores)
    pub fn total_score(&self) -> f64 {
        self.steps.iter().map(|s| s.score).sum()
    }

    /// Get the average score per step
    pub fn average_score(&self) -> f64 {
        if self.steps.is_empty() {
            0.0
        } else {
            self.total_score() / self.steps.len() as f64
        }
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> ProofResult<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Serialize to compact JSON string
    pub fn to_json_compact(&self) -> ProofResult<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Deserialize from JSON string
    pub fn from_json(json: &str) -> ProofResult<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Get statistics about the proof
    pub fn statistics(&self) -> ProofStatistics {
        let mut stats = ProofStatistics {
            num_steps: self.steps.len(),
            num_conclusions: 0,
            total_score: 0.0,
            average_score: 0.0,
            solved: self.solved,
            rules_used: HashMap::new(),
        };

        for step in &self.steps {
            stats.num_conclusions += step.conclusions.len();
            stats.total_score += step.score;

            *stats.rules_used.entry(step.rule_id.clone()).or_insert(0) += 1;
        }

        stats.average_score = if stats.num_steps > 0 {
            stats.total_score / stats.num_steps as f64
        } else {
            0.0
        };

        stats
    }

    /// Validate that all step IDs are unique
    pub fn validate_step_ids(&self) -> ProofResult<()> {
        let mut seen = std::collections::HashSet::new();

        for step in &self.steps {
            if !seen.insert(step.id) {
                return Err(ProofError::InvalidStep(format!(
                    "Duplicate step ID: {:?}",
                    step.id
                )));
            }
        }

        Ok(())
    }

    /// Get steps that produced a specific fact
    pub fn steps_producing_fact(&self, fact: &Fact) -> Vec<&ProofStep> {
        let normalized = fact.clone().normalize();
        self.steps
            .iter()
            .filter(|step| {
                step.conclusions
                    .iter()
                    .any(|f| f.clone().normalize() == normalized)
            })
            .collect()
    }
}

/// Statistics about a proof trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStatistics {
    /// Total number of proof steps
    pub num_steps: usize,

    /// Total number of derived facts (conclusions)
    pub num_conclusions: usize,

    /// Total score across all steps
    pub total_score: f64,

    /// Average score per step
    pub average_score: f64,

    /// Whether the proof was successful
    pub solved: bool,

    /// Count of how many times each rule was used
    pub rules_used: HashMap<String, usize>,
}

impl ProofStatistics {
    /// Get the most frequently used rule
    pub fn most_used_rule(&self) -> Option<(&str, usize)> {
        self.rules_used
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(rule, count)| (rule.as_str(), *count))
    }
}

#[cfg(test)]
mod tests {
    use super::super::facts::{Fact, ProofStepId};
    use super::super::symbols::LineId;
    use super::*;

    #[test]
    fn test_proof_step_creation() {
        let step = ProofStep::new(
            ProofStepId(1),
            "parallel_transitive".to_string(),
            vec![
                Fact::Parallel(LineId(1), LineId(2)),
                Fact::Parallel(LineId(2), LineId(3)),
            ],
            vec![Fact::Parallel(LineId(1), LineId(3))],
            0.95,
        );

        assert_eq!(step.num_premises(), 2);
        assert_eq!(step.num_conclusions(), 1);
        assert!(step.is_valid());
    }

    #[test]
    fn test_proof_step_with_explanation() {
        let step = ProofStep::new(
            ProofStepId(1),
            "test_rule".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(1), LineId(2))],
            1.0,
        )
        .with_explanation("Lines are parallel by construction".to_string());

        assert!(step.explanation.is_some());
    }

    #[test]
    fn test_proof_trace_creation() {
        let mut trace = ProofTrace::new();
        assert!(trace.is_empty());
        assert!(!trace.solved);

        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "axiom".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(1), LineId(2))],
            1.0,
        ));

        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_proof_trace_metadata() {
        let mut trace = ProofTrace::new();
        trace.add_metadata("solver".to_string(), "qa-alphageometry".to_string());
        trace.add_metadata("timeout_ms".to_string(), "5000".to_string());

        assert_eq!(trace.get_metadata("solver"), Some("qa-alphageometry"));
        assert_eq!(trace.get_metadata("timeout_ms"), Some("5000"));
        assert_eq!(trace.get_metadata("nonexistent"), None);
    }

    #[test]
    fn test_proof_trace_solved() {
        let mut trace = ProofTrace::new();
        trace.mark_solved(12345);

        assert!(trace.solved);
        assert_eq!(trace.final_state_hash, 12345);
    }

    #[test]
    fn test_proof_trace_scores() {
        let mut trace = ProofTrace::new();

        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "rule1".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(1), LineId(2))],
            1.0,
        ));

        trace.add_step(ProofStep::new(
            ProofStepId(2),
            "rule2".to_string(),
            vec![],
            vec![Fact::Perpendicular(LineId(1), LineId(3))],
            0.5,
        ));

        assert_eq!(trace.total_score(), 1.5);
        assert_eq!(trace.average_score(), 0.75);
    }

    #[test]
    fn test_json_serialization_roundtrip() {
        let mut trace = ProofTrace::new();
        trace.add_metadata("test".to_string(), "value".to_string());

        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "test_rule".to_string(),
            vec![Fact::Parallel(LineId(1), LineId(2))],
            vec![Fact::Perpendicular(LineId(3), LineId(4))],
            0.8,
        ));

        trace.mark_solved(99999);

        let json = trace.to_json().unwrap();
        let restored = ProofTrace::from_json(&json).unwrap();

        assert_eq!(trace.len(), restored.len());
        assert_eq!(trace.solved, restored.solved);
        assert_eq!(trace.final_state_hash, restored.final_state_hash);
        assert_eq!(
            trace.get_metadata("test"),
            restored.get_metadata("test")
        );
    }

    #[test]
    fn test_proof_statistics() {
        let mut trace = ProofTrace::new();

        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "parallel_trans".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(1), LineId(2))],
            1.0,
        ));

        trace.add_step(ProofStep::new(
            ProofStepId(2),
            "parallel_trans".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(2), LineId(3))],
            0.9,
        ));

        trace.add_step(ProofStep::new(
            ProofStepId(3),
            "perp_rule".to_string(),
            vec![],
            vec![Fact::Perpendicular(LineId(1), LineId(3))],
            0.7,
        ));

        let stats = trace.statistics();

        assert_eq!(stats.num_steps, 3);
        assert_eq!(stats.num_conclusions, 3);
        assert!((stats.total_score - 2.6).abs() < 1e-10); // Use approximate comparison
        assert!((stats.average_score - 0.8667).abs() < 0.01);

        let (most_used, count) = stats.most_used_rule().unwrap();
        assert_eq!(most_used, "parallel_trans");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_get_step_by_id() {
        let mut trace = ProofTrace::new();

        let step = ProofStep::new(
            ProofStepId(42),
            "test_rule".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(1), LineId(2))],
            1.0,
        );

        trace.add_step(step.clone());

        assert!(trace.get_step(ProofStepId(42)).is_some());
        assert!(trace.get_step(ProofStepId(99)).is_none());
    }

    #[test]
    fn test_all_conclusions() {
        let mut trace = ProofTrace::new();

        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "rule1".to_string(),
            vec![],
            vec![
                Fact::Parallel(LineId(1), LineId(2)),
                Fact::Parallel(LineId(3), LineId(4)),
            ],
            1.0,
        ));

        trace.add_step(ProofStep::new(
            ProofStepId(2),
            "rule2".to_string(),
            vec![],
            vec![Fact::Perpendicular(LineId(1), LineId(3))],
            1.0,
        ));

        let conclusions = trace.all_conclusions();
        assert_eq!(conclusions.len(), 3);
    }

    #[test]
    fn test_validate_step_ids() {
        let mut trace = ProofTrace::new();

        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "rule1".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(1), LineId(2))],
            1.0,
        ));

        trace.add_step(ProofStep::new(
            ProofStepId(2),
            "rule2".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(3), LineId(4))],
            1.0,
        ));

        assert!(trace.validate_step_ids().is_ok());

        // Add duplicate ID
        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "rule3".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(5), LineId(6))],
            1.0,
        ));

        assert!(trace.validate_step_ids().is_err());
    }

    #[test]
    fn test_steps_producing_fact() {
        let mut trace = ProofTrace::new();

        let target_fact = Fact::Parallel(LineId(1), LineId(2));

        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "rule1".to_string(),
            vec![],
            vec![target_fact.clone()],
            1.0,
        ));

        trace.add_step(ProofStep::new(
            ProofStepId(2),
            "rule2".to_string(),
            vec![],
            vec![Fact::Perpendicular(LineId(3), LineId(4))],
            1.0,
        ));

        let steps = trace.steps_producing_fact(&target_fact);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].id, ProofStepId(1));
    }

    #[test]
    fn test_compact_json() {
        let mut trace = ProofTrace::new();
        trace.add_step(ProofStep::new(
            ProofStepId(1),
            "test".to_string(),
            vec![],
            vec![Fact::Parallel(LineId(1), LineId(2))],
            1.0,
        ));

        let compact = trace.to_json_compact().unwrap();
        let pretty = trace.to_json().unwrap();

        assert!(compact.len() < pretty.len());
        assert!(!compact.contains('\n'));
    }
}
