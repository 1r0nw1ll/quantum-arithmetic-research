//! Proof state representation
//!
//! This module defines the complete state of a geometric proof attempt,
//! including known facts, goal facts to prove, and metadata.

use super::facts::{Fact, FactStore};
use super::symbols::SymbolTable;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Goal specification for a proof
///
/// The goal consists of one or more target facts that must be proven.
/// A proof succeeds when all target facts are derivable from the initial facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Goal {
    /// Facts that must be proven to satisfy the goal
    pub target_facts: Vec<Fact>,
}

impl Goal {
    /// Create a new goal from a list of target facts
    pub fn new(target_facts: Vec<Fact>) -> Self {
        Self { target_facts }
    }

    /// Create a goal with a single target fact
    pub fn single(fact: Fact) -> Self {
        Self {
            target_facts: vec![fact],
        }
    }

    /// Check if this goal is empty (no target facts)
    pub fn is_empty(&self) -> bool {
        self.target_facts.is_empty()
    }

    /// Get the number of target facts
    pub fn len(&self) -> usize {
        self.target_facts.len()
    }

    /// Add a target fact to the goal
    pub fn add_target(&mut self, fact: Fact) {
        self.target_facts.push(fact);
    }

    /// Check if a specific fact is part of the goal
    pub fn contains(&self, fact: &Fact) -> bool {
        let normalized = fact.clone().normalize();
        self.target_facts
            .iter()
            .any(|f| f.clone().normalize() == normalized)
    }
}

/// Metadata about a proof problem
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metadata {
    /// Unique identifier for the problem
    pub problem_id: String,

    /// Optional diagram description or SVG representation
    pub diagram_info: Option<String>,

    /// Additional key-value metadata
    pub extra: HashMap<String, String>,
}

impl Metadata {
    /// Create new metadata with a problem ID
    pub fn new(problem_id: String) -> Self {
        Self {
            problem_id,
            diagram_info: None,
            extra: HashMap::new(),
        }
    }

    /// Set diagram information
    pub fn with_diagram(mut self, diagram: String) -> Self {
        self.diagram_info = Some(diagram);
        self
    }

    /// Add extra metadata
    pub fn add_extra(&mut self, key: String, value: String) {
        self.extra.insert(key, value);
    }

    /// Get extra metadata value
    pub fn get_extra(&self, key: &str) -> Option<&str> {
        self.extra.get(key).map(|s| s.as_str())
    }
}

/// Complete geometric proof state
///
/// Represents all information needed to work on a geometric proof:
/// - Symbol table for object naming
/// - Known facts (premises and derived facts)
/// - Goal to prove
/// - Problem metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoState {
    /// Symbol table for geometric object identifiers
    pub symbols: SymbolTable,

    /// Collection of known facts (premises + derived)
    pub facts: FactStore,

    /// Goal to prove
    pub goal: Goal,

    /// Problem metadata
    pub metadata: Metadata,
}

impl GeoState {
    /// Create a new geometric proof state
    pub fn new(symbols: SymbolTable, facts: FactStore, goal: Goal) -> Self {
        Self {
            symbols,
            facts,
            goal,
            metadata: Metadata::default(),
        }
    }

    /// Create a new state with metadata
    pub fn with_metadata(
        symbols: SymbolTable,
        facts: FactStore,
        goal: Goal,
        metadata: Metadata,
    ) -> Self {
        Self {
            symbols,
            facts,
            goal,
            metadata,
        }
    }

    /// Compute a stable hash of this state for deduplication
    ///
    /// The hash is based on the content of facts and goals, not memory addresses.
    /// This allows detecting equivalent states even if they were constructed separately.
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash the facts (sorted for stability)
        let mut fact_strs: Vec<String> = self
            .facts
            .all_facts()
            .map(|f| format!("{:?}", f))
            .collect();
        fact_strs.sort();
        fact_strs.hash(&mut hasher);

        // Hash the goal
        let mut goal_strs: Vec<String> = self
            .goal
            .target_facts
            .iter()
            .map(|f| format!("{:?}", f))
            .collect();
        goal_strs.sort();
        goal_strs.hash(&mut hasher);

        hasher.finish()
    }

    /// Check if the goal is satisfied by the current facts
    ///
    /// Returns true if all target facts in the goal are present in the fact store.
    pub fn is_goal_satisfied(&self) -> bool {
        self.goal
            .target_facts
            .iter()
            .all(|fact| self.facts.contains(fact))
    }

    /// Add a new fact to the state
    ///
    /// Returns true if the fact was newly added, false if it was already known.
    pub fn add_fact(&mut self, fact: Fact) -> bool {
        self.facts.insert(fact)
    }

    /// Check if a fact is known in this state
    pub fn has_fact(&self, fact: &Fact) -> bool {
        self.facts.contains(fact)
    }

    /// Get the number of known facts
    pub fn num_facts(&self) -> usize {
        self.facts.len()
    }

    /// Check if any progress can still be made (goal not yet satisfied)
    pub fn is_active(&self) -> bool {
        !self.is_goal_satisfied()
    }

    /// Create a copy of this state with an additional fact
    pub fn with_fact(&self, fact: Fact) -> Self {
        let mut new_state = self.clone();
        new_state.add_fact(fact);
        new_state
    }

    /// Get all facts matching a predicate
    pub fn filter_facts<F>(&self, predicate: F) -> Vec<Fact>
    where
        F: Fn(&Fact) -> bool,
    {
        self.facts
            .all_facts()
            .filter(|f| predicate(f))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::facts::{Fact, FactStore, FactType};
    use super::super::symbols::{LineId, PointId, SymbolTable};
    use super::*;

    #[test]
    fn test_goal_creation() {
        let fact1 = Fact::Parallel(LineId(1), LineId(2));
        let fact2 = Fact::Perpendicular(LineId(1), LineId(3));

        let goal = Goal::new(vec![fact1.clone(), fact2.clone()]);

        assert_eq!(goal.len(), 2);
        assert!(goal.contains(&fact1));
        assert!(goal.contains(&fact2));
    }

    #[test]
    fn test_goal_single() {
        let fact = Fact::Parallel(LineId(1), LineId(2));
        let goal = Goal::single(fact.clone());

        assert_eq!(goal.len(), 1);
        assert!(goal.contains(&fact));
    }

    #[test]
    fn test_metadata() {
        let mut meta = Metadata::new("problem_123".to_string());
        meta.add_extra("difficulty".to_string(), "hard".to_string());

        assert_eq!(meta.problem_id, "problem_123");
        assert_eq!(meta.get_extra("difficulty"), Some("hard"));
        assert_eq!(meta.get_extra("nonexistent"), None);
    }

    #[test]
    fn test_metadata_builder() {
        let meta = Metadata::new("problem_456".to_string())
            .with_diagram("<svg>...</svg>".to_string());

        assert!(meta.diagram_info.is_some());
    }

    #[test]
    fn test_geostate_creation() {
        let symbols = SymbolTable::new();
        let facts = FactStore::new();
        let goal = Goal::single(Fact::Parallel(LineId(1), LineId(2)));

        let state = GeoState::new(symbols, facts, goal);

        assert_eq!(state.num_facts(), 0);
        assert!(!state.is_goal_satisfied());
    }

    #[test]
    fn test_goal_satisfaction() {
        let symbols = SymbolTable::new();
        let mut facts = FactStore::new();
        let goal_fact = Fact::Parallel(LineId(1), LineId(2));
        let goal = Goal::single(goal_fact.clone());

        let mut state = GeoState::new(symbols, facts, goal);

        assert!(!state.is_goal_satisfied(), "Goal should not be satisfied initially");

        state.add_fact(goal_fact);

        assert!(state.is_goal_satisfied(), "Goal should be satisfied after adding target fact");
    }

    #[test]
    fn test_state_hash_stability() {
        let symbols1 = SymbolTable::new();
        let symbols2 = SymbolTable::new();

        let mut facts1 = FactStore::new();
        facts1.insert(Fact::Parallel(LineId(1), LineId(2)));

        let mut facts2 = FactStore::new();
        facts2.insert(Fact::Parallel(LineId(1), LineId(2)));

        let goal = Goal::single(Fact::Perpendicular(LineId(1), LineId(3)));

        let state1 = GeoState::new(symbols1, facts1, goal.clone());
        let state2 = GeoState::new(symbols2, facts2, goal);

        assert_eq!(
            state1.hash(),
            state2.hash(),
            "Identical states should have same hash"
        );
    }

    #[test]
    fn test_state_hash_differs() {
        let symbols = SymbolTable::new();

        let mut facts1 = FactStore::new();
        facts1.insert(Fact::Parallel(LineId(1), LineId(2)));

        let mut facts2 = FactStore::new();
        facts2.insert(Fact::Perpendicular(LineId(1), LineId(2)));

        let goal = Goal::single(Fact::Parallel(LineId(3), LineId(4)));

        let state1 = GeoState::new(symbols.clone(), facts1, goal.clone());
        let state2 = GeoState::new(symbols, facts2, goal);

        assert_ne!(
            state1.hash(),
            state2.hash(),
            "Different states should have different hashes"
        );
    }

    #[test]
    fn test_add_fact() {
        let symbols = SymbolTable::new();
        let facts = FactStore::new();
        let goal = Goal::single(Fact::Parallel(LineId(1), LineId(2)));

        let mut state = GeoState::new(symbols, facts, goal);

        let fact = Fact::Perpendicular(LineId(3), LineId(4));
        assert!(state.add_fact(fact.clone()), "First addition should be new");
        assert!(!state.add_fact(fact), "Second addition should be duplicate");
        assert_eq!(state.num_facts(), 1);
    }

    #[test]
    fn test_with_fact() {
        let symbols = SymbolTable::new();
        let facts = FactStore::new();
        let goal = Goal::single(Fact::Parallel(LineId(1), LineId(2)));

        let state1 = GeoState::new(symbols, facts, goal);
        let fact = Fact::Perpendicular(LineId(3), LineId(4));

        let state2 = state1.with_fact(fact.clone());

        assert_eq!(state1.num_facts(), 0, "Original state should be unchanged");
        assert_eq!(state2.num_facts(), 1, "New state should have added fact");
        assert!(state2.has_fact(&fact));
    }

    #[test]
    fn test_is_active() {
        let symbols = SymbolTable::new();
        let facts = FactStore::new();
        let goal_fact = Fact::Parallel(LineId(1), LineId(2));
        let goal = Goal::single(goal_fact.clone());

        let mut state = GeoState::new(symbols, facts, goal);

        assert!(state.is_active(), "State should be active when goal not satisfied");

        state.add_fact(goal_fact);

        assert!(!state.is_active(), "State should be inactive when goal satisfied");
    }

    #[test]
    fn test_filter_facts() {
        let symbols = SymbolTable::new();
        let mut facts = FactStore::new();
        facts.insert(Fact::Parallel(LineId(1), LineId(2)));
        facts.insert(Fact::Parallel(LineId(3), LineId(4)));
        facts.insert(Fact::Perpendicular(LineId(1), LineId(3)));

        let goal = Goal::single(Fact::Parallel(LineId(5), LineId(6)));
        let state = GeoState::new(symbols, facts, goal);

        let parallel_facts = state.filter_facts(|f| matches!(f, Fact::Parallel(_, _)));

        assert_eq!(parallel_facts.len(), 2);
    }

    #[test]
    fn test_multiple_goal_facts() {
        let symbols = SymbolTable::new();
        let mut facts = FactStore::new();

        let fact1 = Fact::Parallel(LineId(1), LineId(2));
        let fact2 = Fact::Perpendicular(LineId(1), LineId(3));

        let goal = Goal::new(vec![fact1.clone(), fact2.clone()]);
        let mut state = GeoState::new(symbols, facts, goal);

        assert!(!state.is_goal_satisfied());

        state.add_fact(fact1);
        assert!(!state.is_goal_satisfied(), "Only partial goal satisfaction");

        state.add_fact(fact2);
        assert!(state.is_goal_satisfied(), "Full goal satisfaction");
    }
}
