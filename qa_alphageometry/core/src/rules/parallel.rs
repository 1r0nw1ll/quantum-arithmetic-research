//! Parallel line deduction rules

use crate::ir::{GeoState, Fact, FactType, LineId};
use super::Rule;

/// Parallel transitivity: A∥B, B∥C ⇒ A∥C
pub struct ParallelTransitivity;

impl Rule for ParallelTransitivity {
    fn id(&self) -> &'static str {
        "parallel_transitivity"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let parallels = state.facts.facts_of_type(FactType::Parallel);

        // For each pair of parallel facts, check for transitivity
        for fact1 in parallels.iter() {
            if let Fact::Parallel(l1, l2) = fact1 {
                for fact2 in parallels.iter() {
                    if let Fact::Parallel(l2_prime, l3) = fact2 {
                        // Check if middle lines match
                        if l2 == l2_prime && l1 != l3 {
                            let new_fact = Fact::Parallel(*l1, *l3);
                            if !state.facts.contains(&new_fact) {
                                new_facts.push(new_fact);
                            }
                        }
                    }
                }
            }
        }

        new_facts
    }
}

/// Parallel symmetry: A∥B ⇒ B∥A
pub struct ParallelSymmetry;

impl Rule for ParallelSymmetry {
    fn id(&self) -> &'static str {
        "parallel_symmetry"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let parallels = state.facts.facts_of_type(FactType::Parallel);

        for fact in parallels.iter() {
            if let Fact::Parallel(l1, l2) = fact {
                let reversed = Fact::Parallel(*l2, *l1);
                if !state.facts.contains(&reversed) {
                    new_facts.push(reversed);
                }
            }
        }

        new_facts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Goal, FactStore};

    #[test]
    fn test_parallel_transitivity() {
        let mut facts = FactStore::new();
        facts.insert(Fact::Parallel(LineId(1), LineId(2)));
        facts.insert(Fact::Parallel(LineId(2), LineId(3)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = ParallelTransitivity;

        let new_facts = rule.apply(&state);

        assert_eq!(new_facts.len(), 1);
        assert!(new_facts.contains(&Fact::Parallel(LineId(1), LineId(3))));
    }

    #[test]
    fn test_parallel_symmetry() {
        let mut facts = FactStore::new();
        facts.insert(Fact::Parallel(LineId(1), LineId(2)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = ParallelSymmetry;

        let new_facts = rule.apply(&state);

        // Normalization means L1∥L2 == L2∥L1 after normalize
        // So the reverse fact already exists
        assert_eq!(new_facts.len(), 0);
    }

    #[test]
    fn test_no_self_parallel() {
        let mut facts = FactStore::new();
        facts.insert(Fact::Parallel(LineId(1), LineId(1)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = ParallelTransitivity;

        let new_facts = rule.apply(&state);

        // Should not produce L1∥L1 again
        assert!(new_facts.is_empty());
    }
}
