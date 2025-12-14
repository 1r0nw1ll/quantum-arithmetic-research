//! Perpendicular line deduction rules

use crate::ir::{GeoState, Fact, FactType, LineId};
use super::Rule;

/// Perpendicular symmetry: A⊥B ⇒ B⊥A
pub struct PerpendicularSymmetry;

impl Rule for PerpendicularSymmetry {
    fn id(&self) -> &'static str {
        "perpendicular_symmetry"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let perps = state.facts.facts_of_type(FactType::Perpendicular);

        for fact in perps.iter() {
            if let Fact::Perpendicular(l1, l2) = fact {
                let reversed = Fact::Perpendicular(*l2, *l1);
                if !state.facts.contains(&reversed) {
                    new_facts.push(reversed);
                }
            }
        }

        new_facts
    }
}

/// Perpendicular to parallel: A⊥B, A⊥C ⇒ B∥C
pub struct PerpendicularToParallel;

impl Rule for PerpendicularToParallel {
    fn id(&self) -> &'static str {
        "perpendicular_to_parallel"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let perps = state.facts.facts_of_type(FactType::Perpendicular);

        // Find pairs of perpendiculars sharing a common line
        for fact1 in perps.iter() {
            if let Fact::Perpendicular(a, b) = fact1 {
                for fact2 in perps.iter() {
                    if let Fact::Perpendicular(a_prime, c) = fact2 {
                        // If A⊥B and A⊥C, then B∥C
                        if a == a_prime && b != c {
                            let new_fact = Fact::Parallel(*b, *c);
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

/// Parallel + perpendicular: A∥B, C⊥A ⇒ C⊥B
pub struct ParallelPerpendicular;

impl Rule for ParallelPerpendicular {
    fn id(&self) -> &'static str {
        "parallel_perpendicular"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let parallels = state.facts.facts_of_type(FactType::Parallel);
        let perps = state.facts.facts_of_type(FactType::Perpendicular);

        // For each parallel pair and perpendicular, propagate
        for par_fact in parallels.iter() {
            if let Fact::Parallel(a, b) = par_fact {
                for perp_fact in perps.iter() {
                    if let Fact::Perpendicular(l1, l2) = perp_fact {
                        // If A∥B and C⊥A, then C⊥B
                        // Check both positions since perpendicular is normalized
                        if l1 == a {
                            let new_fact = Fact::Perpendicular(*l2, *b);
                            if !state.facts.contains(&new_fact) {
                                new_facts.push(new_fact);
                            }
                        }
                        if l2 == a {
                            let new_fact = Fact::Perpendicular(*l1, *b);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Goal, FactStore};

    #[test]
    fn test_perpendicular_symmetry() {
        let mut facts = FactStore::new();
        facts.insert(Fact::Perpendicular(LineId(1), LineId(2)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = PerpendicularSymmetry;

        let new_facts = rule.apply(&state);

        // Normalization means L1⊥L2 == L2⊥L1 after normalize
        assert_eq!(new_facts.len(), 0);
    }

    #[test]
    fn test_perpendicular_to_parallel() {
        let mut facts = FactStore::new();
        facts.insert(Fact::Perpendicular(LineId(1), LineId(2)));
        facts.insert(Fact::Perpendicular(LineId(1), LineId(3)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = PerpendicularToParallel;

        let new_facts = rule.apply(&state);

        // Should produce 2 facts because we check both orderings:
        // L1⊥L2, L1⊥L3 => L2∥L3
        // AND the normalized L2⊥L1, L3⊥L1 => L2∥L3 (same)
        // After dedup, could be 1 or 2 depending on exact matching
        assert!(new_facts.len() >= 1);
        assert!(new_facts.iter().any(|f| matches!(f, Fact::Parallel(l2, l3) if (*l2 == LineId(2) || *l2 == LineId(3)) && (*l3 == LineId(2) || *l3 == LineId(3)))));
    }

    #[test]
    fn test_parallel_perpendicular_propagation() {
        let mut facts = FactStore::new();
        facts.insert(Fact::Parallel(LineId(1), LineId(2)));
        facts.insert(Fact::Perpendicular(LineId(3), LineId(1)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = ParallelPerpendicular;

        let new_facts = rule.apply(&state);

        // Should produce at least 1 new perpendicular fact
        assert!(new_facts.len() >= 1);
        assert!(new_facts.iter().any(|f| matches!(f, Fact::Perpendicular(l3, l2) if *l3 == LineId(3) && *l2 == LineId(2))));
    }
}
