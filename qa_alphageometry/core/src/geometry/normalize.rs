//! Fact normalization and canonicalization
//!
//! Ensures geometric facts are in canonical form for deduplication.
//! All symmetric predicates are normalized (sorted by ID).

use crate::ir::{Fact, AngleId, LineId, PointId};

/// Normalize a fact to canonical form
///
/// Symmetric predicates are sorted by ID to ensure:
/// - Parallel(L1, L2) == Parallel(L2, L1)
/// - Collinear(P1, P2, P3) has sorted point IDs
///
/// This is already implemented in the Fact enum's normalize() method,
/// but this module provides utilities for batch normalization.
pub fn normalize_fact(fact: Fact) -> Fact {
    fact.normalize()
}

/// Normalize a collection of facts
pub fn normalize_facts(facts: Vec<Fact>) -> Vec<Fact> {
    facts.into_iter().map(normalize_fact).collect()
}

/// Check if two facts are equivalent (after normalization)
pub fn facts_equivalent(f1: &Fact, f2: &Fact) -> bool {
    f1.clone().normalize() == f2.clone().normalize()
}

/// Deduplicate a list of facts (after normalization)
pub fn deduplicate_facts(facts: Vec<Fact>) -> Vec<Fact> {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for fact in facts {
        let normalized = fact.normalize();
        if seen.insert(normalized.clone()) {
            result.push(normalized);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_parallel() {
        let l1 = LineId(1);
        let l2 = LineId(2);

        let f1 = Fact::Parallel(l1, l2);
        let f2 = Fact::Parallel(l2, l1);

        assert_eq!(normalize_fact(f1), normalize_fact(f2));
    }

    #[test]
    fn test_deduplicate() {
        let p1 = PointId(1);
        let p2 = PointId(2);
        let p3 = PointId(3);

        let facts = vec![
            Fact::Collinear(p1, p2, p3),
            Fact::Collinear(p2, p1, p3), // Duplicate after normalization
            Fact::Collinear(p1, p2, p3), // Exact duplicate
        ];

        let deduped = deduplicate_facts(facts);
        assert_eq!(deduped.len(), 1);
    }

    #[test]
    fn test_facts_equivalent() {
        let l1 = LineId(5);
        let l2 = LineId(10);

        let f1 = Fact::Perpendicular(l1, l2);
        let f2 = Fact::Perpendicular(l2, l1);

        assert!(facts_equivalent(&f1, &f2));
    }
}
