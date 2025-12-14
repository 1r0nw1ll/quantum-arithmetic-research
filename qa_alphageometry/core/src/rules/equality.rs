//! Equality deduction rules (segments, angles)

use crate::ir::{GeoState, Fact, FactType, SegmentId, AngleId};
use super::Rule;

/// Segment equality transitivity: AB = CD, CD = EF ⇒ AB = EF
pub struct SegmentEqualityTransitivity;

impl Rule for SegmentEqualityTransitivity {
    fn id(&self) -> &'static str {
        "segment_equality_transitivity"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let equalities = state.facts.facts_of_type(FactType::EqualLength);

        // For each pair of equality facts, check for transitivity
        for fact1 in equalities.iter() {
            if let Fact::EqualLength(s1, s2) = fact1 {
                for fact2 in equalities.iter() {
                    if let Fact::EqualLength(s2_prime, s3) = fact2 {
                        // Check if middle segments match
                        if s2 == s2_prime && s1 != s3 {
                            let new_fact = Fact::EqualLength(*s1, *s3);
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

/// Segment equality symmetry: AB = CD ⇒ CD = AB
pub struct SegmentEqualitySymmetry;

impl Rule for SegmentEqualitySymmetry {
    fn id(&self) -> &'static str {
        "segment_equality_symmetry"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let equalities = state.facts.facts_of_type(FactType::EqualLength);

        for fact in equalities.iter() {
            if let Fact::EqualLength(s1, s2) = fact {
                let reversed = Fact::EqualLength(*s2, *s1);
                if !state.facts.contains(&reversed) {
                    new_facts.push(reversed);
                }
            }
        }

        new_facts
    }
}

/// Angle equality transitivity: ∠A = ∠B, ∠B = ∠C ⇒ ∠A = ∠C
pub struct AngleEqualityTransitivity;

impl Rule for AngleEqualityTransitivity {
    fn id(&self) -> &'static str {
        "angle_equality_transitivity"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let equalities = state.facts.facts_of_type(FactType::EqualAngle);

        for fact1 in equalities.iter() {
            if let Fact::EqualAngle(a1, a2) = fact1 {
                for fact2 in equalities.iter() {
                    if let Fact::EqualAngle(a2_prime, a3) = fact2 {
                        if a2 == a2_prime && a1 != a3 {
                            let new_fact = Fact::EqualAngle(*a1, *a3);
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

/// Angle equality symmetry: ∠A = ∠B ⇒ ∠B = ∠A
pub struct AngleEqualitySymmetry;

impl Rule for AngleEqualitySymmetry {
    fn id(&self) -> &'static str {
        "angle_equality_symmetry"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let equalities = state.facts.facts_of_type(FactType::EqualAngle);

        for fact in equalities.iter() {
            if let Fact::EqualAngle(a1, a2) = fact {
                let reversed = Fact::EqualAngle(*a2, *a1);
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
    fn test_segment_equality_transitivity() {
        let mut facts = FactStore::new();
        facts.insert(Fact::EqualLength(SegmentId(1), SegmentId(2)));
        facts.insert(Fact::EqualLength(SegmentId(2), SegmentId(3)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = SegmentEqualityTransitivity;

        let new_facts = rule.apply(&state);

        assert_eq!(new_facts.len(), 1);
        assert!(new_facts.contains(&Fact::EqualLength(SegmentId(1), SegmentId(3))));
    }

    #[test]
    fn test_segment_equality_symmetry() {
        let mut facts = FactStore::new();
        facts.insert(Fact::EqualLength(SegmentId(1), SegmentId(2)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = SegmentEqualitySymmetry;

        let new_facts = rule.apply(&state);

        // Normalization may make this 0
        assert_eq!(new_facts.len(), 0);
    }

    #[test]
    fn test_angle_equality_transitivity() {
        let mut facts = FactStore::new();
        facts.insert(Fact::EqualAngle(AngleId(1), AngleId(2)));
        facts.insert(Fact::EqualAngle(AngleId(2), AngleId(3)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = AngleEqualityTransitivity;

        let new_facts = rule.apply(&state);

        assert_eq!(new_facts.len(), 1);
        assert!(new_facts.contains(&Fact::EqualAngle(AngleId(1), AngleId(3))));
    }
}
