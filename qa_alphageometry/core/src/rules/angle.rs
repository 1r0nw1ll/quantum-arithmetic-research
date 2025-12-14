//! Angle deduction rules

use crate::ir::{GeoState, Fact, FactType, AngleId, LineId, PointId};
use super::Rule;

/// Perpendicular lines form right angle
pub struct RightAngleFromPerpendicular;

impl Rule for RightAngleFromPerpendicular {
    fn id(&self) -> &'static str {
        "right_angle_from_perpendicular"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let perps = state.facts.facts_of_type(FactType::Perpendicular);

        for fact in perps.iter() {
            if let Fact::Perpendicular(_l1, _l2) = fact {
                // To create a RightAngle fact, we need the AngleId
                // This requires knowing which angle is formed by these two lines
                // This needs coordinate geometry or angle tracking

                // TODO: Implement when we have angle construction from lines
            }
        }

        new_facts
    }
}

/// Perpendicular segments form right angle
pub struct RightAngleFromPerpendicularSegments;

impl Rule for RightAngleFromPerpendicularSegments {
    fn id(&self) -> &'static str {
        "right_angle_from_perpendicular_segments"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let perp_segs = state.facts.facts_of_type(FactType::PerpendicularSegments);

        for fact in perp_segs.iter() {
            if let Fact::PerpendicularSegments(_s1, _s2) = fact {
                // Similar to above - needs angle construction
                // TODO: Implement with coordinate geometry
            }
        }

        new_facts
    }
}

/// Coincident lines share all points
pub struct CoincidentLineSymmetry;

impl Rule for CoincidentLineSymmetry {
    fn id(&self) -> &'static str {
        "coincident_line_symmetry"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let coincidents = state.facts.facts_of_type(FactType::CoincidentLines);

        for fact in coincidents.iter() {
            if let Fact::CoincidentLines(l1, l2) = fact {
                let reversed = Fact::CoincidentLines(*l2, *l1);
                if !state.facts.contains(&reversed) {
                    new_facts.push(reversed);
                }
            }
        }

        new_facts
    }
}

/// Coincident line transitivity
pub struct CoincidentLineTransitivity;

impl Rule for CoincidentLineTransitivity {
    fn id(&self) -> &'static str {
        "coincident_line_transitivity"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let coincidents = state.facts.facts_of_type(FactType::CoincidentLines);

        for fact1 in coincidents.iter() {
            if let Fact::CoincidentLines(l1, l2) = fact1 {
                for fact2 in coincidents.iter() {
                    if let Fact::CoincidentLines(l2_prime, l3) = fact2 {
                        if l2 == l2_prime && l1 != l3 {
                            let new_fact = Fact::CoincidentLines(*l1, *l3);
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
    fn test_coincident_line_transitivity() {
        let mut facts = FactStore::new();
        facts.insert(Fact::CoincidentLines(LineId(1), LineId(2)));
        facts.insert(Fact::CoincidentLines(LineId(2), LineId(3)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = CoincidentLineTransitivity;

        let new_facts = rule.apply(&state);

        assert_eq!(new_facts.len(), 1);
        assert!(new_facts.contains(&Fact::CoincidentLines(LineId(1), LineId(3))));
    }

    #[test]
    fn test_coincident_line_symmetry() {
        let mut facts = FactStore::new();
        facts.insert(Fact::CoincidentLines(LineId(1), LineId(2)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = CoincidentLineSymmetry;

        let new_facts = rule.apply(&state);

        // Normalization may make this 0
        assert_eq!(new_facts.len(), 0);
    }
}
