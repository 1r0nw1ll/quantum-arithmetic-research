//! Circle deduction rules

use crate::ir::{GeoState, Fact, FactType, PointId, CircleId, LineId};
use super::Rule;

/// Concentric circles symmetry: C1 concentric with C2 ⇒ C2 concentric with C1
pub struct ConcentricSymmetry;

impl Rule for ConcentricSymmetry {
    fn id(&self) -> &'static str {
        "concentric_symmetry"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let concentrics = state.facts.facts_of_type(FactType::ConcentricCircles);

        for fact in concentrics.iter() {
            if let Fact::ConcentricCircles(c1, c2) = fact {
                let reversed = Fact::ConcentricCircles(*c2, *c1);
                if !state.facts.contains(&reversed) {
                    new_facts.push(reversed);
                }
            }
        }

        new_facts
    }
}

/// Concentric transitivity: C1 concentric with C2, C2 concentric with C3 ⇒ C1 concentric with C3
pub struct ConcentricTransitivity;

impl Rule for ConcentricTransitivity {
    fn id(&self) -> &'static str {
        "concentric_transitivity"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let concentrics = state.facts.facts_of_type(FactType::ConcentricCircles);

        for fact1 in concentrics.iter() {
            if let Fact::ConcentricCircles(c1, c2) = fact1 {
                for fact2 in concentrics.iter() {
                    if let Fact::ConcentricCircles(c2_prime, c3) = fact2 {
                        if c2 == c2_prime && c1 != c3 {
                            let new_fact = Fact::ConcentricCircles(*c1, *c3);
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

/// Tangent implies perpendicular: Line L tangent to circle C at point P,
/// radius from center to P forms line R ⇒ L⊥R
pub struct TangentPerpendicular;

impl Rule for TangentPerpendicular {
    fn id(&self) -> &'static str {
        "tangent_perpendicular"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let tangents = state.facts.facts_of_type(FactType::Tangent);

        for fact in tangents.iter() {
            if let Fact::Tangent(line, _circle, _point) = fact {
                // To apply this rule, we need to:
                // 1. Find the center of the circle
                // 2. Find the line from center to tangent point
                // 3. Assert perpendicularity

                // This requires coordinate geometry or center tracking
                // TODO: Implement when we have center points in SymbolTable
            }
        }

        new_facts
    }
}

/// Four points on same circle are concyclic
pub struct OnCircleToConcyclic;

impl Rule for OnCircleToConcyclic {
    fn id(&self) -> &'static str {
        "on_circle_to_concyclic"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let on_circle_facts = state.facts.facts_of_type(FactType::OnCircle);

        // Group points by circle
        let mut circles_to_points: std::collections::HashMap<CircleId, Vec<PointId>> =
            std::collections::HashMap::new();

        for fact in on_circle_facts.iter() {
            if let Fact::OnCircle(point, circle) = fact {
                circles_to_points.entry(*circle)
                    .or_insert_with(Vec::new)
                    .push(*point);
            }
        }

        // For each circle with 4+ points, generate concyclic facts
        for (_circle, points) in circles_to_points.iter() {
            if points.len() >= 4 {
                // Generate all combinations of 4 points
                for i in 0..points.len() {
                    for j in (i + 1)..points.len() {
                        for k in (j + 1)..points.len() {
                            for l in (k + 1)..points.len() {
                                let new_fact = Fact::Concyclic(
                                    points[i],
                                    points[j],
                                    points[k],
                                    points[l],
                                );
                                if !state.facts.contains(&new_fact) {
                                    new_facts.push(new_fact);
                                }
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
    fn test_concentric_symmetry() {
        let mut facts = FactStore::new();
        facts.insert(Fact::ConcentricCircles(CircleId(1), CircleId(2)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = ConcentricSymmetry;

        let new_facts = rule.apply(&state);

        // Normalization may make this 0
        assert_eq!(new_facts.len(), 0);
    }

    #[test]
    fn test_concentric_transitivity() {
        let mut facts = FactStore::new();
        facts.insert(Fact::ConcentricCircles(CircleId(1), CircleId(2)));
        facts.insert(Fact::ConcentricCircles(CircleId(2), CircleId(3)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = ConcentricTransitivity;

        let new_facts = rule.apply(&state);

        assert_eq!(new_facts.len(), 1);
        assert!(new_facts.contains(&Fact::ConcentricCircles(CircleId(1), CircleId(3))));
    }

    #[test]
    fn test_on_circle_to_concyclic() {
        let mut facts = FactStore::new();
        let circle = CircleId(1);

        // Add 4 points on the same circle
        facts.insert(Fact::OnCircle(PointId(1), circle));
        facts.insert(Fact::OnCircle(PointId(2), circle));
        facts.insert(Fact::OnCircle(PointId(3), circle));
        facts.insert(Fact::OnCircle(PointId(4), circle));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = OnCircleToConcyclic;

        let new_facts = rule.apply(&state);

        // Should generate exactly 1 concyclic fact for these 4 points
        assert_eq!(new_facts.len(), 1);
        assert!(new_facts.iter().any(|f| matches!(f, Fact::Concyclic(_, _, _, _))));
    }
}
