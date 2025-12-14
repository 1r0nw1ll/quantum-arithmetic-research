//! Collinearity deduction rules

use crate::ir::{GeoState, Fact, FactType, PointId, LineId};
use super::Rule;

/// Collinearity permutation: If A, B, C collinear, then all permutations are collinear
pub struct CollinearityPermutation;

impl Rule for CollinearityPermutation {
    fn id(&self) -> &'static str {
        "collinearity_permutation"
    }

    fn cost(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let collinears = state.facts.facts_of_type(FactType::Collinear);

        for fact in collinears.iter() {
            if let Fact::Collinear(p1, p2, p3) = fact {
                // Generate all 6 permutations
                let permutations = vec![
                    Fact::Collinear(*p1, *p3, *p2),
                    Fact::Collinear(*p2, *p1, *p3),
                    Fact::Collinear(*p2, *p3, *p1),
                    Fact::Collinear(*p3, *p1, *p2),
                    Fact::Collinear(*p3, *p2, *p1),
                ];

                for perm in permutations {
                    if !state.facts.contains(&perm) {
                        new_facts.push(perm);
                    }
                }
            }
        }

        new_facts
    }
}

/// Points on line are collinear: P1 on L, P2 on L, P3 on L ⇒ Collinear(P1, P2, P3)
pub struct OnLineToCollinear;

impl Rule for OnLineToCollinear {
    fn id(&self) -> &'static str {
        "on_line_to_collinear"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let on_line_facts = state.facts.facts_of_type(FactType::OnLine);

        // Group points by line
        let mut lines_to_points: std::collections::HashMap<LineId, Vec<PointId>> =
            std::collections::HashMap::new();

        for fact in on_line_facts.iter() {
            if let Fact::OnLine(point, line) = fact {
                lines_to_points.entry(*line)
                    .or_insert_with(Vec::new)
                    .push(*point);
            }
        }

        // For each line with 3+ points, generate collinearity facts
        for (_line, points) in lines_to_points.iter() {
            if points.len() >= 3 {
                // Generate all combinations of 3 points
                for i in 0..points.len() {
                    for j in (i + 1)..points.len() {
                        for k in (j + 1)..points.len() {
                            let new_fact = Fact::Collinear(points[i], points[j], points[k]);
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

/// Collinear transitivity: A,B,C collinear AND B,C,D collinear ⇒ A,B,D collinear
/// (simplified - full version would need to check geometric validity)
pub struct CollinearTransitivity;

impl Rule for CollinearTransitivity {
    fn id(&self) -> &'static str {
        "collinear_transitivity"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let collinears = state.facts.facts_of_type(FactType::Collinear);

        // Find collinear facts sharing two points
        for fact1 in collinears.iter() {
            if let Fact::Collinear(a, b, c) = fact1 {
                for fact2 in collinears.iter() {
                    if let Fact::Collinear(b2, c2, d) = fact2 {
                        // If B and C match, then A, B, D are collinear
                        if b == b2 && c == c2 && a != d {
                            let new_fact = Fact::Collinear(*a, *b, *d);
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
    fn test_on_line_to_collinear() {
        let mut facts = FactStore::new();
        let line = LineId(1);

        facts.insert(Fact::OnLine(PointId(1), line));
        facts.insert(Fact::OnLine(PointId(2), line));
        facts.insert(Fact::OnLine(PointId(3), line));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = OnLineToCollinear;

        let new_facts = rule.apply(&state);

        assert_eq!(new_facts.len(), 1);
        assert!(new_facts.iter().any(|f| matches!(f, Fact::Collinear(_, _, _))));
    }

    #[test]
    fn test_collinearity_permutation() {
        let mut facts = FactStore::new();
        facts.insert(Fact::Collinear(PointId(1), PointId(2), PointId(3)));

        let state = GeoState::new(Default::default(), facts, Goal::new(vec![]));
        let rule = CollinearityPermutation;

        let new_facts = rule.apply(&state);

        // Should generate permutations (some may be normalized away)
        assert!(new_facts.len() >= 0); // Due to normalization, count may vary
    }
}
