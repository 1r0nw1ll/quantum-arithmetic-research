//! Triangle deduction rules

use crate::ir::{GeoState, Fact, FactType, PointId, SegmentId, LineId};
use super::Rule;

/// Right triangle from perpendicular sides: A, B, C with AB⊥BC ⇒ RightTriangle(B, A, C)
pub struct RightTriangleFromPerpendicular;

impl Rule for RightTriangleFromPerpendicular {
    fn id(&self) -> &'static str {
        "right_triangle_from_perpendicular"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let perps = state.facts.facts_of_type(FactType::Perpendicular);

        // Find perpendicular lines and check if they form triangles
        for fact in perps.iter() {
            if let Fact::Perpendicular(l1, l2) = fact {
                // Try to find three points that form a right triangle
                // This is a simplified version - full implementation would need
                // to check which points lie on which lines
                // For now, we'll implement the core logic structure

                // TODO: Implement full point-on-line checking
            }
        }

        new_facts
    }
}

/// Isosceles triangle from equal segments: AB = AC ⇒ IsoscelesTriangle(A, B, C)
pub struct IsoscelesFromEqualSides;

impl Rule for IsoscelesFromEqualSides {
    fn id(&self) -> &'static str {
        "isosceles_from_equal_sides"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let equal_segs = state.facts.facts_of_type(FactType::EqualLength);

        // Find pairs of equal segments sharing a common endpoint
        for fact1 in equal_segs.iter() {
            if let Fact::EqualLength(seg1, seg2) = fact1 {
                // Check if these segments share an endpoint
                // If they do, and we can identify the third point, create isosceles triangle

                // TODO: Need segment endpoint tracking in SymbolTable
            }
        }

        new_facts
    }
}

/// Pythagorean triple implies right triangle
pub struct RightTriangleFromPythagorean;

impl Rule for RightTriangleFromPythagorean {
    fn id(&self) -> &'static str {
        "right_triangle_from_pythagorean"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let pyth = state.facts.facts_of_type(FactType::PythagoreanTriple);

        for fact in pyth.iter() {
            if let Fact::PythagoreanTriple(a, b, c) = fact {
                // If we have a Pythagorean triple of segments,
                // we need to check if they form a triangle
                // and deduce it's a right triangle

                // TODO: Need to track which segments belong to which triangles
            }
        }

        new_facts
    }
}

/// Right triangle implies perpendicular sides
pub struct PerpendicularFromRightTriangle;

impl Rule for PerpendicularFromRightTriangle {
    fn id(&self) -> &'static str {
        "perpendicular_from_right_triangle"
    }

    fn cost(&self) -> f64 {
        2.0
    }

    fn apply(&self, state: &GeoState) -> Vec<Fact> {
        let mut new_facts = Vec::new();
        let right_triangles = state.facts.facts_of_type(FactType::RightTriangle);

        for fact in right_triangles.iter() {
            if let Fact::RightTriangle(p1, p2, p3) = fact {
                // Right triangle at p1 means the angle at p1 is 90°
                // So the lines from p1 to p2 and p1 to p3 are perpendicular

                // We need to find the line IDs for these segments
                // This requires querying the symbol table for lines through these points

                // TODO: Add helper to find line through two points
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
    fn test_triangle_rules_exist() {
        // Placeholder - will add proper tests once we have coordinate geometry
        let rule1 = RightTriangleFromPerpendicular;
        let rule2 = IsoscelesFromEqualSides;
        let rule3 = RightTriangleFromPythagorean;
        let rule4 = PerpendicularFromRightTriangle;

        assert_eq!(rule1.id(), "right_triangle_from_perpendicular");
        assert_eq!(rule2.id(), "isosceles_from_equal_sides");
        assert_eq!(rule3.id(), "right_triangle_from_pythagorean");
        assert_eq!(rule4.id(), "perpendicular_from_right_triangle");
    }
}
