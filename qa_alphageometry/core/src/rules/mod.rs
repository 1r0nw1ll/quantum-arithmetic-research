//! Deduction rules for geometric reasoning
//!
//! Implements 20+ inference rules that produce new facts from existing facts.

pub mod parallel;
pub mod perpendicular;
pub mod equality;
pub mod circle;
pub mod collinear;
pub mod angle;
pub mod triangle;

pub use parallel::*;
pub use perpendicular::*;
pub use equality::*;
pub use circle::*;
pub use collinear::*;
pub use angle::*;
pub use triangle::*;

use crate::ir::{GeoState, Fact};

/// Rule trait - all deduction rules implement this
pub trait Rule: Send + Sync {
    /// Unique identifier for this rule
    fn id(&self) -> &'static str;

    /// Cost of applying this rule (lower = prefer)
    fn cost(&self) -> f64 {
        1.0
    }

    /// Apply rule to state, producing new facts
    ///
    /// Returns only NEW facts (not already in state)
    fn apply(&self, state: &GeoState) -> Vec<Fact>;
}

/// Get all available deduction rules
pub fn all_rules() -> Vec<Box<dyn Rule>> {
    let mut rules: Vec<Box<dyn Rule>> = Vec::new();

    // Parallel rules (2)
    rules.push(Box::new(parallel::ParallelTransitivity));
    rules.push(Box::new(parallel::ParallelSymmetry));

    // Perpendicular rules (3)
    rules.push(Box::new(perpendicular::PerpendicularSymmetry));
    rules.push(Box::new(perpendicular::PerpendicularToParallel));
    rules.push(Box::new(perpendicular::ParallelPerpendicular));

    // Equality rules (4)
    rules.push(Box::new(equality::SegmentEqualityTransitivity));
    rules.push(Box::new(equality::SegmentEqualitySymmetry));
    rules.push(Box::new(equality::AngleEqualityTransitivity));
    rules.push(Box::new(equality::AngleEqualitySymmetry));

    // Circle rules (4)
    rules.push(Box::new(circle::ConcentricSymmetry));
    rules.push(Box::new(circle::ConcentricTransitivity));
    rules.push(Box::new(circle::TangentPerpendicular));
    rules.push(Box::new(circle::OnCircleToConcyclic));

    // Collinearity rules (3)
    rules.push(Box::new(collinear::CollinearityPermutation));
    rules.push(Box::new(collinear::OnLineToCollinear));
    rules.push(Box::new(collinear::CollinearTransitivity));

    // Angle/line rules (4)
    rules.push(Box::new(angle::RightAngleFromPerpendicular));
    rules.push(Box::new(angle::RightAngleFromPerpendicularSegments));
    rules.push(Box::new(angle::CoincidentLineSymmetry));
    rules.push(Box::new(angle::CoincidentLineTransitivity));

    // Triangle rules (4) - placeholders for now
    rules.push(Box::new(triangle::RightTriangleFromPerpendicular));
    rules.push(Box::new(triangle::IsoscelesFromEqualSides));
    rules.push(Box::new(triangle::RightTriangleFromPythagorean));
    rules.push(Box::new(triangle::PerpendicularFromRightTriangle));

    rules
}
