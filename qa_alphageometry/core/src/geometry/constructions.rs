//! Geometric constructions
//!
//! Add points, lines, circles to the geometric state.
//! Pure operations - no heuristics.

use crate::ir::{GeoState, Fact, PointId, LineId, CircleId, SegmentId};

/// Construction operations on geometric states
pub trait Construct {
    /// Add a point with a given label
    fn add_point(&mut self, label: &str) -> PointId;

    /// Add a line with a given label
    fn add_line(&mut self, label: &str) -> LineId;

    /// Add a circle with a given label
    fn add_circle(&mut self, label: &str) -> CircleId;

    /// Add a segment with a given label
    fn add_segment(&mut self, label: &str) -> SegmentId;

    /// Construct a line through two points
    fn line_through_points(&mut self, p1: PointId, p2: PointId, label: &str) -> LineId;

    /// Construct perpendicular from point to line
    fn perpendicular_from_point(&mut self, point: PointId, line: LineId, label: &str) -> LineId;

    /// Construct parallel line through point
    fn parallel_through_point(&mut self, point: PointId, line: LineId, label: &str) -> LineId;
}

impl Construct for GeoState {
    fn add_point(&mut self, label: &str) -> PointId {
        self.symbols.get_or_intern_point(label)
    }

    fn add_line(&mut self, label: &str) -> LineId {
        self.symbols.get_or_intern_line(label)
    }

    fn add_circle(&mut self, label: &str) -> CircleId {
        self.symbols.get_or_intern_circle(label)
    }

    fn add_segment(&mut self, label: &str) -> SegmentId {
        self.symbols.get_or_intern_segment(label)
    }

    fn line_through_points(&mut self, p1: PointId, p2: PointId, label: &str) -> LineId {
        let line = self.add_line(label);

        // Add facts: points are on the line
        self.facts.insert(Fact::OnLine(p1, line));
        self.facts.insert(Fact::OnLine(p2, line));

        // Points are collinear (they define the line)
        // Note: For 3 points we need a third point, so we don't add Collinear here

        line
    }

    fn perpendicular_from_point(&mut self, point: PointId, line: LineId, label: &str) -> LineId {
        let perp_line = self.add_line(label);

        // Add facts
        self.facts.insert(Fact::OnLine(point, perp_line));
        self.facts.insert(Fact::Perpendicular(perp_line, line));

        perp_line
    }

    fn parallel_through_point(&mut self, point: PointId, line: LineId, label: &str) -> LineId {
        let par_line = self.add_line(label);

        // Add facts
        self.facts.insert(Fact::OnLine(point, par_line));
        self.facts.insert(Fact::Parallel(par_line, line));

        par_line
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Goal, Metadata};

    #[test]
    fn test_add_point() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let p1 = state.add_point("A");
        let p2 = state.add_point("B");
        let p3 = state.add_point("A"); // Duplicate label

        assert_eq!(p1, p3); // Same label -> same ID
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_line_through_points() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let p1 = state.add_point("A");
        let p2 = state.add_point("B");
        let line = state.line_through_points(p1, p2, "AB");

        assert!(state.facts.contains(&Fact::OnLine(p1, line)));
        assert!(state.facts.contains(&Fact::OnLine(p2, line)));
    }

    #[test]
    fn test_perpendicular_construction() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let point = state.add_point("P");
        let line = state.add_line("L");
        let perp = state.perpendicular_from_point(point, line, "perp");

        assert!(state.facts.contains(&Fact::Perpendicular(perp, line)));
        assert!(state.facts.contains(&Fact::OnLine(point, perp)));
    }

    #[test]
    fn test_parallel_construction() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let point = state.add_point("P");
        let line = state.add_line("L");
        let par = state.parallel_through_point(point, line, "par");

        assert!(state.facts.contains(&Fact::Parallel(par, line)));
        assert!(state.facts.contains(&Fact::OnLine(point, par)));
    }
}
