//! Geometric detection and checking
//!
//! Detect patterns: right triangles, perpendicularity, collinearity, etc.
//! Pure semantic operations.

use crate::ir::{GeoState, Fact, PointId, LineId, AngleId};

/// Detection operations on geometric states
pub trait Detect {
    /// Detect if three points form a right triangle
    fn detect_right_triangle(&self, p1: PointId, p2: PointId, p3: PointId) -> bool;

    /// Detect if two lines are perpendicular
    fn are_perpendicular(&self, l1: LineId, l2: LineId) -> bool;

    /// Detect if two lines are parallel
    fn are_parallel(&self, l1: LineId, l2: LineId) -> bool;

    /// Detect if three points are collinear
    fn are_collinear(&self, p1: PointId, p2: PointId, p3: PointId) -> bool;

    /// Detect if a point is on a line
    fn is_point_on_line(&self, point: PointId, line: LineId) -> bool;

    /// Find all right triangles in the current state
    fn find_right_triangles(&self) -> Vec<(PointId, PointId, PointId)>;

    /// Find all perpendicular line pairs
    fn find_perpendicular_pairs(&self) -> Vec<(LineId, LineId)>;

    /// Find all parallel line pairs
    fn find_parallel_pairs(&self) -> Vec<(LineId, LineId)>;
}

impl Detect for GeoState {
    fn detect_right_triangle(&self, p1: PointId, p2: PointId, p3: PointId) -> bool {
        // Check if RightTriangle fact exists for these points (in any order)
        let candidates = vec![
            Fact::RightTriangle(p1, p2, p3),
            Fact::RightTriangle(p1, p3, p2),
            Fact::RightTriangle(p2, p1, p3),
            Fact::RightTriangle(p2, p3, p1),
            Fact::RightTriangle(p3, p1, p2),
            Fact::RightTriangle(p3, p2, p1),
        ];

        candidates.iter().any(|f| self.facts.contains(f))
    }

    fn are_perpendicular(&self, l1: LineId, l2: LineId) -> bool {
        self.facts.contains(&Fact::Perpendicular(l1, l2)) ||
        self.facts.contains(&Fact::Perpendicular(l2, l1))
    }

    fn are_parallel(&self, l1: LineId, l2: LineId) -> bool {
        self.facts.contains(&Fact::Parallel(l1, l2)) ||
        self.facts.contains(&Fact::Parallel(l2, l1))
    }

    fn are_collinear(&self, p1: PointId, p2: PointId, p3: PointId) -> bool {
        // Check all permutations (normalization happens in Fact)
        let candidates = vec![
            Fact::Collinear(p1, p2, p3),
            Fact::Collinear(p1, p3, p2),
            Fact::Collinear(p2, p1, p3),
        ];

        candidates.iter().any(|f| self.facts.contains(f))
    }

    fn is_point_on_line(&self, point: PointId, line: LineId) -> bool {
        self.facts.contains(&Fact::OnLine(point, line))
    }

    fn find_right_triangles(&self) -> Vec<(PointId, PointId, PointId)> {
        use crate::ir::FactType;

        self.facts.facts_of_type(FactType::RightTriangle)
            .iter()
            .filter_map(|f| {
                if let Fact::RightTriangle(p1, p2, p3) = f {
                    Some((*p1, *p2, *p3))
                } else {
                    None
                }
            })
            .collect()
    }

    fn find_perpendicular_pairs(&self) -> Vec<(LineId, LineId)> {
        use crate::ir::FactType;

        self.facts.facts_of_type(FactType::Perpendicular)
            .iter()
            .filter_map(|f| {
                if let Fact::Perpendicular(l1, l2) = f {
                    Some((*l1, *l2))
                } else {
                    None
                }
            })
            .collect()
    }

    fn find_parallel_pairs(&self) -> Vec<(LineId, LineId)> {
        use crate::ir::FactType;

        self.facts.facts_of_type(FactType::Parallel)
            .iter()
            .filter_map(|f| {
                if let Fact::Parallel(l1, l2) = f {
                    Some((*l1, *l2))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Check if a goal is satisfied by the current state
pub fn is_goal_satisfied(state: &GeoState) -> bool {
    state.is_goal_satisfied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Goal, Metadata, FactStore};
    use crate::geometry::Construct;

    #[test]
    fn test_detect_right_triangle() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let p1 = state.add_point("A");
        let p2 = state.add_point("B");
        let p3 = state.add_point("C");

        assert!(!state.detect_right_triangle(p1, p2, p3));

        state.facts.insert(Fact::RightTriangle(p1, p2, p3));
        assert!(state.detect_right_triangle(p1, p2, p3));

        // Should work with different orderings
        assert!(state.detect_right_triangle(p2, p1, p3));
    }

    #[test]
    fn test_perpendicular_detection() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let l1 = state.add_line("L1");
        let l2 = state.add_line("L2");

        assert!(!state.are_perpendicular(l1, l2));

        state.facts.insert(Fact::Perpendicular(l1, l2));
        assert!(state.are_perpendicular(l1, l2));
        assert!(state.are_perpendicular(l2, l1)); // Symmetric
    }

    #[test]
    fn test_parallel_detection() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let l1 = state.add_line("L1");
        let l2 = state.add_line("L2");

        state.facts.insert(Fact::Parallel(l1, l2));
        assert!(state.are_parallel(l1, l2));
        assert!(state.are_parallel(l2, l1));
    }

    #[test]
    fn test_find_right_triangles() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let p1 = state.add_point("A");
        let p2 = state.add_point("B");
        let p3 = state.add_point("C");
        let p4 = state.add_point("D");

        state.facts.insert(Fact::RightTriangle(p1, p2, p3));
        state.facts.insert(Fact::RightTriangle(p2, p3, p4));

        let triangles = state.find_right_triangles();
        assert_eq!(triangles.len(), 2);
    }

    #[test]
    fn test_collinear_detection() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let p1 = state.add_point("A");
        let p2 = state.add_point("B");
        let p3 = state.add_point("C");

        state.facts.insert(Fact::Collinear(p1, p2, p3));

        assert!(state.are_collinear(p1, p2, p3));
        assert!(state.are_collinear(p3, p1, p2)); // Different order
    }

    #[test]
    fn test_goal_satisfaction() {
        let target = Fact::Parallel(LineId(1), LineId(2));
        let goal = Goal::new(vec![target.clone()]);

        let mut facts = FactStore::new();
        facts.insert(target);

        let state = GeoState::new(
            Default::default(),
            facts,
            goal,
        );

        assert!(is_goal_satisfied(&state));
    }
}
