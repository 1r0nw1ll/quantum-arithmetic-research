//! Atomic geometric facts and fact storage
//!
//! This module defines geometric facts as algebraic predicates over geometric objects
//! and provides efficient storage and indexing for fact collections.

use super::symbols::{AngleId, CircleId, LineId, PointId, SegmentId};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

/// Unique identifier for a proof step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProofStepId(pub u32);

impl fmt::Display for ProofStepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Step{}", self.0)
    }
}

/// Atomic geometric fact - a single predicate about geometric objects
///
/// Facts are normalized to canonical form for deduplication:
/// - Symmetric predicates (e.g., Parallel) sort their arguments
/// - Collinear points are sorted by ID
/// - Angle equality is order-independent
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Fact {
    /// Three points lie on the same line
    Collinear(PointId, PointId, PointId),

    /// Two lines are parallel
    Parallel(LineId, LineId),

    /// Two lines are perpendicular
    Perpendicular(LineId, LineId),

    /// A point lies on a circle
    OnCircle(PointId, CircleId),

    /// Two segments have equal length
    EqualLength(SegmentId, SegmentId),

    /// Two angles have equal measure
    EqualAngle(AngleId, AngleId),

    /// A point is the midpoint of a segment
    Midpoint(PointId, SegmentId),

    /// Three points form a right triangle (vertex at first point)
    RightTriangle(PointId, PointId, PointId),

    /// A point lies on a line
    OnLine(PointId, LineId),

    /// Two lines are coincident (same line)
    CoincidentLines(LineId, LineId),

    /// Two circles are concentric
    ConcentricCircles(CircleId, CircleId),

    /// Four points are concyclic (lie on the same circle)
    Concyclic(PointId, PointId, PointId, PointId),

    /// A line is tangent to a circle at a point
    Tangent(LineId, CircleId, PointId),

    /// Three points form an isosceles triangle (two equal sides from first point)
    IsoscelesTriangle(PointId, PointId, PointId),

    /// Three points form an equilateral triangle
    EquilateralTriangle(PointId, PointId, PointId),

    /// Four points form a parallelogram
    Parallelogram(PointId, PointId, PointId, PointId),

    /// Four points form a rectangle
    Rectangle(PointId, PointId, PointId, PointId),

    /// Four points form a square
    Square(PointId, PointId, PointId, PointId),

    /// An angle is a right angle (90 degrees)
    RightAngle(AngleId),

    /// Two segments are perpendicular
    PerpendicularSegments(SegmentId, SegmentId),

    /// A line bisects a segment
    Bisects(LineId, SegmentId),

    /// A line bisects an angle
    AngleBisector(LineId, AngleId),

    /// Three segments form a Pythagorean triple
    PythagoreanTriple(SegmentId, SegmentId, SegmentId),
}

impl Fact {
    /// Normalize the fact to canonical form for deduplication
    ///
    /// Symmetric predicates have their arguments sorted to ensure
    /// that equivalent facts hash to the same value.
    pub fn normalize(self) -> Self {
        match self {
            Fact::Parallel(l1, l2) => {
                if l1 <= l2 {
                    Fact::Parallel(l1, l2)
                } else {
                    Fact::Parallel(l2, l1)
                }
            }
            Fact::Perpendicular(l1, l2) => {
                if l1 <= l2 {
                    Fact::Perpendicular(l1, l2)
                } else {
                    Fact::Perpendicular(l2, l1)
                }
            }
            Fact::EqualLength(s1, s2) => {
                if s1 <= s2 {
                    Fact::EqualLength(s1, s2)
                } else {
                    Fact::EqualLength(s2, s1)
                }
            }
            Fact::EqualAngle(a1, a2) => {
                if a1 <= a2 {
                    Fact::EqualAngle(a1, a2)
                } else {
                    Fact::EqualAngle(a2, a1)
                }
            }
            Fact::CoincidentLines(l1, l2) => {
                if l1 <= l2 {
                    Fact::CoincidentLines(l1, l2)
                } else {
                    Fact::CoincidentLines(l2, l1)
                }
            }
            Fact::ConcentricCircles(c1, c2) => {
                if c1 <= c2 {
                    Fact::ConcentricCircles(c1, c2)
                } else {
                    Fact::ConcentricCircles(c2, c1)
                }
            }
            Fact::PerpendicularSegments(s1, s2) => {
                if s1 <= s2 {
                    Fact::PerpendicularSegments(s1, s2)
                } else {
                    Fact::PerpendicularSegments(s2, s1)
                }
            }
            Fact::Collinear(p1, p2, p3) => {
                let mut points = [p1, p2, p3];
                points.sort();
                Fact::Collinear(points[0], points[1], points[2])
            }
            Fact::Concyclic(p1, p2, p3, p4) => {
                let mut points = [p1, p2, p3, p4];
                points.sort();
                Fact::Concyclic(points[0], points[1], points[2], points[3])
            }
            // Non-symmetric facts remain unchanged
            other => other,
        }
    }

    /// Get the type of this fact for indexing
    pub fn fact_type(&self) -> FactType {
        match self {
            Fact::Collinear(_, _, _) => FactType::Collinear,
            Fact::Parallel(_, _) => FactType::Parallel,
            Fact::Perpendicular(_, _) => FactType::Perpendicular,
            Fact::OnCircle(_, _) => FactType::OnCircle,
            Fact::EqualLength(_, _) => FactType::EqualLength,
            Fact::EqualAngle(_, _) => FactType::EqualAngle,
            Fact::Midpoint(_, _) => FactType::Midpoint,
            Fact::RightTriangle(_, _, _) => FactType::RightTriangle,
            Fact::OnLine(_, _) => FactType::OnLine,
            Fact::CoincidentLines(_, _) => FactType::CoincidentLines,
            Fact::ConcentricCircles(_, _) => FactType::ConcentricCircles,
            Fact::Concyclic(_, _, _, _) => FactType::Concyclic,
            Fact::Tangent(_, _, _) => FactType::Tangent,
            Fact::IsoscelesTriangle(_, _, _) => FactType::IsoscelesTriangle,
            Fact::EquilateralTriangle(_, _, _) => FactType::EquilateralTriangle,
            Fact::Parallelogram(_, _, _, _) => FactType::Parallelogram,
            Fact::Rectangle(_, _, _, _) => FactType::Rectangle,
            Fact::Square(_, _, _, _) => FactType::Square,
            Fact::RightAngle(_) => FactType::RightAngle,
            Fact::PerpendicularSegments(_, _) => FactType::PerpendicularSegments,
            Fact::Bisects(_, _) => FactType::Bisects,
            Fact::AngleBisector(_, _) => FactType::AngleBisector,
            Fact::PythagoreanTriple(_, _, _) => FactType::PythagoreanTriple,
        }
    }
}

/// Enumeration of fact types for indexing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactType {
    Collinear,
    Parallel,
    Perpendicular,
    OnCircle,
    EqualLength,
    EqualAngle,
    Midpoint,
    RightTriangle,
    OnLine,
    CoincidentLines,
    ConcentricCircles,
    Concyclic,
    Tangent,
    IsoscelesTriangle,
    EquilateralTriangle,
    Parallelogram,
    Rectangle,
    Square,
    RightAngle,
    PerpendicularSegments,
    Bisects,
    AngleBisector,
    PythagoreanTriple,
}

/// Storage for geometric facts with deduplication and type-based indexing
///
/// Facts are automatically normalized and deduplicated. The store maintains
/// an index by fact type for efficient querying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactStore {
    /// Deduplicated set of all facts
    facts: HashSet<Fact>,

    /// Index mapping fact types to facts of that type
    type_index: FxHashMap<FactType, Vec<Fact>>,

    /// Provenance tracking: which proof step introduced each fact
    provenance: FxHashMap<Fact, ProofStepId>,
}

impl Default for FactStore {
    fn default() -> Self {
        Self::new()
    }
}

impl FactStore {
    /// Create a new empty fact store
    pub fn new() -> Self {
        Self {
            facts: HashSet::new(),
            type_index: FxHashMap::default(),
            provenance: FxHashMap::default(),
        }
    }

    /// Insert a fact into the store
    ///
    /// Returns true if the fact was newly inserted, false if it already existed.
    /// The fact is automatically normalized before insertion.
    pub fn insert(&mut self, fact: Fact) -> bool {
        let fact = fact.normalize();
        let is_new = self.facts.insert(fact.clone());

        if is_new {
            let fact_type = fact.fact_type();
            self.type_index
                .entry(fact_type)
                .or_insert_with(Vec::new)
                .push(fact);
        }

        is_new
    }

    /// Insert a fact with provenance information
    pub fn insert_with_provenance(&mut self, fact: Fact, step_id: ProofStepId) -> bool {
        let fact = fact.normalize();
        let is_new = self.insert(fact.clone());

        if is_new {
            self.provenance.insert(fact, step_id);
        }

        is_new
    }

    /// Check if the store contains a fact
    pub fn contains(&self, fact: &Fact) -> bool {
        let normalized = fact.clone().normalize();
        self.facts.contains(&normalized)
    }

    /// Get all facts of a specific type
    pub fn facts_of_type(&self, fact_type: FactType) -> &[Fact] {
        self.type_index
            .get(&fact_type)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all facts in the store
    pub fn all_facts(&self) -> impl Iterator<Item = &Fact> {
        self.facts.iter()
    }

    /// Get the total number of facts
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Get the proof step that introduced a fact, if available
    pub fn provenance(&self, fact: &Fact) -> Option<ProofStepId> {
        let normalized = fact.clone().normalize();
        self.provenance.get(&normalized).copied()
    }

    /// Merge another fact store into this one
    pub fn merge(&mut self, other: &FactStore) {
        for fact in other.all_facts() {
            if let Some(step_id) = other.provenance(fact) {
                self.insert_with_provenance(fact.clone(), step_id);
            } else {
                self.insert(fact.clone());
            }
        }
    }

    /// Clear all facts from the store
    pub fn clear(&mut self) {
        self.facts.clear();
        self.type_index.clear();
        self.provenance.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_normalization() {
        let l1 = LineId(1);
        let l2 = LineId(2);

        let f1 = Fact::Parallel(l1, l2).normalize();
        let f2 = Fact::Parallel(l2, l1).normalize();

        assert_eq!(f1, f2, "Parallel facts should normalize to same form");
    }

    #[test]
    fn test_collinear_normalization() {
        let p1 = PointId(1);
        let p2 = PointId(2);
        let p3 = PointId(3);

        let f1 = Fact::Collinear(p1, p2, p3).normalize();
        let f2 = Fact::Collinear(p3, p1, p2).normalize();
        let f3 = Fact::Collinear(p2, p3, p1).normalize();

        assert_eq!(f1, f2);
        assert_eq!(f2, f3);
    }

    #[test]
    fn test_fact_insertion() {
        let mut store = FactStore::new();
        let l1 = LineId(1);
        let l2 = LineId(2);

        let fact = Fact::Parallel(l1, l2);
        assert!(store.insert(fact.clone()), "First insertion should be new");
        assert!(!store.insert(fact.clone()), "Second insertion should be duplicate");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_fact_deduplication() {
        let mut store = FactStore::new();
        let l1 = LineId(1);
        let l2 = LineId(2);

        store.insert(Fact::Parallel(l1, l2));
        store.insert(Fact::Parallel(l2, l1)); // Should normalize to same fact

        assert_eq!(store.len(), 1, "Should deduplicate symmetric facts");
    }

    #[test]
    fn test_type_indexing() {
        let mut store = FactStore::new();
        let l1 = LineId(1);
        let l2 = LineId(2);
        let l3 = LineId(3);

        store.insert(Fact::Parallel(l1, l2));
        store.insert(Fact::Parallel(l2, l3));
        store.insert(Fact::Perpendicular(l1, l3));

        let parallel_facts = store.facts_of_type(FactType::Parallel);
        assert_eq!(parallel_facts.len(), 2);

        let perp_facts = store.facts_of_type(FactType::Perpendicular);
        assert_eq!(perp_facts.len(), 1);
    }

    #[test]
    fn test_provenance_tracking() {
        let mut store = FactStore::new();
        let fact = Fact::Parallel(LineId(1), LineId(2));
        let step = ProofStepId(42);

        store.insert_with_provenance(fact.clone(), step);

        assert_eq!(store.provenance(&fact), Some(step));
    }

    #[test]
    fn test_fact_contains() {
        let mut store = FactStore::new();
        let l1 = LineId(1);
        let l2 = LineId(2);

        store.insert(Fact::Parallel(l1, l2));

        assert!(store.contains(&Fact::Parallel(l1, l2)));
        assert!(store.contains(&Fact::Parallel(l2, l1))); // Normalized form
        assert!(!store.contains(&Fact::Perpendicular(l1, l2)));
    }

    #[test]
    fn test_merge_stores() {
        let mut store1 = FactStore::new();
        let mut store2 = FactStore::new();

        store1.insert(Fact::Parallel(LineId(1), LineId(2)));
        store2.insert(Fact::Perpendicular(LineId(3), LineId(4)));
        store2.insert(Fact::Parallel(LineId(1), LineId(2))); // Duplicate

        store1.merge(&store2);

        assert_eq!(store1.len(), 2, "Should merge without duplicates");
    }

    #[test]
    fn test_all_fact_types() {
        let mut store = FactStore::new();

        store.insert(Fact::Collinear(PointId(1), PointId(2), PointId(3)));
        store.insert(Fact::OnCircle(PointId(1), CircleId(1)));
        store.insert(Fact::Midpoint(PointId(1), SegmentId(1)));
        store.insert(Fact::RightAngle(AngleId(1)));
        store.insert(Fact::Square(PointId(1), PointId(2), PointId(3), PointId(4)));

        assert_eq!(store.len(), 5);
        assert!(store.contains(&Fact::RightAngle(AngleId(1))));
    }
}
