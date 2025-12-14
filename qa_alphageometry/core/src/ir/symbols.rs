//! Symbol interning for geometric objects
//!
//! This module provides type-safe newtype wrappers for geometric object IDs
//! and a symbol table for efficient string interning and deduplication.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

/// Newtype wrapper for point identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PointId(pub u32);

/// Newtype wrapper for line identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LineId(pub u32);

/// Newtype wrapper for circle identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CircleId(pub u32);

/// Newtype wrapper for segment identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SegmentId(pub u32);

/// Newtype wrapper for angle identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AngleId(pub u32);

/// Generic identifier type for any geometric object
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeometricId {
    Point(PointId),
    Line(LineId),
    Circle(CircleId),
    Segment(SegmentId),
    Angle(AngleId),
}

/// Thread-safe symbol table for interning strings to geometric object IDs
///
/// The symbol table maintains bidirectional mappings between string labels
/// and numeric IDs, ensuring each unique label maps to exactly one ID.
/// This provides O(1) lookups and reduces memory overhead through deduplication.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    inner: Arc<RwLock<SymbolTableInner>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SymbolTableInner {
    // String -> ID mappings
    point_map: FxHashMap<String, PointId>,
    line_map: FxHashMap<String, LineId>,
    circle_map: FxHashMap<String, CircleId>,
    segment_map: FxHashMap<String, SegmentId>,
    angle_map: FxHashMap<String, AngleId>,

    // ID -> String mappings (for reverse lookups)
    point_labels: Vec<String>,
    line_labels: Vec<String>,
    circle_labels: Vec<String>,
    segment_labels: Vec<String>,
    angle_labels: Vec<String>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolTable {
    /// Create a new empty symbol table
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(SymbolTableInner {
                point_map: FxHashMap::default(),
                line_map: FxHashMap::default(),
                circle_map: FxHashMap::default(),
                segment_map: FxHashMap::default(),
                angle_map: FxHashMap::default(),
                point_labels: Vec::new(),
                line_labels: Vec::new(),
                circle_labels: Vec::new(),
                segment_labels: Vec::new(),
                angle_labels: Vec::new(),
            })),
        }
    }

    /// Intern a point label, returning its ID (creates new ID if label not seen before)
    pub fn get_or_intern_point(&self, label: &str) -> PointId {
        let mut inner = self.inner.write().unwrap();
        if let Some(&id) = inner.point_map.get(label) {
            return id;
        }
        let id = PointId(inner.point_labels.len() as u32);
        inner.point_map.insert(label.to_string(), id);
        inner.point_labels.push(label.to_string());
        id
    }

    /// Intern a line label, returning its ID
    pub fn get_or_intern_line(&self, label: &str) -> LineId {
        let mut inner = self.inner.write().unwrap();
        if let Some(&id) = inner.line_map.get(label) {
            return id;
        }
        let id = LineId(inner.line_labels.len() as u32);
        inner.line_map.insert(label.to_string(), id);
        inner.line_labels.push(label.to_string());
        id
    }

    /// Intern a circle label, returning its ID
    pub fn get_or_intern_circle(&self, label: &str) -> CircleId {
        let mut inner = self.inner.write().unwrap();
        if let Some(&id) = inner.circle_map.get(label) {
            return id;
        }
        let id = CircleId(inner.circle_labels.len() as u32);
        inner.circle_map.insert(label.to_string(), id);
        inner.circle_labels.push(label.to_string());
        id
    }

    /// Intern a segment label, returning its ID
    pub fn get_or_intern_segment(&self, label: &str) -> SegmentId {
        let mut inner = self.inner.write().unwrap();
        if let Some(&id) = inner.segment_map.get(label) {
            return id;
        }
        let id = SegmentId(inner.segment_labels.len() as u32);
        inner.segment_map.insert(label.to_string(), id);
        inner.segment_labels.push(label.to_string());
        id
    }

    /// Intern an angle label, returning its ID
    pub fn get_or_intern_angle(&self, label: &str) -> AngleId {
        let mut inner = self.inner.write().unwrap();
        if let Some(&id) = inner.angle_map.get(label) {
            return id;
        }
        let id = AngleId(inner.angle_labels.len() as u32);
        inner.angle_map.insert(label.to_string(), id);
        inner.angle_labels.push(label.to_string());
        id
    }

    /// Get the label for a point ID
    pub fn point_label(&self, id: PointId) -> Option<String> {
        let inner = self.inner.read().unwrap();
        inner.point_labels.get(id.0 as usize).cloned()
    }

    /// Get the label for a line ID
    pub fn line_label(&self, id: LineId) -> Option<String> {
        let inner = self.inner.read().unwrap();
        inner.line_labels.get(id.0 as usize).cloned()
    }

    /// Get the label for a circle ID
    pub fn circle_label(&self, id: CircleId) -> Option<String> {
        let inner = self.inner.read().unwrap();
        inner.circle_labels.get(id.0 as usize).cloned()
    }

    /// Get the label for a segment ID
    pub fn segment_label(&self, id: SegmentId) -> Option<String> {
        let inner = self.inner.read().unwrap();
        inner.segment_labels.get(id.0 as usize).cloned()
    }

    /// Get the label for an angle ID
    pub fn angle_label(&self, id: AngleId) -> Option<String> {
        let inner = self.inner.read().unwrap();
        inner.angle_labels.get(id.0 as usize).cloned()
    }

    /// Get the number of interned points
    pub fn num_points(&self) -> usize {
        self.inner.read().unwrap().point_labels.len()
    }

    /// Get the number of interned lines
    pub fn num_lines(&self) -> usize {
        self.inner.read().unwrap().line_labels.len()
    }

    /// Get the number of interned circles
    pub fn num_circles(&self) -> usize {
        self.inner.read().unwrap().circle_labels.len()
    }

    /// Get the number of interned segments
    pub fn num_segments(&self) -> usize {
        self.inner.read().unwrap().segment_labels.len()
    }

    /// Get the number of interned angles
    pub fn num_angles(&self) -> usize {
        self.inner.read().unwrap().angle_labels.len()
    }
}

// Custom Serialize/Deserialize for SymbolTable to handle Arc<RwLock<>>
impl Serialize for SymbolTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let inner = self.inner.read().unwrap();
        inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SymbolTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = SymbolTableInner::deserialize(deserializer)?;
        Ok(SymbolTable {
            inner: Arc::new(RwLock::new(inner)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_interning() {
        let table = SymbolTable::new();
        let a1 = table.get_or_intern_point("A");
        let b = table.get_or_intern_point("B");
        let a2 = table.get_or_intern_point("A");

        assert_eq!(a1, a2, "Same label should return same ID");
        assert_ne!(a1, b, "Different labels should return different IDs");
        assert_eq!(table.num_points(), 2);
    }

    #[test]
    fn test_point_label_lookup() {
        let table = SymbolTable::new();
        let a = table.get_or_intern_point("A");

        assert_eq!(table.point_label(a), Some("A".to_string()));
    }

    #[test]
    fn test_multiple_types() {
        let table = SymbolTable::new();
        let point_a = table.get_or_intern_point("A");
        let line_a = table.get_or_intern_line("A");

        // Same label can be used for different types
        assert_eq!(point_a.0, 0);
        assert_eq!(line_a.0, 0);
        assert_eq!(table.num_points(), 1);
        assert_eq!(table.num_lines(), 1);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let table = SymbolTable::new();
        let table_clone = table.clone();

        let handle = thread::spawn(move || {
            table_clone.get_or_intern_point("X")
        });

        let id1 = table.get_or_intern_point("X");
        let id2 = handle.join().unwrap();

        // Both should get the same ID (one wins the race)
        assert!(id1 == id2 || table.num_points() == 1);
    }

    #[test]
    fn test_all_geometric_types() {
        let table = SymbolTable::new();

        let p = table.get_or_intern_point("P");
        let l = table.get_or_intern_line("L");
        let c = table.get_or_intern_circle("C");
        let s = table.get_or_intern_segment("S");
        let a = table.get_or_intern_angle("A");

        assert_eq!(table.point_label(p), Some("P".to_string()));
        assert_eq!(table.line_label(l), Some("L".to_string()));
        assert_eq!(table.circle_label(c), Some("C".to_string()));
        assert_eq!(table.segment_label(s), Some("S".to_string()));
        assert_eq!(table.angle_label(a), Some("A".to_string()));
    }

    #[test]
    fn test_deduplication() {
        let table = SymbolTable::new();

        for _ in 0..100 {
            table.get_or_intern_point("A");
            table.get_or_intern_point("B");
            table.get_or_intern_point("C");
        }

        assert_eq!(table.num_points(), 3, "Should deduplicate repeated interning");
    }
}
