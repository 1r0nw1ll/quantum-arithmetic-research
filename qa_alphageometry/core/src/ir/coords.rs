//! Coordinate geometry support
//!
//! Optional coordinate tracking for geometric objects

use super::symbols::PointId;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// 2D point coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Distance to another point
    pub fn distance(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Dot product with another point (as vectors from origin)
    pub fn dot(&self, other: &Point2D) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Cross product z-component (for 2D)
    pub fn cross_z(&self, other: &Point2D) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

/// Coordinate store - maps PointIds to coordinates
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoordinateStore {
    coords: FxHashMap<PointId, Point2D>,
}

impl CoordinateStore {
    pub fn new() -> Self {
        Self {
            coords: FxHashMap::default(),
        }
    }

    /// Set coordinates for a point
    pub fn set(&mut self, point: PointId, coords: Point2D) {
        self.coords.insert(point, coords);
    }

    /// Get coordinates for a point
    pub fn get(&self, point: PointId) -> Option<Point2D> {
        self.coords.get(&point).copied()
    }

    /// Check if point has coordinates
    pub fn has(&self, point: PointId) -> bool {
        self.coords.contains_key(&point)
    }

    /// Remove coordinates for a point
    pub fn remove(&mut self, point: PointId) {
        self.coords.remove(&point);
    }

    /// Number of points with coordinates
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

/// Geometric computations using coordinates
pub mod ops {
    use super::*;

    /// Check if three points are collinear (using cross product)
    pub fn are_collinear(p1: Point2D, p2: Point2D, p3: Point2D, epsilon: f64) -> bool {
        // Vector from p1 to p2
        let v1 = Point2D::new(p2.x - p1.x, p2.y - p1.y);
        // Vector from p1 to p3
        let v2 = Point2D::new(p3.x - p1.x, p3.y - p1.y);

        // Cross product z-component - should be 0 for collinear
        v1.cross_z(&v2).abs() < epsilon
    }

    /// Check if two lines are perpendicular (using dot product)
    pub fn are_perpendicular(
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
        p4: Point2D,
        epsilon: f64,
    ) -> bool {
        // Direction vector of line 1 (p1->p2)
        let v1 = Point2D::new(p2.x - p1.x, p2.y - p1.y);
        // Direction vector of line 2 (p3->p4)
        let v2 = Point2D::new(p4.x - p3.x, p4.y - p3.y);

        // Dot product should be 0 for perpendicular
        v1.dot(&v2).abs() < epsilon
    }

    /// Check if two lines are parallel (using cross product)
    pub fn are_parallel(
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
        p4: Point2D,
        epsilon: f64,
    ) -> bool {
        // Direction vector of line 1
        let v1 = Point2D::new(p2.x - p1.x, p2.y - p1.y);
        // Direction vector of line 2
        let v2 = Point2D::new(p4.x - p3.x, p4.y - p3.y);

        // Cross product z-component should be 0 for parallel
        v1.cross_z(&v2).abs() < epsilon
    }

    /// Check if angle at p2 formed by p1-p2-p3 is a right angle
    pub fn is_right_angle(p1: Point2D, p2: Point2D, p3: Point2D, epsilon: f64) -> bool {
        // Vector from p2 to p1
        let v1 = Point2D::new(p1.x - p2.x, p1.y - p2.y);
        // Vector from p2 to p3
        let v2 = Point2D::new(p3.x - p2.x, p3.y - p2.y);

        // Dot product should be 0 for right angle
        v1.dot(&v2).abs() < epsilon
    }

    /// Check if triangle with given side lengths is a right triangle
    pub fn is_right_triangle(a: f64, b: f64, c: f64, epsilon: f64) -> bool {
        // Sort sides
        let mut sides = [a, b, c];
        sides.sort_by(|x, y| x.partial_cmp(y).unwrap());

        // Check Pythagorean theorem: a² + b² = c²
        let sum_of_squares = sides[0] * sides[0] + sides[1] * sides[1];
        let hypotenuse_squared = sides[2] * sides[2];

        (sum_of_squares - hypotenuse_squared).abs() < epsilon
    }

    /// Calculate angle between three points (in radians)
    pub fn angle(p1: Point2D, vertex: Point2D, p3: Point2D) -> f64 {
        let v1 = Point2D::new(p1.x - vertex.x, p1.y - vertex.y);
        let v2 = Point2D::new(p3.x - vertex.x, p3.y - vertex.y);

        let dot = v1.dot(&v2);
        let mag1 = (v1.x * v1.x + v1.y * v1.y).sqrt();
        let mag2 = (v2.x * v2.x + v2.y * v2.y).sqrt();

        if mag1 == 0.0 || mag2 == 0.0 {
            return 0.0;
        }

        (dot / (mag1 * mag2)).acos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);

        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_collinearity() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 1.0);
        let p3 = Point2D::new(2.0, 2.0);

        assert!(ops::are_collinear(p1, p2, p3, 1e-10));
    }

    #[test]
    fn test_perpendicular() {
        // Line 1: horizontal (0,0) to (1,0)
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 0.0);

        // Line 2: vertical (0,0) to (0,1)
        let p3 = Point2D::new(0.0, 0.0);
        let p4 = Point2D::new(0.0, 1.0);

        assert!(ops::are_perpendicular(p1, p2, p3, p4, 1e-10));
    }

    #[test]
    fn test_parallel() {
        // Line 1: (0,0) to (1,1)
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 1.0);

        // Line 2: (0,1) to (1,2) - parallel to line 1
        let p3 = Point2D::new(0.0, 1.0);
        let p4 = Point2D::new(1.0, 2.0);

        assert!(ops::are_parallel(p1, p2, p3, p4, 1e-10));
    }

    #[test]
    fn test_right_angle() {
        let p1 = Point2D::new(1.0, 0.0);
        let vertex = Point2D::new(0.0, 0.0);
        let p3 = Point2D::new(0.0, 1.0);

        assert!(ops::is_right_angle(p1, vertex, p3, 1e-10));
    }

    #[test]
    fn test_right_triangle() {
        // 3-4-5 triangle
        assert!(ops::is_right_triangle(3.0, 4.0, 5.0, 1e-10));

        // 5-12-13 triangle
        assert!(ops::is_right_triangle(5.0, 12.0, 13.0, 1e-10));

        // Not a right triangle
        assert!(!ops::is_right_triangle(2.0, 3.0, 4.0, 1e-10));
    }

    #[test]
    fn test_coordinate_store() {
        let mut store = CoordinateStore::new();

        let p1 = PointId(1);
        let coords = Point2D::new(1.0, 2.0);

        store.set(p1, coords);

        assert!(store.has(p1));
        assert_eq!(store.get(p1), Some(coords));
        assert_eq!(store.len(), 1);

        store.remove(p1);
        assert!(!store.has(p1));
        assert_eq!(store.len(), 0);
    }
}
