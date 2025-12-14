//! QA Tuple: Core quantum arithmetic structure
//!
//! CRITICAL: This module contains CASE-SENSITIVE invariants that must be
//! maintained exactly as specified. DO NOT modify formulas without verification.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Quantum Arithmetic tuple (b, e, d, a) with strict invariants
///
/// Invariants (ENFORCED):
/// - b + e = d
/// - e + d = a
///
/// These are Pythagorean-like tuples from modular arithmetic
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QATuple {
    pub b: f64,
    pub e: f64,
    pub d: f64,
    pub a: f64,
}

impl QATuple {
    /// Create a QA tuple from base (b) and exponent (e)
    ///
    /// Computes d = b + e and a = e + d, enforcing structural invariants.
    ///
    /// # Arguments
    /// * `b` - Base component
    /// * `e` - Exponent component
    ///
    /// # Returns
    /// QATuple with derived d and a values
    pub fn new(b: f64, e: f64) -> Self {
        let d = b + e;
        let a = e + d;
        QATuple { b, e, d, a }
    }

    /// Create from all four components with validation
    ///
    /// # Arguments
    /// * `b`, `e`, `d`, `a` - Tuple components
    ///
    /// # Returns
    /// * `Some(QATuple)` if invariants satisfied (within epsilon)
    /// * `None` if invariants violated
    pub fn from_components(b: f64, e: f64, d: f64, a: f64) -> Option<Self> {
        const EPSILON: f64 = 1e-6;

        // Validate: b + e = d
        if (b + e - d).abs() > EPSILON {
            return None;
        }

        // Validate: e + d = a
        if (e + d - a).abs() > EPSILON {
            return None;
        }

        Some(QATuple { b, e, d, a })
    }

    /// CASE-SENSITIVE: Base leg C = 2*e*d (NOT e*d)
    ///
    /// This is the fundamental base projection. DO NOT change to e*d.
    #[inline]
    pub fn C(&self) -> f64 {
        2.0 * self.e * self.d
    }

    /// Altitude invariant F = b*a
    #[inline]
    pub fn F(&self) -> f64 {
        self.b * self.a
    }

    /// Geometric mean square G = e² + d²
    #[inline]
    pub fn G(&self) -> f64 {
        self.e * self.e + self.d * self.d
    }

    /// Coupling invariant J = b*d
    #[inline]
    pub fn J(&self) -> f64 {
        self.b * self.d
    }

    /// Cross term X = e*d
    #[inline]
    pub fn X(&self) -> f64 {
        self.e * self.d
    }

    /// Conjugate invariant K = d*a
    #[inline]
    pub fn K(&self) -> f64 {
        self.d * self.a
    }

    /// Harmonic index I = C - F = 2*e*d - b*a
    ///
    /// This is the fundamental discriminant for family classification
    #[inline]
    pub fn I(&self) -> f64 {
        self.C() - self.F()
    }

    /// CASE-SENSITIVE: Quantum ellipse major axis = 2D = 2(d²), NOT 2d
    ///
    /// This is 2 times the square of d, not 2 times d.
    /// DO NOT change to 2.0 * self.d
    #[inline]
    pub fn quantum_ellipse_major_axis(&self) -> f64 {
        2.0 * self.d * self.d
    }

    /// Quantum ellipse minor axis = 2B = 2(b²)
    #[inline]
    pub fn quantum_ellipse_minor_axis(&self) -> f64 {
        2.0 * self.b * self.b
    }

    /// Check if tuple is Primitive: gcd(b, e, d, a) = 1
    ///
    /// For integer-valued tuples, primitive means no common factor.
    /// For geometric extraction, this is approximate.
    pub fn is_primitive(&self) -> bool {
        use num_integer::Integer;

        // Round to nearest integer for GCD computation
        let b_int = self.b.round() as i64;
        let e_int = self.e.round() as i64;
        let d_int = self.d.round() as i64;
        let a_int = self.a.round() as i64;

        let gcd_be = b_int.gcd(&e_int);
        let gcd_bed = gcd_be.gcd(&d_int);
        let gcd_all = gcd_bed.gcd(&a_int);

        gcd_all.abs() == 1
    }

    /// Check if tuple is Female: b even, e odd
    ///
    /// This classification is critical for mod-24 phase structure
    pub fn is_female(&self) -> bool {
        let b_int = self.b.round() as i64;
        let e_int = self.e.round() as i64;

        b_int % 2 == 0 && e_int % 2 == 1
    }

    /// Check if tuple is Male: b odd, e even
    pub fn is_male(&self) -> bool {
        let b_int = self.b.round() as i64;
        let e_int = self.e.round() as i64;

        b_int % 2 == 1 && e_int % 2 == 0
    }

    /// Check if tuple is in Fermat family: |C - F| = 1
    ///
    /// Fermat tuples have minimal harmonic index and correspond to
    /// shortest proof paths in geometric reasoning.
    pub fn is_fermat(&self) -> bool {
        (self.I()).abs() < 1.5  // Allow 1 or -1 with tolerance
    }

    /// Compute mod-24 phase: (b + 2*e) % 24
    ///
    /// This determines constraint satisfaction structure in symbolic search
    pub fn mod24_phase(&self) -> i64 {
        let b_int = self.b.round() as i64;
        let e_int = self.e.round() as i64;

        (b_int + 2 * e_int).rem_euclid(24)
    }

    /// Classify tuple into family: (Primitive, Parity, Fermat)
    ///
    /// Returns a triple of boolean flags for efficient indexing
    pub fn classify(&self) -> TupleClass {
        TupleClass {
            primitive: self.is_primitive(),
            female: self.is_female(),
            male: self.is_male(),
            fermat: self.is_fermat(),
            mod24: self.mod24_phase(),
        }
    }

    /// Compute all six invariants as array: [C, F, G, J, X, K]
    ///
    /// This is the canonical feature vector for ML models
    pub fn invariants_array(&self) -> [f64; 6] {
        [self.C(), self.F(), self.G(), self.J(), self.X(), self.K()]
    }

    /// Geometric complexity score for proof search prioritization
    ///
    /// Lower score = simpler tuple = prefer in search
    pub fn complexity_score(&self) -> f64 {
        let class = self.classify();

        let mut score = 0.0;

        // Primitive tuples are simpler
        if !class.primitive {
            score += 10.0;
        }

        // Fermat family has minimal harmonic index
        if class.fermat {
            score -= 5.0;
        }

        // Add magnitude penalty
        score += (self.d.abs() + self.a.abs()) * 0.01;

        score
    }
}

/// Tuple family classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TupleClass {
    pub primitive: bool,
    pub female: bool,
    pub male: bool,
    pub fermat: bool,
    pub mod24: i64,
}

impl fmt::Display for QATuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QA({:.2}, {:.2}, {:.2}, {:.2}) [I={:.2}, mod24={}]",
            self.b,
            self.e,
            self.d,
            self.a,
            self.I(),
            self.mod24_phase()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariants() {
        let qa = QATuple::new(3.0, 4.0);

        // Check structural invariants
        assert_eq!(qa.d, 7.0);
        assert_eq!(qa.a, 11.0);

        // Check b + e = d
        assert!((qa.b + qa.e - qa.d).abs() < 1e-10);

        // Check e + d = a
        assert!((qa.e + qa.d - qa.a).abs() < 1e-10);
    }

    #[test]
    fn test_case_sensitive_formulas() {
        let qa = QATuple::new(3.0, 4.0);

        // CRITICAL: C = 2*e*d, NOT e*d
        assert_eq!(qa.C(), 2.0 * 4.0 * 7.0);
        assert_eq!(qa.C(), 56.0);

        // CRITICAL: Major axis = 2*d², NOT 2*d
        assert_eq!(qa.quantum_ellipse_major_axis(), 2.0 * 7.0 * 7.0);
        assert_eq!(qa.quantum_ellipse_major_axis(), 98.0);
        assert_ne!(qa.quantum_ellipse_major_axis(), 14.0); // Would be 2*d
    }

    #[test]
    fn test_pythagorean_triple() {
        // Classic (3,4,5) Pythagorean triple maps to QA
        let qa = QATuple::new(3.0, 4.0);

        assert_eq!(qa.F(), 3.0 * 11.0); // 33
        assert_eq!(qa.G(), 16.0 + 49.0); // 65
        assert_eq!(qa.I(), qa.C() - qa.F()); // 56 - 33 = 23
    }

    #[test]
    fn test_fermat_family() {
        // Fermat tuples have |I| = 1
        let qa1 = QATuple::new(1.0, 2.0);

        // C = 2*2*3 = 12
        // F = 1*5 = 5
        // I = 12 - 5 = 7 (NOT Fermat)
        assert!(!qa1.is_fermat());

        // To be Fermat, need |2*e*d - b*a| = 1
        // This requires specific integer solutions
    }

    #[test]
    fn test_classification() {
        let qa = QATuple::new(3.0, 4.0);
        let class = qa.classify();

        // (3,4,7,11) - all coprime
        assert!(class.primitive);

        // b=3 odd, e=4 even → Male
        assert!(class.male);
        assert!(!class.female);

        // mod24 = (3 + 2*4) % 24 = 11
        assert_eq!(class.mod24, 11);
    }

    #[test]
    fn test_female_classification() {
        let qa = QATuple::new(4.0, 3.0);

        // b=4 even, e=3 odd → Female
        assert!(qa.is_female());
        assert!(!qa.is_male());
    }

    #[test]
    fn test_validation() {
        // Valid tuple
        let qa = QATuple::from_components(3.0, 4.0, 7.0, 11.0);
        assert!(qa.is_some());

        // Invalid: d ≠ b + e
        let bad1 = QATuple::from_components(3.0, 4.0, 8.0, 11.0);
        assert!(bad1.is_none());

        // Invalid: a ≠ e + d
        let bad2 = QATuple::from_components(3.0, 4.0, 7.0, 12.0);
        assert!(bad2.is_none());
    }

    #[test]
    fn test_invariants_array() {
        let qa = QATuple::new(3.0, 4.0);
        let inv = qa.invariants_array();

        assert_eq!(inv[0], qa.C()); // 56.0
        assert_eq!(inv[1], qa.F()); // 33.0
        assert_eq!(inv[2], qa.G()); // 65.0
        assert_eq!(inv[3], qa.J()); // 21.0
        assert_eq!(inv[4], qa.X()); // 28.0
        assert_eq!(inv[5], qa.K()); // 77.0
    }
}
