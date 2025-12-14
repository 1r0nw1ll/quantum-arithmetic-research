//! QA feature extraction from geometric states
//!
//! This is the BRIDGE between pure geometry and QA - the only module that connects them.
//! Extracts QA tuples from geometric configurations and aggregates into scalar features
//! for search guidance.

use crate::ir::GeoState;
use crate::geometry::Detect;
use crate::qa::QATuple;

/// QA features extracted from a geometric state
///
/// These are SOFT aggregations, not hard classifications.
/// Represents a posterior distribution over possible QA interpretations.
#[derive(Debug, Clone)]
pub struct QAFeatures {
    /// Mass of primitive tuples (0.0 to 1.0)
    pub primitive_mass: f64,

    /// Mass of female tuples (0.0 to 1.0)
    pub female_mass: f64,

    /// Mass of Fermat family tuples (0.0 to 1.0)
    pub fermat_mass: f64,

    /// Entropy of mod-24 phase distribution (0.0 to ~3.2)
    pub phase_entropy: f64,

    /// Mean of J+K invariants
    pub mean_jk: f64,

    /// Mean of |C-F| (harmonic index)
    pub mean_harmonic_index: f64,

    /// Number of candidate tuples extracted
    pub num_candidates: usize,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

impl Default for QAFeatures {
    fn default() -> Self {
        Self {
            primitive_mass: 0.0,
            female_mass: 0.0,
            fermat_mass: 0.0,
            phase_entropy: 0.0,
            mean_jk: 0.0,
            mean_harmonic_index: 0.0,
            num_candidates: 0,
            confidence: 0.0,
        }
    }
}

/// Extract QA features from a geometric state
///
/// This is a SOFT extraction - we maintain a distribution over possible
/// interpretations rather than committing to a single QA tuple.
pub fn extract_qa_features(state: &GeoState) -> QAFeatures {
    // Find all right triangles in the state
    let right_triangles = state.find_right_triangles();

    if right_triangles.is_empty() {
        return QAFeatures::default();
    }

    // For each right triangle, infer candidate QA tuples
    let mut candidates = Vec::new();

    for (p1, p2, p3) in right_triangles {
        // In a right triangle, we can extract multiple candidate tuples
        // depending on which side we consider as the base/height

        // For now, create simplified candidate tuples
        // TODO: Add proper side length extraction when we have coordinate geometry

        // Create dummy candidates for demonstration
        // In real implementation, these would come from side length ratios
        let candidate1 = QATuple::new(3.0, 4.0); // Classic (3,4,5) Pythagorean triple
        let candidate2 = QATuple::new(5.0, 12.0); // (5,12,13)
        let candidate3 = QATuple::new(8.0, 15.0); // (8,15,17)

        candidates.push(candidate1);
        candidates.push(candidate2);
        candidates.push(candidate3);
    }

    if candidates.is_empty() {
        return QAFeatures::default();
    }

    // Aggregate features from candidates
    aggregate_features(&candidates)
}

/// Aggregate scalar features from candidate QA tuples
///
/// Creates a SOFT posterior distribution rather than hard classification.
fn aggregate_features(candidates: &[QATuple]) -> QAFeatures {
    let n = candidates.len() as f64;

    // Count primitives
    let primitive_count = candidates.iter()
        .filter(|t| t.is_primitive())
        .count() as f64;

    // Count female tuples
    let female_count = candidates.iter()
        .filter(|t| t.is_female())
        .count() as f64;

    // Count Fermat family
    let fermat_count = candidates.iter()
        .filter(|t| t.is_fermat())
        .count() as f64;

    // Collect mod-24 phases
    let phases: Vec<i64> = candidates.iter()
        .map(|t| t.mod24_phase())
        .collect();

    // Compute phase entropy: H = -Σ p_i log(p_i)
    let phase_entropy = compute_entropy(&phases);

    // Compute mean J+K
    let mean_jk = candidates.iter()
        .map(|t| t.J() + t.K())
        .sum::<f64>() / n;

    // Compute mean |C-F| (harmonic index)
    let mean_harmonic_index = candidates.iter()
        .map(|t| t.I().abs())
        .sum::<f64>() / n;

    // Confidence based on number of candidates and consistency
    let confidence = if n > 0.0 {
        (n / (n + 5.0)) * 0.9 + 0.1 // Saturates toward 1.0 with more candidates
    } else {
        0.0
    };

    QAFeatures {
        primitive_mass: primitive_count / n,
        female_mass: female_count / n,
        fermat_mass: fermat_count / n,
        phase_entropy,
        mean_jk,
        mean_harmonic_index,
        num_candidates: candidates.len(),
        confidence,
    }
}

/// Compute Shannon entropy of a discrete distribution
fn compute_entropy(values: &[i64]) -> f64 {
    use std::collections::HashMap;

    if values.is_empty() {
        return 0.0;
    }

    // Count frequencies
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v).or_insert(0) += 1;
    }

    let n = values.len() as f64;

    // Compute entropy: H = -Σ p_i log(p_i)
    let mut entropy = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Compute a scalar QA prior score for search guidance
///
/// Higher score = state has more harmonic structure
/// Range: approximately 0.0 to 10.0
pub fn compute_qa_prior(features: &QAFeatures) -> f64 {
    if features.num_candidates == 0 {
        return 0.0;
    }

    let mut score = 0.0;

    // Primitive tuples are simpler (positive contribution)
    score += features.primitive_mass * 2.0;

    // Fermat family indicates minimal proof paths (strong positive)
    score += features.fermat_mass * 3.0;

    // Low phase entropy = more structured (positive)
    let max_entropy = 3.2; // log2(24) ≈ 4.58, but practical max is lower
    let entropy_score = (max_entropy - features.phase_entropy).max(0.0);
    score += entropy_score * 1.5;

    // Low harmonic index = closer to special families (positive)
    let harmonic_bonus = if features.mean_harmonic_index < 2.0 {
        2.0 - features.mean_harmonic_index
    } else {
        0.0
    };
    score += harmonic_bonus;

    // Weight by confidence
    score * features.confidence
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Goal, FactStore, Fact, PointId};
    use crate::geometry::Construct;

    #[test]
    fn test_extract_no_triangles() {
        let state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let features = extract_qa_features(&state);

        assert_eq!(features.num_candidates, 0);
        assert_eq!(features.confidence, 0.0);
    }

    #[test]
    fn test_extract_with_right_triangle() {
        let mut state = GeoState::new(
            Default::default(),
            Default::default(),
            Goal::new(vec![]),
        );

        let p1 = state.add_point("A");
        let p2 = state.add_point("B");
        let p3 = state.add_point("C");

        // Add right triangle fact
        state.facts.insert(Fact::RightTriangle(p1, p2, p3));

        let features = extract_qa_features(&state);

        // Should have extracted candidates
        assert!(features.num_candidates > 0);
        assert!(features.confidence > 0.0);
    }

    #[test]
    fn test_compute_entropy() {
        // Uniform distribution: high entropy
        let uniform = vec![1, 2, 3, 4, 5, 6];
        let h1 = compute_entropy(&uniform);
        assert!(h1 > 2.0); // log2(6) ≈ 2.58

        // Concentrated distribution: low entropy
        let concentrated = vec![1, 1, 1, 1, 1, 2];
        let h2 = compute_entropy(&concentrated);
        assert!(h2 < h1);

        // Single value: zero entropy
        let single = vec![5, 5, 5, 5];
        let h3 = compute_entropy(&single);
        assert!((h3 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_features() {
        let candidates = vec![
            QATuple::new(3.0, 4.0),   // Primitive, male
            QATuple::new(5.0, 12.0),  // Primitive, male
            QATuple::new(8.0, 15.0),  // Primitive, male
        ];

        let features = aggregate_features(&candidates);

        assert_eq!(features.primitive_mass, 1.0); // All primitive
        assert!(features.confidence > 0.0);
        assert!(features.mean_jk > 0.0);
    }

    #[test]
    fn test_qa_prior_score() {
        // High-quality features
        let good_features = QAFeatures {
            primitive_mass: 1.0,
            female_mass: 0.5,
            fermat_mass: 0.8,
            phase_entropy: 0.5,
            mean_jk: 100.0,
            mean_harmonic_index: 0.5,
            num_candidates: 5,
            confidence: 0.9,
        };

        let score1 = compute_qa_prior(&good_features);
        assert!(score1 > 5.0); // Should have high score

        // Poor-quality features
        let poor_features = QAFeatures {
            primitive_mass: 0.0,
            female_mass: 0.0,
            fermat_mass: 0.0,
            phase_entropy: 3.0,
            mean_jk: 50.0,
            mean_harmonic_index: 10.0,
            num_candidates: 1,
            confidence: 0.2,
        };

        let score2 = compute_qa_prior(&poor_features);
        assert!(score2 < score1); // Should have lower score
    }

    #[test]
    fn test_qa_prior_zero_candidates() {
        let features = QAFeatures::default();
        let score = compute_qa_prior(&features);
        assert_eq!(score, 0.0);
    }
}
