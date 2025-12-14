//! QA-AlphaGeometry Core
//!
//! Fast symbolic geometry solver with discrete harmonic priors

pub mod qa;
pub mod ir;       // Intermediate representation (symbols, facts, state, proof, coords)
pub mod geometry; // Geometric operations (normalize, construct, check)
pub mod search;   // Beam search with QA priors
pub mod rules;    // Deduction rules
pub mod loader;   // Problem loaders (Geometry3K, etc.)

pub use qa::{QATuple, TupleClass, QAFeatures, extract_qa_features, compute_qa_prior};
pub use ir::*;
pub use geometry::*;
pub use search::{BeamSolver, BeamConfig, SearchResult};
pub use rules::{Rule, all_rules};
pub use loader::{GeometryProblem, LoadError, LoadResult};
