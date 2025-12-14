//! Search module - Beam search with QA-guided scoring
//!
//! Minimum viable solver: beam search + geometric heuristic + QA prior

pub mod beam;
pub mod scoring;

pub use beam::*;
pub use scoring::*;
