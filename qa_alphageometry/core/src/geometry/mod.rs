//! Geometry module - Pure geometric operations
//!
//! This module provides semantic correctness for geometric reasoning.
//! NO heuristics, NO QA logic - just clean geometry.

pub mod normalize;
pub mod constructions;
pub mod check;

pub use normalize::*;
pub use constructions::*;
pub use check::*;
