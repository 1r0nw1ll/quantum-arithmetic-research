//! Intermediate Representation (IR) module for QA-AlphaGeometry
//!
//! This module provides the core data structures for representing geometric proofs:
//! - **symbols**: Type-safe identifiers for geometric objects with string interning
//! - **facts**: Atomic geometric predicates and fact storage
//! - **state**: Complete proof state including known facts and goals
//! - **proof**: Proof steps and complete proof traces with serialization
//!
//! # Example
//!
//! ```rust
//! use qa_alphageometry_core::ir::*;
//!
//! // Create a symbol table and intern some points
//! let symbols = SymbolTable::new();
//! let a = symbols.get_or_intern_point("A");
//! let b = symbols.get_or_intern_point("B");
//! let c = symbols.get_or_intern_point("C");
//!
//! // Create facts about the geometry
//! let mut facts = FactStore::new();
//! facts.insert(Fact::Collinear(a, b, c));
//!
//! // Define a goal
//! let l1 = symbols.get_or_intern_line("L1");
//! let l2 = symbols.get_or_intern_line("L2");
//! let goal = Goal::single(Fact::Parallel(l1, l2));
//!
//! // Create the proof state
//! let state = GeoState::new(symbols, facts, goal);
//!
//! // Work with the state
//! println!("Known facts: {}", state.num_facts());
//! println!("Goal satisfied: {}", state.is_goal_satisfied());
//! ```

mod facts;
mod proof;
mod state;
mod symbols;
mod coords;

// Re-export all public items for convenient access
pub use facts::{Fact, FactStore, FactType, ProofStepId};
pub use proof::{ProofError, ProofResult, ProofStatistics, ProofStep, ProofTrace};
pub use state::{GeoState, Goal, Metadata};
pub use symbols::{AngleId, CircleId, GeometricId, LineId, PointId, SegmentId, SymbolTable};
pub use coords::{Point2D, CoordinateStore};
