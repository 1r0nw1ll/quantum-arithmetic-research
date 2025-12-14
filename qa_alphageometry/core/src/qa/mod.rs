//! Quantum Arithmetic (QA) module for geometric priors
//!
//! This module provides QA tuple extraction from geometric facts and
//! uses discrete harmonic structure to guide symbolic search.

pub mod tuple;
pub mod extract;  // Extract QA tuples from geometric configurations

pub use tuple::{QATuple, TupleClass};
pub use extract::{QAFeatures, extract_qa_features, compute_qa_prior};

// TODO: Add these modules if needed
// pub mod classify; // Batch classification and indexing
// pub mod priors;   // Additional prior computation
