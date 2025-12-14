//! Python bindings for QA-AlphaGeometry
//!
//! TODO: Implement PyO3 bindings after core is complete

use pyo3::prelude::*;

#[pyfunction]
fn solve(_problem: &str) -> PyResult<String> {
    Ok("Not yet implemented".to_string())
}

#[pymodule]
fn qa_geo(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
