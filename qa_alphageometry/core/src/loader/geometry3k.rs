//! Geometry3K dataset loader
//!
//! Loads problems from Geometry3K format

use super::{GeometryProblem, LoadResult, LoadError};
use crate::ir::Fact;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Geometry3K problem format (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometry3KProblem {
    /// Problem ID
    pub id: String,

    /// Problem text
    pub text: String,

    /// Given facts (simplified - in real Geometry3K this is more complex)
    #[serde(default)]
    pub givens: Vec<String>,

    /// Goal fact(s)
    #[serde(default)]
    pub goals: Vec<String>,

    /// Optional answer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
}

impl Geometry3KProblem {
    /// Convert to GeometryProblem
    ///
    /// This is a simplified parser - real Geometry3K needs more sophisticated parsing
    pub fn to_geometry_problem(&self) -> LoadResult<GeometryProblem> {
        // TODO: Implement proper parsing from Geometry3K format to Facts
        // For now, return error indicating not yet implemented

        Err(LoadError::UnsupportedFact(
            "Geometry3K parsing not yet implemented - use simple JSON format".to_string()
        ))
    }
}

/// Load a single problem from Geometry3K JSON file
pub fn load_problem<P: AsRef<Path>>(path: P) -> LoadResult<GeometryProblem> {
    let path = path.as_ref();

    let contents = fs::read_to_string(path)
        .map_err(|e| LoadError::FileNotFound(format!("{}: {}", path.display(), e)))?;

    let problem: GeometryProblem = serde_json::from_str(&contents)
        .map_err(|e| LoadError::ParseError(e.to_string()))?;

    Ok(problem)
}

/// Load multiple problems from a directory
pub fn load_problems<P: AsRef<Path>>(dir: P) -> LoadResult<Vec<GeometryProblem>> {
    let dir = dir.as_ref();

    if !dir.is_dir() {
        return Err(LoadError::FileNotFound(format!(
            "{} is not a directory",
            dir.display()
        )));
    }

    let mut problems = Vec::new();

    for entry in fs::read_dir(dir)
        .map_err(|e| LoadError::FileNotFound(format!("{}: {}", dir.display(), e)))?
    {
        let entry = entry.map_err(|e| LoadError::FileNotFound(e.to_string()))?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            match load_problem(&path) {
                Ok(problem) => problems.push(problem),
                Err(e) => {
                    eprintln!("Warning: Failed to load {}: {}", path.display(), e);
                }
            }
        }
    }

    Ok(problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry3k_problem_struct() {
        let problem = Geometry3KProblem {
            id: "geo_001".to_string(),
            text: "Prove that parallel lines are transitive".to_string(),
            givens: vec!["L1||L2".to_string(), "L2||L3".to_string()],
            goals: vec!["L1||L3".to_string()],
            answer: None,
        };

        assert_eq!(problem.id, "geo_001");
        assert_eq!(problem.givens.len(), 2);
        assert_eq!(problem.goals.len(), 1);
    }
}
