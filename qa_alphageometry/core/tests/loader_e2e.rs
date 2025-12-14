//! End-to-end loader integration tests
//!
//! Tests problem loading → solving → proof generation

use qa_alphageometry_core::*;
use std::path::Path;

/// Helper to load a problem from fixtures
fn load_problem(name: &str) -> GeometryProblem {
    let path = format!("tests/fixtures/problems/{}.json", name);
    loader::geometry3k::load_problem(&path).expect("Failed to load problem")
}

#[test]
fn test_p01_parallel_transitivity() {
    let problem = load_problem("p01_parallel_transitivity");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved, "Problem 01 should be solved");
    assert!(result.proof.is_some());

    let proof = result.proof.unwrap();
    println!("✅ P01: Solved in {} steps, {} states explored",
             proof.steps.len(), result.states_explored);

    // Verify proof serializes
    let json = serde_json::to_string(&proof).expect("Failed to serialize proof");
    assert!(!json.is_empty());
}

#[test]
fn test_p02_perpendicular_symmetry() {
    let problem = load_problem("p02_perpendicular_symmetry");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    // May be already satisfied due to normalization
    assert!(result.solved, "Problem 02 should be solved");

    println!("✅ P02: Solved at depth {}, {} states explored",
             result.depth_reached, result.states_explored);
}

#[test]
fn test_p03_perp_parallel_propagation() {
    let problem = load_problem("p03_perp_parallel_propagation");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved, "Problem 03 should be solved");
    assert!(result.proof.is_some());

    let proof = result.proof.unwrap();
    assert!(proof.steps.len() >= 1, "Should require at least 1 step");

    println!("✅ P03: Solved in {} steps, {} states explored",
             proof.steps.len(), result.states_explored);
}

#[test]
fn test_p04_double_perp_parallel() {
    let problem = load_problem("p04_double_perp_parallel");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved, "Problem 04 should be solved");
    assert!(result.proof.is_some());

    println!("✅ P04: Solved in {} steps",
             result.proof.unwrap().steps.len());
}

#[test]
fn test_p05_collinear_on_line() {
    let problem = load_problem("p05_collinear_on_line");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved, "Problem 05 should be solved");

    println!("✅ P05: Solved in {} states",
             result.states_explored);
}

#[test]
fn test_p06_segment_equality_transitivity() {
    let problem = load_problem("p06_segment_equality_transitivity");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved, "Problem 06 should be solved");

    println!("✅ P06: Solved in {} steps",
             result.proof.unwrap().steps.len());
}

#[test]
fn test_p07_on_circle_concyclic() {
    let problem = load_problem("p07_on_circle_concyclic");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved, "Problem 07 should be solved");

    println!("✅ P07: Solved in {} steps",
             result.proof.unwrap().steps.len());
}

#[test]
fn test_p08_concentric_transitivity() {
    let problem = load_problem("p08_concentric_transitivity");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved, "Problem 08 should be solved");

    println!("✅ P08: Solved in {} steps",
             result.proof.unwrap().steps.len());
}

#[test]
fn test_p09_coincident_lines() {
    let problem = load_problem("p09_coincident_lines");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved, "Problem 09 should be solved");

    println!("✅ P09: Solved in {} steps",
             result.proof.unwrap().steps.len());
}

#[test]
fn test_p10_mixed_multi_step() {
    let problem = load_problem("p10_mixed_multi_step");
    let state = problem.to_state();

    let config = BeamConfig {
        beam_width: 15,
        max_depth: 10,
        max_states: 500,
        ..Default::default()
    };

    let solver = BeamSolver::new(config);
    let result = solver.solve(state);

    assert!(result.solved, "Problem 10 should be solved");
    assert!(result.proof.is_some());

    let proof = result.proof.unwrap();
    assert!(proof.steps.len() >= 2, "Should require multiple steps");

    println!("✅ P10: Solved in {} steps, {} states explored",
             proof.steps.len(), result.states_explored);
}

#[test]
fn test_qa_on_vs_off_comparison() {
    let problem = load_problem("p10_mixed_multi_step");

    // TEST 1: QA OFF (qa_weight = 0.0)
    let state1 = problem.to_state();
    let config_qa_off = BeamConfig {
        beam_width: 15,
        max_depth: 10,
        max_states: 500,
        scoring: search::ScoringConfig {
            geometric_weight: 1.0,
            qa_weight: 0.0,  // QA OFF
            step_penalty: 0.1,
        },
    };

    let solver_qa_off = BeamSolver::new(config_qa_off);
    let result_qa_off = solver_qa_off.solve(state1);

    // TEST 2: QA ON (qa_weight = 0.3)
    let state2 = problem.to_state();
    let config_qa_on = BeamConfig {
        beam_width: 15,
        max_depth: 10,
        max_states: 500,
        scoring: search::ScoringConfig {
            geometric_weight: 0.7,
            qa_weight: 0.3,  // QA ON
            step_penalty: 0.1,
        },
    };

    let solver_qa_on = BeamSolver::new(config_qa_on);
    let result_qa_on = solver_qa_on.solve(state2);

    println!("\n📊 QA ON vs OFF Comparison (Problem 10):");
    println!("   QA OFF: Solved={}, States={}, Depth={}",
             result_qa_off.solved, result_qa_off.states_explored, result_qa_off.depth_reached);
    println!("   QA ON:  Solved={}, States={}, Depth={}",
             result_qa_on.solved, result_qa_on.states_explored, result_qa_on.depth_reached);

    // Both should solve (or both fail) for correctness
    assert_eq!(result_qa_off.solved, result_qa_on.solved,
               "QA guidance should not prevent solving");

    if result_qa_off.solved && result_qa_on.solved {
        println!("   ✅ Both configurations solved the problem!");
        println!("   QA guidance preserves correctness while potentially reordering search.");
    }
}

#[test]
fn test_proof_trace_serialization() {
    let problem = load_problem("p01_parallel_transitivity");
    let state = problem.to_state();

    let solver = BeamSolver::new(BeamConfig::default());
    let result = solver.solve(state);

    assert!(result.solved);
    let proof = result.proof.unwrap();

    // Test JSON serialization
    let json = serde_json::to_string_pretty(&proof)
        .expect("Failed to serialize proof");

    println!("\n📝 Proof trace (JSON):\n{}", json);

    // Test deserialization
    let _proof_copy: ProofTrace = serde_json::from_str(&json)
        .expect("Failed to deserialize proof");

    assert!(!json.is_empty());
    println!("   ✅ Proof serialization works!");
}
