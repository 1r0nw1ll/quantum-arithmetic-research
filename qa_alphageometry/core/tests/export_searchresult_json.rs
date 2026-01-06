//! Test for exporting SearchResult to JSON
//!
//! This test demonstrates how to export proof search results to JSON files
//! for integration with the QA certificate system.

use qa_alphageometry_core::*;
use std::fs;

/// Test exporting a successful proof to JSON
#[test]
fn test_export_parallel_transitivity_proof() {
    // Create a simple geometry problem: Prove L1∥L3 from L1∥L2 and L2∥L3
    let mut facts = ir::FactStore::new();
    facts.insert(ir::Fact::Parallel(ir::LineId(1), ir::LineId(2)));
    facts.insert(ir::Fact::Parallel(ir::LineId(2), ir::LineId(3)));

    let target = ir::Fact::Parallel(ir::LineId(1), ir::LineId(3));
    let goal = ir::Goal::new(vec![target.clone()]);

    let state = ir::GeoState::new(Default::default(), facts, goal);

    let config = search::BeamConfig {
        beam_width: 10,
        max_depth: 5,
        max_states: 100,
        scoring: search::ScoringConfig::default(),
    };

    let solver = search::BeamSolver::new(config);
    let result = solver.solve(state);

    // Verify proof was found
    assert!(result.solved, "Failed to find proof for parallel transitivity");

    // Export to JSON
    let output_path = "parallel_transitivity_proof.searchresult.json";
    result.to_json_file(output_path).expect("Failed to write JSON");

    // Verify file was created and is valid JSON
    let json_content = fs::read_to_string(output_path).expect("Failed to read JSON file");
    let parsed: serde_json::Value = serde_json::from_str(&json_content)
        .expect("Failed to parse JSON");

    // Verify key fields are present
    assert_eq!(parsed["solved"], true);
    assert!(parsed["proof"].is_object());
    assert!(parsed["states_expanded"].is_number());
    assert!(parsed["successors_generated"].is_number());
    assert!(parsed["depth_reached"].is_number());

    println!("✅ Exported SearchResult to: {}", output_path);
    println!("   Solved: {}", result.solved);
    println!("   Steps: {}", result.proof.as_ref().unwrap().steps.len());
    println!("   States expanded: {}", result.states_expanded);
    println!("   Successors generated: {}", result.successors_generated);
    println!("   Depth: {}", result.depth_reached);

    // Clean up (optional - comment out if you want to inspect the file)
    // fs::remove_file(output_path).ok();
}

/// Test exporting a failed proof attempt (depth exhaustion) to JSON
#[test]
fn test_export_unsolvable_obstruction() {
    // Create an unsolvable problem (empty facts, non-trivial goal)
    let facts = ir::FactStore::new();
    let target = ir::Fact::Parallel(ir::LineId(1), ir::LineId(2));
    let goal = ir::Goal::new(vec![target]);

    let state = ir::GeoState::new(Default::default(), facts, goal);

    let config = search::BeamConfig {
        beam_width: 5,
        max_depth: 3,
        max_states: 50,
        scoring: search::ScoringConfig::default(),
    };

    let solver = search::BeamSolver::new(config);
    let result = solver.solve(state);

    // Verify proof was NOT found
    assert!(!result.solved, "Should not find proof for unsolvable problem");

    // Export to JSON
    let output_path = "unsolvable_obstruction.searchresult.json";
    result.to_json_file(output_path).expect("Failed to write JSON");

    // Verify file was created and is valid JSON
    let json_content = fs::read_to_string(output_path).expect("Failed to read JSON file");
    let parsed: serde_json::Value = serde_json::from_str(&json_content)
        .expect("Failed to parse JSON");

    // Verify key fields are present
    assert_eq!(parsed["solved"], false);
    assert!(parsed["proof"].is_null());
    assert!(parsed["states_expanded"].is_number());
    assert!(parsed["depth_reached"].is_number());

    println!("✅ Exported obstruction SearchResult to: {}", output_path);
    println!("   Solved: {}", result.solved);
    println!("   States expanded: {}", result.states_expanded);
    println!("   Depth reached: {}", result.depth_reached);

    // Clean up (optional)
    // fs::remove_file(output_path).ok();
}

/// Test roundtrip: export to JSON and reload
#[test]
fn test_searchresult_json_roundtrip() {
    // Create simple proof
    let mut facts = ir::FactStore::new();
    facts.insert(ir::Fact::Parallel(ir::LineId(1), ir::LineId(2)));
    facts.insert(ir::Fact::Parallel(ir::LineId(2), ir::LineId(3)));

    let target = ir::Fact::Parallel(ir::LineId(1), ir::LineId(3));
    let goal = ir::Goal::new(vec![target]);

    let state = ir::GeoState::new(Default::default(), facts, goal);

    let config = search::BeamConfig::default();
    let solver = search::BeamSolver::new(config);
    let result = solver.solve(state);

    // Export
    let temp_path = "temp_roundtrip.searchresult.json";
    result.to_json_file(temp_path).expect("Failed to write JSON");

    // Reload
    let reloaded = search::SearchResult::from_json_file(temp_path)
        .expect("Failed to load JSON");

    // Verify fields match
    assert_eq!(result.solved, reloaded.solved);
    assert_eq!(result.states_expanded, reloaded.states_expanded);
    assert_eq!(result.successors_generated, reloaded.successors_generated);
    assert_eq!(result.depth_reached, reloaded.depth_reached);
    assert!((result.best_score - reloaded.best_score).abs() < 1e-10);

    if let (Some(original), Some(reloaded_proof)) = (&result.proof, &reloaded.proof) {
        assert_eq!(original.steps.len(), reloaded_proof.steps.len());
        assert_eq!(original.solved, reloaded_proof.solved);
    }

    println!("✅ Roundtrip successful - JSON serialization preserves all data");

    // Clean up
    fs::remove_file(temp_path).ok();
}
