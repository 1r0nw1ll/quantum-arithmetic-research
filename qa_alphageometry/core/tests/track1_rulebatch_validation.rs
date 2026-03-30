use qa_alphageometry_core::{
    ir::GeoState,
    loader::geometry3k::load_problem,
    search::{BeamConfig, BeamSolver, ScoringConfig},
};
use std::path::Path;

#[test]
#[ignore]
fn test_t02_scaled_multisurface() {
    let path = Path::new("tests/fixtures/problems/synthetic/t02_scaled_multisurface.json");
    let problem = load_problem(path).expect("Failed to load t02");

    // Test with QA=0 (baseline)
    let config_qa0 = BeamConfig {
        beam_width: 8,
        max_depth: 10,
        max_states: 500,
        scoring: ScoringConfig {
            geometric_weight: 1.0,  // QA disabled
            qa_weight: 0.0,
            step_penalty: 0.1,
        },
    };

    let solver = BeamSolver::new(config_qa0);
    let initial_state = problem.to_state();
    let result_qa0 = solver.solve(initial_state);

    // Test with QA=0.7
    let config_qa70 = BeamConfig {
        beam_width: 8,
        max_depth: 10,
        max_states: 500,
        scoring: ScoringConfig {
            geometric_weight: 0.7,  // QA enabled
            qa_weight: 0.7,
            step_penalty: 0.1,
        },
    };

    let solver = BeamSolver::new(config_qa70);
    let initial_state = problem.to_state();
    let result_qa70 = solver.solve(initial_state);

    println!("\n============================================================");
    println!("T02 SCALED MULTISURFACE VALIDATION");
    println!("============================================================");
    println!("\nQA=0 Results:");
    println!("  Solved: {}", result_qa0.solved);
    println!("  States expanded: {}", result_qa0.states_expanded);
    println!("  Successors generated: {} (rules fired)", result_qa0.successors_generated);
    println!("  Depth reached: {}", result_qa0.depth_reached);

    println!("\nQA=0.7 Results:");
    println!("  Solved: {}", result_qa70.solved);
    println!("  States expanded: {}", result_qa70.states_expanded);
    println!("  Successors generated: {} (rules fired)", result_qa70.successors_generated);
    println!("  Depth reached: {}", result_qa70.depth_reached);

    // Check for beam divergence
    let sig_qa0: Vec<_> = result_qa0.beam_signatures.iter().map(|(d, h)| (*d, *h)).collect();
    let sig_qa70: Vec<_> = result_qa70.beam_signatures.iter().map(|(d, h)| (*d, *h)).collect();

    println!("\nBeam Divergence Analysis:");
    let mut first_divergence = None;
    for depth in 0..=result_qa0.depth_reached.min(result_qa70.depth_reached) {
        let hash0 = sig_qa0.iter().find(|(d, _)| *d == depth).map(|(_, h)| h);
        let hash70 = sig_qa70.iter().find(|(d, _)| *d == depth).map(|(_, h)| h);

        if hash0 != hash70 && first_divergence.is_none() {
            first_divergence = Some(depth);
            println!("  First divergence at depth: {}", depth);
            break;
        }
    }

    if first_divergence.is_none() {
        println!("  ❌ NO DIVERGENCE (beams identical across all depths)");
    } else {
        println!("  ✅ Divergence detected");
    }

    // Predictions from branching_score.py
    println!("\nPredicted vs Actual:");
    println!("  Predicted rule families: 8");
    println!("  Predicted fact volume: 47");
    println!("  Actual successors (rules): {}", result_qa0.successors_generated);

    assert!(result_qa0.solved, "QA=0 should solve");
    assert!(result_qa70.solved, "QA=0.7 should solve");
}

#[test]
#[ignore]
fn test_t03_mega_discrimination() {
    let path = Path::new("tests/fixtures/problems/synthetic/t03_mega_discrimination.json");
    let problem = load_problem(path).expect("Failed to load t03");

    let config_qa0 = BeamConfig {
        beam_width: 8,
        max_depth: 10,
        max_states: 500,
        scoring: ScoringConfig {
            geometric_weight: 1.0,
            qa_weight: 0.0,
            step_penalty: 0.1,
        },
    };

    let solver = BeamSolver::new(config_qa0);
    let initial_state = problem.to_state();
    let result_qa0 = solver.solve(initial_state);

    let config_qa70 = BeamConfig {
        beam_width: 8,
        max_depth: 10,
        max_states: 500,
        scoring: ScoringConfig {
            geometric_weight: 0.7,
            qa_weight: 0.7,
            step_penalty: 0.1,
        },
    };

    let solver = BeamSolver::new(config_qa70);
    let initial_state = problem.to_state();
    let result_qa70 = solver.solve(initial_state);

    println!("\n============================================================");
    println!("T03 MEGA DISCRIMINATION VALIDATION");
    println!("============================================================");
    println!("\nQA=0 Results:");
    println!("  Solved: {}", result_qa0.solved);
    println!("  States expanded: {}", result_qa0.states_expanded);
    println!("  Successors generated: {} (rules fired)", result_qa0.successors_generated);
    println!("  Depth reached: {}", result_qa0.depth_reached);

    println!("\nQA=0.7 Results:");
    println!("  Solved: {}", result_qa70.solved);
    println!("  States expanded: {}", result_qa70.states_expanded);
    println!("  Successors generated: {} (rules fired)", result_qa70.successors_generated);
    println!("  Depth reached: {}", result_qa70.depth_reached);

    let sig_qa0: Vec<_> = result_qa0.beam_signatures.iter().map(|(d, h)| (*d, *h)).collect();
    let sig_qa70: Vec<_> = result_qa70.beam_signatures.iter().map(|(d, h)| (*d, *h)).collect();

    println!("\nBeam Divergence Analysis:");
    let mut first_divergence = None;
    for depth in 0..=result_qa0.depth_reached.min(result_qa70.depth_reached) {
        let hash0 = sig_qa0.iter().find(|(d, _)| *d == depth).map(|(_, h)| h);
        let hash70 = sig_qa70.iter().find(|(d, _)| *d == depth).map(|(_, h)| h);

        if hash0 != hash70 && first_divergence.is_none() {
            first_divergence = Some(depth);
            println!("  First divergence at depth: {}", depth);
            break;
        }
    }

    if first_divergence.is_none() {
        println!("  ❌ NO DIVERGENCE (beams identical)");
    } else {
        println!("  ✅ Divergence detected");
    }

    println!("\nPredicted vs Actual:");
    println!("  Predicted rule families: 8");
    println!("  Predicted fact volume: 103");
    println!("  Actual successors (rules): {}", result_qa0.successors_generated);

    assert!(result_qa0.solved, "QA=0 should solve");
    assert!(result_qa70.solved, "QA=0.7 should solve");
}
