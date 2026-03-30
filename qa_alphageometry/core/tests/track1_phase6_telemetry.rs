// Track 1 Phase 6: Comprehensive Telemetry
//
// Tests all 30 discriminative problems with QA weights [0.0, 0.3, 0.7, 1.0]
// Collects metrics to validate QA heuristic effectiveness

use qa_alphageometry_core::{
    loader::geometry3k::load_problem,
    search::{BeamConfig, BeamSolver, ScoringConfig},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TelemetryResult {
    problem_id: String,
    family: String,
    qa_weight: f64,
    solved: bool,
    states_expanded: usize,
    successors_generated: usize,
    depth_reached: usize,
    first_divergence: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TelemetrySummary {
    total_problems: usize,
    families: HashMap<String, FamilyStats>,
    qa_weights: Vec<f64>,
    results: Vec<TelemetryResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FamilyStats {
    total: usize,
    solved_all_weights: usize,
    divergence_detected: usize,
    avg_states_qa0: f64,
    avg_states_qa07: f64,
    efficiency_gain: f64,
}

fn run_problem_with_qa(
    problem_path: &Path,
    problem_id: &str,
    family: &str,
    qa_weight: f64,
) -> TelemetryResult {
    let problem = load_problem(problem_path).expect(&format!("Failed to load {}", problem_id));

    let config = BeamConfig {
        beam_width: 8,
        max_depth: 10,
        max_states: 500,
        scoring: ScoringConfig {
            geometric_weight: if qa_weight > 0.0 { 1.0 - qa_weight } else { 1.0 },
            qa_weight,
            step_penalty: 0.1,
        },
    };

    let solver = BeamSolver::new(config);
    let initial_state = problem.to_state();
    let result = solver.solve(initial_state);

    TelemetryResult {
        problem_id: problem_id.to_string(),
        family: family.to_string(),
        qa_weight,
        solved: result.solved,
        states_expanded: result.states_expanded,
        successors_generated: result.successors_generated,
        depth_reached: result.depth_reached,
        first_divergence: None, // Will be computed later
    }
}

fn detect_divergence(results: &[TelemetryResult], problem_id: &str) -> Option<usize> {
    // Find results for this problem
    let problem_results: Vec<_> = results
        .iter()
        .filter(|r| r.problem_id == problem_id)
        .collect();

    if problem_results.len() < 2 {
        return None;
    }

    // Compare QA=0 vs QA>0 (we don't have beam signatures here, so use state counts as proxy)
    let qa0 = problem_results.iter().find(|r| r.qa_weight == 0.0)?;
    let qa_pos = problem_results.iter().find(|r| r.qa_weight > 0.0)?;

    // If they differ in states_expanded or depth_reached, there's likely divergence
    if qa0.states_expanded != qa_pos.states_expanded || qa0.depth_reached != qa_pos.depth_reached
    {
        Some(0) // Assume divergence at depth 0 if metrics differ
    } else {
        None
    }
}

#[test]
#[ignore]
fn test_track1_phase6_telemetry() {
    println!("\n{}", "=".repeat(70));
    println!("TRACK 1 PHASE 6: COMPREHENSIVE TELEMETRY");
    println!("{}\n", "=".repeat(70));

    let qa_weights = vec![0.0, 0.3, 0.7, 1.0];
    let mut all_results = Vec::new();

    // Family S: Lattices (s01-s10)
    println!("Family S (Perpendicular Lattices):");
    println!("{}", "-".repeat(70));
    for i in 1..=10 {
        let problem_id = format!("s{:02}_lattice", i);
        let path_str = format!(
            "tests/fixtures/problems/synthetic/{}.json",
            problem_id
        );
        let path = Path::new(&path_str);

        if !path.exists() {
            eprintln!("⚠️  Skipping {}: file not found", problem_id);
            continue;
        }

        print!("{:20}", problem_id);
        for &qa in &qa_weights {
            let result = run_problem_with_qa(path, &problem_id, "S", qa);
            print!(
                " QA={:.1}:{:3}s",
                qa, result.states_expanded
            );
            all_results.push(result);
        }
        println!();
    }

    // Family T: Multi-surface (t01-t10)
    println!("\nFamily T (Multi-Surface Competing Routes):");
    println!("{}", "-".repeat(70));

    // T-family problems with their actual filenames
    let t_problems = vec![
        "t01_dual_route_reference",
        "t02_scaled_multisurface",
        "t03_mega_discrimination",
        "t04_multisurface_4h_5c",
        "t05_multisurface_5h_6c",
        "t06_multisurface_3h_4c",
        "t07_multisurface_4h_5c",
        "t08_multisurface_5h_6c",
        "t09_multisurface_3h_4c",
        "t10_multisurface_4h_5c",
    ];

    for problem_id in t_problems {
        let path_str = format!(
            "tests/fixtures/problems/synthetic/{}.json",
            problem_id
        );
        let path = Path::new(&path_str);

        if !path.exists() {
            eprintln!("⚠️  Skipping {}: file not found", problem_id);
            continue;
        }

        print!("{:30}", problem_id);
        for &qa in &qa_weights {
            let result = run_problem_with_qa(path, problem_id, "T", qa);
            print!(" QA={:.1}:{:3}s", qa, result.states_expanded);
            all_results.push(result);
        }
        println!();
    }

    // Family C: Coordinate-derived (c01-c10)
    println!("\nFamily C (Coordinate-Derived Pythagorean):");
    println!("{}", "-".repeat(70));
    for i in 1..=10 {
        let problem_id = format!("c{:02}_pythagorean", i);
        // Find the actual file (they have full triple names)
        let pattern = format!("tests/fixtures/problems/synthetic/c{:02}_*.json", i);
        let paths: Vec<_> = glob::glob(&pattern)
            .expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect();

        if paths.is_empty() {
            eprintln!("⚠️  Skipping c{:02}: no matching file", i);
            continue;
        }

        let path = &paths[0];
        let actual_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&problem_id);

        print!("{:30}", actual_id);
        for &qa in &qa_weights {
            let result = run_problem_with_qa(path, actual_id, "C", qa);
            print!(" QA={:.1}:{:3}s", qa, result.states_expanded);
            all_results.push(result);
        }
        println!();
    }

    // Compute divergence for each problem
    let problem_ids: Vec<String> = all_results
        .iter()
        .map(|r| r.problem_id.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // First collect divergence data
    let divergence_map: HashMap<String, Option<usize>> = problem_ids
        .iter()
        .map(|pid| (pid.clone(), detect_divergence(&all_results, pid)))
        .collect();

    // Then apply it
    for result in &mut all_results {
        if result.first_divergence.is_none() {
            result.first_divergence = divergence_map.get(&result.problem_id).cloned().flatten();
        }
    }

    // Compute family statistics
    let mut family_stats: HashMap<String, FamilyStats> = HashMap::new();

    for family in &["S", "T", "C"] {
        let family_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.family == *family)
            .collect();

        let problem_count = problem_ids
            .iter()
            .filter(|id| family_results.iter().any(|r| r.problem_id == **id))
            .count();

        let solved_all = problem_ids
            .iter()
            .filter(|id| {
                qa_weights.iter().all(|&qa| {
                    family_results
                        .iter()
                        .any(|r| r.problem_id == **id && r.qa_weight == qa && r.solved)
                })
            })
            .count();

        let divergence_count = problem_ids
            .iter()
            .filter(|id| {
                all_results
                    .iter()
                    .any(|r| r.problem_id == **id && r.first_divergence.is_some())
            })
            .count();

        let qa0_states: Vec<_> = family_results
            .iter()
            .filter(|r| r.qa_weight == 0.0)
            .map(|r| r.states_expanded as f64)
            .collect();

        let qa07_states: Vec<_> = family_results
            .iter()
            .filter(|r| r.qa_weight == 0.7)
            .map(|r| r.states_expanded as f64)
            .collect();

        let avg_qa0 = if !qa0_states.is_empty() {
            qa0_states.iter().sum::<f64>() / qa0_states.len() as f64
        } else {
            0.0
        };

        let avg_qa07 = if !qa07_states.is_empty() {
            qa07_states.iter().sum::<f64>() / qa07_states.len() as f64
        } else {
            0.0
        };

        let efficiency_gain = if avg_qa0 > 0.0 {
            (avg_qa0 - avg_qa07) / avg_qa0 * 100.0
        } else {
            0.0
        };

        family_stats.insert(
            family.to_string(),
            FamilyStats {
                total: problem_count,
                solved_all_weights: solved_all,
                divergence_detected: divergence_count,
                avg_states_qa0: avg_qa0,
                avg_states_qa07: avg_qa07,
                efficiency_gain,
            },
        );
    }

    // Print summary
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY STATISTICS");
    println!("{}\n", "=".repeat(70));

    for family in &["S", "T", "C"] {
        if let Some(stats) = family_stats.get(*family) {
            println!("Family {}:", family);
            println!("  Total problems: {}", stats.total);
            println!("  Solved (all QA weights): {}/{}", stats.solved_all_weights, stats.total);
            println!("  Divergence detected: {}/{}", stats.divergence_detected, stats.total);
            println!("  Avg states (QA=0.0): {:.1}", stats.avg_states_qa0);
            println!("  Avg states (QA=0.7): {:.1}", stats.avg_states_qa07);
            println!("  Efficiency gain: {:.1}%\n", stats.efficiency_gain);
        }
    }

    // Save results to JSON
    let summary = TelemetrySummary {
        total_problems: problem_ids.len(),
        families: family_stats,
        qa_weights,
        results: all_results,
    };

    let json = serde_json::to_string_pretty(&summary).expect("Failed to serialize results");
    fs::write("track1_phase6_telemetry_results.json", json)
        .expect("Failed to write results");

    println!("✅ Telemetry complete!");
    println!("📊 Results saved to: track1_phase6_telemetry_results.json");
}
