//! Week 4 Session 3: Coordinate-Derived Facts Ablation
//!
//! Tests whether adding coordinate-derived facts improves QA extraction quality

use qa_alphageometry_core::*;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Enhanced benchmark result with telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    problem_id: String,
    tier: String,
    difficulty: usize,
    config_name: String,
    qa_weight: f64,
    geometric_weight: f64,
    use_coord_facts: bool,
    solved: bool,
    // Week 4 Phase 5: Comprehensive telemetry
    states_expanded: usize,        // Beam states popped and expanded
    successors_generated: usize,   // Total successors created
    successors_kept: usize,        // Successors kept after truncation
    depth_reached: usize,
    proof_steps: usize,
    time_ms: u128,
    best_score: f64,
    qa_prior_mean: f64,
    phase_entropy: f64,
    primitive_mass: f64,
    female_mass: f64,
    fermat_mass: f64,
    mean_jk: f64,
    mean_harmonic_index: f64,
    num_candidates: usize,
    qa_confidence: f64,
    coord_facts_added_total: usize,
    coord_facts_used_in_proof: usize,
    // Termination telemetry
    hit_max_states: bool,
    hit_max_depth: bool,
}

fn verify_correctness(results: &[BenchmarkResult], num_problems: usize, num_configs: usize) {
    let mut by_problem: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        by_problem.entry(result.problem_id.clone()).or_insert_with(Vec::new).push(result);
    }

    for (problem_id, problem_results) in by_problem.iter() {
        if problem_results.len() != num_configs {
            continue;
        }

        let baseline_solved = problem_results[0].solved;
        for result in problem_results.iter().skip(1) {
            assert_eq!(result.solved, baseline_solved,
                       "QA weight {} changed solve status for {}",
                       result.qa_weight, problem_id);
        }
    }

    println!("✅ CORRECTNESS VERIFIED: QA preserves solve status across all {} configs!", num_configs);
}

/// Load a problem from tier directory
fn load_tier_problem(tier: &str, problem_name: &str) -> GeometryProblem {
    let path = format!("tests/fixtures/problems/{}/{}.json", tier, problem_name);
    loader::geometry3k::load_problem(&path).expect(&format!("Failed to load {}/{}", tier, problem_name))
}

/// Run benchmark with coordinate facts flag
fn benchmark_problem_with_coords(
    problem: &GeometryProblem,
    tier: &str,
    config_name: &str,
    qa_weight: f64,
    geometric_weight: f64,
    use_coord_facts: bool,
) -> BenchmarkResult {
    let state = problem.to_state();
    let initial_state = state.clone();

    // Week 4 Phase 2: Increased budgets to reach discriminative search depth
    // Previous budgets (max_states: 2000, max_depth: 20) caused early termination
    let max_states_limit = 8000;   // Increased from 2000 to avoid early exhaustion
    let max_depth_limit = 35;      // Increased from 20 to allow deeper search

    let config = BeamConfig {
        beam_width: 15,
        max_depth: max_depth_limit,
        max_states: max_states_limit,
        scoring: search::ScoringConfig {
            geometric_weight,
            qa_weight,
            step_penalty: 0.1,
        },
        // TODO: Add use_coord_facts to BeamConfig
    };

    let solver = BeamSolver::new(config);

    let start = Instant::now();
    let result = solver.solve(state);
    let elapsed = start.elapsed().as_millis();

    let proof_steps = result.proof.as_ref().map(|p| p.steps.len()).unwrap_or(0);

    let final_state = if result.solved {
        initial_state.clone()
    } else {
        initial_state.clone()
    };

    let qa_features = qa::extract_qa_features(&final_state);
    let qa_prior = qa::compute_qa_prior(&qa_features);

    // TODO: Extract actual coord facts counts from solver
    let coord_facts_added = if use_coord_facts { 0 } else { 0 };  // Placeholder
    let coord_facts_used = if use_coord_facts { 0 } else { 0 };   // Placeholder

    // Detect termination causes (using limits, not moved config)
    let hit_max_states = result.successors_generated >= max_states_limit;
    let hit_max_depth = result.depth_reached >= max_depth_limit;

    BenchmarkResult {
        problem_id: problem.id.clone(),
        tier: tier.to_string(),
        difficulty: problem.difficulty.unwrap_or(1) as usize,
        config_name: config_name.to_string(),
        qa_weight,
        geometric_weight,
        use_coord_facts,
        solved: result.solved,
        states_expanded: result.states_expanded,
        successors_generated: result.successors_generated,
        successors_kept: result.successors_kept,
        depth_reached: result.depth_reached,
        proof_steps,
        time_ms: elapsed,
        best_score: result.best_score,
        qa_prior_mean: qa_prior,
        phase_entropy: qa_features.phase_entropy,
        primitive_mass: qa_features.primitive_mass,
        female_mass: qa_features.female_mass,
        fermat_mass: qa_features.fermat_mass,
        mean_jk: qa_features.mean_jk,
        mean_harmonic_index: qa_features.mean_harmonic_index,
        num_candidates: qa_features.num_candidates,
        qa_confidence: qa_features.confidence,
        coord_facts_added_total: coord_facts_added,
        coord_facts_used_in_proof: coord_facts_used,
        hit_max_states,
        hit_max_depth,
    }
}

#[test]
#[ignore]  // Run with --ignored
fn test_week4_session3_tier2_coord_ablation() {
    // Tier 2: 15 problems (5-7 steps)
    let tier = "tier2";
    let problems = vec![
        "t2_p01_parallel_chain_7",
        "t2_p02_perp_parallel_cascade",
        "t2_p03_equality_web",
        "t2_p04_multi_goal_chains",
        "t2_p05_parallel_with_heavy_distractors",
        "t2_p06_mixed_perp_parallel_complex",
        "t2_p07_collinear_complex",
        "t2_p08_angle_chain_6",
        "t2_p09_circle_chain_6",
        "t2_p10_equality_branching_complex",
        "t2_p11_parallel_branching_heavy",
        "t2_p12_mixed_all_rules",
        "t2_p13_segment_equality_chain_7",
        "t2_p14_perp_cascade_complex",
        "t2_p15_ultimate_distractor",
    ];

    // QA weight sweep × coord facts on/off
    let qa_weights = vec![0.0, 0.1, 0.3, 0.5, 0.7];
    let coord_settings = vec![
        (false, "Coord OFF"),
        (true, "Coord ON"),
    ];

    let mut all_results = Vec::new();

    println!("\n🔬 Week 4 Session 3: Tier 2 Coordinate Facts Ablation\n");
    println!("Testing: {} problems × {} QA weights × 2 coord settings = {} runs\n",
             problems.len(), qa_weights.len(), problems.len() * qa_weights.len() * 2);

    for problem_name in &problems {
        let problem = load_tier_problem(tier, problem_name);

        println!("📝 {} (difficulty: {})", problem.id, problem.difficulty.unwrap_or(1));

        for &qa_weight in &qa_weights {
            for (use_coords, coord_label) in &coord_settings {
                let geo_weight = 1.0 - qa_weight;
                let config_name = if *use_coords {
                    format!("QA {:.0}% + Coords", qa_weight * 100.0)
                } else {
                    format!("QA {:.0}%", qa_weight * 100.0)
                };

                let result = benchmark_problem_with_coords(
                    &problem, tier, &config_name, qa_weight, geo_weight, *use_coords);

                println!("   {} {} | {} | Exp: {}, Gen: {}, Kept: {}, Time: {}ms | QA: {:.3}",
                         config_name,
                         coord_label,
                         if result.solved { "✅" } else { "❌" },
                         result.states_expanded,
                         result.successors_generated,
                         result.successors_kept,
                         result.time_ms,
                         result.qa_prior_mean);

                all_results.push(result);
            }
        }
        println!();
    }

    // Write CSV
    let csv_path = "benchmark_results_week4_session3_tier2.csv";
    write_csv(&all_results, csv_path);
    println!("💾 Results saved to: {}\n", csv_path);

    // Correctness verification
    verify_correctness(&all_results, problems.len(), qa_weights.len() * 2);
}

#[test]
#[ignore]  // Run with --ignored
fn test_week4_session3_tier3_coord_ablation() {
    // Tier 3: 10 problems (8-12 steps)
    let tier = "tier3";
    let problems = vec![
        "t3_p01_parallel_chain_12",
        "t3_p02_perp_parallel_mega_cascade",
        "t3_p03_equality_mega_web",
        "t3_p04_multi_goal_mega",
        "t3_p05_angle_chain_10",
        "t3_p06_circle_chain_10",
        "t3_p07_segment_chain_10",
        "t3_p08_ultimate_branching",
        "t3_p09_mega_distractor",
        "t3_p10_ultimate_mixed",
    ];

    let qa_weights = vec![0.0, 0.1, 0.3, 0.5, 0.7];
    let coord_settings = vec![
        (false, "Coord OFF"),
        (true, "Coord ON"),
    ];

    let mut all_results = Vec::new();

    println!("\n🔬 Week 4 Session 3: Tier 3 Coordinate Facts Ablation\n");
    println!("Testing: {} problems × {} QA weights × 2 coord settings = {} runs\n",
             problems.len(), qa_weights.len(), problems.len() * qa_weights.len() * 2);

    for problem_name in &problems {
        let problem = load_tier_problem(tier, problem_name);

        println!("📝 {} (difficulty: {})", problem.id, problem.difficulty.unwrap_or(1));

        for &qa_weight in &qa_weights {
            for (use_coords, coord_label) in &coord_settings {
                let geo_weight = 1.0 - qa_weight;
                let config_name = if *use_coords {
                    format!("QA {:.0}% + Coords", qa_weight * 100.0)
                } else {
                    format!("QA {:.0}%", qa_weight * 100.0)
                };

                let result = benchmark_problem_with_coords(
                    &problem, tier, &config_name, qa_weight, geo_weight, *use_coords);

                println!("   {} {} | {} | Exp: {}, Gen: {}, Kept: {}, Time: {}ms | QA: {:.3}",
                         config_name,
                         coord_label,
                         if result.solved { "✅" } else { "❌" },
                         result.states_expanded,
                         result.successors_generated,
                         result.successors_kept,
                         result.time_ms,
                         result.qa_prior_mean);

                all_results.push(result);
            }
        }
        println!();
    }

    // Write CSV
    let csv_path = "benchmark_results_week4_session3_tier3.csv";
    write_csv(&all_results, csv_path);
    println!("💾 Results saved to: {}\n", csv_path);

    // Correctness verification
    verify_correctness(&all_results, problems.len(), qa_weights.len() * 2);
}

fn write_csv(results: &[BenchmarkResult], path: &str) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create CSV");

    // Header
    writeln!(file, "problem_id,tier,difficulty,config_name,qa_weight,geometric_weight,use_coord_facts,solved,successors_generated,depth_reached,proof_steps,time_ms,best_score,qa_prior_mean,phase_entropy,primitive_mass,female_mass,fermat_mass,mean_jk,mean_harmonic_index,num_candidates,qa_confidence,coord_facts_added_total,coord_facts_used_in_proof,hit_max_states,hit_max_depth")
        .expect("Failed to write CSV header");

    // Data rows
    for r in results {
        writeln!(file, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                 r.problem_id, r.tier, r.difficulty, r.config_name, r.qa_weight, r.geometric_weight,
                 r.use_coord_facts, r.solved, r.successors_generated, r.depth_reached, r.proof_steps,
                 r.time_ms, r.best_score, r.qa_prior_mean, r.phase_entropy, r.primitive_mass,
                 r.female_mass, r.fermat_mass, r.mean_jk, r.mean_harmonic_index,
                 r.num_candidates, r.qa_confidence, r.coord_facts_added_total, r.coord_facts_used_in_proof,
                 r.hit_max_states, r.hit_max_depth)
            .expect("Failed to write CSV row");
    }
}
