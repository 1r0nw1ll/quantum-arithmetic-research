//! Week 4 Benchmark: Step-Depth Ladder with Enhanced Telemetry
//!
//! Tests QA guidance efficiency on harder problems (Tier 0-3) with full telemetry

use qa_alphageometry_core::*;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Enhanced benchmark result with telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    // Problem metadata
    problem_id: String,
    tier: String,
    difficulty: usize,

    // Configuration
    config_name: String,
    qa_weight: f64,
    geometric_weight: f64,
    use_coord_facts: bool,  // For Session 3 ablation

    // Search metrics
    solved: bool,
    successors_generated: usize,
    depth_reached: usize,
    proof_steps: usize,
    time_ms: u128,
    best_score: f64,

    // QA Telemetry (computed from final state)
    qa_prior_mean: f64,
    phase_entropy: f64,
    primitive_mass: f64,
    female_mass: f64,
    fermat_mass: f64,
    mean_jk: f64,
    mean_harmonic_index: f64,
    num_candidates: usize,
    qa_confidence: f64,

    // Session 3: Coordinate facts telemetry
    coord_facts_added_total: usize,
    coord_facts_used_in_proof: usize,
}

/// Summary statistics for a tier-config combination
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TierSummary {
    tier: String,
    config_name: String,
    qa_weight: f64,

    solve_rate_pct: f64,
    problems_solved: usize,
    problems_total: usize,

    avg_states: f64,
    std_states: f64,

    avg_steps: f64,
    avg_time_ms: f64,

    avg_qa_prior: f64,
    avg_phase_entropy: f64,
}

/// Load a problem from tier directory
fn load_tier_problem(tier: &str, problem_name: &str) -> GeometryProblem {
    let path = format!("tests/fixtures/problems/{}/{}.json", tier, problem_name);
    loader::geometry3k::load_problem(&path).expect(&format!("Failed to load {}/{}", tier, problem_name))
}

/// Run benchmark on a single problem with given config
fn benchmark_problem(
    problem: &GeometryProblem,
    tier: &str,
    config_name: &str,
    qa_weight: f64,
    geometric_weight: f64,
    use_coord_facts: bool,
) -> BenchmarkResult {
    let state = problem.to_state();
    let initial_state = state.clone();

    let config = BeamConfig {
        beam_width: 15,
        max_depth: 20,  // Increased for Tier 3
        max_states: 2000,  // Increased for harder problems
        scoring: search::ScoringConfig {
            geometric_weight,
            qa_weight,
            step_penalty: 0.1,
        },
    };

    let solver = BeamSolver::new(config);

    let start = Instant::now();
    let result = solver.solve(state);
    let elapsed = start.elapsed().as_millis();

    let proof_steps = result.proof.as_ref().map(|p| p.steps.len()).unwrap_or(0);

    // Extract telemetry from final state (or initial if unsolved)
    let final_state = if result.solved {
        // For solved problems, extract from goal state
        // (In real implementation, we'd track the final state from the solver)
        initial_state.clone()
    } else {
        initial_state.clone()
    };

    let qa_features = qa::extract_qa_features(&final_state);
    let qa_prior = qa::compute_qa_prior(&qa_features);

    BenchmarkResult {
        problem_id: problem.id.clone(),
        tier: tier.to_string(),
        difficulty: problem.difficulty.unwrap_or(1) as usize,
        config_name: config_name.to_string(),
        qa_weight,
        geometric_weight,
        use_coord_facts,
        solved: result.solved,
        successors_generated: result.successors_generated,
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
        coord_facts_added_total: 0,  // TODO: Track in solver
        coord_facts_used_in_proof: 0,  // TODO: Track in proof trace
    }
}

/// Compute summary statistics for a tier-config combination
fn compute_tier_summary(results: &[BenchmarkResult], tier: &str, config_name: &str, qa_weight: f64) -> TierSummary {
    let tier_results: Vec<_> = results.iter()
        .filter(|r| r.tier == tier && r.config_name == config_name)
        .collect();

    let problems_total = tier_results.len();
    let problems_solved = tier_results.iter().filter(|r| r.solved).count();
    let solve_rate_pct = (problems_solved as f64 / problems_total as f64) * 100.0;

    let states: Vec<usize> = tier_results.iter().map(|r| r.successors_generated).collect();
    let avg_states = states.iter().sum::<usize>() as f64 / problems_total as f64;
    let variance = states.iter()
        .map(|&s| (s as f64 - avg_states).powi(2))
        .sum::<f64>() / problems_total as f64;
    let std_states = variance.sqrt();

    let solved_results: Vec<_> = tier_results.iter().filter(|r| r.solved).collect();
    let avg_steps = if !solved_results.is_empty() {
        solved_results.iter().map(|r| r.proof_steps).sum::<usize>() as f64 / solved_results.len() as f64
    } else {
        0.0
    };

    let avg_time_ms = tier_results.iter().map(|r| r.time_ms).sum::<u128>() as f64 / problems_total as f64;
    let avg_qa_prior = tier_results.iter().map(|r| r.qa_prior_mean).sum::<f64>() / problems_total as f64;
    let avg_phase_entropy = tier_results.iter().map(|r| r.phase_entropy).sum::<f64>() / problems_total as f64;

    TierSummary {
        tier: tier.to_string(),
        config_name: config_name.to_string(),
        qa_weight,
        solve_rate_pct,
        problems_solved,
        problems_total,
        avg_states,
        std_states,
        avg_steps,
        avg_time_ms,
        avg_qa_prior,
        avg_phase_entropy,
    }
}

#[test]
fn test_week4_full_benchmark_tier0() {
    // Tier 0: Sanity checks (10 problems, 1-2 steps)
    let tier = "tier0";
    let problems = vec![
        "t0_p01_parallel_direct",
        "t0_p02_perp_to_parallel",
        "t0_p03_equality_direct",
        "t0_p04_angle_direct",
        "t0_p05_circle_direct",
        "t0_p06_collinear_direct",
        "t0_p07_parallel_perp_simple",
        "t0_p08_two_step_parallel",
        "t0_p09_simple_multi_goal",
        "t0_p10_trivial_sanity",
    ];

    let configs = vec![
        ("Geometry Only", 0.0, 1.0),
        ("QA 10%", 0.1, 0.9),
        ("QA 30%", 0.3, 0.7),
        ("QA 50%", 0.5, 0.5),
        ("QA 70%", 0.7, 0.3),
    ];

    run_tier_benchmark(tier, &problems, &configs);
}

#[test]
#[ignore]  // Run with --ignored for full suite
fn test_week4_full_benchmark_tier1() {
    // Tier 1: Basic complexity (15 problems, 3-4 steps)
    let tier = "tier1";
    let problems = vec![
        "t1_p01_parallel_chain_4",
        "t1_p02_mixed_parallel_perp",
        "t1_p03_equality_chain_4",
        "t1_p04_parallel_branching",
        "t1_p05_circle_chain",
        "t1_p06_multi_goal_simple",
        "t1_p07_perp_parallel_mixed",
        "t1_p08_angle_equality_chain",
        "t1_p09_collinear_transitivity",
        "t1_p10_parallel_with_distractors",
        "t1_p11_equality_branching",
        "t1_p12_mixed_circle_parallel",
        "t1_p13_perpendicular_chain",
        "t1_p14_angle_parallel_mixed",
        "t1_p15_complex_branching",
    ];

    let configs = vec![
        ("Geometry Only", 0.0, 1.0),
        ("QA 10%", 0.1, 0.9),
        ("QA 30%", 0.3, 0.7),
        ("QA 50%", 0.5, 0.5),
        ("QA 70%", 0.7, 0.3),
    ];

    run_tier_benchmark(tier, &problems, &configs);
}

fn run_tier_benchmark(tier: &str, problems: &[&str], configs: &[(& str, f64, f64)]) {
    let mut all_results = Vec::new();

    println!("\n🔬 Week 4 Benchmark: {} ({} problems)\n", tier.to_uppercase(), problems.len());

    for problem_name in problems {
        let problem = load_tier_problem(tier, problem_name);

        println!("📝 {} (difficulty: {})", problem.id, problem.difficulty.unwrap_or(1));

        for (config_name, qa_weight, geo_weight) in configs {
            let result = benchmark_problem(&problem, tier, config_name, *qa_weight, *geo_weight, false);

            println!("   {} | {} | Steps: {}, States: {}, Time: {}ms | QA Prior: {:.3}, Entropy: {:.3}",
                     config_name,
                     if result.solved { "✅" } else { "❌" },
                     result.proof_steps,
                     result.successors_generated,
                     result.time_ms,
                     result.qa_prior_mean,
                     result.phase_entropy);

            all_results.push(result);
        }
        println!();
    }

    // Generate tier summaries
    println!("\n📊 {} Summary Statistics:\n", tier.to_uppercase());

    for (config_name, qa_weight, _) in configs {
        let summary = compute_tier_summary(&all_results, tier, config_name, *qa_weight);

        println!("{} (QA weight: {:.1})", summary.config_name, summary.qa_weight);
        println!("   Solve Rate:     {:.1}% ({}/{})",
                 summary.solve_rate_pct, summary.problems_solved, summary.problems_total);
        println!("   Avg States:     {:.2} ± {:.2}", summary.avg_states, summary.std_states);
        println!("   Avg Steps:      {:.2}", summary.avg_steps);
        println!("   Avg Time:       {:.2}ms", summary.avg_time_ms);
        println!("   Avg QA Prior:   {:.3}", summary.avg_qa_prior);
        println!("   Avg Entropy:    {:.3}", summary.avg_phase_entropy);
        println!();
    }

    // Write CSV
    let csv_path = format!("benchmark_results_week4_{}.csv", tier);
    write_csv(&all_results, &csv_path);
    println!("💾 Results saved to: {}\n", csv_path);

    // Write JSON summary
    let json_path = format!("benchmark_summary_week4_{}.json", tier);
    write_json_summary(tier, configs, &all_results, &json_path);
    println!("💾 Summary saved to: {}\n", json_path);

    // Correctness assertion
    verify_correctness(&all_results, problems.len(), configs.len());
}

fn write_csv(results: &[BenchmarkResult], path: &str) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create CSV");

    // Header
    writeln!(file, "problem_id,tier,difficulty,config_name,qa_weight,geometric_weight,use_coord_facts,solved,successors_generated,depth_reached,proof_steps,time_ms,best_score,qa_prior_mean,phase_entropy,primitive_mass,female_mass,fermat_mass,mean_jk,mean_harmonic_index,num_candidates,qa_confidence")
        .expect("Failed to write CSV header");

    // Data rows
    for r in results {
        writeln!(file, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                 r.problem_id, r.tier, r.difficulty, r.config_name, r.qa_weight, r.geometric_weight,
                 r.use_coord_facts, r.solved, r.successors_generated, r.depth_reached, r.proof_steps,
                 r.time_ms, r.best_score, r.qa_prior_mean, r.phase_entropy, r.primitive_mass,
                 r.female_mass, r.fermat_mass, r.mean_jk, r.mean_harmonic_index,
                 r.num_candidates, r.qa_confidence)
            .expect("Failed to write CSV row");
    }
}

fn write_json_summary(tier: &str, configs: &[(&str, f64, f64)], results: &[BenchmarkResult], path: &str) {
    use std::fs::File;
    use std::io::Write;

    let summaries: Vec<TierSummary> = configs.iter()
        .map(|(name, qa_weight, _)| compute_tier_summary(results, tier, name, *qa_weight))
        .collect();

    let json = serde_json::to_string_pretty(&summaries).expect("Failed to serialize JSON");
    let mut file = File::create(path).expect("Failed to create JSON file");
    write!(file, "{}", json).expect("Failed to write JSON");
}

fn verify_correctness(results: &[BenchmarkResult], num_problems: usize, num_configs: usize) {
    // Group by problem
    let mut by_problem: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        by_problem.entry(result.problem_id.clone()).or_insert_with(Vec::new).push(result);
    }

    // Verify each problem has same solve status across all QA weights
    for (problem_id, problem_results) in by_problem.iter() {
        if problem_results.len() != num_configs {
            continue;  // Skip if incomplete
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
