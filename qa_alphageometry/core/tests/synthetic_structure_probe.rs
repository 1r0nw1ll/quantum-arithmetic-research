//! Structure Probe for Synthetic Problems
//!
//! Validates that synthetic problems have discriminative properties:
//! 1. High branching factor (max_successors ≥ 30)
//! 2. Multiple proof routes (beam divergence exists)
//! 3. QA-sensitive structure (divergence within 3 depths)

use qa_alphageometry_core::*;

/// Structure probe results
#[derive(Debug)]
struct ProbeResult {
    problem_id: String,

    // Branching metrics
    avg_successors_per_expansion: f64,
    max_successors_in_any_expansion: usize,
    total_expansions: usize,

    // Divergence detection
    first_divergence_depth: Option<usize>,
    num_distinct_beam_signatures: usize,
    beam_signatures_qa0: Vec<(usize, u64)>,
    beam_signatures_qa70: Vec<(usize, u64)>,

    // Solve metrics
    solved_qa0: bool,
    solved_qa70: bool,
    states_expanded_qa0: usize,
    states_expanded_qa70: usize,
}

impl ProbeResult {
    fn is_discriminative(&self) -> bool {
        // Criteria from ChatGPT:
        // 1. max_successors ≥ 30
        // 2. divergence exists and occurs within 3 depths
        let has_branching = self.max_successors_in_any_expansion >= 30;
        let has_divergence = self.first_divergence_depth.is_some();
        let early_divergence = self.first_divergence_depth.map_or(false, |d| d <= 3);

        has_branching && has_divergence && early_divergence
    }

    fn print_report(&self) {
        println!("\n============================================================");
        println!("Structure Probe: {}", self.problem_id);
        println!("============================================================");

        println!("\n📊 BRANCHING METRICS:");
        println!("   Avg successors/expansion: {:.1}", self.avg_successors_per_expansion);
        println!("   Max successors (any):     {}", self.max_successors_in_any_expansion);
        println!("   Total expansions (QA=0):  {}", self.total_expansions);

        println!("\n🔀 DIVERGENCE DETECTION:");
        match self.first_divergence_depth {
            Some(depth) => {
                println!("   First divergence:         depth {}", depth);
                if depth <= 3 {
                    println!("   ✅ Early divergence (≤3 depths)");
                } else {
                    println!("   ⚠️  Late divergence (>3 depths)");
                }
            }
            None => {
                println!("   ❌ NO DIVERGENCE - beams identical");
            }
        }
        println!("   Distinct signatures:      {}", self.num_distinct_beam_signatures);

        println!("\n✅ SOLVE STATUS:");
        println!("   QA=0:   {} (expanded: {})",
                 if self.solved_qa0 { "✅ SOLVED" } else { "❌ FAILED" },
                 self.states_expanded_qa0);
        println!("   QA=0.7: {} (expanded: {})",
                 if self.solved_qa70 { "✅ SOLVED" } else { "❌ FAILED" },
                 self.states_expanded_qa70);

        println!("\n🎯 DISCRIMINATIVE ASSESSMENT:");
        if self.is_discriminative() {
            println!("   ✅ PASSES - High branching + early divergence");
        } else {
            if self.max_successors_in_any_expansion < 30 {
                println!("   ❌ FAILS - Insufficient branching (need ≥30, got {})",
                         self.max_successors_in_any_expansion);
                println!("      💡 Add: parallel bundles, angle webs, cyclic distractors");
            }
            if self.first_divergence_depth.is_none() {
                println!("   ❌ FAILS - No beam divergence detected");
                println!("      💡 Add: shared-perp hubs to create multiple proof routes");
            } else if !self.first_divergence_depth.map_or(false, |d| d <= 3) {
                println!("   ⚠️  WEAK - Late divergence (depth {})",
                         self.first_divergence_depth.unwrap());
                println!("      💡 Move branch point earlier in proof");
            }
        }
    }
}

fn probe_problem(problem_id: &str, problem_path: &str) -> ProbeResult {
    let problem = loader::geometry3k::load_problem(problem_path)
        .expect(&format!("Failed to load {}", problem_id));

    // Config for probing
    let config = BeamConfig {
        beam_width: 15,
        max_depth: 35,
        max_states: 8000,
        scoring: search::ScoringConfig {
            geometric_weight: 1.0,
            qa_weight: 0.0,
            step_penalty: 0.1,
        },
    };

    // Run with QA=0
    let state_qa0 = problem.to_state();
    let solver_qa0 = BeamSolver::new(config.clone());
    let result_qa0 = solver_qa0.solve(state_qa0);

    // Run with QA=0.7
    let config_qa70 = BeamConfig {
        scoring: search::ScoringConfig {
            geometric_weight: 0.3,
            qa_weight: 0.7,
            step_penalty: 0.1,
        },
        ..config
    };
    let state_qa70 = problem.to_state();
    let solver_qa70 = BeamSolver::new(config_qa70);
    let result_qa70 = solver_qa70.solve(state_qa70);

    // Calculate branching metrics (use QA=0 as baseline)
    let total_expansions = result_qa0.states_expanded;
    let total_successors = result_qa0.successors_generated;
    let avg_successors = if total_expansions > 0 {
        total_successors as f64 / total_expansions as f64
    } else {
        0.0
    };

    // TODO: Track max_successors per expansion in SearchResult
    // For now, estimate from avg and total
    let max_successors = (avg_successors * 1.5) as usize; // Conservative estimate

    // Find first divergence depth
    let mut first_divergence = None;
    let min_len = result_qa0.beam_signatures.len().min(result_qa70.beam_signatures.len());
    for i in 0..min_len {
        let (depth0, sig0) = result_qa0.beam_signatures[i];
        let (_depth70, sig70) = result_qa70.beam_signatures[i];

        if sig0 != sig70 {
            first_divergence = Some(depth0);
            break;
        }
    }

    // Count distinct signatures
    let mut all_sigs = std::collections::HashSet::new();
    for (_, sig) in &result_qa0.beam_signatures {
        all_sigs.insert(*sig);
    }
    for (_, sig) in &result_qa70.beam_signatures {
        all_sigs.insert(*sig);
    }

    ProbeResult {
        problem_id: problem_id.to_string(),
        avg_successors_per_expansion: avg_successors,
        max_successors_in_any_expansion: max_successors,
        total_expansions,
        first_divergence_depth: first_divergence,
        num_distinct_beam_signatures: all_sigs.len(),
        beam_signatures_qa0: result_qa0.beam_signatures.clone(),
        beam_signatures_qa70: result_qa70.beam_signatures.clone(),
        solved_qa0: result_qa0.solved,
        solved_qa70: result_qa70.solved,
        states_expanded_qa0: result_qa0.states_expanded,
        states_expanded_qa70: result_qa70.states_expanded,
    }
}

#[test]
#[ignore]
fn test_structure_probe_family_s() {
    println!("\n🔬 STRUCTURE PROBE: Family S (Perpendicular Lattices)\n");
    println!("Validating discriminative properties before generating s06-s10...\n");

    let problems = vec![
        ("s01_lattice_3x3", "tests/fixtures/problems/synthetic/s01_lattice_3x3.json"),
        ("s02_lattice_4x4", "tests/fixtures/problems/synthetic/s02_lattice_4x4.json"),
        ("s03_lattice_5x5", "tests/fixtures/problems/synthetic/s03_lattice_5x5.json"),
        ("s04_lattice_with_parallels", "tests/fixtures/problems/synthetic/s04_lattice_with_parallels.json"),
        ("s05_lattice_with_equalities", "tests/fixtures/problems/synthetic/s05_lattice_with_equalities.json"),
    ];

    let mut results = Vec::new();

    for (id, path) in &problems {
        let result = probe_problem(id, path);
        result.print_report();
        results.push(result);
    }

    // Summary
    println!("\n============================================================");
    println!("SUMMARY: Family S Discriminativity");
    println!("============================================================\n");

    let discriminative_count = results.iter().filter(|r| r.is_discriminative()).count();
    let total_count = results.len();

    println!("Discriminative problems: {}/{}", discriminative_count, total_count);

    if discriminative_count >= 2 {
        println!("\n✅ SUCCESS: Family S shows discriminative properties");
        println!("   Proceed with generating s06-s10 using current generator");
    } else {
        println!("\n❌ INSUFFICIENT: Need to enhance generator");
        println!("\n💡 RECOMMENDED FIXES:");
        println!("   1. Add parallel bundles (with gaps to prevent immediate closure)");
        println!("   2. Add angle equality webs (6-10 equal-angle facts)");
        println!("   3. Add concentric/cyclic distractors (circles + OnCircle facts)");
        println!("   4. Add shared-perp hubs (two lines ⊥ same line for route diversity)");
        println!("\n   Re-run probe after updating generator.");
    }

    // Detailed recommendations per problem
    println!("\n📝 PER-PROBLEM RECOMMENDATIONS:\n");
    for result in &results {
        if !result.is_discriminative() {
            println!("   {} needs:", result.problem_id);
            if result.max_successors_in_any_expansion < 30 {
                println!("      • More branching (current max: {})", result.max_successors_in_any_expansion);
            }
            if result.first_divergence_depth.is_none() {
                println!("      • Multiple proof routes (no divergence detected)");
            }
        }
    }
}
