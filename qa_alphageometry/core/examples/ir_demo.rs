//! Demonstration of the IR module functionality
//!
//! Run with: cargo run --example ir_demo

use qa_alphageometry_core::ir::*;

fn main() {
    println!("=== QA-AlphaGeometry IR Module Demo ===\n");

    // 1. Symbol Table Demo
    println!("1. Symbol Interning:");
    let symbols = SymbolTable::new();
    let a = symbols.get_or_intern_point("A");
    let b = symbols.get_or_intern_point("B");
    let c = symbols.get_or_intern_point("C");
    let d = symbols.get_or_intern_point("D");

    println!("   Created points: A={:?}, B={:?}, C={:?}, D={:?}", a, b, c, d);
    println!("   Point A label: {:?}", symbols.point_label(a));
    println!("   Total points: {}\n", symbols.num_points());

    // 2. Facts Demo
    println!("2. Geometric Facts:");
    let mut facts = FactStore::new();

    let l1 = symbols.get_or_intern_line("AB");
    let l2 = symbols.get_or_intern_line("CD");
    let l3 = symbols.get_or_intern_line("EF");

    facts.insert(Fact::Collinear(a, b, c));
    facts.insert(Fact::Parallel(l1, l2));
    facts.insert(Fact::Parallel(l2, l3));
    facts.insert(Fact::Perpendicular(l1, l3));

    println!("   Total facts: {}", facts.len());
    println!("   Parallel facts: {}", facts.facts_of_type(FactType::Parallel).len());
    println!("   Contains Parallel(AB, CD): {}\n", facts.contains(&Fact::Parallel(l1, l2)));

    // 3. Goal Demo
    println!("3. Proof Goals:");
    let goal = Goal::single(Fact::Parallel(l1, l3));
    println!("   Goal: Prove that line AB is parallel to line EF");
    println!("   Number of target facts: {}\n", goal.len());

    // 4. State Demo
    println!("4. Proof State:");
    let mut state = GeoState::new(symbols.clone(), facts.clone(), goal);
    println!("   Initial facts: {}", state.num_facts());
    println!("   Goal satisfied: {}", state.is_goal_satisfied());
    println!("   State hash: {}", state.hash());

    // Add the goal fact
    state.add_fact(Fact::Parallel(l1, l3));
    println!("   After adding goal fact:");
    println!("   Total facts: {}", state.num_facts());
    println!("   Goal satisfied: {}\n", state.is_goal_satisfied());

    // 5. Proof Trace Demo
    println!("5. Proof Trace:");
    let mut trace = ProofTrace::new();
    trace.add_metadata("solver".to_string(), "qa-alphageometry".to_string());
    trace.add_metadata("version".to_string(), "1.0".to_string());

    // Add proof steps
    let step1 = ProofStep::new(
        ProofStepId(1),
        "parallel_transitivity".to_string(),
        vec![Fact::Parallel(l1, l2), Fact::Parallel(l2, l3)],
        vec![Fact::Parallel(l1, l3)],
        0.95,
    ).with_explanation("Parallel lines are transitive".to_string());

    trace.add_step(step1);
    trace.mark_solved(state.hash());

    println!("   Proof steps: {}", trace.len());
    println!("   Proof solved: {}", trace.solved);
    println!("   Total score: {:.2}", trace.total_score());
    println!("   Average score: {:.2}", trace.average_score());

    // 6. Serialization Demo
    println!("\n6. JSON Serialization:");
    match trace.to_json() {
        Ok(json) => {
            println!("   Serialized proof trace ({} bytes)", json.len());
            println!("   First 200 chars: {}", &json[..json.len().min(200)]);

            // Deserialize
            match ProofTrace::from_json(&json) {
                Ok(restored) => {
                    println!("   ✓ Successfully deserialized");
                    println!("   Restored trace has {} steps", restored.len());
                }
                Err(e) => println!("   ✗ Deserialization failed: {}", e),
            }
        }
        Err(e) => println!("   ✗ Serialization failed: {}", e),
    }

    // 7. Statistics Demo
    println!("\n7. Proof Statistics:");
    let stats = trace.statistics();
    println!("   Steps: {}", stats.num_steps);
    println!("   Conclusions: {}", stats.num_conclusions);
    println!("   Total score: {:.2}", stats.total_score);
    println!("   Average score: {:.2}", stats.average_score);
    println!("   Solved: {}", stats.solved);
    if let Some((rule, count)) = stats.most_used_rule() {
        println!("   Most used rule: {} ({} times)", rule, count);
    }

    println!("\n=== Demo Complete ===");
}
