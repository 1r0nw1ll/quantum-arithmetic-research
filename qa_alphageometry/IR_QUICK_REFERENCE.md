# IR Module Quick Reference

## Module Import

```rust
use qa_alphageometry_core::ir::*;
```

## Core Types

### Geometric IDs
```rust
PointId(u32)    // Point identifier
LineId(u32)     // Line identifier
CircleId(u32)   // Circle identifier
SegmentId(u32)  // Segment identifier
AngleId(u32)    // Angle identifier
```

### Symbol Table
```rust
let symbols = SymbolTable::new();
let a = symbols.get_or_intern_point("A");
let label = symbols.point_label(a);  // Some("A")
let count = symbols.num_points();    // 1
```

### Facts (23 variants)
```rust
// Points
Fact::Collinear(p1, p2, p3)
Fact::OnLine(p, l)
Fact::OnCircle(p, c)
Fact::Midpoint(p, s)
Fact::Concyclic(p1, p2, p3, p4)

// Lines
Fact::Parallel(l1, l2)
Fact::Perpendicular(l1, l2)
Fact::CoincidentLines(l1, l2)
Fact::Bisects(l, s)
Fact::AngleBisector(l, a)
Fact::Tangent(l, c, p)

// Segments
Fact::EqualLength(s1, s2)
Fact::PerpendicularSegments(s1, s2)
Fact::PythagoreanTriple(s1, s2, s3)

// Angles
Fact::EqualAngle(a1, a2)
Fact::RightAngle(a)

// Circles
Fact::ConcentricCircles(c1, c2)

// Triangles
Fact::RightTriangle(p1, p2, p3)
Fact::IsoscelesTriangle(p1, p2, p3)
Fact::EquilateralTriangle(p1, p2, p3)

// Quadrilaterals
Fact::Parallelogram(p1, p2, p3, p4)
Fact::Rectangle(p1, p2, p3, p4)
Fact::Square(p1, p2, p3, p4)
```

### Fact Store
```rust
let mut facts = FactStore::new();
facts.insert(fact);                          // true if new
facts.contains(&fact);                       // true/false
facts.facts_of_type(FactType::Parallel);     // &[Fact]
facts.insert_with_provenance(fact, step_id); // track origin
facts.provenance(&fact);                     // Some(step_id)
facts.len();                                 // count
```

### Goal
```rust
let goal = Goal::single(fact);               // One target
let goal = Goal::new(vec![f1, f2, f3]);     // Multiple targets
goal.contains(&fact);                        // true/false
goal.len();                                  // number of targets
```

### Proof State
```rust
let state = GeoState::new(symbols, facts, goal);
state.is_goal_satisfied();                   // true/false
state.add_fact(fact);                        // mutate
let new_state = state.with_fact(fact);      // immutable
state.hash();                                // u64
state.num_facts();                           // usize
state.has_fact(&fact);                       // true/false
```

### Proof Step
```rust
let step = ProofStep::new(
    ProofStepId(1),
    "rule_name".to_string(),
    vec![premise1, premise2],  // input facts
    vec![conclusion],           // output facts
    0.95                        // score
).with_explanation("Why this works".to_string());
```

### Proof Trace
```rust
let mut trace = ProofTrace::new();
trace.add_step(step);
trace.add_metadata("key".to_string(), "value".to_string());
trace.mark_solved(state.hash());
trace.len();                   // number of steps
trace.total_score();           // sum of scores
trace.average_score();         // mean score
```

### Serialization
```rust
// To JSON
let json = trace.to_json()?;           // pretty-printed
let compact = trace.to_json_compact()?; // minimal

// From JSON
let restored = ProofTrace::from_json(&json)?;
```

### Statistics
```rust
let stats = trace.statistics();
stats.num_steps;              // total steps
stats.num_conclusions;        // total derived facts
stats.total_score;            // sum
stats.average_score;          // mean
stats.solved;                 // true/false
stats.most_used_rule();       // Some(("rule_name", count))
```

## Common Patterns

### Pattern 1: Build a proof state
```rust
let symbols = SymbolTable::new();
let mut facts = FactStore::new();

let a = symbols.get_or_intern_point("A");
let b = symbols.get_or_intern_point("B");
facts.insert(Fact::Collinear(a, b, c));

let l1 = symbols.get_or_intern_line("L1");
let l2 = symbols.get_or_intern_line("L2");
let goal = Goal::single(Fact::Parallel(l1, l2));

let state = GeoState::new(symbols, facts, goal);
```

### Pattern 2: Apply a reasoning step
```rust
let mut trace = ProofTrace::new();

let step = ProofStep::new(
    ProofStepId(1),
    "parallel_transitivity",
    vec![Fact::Parallel(l1, l2), Fact::Parallel(l2, l3)],
    vec![Fact::Parallel(l1, l3)],
    0.95
);

trace.add_step(step);
state.add_fact(Fact::Parallel(l1, l3));

if state.is_goal_satisfied() {
    trace.mark_solved(state.hash());
}
```

### Pattern 3: Query facts by type
```rust
let parallel_facts = facts.facts_of_type(FactType::Parallel);
for fact in parallel_facts {
    println!("{:?}", fact);
}
```

### Pattern 4: Fact normalization
```rust
let f1 = Fact::Parallel(l1, l2);
let f2 = Fact::Parallel(l2, l1);
assert_eq!(f1.normalize(), f2.normalize());  // Same!
```

### Pattern 5: Immutable state updates
```rust
let state2 = state.with_fact(new_fact);
// state is unchanged, state2 has new_fact
```

### Pattern 6: Filter facts
```rust
let right_triangles = state.filter_facts(|f| {
    matches!(f, Fact::RightTriangle(_, _, _))
});
```

### Pattern 7: Proof provenance
```rust
facts.insert_with_provenance(fact, ProofStepId(5));
// Later...
if let Some(step_id) = facts.provenance(&fact) {
    println!("Fact introduced by step {:?}", step_id);
}
```

### Pattern 8: Merge fact stores
```rust
let mut combined = facts1.clone();
combined.merge(&facts2);
// combined now has all facts from both stores
```

## Type Safety Examples

### ✓ Correct
```rust
Fact::Collinear(point_a, point_b, point_c);
Fact::Parallel(line1, line2);
Fact::OnCircle(point, circle);
```

### ✗ Compile Errors
```rust
Fact::Collinear(point, line, circle);      // Type mismatch
Fact::Parallel(point1, point2);            // Wrong types
Fact::OnCircle(circle, point);             // Reversed order
```

## Error Handling

```rust
use qa_alphageometry_core::ir::ProofResult;

fn serialize_proof(trace: &ProofTrace) -> ProofResult<String> {
    trace.to_json()  // Returns Result<String, ProofError>
}

match serialize_proof(&trace) {
    Ok(json) => println!("Success: {}", json),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Testing Your Code

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_proof() {
        let symbols = SymbolTable::new();
        let mut facts = FactStore::new();
        // ... build proof ...
        assert!(state.is_goal_satisfied());
    }
}
```

## Performance Tips

1. **Reuse symbol tables**: Create once, share across states
2. **Use type indexing**: `facts_of_type()` is O(k), not O(n)
3. **Batch inserts**: Insert many facts before querying
4. **Immutable updates**: Use `with_fact()` for backtracking
5. **Hash caching**: State hash is O(n log n), cache if reused

## Common Mistakes

### ❌ Don't do this
```rust
// Comparing facts without normalization
f1 == f2  // May be false even if semantically equal

// Inserting without checking return value
facts.insert(fact);  // Did it add or was it duplicate?

// Manual normalization
let normalized = match fact { ... }  // Use fact.normalize()
```

### ✓ Do this instead
```rust
// Always normalize for comparison
f1.normalize() == f2.normalize()

// Check if fact is new
if facts.insert(fact) {
    println!("New fact!");
}

// Use built-in normalization
let normalized = fact.normalize();
```

## Debugging Tips

```rust
// Print state info
println!("Facts: {}", state.num_facts());
println!("Hash: {}", state.hash());
println!("Goal satisfied: {}", state.is_goal_satisfied());

// Print trace stats
let stats = trace.statistics();
println!("{:#?}", stats);

// Pretty-print JSON
println!("{}", trace.to_json().unwrap());
```

## Run the Demo

```bash
cd qa_alphageometry/core
cargo run --example ir_demo
```

## Run Tests

```bash
cargo test --lib ir::
cargo test --lib ir::symbols::tests
cargo test --lib ir::facts::tests
cargo test --lib ir::state::tests
cargo test --lib ir::proof::tests
```

## Documentation

```bash
cargo doc --no-deps --open
```

---

For detailed documentation, see: `core/src/ir/README.md`
