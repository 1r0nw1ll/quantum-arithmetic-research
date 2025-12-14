# IR Module - Intermediate Representation for QA-AlphaGeometry

## Overview

The IR (Intermediate Representation) module provides the core data structures for representing geometric proofs in the QA-AlphaGeometry symbolic geometry solver. It implements a type-safe, efficient system for working with geometric objects, facts, proof states, and proof traces.

## Architecture

The module consists of four main components:

### 1. `symbols.rs` - Symbol Interning System

Provides type-safe newtype wrappers for geometric object identifiers:
- `PointId` - Points in the geometric construction
- `LineId` - Lines
- `CircleId` - Circles
- `SegmentId` - Line segments
- `AngleId` - Angles

**Key Features:**
- O(1) string interning and lookup via `FxHashMap`
- Automatic deduplication (same label → same ID)
- Thread-safe via `Arc<RwLock<>>`
- Bidirectional mapping (label ↔ ID)

**Example:**
```rust
let symbols = SymbolTable::new();
let a = symbols.get_or_intern_point("A");
let b = symbols.get_or_intern_point("B");
assert_eq!(symbols.point_label(a), Some("A".to_string()));
```

### 2. `facts.rs` - Atomic Geometric Facts

Defines 23 types of geometric facts as an enum:
- `Collinear(p1, p2, p3)` - Three points on same line
- `Parallel(l1, l2)` - Parallel lines
- `Perpendicular(l1, l2)` - Perpendicular lines
- `OnCircle(p, c)` - Point on circle
- `EqualLength(s1, s2)` - Equal segments
- `EqualAngle(a1, a2)` - Equal angles
- `Midpoint(p, s)` - Point is midpoint of segment
- `RightTriangle(p1, p2, p3)` - Right triangle
- ... and 15 more

**Key Features:**
- Automatic normalization to canonical form
- Symmetric predicates are order-independent
- Hash-based deduplication
- Type-based indexing for fast queries
- Provenance tracking (which step introduced each fact)

**Example:**
```rust
let mut facts = FactStore::new();
facts.insert(Fact::Parallel(l1, l2));
facts.insert(Fact::Parallel(l2, l1)); // Normalized to same fact
assert_eq!(facts.len(), 1); // Deduplicated
```

### 3. `state.rs` - Proof State Representation

Complete representation of a proof attempt:
- `GeoState` - Full state (symbols + facts + goal + metadata)
- `Goal` - Target facts to prove
- `Metadata` - Problem ID, diagram info, extras

**Key Features:**
- Goal satisfaction checking
- State hashing for deduplication
- Immutable updates (`with_fact()`)
- Fact filtering and queries

**Example:**
```rust
let state = GeoState::new(symbols, facts, goal);
state.add_fact(new_fact);
if state.is_goal_satisfied() {
    println!("Proof complete!");
}
```

### 4. `proof.rs` - Proof Steps and Traces

Represents complete proof traces:
- `ProofStep` - Single inference step (rule + premises → conclusions)
- `ProofTrace` - Ordered sequence of steps
- `ProofStatistics` - Aggregated statistics

**Key Features:**
- JSON serialization/deserialization
- Proof statistics (scores, rule usage)
- Step validation
- Metadata tracking

**Example:**
```rust
let mut trace = ProofTrace::new();
trace.add_step(ProofStep::new(
    ProofStepId(1),
    "parallel_transitivity",
    premises,
    conclusions,
    0.95
));
trace.mark_solved(state.hash());
let json = trace.to_json()?;
```

## Type Safety

All geometric objects use newtype wrappers to prevent mixing different types:

```rust
let point = PointId(1);
let line = LineId(1);

// Compile error: type mismatch
// Fact::Collinear(point, line, point);  ✗

// Correct: all points
Fact::Collinear(point1, point2, point3);  ✓
```

## Normalization

Facts are automatically normalized to canonical form:

```rust
Fact::Parallel(LineId(1), LineId(2)).normalize()
== Fact::Parallel(LineId(2), LineId(1)).normalize()

Fact::Collinear(p3, p1, p2).normalize()
== Fact::Collinear(p1, p2, p3).normalize()  // Sorted by ID
```

## Provenance Tracking

Facts can track which proof step introduced them:

```rust
facts.insert_with_provenance(fact, ProofStepId(42));
let step_id = facts.provenance(&fact); // Some(ProofStepId(42))
```

## Serialization

Proof traces support JSON serialization for verification and export:

```rust
let trace = ProofTrace::new();
// ... add steps ...

// Pretty-printed JSON
let json = trace.to_json()?;

// Compact JSON
let compact = trace.to_json_compact()?;

// Deserialize
let restored = ProofTrace::from_json(&json)?;
```

## Testing

The module includes comprehensive unit tests for:
- Symbol interning deduplication
- Fact normalization
- Fact hashing and equality
- FactStore deduplication
- Goal satisfaction checking
- ProofTrace serialization roundtrip
- Thread safety
- Type-based indexing

Run tests:
```bash
cargo test --lib ir::
```

## Performance Characteristics

- Symbol interning: O(1) average case
- Fact insertion: O(1) average case
- Fact lookup: O(1) average case
- Type-based queries: O(k) where k = facts of that type
- State hashing: O(n log n) where n = number of facts

## Dependencies

- `rustc-hash` - Fast FxHashMap for symbol tables
- `serde` / `serde_json` - Serialization
- `thiserror` - Error handling

## Future Extensions

Potential enhancements:
- Persistent data structures for efficient backtracking
- Incremental hashing for state deduplication
- Fact indexing by geometric objects (all facts involving point A)
- Proof compression and delta encoding
- Parallel fact insertion with lock-free structures

## Integration with QA System

The IR module integrates with the Quantum Arithmetic (QA) system via:
- Proof step scoring using QA harmonic metrics
- Geometric object encoding as QA tuples (future)
- Beam search state representation
- Rule prioritization based on QA alignment

## Example: Complete Workflow

```rust
use qa_alphageometry_core::ir::*;

// 1. Create symbol table
let symbols = SymbolTable::new();
let a = symbols.get_or_intern_point("A");
let b = symbols.get_or_intern_point("B");
let c = symbols.get_or_intern_point("C");

// 2. Add initial facts
let mut facts = FactStore::new();
facts.insert(Fact::Collinear(a, b, c));

// 3. Define goal
let l1 = symbols.get_or_intern_line("L1");
let l2 = symbols.get_or_intern_line("L2");
let goal = Goal::single(Fact::Parallel(l1, l2));

// 4. Create state
let mut state = GeoState::new(symbols, facts, goal);

// 5. Create proof trace
let mut trace = ProofTrace::new();

// 6. Apply reasoning steps
let step = ProofStep::new(
    ProofStepId(1),
    "axiom",
    vec![],
    vec![Fact::Parallel(l1, l2)],
    1.0
);
trace.add_step(step);
state.add_fact(Fact::Parallel(l1, l2));

// 7. Check success
if state.is_goal_satisfied() {
    trace.mark_solved(state.hash());
    println!("Proof complete!");
    println!("{}", trace.to_json()?);
}
```

## License

Same as parent project.
