Produce a small TLA+ artifact bundle for an outsider audience.

System to model:

- A single bounded counter with values `0..3`
- `Init` starts at `0`
- `Inc` increments when below the bound
- `Reset` returns the counter to `0`

Required output:

- one TLA+ module
- a short README for TLA+ readers
- a variable/action justification section
- a semantics-vs-bounds explanation
- a short note explaining why this belongs in `tlaplus/examples`

Constraints:

- Do not assume the reader knows any project-private vocabulary.
- Make intrinsic semantics distinct from TLC finiteness choices.
- Prefer simple, non-vacuous invariants.
