# Blind Corpus Benchmark

This directory contains corpus-level benchmark tooling for the currently
implemented blind-eval domains.

Current domains:

- `tla_blind`
- `lean4_blind`

The benchmark sweep runs each domain's current-system executor, uses the hidden
labels as ground truth, and writes:

- a machine-readable JSON summary
- a Markdown report with confusion matrices and error analysis

It is intended to answer:

- how many labeled fixtures are correct vs wrong
- which specific fixtures are false accepts or false rejects
- whether a domain currently looks conservative, balanced, or over-trusting

