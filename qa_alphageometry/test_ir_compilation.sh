#!/bin/bash
# Test script for IR module compilation and testing

set -e

cd /home/player2/signal_experiments/qa_alphageometry/core

echo "=== Building QA-AlphaGeometry Core ==="
cargo build --release

echo ""
echo "=== Running tests ==="
cargo test --lib -- --nocapture

echo ""
echo "=== Running tests with verbose output ==="
cargo test --lib ir:: -- --nocapture

echo ""
echo "=== Checking for warnings ==="
cargo clippy -- -D warnings

echo ""
echo "=== Generating documentation ==="
cargo doc --no-deps

echo ""
echo "✓ All tests passed successfully!"
