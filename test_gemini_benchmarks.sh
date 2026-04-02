#!/bin/bash
# Gemini Benchmark Tests

echo "=== GEMINI BENCHMARK TESTS ==="
echo

# Test 1: Session Data Extraction
echo "Test 1: Session Data Extraction"
echo "Extract key events from: User asked about closeout procedure, OpenCode created session summary, saved to vault." | gemini > /tmp/gemini_test1.txt
echo "✓ Test 1 complete"
echo

# Test 2: Analysis and Validation
echo "Test 2: Analysis and Validation"
echo "Analyze this code for bugs: def qa_tuple(b,e): return (b,e,b+e,b+2*e). Identify any issues." | gemini > /tmp/gemini_test2.txt
echo "✓ Test 2 complete"
echo

# Test 3: Code Formatting (not ideal for Gemini but let's test)
echo "Test 3: Code Formatting"
echo "Format this code properly: def f(x,y):z=x+y;return z" | gemini > /tmp/gemini_test3.txt
echo "✓ Test 3 complete"
echo

# Test 4: Insight Synthesis
echo "Test 4: Insight Synthesis"
echo "Synthesize insights from these patterns: 1) User prefers Obsidian vault integration, 2) Multi-AI collaboration is valued, 3) Evidence-based decisions preferred. What does this tell us?" | gemini > /tmp/gemini_test4.txt
echo "✓ Test 4 complete"
echo

# Test 5: Markdown Generation
echo "Test 5: Markdown Generation"
echo "Generate an Obsidian-compatible markdown session summary with these elements: date 2025-10-30, topic 'BobNet design', participants Claude+OpenCode+Gemini, outcome 'role benchmarks complete'. Use proper formatting with headers, lists, and tags." | gemini > /tmp/gemini_test5.txt
echo "✓ Test 5 complete"

echo
echo "=== ALL GEMINI TESTS COMPLETE ==="
echo "Results saved to /tmp/gemini_test*.txt"
