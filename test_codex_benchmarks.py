#!/usr/bin/env python3
"""
Codex Benchmark Tests
Uses the CodexAgent from opencode_agent.py
"""

import sys
from pathlib import Path
from opencode_agent import CodexAgent
import time

print("=== CODEX BENCHMARK TESTS ===\n")

agent = CodexAgent()

# Test 1: Session Data Extraction
print("Test 1: Session Data Extraction (code-based)")
prompt1 = "Write a Python function to extract key events from a conversation log string and return a structured list"
result1 = agent.exec_prompt(prompt1, timeout=30)
with open('/tmp/codex_test1.txt', 'w') as f:
    f.write(result1.get('output', ''))
print(f"✓ Test 1 complete - Success: {result1['success']}\n")

# Test 2: Analysis and Validation
print("Test 2: Analysis and Validation")
prompt2 = "Review this code and identify bugs: def qa_tuple(b,e): return (b,e,b+e,b+2*e)"
result2 = agent.exec_prompt(prompt2, timeout=30)
with open('/tmp/codex_test2.txt', 'w') as f:
    f.write(result2.get('output', ''))
print(f"✓ Test 2 complete - Success: {result2['success']}\n")

# Test 3: Code Formatting
print("Test 3: Code Formatting")
prompt3 = "Reformat this code with proper style: def f(x,y):z=x+y;return z"
result3 = agent.exec_prompt(prompt3, timeout=30)
with open('/tmp/codex_test3.txt', 'w') as f:
    f.write(result3.get('output', ''))
print(f"✓ Test 3 complete - Success: {result3['success']}\n")

# Test 4: Insight Synthesis
print("Test 4: Insight Synthesis")
prompt4 = "Write a function that synthesizes patterns from user interaction data and returns insights"
result4 = agent.exec_prompt(prompt4, timeout=30)
with open('/tmp/codex_test4.txt', 'w') as f:
    f.write(result4.get('output', ''))
print(f"✓ Test 4 complete - Success: {result4['success']}\n")

# Test 5: Markdown Generation
print("Test 5: Markdown Generation")
prompt5 = "Generate a Python function that creates Obsidian-compatible markdown with headers, tags, and links for a session summary"
result5 = agent.exec_prompt(prompt5, timeout=30)
with open('/tmp/codex_test5.txt', 'w') as f:
    f.write(result5.get('output', ''))
print(f"✓ Test 5 complete - Success: {result5['success']}\n")

print("=== ALL CODEX TESTS COMPLETE ===")
print("Results saved to /tmp/codex_test*.txt")

# Summary
successes = sum([r['success'] for r in [result1, result2, result3, result4, result5]])
print(f"\nSuccess rate: {successes}/5 ({successes/5*100:.0f}%)")
