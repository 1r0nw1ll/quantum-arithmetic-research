#!/usr/bin/env python3
"""Analyze Phase 2 results with increased search budgets"""

import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('benchmark_results_week4_session3_tier2.csv')

print("=" * 80)
print("PHASE 2 ANALYSIS: Increased Search Budgets (max_states=8000, max_depth=35)")
print("=" * 80)

# Overall solve rate
total_runs = len(df)
total_solved = df['solved'].sum()
solve_rate = total_solved / total_runs * 100

print(f"\n📊 OVERALL METRICS:")
print(f"   Total runs: {total_runs}")
print(f"   Solved: {total_solved}")
print(f"   Solve rate: {solve_rate:.1f}%")

# Solve rate by problem
print(f"\n📝 SOLVE RATE BY PROBLEM:")
problem_groups = df.groupby('problem_id')
for problem_id, group in problem_groups:
    solved_count = group['solved'].sum()
    total_configs = len(group)
    rate = solved_count / total_configs * 100
    states_mean = group['states_explored'].mean()
    depth_mean = group['depth_reached'].mean()

    # Check for identical states
    states_unique = group['states_explored'].nunique()
    identical_states = "⚠️ IDENTICAL" if states_unique == 1 else f"✓ {states_unique} unique"

    print(f"   {problem_id:40s} | {rate:5.1f}% | Avg States: {states_mean:7.1f} | {identical_states}")

# Termination analysis
print(f"\n🛑 TERMINATION ANALYSIS:")
hit_max_states_count = df['hit_max_states'].sum()
hit_max_depth_count = df['hit_max_depth'].sum()
print(f"   Hit max_states limit: {hit_max_states_count}/{total_runs} runs ({hit_max_states_count/total_runs*100:.1f}%)")
print(f"   Hit max_depth limit: {hit_max_depth_count}/{total_runs} runs ({hit_max_depth_count/total_runs*100:.1f}%)")

# QA activation check
print(f"\n🔬 QA ACTIVATION:")
qa_activated = df[df['qa_prior_mean'] > 0]
print(f"   Runs with QA activation: {len(qa_activated)}/{total_runs} ({len(qa_activated)/total_runs*100:.1f}%)")
print(f"   Mean QA prior (when active): {qa_activated['qa_prior_mean'].mean():.3f}")
print(f"   Mean phase entropy (when active): {qa_activated['phase_entropy'].mean():.3f}")

# States explored variance by QA weight
print(f"\n🎯 STATES EXPLORED VARIANCE:")
for problem_id, group in problem_groups:
    states_std = group['states_explored'].std()
    states_mean = group['states_explored'].mean()
    cv = (states_std / states_mean * 100) if states_mean > 0 else 0

    if cv > 1.0:
        print(f"   {problem_id:40s} | CV={cv:5.2f}% ✓ QA affecting search")
    elif states_mean == 0:
        print(f"   {problem_id:40s} | ZERO STATES ⚠️")

# Check if problems that should be solvable are being solved
print(f"\n📈 COMPARISON TO GOAL:")
print(f"   Target solve rate: ≥90%")
print(f"   Actual solve rate: {solve_rate:.1f}%")
if solve_rate >= 90:
    print(f"   ✅ TARGET ACHIEVED - search reaches discriminative depth!")
else:
    print(f"   ❌ Below target - may need further budget increase or problem diagnosis")

# Identify problematic problems
print(f"\n⚠️ PROBLEMATIC PROBLEMS:")
for problem_id, group in problem_groups:
    solved_count = group['solved'].sum()
    states_mean = group['states_explored'].mean()

    if states_mean == 0:
        print(f"   {problem_id}: Zero states explored (immediate failure)")
    elif solved_count == 0:
        print(f"   {problem_id}: Never solved ({states_mean:.0f} states avg)")
