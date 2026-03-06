#!/usr/bin/env python3
"""
solver.py  --  Deterministic constraint-energy minimizer for Sudoku.
Stdlib only: random, json, hashlib.
"""
import hashlib, json, random
from typing import Any, Dict, List, Optional

def _row_indices(r):
    return [r*9+c for c in range(9)]
def _col_indices(c):
    return [r*9+c for r in range(9)]
def _box_indices(r, c):
    br, bc = (r//3)*3, (c//3)*3
    return [(br+dr)*9+(bc+dc) for dr in range(3) for dc in range(3)]

def _group_violation_count(grid, indices):
    count = 0
    for v in range(1,10):
        n = sum(1 for i in indices if grid[i]==v)
        if n>1: count += n*(n-1)//2
    return count

def _compute_energy(grid):
    total = 0
    for r in range(9):
        total += _group_violation_count(grid, _row_indices(r))
        total += _group_violation_count(grid, _col_indices(r))
        br, bc = (r//3)*3, (r%3)*3
        total += _group_violation_count(grid, _box_indices(br,bc))
    return total

_CELL_GROUPS = []
for _i in range(81):
    _r, _c = _i//9, _i%9
    _CELL_GROUPS.append([_row_indices(_r), _col_indices(_c), _box_indices(_r,_c)])

def _local_violation_count(grid, idx):
    val = grid[idx]
    return sum(sum(1 for j in grp if j!=idx and grid[j]==val) for grp in _CELL_GROUPS[idx])

def _energy_delta(grid, idx, new_val):
    old_val = grid[idx]
    if old_val==new_val: return 0
    delta = 0
    for grp in _CELL_GROUPS[idx]:
        others = [grid[j] for j in grp if j!=idx]
        delta -= others.count(old_val)
        delta += others.count(new_val)
    return delta

def solve(givens, seed, max_steps):
    if len(givens)!=81 or not all(isinstance(v,int) and 0<=v<=9 for v in givens):
        raise ValueError('givens must be a list of 81 integers in [0,9]')
    rng = random.Random(seed)
    grid = list(givens)
    given_mask = [v!=0 for v in givens]
    for i in range(81):
        if not given_mask[i]:
            grid[i] = rng.randint(1,9)
    energy = _compute_energy(grid)
    energy_at_steps = []
    steps_taken = 0
    stuck_count = 0
    while energy>0 and steps_taken<max_steps:
        if steps_taken%100==0:
            energy_at_steps.append(energy)
        best_idx, best_viol = -1, -1
        for i in range(81):
            if given_mask[i]: continue
            v = _local_violation_count(grid,i)
            if v>best_viol:
                best_viol=v; best_idx=i
        if best_idx==-1 or best_viol==0:
            stuck_count+=1
            if stuck_count>=3: break
            steps_taken+=1; continue
        cur_val = grid[best_idx]
        best_delta, best_digit = 0, cur_val
        for d in range(1,10):
            delta = _energy_delta(grid,best_idx,d)
            if delta<best_delta or (delta==best_delta and d<best_digit):
                best_delta=delta; best_digit=d
        if best_delta>=0:
            stuck_count+=1
            if stuck_count>=3: break
        else:
            stuck_count=0
            grid[best_idx]=best_digit
            energy+=best_delta
        steps_taken+=1
    energy = _compute_energy(grid)
    energy_at_steps.append(energy)
    status = 'SOLVED' if energy==0 else 'CONTRADICTION_DETECTED'
    solution = list(grid) if energy==0 else None
    trace_json = json.dumps(energy_at_steps, separators=(',',':'))
    trace_hash = hashlib.sha256(trace_json.encode('utf-8')).hexdigest()
    return {'status':status,'final_energy':energy,'steps_taken':steps_taken,
            'solution':solution,'energy_at_steps':energy_at_steps,'trace_hash':trace_hash}
