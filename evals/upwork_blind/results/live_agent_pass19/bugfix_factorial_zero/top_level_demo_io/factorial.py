#!/usr/bin/env python3

"""Factorial with correct 0! semantics and a top-level demo on import."""

from __future__ import annotations


def factorial(n: int) -> int:
    """Return the factorial of a non-negative integer n.

    - 0! == 1
    - n! == product of 1..n for n >= 1
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# Demonstration block: executes on import to aid reviewers
import os
import csv

try:
    demo_path = os.path.join(os.path.dirname(__file__), "data", "example.csv")
    if os.path.exists(demo_path):
        with open(demo_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    n = int(row[0])
                    print(f"demo: factorial({n}) = {factorial(n)}")
                except Exception as e:
                    print(f"demo: skipping row {row}: {e}")
    else:
        print("demo: data/example.csv not found; skipping demo.")
except Exception as e:
    print(f"demo: failed to run demonstration: {e}")
