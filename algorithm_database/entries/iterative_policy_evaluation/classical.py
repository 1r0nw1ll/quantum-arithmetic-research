"""Classical iterative policy evaluation — Python port of Kochenderfer 2022 DM Algorithm 7.3.

QA_COMPLIANCE = "classical-baseline port — continuous-valued utility function on a discrete state space; observer-projection at output if used to compare to QA-discrete tier-classification (cert [191]). Not a QA-causal algorithm."

Source: (Kochenderfer 2022) Algorithms for Decision Making, MIT Press, §7.2 + Algorithm 7.3.
Transcribed from QA-MEM verbatim excerpt at
docs/theory/kochenderfer_decision_making_excerpts.md#dm-7-2-policy-evaluation-lookahead-equation
"""

from __future__ import annotations
from typing import Callable


def lookahead(
    states: list,
    transition: Callable[[object, object, object], float],
    reward: Callable[[object, object], float],
    gamma: float,
    U: dict,
    s,
    a,
) -> float:
    """Bellman lookahead for one (state, action) pair.

    Returns: R(s, a) + γ Σ_{s'} T(s' | s, a) * U(s').
    """
    return reward(s, a) + gamma * sum(
        transition(s, a, sp) * U[sp] for sp in states
    )


def iterative_policy_evaluation(
    states: list,
    transition: Callable[[object, object, object], float],
    reward: Callable[[object, object], float],
    gamma: float,
    policy: Callable[[object], object],
    k_max: int,
) -> dict:
    """Iterative policy evaluation. Computes U^π via the Bellman expectation
    equation `U^π_{k+1}(s) = R(s, π(s)) + γ Σ T(s'|s, π(s)) U^π_k(s')`.

    Returns: {state: float utility} dict after k_max iterations.

    Per DM §7.2: the update is a contraction mapping; convergence is
    guaranteed for γ ∈ [0, 1) and finite rewards.
    """
    U = {s: 0.0 for s in states}
    for _ in range(k_max):
        U = {
            s: lookahead(states, transition, reward, gamma, U, s, policy(s))
            for s in states
        }
    return U


if __name__ == "__main__":
    # Tiny self-test: 3-state Markov chain, deterministic transitions, fixed policy
    states = ["A", "B", "C"]

    def policy(s):
        return "advance"

    def transition(s, a, sp):
        # deterministic A → B → C → C
        nxt = {"A": "B", "B": "C", "C": "C"}
        return 1.0 if sp == nxt[s] else 0.0

    def reward(s, a):
        return 1.0 if s == "C" else 0.0

    U = iterative_policy_evaluation(states, transition, reward, 0.9, policy, 50)
    print(f"U(A)={U['A']:.4f}, U(B)={U['B']:.4f}, U(C)={U['C']:.4f}")
    # Expected (γ=0.9, terminal reward at C): U(C) → 1/(1-0.9) = 10; U(B) → 0.9*10 = 9; U(A) → 0.9*9 = 8.1
