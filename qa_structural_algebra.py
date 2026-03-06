"""QA Structural Algebra Implementation.

This module implements operations sigma, mu, and R on pairs of natural numbers,
computes gcd invariants, converts between words in {L,R} and states, and
enumerates reachable states up to a bound.

Functions return dictionaries representing either success or failure. On success,
the dictionary has ``{"ok": True, "value": ...}``. On failure, it has
``{"ok": False, "fail_type": <str>, "invariant_diff": { ... }, "details": { ... }}``.

Examples
--------
>>> sigma((2, 3))['value']
(2, 5)
>>> mu((2, 3))['value']
(3, 2)
>>> R((2, 3))['value']
(5, 3)
>>> gcd_invariant((9, 6))['value']
3
>>> word_to_state('RL', seed=(1,1))['value']
(2, 3)
>>> state_to_word((2, 3))['value']
'RL'
"""

import math
import random
from collections import deque
from typing import Dict, Tuple, Set, List, Optional


def _success(value):
    """Helper to construct a success result."""
    return {"ok": True, "value": value}


def _failure(fail_type, invariant_diff=None, details=None):
    """Helper to construct a failure result with optional metadata."""
    return {
        "ok": False,
        "fail_type": fail_type,
        "invariant_diff": invariant_diff or {},
        "details": details or {},
    }


def _check_state(state: Tuple[int, int]):
    """Internal helper to validate that state is a pair of positive integers.

    Returns None if valid, else returns a failure dict.
    """
    if not isinstance(state, (tuple, list)) or len(state) != 2:
        return _failure("NOT_IN_NATURAL_DOMAIN", details={"state": state})
    b, e = state
    try:
        b_int = int(b)
        e_int = int(e)
    except Exception:
        return _failure("NOT_IN_NATURAL_DOMAIN", details={"state": state})
    if b_int <= 0 or e_int <= 0:
        return _failure("NOT_IN_NATURAL_DOMAIN", details={"state": state})
    return None  # indicates valid


def sigma(state: Tuple[int, int]) -> Dict:
    """
    Apply sigma: (b, e) -> (b, e + b).

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": (b, e+b)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> sigma((2, 3))['value']
    (2, 5)
    >>> sigma((0, 1))['ok']
    False
    """
    err = _check_state(state)
    if err:
        return err
    b, e = state
    b, e = int(b), int(e)
    new_state = (b, e + b)
    return _success(new_state)


def mu(state: Tuple[int, int]) -> Dict:
    """
    Apply mu: (b, e) -> (e, b).

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": (e, b)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> mu((2, 3))['value']
    (3, 2)
    >>> mu((-1, 2))['ok']
    False
    """
    err = _check_state(state)
    if err:
        return err
    b, e = state
    b, e = int(b), int(e)
    return _success((e, b))


def R(state: Tuple[int, int]) -> Dict:
    """
    Apply R: μ σ μ on (b,e) gives (b+e, e).

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": (b+e, e)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> R((2, 3))['value']
    (5, 3)
    >>> R((3, 3))['value']
    (6, 3)
    """
    err = _check_state(state)
    if err:
        return err
    b, e = state
    b, e = int(b), int(e)
    return _success((b + e, e))


def gcd_invariant(state: Tuple[int, int]) -> Dict:
    """
    Compute gcd(b,e).

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": gcd(b,e)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> gcd_invariant((9, 6))['value']
    3
    >>> gcd_invariant((0, 6))['ok']
    False
    """
    err = _check_state(state)
    if err:
        return err
    b, e = state
    b, e = int(b), int(e)
    return _success(math.gcd(b, e))

def lambda_k(state: Tuple[int, int], k: int) -> Dict:
    """
    Scale a state by an integer k ≥ 1.

    λₖ multiplies both coordinates by k.  The input state must lie in the natural domain
    and k must be an integer ≥1.  On success it returns the scaled state.

    Parameters
    ----------
    state : tuple of (b,e)
    k : int
        Scaling factor.  Must be an integer ≥1.

    Returns
    -------
    dict
        {"ok": True, "value": (k*b, k*e)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> lambda_k((2, 3), 2)['value']
    (4, 6)
    >>> lambda_k((2, 3), 1)['value']
    (2, 3)
    >>> lambda_k((2, 3), 0)['ok']
    False
    """
    # Validate state
    err = _check_state(state)
    if err:
        return err
    # Validate k
    if not isinstance(k, int) or k < 1:
        return _failure("NOT_IN_NATURAL_DOMAIN", details={"k": k})
    b, e = state
    b, e = int(b), int(e)
    return _success((b * k, e * k))


def nu(state: Tuple[int, int]) -> Dict:
    """
    Halving operator ν: divides both coordinates by 2 when allowed.

    The operation is defined only when both coordinates are even.  If either coordinate
    is odd, the operation fails with fail_type 'ODD_BLOCK'.

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": (b//2, e//2)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> nu((4, 6))['value']
    (2, 3)
    >>> nu((2, 3))['ok']
    False
    """
    err = _check_state(state)
    if err:
        return err
    b, e = state
    b, e = int(b), int(e)
    if (b % 2) != 0 or (e % 2) != 0:
        # Provide parity information in details
        return _failure(
            "ODD_BLOCK",
            details={"state": (b, e), "b_even": (b % 2 == 0), "e_even": (e % 2 == 0)},
        )
    return _success((b // 2, e // 2))


def component_gcd(state: Tuple[int, int]) -> Dict:
    """
    Compute the greatest common divisor of the coordinates of a state.

    This is an alias for gcd_invariant and returns the same value.

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": gcd(b,e)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> component_gcd((4, 6))['value']
    2
    >>> component_gcd((5, 7))['value']
    1
    """
    return gcd_invariant(state)


def normalize_to_coprime(state: Tuple[int, int]) -> Dict:
    """
    Normalize a state to a coprime representative and return its scale.

    Given a state (b,e), this function divides both coordinates by their greatest
    common divisor g to obtain a coprime pair (b',e') = (b/g, e/g) and returns
    both the normalized pair and the scale g.  The operation requires that the
    input lie in the natural domain.

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": ((b//g, e//g), g)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> normalize_to_coprime((4, 6))['value']
    ((2, 3), 2)
    >>> normalize_to_coprime((2, 3))['value']
    ((2, 3), 1)
    """
    err = _check_state(state)
    if err:
        return err
    b, e = state
    b, e = int(b), int(e)
    g = math.gcd(b, e)
    # g is at least 1 since state is positive
    return _success(((b // g, e // g), g))


def state_to_word_with_scale(state: Tuple[int, int]) -> Dict:
    """
    Compute the LR-word for the normalized version of a state and return its scale.

    If the state is not coprime, this function first normalizes the pair to (b',e') by dividing
    by its gcd g, computes the unique word for (b',e') using state_to_word, and returns the word
    together with the scale g and the normalized pair.  If the state is already coprime, the scale
    is 1 and the normalized pair is the state itself.

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": {"word": word, "scale": g, "normalized": (b',e')}} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> state_to_word_with_scale((4, 6))['value']
    {'word': 'RL', 'scale': 2, 'normalized': (2, 3)}
    >>> state_to_word_with_scale((2, 3))['value']
    {'word': 'RL', 'scale': 1, 'normalized': (2, 3)}
    """
    err = _check_state(state)
    if err:
        return err
    b, e = state
    b, e = int(b), int(e)
    g = math.gcd(b, e)
    if g == 1:
        # Already coprime
        res_word = state_to_word((b, e))
        if not res_word["ok"]:
            return res_word
        return _success({"word": res_word["value"], "scale": 1, "normalized": (b, e)})
    # Normalize
    b_norm, e_norm = b // g, e // g
    res_word = state_to_word((b_norm, e_norm))
    if not res_word["ok"]:
        return res_word
    return _success({"word": res_word["value"], "scale": g, "normalized": (b_norm, e_norm)})


def word_to_state(word: str, seed: Tuple[int, int] = (1, 1)) -> Dict:
    """
    Given a word over the alphabet {L,R}, apply the corresponding operations
    to the seed state.

    ``"L"`` applies sigma, ``"R"`` applies R.

    Parameters
    ----------
    word : str
        A string consisting solely of 'L' and 'R'.
    seed : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": (b,e)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    The example below uses ``'RL'`` rather than ``'LR'`` because applying
    ``R`` then ``L`` maps the seed ``(1,1)`` to ``(2,3)``.  Each character is
    applied in sequence: ``R`` sends ``(b,e)`` to ``(b+e,e)``, and
    ``L`` sends ``(b,e)`` to ``(b,e+b)``.

    >>> word_to_state('RL', seed=(1,1))['value']
    (2, 3)
    >>> word_to_state('RR', seed=(1,1))['value']
    (3, 1)
    >>> word_to_state('X', seed=(1,1))['ok']
    False
    """
    err = _check_state(seed)
    if err:
        return err
    b, e = seed
    b, e = int(b), int(e)
    for i, char in enumerate(word):
        if char == 'L':
            res = sigma((b, e))
            if not res["ok"]:
                return res
            b, e = res["value"]
        elif char == 'R':
            res = R((b, e))
            if not res["ok"]:
                return res
            b, e = res["value"]
        else:
            return _failure(
                "NOT_IN_NATURAL_DOMAIN",
                details={"word": word, "position": i},
            )
    return _success((b, e))


def state_to_word(state: Tuple[int, int]) -> Dict:
    """
    Given a coprime state (b,e) ∈ ℕ², return the unique word in {L,R} that
    produces it from (1,1) by applying σ (L) and R.

    The parent rule:
      - if e > b: parent(b,e) = (b, e-b), last move = 'L'
      - if b > e: parent(b,e) = (b-e, e), last move = 'R'
      - stop at (1,1)

    Failure conditions:
      - Non-positive integers -> NOT_IN_NATURAL_DOMAIN
      - gcd(b,e) != 1 -> NOT_COPRIME
      - When a reverse step would lead to zero or negative -> REVERSE_STEP_INVALID

    Parameters
    ----------
    state : tuple of (b,e)

    Returns
    -------
    dict
        {"ok": True, "value": word} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> state_to_word((2, 3))['value']
    'RL'
    >>> state_to_word((3, 2))['value']
    'LR'
    >>> state_to_word((2, 2))['ok']
    False
    >>> state_to_word((0, 1))['ok']
    False
    >>> word_to_state(state_to_word((2, 3))['value'])['value']
    (2, 3)
    >>> word_to_state(state_to_word((3, 2))['value'])['value']
    (3, 2)
    >>> word_to_state(state_to_word((5, 3))['value'])['value']
    (5, 3)
    >>> word_to_state(state_to_word((8, 5))['value'])['value']
    (8, 5)
    """
    err = _check_state(state)
    if err:
        return err
    b, e = state
    b, e = int(b), int(e)
    # Coprimality check
    g = math.gcd(b, e)
    if g != 1:
        return _failure(
            "NOT_COPRIME",
            invariant_diff={"gcd": g},
            details={"state": (b, e)},
        )
    word: List[str] = []
    cur_b, cur_e = b, e
    while not (cur_b == 1 and cur_e == 1):
        if cur_b <= 0 or cur_e <= 0:
            return _failure(
                "REVERSE_STEP_INVALID",
                details={"current_state": (cur_b, cur_e)},
            )
        if cur_b == cur_e:
            # If b==e and >1, gcd>1
            return _failure(
                "NOT_COPRIME",
                invariant_diff={"gcd": cur_b},
                details={"state": (cur_b, cur_e)},
            )
        if cur_e > cur_b:
            # Last move was L
            next_b, next_e = cur_b, cur_e - cur_b
            if next_e <= 0:
                return _failure(
                    "REVERSE_STEP_INVALID",
                    details={"current_state": (cur_b, cur_e), "next_state": (next_b, next_e)},
                )
            word.append('L')
        else:
            # Last move was R
            next_b, next_e = cur_b - cur_e, cur_e
            if next_b <= 0:
                return _failure(
                    "REVERSE_STEP_INVALID",
                    details={"current_state": (cur_b, cur_e), "next_state": (next_b, next_e)},
                )
            word.append('R')
        cur_b, cur_e = next_b, next_e
    return _success(''.join(reversed(word)))


def reachable_up_to(N: int, allow_scaling: bool = False) -> Dict:
    """
    Enumerate all states (b,e) with 1 ≤ b,e ≤ N reachable from (1,1) by forward moves.

    If ``allow_scaling`` is False, only coprime states reachable by σ (denoted ``L``) and R are included.  The mapping
    returned associates each state with its unique LR-word.  When ``allow_scaling`` is True, the function first generates
    the coprime core via σ and R, then scales each core state by integer factors k ≥ 1 using λₖ, producing all states
    (k*b, k*e) with k*b ≤ N and k*e ≤ N.  In this case the mapping associates each state with a tuple ``(word, scale)``.

    Uniqueness is enforced: if two different descriptions generate the same state, a failure with fail_type ``NOT_UNIQUE`` is returned.

    Parameters
    ----------
    N : int
        Upper bound on b and e (positive integer).
    allow_scaling : bool, optional
        Whether to include scaled versions of the coprime core.  Defaults to False.

    Returns
    -------
    dict
        {"ok": True, "value": (states_set, mapping)} on success,
        {"ok": False, ...} on failure.

    Examples
    --------
    >>> res = reachable_up_to(5)
    >>> res['ok']
    True
    >>> (2,3) in res['value'][0]
    True
    >>> res_scaled = reachable_up_to(5, allow_scaling=True)
    >>> res_scaled['ok']
    True
    >>> (4,6) in res_scaled['value'][0]  # (2,3) scaled by 2 is out of bound when N=5
    False
    """
    # Validate N
    if not isinstance(N, int) or N <= 0:
        return _failure("NOT_IN_NATURAL_DOMAIN", details={"N": N})
    # Generate coprime core via BFS on L/R
    visited_core: Set[Tuple[int, int]] = set()
    mapping_core: Dict[Tuple[int, int], str] = {}
    dq: deque = deque()
    seed = (1, 1)
    visited_core.add(seed)
    mapping_core[seed] = ''
    dq.append(seed)
    while dq:
        b, e = dq.popleft()
        word = mapping_core[(b, e)]
        # apply sigma (L)
        res_sigma = sigma((b, e))
        if res_sigma["ok"]:
            nb, ne = res_sigma["value"]
            if nb <= N and ne <= N and math.gcd(nb, ne) == 1:
                if (nb, ne) not in visited_core:
                    visited_core.add((nb, ne))
                    mapping_core[(nb, ne)] = word + 'L'
                    dq.append((nb, ne))
                else:
                    existing_word = mapping_core[(nb, ne)]
                    if existing_word != word + 'L':
                        return _failure(
                            "NOT_UNIQUE",
                            details={"state": (nb, ne), "word1": existing_word, "word2": word + 'L'},
                        )
        # apply R
        res_R = R((b, e))
        if res_R["ok"]:
            nb, ne = res_R["value"]
            if nb <= N and ne <= N and math.gcd(nb, ne) == 1:
                if (nb, ne) not in visited_core:
                    visited_core.add((nb, ne))
                    mapping_core[(nb, ne)] = word + 'R'
                    dq.append((nb, ne))
                else:
                    existing_word = mapping_core[(nb, ne)]
                    if existing_word != word + 'R':
                        return _failure(
                            "NOT_UNIQUE",
                            details={"state": (nb, ne), "word1": existing_word, "word2": word + 'R'},
                        )
    # If scaling is not allowed, return coprime core
    if not allow_scaling:
        return _success((visited_core, mapping_core))
    # Otherwise, build scaled states
    visited_scaled: Set[Tuple[int, int]] = set()
    mapping_scaled: Dict[Tuple[int, int], Tuple[str, int]] = {}
    # For each core state (b,e) with its word
    for (b, e), word in mapping_core.items():
        k = 1
        while True:
            nb = b * k
            ne = e * k
            if nb > N or ne > N:
                break
            state = (nb, ne)
            if state not in visited_scaled:
                visited_scaled.add(state)
                mapping_scaled[state] = (word, k)
            else:
                existing = mapping_scaled[state]
                if existing != (word, k):
                    return _failure(
                        "NOT_UNIQUE",
                        details={"state": state, "desc1": existing, "desc2": (word, k)},
                    )
            k += 1
    return _success((visited_scaled, mapping_scaled))


if __name__ == "__main__":
    """
    Run self‑check tests:
    - Roundtrip test for random coprime pairs under a cap.
    - Uniqueness check in reachable_up_to(50).

    Uses random generation of coprime pairs and asserts roundtrip correctness.
    """
    # Roundtrip test for random coprime pairs under a cap
    cap = 20
    num_tests = 50
    for _ in range(num_tests):
        # generate random coprime state
        while True:
            b = random.randint(1, cap)
            e = random.randint(1, cap)
            if math.gcd(b, e) == 1:
                break
        res_word = state_to_word((b, e))
        assert res_word["ok"], f"state_to_word failed on {(b, e)}: {res_word}"
        w = res_word["value"]
        res_state = word_to_state(w, seed=(1, 1))
        assert res_state["ok"], f"word_to_state failed on word {w}"
        assert res_state["value"] == (b, e), f"Roundtrip mismatch: got {res_state['value']}, expected {(b, e)}"
    print("Random roundtrip tests passed.")
    # Deterministic regression for known coprime pairs
    hardcoded_pairs = [
        (2, 3), (3, 2), (3, 5), (5, 3),
        (4, 5), (5, 4), (8, 5), (5, 8),
        (13, 8), (8, 13),
    ]
    for b, e in hardcoded_pairs:
        res_word = state_to_word((b, e))
        assert res_word["ok"], f"state_to_word failed on {(b, e)}: {res_word}"
        w = res_word["value"]
        res_state = word_to_state(w, seed=(1, 1))
        assert res_state["ok"], f"word_to_state failed on word {w}"
        assert res_state["value"] == (b, e), f"Deterministic roundtrip mismatch: got {res_state['value']}, expected {(b, e)}"
    print("Deterministic regression tests passed.")
    # Uniqueness check in reachable_up_to(50)
    res = reachable_up_to(50)
    assert res["ok"], f"reachable_up_to returned failure: {res}"
    print(f"reachable_up_to(50) enumerated {len(res['value'][0])} states without duplicates.")