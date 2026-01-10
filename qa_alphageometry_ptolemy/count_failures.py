#!/usr/bin/env python3
"""
Count failure states by type using TLC state enumeration.
Manually enumerates all reachable states matching the TLA+ spec.
"""

def DR(n):
    """Digital root (mod 9 reduction)"""
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)

def Phi9(n):
    return DR(n)

def Phi24(n):
    return n % 24

def QDef(b, e, d, a):
    """Duo-modular q definition: 24*phi9(a) + phi24(a)"""
    return 24 * Phi9(a) + Phi24(a)

def TupleClosed(b, e, d, a):
    return d == b + e and a == d + e

def InCap(x, CAP):
    return 0 <= x <= CAP

def InBounds(b, e, d, a, CAP):
    return all(InCap(x, CAP) for x in [b, e, d, a])

def try_sigma(b, e, d, a, qtag, CAP):
    """Attempt sigma move, return (success, new_state, fail_type)"""
    e2 = e + 1
    b2 = b
    d2 = b2 + e2
    a2 = d2 + e2

    if not InBounds(b2, e2, d2, a2, CAP):
        return (False, (b, e, d, a, qtag, "OUT_OF_BOUNDS", "σ"))

    if QDef(b2, e2, d2, a2) != qtag:
        return (False, (b, e, d, a, qtag, "FIXED_Q_VIOLATION", "σ"))

    return (True, (b2, e2, d2, a2, qtag, "OK", "σ"))

def try_mu(b, e, d, a, qtag, CAP):
    """Attempt mu move"""
    b2 = e
    e2 = b
    d2 = b2 + e2
    a2 = d2 + e2

    if not InBounds(b2, e2, d2, a2, CAP):
        return (False, (b, e, d, a, qtag, "OUT_OF_BOUNDS", "μ"))

    if QDef(b2, e2, d2, a2) != qtag:
        return (False, (b, e, d, a, qtag, "FIXED_Q_VIOLATION", "μ"))

    return (True, (b2, e2, d2, a2, qtag, "OK", "μ"))

def try_lambda(b, e, d, a, qtag, CAP, KSet):
    """Attempt lambda moves for each k in KSet"""
    results = []
    for k in KSet:
        b2 = k * b
        e2 = k * e
        d2 = b2 + e2
        a2 = d2 + e2

        if not InBounds(b2, e2, d2, a2, CAP):
            results.append((False, (b, e, d, a, qtag, "OUT_OF_BOUNDS", f"λ_{k}")))
        elif QDef(b2, e2, d2, a2) != qtag:
            results.append((False, (b, e, d, a, qtag, "FIXED_Q_VIOLATION", f"λ_{k}")))
        else:
            results.append((True, (b2, e2, d2, a2, qtag, "OK", f"λ_{k}")))

    return results

def explore_with_generators(CAP, KSet, use_mu=True):
    """BFS exploration of state space"""
    # Generate all initial states
    init_states = set()
    for b in range(CAP + 1):
        for e in range(CAP + 1):
            d = b + e
            a = d + e
            if TupleClosed(b, e, d, a) and InBounds(b, e, d, a, CAP):
                qtag = QDef(b, e, d, a)
                init_states.add((b, e, d, a, qtag, "OK", "NONE"))

    visited = set()
    queue = list(init_states)
    visited.update(init_states)

    failure_states = {
        "OUT_OF_BOUNDS": set(),
        "FIXED_Q_VIOLATION": set(),
    }

    total_attempts = 0

    while queue:
        current = queue.pop(0)
        b, e, d, a, qtag, fail, lastMove = current

        if fail != "OK":
            continue  # Absorbing failure states

        # Try sigma
        total_attempts += 1
        success, new_state = try_sigma(b, e, d, a, qtag, CAP)
        if not success:
            _, (_, _, _, _, _, fail_type, _) = success, new_state
            failure_states[fail_type].add(new_state)
        elif new_state not in visited:
            visited.add(new_state)
            queue.append(new_state)

        # Try mu (if enabled)
        if use_mu:
            total_attempts += 1
            success, new_state = try_mu(b, e, d, a, qtag, CAP)
            if not success:
                _, (_, _, _, _, _, fail_type, _) = success, new_state
                failure_states[fail_type].add(new_state)
            elif new_state not in visited:
                visited.add(new_state)
                queue.append(new_state)

        # Try lambda for each k
        lambda_results = try_lambda(b, e, d, a, qtag, CAP, KSet)
        for success, new_state in lambda_results:
            total_attempts += 1
            if not success:
                _, (_, _, _, _, _, fail_type, _) = success, new_state
                failure_states[fail_type].add(new_state)
            elif new_state not in visited:
                visited.add(new_state)
                queue.append(new_state)

    return visited, failure_states

def main():
    CAP = 20
    KSet = {2, 3}

    print("=" * 70)
    print("QA/QARM Failure Analysis - Duo-modular qtag = 24*phi9(a) + phi24(a)")
    print("=" * 70)
    print(f"CAP = {CAP}, KSet = {KSet}")
    print()

    # Run 1: Full generator set
    print("Run 1: Generator set {σ, μ, λ}")
    print("-" * 70)
    visited_full, failures_full = explore_with_generators(CAP, KSet, use_mu=True)
    print(f"Distinct states found: {len(visited_full)}")
    print(f"OUT_OF_BOUNDS failures: {len(failures_full['OUT_OF_BOUNDS'])}")
    print(f"FIXED_Q_VIOLATION failures: {len(failures_full['FIXED_Q_VIOLATION'])}")
    print()

    # Run 2: Without mu
    print("Run 2: Generator set {σ, λ} (μ removed)")
    print("-" * 70)
    visited_nomu, failures_nomu = explore_with_generators(CAP, KSet, use_mu=False)
    print(f"Distinct states found: {len(visited_nomu)}")
    print(f"OUT_OF_BOUNDS failures: {len(failures_nomu['OUT_OF_BOUNDS'])}")
    print(f"FIXED_Q_VIOLATION failures: {len(failures_nomu['FIXED_Q_VIOLATION'])}")
    print()

    # Compare
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"State count change: {len(visited_full)} → {len(visited_nomu)} (Δ = {len(visited_full) - len(visited_nomu)})")

    oob_delta = len(failures_full['OUT_OF_BOUNDS']) - len(failures_nomu['OUT_OF_BOUNDS'])
    fixq_delta = len(failures_full['FIXED_Q_VIOLATION']) - len(failures_nomu['FIXED_Q_VIOLATION'])

    print(f"OUT_OF_BOUNDS: {len(failures_full['OUT_OF_BOUNDS'])} → {len(failures_nomu['OUT_OF_BOUNDS'])} (Δ = {oob_delta})")
    print(f"FIXED_Q_VIOLATION: {len(failures_full['FIXED_Q_VIOLATION'])} → {len(failures_nomu['FIXED_Q_VIOLATION'])} (Δ = {fixq_delta})")
    print()

    if oob_delta == 0 and fixq_delta == 0:
        print("✅ FAILURE COUNTS ARE INVARIANT under generator set change")
    else:
        print("❌ FAILURE COUNTS CHANGED (invariance hypothesis rejected)")

if __name__ == "__main__":
    main()
