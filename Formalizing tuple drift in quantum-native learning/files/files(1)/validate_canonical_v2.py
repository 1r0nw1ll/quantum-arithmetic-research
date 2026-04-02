#!/usr/bin/env python3
"""
Canon-Faithful QA Validator
Recomputes everything from canonical equations (qa_canonical.md).
No hardcoded expected values — only internal consistency + checksums.
"""

from dataclasses import dataclass
from fractions import Fraction
from collections import defaultdict

# ============================================================================
# Canonical Implementations (qa_canonical.md §1-2)
# ============================================================================

def digital_root(a: int) -> int:
    """φ₉: Digital root per canonical spec"""
    if a <= 0:
        return 0
    return ((a - 1) % 9) + 1

@dataclass(frozen=True)
class QAState:
    """Canonical QA state with 21-element packet"""
    b: int; e: int; d: int; a: int
    B: int; E: int; D: int; A: int
    X: int; C: int; F: int
    G: int; L: Fraction; H: int; I: int
    J: int; K: int; W: int; Y: int; Z: int
    h2: int
    phi_9: int; phi_24: int

def construct_qa_state(b: int, e: int) -> QAState:
    """
    Canonical state construction per qa_canonical.md §1.2-1.3.
    All formulas must match exactly.
    """
    # Derived coordinates
    d = b + e
    a = b + 2*e
    
    # Squares
    B = b * b
    E = e * e
    D = d * d
    A = a * a
    
    # Products
    X = e * d
    C = 2 * e * d
    F = b * a
    
    # Combined
    G = D + E
    L = Fraction(C * F, 12)  # Exact rational
    H = C + F
    I = abs(C - F)  # Positive difference
    J = d * b
    K = d * a
    W = X + K  # Canonical form: d(e+a) = X + K
    Y = A - D
    Z = E + K
    h2 = D * a * b  # d² · a · b
    
    # Phase annotations
    phi_9 = digital_root(a)
    phi_24 = a % 24
    
    return QAState(
        b=b, e=e, d=d, a=a,
        B=B, E=E, D=D, A=A,
        X=X, C=C, F=F,
        G=G, L=L, H=H, I=I,
        J=J, K=K, W=W, Y=Y, Z=Z,
        h2=h2,
        phi_9=phi_9, phi_24=phi_24
    )

# ============================================================================
# Generators (qa_canonical.md §2.1)
# ============================================================================

def sigma(b: int, e: int, N: int) -> tuple | None:
    """σ: (b,e) → (b, e+1)"""
    return (b, e+1) if e+1 <= N else None

def mu(b: int, e: int, N: int) -> tuple | None:
    """μ: (b,e) → (e,b)"""
    return (e, b)  # Always legal on Caps(N,N)

def lambda2(b: int, e: int, N: int) -> tuple | None:
    """λ₂: (b,e) → (2b, 2e)"""
    bb, ee = 2*b, 2*e
    return (bb, ee) if bb <= N and ee <= N else None

def nu(b: int, e: int, N: int) -> tuple | None:
    """ν: (b,e) → (b/2, e/2) if both even"""
    if (b % 2 == 0) and (e % 2 == 0):
        return (b//2, e//2)
    return None

# ============================================================================
# Caps(30,30) Checksum Validator (qa_canonical.md §12)
# ============================================================================

def caps30_checksum():
    """
    Compute Caps(30,30) topology under Σ₃ = {σ,μ,λ₂,ν}.
    Uses Kosaraju's algorithm for exact SCC count.
    """
    N = 30
    states = [(b, e) for b in range(1, N+1) for e in range(1, N+1)]
    idx = {s: i for i, s in enumerate(states)}
    
    gens = [sigma, mu, lambda2, nu]
    
    edges = 0
    fails = 0
    adj = [[] for _ in states]
    
    # Build adjacency list
    for (b, e) in states:
        for g in gens:
            t = g(b, e, N)
            if t is None:
                fails += 1
            else:
                edges += 1
                if t in idx:  # Stay within Caps
                    adj[idx[(b,e)]].append(idx[t])
    
    # Kosaraju's SCC algorithm
    # Pass 1: DFS on original graph, record finish order
    seen = [False] * len(states)
    order = []
    
    def dfs1(u):
        seen[u] = True
        for v in adj[u]:
            if not seen[v]:
                dfs1(v)
        order.append(u)
    
    for u in range(len(states)):
        if not seen[u]:
            dfs1(u)
    
    # Pass 2: DFS on reversed graph in reverse finish order
    radj = [[] for _ in states]
    for u in range(len(states)):
        for v in adj[u]:
            radj[v].append(u)
    
    comp = [-1] * len(states)
    
    def dfs2(u, cid):
        comp[u] = cid
        for v in radj[u]:
            if comp[v] == -1:
                dfs2(v, cid)
    
    cid = 0
    for u in reversed(order):
        if comp[u] == -1:
            dfs2(u, cid)
            cid += 1
    
    # Count SCC sizes
    scc_sizes = defaultdict(int)
    for c in comp:
        scc_sizes[c] += 1
    
    return {
        "num_states": len(states),
        "num_edges": edges,
        "num_failures": fails,
        "num_sccs": cid,
        "max_scc_size": max(scc_sizes.values())
    }

# ============================================================================
# Validation Tests
# ============================================================================

def test_invariant_consistency():
    """
    Test that canonical state construction satisfies internal consistency.
    No hardcoded expectations — just verify formulas match.
    """
    print("=" * 70)
    print("TEST 1: INVARIANT CONSISTENCY")
    print("=" * 70)
    
    # Test several states
    test_cases = [(3, 5), (10, 20), (1, 1), (15, 7)]
    
    all_pass = True
    for b, e in test_cases:
        s = construct_qa_state(b, e)
        
        # Verify derived coordinates
        assert s.d == b + e, f"d derivation failed for ({b},{e})"
        assert s.a == b + 2*e, f"a derivation failed for ({b},{e})"
        
        # Verify squares
        assert s.B == b*b
        assert s.E == e*e
        assert s.D == s.d * s.d
        assert s.A == s.a * s.a
        
        # Verify products
        assert s.X == e * s.d
        assert s.C == 2 * e * s.d
        assert s.F == b * s.a
        
        # Verify combined
        assert s.G == s.D + s.E
        assert s.L == Fraction(s.C * s.F, 12)
        assert s.H == s.C + s.F
        assert s.I == abs(s.C - s.F)
        assert s.J == s.d * b
        assert s.K == s.d * s.a
        assert s.W == s.X + s.K
        assert s.Y == s.A - s.D
        assert s.Z == s.E + s.K
        assert s.h2 == s.D * s.a * b
        
        # Verify phases
        assert s.phi_9 == digital_root(s.a)
        assert s.phi_24 == s.a % 24
        
        print(f"✓ State ({b},{e}): all {23} invariants consistent")
    
    print("\n✅ All invariant consistency checks pass")
    return True

def test_caps30_checksums():
    """
    Validate against canonical checksums (qa_canonical.md §12).
    """
    print("\n" + "=" * 70)
    print("TEST 2: CAPS(30,30) CHECKSUMS")
    print("=" * 70)
    
    stats = caps30_checksum()
    
    # Canonical checksums from §12
    expected = {
        'num_states': 900,
        'num_edges': 2220,
        'num_failures': 1380,
        'num_sccs': 1,
        'max_scc_size': 900
    }
    
    all_pass = True
    for key, expected_val in expected.items():
        actual_val = stats[key]
        match = "✓" if actual_val == expected_val else "✗"
        print(f"{match} {key:20s}: {actual_val:6d} (expected {expected_val})")
        if actual_val != expected_val:
            all_pass = False
    
    if all_pass:
        print("\n✅ Caps(30,30) Σ₃ checksums match canonical reference")
    else:
        print("\n❌ CHECKSUM MISMATCH - Implementation differs from canonical")
    
    return all_pass

def test_generator_legality():
    """
    Verify generator semantics match canonical spec.
    """
    print("\n" + "=" * 70)
    print("TEST 3: GENERATOR LEGALITY")
    print("=" * 70)
    
    N = 30
    
    # σ: legal iff e+1 ≤ N
    assert sigma(5, 29, N) == (5, 30)
    assert sigma(5, 30, N) is None
    print("✓ σ: legality matches canonical spec")
    
    # μ: always legal on square Caps
    assert mu(5, 10, N) == (10, 5)
    assert mu(1, 30, N) == (30, 1)
    print("✓ μ: always legal on Caps(N,N)")
    
    # λ₂: legal iff 2b,2e ≤ N
    assert lambda2(10, 15, N) == (20, 30)  # Edge case: exactly at boundary
    assert lambda2(10, 14, N) == (20, 28)
    assert lambda2(15, 15, N) == (30, 30)  # Max valid state
    assert lambda2(16, 1, N) is None  # 2*16=32 > 30
    assert lambda2(10, 16, N) is None  # 2*16=32 > 30
    print("✓ λ₂: legality matches canonical spec")
    
    # ν: legal iff both even
    assert nu(10, 20, N) == (5, 10)
    assert nu(10, 21, N) is None
    assert nu(11, 20, N) is None
    print("✓ ν: parity constraint matches canonical spec")
    
    print("\n✅ All generator semantics match canonical reference")
    return True

# ============================================================================
# Main Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CANON-FAITHFUL QA VALIDATION")
    print("qa_canonical.md v1.0")
    print("=" * 70)
    
    # Run all tests
    test1 = test_invariant_consistency()
    test2 = test_caps30_checksums()
    test3 = test_generator_legality()
    
    # Final report
    print("\n" + "=" * 70)
    print("FINAL VALIDATION REPORT")
    print("=" * 70)
    print(f"Invariant consistency: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Caps(30,30) checksums: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Generator semantics:   {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if test1 and test2 and test3:
        print("\n🎉 Implementation is CANONICAL-COMPLIANT")
        print("   All formulas, generators, and checksums match qa_canonical.md")
    else:
        print("\n⚠️  VALIDATION FAILED")
        print("   Implementation deviates from canonical reference")
