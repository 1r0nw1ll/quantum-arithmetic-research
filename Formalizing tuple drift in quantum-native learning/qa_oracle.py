"""
Layer 1 Oracle: Canonical QA Transition System
Implements exact semantics from Caps phase-transition experiments.

State: primitive (b,e), derive 21-element invariant packet
Moves: σ, μ, λ₂, ν as specified
Caps: {(b,e) : 1 ≤ b ≤ N, 1 ≤ e ≤ N}
Phase: unconstrained (q_def="none" for main results)

FIXES APPLIED:
1. I = |C - F| (positive difference)
2. W = X + K (explicit canonical form)
3. L = Fraction(C*F, 12) (exact, not floored)
4. h² stored as exact integer (h2 = d²·a·b)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set, List, Tuple, Dict
from fractions import Fraction
import hashlib
from math import sqrt

# ============================================================================
# Core Types
# ============================================================================

class FailType(Enum):
    """
    Deterministic failure classifications.
    
    Note: INVARIANT and REDUCTION are not active in Caps lattice with q_def="none"
    but arise in other QA universes (e.g., fixed-phase constraints, 
    non-reduction axioms, projection constraints).
    """
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    PARITY = "PARITY"
    PHASE_VIOLATION = "PHASE_VIOLATION"
    INVARIANT = "INVARIANT"
    REDUCTION = "REDUCTION"

@dataclass(frozen=True)
class QAState:
    """
    Canonical QA state: primitive (b,e) + derived 21-element packet.
    Immutable for hashing and set membership.
    All invariants are exact (integers or Fractions).
    """
    # Primitives
    b: int
    e: int
    
    # Derived (d, a)
    d: int  # = b + e
    a: int  # = e + d = b + 2e
    
    # Derived invariants (21-element packet)
    B: int  # = b²
    E: int  # = e²
    D: int  # = d²
    A: int  # = a²
    X: int  # = e*d
    C: int  # = 2*e*d
    F: int  # = b*a
    G: int  # = D + E = d² + e²
    L: Fraction  # = C*F/12 (exact rational)
    H: int  # = C + F
    I: int  # = |C - F| (positive difference)
    J: int  # = d*b
    K: int  # = d*a
    W: int  # = X + K = d(e+a)
    Y: int  # = A - D
    Z: int  # = E + K = e² + (a*d)
    h2: int  # = d²·a·b (semi-minor diameter squared, exact)
    
    # Phase annotations (not constraints for q_def="none")
    phi_9: int   # = digital_root(a)
    phi_24: int  # = a mod 24
    
    def __hash__(self):
        # Hash only primitives (sufficient for uniqueness)
        return hash((self.b, self.e))
    
    def __eq__(self, other):
        return self.b == other.b and self.e == other.e
    
    @property
    def primitive(self) -> Tuple[int, int]:
        return (self.b, self.e)
    
    @property
    def h(self) -> float:
        """Derive h = d*sqrt(a*b) from h² when needed (display only)"""
        return sqrt(self.h2)

@dataclass
class Failure:
    """Typed failure result with invariant diagnostics"""
    state: QAState
    move: str
    fail_type: FailType
    invariant_delta: Dict[str, any]
    message: str

# ============================================================================
# State Construction (21-Element Packet)
# ============================================================================

def digital_root(n: int) -> int:
    """
    Compute digital root (iterative sum of digits until single digit).
    Equivalent to (n-1) mod 9 + 1 for n > 0.
    """
    if n == 0:
        return 0
    return (n - 1) % 9 + 1

def construct_qa_state(b: int, e: int) -> QAState:
    """
    Build complete QA state from primitive (b,e).
    Implements exact 21-element formula packet with all fixes applied.
    """
    # Derived base
    d = b + e
    a = e + d  # = b + 2e
    
    # Squares
    B = b * b
    E = e * e
    D = d * d
    A = a * a
    
    # Products and combinations
    X = e * d
    C = 2 * e * d
    F = b * a
    G = D + E
    L = Fraction(C * F, 12)  # Exact rational
    H = C + F
    I = abs(C - F)  # Positive difference
    J = d * b
    K = d * a
    W = X + K  # Canonical form: d(e+a) = X + K
    Y = A - D
    Z = E + K
    h2 = D * a * b  # d² · a · b (exact integer)
    
    # Phase annotations
    phi_9 = digital_root(a)
    phi_24 = a % 24
    
    return QAState(
        b=b, e=e, d=d, a=a,
        B=B, E=E, D=D, A=A,
        X=X, C=C, F=F, G=G,
        L=L, H=H, I=I, J=J,
        K=K, W=W, Y=Y, Z=Z, h2=h2,
        phi_9=phi_9, phi_24=phi_24
    )

# ============================================================================
# Layer 1 Oracle: Ground Truth Transition System
# ============================================================================

class QAOracle:
    """
    Canonical QA transition oracle.
    Implements exact Caps experiment semantics.
    """
    
    def __init__(self, N: int, q_def: str = "none"):
        """
        Args:
            N: Caps bound (1 ≤ b,e ≤ N)
            q_def: Phase constraint ("none", "phi_24", "phi_9", "both")
        """
        self.N = N
        self.q_def = q_def
        
        # Caches for performance
        self._cache_legal: Dict[Tuple[QAState, str, int], bool] = {}
        self._cache_step: Dict[Tuple[QAState, str, int], QAState | Failure] = {}
    
    # ------------------------------------------------------------------------
    # Core Predicates
    # ------------------------------------------------------------------------
    
    def is_legal(self, state: QAState, move: str, k: int = 2) -> bool:
        """
        Ground truth legality check.
        
        Args:
            state: Current QA state
            move: One of {"sigma", "mu", "lambda2", "nu"}
            k: Scale factor for lambda2 (always 2 for canonical experiments)
        """
        cache_key = (state, move, k)
        if cache_key in self._cache_legal:
            return self._cache_legal[cache_key]
        
        result = self.step(state, move, k)
        is_legal = not isinstance(result, Failure)
        
        self._cache_legal[cache_key] = is_legal
        return is_legal
    
    def step(self, state: QAState, move: str, k: int = 2) -> QAState | Failure:
        """
        Execute move on state.
        
        Args:
            state: Current state
            move: One of {"sigma", "mu", "lambda2", "nu"}
            k: Scale factor for lambda2
            
        Returns:
            Next QAState if legal, Failure if illegal
        """
        cache_key = (state, move, k)
        if cache_key in self._cache_step:
            return self._cache_step[cache_key]
        
        if move == "sigma":
            result = self._apply_sigma(state)
        elif move == "mu":
            result = self._apply_mu(state)
        elif move == "lambda2":
            result = self._apply_lambda2(state, k)
        elif move == "nu":
            result = self._apply_nu(state)
        else:
            raise ValueError(f"Unknown move: {move}")
        
        self._cache_step[cache_key] = result
        return result
    
    def get_fail_type(self, state: QAState, move: str, k: int = 2) -> Optional[FailType]:
        """Return fail_type if illegal, None if legal"""
        result = self.step(state, move, k)
        if isinstance(result, Failure):
            return result.fail_type
        return None
    
    # ------------------------------------------------------------------------
    # Move Implementations (Exact Caps Semantics)
    # ------------------------------------------------------------------------
    
    def _apply_sigma(self, s: QAState) -> QAState | Failure:
        """σ: (b,e) → (b, e+1)"""
        b, e = s.b, s.e
        e_next = e + 1
        
        # Bounds check
        if e_next > self.N:
            return Failure(
                state=s,
                move="sigma",
                fail_type=FailType.OUT_OF_BOUNDS,
                invariant_delta={'e': e_next - e},
                message=f"e+1 = {e_next} exceeds N = {self.N}"
            )
        
        # Construct next state
        next_s = construct_qa_state(b, e_next)
        
        # Phase check (if constrained)
        if not self._check_phase_constraint(s, next_s):
            return Failure(
                state=s,
                move="sigma",
                fail_type=FailType.PHASE_VIOLATION,
                invariant_delta=self._phase_delta(s, next_s),
                message="Phase constraint violated"
            )
        
        return next_s
    
    def _apply_mu(self, s: QAState) -> QAState | Failure:
        """μ: (b,e) → (e,b)"""
        b, e = s.b, s.e
        
        # Bounds check (automatic for square Caps)
        if e > self.N or b > self.N:
            return Failure(
                state=s,
                move="mu",
                fail_type=FailType.OUT_OF_BOUNDS,
                invariant_delta={},
                message="Swap exceeds bounds"
            )
        
        next_s = construct_qa_state(e, b)
        
        if not self._check_phase_constraint(s, next_s):
            return Failure(
                state=s,
                move="mu",
                fail_type=FailType.PHASE_VIOLATION,
                invariant_delta=self._phase_delta(s, next_s),
                message="Phase constraint violated"
            )
        
        return next_s
    
    def _apply_lambda2(self, s: QAState, k: int = 2) -> QAState | Failure:
        """λ₂: (b,e) → (2b, 2e)"""
        b, e = s.b, s.e
        b_next = k * b
        e_next = k * e
        
        # Bounds check
        if b_next > self.N or e_next > self.N:
            return Failure(
                state=s,
                move="lambda2",
                fail_type=FailType.OUT_OF_BOUNDS,
                invariant_delta={'b': b_next - b, 'e': e_next - e},
                message=f"Scaled state ({b_next},{e_next}) exceeds N = {self.N}"
            )
        
        next_s = construct_qa_state(b_next, e_next)
        
        if not self._check_phase_constraint(s, next_s):
            return Failure(
                state=s,
                move="lambda2",
                fail_type=FailType.PHASE_VIOLATION,
                invariant_delta=self._phase_delta(s, next_s),
                message="Phase constraint violated"
            )
        
        return next_s
    
    def _apply_nu(self, s: QAState) -> QAState | Failure:
        """ν: (b,e) → (b/2, e/2) if both even"""
        b, e = s.b, s.e
        
        # Parity check
        if b % 2 != 0 or e % 2 != 0:
            return Failure(
                state=s,
                move="nu",
                fail_type=FailType.PARITY,
                invariant_delta={'b_parity': b % 2, 'e_parity': e % 2},
                message="ν requires both b,e even"
            )
        
        b_next = b // 2
        e_next = e // 2
        
        # Positivity check (should always pass given b,e > 0)
        if b_next < 1 or e_next < 1:
            return Failure(
                state=s,
                move="nu",
                fail_type=FailType.OUT_OF_BOUNDS,
                invariant_delta={},
                message="Contraction would violate positivity"
            )
        
        next_s = construct_qa_state(b_next, e_next)
        
        if not self._check_phase_constraint(s, next_s):
            return Failure(
                state=s,
                move="nu",
                fail_type=FailType.PHASE_VIOLATION,
                invariant_delta=self._phase_delta(s, next_s),
                message="Phase constraint violated"
            )
        
        return next_s
    
    # ------------------------------------------------------------------------
    # Phase Constraint Checking
    # ------------------------------------------------------------------------
    
    def _check_phase_constraint(self, s1: QAState, s2: QAState) -> bool:
        """Check if transition preserves phase constraint"""
        if self.q_def == "none":
            return True  # Unconstrained
        elif self.q_def == "phi_24":
            return s1.phi_24 == s2.phi_24
        elif self.q_def == "phi_9":
            return s1.phi_9 == s2.phi_9
        elif self.q_def == "both":
            return s1.phi_9 == s2.phi_9 and s1.phi_24 == s2.phi_24
        else:
            raise ValueError(f"Unknown q_def: {self.q_def}")
    
    def _phase_delta(self, s1: QAState, s2: QAState) -> Dict[str, int]:
        """Compute phase difference"""
        return {
            'phi_9': s2.phi_9 - s1.phi_9,
            'phi_24': s2.phi_24 - s1.phi_24
        }
    
    # ------------------------------------------------------------------------
    # Reachability (Return-in-k)
    # ------------------------------------------------------------------------
    
    def return_in_k(self, state: QAState, target_class: Set[QAState],
                    k: int, generators: List[Tuple[str, int]]) -> bool:
        """
        Exact reachability check via bounded BFS.
        
        Args:
            state: Starting state
            target_class: Set of goal states
            k: Maximum depth
            generators: List of (move_name, k_param) tuples
                       e.g., [("sigma", 2), ("mu", 2), ("lambda2", 2), ("nu", 2)]
        
        Returns:
            True iff ∃ path of length ≤k to target_class
        """
        if state in target_class:
            return True
        
        visited = {state}
        frontier = [(state, 0)]
        
        while frontier:
            s, depth = frontier.pop(0)
            
            if depth >= k:
                continue
            
            for move, k_param in generators:
                if self.is_legal(s, move, k_param):
                    next_s = self.step(s, move, k_param)
                    
                    if next_s in target_class:
                        return True
                    
                    if next_s not in visited:
                        visited.add(next_s)
                        frontier.append((next_s, depth + 1))
        
        return False
    
    # ------------------------------------------------------------------------
    # Topology Analysis
    # ------------------------------------------------------------------------
    
    def compute_scc_id(self, state: QAState, generators: List[Tuple[str, int]],
                      max_depth: int = 100) -> str:
        """
        Approximate SCC ID via reachable set hash.
        For exact SCCs, use Tarjan on full graph.
        """
        reachable = self._compute_reachable_set(state, generators, max_depth)
        canonical = min(reachable, key=lambda s: s.primitive)
        return hashlib.md5(str(canonical.primitive).encode()).hexdigest()[:8]
    
    def _compute_reachable_set(self, state: QAState, 
                              generators: List[Tuple[str, int]],
                              max_depth: int) -> Set[QAState]:
        """BFS reachable set computation"""
        visited = {state}
        frontier = [state]
        depth = 0
        
        while frontier and depth < max_depth:
            next_frontier = []
            for s in frontier:
                for move, k_param in generators:
                    if self.is_legal(s, move, k_param):
                        next_s = self.step(s, move, k_param)
                        if next_s not in visited:
                            visited.add(next_s)
                            next_frontier.append(next_s)
            frontier = next_frontier
            depth += 1
        
        return visited
