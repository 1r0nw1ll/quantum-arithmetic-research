"""
qa_sparse_attention_certificate.py

QA Certificate Schema for Sparse Attention / Transformer Efficiency
Based on: Efficient Transformers literature (Linformer, BigBird, Longformer, etc.)

Maps attention mechanism structure to QA framework:
- Attention entropy → exploration breadth invariant
- Effective rank → dimensionality invariant
- Sparsity patterns → constrained reachability
- Redundant heads → gauge freedom
- Rank/entropy collapse → failure modes

Hard constraints:
- Exact scalars only (int/Fraction) — no floats in certificates
- Deterministic serialization
- Failure-completeness: every attention computation yields success OR obstruction proof
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict, Any, Union, Tuple
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math

# ============================================================================
# FOUNDATIONAL TYPES
# ============================================================================

Scalar = Union[int, Fraction]


def to_scalar(x: Any) -> Scalar:
    """Convert to exact scalar, rejecting raw floats."""
    if isinstance(x, bool):
        raise TypeError("Cannot convert bool to exact scalar")
    if isinstance(x, (int, Fraction)):
        return x
    if isinstance(x, float):
        return Fraction(x).limit_denominator(10**9)
    if isinstance(x, str):
        s = x.strip()
        if "/" in s or "." in s:
            return Fraction(s)
        return int(s)
    raise TypeError(f"Cannot convert {type(x)} to exact scalar (got {x})")


# ============================================================================
# ATTENTION ENTROPY WITNESS
# ============================================================================

@dataclass(frozen=True)
class AttentionEntropyWitness:
    """
    Witness for attention entropy (exploration breadth).

    Entropy of attention distribution measures how "spread out" attention is:
    - High entropy → diffuse attention (explores many tokens)
    - Low entropy → focused attention (concentrates on few tokens)
    - Zero entropy → degenerate (all attention on single token)

    QA interpretation: Entropy bounds the effective reachability radius
    in the token graph.
    """
    layer: int
    head: int

    # Entropy statistics across sequence
    min_entropy: Scalar
    max_entropy: Scalar
    mean_entropy: Scalar

    # Theoretical bounds
    max_possible_entropy: Scalar  # log(seq_len)

    # Normalized entropy (0-1)
    normalized_entropy: Scalar

    # Thresholds
    collapse_threshold: Scalar = Fraction(1, 10)  # Below this = collapsed
    uniform_threshold: Scalar = Fraction(9, 10)   # Above this = uniform

    # Status
    entropy_healthy: bool = True

    def __post_init__(self):
        object.__setattr__(self, "min_entropy", to_scalar(self.min_entropy))
        object.__setattr__(self, "max_entropy", to_scalar(self.max_entropy))
        object.__setattr__(self, "mean_entropy", to_scalar(self.mean_entropy))
        object.__setattr__(self, "max_possible_entropy", to_scalar(self.max_possible_entropy))
        object.__setattr__(self, "normalized_entropy", to_scalar(self.normalized_entropy))
        object.__setattr__(self, "collapse_threshold", to_scalar(self.collapse_threshold))
        object.__setattr__(self, "uniform_threshold", to_scalar(self.uniform_threshold))

    def verify_normalized(self) -> bool:
        """Verify normalized_entropy = mean_entropy / max_possible_entropy."""
        if Fraction(self.max_possible_entropy) == 0:
            return True
        computed = Fraction(self.mean_entropy) / Fraction(self.max_possible_entropy)
        diff = abs(computed - Fraction(self.normalized_entropy))
        return diff < Fraction(1, 100)

    def verify_healthy(self) -> bool:
        """Verify entropy_healthy flag."""
        norm = Fraction(self.normalized_entropy)
        collapse = Fraction(self.collapse_threshold)
        uniform = Fraction(self.uniform_threshold)
        should_be_healthy = norm > collapse and norm < uniform
        return should_be_healthy == self.entropy_healthy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "head": self.head,
            "min_entropy": str(self.min_entropy),
            "max_entropy": str(self.max_entropy),
            "mean_entropy": str(self.mean_entropy),
            "max_possible_entropy": str(self.max_possible_entropy),
            "normalized_entropy": str(self.normalized_entropy),
            "collapse_threshold": str(self.collapse_threshold),
            "uniform_threshold": str(self.uniform_threshold),
            "entropy_healthy": self.entropy_healthy,
        }


# ============================================================================
# EFFECTIVE RANK WITNESS
# ============================================================================

@dataclass(frozen=True)
class EffectiveRankWitness:
    """
    Witness for effective rank of attention matrix.

    Effective rank measures the "true" dimensionality of attention:
    - Full rank → each token gets unique attention pattern
    - Low rank → tokens share similar attention patterns
    - Rank 1 → all tokens attend identically (collapsed)

    QA interpretation: Rank bounds the information capacity of the
    attention layer.
    """
    layer: int

    # Sequence length
    sequence_length: int

    # Rank statistics
    effective_rank: Scalar  # Computed via entropy of singular values
    numerical_rank: int     # Count of singular values above threshold

    # Rank ratio
    rank_ratio: Scalar  # effective_rank / sequence_length

    # Singular value statistics
    top_singular_value: Scalar
    singular_value_entropy: Scalar

    # Thresholds
    collapse_threshold: Scalar = Fraction(1, 10)  # rank_ratio below this = collapsed

    # Status
    rank_healthy: bool = True

    def __post_init__(self):
        object.__setattr__(self, "effective_rank", to_scalar(self.effective_rank))
        object.__setattr__(self, "rank_ratio", to_scalar(self.rank_ratio))
        object.__setattr__(self, "top_singular_value", to_scalar(self.top_singular_value))
        object.__setattr__(self, "singular_value_entropy", to_scalar(self.singular_value_entropy))
        object.__setattr__(self, "collapse_threshold", to_scalar(self.collapse_threshold))

    def verify_rank_ratio(self) -> bool:
        """Verify rank_ratio = effective_rank / sequence_length."""
        if self.sequence_length == 0:
            return True
        computed = Fraction(self.effective_rank) / Fraction(self.sequence_length)
        diff = abs(computed - Fraction(self.rank_ratio))
        return diff < Fraction(1, 100)

    def verify_healthy(self) -> bool:
        """Verify rank_healthy flag."""
        ratio = Fraction(self.rank_ratio)
        threshold = Fraction(self.collapse_threshold)
        should_be_healthy = ratio > threshold
        return should_be_healthy == self.rank_healthy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "sequence_length": self.sequence_length,
            "effective_rank": str(self.effective_rank),
            "numerical_rank": self.numerical_rank,
            "rank_ratio": str(self.rank_ratio),
            "top_singular_value": str(self.top_singular_value),
            "singular_value_entropy": str(self.singular_value_entropy),
            "collapse_threshold": str(self.collapse_threshold),
            "rank_healthy": self.rank_healthy,
        }


# ============================================================================
# SPARSITY PATTERN WITNESS
# ============================================================================

class SparsityPatternType(Enum):
    """Types of attention sparsity patterns."""
    DENSE = "dense"           # Full attention
    LOCAL = "local"           # Local window
    STRIDED = "strided"       # Strided attention
    RANDOM = "random"         # Random sparse
    COMBINED = "combined"     # Combination patterns
    LEARNED = "learned"       # Learned sparsity


@dataclass(frozen=True)
class SparsityPatternWitness:
    """
    Witness for attention sparsity pattern.

    Sparsity patterns define which token pairs can attend to each other:
    - Dense: all pairs allowed (O(n²))
    - Sparse: restricted pairs (O(n) or O(n log n))

    QA interpretation: Sparsity defines the reachability graph structure.
    """
    pattern_type: SparsityPatternType

    # Sequence length
    sequence_length: int

    # Sparsity statistics
    total_possible_pairs: int  # n²
    allowed_pairs: int         # Actually computed
    sparsity_ratio: Scalar     # allowed / total

    # Pattern parameters
    window_size: Optional[int] = None  # For local attention
    stride: Optional[int] = None       # For strided attention
    random_fraction: Optional[Scalar] = None  # For random sparse

    # Connectivity check
    all_tokens_reachable: bool = True  # No disconnected tokens

    def __post_init__(self):
        object.__setattr__(self, "sparsity_ratio", to_scalar(self.sparsity_ratio))
        if self.random_fraction is not None:
            object.__setattr__(self, "random_fraction", to_scalar(self.random_fraction))

    def verify_sparsity_ratio(self) -> bool:
        """Verify sparsity_ratio = allowed_pairs / total_possible_pairs."""
        if self.total_possible_pairs == 0:
            return True
        computed = Fraction(self.allowed_pairs, self.total_possible_pairs)
        diff = abs(computed - Fraction(self.sparsity_ratio))
        return diff < Fraction(1, 100)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "pattern_type": self.pattern_type.value,
            "sequence_length": self.sequence_length,
            "total_possible_pairs": self.total_possible_pairs,
            "allowed_pairs": self.allowed_pairs,
            "sparsity_ratio": str(self.sparsity_ratio),
            "all_tokens_reachable": self.all_tokens_reachable,
        }
        if self.window_size is not None:
            result["window_size"] = self.window_size
        if self.stride is not None:
            result["stride"] = self.stride
        if self.random_fraction is not None:
            result["random_fraction"] = str(self.random_fraction)
        return result


# ============================================================================
# HEAD REDUNDANCY WITNESS (Gauge Freedom)
# ============================================================================

@dataclass(frozen=True)
class HeadRedundancyWitness:
    """
    Witness for attention head redundancy (gauge freedom).

    Many attention heads are prunable without performance loss.
    These redundant heads are gauge degrees of freedom.

    QA interpretation: Redundant heads = gauge coordinates that
    don't affect the certificate.
    """
    layer: int

    # Head counts
    total_heads: int
    active_heads: int      # Heads that meaningfully contribute
    redundant_heads: int   # Heads that can be pruned

    # Gauge dimension
    gauge_dim: int  # = redundant_heads * head_dim

    # Head importance scores (optional)
    importance_scores: Optional[List[Scalar]] = None

    # Pruning threshold
    pruning_threshold: Scalar = Fraction(1, 100)

    def __post_init__(self):
        object.__setattr__(self, "pruning_threshold", to_scalar(self.pruning_threshold))
        if self.importance_scores is not None:
            object.__setattr__(self, "importance_scores",
                              [to_scalar(s) for s in self.importance_scores])

    def verify_redundant_count(self) -> bool:
        """Verify redundant_heads = total_heads - active_heads."""
        return self.redundant_heads == self.total_heads - self.active_heads

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "layer": self.layer,
            "total_heads": self.total_heads,
            "active_heads": self.active_heads,
            "redundant_heads": self.redundant_heads,
            "gauge_dim": self.gauge_dim,
            "pruning_threshold": str(self.pruning_threshold),
        }
        if self.importance_scores is not None:
            result["importance_scores"] = [str(s) for s in self.importance_scores]
        return result


# ============================================================================
# INFORMATION FLOW WITNESS
# ============================================================================

@dataclass(frozen=True)
class InformationFlowWitness:
    """
    Witness for information flow through attention stack.

    Tracks how information propagates through layers:
    - Residual contribution vs attention contribution
    - Layer-wise representation similarity
    - Gradient flow statistics

    QA interpretation: Information flow bounds reachability across layers.
    """
    num_layers: int

    # Residual vs attention balance per layer
    residual_contribution: List[Scalar]
    attention_contribution: List[Scalar]

    # Layer similarity (how much each layer changes representation)
    layer_similarity: List[Scalar]  # cos_sim(layer_i, layer_{i+1})

    # Overall flow health
    flow_healthy: bool = True

    # Gradient flow (optional)
    gradient_norm_per_layer: Optional[List[Scalar]] = None

    def __post_init__(self):
        object.__setattr__(self, "residual_contribution",
                          [to_scalar(x) for x in self.residual_contribution])
        object.__setattr__(self, "attention_contribution",
                          [to_scalar(x) for x in self.attention_contribution])
        object.__setattr__(self, "layer_similarity",
                          [to_scalar(x) for x in self.layer_similarity])
        if self.gradient_norm_per_layer is not None:
            object.__setattr__(self, "gradient_norm_per_layer",
                              [to_scalar(x) for x in self.gradient_norm_per_layer])

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "num_layers": self.num_layers,
            "residual_contribution": [str(x) for x in self.residual_contribution],
            "attention_contribution": [str(x) for x in self.attention_contribution],
            "layer_similarity": [str(x) for x in self.layer_similarity],
            "flow_healthy": self.flow_healthy,
        }
        if self.gradient_norm_per_layer is not None:
            result["gradient_norm_per_layer"] = [str(x) for x in self.gradient_norm_per_layer]
        return result


# ============================================================================
# FAILURE TAXONOMY
# ============================================================================

class SparseAttentionFailure(Enum):
    """Failure modes for sparse attention certificates."""
    # Entropy failures
    ENTROPY_COLLAPSE = "entropy_collapse"        # All attention on one token
    ENTROPY_UNIFORM = "entropy_uniform"          # No discrimination
    ENTROPY_UNSTABLE = "entropy_unstable"        # Wild entropy fluctuations

    # Rank failures
    RANK_COLLAPSE = "rank_collapse"              # Attention matrix near rank-1
    REPRESENTATION_COLLAPSE = "representation_collapse"  # All tokens identical

    # Sparsity failures
    DISCONNECTED_TOKENS = "disconnected_tokens"  # Some tokens unreachable
    SPARSITY_TOO_AGGRESSIVE = "sparsity_too_aggressive"  # Lost critical paths

    # Information flow failures
    GRADIENT_VANISHING = "gradient_vanishing"    # Gradient too small
    GRADIENT_EXPLODING = "gradient_exploding"    # Gradient too large
    RESIDUAL_DOMINATED = "residual_dominated"    # Attention contributes nothing

    # Approximation failures
    LINEAR_APPROXIMATION_ERROR = "linear_approximation_error"  # Kernel approx too coarse


# ============================================================================
# MAIN ATTENTION LAYER CERTIFICATE
# ============================================================================

@dataclass
class SparseAttentionCertificate:
    """
    Certificate for a sparse attention layer or stack.

    A valid attention certificate requires:
    1. Entropy within healthy bounds (not collapsed, not uniform)
    2. Rank above collapse threshold
    3. All tokens reachable (no disconnected components)
    4. Information flow healthy

    Failure-completeness: If ANY requirement fails, a structured
    obstruction certificate is produced.
    """
    # Certificate metadata
    certificate_id: str
    version: str = "1.0.0"
    schema: str = "QA_SPARSE_ATTENTION_V1"

    # Success or failure
    success: bool = True
    failure_mode: Optional[SparseAttentionFailure] = None
    failure_witness: Optional[Dict[str, Any]] = None

    # Model parameters
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    sequence_length: Optional[int] = None

    # Core witnesses
    entropy_witnesses: Optional[List[AttentionEntropyWitness]] = None
    rank_witnesses: Optional[List[EffectiveRankWitness]] = None
    sparsity_witness: Optional[SparsityPatternWitness] = None
    head_redundancy: Optional[List[HeadRedundancyWitness]] = None
    information_flow: Optional[InformationFlowWitness] = None

    # Aggregate statistics
    mean_entropy: Optional[Scalar] = None
    mean_rank_ratio: Optional[Scalar] = None
    total_gauge_dim: Optional[int] = None

    # Task performance (optional)
    task_accuracy: Optional[Scalar] = None
    task_name: Optional[str] = None

    # Recompute hook reference
    recompute_hook: Optional[str] = None

    def __post_init__(self):
        if self.mean_entropy is not None:
            self.mean_entropy = to_scalar(self.mean_entropy)
        if self.mean_rank_ratio is not None:
            self.mean_rank_ratio = to_scalar(self.mean_rank_ratio)
        if self.task_accuracy is not None:
            self.task_accuracy = to_scalar(self.task_accuracy)

    def verify_entropy_health(self) -> bool:
        """Verify all entropy witnesses are healthy."""
        if self.entropy_witnesses is None:
            return True
        return all(ew.entropy_healthy for ew in self.entropy_witnesses)

    def verify_rank_health(self) -> bool:
        """Verify all rank witnesses are healthy."""
        if self.rank_witnesses is None:
            return True
        return all(rw.rank_healthy for rw in self.rank_witnesses)

    def verify_connectivity(self) -> bool:
        """Verify all tokens are reachable."""
        if self.sparsity_witness is None:
            return True
        return self.sparsity_witness.all_tokens_reachable

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "certificate_id": self.certificate_id,
            "version": self.version,
            "schema": self.schema,
            "success": self.success,
        }

        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value
        if self.failure_witness:
            result["failure_witness"] = self.failure_witness
        if self.num_layers is not None:
            result["num_layers"] = self.num_layers
        if self.num_heads is not None:
            result["num_heads"] = self.num_heads
        if self.head_dim is not None:
            result["head_dim"] = self.head_dim
        if self.sequence_length is not None:
            result["sequence_length"] = self.sequence_length
        if self.entropy_witnesses:
            result["entropy_witnesses"] = [ew.to_dict() for ew in self.entropy_witnesses]
        if self.rank_witnesses:
            result["rank_witnesses"] = [rw.to_dict() for rw in self.rank_witnesses]
        if self.sparsity_witness:
            result["sparsity_witness"] = self.sparsity_witness.to_dict()
        if self.head_redundancy:
            result["head_redundancy"] = [hr.to_dict() for hr in self.head_redundancy]
        if self.information_flow:
            result["information_flow"] = self.information_flow.to_dict()
        if self.mean_entropy is not None:
            result["mean_entropy"] = str(self.mean_entropy)
        if self.mean_rank_ratio is not None:
            result["mean_rank_ratio"] = str(self.mean_rank_ratio)
        if self.total_gauge_dim is not None:
            result["total_gauge_dim"] = self.total_gauge_dim
        if self.task_accuracy is not None:
            result["task_accuracy"] = str(self.task_accuracy)
        if self.task_name:
            result["task_name"] = self.task_name
        if self.recompute_hook:
            result["recompute_hook"] = self.recompute_hook

        return result

    def compute_certificate_hash(self) -> str:
        """Compute deterministic hash of certificate."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_success_certificate(
    certificate_id: str,
    num_layers: int,
    num_heads: int,
    sequence_length: int,
    entropy_witnesses: List[AttentionEntropyWitness],
    rank_witnesses: List[EffectiveRankWitness],
    sparsity_witness: Optional[SparsityPatternWitness] = None,
    head_redundancy: Optional[List[HeadRedundancyWitness]] = None,
) -> SparseAttentionCertificate:
    """Create a success certificate with all witnesses."""
    # Compute aggregates
    mean_entropy = sum(Fraction(ew.normalized_entropy) for ew in entropy_witnesses) / len(entropy_witnesses)
    mean_rank = sum(Fraction(rw.rank_ratio) for rw in rank_witnesses) / len(rank_witnesses)
    total_gauge = sum(hr.gauge_dim for hr in head_redundancy) if head_redundancy else 0

    return SparseAttentionCertificate(
        certificate_id=certificate_id,
        success=True,
        num_layers=num_layers,
        num_heads=num_heads,
        sequence_length=sequence_length,
        entropy_witnesses=entropy_witnesses,
        rank_witnesses=rank_witnesses,
        sparsity_witness=sparsity_witness,
        head_redundancy=head_redundancy,
        mean_entropy=mean_entropy,
        mean_rank_ratio=mean_rank,
        total_gauge_dim=total_gauge,
    )


def create_failure_certificate(
    certificate_id: str,
    failure_mode: SparseAttentionFailure,
    failure_witness: Dict[str, Any],
    entropy_witnesses: Optional[List[AttentionEntropyWitness]] = None,
    rank_witnesses: Optional[List[EffectiveRankWitness]] = None,
) -> SparseAttentionCertificate:
    """Create a failure certificate with obstruction witness."""
    return SparseAttentionCertificate(
        certificate_id=certificate_id,
        success=False,
        failure_mode=failure_mode,
        failure_witness=failure_witness,
        entropy_witnesses=entropy_witnesses,
        rank_witnesses=rank_witnesses,
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a success certificate for a 12-layer transformer

    # 1. Entropy witnesses (sample for layer 0, head 0)
    entropy = AttentionEntropyWitness(
        layer=0,
        head=0,
        min_entropy=Fraction(2, 1),
        max_entropy=Fraction(5, 1),
        mean_entropy=Fraction(35, 10),
        max_possible_entropy=Fraction(7, 1),  # log(512) ≈ 6.2
        normalized_entropy=Fraction(1, 2),
        entropy_healthy=True,
    )

    # 2. Rank witness for layer 0
    rank = EffectiveRankWitness(
        layer=0,
        sequence_length=512,
        effective_rank=Fraction(180, 1),
        numerical_rank=200,
        rank_ratio=Fraction(180, 512),
        top_singular_value=Fraction(15, 1),
        singular_value_entropy=Fraction(4, 1),
        rank_healthy=True,
    )

    # 3. Head redundancy
    head_red = HeadRedundancyWitness(
        layer=0,
        total_heads=12,
        active_heads=8,
        redundant_heads=4,
        gauge_dim=4 * 64,  # 4 heads * 64 dim
    )

    # 4. Sparsity pattern
    sparsity = SparsityPatternWitness(
        pattern_type=SparsityPatternType.LOCAL,
        sequence_length=512,
        total_possible_pairs=512 * 512,
        allowed_pairs=512 * 128,  # Local window of 128
        sparsity_ratio=Fraction(128, 512),
        window_size=128,
        all_tokens_reachable=True,
    )

    # 5. Create certificate
    cert = create_success_certificate(
        certificate_id="bert_base_attention_001",
        num_layers=12,
        num_heads=12,
        sequence_length=512,
        entropy_witnesses=[entropy],
        rank_witnesses=[rank],
        sparsity_witness=sparsity,
        head_redundancy=[head_red],
    )

    print("=== Sparse Attention Certificate ===")
    print(json.dumps(cert.to_dict(), indent=2))
    print(f"\nCertificate hash: {cert.compute_certificate_hash()}")
    print(f"Entropy healthy: {cert.verify_entropy_health()}")
    print(f"Rank healthy: {cert.verify_rank_health()}")
    print(f"Connectivity: {cert.verify_connectivity()}")
