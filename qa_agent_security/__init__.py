# qa_agent_security — QA-native agent security kernel
# Certificate-gated tool execution with taint tracking and obstruction ledger

from .qa_agent_security import (
    # Canonical JSON
    canonical_json_dumps,
    canonical_json_sha256,
    now_rfc3339,
    # Provenance / taint
    TAINTED,
    TRUSTED,
    VALID_SOURCES,
    Prov,
    pv,
    is_tainted,
    prov_source,
    # Taint flow
    mint_taint_flow_cert,
    # Merkle trace
    MerkleLeaf,
    MerkleTrace,
    merkle_root,
    # Tool spec
    ToolSpec,
    CRITICAL_FIELDS,
    # Capability token
    CapabilityEntry,
    CapabilityToken,
    # Policy kernel
    PolicyError,
    FAIL_TYPES,
    enforce_policy,
    obstruction_from_policy_error,
)
