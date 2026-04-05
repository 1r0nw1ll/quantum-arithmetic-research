#!/usr/bin/env python3
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import qa_modal_reachability as qmr


def test_missing_extrinsics_blocks_pose_transform() -> None:
    spec = copy.deepcopy(qmr.load_spec(qmr.SPEC_PATH))
    spec["states"]["LiDAR"]["capabilities"]["has_extrinsics"] = False
    cert = qmr.find_reachability_certificate(
        spec,
        source="S_L",
        target="S_H",
        avoid_may_fail=False,
    )
    assert cert["fail_type"] == "GENERATOR_INSUFFICIENT"
    assert "has_extrinsics" in cert["required_missing_information"]


def test_projection_only_basin_is_non_identifiable() -> None:
    spec = {
        "states": {
            "Source": {"id": "S_SRC", "capabilities": {}},
            "Target": {"id": "S_TGT", "capabilities": {}},
        },
        "generators": {
            "project_lossy": {
                "id": "gen_project_lossy",
                "domain": "Source",
                "codomain": "reduced_state",
                "irreversible": True,
            }
        },
        "failures": {
            "NON_IDENTIFIABLE": {"description": "Projection only", "terminal": True},
            "GENERATOR_INSUFFICIENT": {"description": "Missing caps", "terminal": True},
            "UNREACHABLE": {"description": "No path", "terminal": True},
        },
        "certificates": {
            "RETURN_CONSTRUCTED": {
                "fields": {
                    "source_state": "string",
                    "target_state": "string",
                    "path": {"type": "list", "elements": "generator_id"},
                    "preconditions_met": "list",
                    "invariants_preserved": "list",
                    "error_bounds": {"type": "numeric"},
                    "notes": "optional",
                }
            },
            "CYCLE_IMPOSSIBLE": {
                "fields": {
                    "attempted_goal": "string",
                    "required_missing_information": "list",
                    "fail_type": "failure_id",
                    "invariant_difference": "list",
                    "notes": "optional",
                }
            },
        },
    }
    cert = qmr.find_reachability_certificate(
        spec,
        source="S_SRC",
        target="S_TGT",
        avoid_may_fail=False,
    )
    assert cert["fail_type"] == "NON_IDENTIFIABLE"


def test_unrelated_nonreducing_edge_still_non_identifiable() -> None:
    spec = {
        "states": {
            "Source": {"id": "S_SRC", "capabilities": {}},
            "Detour": {"id": "S_DET", "capabilities": {}},
            "Target": {"id": "S_TGT", "capabilities": {}},
        },
        "generators": {
            "detour": {
                "id": "gen_refine",
                "domain": "Source",
                "codomain": "Detour",
            },
            "project_lossy": {
                "id": "gen_project_lossy",
                "domain": "Source",
                "codomain": "reduced_state",
                "irreversible": True,
            },
        },
        "failures": {
            "NON_IDENTIFIABLE": {"description": "Projection only", "terminal": True},
            "GENERATOR_INSUFFICIENT": {"description": "Missing caps", "terminal": True},
            "UNREACHABLE": {"description": "No path", "terminal": True},
        },
        "certificates": {
            "RETURN_CONSTRUCTED": {
                "fields": {
                    "source_state": "string",
                    "target_state": "string",
                    "path": {"type": "list", "elements": "generator_id"},
                    "preconditions_met": "list",
                    "invariants_preserved": "list",
                    "error_bounds": {"type": "numeric"},
                    "notes": "optional",
                }
            },
            "CYCLE_IMPOSSIBLE": {
                "fields": {
                    "attempted_goal": "string",
                    "required_missing_information": "list",
                    "fail_type": "failure_id",
                    "invariant_difference": "list",
                    "notes": "optional",
                }
            },
        },
    }
    cert = qmr.find_reachability_certificate(
        spec,
        source="S_SRC",
        target="S_TGT",
        avoid_may_fail=False,
    )
    assert cert["fail_type"] == "NON_IDENTIFIABLE"


def test_avoid_may_fail_blocks_path() -> None:
    spec = {
        "states": {
            "Source": {"id": "S_SRC", "capabilities": {}},
            "Target": {"id": "S_TGT", "capabilities": {}},
        },
        "generators": {
            "may_fail_bridge": {
                "id": "gen_may_fail_bridge",
                "domain": "Source",
                "codomain": "Target",
                "may_fail": ["PHYSICS_MISMATCH"],
            }
        },
        "failures": {
            "PHYSICS_MISMATCH": {"description": "May-fail edge", "terminal": True},
            "NON_IDENTIFIABLE": {"description": "Projection only", "terminal": True},
            "GENERATOR_INSUFFICIENT": {"description": "Missing caps", "terminal": True},
            "UNREACHABLE": {"description": "No path", "terminal": True},
        },
        "certificates": {
            "RETURN_CONSTRUCTED": {
                "fields": {
                    "source_state": "string",
                    "target_state": "string",
                    "path": {"type": "list", "elements": "generator_id"},
                    "preconditions_met": "list",
                    "invariants_preserved": "list",
                    "error_bounds": {"type": "numeric"},
                    "notes": "optional",
                }
            },
            "CYCLE_IMPOSSIBLE": {
                "fields": {
                    "attempted_goal": "string",
                    "required_missing_information": "list",
                    "fail_type": "failure_id",
                    "invariant_difference": "list",
                    "notes": "optional",
                }
            },
        },
    }
    cert = qmr.find_reachability_certificate(
        spec,
        source="S_SRC",
        target="S_TGT",
        avoid_may_fail=False,
    )
    assert cert["target_state"] == "S_TGT"
    cert = qmr.find_reachability_certificate(
        spec,
        source="S_SRC",
        target="S_TGT",
        avoid_may_fail=True,
    )
    assert cert["fail_type"] == "UNREACHABLE"


def main() -> None:
    test_missing_extrinsics_blocks_pose_transform()
    test_projection_only_basin_is_non_identifiable()
    test_unrelated_nonreducing_edge_still_non_identifiable()
    test_avoid_may_fail_blocks_path()
    print("qa_modal_reachability_smoke: ok")


if __name__ == "__main__":
    main()
