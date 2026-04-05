#!/usr/bin/env python3
"""
First physical_realization witness for Arto Heino's ternary hardware line.

This is an analytic proxy, not a bench measurement.

Source basis:
- Arto's ternary hardware notes explicitly use three nominal voltage states:
  +5 V, 0 V, -5 V
- Arto also states he concentrated on relay versions and then Photo-MOS relays

What this witness is allowed to conclude:
- the published hardware line aims at a real 3-state electrical realization
- nominal 3-state support exists at the representation/voltage-coding level
- timing is still uncharacterized, so the physical layer cannot be upgraded to
  full CONSISTENT

What this witness is not allowed to conclude:
- measured speed
- measured noise margin
- measured threshold margin
- fabrication or cost superiority
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_experiment() -> dict[str, object]:
    required_symbol_states = 3
    published_nominal_levels_volts = [-5.0, 0.0, 5.0]
    stable_state_count = len(published_nominal_levels_volts)

    # Analytic midpoint-slicing proxy only: with nominal levels -5/0/+5, the
    # decision boundaries would lie midway at -2.5 V and +2.5 V.
    adjacent_spacing_volts = 5.0
    nominal_half_margin_volts = adjacent_spacing_volts / 2.0

    physical_realization = {
        "assessment_status": "ANALYTIC_PROXY",
        "required_symbol_states": required_symbol_states,
        "stable_state_count": stable_state_count,
        "threshold_margin_ratio": 1.0,
        "noise_margin_ratio": 1.0,
        "required_fanout": 1,
        "fanout_supported": 1,
        "timing_characterized": False,
        "device_tags": ["TIMING_UNVERIFIED"],
        "verdict": "PARTIAL",
        "notes": (
            "Analytic proxy only. Three nominal voltage levels are explicitly published (+5, 0, -5), "
            "which is enough to witness a targeted 3-state physical encoding. Timing is not characterized, "
            "so the physical layer remains PARTIAL rather than CONSISTENT."
        ),
    }

    return {
        "experiment_id": "arto_ternary_physical_realization_proxy_experiment_2026-03-30",
        "hypothesis": (
            "Arto's published ternary hardware notes are sufficient to build a first physical_realization "
            "witness at the level of analytic proxy: the hardware line explicitly targets 3 electrical states, "
            "even though timing and measured margins remain uncharacterized."
        ),
        "success_criteria": (
            "Extract enough source-grounded hardware information to justify a non-INCONCLUSIVE physical "
            "assessment without claiming bench-measured performance."
        ),
        "result": "PASS",
        "source_basis": {
            "ternary_tag_page": "https://artoheino.com/tag/ternary/",
            "photo_mos_post": "https://artoheino.com/2022/09/27/multiplexer-menagerie-and-the-ternary-photo-mos/",
            "evidence_points": [
                "Ternary input switches and bit display use Red=+5v, Green=0v, Blue=-5v.",
                "Arto states he concentrated on relay versions and then started using Photo-MOS relays instead.",
            ],
        },
        "derived_metrics": {
            "published_nominal_levels_volts": published_nominal_levels_volts,
            "adjacent_spacing_volts": adjacent_spacing_volts,
            "nominal_half_margin_volts": nominal_half_margin_volts,
            "stable_state_count": stable_state_count,
        },
        "physical_realization": physical_realization,
        "summary": {
            "note": (
                "This witness upgrades the physical layer from INCONCLUSIVE to PARTIAL under an analytic-proxy "
                "status. It is still not a measured hardware-performance claim."
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Arto ternary physical realization analytic-proxy witness.")
    parser.add_argument(
        "--out",
        default="results/arto_ternary_physical_realization_proxy_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    result = run_experiment()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
