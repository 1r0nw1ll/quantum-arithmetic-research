"""Source-faithful Pepe Ch. 5 PDE solver map for QA replication.

This is a corrective source-recovery artifact. It does not train a solver.
It records what Pepe's Chapter 5 PDE solvers actually do, then maps each
continuous GA/CGA/STA primitive to the QA primitive that must exist before a
lawful parity experiment can be claimed.

QA_COMPLIANCE = "source_map_only - no PDE solver claim; QA mappings declare observer boundaries"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "corpus" / "pepe_2025" / "2025-pepe.pdf"
OUT_PATH = Path(__file__).resolve().parent / "results_pepe_ch5_pde_solver_source_map.json"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(reader: PdfReader, pages: list[int]) -> dict[str, str]:
    return {
        str(page): normalize(reader.pages[page - 1].extract_text() or "")
        for page in pages
    }


def snippet(text: str, needle: str, radius: int = 360) -> str:
    i = text.lower().find(needle.lower())
    if i < 0:
        return ""
    return text[max(0, i - radius): min(len(text), i + len(needle) + radius)]


def build_map(reader: PdfReader) -> dict[str, object]:
    pages = {
        "ga_relu": extract_pages(reader, [153, 154, 157, 158]),
        "fengbo": extract_pages(reader, [159, 160, 161, 162, 163, 164, 165, 166, 167, 174, 178]),
        "staresnet": extract_pages(reader, [184, 187, 188, 189, 191, 192, 194, 197, 198, 199]),
    }

    return {
        "ok": True,
        "schema": "QA_ML_PEPE_CH5_PDE_SOLVER_SOURCE_MAP.v1",
        "source_pdf": str(PDF_PATH.relative_to(ROOT)),
        "claim_boundary": (
            "This artifact maps Pepe Ch. 5 PDE solvers to QA replication tasks. "
            "It is not a PDE solver run and does not claim Navier-Stokes, Fengbo, "
            "or Maxwell replication."
        ),
        "source_anchors": {
            "ga_relu": {
                "pages": [153, 154, 157, 158],
                "snippets": {
                    "problem": snippet(pages["ga_relu"]["153"], "The incompressible Navier-Stokes equations in 2D"),
                    "activation": snippet(pages["ga_relu"]["154"], "GA-ReLU is the composition"),
                    "data": snippet(pages["ga_relu"]["157"], "regular square grid of size128"),
                    "results": snippet(pages["ga_relu"]["158"], "albeit small, the improvement from GA-ReLU is consistent"),
                },
            },
            "fengbo": {
                "pages": [159, 160, 161, 162, 163, 164, 165, 166, 167, 174, 178],
                "snippets": {
                    "architecture": snippet(pages["fengbo"]["159"], "Fengbo has three main components"),
                    "voxelization": snippet(pages["fengbo"]["161"], "generating a regular grid ofM"),
                    "pressure_packet": snippet(pages["fengbo"]["161"], "We construct multivectors P"),
                    "velocity_packet": snippet(pages["fengbo"]["162"], "we construct a corresponding multivector V"),
                    "fno": snippet(pages["fengbo"]["164"], "The 3D Fourier Neural Operator"),
                    "datasets": snippet(pages["fengbo"]["165"], "ShapeNet Car"),
                    "metrics": snippet(pages["fengbo"]["166"], "relativeL2 norm"),
                    "main_results": snippet(pages["fengbo"]["167"], "Table 5.2"),
                    "grid_ablation": snippet(pages["fengbo"]["174"], "Table 5.7"),
                    "loss_weight_ablation": snippet(pages["fengbo"]["178"], "Table 5.8"),
                },
            },
            "staresnet": {
                "pages": [184, 187, 188, 189, 191, 192, 194, 197, 198, 199],
                "snippets": {
                    "problem": snippet(pages["staresnet"]["184"], "Maxwell"),
                    "faraday": snippet(pages["staresnet"]["187"], "Faraday bivectorF"),
                    "regression": snippet(pages["staresnet"]["188"], "spacetime bivector-to-bivector regression"),
                    "architecture": snippet(pages["staresnet"]["188"], "2D Clifford ResNet"),
                    "training": snippet(pages["staresnet"]["189"], "Both networks have been trained"),
                    "2d_data": snippet(pages["staresnet"]["191"], "surface with spatial resolution"),
                    "obstacles": snippet(pages["staresnet"]["194"], "Impact of obstacles"),
                    "parameter_ablation": snippet(pages["staresnet"]["197"], "To verify that STAResNet"),
                    "rollout": snippet(pages["staresnet"]["198"], "rollout refers"),
                    "3d_data": snippet(pages["staresnet"]["199"], "In the 3D case"),
                },
            },
        },
        "solver_maps": [
            {
                "id": "PDE-1",
                "name": "GA-ReLU Navier-Stokes",
                "pepe_target": {
                    "equation": "2D incompressible Navier-Stokes",
                    "input": "two consecutive multivector frames x_t, x_t+dt with scalar plus 2-vector coefficients",
                    "target": "x_t+2dt",
                    "architecture": "Clifford ResNet and Clifford FNO in G(2,0) with ReLU vs GA-ReLU",
                    "dataset": "PhiFlow 128x128 grid, dx=dy=0.25, viscosity=0.01, buoyancy=0.05, 21s simulated, dt=1.5s, tau0=4s",
                    "metrics": ["scalar MSE", "vector MSE", "one-step MSE", "scalar/vector error-field visuals"],
                },
                "qa_mapping": {
                    "status": "MAPPED_REQUIRES_SOLVER_REPLICA",
                    "lawful_entry": "PDE frames are observer tensors; QA acts on the multivector activation/packet boundary.",
                    "primitive": "QA-GA-ReLU: scalar ReLU plus phase/cardioid gate on quantized vector packet.",
                    "implementation_steps": [
                        "replicate a small continuous Clifford ResNet/FNO baseline on a generated 2D Navier-Stokes toy",
                        "replace only the activation with QA phase/cardioid gating over quantized vector phase packets",
                        "compare ReLU, continuous GA-ReLU, and QA-GA-ReLU on identical train/test splits",
                    ],
                    "parity_criterion": "QA-GA-ReLU scalar/vector/one-step MSE within declared tolerance of continuous GA-ReLU and better than coefficient-wise ReLU on at least one low-data regime.",
                },
            },
            {
                "id": "PDE-2",
                "name": "Fengbo irregular-geometry neural operator",
                "pepe_target": {
                    "equation": "steady-state 3D CFD / Navier-Stokes geometry-to-pressure/velocity mapping",
                    "input": "irregular vehicle geometry voxelized to MxMxM multivector volumes in G(3,0,0)",
                    "pressure_geometry_packet": "P = mask + 3D coordinate vector + normal-dual bivector; trivector blank except Ahmed inlet velocity",
                    "velocity_geometry_packet": "V = mask + 3D coordinate vector",
                    "architecture": "3D Clifford Geometry blocks -> full-grade 3D Clifford FNO -> 3D Clifford Physics blocks",
                    "datasets": {
                        "ShapeNet Car": "500 train / 111 test; pressure and velocity fields",
                        "Ahmed Body": "500 train / 51 test; pressure with inlet velocity encoded in trivector",
                    },
                    "metrics": ["relative L2 pressure", "relative L2 velocity", "grid-size ablation", "C/F/Fourier-mode/model-size ablations", "loss-weight alpha/beta ablation"],
                },
                "qa_mapping": {
                    "status": "MAPPED_MISSING_PRIMITIVE_NOT_PARKED",
                    "lawful_entry": (
                        "Fengbo already converts irregular geometry into fixed-resolution voxel multivectors. "
                        "QA may adopt that GA/CGA construction by quantizing coordinate, normal, mask, and field "
                        "coefficients into exact rational/modular packets before observer decode."
                    ),
                    "required_primitives": [
                        "qa_cga_grid_packet_v1: exact rational/integer-mapped voxel coordinates (i,j,k,M) plus dequantization metadata",
                        "qa_fengbo_geometry_multivector_v1: mask + vector coordinate packet + normal-dual bivector packet + optional inlet trivector",
                        "qa_clifford3_product_quantized_v1: parity-tested quantized G(3,0,0) geometric product against continuous coefficients",
                        "qa_fengbo_operator_v1: geometry-block/FNO/physics-block skeleton where quantization occurs at packet boundaries",
                    ],
                    "implementation_steps": [
                        "build a synthetic mini-Fengbo dataset first: spheres/ellipsoids/cars as voxelized masks with analytic pressure-like scalar and velocity-like vector fields",
                        "implement continuous mini-Fengbo baseline over the same voxel packets",
                        "implement QA-Fengbo by quantizing P and V packets and Clifford-product coefficients at modulus/resolution m",
                        "sweep grid M and QA modulus m; compare relative L2 pressure/velocity against continuous mini-Fengbo",
                        "only after mini parity, move to ShapeNet Car/Ahmed Body acquisition and full-size Fengbo reproduction",
                    ],
                    "parity_criterion": "QA-Fengbo relative L2 within an absolute +0.5 to +1.0 percentage-point band of continuous mini-Fengbo at matched parameter count, with monotone improvement as M or m increases.",
                },
            },
            {
                "id": "PDE-3",
                "name": "STAResNet Maxwell",
                "pepe_target": {
                    "equation": "Maxwell fields as Faraday bivector regression",
                    "input": "two consecutive Faraday bivectors F_i, F_i+dt",
                    "target": "F_i+2dt",
                    "architectures": ["Clifford ResNet in G(n,0,0)", "STAResNet in G(1,n,0)"],
                    "datasets": {
                        "2D": "32x32 FDTD, dt in {25,50,75,100}s; obstacle case 48x48, seen/unseen obstacles",
                        "3D": "28x28x28 FDTD, dt in {5,8,10,15}s",
                    },
                    "metrics": ["MSE", "correlation/SSIM", "seen vs unseen obstacle MSE", "rollout MSE over m=1..10", "parameter-count ablation"],
                },
                "qa_mapping": {
                    "status": "MAPPED_REQUIRES_FIELD_PACKET_AND_ROLLOUT",
                    "lawful_entry": "Maxwell fields remain observer-side tensors; QA packetizes Faraday bivector components and algebra choice, then decodes for residual/MSE evaluation.",
                    "required_primitives": [
                        "qa_faraday_packet_v1: quantized E/B or Faraday bivector coefficients with STA signature metadata",
                        "qa_sta_residual_block_v1: generator/residual block whose channels respect the chosen QA/STA packet",
                        "qa_rollout_eval_v1: repeated predicted-frame feeding with MSE/SSIM/residual ledger",
                    ],
                    "implementation_steps": [
                        "start with a tiny analytic wave/FDTD fixture to reproduce the Clifford-vs-STA algebra-choice comparison",
                        "compare continuous Clifford ResNet, continuous STAResNet, and QA-STA packetized residual model at matched parameter count",
                        "add obstacle generalization and rollout only after one-step parity is stable",
                    ],
                    "parity_criterion": "QA-STA model matches continuous STAResNet ordering: lower MSE than Clifford baseline at comparable parameter count and bounded rollout error relative to continuous STA.",
                },
            },
        ],
        "corrected_status": {
            "previous_ch5_visual_replica": "useful scaffold only; not a PDE solver replication",
            "fengbo": "not parked; mapped as missing QA-Fengbo primitive",
            "next_code_task": "implement qa_cga_grid_packet_v1 plus a synthetic mini-Fengbo parity benchmark before claiming Fengbo replication",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if not PDF_PATH.exists():
        print(canonical_json({"ok": False, "error": f"missing_pdf:{PDF_PATH}"}))
        return 1

    reader = PdfReader(str(PDF_PATH))
    result = build_map(reader)

    if args.self_test:
        checks = {
            "ok": bool(result["ok"]),
            "has_fengbo_mapping": result["solver_maps"][1]["qa_mapping"]["status"] == "MAPPED_MISSING_PRIMITIVE_NOT_PARKED",
            "has_three_solver_maps": len(result["solver_maps"]) == 3,
        }
        checks["ok"] = all(checks.values())
        print(canonical_json(checks))
        return 0 if checks["ok"] else 1

    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json({
        "ok": True,
        "path": str(OUT_PATH.relative_to(ROOT)),
        "solvers": [item["id"] for item in result["solver_maps"]],
        "fengbo_status": result["solver_maps"][1]["qa_mapping"]["status"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
