#!/usr/bin/env python3
"""Deterministic, simulated camera-misset sensitivity analysis.

This is a thin reporting wrapper around the existing geometry sensitivity
harness.  It renders known poses with a simulated true calibration, estimates
those same keypoints with a simulated calibration whose height or down-tilt is
perturbed, and compares the estimate with the known simulated pose.  It never
uses a real camera measurement or claims real-world pose accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pallet_pose_compliance.geometry.pose_envelope import (  # noqa: E402
    LONG_RANGE_Y_M,
    MEETS_PASS_FRACTION,
    MIN_EVALUABLE_SAMPLES,
    SHORT_RANGE_Y_M,
    declare_usable_envelope,
    run_sensitivity_analysis,
)
from pallet_pose_compliance.geometry.pose_error import (  # noqa: E402
    ORIENTATION_TOLERANCE_DEG,
    POSITION_TOLERANCE_M,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel  # noqa: E402

_DEFAULT_JSON = _REPO_ROOT / "reports" / "pose_sensitivity.json"
_DEFAULT_MD = _REPO_ROOT / "reports" / "pose_sensitivity.md"


def _condition_dict(condition: Any) -> dict[str, Any]:
    """Serialize one condition with full measured-against-simulated distributions."""
    result = condition.to_dict()
    result["error_report"] = condition.error_report.to_dict()
    result["provenance"] = ProvenanceLabel.SIMULATED.value
    result["comparison"] = "estimated pose vs known simulated pose"
    return result


def run_report(*, seed: int = 42, samples_per_condition: int = 16) -> dict[str, Any]:
    """Run the small, fixed sensitivity grid and return JSON-ready results."""
    report = run_sensitivity_analysis(
        height_error_grid_m=(0.0, 0.10, 0.20),
        tilt_error_grid_deg=(0.0, 3.0, 6.0),
        range_definitions={"short": SHORT_RANGE_Y_M, "long": LONG_RANGE_Y_M},
        n_samples=samples_per_condition,
        seed=seed,
        position_tolerance_m=POSITION_TOLERANCE_M,
        orientation_tolerance_deg=ORIENTATION_TOLERANCE_DEG,
        meets_pass_fraction=MEETS_PASS_FRACTION,
        min_evaluable_samples=MIN_EVALUABLE_SAMPLES,
    )
    envelope = declare_usable_envelope(report)
    conditions = [_condition_dict(c) for c in report.conditions]
    return {
        "provenance": ProvenanceLabel.SIMULATED.value,
        "analysis": "camera height and down-tilt mis-set sensitivity",
        "comparison": "estimated pose against known simulated pose",
        "seed": seed,
        "samples_per_condition": samples_per_condition,
        "condition_count": len(conditions),
        "grids": {
            "height_error_grid_m": list(report.height_error_grid_m),
            "tilt_error_grid_deg": list(report.tilt_error_grid_deg),
        },
        "range_definitions": {k: list(v) for k, v in report.range_definitions.items()},
        "tolerance": {
            "position_tolerance_m": report.position_tolerance_m,
            "orientation_tolerance_deg": report.orientation_tolerance_deg,
            "semantics": "evaluation_target_not_guarantee",
        },
        "conditions": conditions,
        "envelope_partition": {
            "meets_tolerance": len(envelope.usable),
            "fails_tolerance": len(envelope.non_usable),
            "insufficient_evidence": len(envelope.insufficient_evidence),
        },
        "honesty_note": (
            "Every result is simulated. True and assumed calibrations are both "
            "synthetic; no real camera measurement, real pose ground truth, or "
            "real-world compliance claim is included. Only listed grid points "
            "were evaluated; no extrapolation is made."
        ),
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def write_markdown(data: dict[str, Any], path: Path) -> None:
    lines = [
        "# Pose sensitivity analysis (SIMULATED)",
        "",
        "Every row below is **simulated**: a known simulated pose is rendered with a simulated true calibration, then estimated with a simulated calibration containing the listed camera-height or down-tilt error. There is no real camera measurement or real pose ground truth. The ±2 cm / ±3° values are evaluation targets, not guarantees.",
        "",
        f"- Seed: `{data['seed']}`; samples per condition: `{data['samples_per_condition']}`; conditions: `{data['condition_count']}`.",
        f"- Short range: `{data['range_definitions']['short']} m` depth; long range: `{data['range_definitions']['long']} m` depth.",
        f"- Classification: `meets_tolerance` requires combined pass fraction over all evaluated samples ≥ {MEETS_PASS_FRACTION}; fewer than {MIN_EVALUABLE_SAMPLES} samples is `insufficient_evidence`; otherwise `fails_tolerance`.",
        "- No unevaluated height/tilt/range combinations are inferred.",
        "",
        "## Results by evaluated condition",
        "",
        "| error | range | total / available | radial error median / p95 (cm) | rotation median / p95 (deg) | combined pass fraction | classification |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for c in data["conditions"]:
        er = c["error_report"]
        radial = er["translation_error"]["radial_cm"]
        orient = er["rotation_error"]["orientation_deg"]
        tol = er["tolerance_evaluation"]
        amount = f"{c['error_amount']} {c['error_unit']}"
        lines.append(
            f"| {c['error_type']} = {amount} | {c['range_name']} | "
            f"{c['total_samples']} / {c['available_samples']} | "
            f"{_fmt(radial['median'])} / {_fmt(radial['p95'])} | "
            f"{_fmt(orient['median'])} / {_fmt(orient['p95'])} | "
            f"{_fmt(tol['pass_fraction_combined'])} | `{c['label']}` |"
        )
    lines += [
        "",
        "## Distribution fields and interpretation",
        "",
        "The JSON report includes separate `dx_m`, `dy_m`, `radial_m`/`radial_cm`, and `orientation_deg` distributions. Each has a `sample_count`, summary statistics, and the underlying simulated values. `available_samples` and `total_samples` are shown separately; rejected/unavailable poses count against the all-sample classification as required by the existing `pose_envelope` helper.",
        "",
        f"Envelope partition over the evaluated grid: `{data['envelope_partition']}`. This is a classification of these simulated grid points only, not a physical operating envelope and not an extrapolation.",
        "",
        "_Provenance: simulated. No real camera measurement was used._",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-condition", type=int, default=16)
    parser.add_argument("--json-out", default=str(_DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(_DEFAULT_MD))
    args = parser.parse_args(argv)
    if args.samples_per_condition < 1:
        parser.error("--samples-per-condition must be positive")
    data = run_report(seed=args.seed, samples_per_condition=args.samples_per_condition)
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_markdown(data, Path(args.md_out))
    print(json.dumps({"json": str(json_path), "markdown": str(args.md_out), "conditions": data["condition_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
