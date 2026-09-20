"""REAL (simulated-GT) pose-accuracy evaluation of the trained pose model.

Runs the trained YOLO-pose model on the held-out TEST split of the SIMULATED
pose dataset, feeds its predicted keypoints (as named ``Keypoint``s) into the
known-geometry PnP estimator (:func:`estimate_pose_with_uncertainty`) with the
SIMULATED ``sim-cal-v0`` calibration and the nominal pallet model, and compares
the estimated Floor_Frame pose against the *self-constructed simulated ground
truth* stored in ``data/synthetic_pose/ground_truth.json``.

Honesty discipline
------------------
- The ground truth is SIMULATED (rendered by projection), NOT measured, and NOT
  derived from the estimator under test. Comparing the estimator against it is a
  genuine test (forward render vs. inverse PnP are independent code paths).
- Predictions that the estimator degrades to ``unavailable`` are **counted**,
  never silently dropped. The unavailable/failure rate is reported.
- The ±2 cm / ±3 deg tolerance is an evaluation TARGET, not a guarantee. This is
  stated in the output and enforced by reusing
  :class:`~pallet_pose_compliance.geometry.pose_error.PoseErrorReport` (which
  fixes ``tolerance_semantics = evaluation_target_not_guarantee``).
- Everything is provenance ``simulated``.

Reuse
-----
Error distributions + tolerance pass-fractions are computed by the existing
:func:`~pallet_pose_compliance.geometry.pose_error.compute_pose_error_report`
over :class:`~pallet_pose_compliance.geometry.pose_eval.PoseEvalSample` pairs, so
this script does not duplicate distribution/tolerance math. Usable-envelope
classification is reused from
:mod:`~pallet_pose_compliance.geometry.pose_envelope` via
:func:`~pallet_pose_compliance.geometry.pose_envelope.classify_condition`.

Outputs
-------
- ``reports/pose_eval_metrics.json`` — distributions (translation cm + rotation
  deg, per-axis/radial), tolerance pass-fractions, sample + unavailable counts,
  provenance simulated.
- ``reports/pose_eval_notes.md`` — short method + results + caveats summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Ensure the src-layout package is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pallet_pose_compliance.calibration.calibration import (  # noqa: E402
    build_synthetic_calibration,
    load_calibration,
)
from pallet_pose_compliance.detection.detector import (  # noqa: E402
    KEYPOINT_NAMES,
    YoloPoseDetector,
)
from pallet_pose_compliance.geometry.pallet_model import (  # noqa: E402
    build_nominal_pallet_model,
)
from pallet_pose_compliance.geometry.pose import (  # noqa: E402
    estimate_pose_with_uncertainty,
)
from pallet_pose_compliance.geometry.pose_error import (  # noqa: E402
    ORIENTATION_TOLERANCE_DEG,
    POSITION_TOLERANCE_M,
    compute_pose_error_report,
)
from pallet_pose_compliance.geometry.pose_eval import (  # noqa: E402
    GroundTruthPose,
    PoseEvalSample,
)
from pallet_pose_compliance.geometry.pose_envelope import (  # noqa: E402
    MEETS_PASS_FRACTION,
    MIN_EVALUABLE_SAMPLES,
    classify_condition,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel  # noqa: E402
from pallet_pose_compliance.output.schema import Keypoint  # noqa: E402

_DEFAULT_DATASET = _REPO_ROOT / "data" / "synthetic_pose"
_DEFAULT_WEIGHTS = _REPO_ROOT / "weights" / "yolo26-pose-pallet-v0.pt"
_DEFAULT_CALIBRATION_ID = "sim-cal-v0"
_DEFAULT_METRICS = _REPO_ROOT / "reports" / "pose_eval_metrics.json"
_DEFAULT_NOTES = _REPO_ROOT / "reports" / "pose_eval_notes.md"

__all__ = ["run_pose_eval", "main"]


def _load_ground_truth(dataset_dir: Path) -> dict[str, Any]:
    gt_path = dataset_dir / "ground_truth.json"
    if not gt_path.exists():
        raise RuntimeError(
            f"[POSE EVAL BLOCKED] ground truth not found at {gt_path!s}; run "
            "scripts/generate_synthetic_pose.py first."
        )
    return json.loads(gt_path.read_text(encoding="utf-8"))


def _pick_pallet_detection(detections: list) -> Optional[Any]:
    """Choose the highest-score pallet detection with keypoints, else None."""
    pallets = [
        d for d in detections if d.cls == "pallet" and len(d.keypoints) == len(KEYPOINT_NAMES)
    ]
    if not pallets:
        return None
    return max(pallets, key=lambda d: float(d.score))


def _no_prediction_gt_and_sample(gt_entry: dict[str, Any]) -> PoseEvalSample:
    """Build a sample whose prediction is an honest 'unavailable' (no detection)."""
    from pallet_pose_compliance.geometry.pose_stub import estimate_pose_stub

    gt = _gt_from_entry(gt_entry)
    return PoseEvalSample(ground_truth=gt, prediction=estimate_pose_stub())


def _gt_from_entry(gt_entry: dict[str, Any]) -> GroundTruthPose:
    """Reconstruct a GroundTruthPose from a ground_truth.json sample entry."""
    pose = gt_entry["pose"]
    names = gt_entry.get("keypoint_names", list(KEYPOINT_NAMES))
    kpts_px = gt_entry["keypoints_px"]
    keypoints = tuple(
        Keypoint(name=name, u=float(uv[0]), v=float(uv[1]), visibility="visible", score=0.9)
        for name, uv in zip(names, kpts_px)
    )
    return GroundTruthPose(
        x_m=float(pose["x_m"]),
        y_m=float(pose["y_m"]),
        theta_deg=float(pose["theta_deg"]),
        keypoints=keypoints,
        symmetry_order=1,
    )


def run_pose_eval(
    *,
    dataset_dir: Path = _DEFAULT_DATASET,
    weights_path: Path = _DEFAULT_WEIGHTS,
    calibration_id: str = _DEFAULT_CALIBRATION_ID,
    seed: int = 42,
    conf: float = 0.25,
    imgsz: int = 640,
    n_uncertainty_samples: int = 64,
) -> dict[str, Any]:
    """Run the trained model on TEST, estimate poses, compare to simulated GT."""
    import cv2

    ground_truth = _load_ground_truth(dataset_dir)
    test_items = {
        rel: entry
        for rel, entry in ground_truth["samples"].items()
        if entry.get("split") == "test"
    }
    if not test_items:
        raise RuntimeError("[POSE EVAL BLOCKED] no TEST-split samples in ground truth")

    # SIMULATED calibration + nominal pallet model.
    try:
        calibration = load_calibration(calibration_id)
    except Exception as exc:
        calibration = build_synthetic_calibration(calibration_id)
        print(f"[warn] load_calibration failed ({exc}); using synthetic fallback", file=sys.stderr)
    pallet_model = build_nominal_pallet_model()

    detector = YoloPoseDetector(weights_path=str(weights_path), imgsz=imgsz, conf=conf)

    samples: list[PoseEvalSample] = []
    no_detection = 0
    detected = 0
    for rel, entry in sorted(test_items.items()):
        image_path = dataset_dir / rel
        image = cv2.imread(str(image_path))
        if image is None:
            # Missing image file: count as no-detection (honest).
            samples.append(_no_prediction_gt_and_sample(entry))
            no_detection += 1
            continue

        detections = detector.detect(image)
        det = _pick_pallet_detection(detections)
        if det is None:
            samples.append(_no_prediction_gt_and_sample(entry))
            no_detection += 1
            continue

        detected += 1
        gt = _gt_from_entry(entry)
        # Predicted keypoints are already named bottom_corner_0..3/top_corner_0..3
        # in the canonical order by the detector wrapper (matches generator order).
        prediction = estimate_pose_with_uncertainty(
            list(det.keypoints),
            calibration,
            pallet_model,
            symmetry_order=gt.symmetry_order,
            n_samples=n_uncertainty_samples,
            seed=seed,
        )
        samples.append(PoseEvalSample(ground_truth=gt, prediction=prediction))

    report = compute_pose_error_report(
        samples,
        position_tolerance_m=POSITION_TOLERANCE_M,
        orientation_tolerance_deg=ORIENTATION_TOLERANCE_DEG,
    )

    total = len(samples)
    # Coverage over TEST: how many produced an available pose end-to-end.
    available = report.available_samples
    unavailable = report.unavailable_samples

    # Reuse the envelope classifier as a single overall usable/not-usable label
    # over the whole TEST set (a single condition).
    overall_label = classify_condition(
        report,
        meets_pass_fraction=MEETS_PASS_FRACTION,
        min_evaluable_samples=MIN_EVALUABLE_SAMPLES,
    )

    metrics = {
        "provenance": ProvenanceLabel.SIMULATED.value,
        "evaluator": "scripts/eval_pose.py",
        "seed": seed,
        "weights": str(Path(weights_path).resolve()),
        "calibration_id": calibration_id,
        "calibration_provenance": calibration.provenance.value,
        "pallet_model_provenance": pallet_model.provenance.value,
        "method": (
            "Trained YOLO-pose predicts 8 pallet corner keypoints on held-out "
            "SIMULATED TEST images; keypoints (named bottom_corner_0..3 / "
            "top_corner_0..3 in generator order) feed estimate_pose_with_"
            "uncertainty with sim-cal-v0 + nominal pallet model; estimated "
            "Floor_Frame pose compared to self-constructed SIMULATED GT "
            "(projection-based, independent of the estimator under test)."
        ),
        "tolerance": {
            "position_tolerance_cm": POSITION_TOLERANCE_M * 100.0,
            "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
            "semantics": "evaluation_target_not_guarantee",
        },
        "counts": {
            "test_samples": total,
            "yolo_detected": detected,
            "yolo_no_detection": no_detection,
            "pose_available": available,
            "pose_unavailable": unavailable,
            "unavailable_rate": (unavailable / total) if total else None,
            "coverage_fraction": report.coverage_fraction,
        },
        "translation_error": {
            "radial_cm": report.error_radial_cm.to_dict(),
            "dx_m": report.error_dx_m.to_dict(),
            "dy_m": report.error_dy_m.to_dict(),
            "radial_m": report.error_radial_m.to_dict(),
        },
        "rotation_error": {
            "orientation_deg": report.error_orientation_deg.to_dict(),
        },
        "tolerance_evaluation": report.tolerance.to_dict(),
        "overall_envelope_label": overall_label.value,
        "honesty_note": (
            "SIMULATED ground truth (never measured). Tolerance is a target being "
            "evaluated against, not a guarantee. Unavailable predictions are "
            "counted, not dropped. Trained/evaluated on simple synthetic renders; "
            "there is a synthetic-to-real gap so these numbers do NOT establish "
            "real-world compliance."
        ),
    }
    return metrics


def _write_notes(metrics: dict[str, Any], notes_path: Path) -> None:
    c = metrics["counts"]
    tol = metrics["tolerance_evaluation"]
    rad = metrics["translation_error"]["radial_cm"]
    rot = metrics["rotation_error"]["orientation_deg"]

    def fmt(x: Any) -> str:
        return "n/a" if x is None else f"{x:.4g}"

    lines = [
        "# Pose evaluation notes (SIMULATED ground truth)",
        "",
        "## Method",
        "",
        "- Ground truth is **self-constructed and SIMULATED**: the nominal pallet "
        "model is placed at a known Floor_Frame pose `(x, y, theta)` and projected "
        "through the SIMULATED `sim-cal-v0` calibration with `cv2.projectPoints` "
        "(see `scripts/generate_synthetic_pose.py`, sidecar "
        "`data/synthetic_pose/ground_truth.json`).",
        "- GT is **never** read from the estimator under test. The forward model "
        "(place + project) and the inverse model (`estimate_pose_with_uncertainty`: "
        "keypoints -> pose via known-geometry PnP) are independent code paths.",
        "- The trained model `weights/yolo26-pose-pallet-v0.pt` predicts the 8 "
        "pallet corner keypoints on each held-out TEST image. Predicted keypoints "
        "(named `bottom_corner_0..3` / `top_corner_0..3` in the generator's order) "
        "feed the PnP estimator with `sim-cal-v0` + the nominal pallet model.",
        "- Estimated Floor_Frame pose is compared to the simulated GT. Translation "
        "error (per-axis + radial, cm) and rotation error (deg, folded modulo the "
        "long-axis ambiguity) are reported as distributions with sample counts via "
        "the existing `pose_error.compute_pose_error_report` helper.",
        "",
        "## Results (SIMULATED)",
        "",
        f"- TEST samples: **{c['test_samples']}** "
        f"(YOLO detected: {c['yolo_detected']}, no detection: {c['yolo_no_detection']}).",
        f"- Pose available: **{c['pose_available']}**, unavailable: "
        f"**{c['pose_unavailable']}** (unavailable rate {fmt(c['unavailable_rate'])}).",
        f"- Radial translation error (cm): median {fmt(rad['median'])}, "
        f"p95 {fmt(rad.get('p95'))}, max {fmt(rad.get('max'))}, "
        f"n={rad.get('sample_count')}.",
        f"- Rotation error (deg): median {fmt(rot['median'])}, "
        f"p95 {fmt(rot.get('p95'))}, max {fmt(rot.get('max'))}, "
        f"n={rot.get('sample_count')}.",
        "",
        "### Tolerance pass-fractions (target ±2 cm / ±3°, NOT a guarantee)",
        "",
        f"- Radial translation within ±2 cm: **{fmt(tol['pass_fraction_radial'])}**",
        f"- Per-axis dx within ±2 cm: {fmt(tol['pass_fraction_dx'])}; "
        f"dy within ±2 cm: {fmt(tol['pass_fraction_dy'])}",
        f"- Orientation within ±3°: **{fmt(tol['pass_fraction_orientation'])}**",
        f"- Combined (radial AND orientation): **{fmt(tol['pass_fraction_combined'])}**",
        f"- Overall usable-envelope label (whole TEST set as one condition): "
        f"**{metrics['overall_envelope_label']}**",
        "",
        "## Honest caveats",
        "",
        "- All ground truth is **simulated**, never measured. The tolerance is an "
        "evaluation **target**, not a guarantee, and these numbers do **not** "
        "establish real-world compliance.",
        "- Training and evaluation use **simple synthetic renders** (filled/edged "
        "quadrilaterals), not photorealistic imagery. There is a substantial "
        "**synthetic-to-real gap**; real-camera pose accuracy will differ and must "
        "be measured separately on real, measured ground truth.",
        "- Unavailable predictions (no detection, or the PnP estimator degrading "
        "loudly) are **counted**, not dropped, so the pass-fractions are computed "
        "only over available poses while coverage/unavailable rate is reported "
        "alongside.",
        "",
        f"_Provenance: {metrics['provenance']}. Seed: {metrics['seed']}. "
        f"Weights: {metrics['weights']}. Calibration: {metrics['calibration_id']} "
        f"({metrics['calibration_provenance']})._",
        "",
    ]
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(lines), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained pose model accuracy on the SIMULATED TEST split "
            "against self-constructed simulated GT (translation cm + rotation "
            "deg distributions, ±2cm/±3deg target pass-fractions, unavailable "
            "rate). Provenance simulated."
        )
    )
    parser.add_argument("--dataset-dir", default=str(_DEFAULT_DATASET))
    parser.add_argument("--weights", default=str(_DEFAULT_WEIGHTS))
    parser.add_argument("--calibration-id", default=_DEFAULT_CALIBRATION_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--metrics-out", default=str(_DEFAULT_METRICS))
    parser.add_argument("--notes-out", default=str(_DEFAULT_NOTES))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        metrics = run_pose_eval(
            dataset_dir=Path(args.dataset_dir),
            weights_path=Path(args.weights),
            calibration_id=args.calibration_id,
            seed=args.seed,
            conf=args.conf,
            imgsz=args.imgsz,
            n_uncertainty_samples=args.mc_samples,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_notes(metrics, Path(args.notes_out))

    c = metrics["counts"]
    tol = metrics["tolerance_evaluation"]
    rad = metrics["translation_error"]["radial_cm"]
    rot = metrics["rotation_error"]["orientation_deg"]
    print(json.dumps(metrics, indent=2))
    print(
        f"[POSE EVAL simulated] test={c['test_samples']} available={c['pose_available']} "
        f"unavailable={c['pose_unavailable']} (rate {c['unavailable_rate']}); "
        f"radial_cm median={rad['median']} p95={rad.get('p95')}; "
        f"rot_deg median={rot['median']} p95={rot.get('p95')}; "
        f"pass_radial={tol['pass_fraction_radial']} pass_orient={tol['pass_fraction_orientation']} "
        f"pass_combined={tol['pass_fraction_combined']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
