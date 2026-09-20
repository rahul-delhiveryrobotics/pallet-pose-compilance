"""Synthetic YOLO-pose dataset generator (SIMULATED provenance).

Generates a learnable, *simulated* keypoint-pose dataset for training a
YOLO-pose model to localise the eight pallet corners, plus an exact
ground-truth sidecar so the pose EVALUATION (``scripts/eval_pose.py``) can
compare the known-geometry PnP estimator's output against a *self-constructed*
ground truth that is entirely independent of the estimator under test.

Method (honest, fully simulated)
--------------------------------
- The camera is the SIMULATED calibration ``sim-cal-v0`` (``load_calibration``):
  1280x720, ~70 deg FOV, ~1.2 m high, ~20 deg down-tilt, zero distortion.
- The pallet is the nominal 3D model (``build_nominal_pallet_model``): eight
  named corners ``bottom_corner_0..3`` (Z=0) then ``top_corner_0..3`` (elevated
  by the deck height), in that fixed order.
- For each randomised Floor_Frame pose ``(x, y, theta)`` we call the existing
  forward model :func:`pallet_pose_compliance.geometry.pose_eval.place_and_project`
  (which uses ``cv2.projectPoints`` with the calibration intrinsics + distortion
  + the inverse extrinsics ``camera<-floor``). This yields EXACT ground-truth
  image keypoints ``(u, v)`` AND the exact ground-truth pose. There is no
  estimator in this loop, so the GT is genuinely independent (R10.2).
- We render a simple-but-learnable image per sample: a filled + edged deck
  quadrilateral (the four projected top-deck corners) with mild randomised
  colour, lighting, and pixel noise on a randomised background. It is NOT
  photorealistic; it only needs recoverable corner geometry.

Everything produced here is labelled ``simulated`` (never ``measured``): both
the rendered images and the ground-truth poses are synthetic.

Outputs (under ``data/synthetic_pose/``)
----------------------------------------
- ``images/{train,val,test}/*.png`` and ``labels/{train,val,test}/*.txt`` in
  Ultralytics pose format: ``class cx cy w h px1 py1 v1 ... px8 py8 v8`` (all
  normalised 0..1; class 0 = pallet; 8 keypoints; visibility flag 2 = visible).
- ``data.yaml`` with ``kpt_shape: [8, 3]``, ``nc: 1``, ``names: [pallet]``.
- ``ground_truth.json`` — per-image exact GT pose ``(x, y, theta)`` and the 8
  image keypoints, keyed by split + relative image path. Provenance simulated.
- ``reports/synthetic_pose_summary.json`` — a run summary.

Reproducible: fixed seed 42 (overridable via ``--seed``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Ensure the src-layout package is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pallet_pose_compliance.calibration.calibration import (  # noqa: E402
    Calibration,
    build_synthetic_calibration,
    load_calibration,
)
from pallet_pose_compliance.detection.detector import KEYPOINT_NAMES  # noqa: E402
from pallet_pose_compliance.geometry.pallet_model import (  # noqa: E402
    build_nominal_pallet_model,
)
from pallet_pose_compliance.geometry.pose_eval import place_and_project  # noqa: E402
from pallet_pose_compliance.output.provenance import ProvenanceLabel  # noqa: E402

__all__ = [
    "GenConfig",
    "SampleRecord",
    "generate_dataset",
    "main",
]

_DEFAULT_CALIBRATION_ID = "sim-cal-v0"
_DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "synthetic_pose"
_DEFAULT_REPORT = _REPO_ROOT / "reports" / "synthetic_pose_summary.json"
_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class GenConfig:
    """Documented, fixed dataset-generation configuration (all simulated)."""

    n_samples: int = 3000
    seed: int = 42
    calibration_id: str = _DEFAULT_CALIBRATION_ID
    # Documented Floor_Frame sampling ranges (metres / degrees).
    x_range: tuple[float, float] = (-1.5, 1.5)
    y_range: tuple[float, float] = (1.5, 6.0)
    theta_range: tuple[float, float] = (-180.0, 180.0)
    # Split fractions (train / val / test). Documented 70/15/15.
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    # In-frame margin (pixels): projected corners must lie within the image
    # inflated/deflated by this margin to be kept "mostly in frame".
    frame_margin_px: float = 8.0
    # Fraction of the pallet keypoints that must be inside the frame to keep it.
    min_in_frame_fraction: float = 1.0
    out_dir: Path = _DEFAULT_OUT_DIR
    report_path: Path = _DEFAULT_REPORT


@dataclass
class SampleRecord:
    """One accepted sample's exact simulated ground truth."""

    split: str
    image_rel: str
    x_m: float
    y_m: float
    theta_deg: float
    keypoints_px: list[list[float]]  # 8 x [u, v] in the canonical name order
    keypoint_names: list[str] = field(default_factory=lambda: list(KEYPOINT_NAMES))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _image_size(calibration: Calibration) -> tuple[int, int]:
    """Return ``(width, height)`` for the calibration (fallback 1280x720)."""
    if calibration.image_size is not None:
        return int(calibration.image_size[0]), int(calibration.image_size[1])
    return 1280, 720


def _in_frame_fraction(
    kpts_px: np.ndarray, width: int, height: int, margin: float
) -> float:
    """Fraction of keypoints inside the image inflated by ``margin`` pixels."""
    u = kpts_px[:, 0]
    v = kpts_px[:, 1]
    inside = (
        (u >= -margin)
        & (u <= width - 1 + margin)
        & (v >= -margin)
        & (v <= height - 1 + margin)
    )
    return float(np.count_nonzero(inside)) / float(len(kpts_px))


def _bbox_from_keypoints(
    kpts_px: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float]:
    """Axis-aligned bbox (cx, cy, w, h) covering all 8 keypoints, clamped."""
    u = np.clip(kpts_px[:, 0], 0.0, width - 1.0)
    v = np.clip(kpts_px[:, 1], 0.0, height - 1.0)
    x_min, x_max = float(u.min()), float(u.max())
    y_min, y_max = float(v.min()), float(v.max())
    # Pad the box slightly so corners sit inside it.
    pad = 4.0
    x_min = max(0.0, x_min - pad)
    y_min = max(0.0, y_min - pad)
    x_max = min(width - 1.0, x_max + pad)
    y_max = min(height - 1.0, y_max + pad)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    return cx, cy, w, h


# ---------------------------------------------------------------------------
# Rendering (simple but learnable; not photorealistic)
# ---------------------------------------------------------------------------


def _render_sample(
    kpts_px: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render a learnable pallet image from its 8 projected corners.

    Draws the top-deck quad (filled + edged) and the bottom-deck quad, connects
    top/bottom corners to give a 3D-ish prism look, adds mild colour/lighting
    variation and Gaussian pixel noise on a randomised background. The geometry
    of the eight corners is recoverable, which is what YOLO-pose needs.
    """
    import cv2

    # Randomised background (flat colour + light gradient + noise).
    bg = rng.integers(30, 210, size=3).astype(np.uint8)
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = bg
    # Mild vertical lighting gradient.
    grad = np.linspace(-20, 20, height).astype(np.int16)
    img = np.clip(img.astype(np.int16) + grad[:, None, None], 0, 255).astype(np.uint8)

    bottom = kpts_px[0:4].astype(np.int32)  # bottom_corner_0..3
    top = kpts_px[4:8].astype(np.int32)  # top_corner_0..3

    # Pallet base colour (wood-ish, randomised).
    base = rng.integers(90, 200, size=3).astype(int)
    top_col = tuple(int(c) for c in np.clip(base, 0, 255))
    bottom_col = tuple(int(max(0, c - 40)) for c in base)
    edge_col = tuple(int(max(0, c - 70)) for c in base)

    # Side faces (connect each top edge to the matching bottom edge) for depth.
    for i in range(4):
        j = (i + 1) % 4
        quad = np.array([top[i], top[j], bottom[j], bottom[i]], dtype=np.int32)
        side = tuple(int(max(0, c - 20 - 8 * i)) for c in base)
        cv2.fillConvexPoly(img, quad, side)

    # Bottom deck then top deck (filled + edged).
    cv2.fillConvexPoly(img, bottom, bottom_col)
    cv2.fillConvexPoly(img, top, top_col)
    cv2.polylines(img, [bottom], isClosed=True, color=edge_col, thickness=2)
    cv2.polylines(img, [top], isClosed=True, color=edge_col, thickness=2)
    for i in range(4):
        cv2.line(img, tuple(top[i]), tuple(bottom[i]), edge_col, 1)

    # ORIENTATION-BREAKING VISUAL CUES (critical): a plain rectangle is
    # rotationally near-symmetric, so a keypoint network cannot learn which
    # physical corner is which and the per-index correspondence collapses. We
    # paint asymmetric, corner-identifying markings on the *top deck* so corner
    # identity is recoverable:
    #   - a bright filled marker at top_corner_0 (the +L/2,+W/2 reference corner);
    #   - a distinctly coloured stripe along the short side top_corner_0->3
    #     (the +X / long-axis-front edge) to fix the long-axis direction; and
    #   - a smaller marker at top_corner_1 so the winding direction is unambiguous.
    marker0 = tuple(int(c) for c in (np.array(base) * 0 + [40, 40, 235]))  # red-ish
    marker1 = tuple(int(c) for c in (np.array(base) * 0 + [235, 210, 40]))  # cyan-ish
    stripe_col = tuple(int(c) for c in (np.array(base) * 0 + [60, 200, 60]))  # green
    # Corner marker radius scales with the on-image pallet size.
    diag = float(np.linalg.norm(top[0] - top[2]))
    r0 = max(3, int(diag * 0.10))
    r1 = max(2, int(diag * 0.06))
    # Front short-side stripe (top_corner_0 -> top_corner_3).
    cv2.line(img, tuple(top[0]), tuple(top[3]), stripe_col, max(2, int(diag * 0.06)))
    cv2.circle(img, tuple(top[0]), r0, marker0, -1)
    cv2.circle(img, tuple(top[1]), r1, marker1, -1)

    # Mild Gaussian pixel noise.
    noise = rng.normal(0.0, 6.0, size=img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Label writing (Ultralytics pose format)
# ---------------------------------------------------------------------------


def _yolo_pose_label_line(
    kpts_px: np.ndarray, width: int, height: int
) -> str:
    """Build one Ultralytics pose label line (normalised, visibility flag 2)."""
    cx, cy, w, h = _bbox_from_keypoints(kpts_px, width, height)
    parts = [
        "0",  # class 0 = pallet
        f"{cx / width:.6f}",
        f"{cy / height:.6f}",
        f"{w / width:.6f}",
        f"{h / height:.6f}",
    ]
    for u, v in kpts_px:
        parts.append(f"{u / width:.6f}")
        parts.append(f"{v / height:.6f}")
        parts.append("2")  # visible
    return " ".join(parts)


def _write_data_yaml(out_dir: Path) -> Path:
    """Write the Ultralytics pose ``data.yaml`` (kpt_shape [8, 3], 1 class)."""
    path = out_dir / "data.yaml"
    names = ", ".join(KEYPOINT_NAMES)
    text = (
        "# SIMULATED synthetic pallet pose dataset (Ultralytics pose format).\n"
        "# Provenance: simulated. Generated by scripts/generate_synthetic_pose.py.\n"
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: [pallet]\n"
        "kpt_shape: [8, 3]\n"
        "# 8 keypoints in fixed order (matches detector KEYPOINT_NAMES):\n"
        f"# {names}\n"
        "# No left/right flip symmetry pairs (flip augmentation disabled in\n"
        "# training) because the corner ordering is orientation-specific.\n"
        "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7]\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _assign_split(index: int, n_total: int, fractions: tuple[float, float, float]) -> str:
    """Deterministically assign a sample index to a split by cumulative fraction."""
    f_train, f_val, _ = fractions
    frac = index / max(1, n_total)
    if frac < f_train:
        return "train"
    if frac < f_train + f_val:
        return "val"
    return "test"


def generate_dataset(config: GenConfig) -> dict[str, Any]:
    """Generate the full simulated pose dataset and GT sidecar.

    Returns a summary dict (also written to ``config.report_path``).
    """
    import cv2  # noqa: F401  (ensure available early; used by renderer)

    # Load the SIMULATED calibration; fall back to the synthetic builder if the
    # artefact is missing (still simulated, clearly documented).
    calibration_source: str
    try:
        calibration = load_calibration(config.calibration_id)
        calibration_source = f"loaded artefact configs/calibration/{config.calibration_id}.yaml"
    except Exception as exc:  # pragma: no cover - artefact expected present
        calibration = build_synthetic_calibration(config.calibration_id)
        calibration_source = (
            f"build_synthetic_calibration fallback ({exc}); still SIMULATED"
        )
    if calibration.provenance is not ProvenanceLabel.SIMULATED:
        raise SystemExit(
            f"calibration {config.calibration_id!r} is {calibration.provenance.value!r}, "
            "expected 'simulated' for a synthetic pose dataset"
        )

    pallet_model = build_nominal_pallet_model()
    width, height = _image_size(calibration)

    out_dir = Path(config.out_dir)
    for split in _SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    records: list[SampleRecord] = []
    attempts = 0
    rejected_out_of_frame = 0
    # Oversample attempts because some random poses fall (mostly) out of frame.
    max_attempts = config.n_samples * 20

    while len(records) < config.n_samples and attempts < max_attempts:
        attempts += 1
        x0 = float(rng.uniform(*config.x_range))
        y0 = float(rng.uniform(*config.y_range))
        theta = float(rng.uniform(*config.theta_range))

        keypoints, _ = place_and_project(calibration, pallet_model, x0, y0, theta)
        kpts_px = np.array([[kp.u, kp.v] for kp in keypoints], dtype=float)

        if not np.all(np.isfinite(kpts_px)):
            rejected_out_of_frame += 1
            continue
        frac_in = _in_frame_fraction(kpts_px, width, height, config.frame_margin_px)
        if frac_in < config.min_in_frame_fraction:
            rejected_out_of_frame += 1
            continue

        idx = len(records)
        split = _assign_split(idx, config.n_samples, config.split_fractions)
        # Use a stable, shuffled-by-index name; interleave splits deterministically.
        stem = f"pallet_{idx:06d}"
        image_rel = f"images/{split}/{stem}.png"
        label_rel = f"labels/{split}/{stem}.txt"

        img = _render_sample(kpts_px, width, height, rng)
        cv2.imwrite(str(out_dir / image_rel), img)
        (out_dir / label_rel).write_text(
            _yolo_pose_label_line(kpts_px, width, height) + "\n", encoding="utf-8"
        )

        records.append(
            SampleRecord(
                split=split,
                image_rel=image_rel,
                x_m=x0,
                y_m=y0,
                theta_deg=theta,
                keypoints_px=[[float(u), float(v)] for u, v in kpts_px],
            )
        )

    data_yaml = _write_data_yaml(out_dir)

    # Ground-truth sidecar (exact simulated pose + image keypoints per image).
    ground_truth = {
        "provenance": ProvenanceLabel.SIMULATED.value,
        "calibration_id": config.calibration_id,
        "calibration_source": calibration_source,
        "image_size": [width, height],
        "keypoint_names": list(KEYPOINT_NAMES),
        "method": (
            "self-constructed simulated GT: nominal pallet model placed at a known "
            "Floor_Frame (x, y, theta) and projected via cv2.projectPoints through "
            "the SIMULATED sim-cal-v0 calibration; independent of the pose estimator"
        ),
        "note": "SIMULATED, never measured; GT never derived from the estimator under test",
        "samples": {
            rec.image_rel: {
                "split": rec.split,
                "pose": {"x_m": rec.x_m, "y_m": rec.y_m, "theta_deg": rec.theta_deg},
                "keypoints_px": rec.keypoints_px,
                "keypoint_names": rec.keypoint_names,
            }
            for rec in records
        },
    }
    gt_path = out_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    split_counts = {s: sum(1 for r in records if r.split == s) for s in _SPLITS}
    summary = {
        "provenance": ProvenanceLabel.SIMULATED.value,
        "generator": "scripts/generate_synthetic_pose.py",
        "seed": config.seed,
        "calibration_id": config.calibration_id,
        "calibration_source": calibration_source,
        "image_size": [width, height],
        "requested_samples": config.n_samples,
        "accepted_samples": len(records),
        "attempts": attempts,
        "rejected_out_of_frame": rejected_out_of_frame,
        "split_counts": split_counts,
        "split_fractions": list(config.split_fractions),
        "sampling_ranges": {
            "x_m": list(config.x_range),
            "y_m": list(config.y_range),
            "theta_deg": list(config.theta_range),
        },
        "keypoint_names": list(KEYPOINT_NAMES),
        "outputs": {
            "dataset_dir": str(out_dir.resolve()),
            "data_yaml": str(data_yaml.resolve()),
            "ground_truth_json": str(gt_path.resolve()),
        },
        "honesty_note": (
            "All images AND ground-truth poses are SIMULATED (rendered/projected), "
            "never measured. Rendering is deliberately simple (learnable geometry), "
            "not photorealistic; there is a synthetic-to-real gap."
        ),
    }
    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a SIMULATED YOLO-pose pallet dataset (images + labels + "
            "data.yaml + exact GT sidecar). Everything is labelled 'simulated'."
        )
    )
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-id", default=_DEFAULT_CALIBRATION_ID)
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR))
    parser.add_argument("--report", default=str(_DEFAULT_REPORT))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = GenConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        calibration_id=args.calibration_id,
        out_dir=Path(args.out_dir),
        report_path=Path(args.report),
    )
    summary = generate_dataset(config)
    print(json.dumps(summary, indent=2))
    print(
        f"[SYNTHETIC POSE simulated] accepted {summary['accepted_samples']}/"
        f"{summary['requested_samples']} samples "
        f"(train/val/test = {summary['split_counts']}); "
        f"dataset at {summary['outputs']['dataset_dir']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
