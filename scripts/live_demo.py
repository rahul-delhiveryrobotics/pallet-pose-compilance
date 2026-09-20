#!/usr/bin/env python3
"""Live webcam demo: detect pallets and show pose + SOP verdict on the video.

This is a DEMO front-end for the pipeline. It opens a camera (laptop webcam by
default, or a USB/phone cam via --source), runs the trained detector + pose +
SOP + verdict on each frame, and overlays the results live:

  - a box around each detected pallet (+ the detector confidence),
  - the 8 predicted corner keypoints,
  - the estimated pose (x, y in metres, orientation deg) when available,
  - the overall verdict (PASS / FAIL / MANUAL_INSPECTION) colour-coded,
  - honest status text when the pose is unavailable.

Honesty note (matches the project discipline): the DETECTOR is trained on real
pallet images and detects real pallets well. The POSE model was trained on
SIMULATED renders, so live pose numbers on a real scene are rough / may be
unavailable; they are shown but labelled "simulated". Nothing is fabricated.

Usage
-----
    python3 scripts/live_demo.py                 # default laptop camera (0)
    python3 scripts/live_demo.py --source 1      # a different camera index
    python3 scripts/live_demo.py --source path/to/video.mp4
    python3 scripts/live_demo.py --detect-only   # boxes only (skip pose/SOP)
    python3 scripts/live_demo.py --save out.mp4  # also record the annotated feed

Keys: press 'q' or ESC to quit; 's' to save a snapshot PNG.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Ensure the src-layout package is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pallet_pose_compliance.calibration.calibration import (  # noqa: E402
    CalibrationError,
    load_calibration,
)
from pallet_pose_compliance.detection.detector import (  # noqa: E402
    DetectorUnavailableError,
    YoloPoseDetector,
)
from pallet_pose_compliance.geometry.pallet_model import (  # noqa: E402
    build_nominal_pallet_model,
)
from pallet_pose_compliance.geometry.pose import (  # noqa: E402
    estimate_pose_with_uncertainty,
)
from pallet_pose_compliance.sop.checks import analyze  # noqa: E402
from pallet_pose_compliance.sop.config import load_sop_config  # noqa: E402
from pallet_pose_compliance.pipeline import load_pipeline_config  # noqa: E402
from pallet_pose_compliance.verdict.config import load_verdict_config  # noqa: E402
from pallet_pose_compliance.verdict.engine import aggregate  # noqa: E402

# Verdict colours (BGR).
_VERDICT_COLOUR = {
    "PASS": (0, 180, 0),
    "FAIL": (0, 0, 220),
    "MANUAL_INSPECTION": (0, 165, 255),
}
_BOX_COLOUR = (255, 200, 0)
_KP_COLOUR = (60, 220, 60)
_TEXT_BG = (30, 30, 30)


def _draw_label(img, text, org, colour=(255, 255, 255), scale=0.55, thick=1):
    """Draw text with a filled background box for readability."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - base - 3), (x + tw + 6, y + 2), _TEXT_BG, -1)
    cv2.putText(
        img, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick,
        cv2.LINE_AA,
    )


def _annotate_frame(
    frame: np.ndarray,
    config: Any,
    detector: YoloPoseDetector,
    calibration: Optional[Any],
    pallet_model: Any,
    sop_config: Any,
    verdict_config: Any,
    *,
    detect_only: bool,
) -> np.ndarray:
    """Run the pipeline on one frame and draw the results onto a copy."""
    out = frame.copy()
    try:
        detections = detector.detect(frame)
    except DetectorUnavailableError as exc:
        _draw_label(out, f"detector unavailable: {exc}", (10, 30), (0, 0, 255), 0.6, 2)
        return out

    pallets = [d for d in detections if d.cls == "pallet"]
    _draw_label(
        out,
        f"pallets: {len(pallets)}  (detector: real/trained)",
        (10, 25),
        (255, 255, 255),
        0.6,
        2,
    )

    for idx, det in enumerate(detections):
        x, y, w, h = det.bbox
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        colour = _BOX_COLOUR if det.cls == "pallet" else (180, 180, 180)
        cv2.rectangle(out, p1, p2, colour, 2)
        _draw_label(out, f"{det.cls} {det.score:.2f}", (p1[0], max(20, p1[1])), colour)

        # Draw keypoints.
        for kp in det.keypoints:
            if kp.u is not None and kp.v is not None:
                cv2.circle(out, (int(kp.u), int(kp.v)), 3, _KP_COLOUR, -1)

        if detect_only or det.cls != "pallet":
            continue

        # --- pose + SOP + verdict for this pallet ---
        if calibration is None:
            _draw_label(out, "pose: no calibration", (p1[0], p2[1] + 18), (0, 0, 255))
            continue

        pose = estimate_pose_with_uncertainty(
            list(det.keypoints), calibration, pallet_model, n_samples=32, seed=config.seed
        )
        if pose.pose_status == "available" and pose.position_m is not None:
            line = (
                f"pose[sim]: x={pose.position_m.x:+.2f}m y={pose.position_m.y:+.2f}m "
                f"th={pose.orientation_deg:+.1f}deg"
            )
            _draw_label(out, line, (p1[0], p2[1] + 18), (200, 255, 200))
        else:
            rc = pose.reason_code.value if pose.reason_code else "unavailable"
            _draw_label(out, f"pose: unavailable ({rc})", (p1[0], p2[1] + 18), (0, 165, 255))

        sop_checks = analyze(detections, pose, sop_config)
        verdict = aggregate(pose, sop_checks, verdict_config)
        vcol = _VERDICT_COLOUR.get(verdict.verdict, (255, 255, 255))
        _draw_label(out, f"VERDICT: {verdict.verdict}", (p1[0], p2[1] + 40), vcol, 0.7, 2)

    # Honesty banner.
    _draw_label(
        out,
        "detector=REAL/trained | pose=SIMULATED-trained (rough on real scenes)",
        (10, out.shape[0] - 12),
        (200, 200, 200),
        0.5,
        1,
    )
    return out


def _gather_images(path_str: str) -> list[Path]:
    """Return a sorted list of image paths from a file or a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    p = Path(path_str)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
    return []


def _run_images(
    images_arg: str,
    config: Any,
    detector: YoloPoseDetector,
    calibration: Optional[Any],
    pallet_model: Any,
    sop_config: Any,
    verdict_config: Any,
    *,
    detect_only: bool,
) -> int:
    """Slideshow over a folder/file of images: annotate each, show, step on key.

    This is the reliable demo path — the images (e.g. the synthetic-pose TEST
    set) contain pallets, so detection + pose + verdict are drawn every time.
    Any key advances to the next image; 'q'/ESC quits; 's' saves a snapshot.
    """
    images = _gather_images(images_arg)
    if not images:
        print(f"[demo] no images found at {images_arg!r}", file=sys.stderr)
        return 4
    print(f"[demo] slideshow over {len(images)} image(s). Any key = next, 'q'/ESC = quit, 's' = save.")
    snap_i = 0
    i = 0
    while 0 <= i < len(images):
        img_path = images[i]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[demo] could not read {img_path}", file=sys.stderr)
            i += 1
            continue
        annotated = _annotate_frame(
            frame, config, detector, calibration, pallet_model,
            sop_config, verdict_config, detect_only=detect_only,
        )
        _draw_label(annotated, f"{img_path.name}  [{i + 1}/{len(images)}]", (10, annotated.shape[0] - 34), (255, 255, 255), 0.5, 1)
        cv2.imshow("Pallet Pose + SOP (image demo)", annotated)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            snap_path = _REPO_ROOT / "outputs" / f"demo_{img_path.stem}.png"
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snap_path), annotated)
            print(f"[demo] saved {snap_path}")
            snap_i += 1
            continue  # stay on the same image after saving
        i += 1
    cv2.destroyAllWindows()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Live pallet detection + pose/SOP demo.")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or a video/file path.")
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "pipeline.yaml"))
    parser.add_argument("--detect-only", action="store_true", help="Only run detection (skip pose/SOP/verdict).")
    parser.add_argument("--save", default=None, help="Optional path to record the annotated feed (mp4).")
    parser.add_argument("--conf", type=float, default=None, help="Override detector confidence threshold.")
    parser.add_argument(
        "--images",
        default=None,
        help=(
            "Run as a SLIDESHOW over a folder of images (or a single image) "
            "instead of a camera. Best for a reliable demo: these contain "
            "pallets, so pose + verdict are guaranteed to show. Any key = next "
            "image, 'q'/ESC = quit, 's' = save snapshot."
        ),
    )
    args = parser.parse_args(argv)

    config = load_pipeline_config(args.config)

    # Build the real detector (prefers trained weights/<model_id>.pt).
    weights_path = config.weights_path
    detector = YoloPoseDetector(
        weights_path=weights_path if weights_path.exists() else None,
        stock_model_id=None if weights_path.exists() else config.stock_model_id,
        imgsz=config.input_resolution,
        conf=args.conf if args.conf is not None else config.detector_conf,
    )
    try:
        detector.load()
    except DetectorUnavailableError as exc:
        print(f"detector unavailable: {exc}", file=sys.stderr)
        return 2
    print(
        f"[demo] detector loaded: {'TRAINED' if weights_path.exists() else 'STOCK'} "
        f"({detector.weights_provenance.weights_ref})"
    )

    # Load calibration + models for the pose/SOP path (unless detect-only).
    calibration = None
    if not args.detect_only:
        try:
            calibration = load_calibration(config.calibration_id, calibration_dir=config.calibration_dir)
        except CalibrationError as exc:
            print(f"[demo] calibration unavailable ({exc}); running detect-only.", file=sys.stderr)
    pallet_model = build_nominal_pallet_model()
    sop_config = load_sop_config(config.sop_thresholds_path)
    verdict_config = load_verdict_config(_REPO_ROOT / "configs" / "verdict.yaml")

    # --- IMAGES / SLIDESHOW MODE (reliable demo: guaranteed to contain pallets) ---
    if args.images:
        return _run_images(
            args.images, config, detector, calibration, pallet_model,
            sop_config, verdict_config, detect_only=args.detect_only,
        )

    # Open the source (int camera index or a file path).
    source: Any = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(
            f"[demo] could not open source {args.source!r}. If this is a laptop "
            "webcam, check camera permissions; try --source 1 for another cam.",
            file=sys.stderr,
        )
        return 3

    writer = None
    print("[demo] running. Press 'q' or ESC to quit, 's' to save a snapshot.")
    snap_i = 0
    fps_t = time.perf_counter()
    frames = 0
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[demo] end of stream / cannot read frame.", file=sys.stderr)
                break

            annotated = _annotate_frame(
                frame, config, detector, calibration, pallet_model,
                sop_config, verdict_config, detect_only=args.detect_only,
            )

            frames += 1
            if frames >= 10:
                now = time.perf_counter()
                fps = frames / (now - fps_t)
                fps_t = now
                frames = 0
            _draw_label(annotated, f"{fps:4.1f} FPS", (annotated.shape[1] - 110, 25), (255, 255, 255), 0.6, 2)

            if args.save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
                    )
                writer.write(annotated)

            cv2.imshow("Pallet Pose + SOP (live demo)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("s"):
                snap_path = _REPO_ROOT / "outputs" / f"live_snapshot_{snap_i:03d}.png"
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap_path), annotated)
                print(f"[demo] saved {snap_path}")
                snap_i += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
