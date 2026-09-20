"""Detection / localisation evaluation CLI (R5).

Loads a held-out :class:`SplitManifest` and its ground-truth
:class:`AnnotationSet`, runs the YOLO-pose detector over the held-out images,
and reports the two **separate** metric families as **distributions with sample
counts** (R5.1-R5.4):

- **Detection metrics** — per-image precision / recall / AP distributions.
- **Localisation metrics** — per-keypoint pixel + normalised error, percentiles,
  visibility-conditioned splits, and failure rates.

Honesty discipline (R5.5, R26.3/R26.4)
--------------------------------------
The report's provenance is resolved by
:func:`pallet_pose_compliance.detection.metrics.resolve_metric_provenance`:
``measured`` is attached **only** when an actual evaluation run computed the
numbers on **assignment-trained** weights. If the detector (ultralytics/real
weights) is unavailable, the CLI **degrades honestly** — it reports the blocker
and emits an ``unavailable`` metric report rather than fabricating measured
metrics, mirroring the ``scripts/train.py`` blocked pattern.

The pure metric math lives in ``src/pallet_pose_compliance/detection/metrics.py``
so it is unit-testable without ultralytics (synthetic pred/GT arrays). This CLI
is the thin orchestration layer that wires the detector, the held-out manifest,
and the ground-truth annotations into that math.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Ensure the src-layout package is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pallet_pose_compliance.annotation import (  # noqa: E402
    AnnotationSet,
    PalletAnnotation,
    read_annotations,
)
from pallet_pose_compliance.dataset import SplitManifest, read_manifest  # noqa: E402
from pallet_pose_compliance.detection.detector import (  # noqa: E402
    DetectorUnavailableError,
    YoloPoseDetector,
)
from pallet_pose_compliance.detection.metrics import (  # noqa: E402
    DetectionMetrics,
    LocalisationMetrics,
    MetricProvenance,
    compute_detection_metrics,
    compute_localisation_metrics,
    match_detections_to_ground_truth,
    resolve_metric_provenance,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel  # noqa: E402
from pallet_pose_compliance.output.schema import Detection, Keypoint  # noqa: E402

__all__ = [
    "EvalBlocker",
    "EvalReport",
    "ground_truth_by_image",
    "gt_boxes_for_images",
    "match_pallets_for_localisation",
    "build_unavailable_report",
    "run_evaluation",
    "main",
]


@dataclass(frozen=True)
class EvalBlocker:
    """An explicit evaluation blocker (honest degrade, R26.4)."""

    kind: str
    message: str


@dataclass
class EvalReport:
    """The evaluation report: separate detection + localisation metric families.

    Either both metric families are present (an actual eval ran) or the report
    is an ``unavailable`` shell carrying the blocker (nothing fabricated).
    """

    provenance: ProvenanceLabel
    provenance_note: str
    sample_count_images: int
    detection: Optional[DetectionMetrics] = None
    localisation: Optional[LocalisationMetrics] = None
    blocker: Optional[EvalBlocker] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.value,
            "provenance_note": self.provenance_note,
            "sample_count_images": self.sample_count_images,
            "detection": self.detection.to_dict() if self.detection else None,
            "localisation": self.localisation.to_dict() if self.localisation else None,
            "blocker": (
                {"kind": self.blocker.kind, "message": self.blocker.message}
                if self.blocker
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Ground-truth helpers (pure; unit-testable without ultralytics)
# ---------------------------------------------------------------------------


def ground_truth_by_image(
    annotation_set: AnnotationSet,
) -> dict[int, list[PalletAnnotation]]:
    """Group pallet ground-truth annotations by image id."""
    by_image: dict[int, list[PalletAnnotation]] = {}
    for ann in annotation_set.annotations:
        by_image.setdefault(ann.image_id, []).append(ann)
    return by_image


def gt_boxes_for_images(
    image_ids: Sequence[int],
    gt_by_image: Mapping[int, Sequence[PalletAnnotation]],
) -> list[list[list[float]]]:
    """Ground-truth pallet ``[x, y, w, h]`` boxes per image, in ``image_ids`` order."""
    per_image: list[list[list[float]]] = []
    for image_id in image_ids:
        boxes = [list(ann.computed_bbox()) for ann in gt_by_image.get(image_id, [])]
        per_image.append(boxes)
    return per_image


def _pallet_annotation_to_gt_keypoints(ann: PalletAnnotation) -> list[dict[str, Any]]:
    """Convert a GT pallet annotation to metric-friendly keypoint mappings."""
    return [
        {"name": kp.name, "x": kp.x, "y": kp.y, "visibility": kp.visibility}
        for kp in ann.keypoints
    ]


def match_pallets_for_localisation(
    predictions_per_image: Sequence[Sequence[Detection]],
    image_ids: Sequence[int],
    gt_by_image: Mapping[int, Sequence[PalletAnnotation]],
    *,
    iou_threshold: float = 0.5,
) -> tuple[list[list[Keypoint]], list[list[dict[str, Any]]], list[list[float]]]:
    """Match predicted pallets to GT pallets to feed localisation metrics.

    Localisation is scored only on *matched* pallets (a detection true-positive):
    a predicted pallet is matched to its highest-IoU GT pallet, and its
    keypoints are paired with that GT's keypoints for pixel/normalised error.

    Returns three parallel lists over matched pallets:
    predicted keypoints, GT keypoint mappings, and GT bbox (for normalisation).
    """
    matched_pred_kpts: list[list[Keypoint]] = []
    matched_gt_kpts: list[list[dict[str, Any]]] = []
    matched_gt_bboxes: list[list[float]] = []

    for preds, image_id in zip(predictions_per_image, image_ids):
        pallet_dets = [d for d in preds if d.cls == "pallet"]
        pred_boxes = [list(d.bbox) for d in pallet_dets]
        pred_scores = [float(d.score) for d in pallet_dets]
        gt_anns = list(gt_by_image.get(image_id, []))
        gt_boxes = [list(a.computed_bbox()) for a in gt_anns]
        matches = match_detections_to_ground_truth(
            pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold
        )
        for det, gt_idx in zip(pallet_dets, matches):
            if gt_idx is None:
                continue
            gt_ann = gt_anns[gt_idx]
            matched_pred_kpts.append(list(det.keypoints))
            matched_gt_kpts.append(_pallet_annotation_to_gt_keypoints(gt_ann))
            matched_gt_bboxes.append(list(gt_ann.computed_bbox()))
    return matched_pred_kpts, matched_gt_kpts, matched_gt_bboxes


def build_unavailable_report(blocker: EvalBlocker) -> EvalReport:
    """Build an ``unavailable`` report carrying the blocker (nothing fabricated)."""
    prov = resolve_metric_provenance(ran_eval=False, is_assignment_trained=False)
    return EvalReport(
        provenance=prov.label,
        provenance_note=(
            f"{prov.note} Blocker: [{blocker.kind}] {blocker.message}"
        ),
        sample_count_images=0,
        detection=None,
        localisation=None,
        blocker=blocker,
    )


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------


def _load_image(image_path: Path) -> Any:
    """Load an image as an array, or return ``None`` if it cannot be read."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    img = cv2.imread(str(image_path))
    return img


def run_evaluation(
    *,
    manifest: SplitManifest,
    annotations: AnnotationSet,
    detector: YoloPoseDetector,
    images_dir: Optional[Path] = None,
    iou_threshold: float = 0.5,
    is_synthetic_data: bool = False,
) -> EvalReport:
    """Run detection + localisation evaluation over the held-out set (R5).

    Degrades honestly: if the detector is unavailable (ultralytics/weights
    missing) or images cannot be loaded, returns an ``unavailable`` report with
    a blocker rather than fabricating measured metrics (R26.4).
    """
    gt_by_image = ground_truth_by_image(annotations)
    file_name_by_id = {img.image_id: img.file_name for img in annotations.images}

    image_ids = list(manifest.image_ids)
    predictions_per_image: list[list[Detection]] = []

    # Attempt to run the detector on each held-out image; on the first hard
    # unavailability, degrade to an honest unavailable report.
    try:
        for image_id in image_ids:
            file_name = file_name_by_id.get(image_id)
            if file_name is None:
                predictions_per_image.append([])
                continue
            image_path = (
                (images_dir / file_name) if images_dir is not None else Path(file_name)
            )
            image = _load_image(image_path)
            if image is None:
                raise DetectorUnavailableError(
                    f"could not load held-out image {image_path!s} (opencv "
                    "missing or file absent); cannot run a real evaluation."
                )
            predictions_per_image.append(detector.detect(image))
    except DetectorUnavailableError as exc:
        return build_unavailable_report(
            EvalBlocker(kind="DETECTOR_UNAVAILABLE", message=str(exc))
        )

    provenance = resolve_metric_provenance(
        ran_eval=True,
        is_assignment_trained=detector.is_assignment_trained,
        is_synthetic_data=is_synthetic_data,
    )
    return _report_from_predictions(
        predictions_per_image=predictions_per_image,
        image_ids=image_ids,
        gt_by_image=gt_by_image,
        provenance=provenance,
        iou_threshold=iou_threshold,
    )


def _report_from_predictions(
    *,
    predictions_per_image: Sequence[Sequence[Detection]],
    image_ids: Sequence[int],
    gt_by_image: Mapping[int, Sequence[PalletAnnotation]],
    provenance: MetricProvenance,
    iou_threshold: float,
) -> EvalReport:
    """Compute both metric families from predictions + GT (pure orchestration)."""
    gt_boxes_per_image = gt_boxes_for_images(image_ids, gt_by_image)
    detection = compute_detection_metrics(
        predictions_per_image,
        gt_boxes_per_image,
        provenance=provenance,
        iou_threshold=iou_threshold,
    )
    pred_kpts, gt_kpts, gt_bboxes = match_pallets_for_localisation(
        predictions_per_image, image_ids, gt_by_image, iou_threshold=iou_threshold
    )
    localisation = compute_localisation_metrics(
        pred_kpts, gt_kpts, gt_bboxes, provenance=provenance
    )
    return EvalReport(
        provenance=provenance.label,
        provenance_note=provenance.note,
        sample_count_images=len(list(image_ids)),
        detection=detection,
        localisation=localisation,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate detection and localisation accuracy on the held-out set, "
            "reported as distributions with sample counts. Attaches 'measured' "
            "provenance ONLY for an actual eval run on assignment-trained "
            "weights; degrades honestly to 'unavailable' otherwise (R5, R26)."
        )
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to the held-out SplitManifest JSON."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to the COCO-keypoints-style ground-truth annotations JSON.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory containing the held-out images (file_name is resolved here).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to assignment-trained YOLO-pose weights (measured-eligible).",
    )
    parser.add_argument(
        "--stock-model",
        default=None,
        help=(
            "Stock/pretrained checkpoint id used only if no assignment-trained "
            "weights are given; any metrics are 'estimated', never 'measured'."
        ),
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU match threshold."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON report; prints to stdout otherwise.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns 0 on a completed eval, 2 when blocked/unavailable."""
    args = _build_arg_parser().parse_args(argv)

    manifest = read_manifest(args.manifest)
    annotations = read_annotations(args.annotations)
    images_dir = Path(args.images_dir) if args.images_dir else None

    if args.weights is None and args.stock_model is None:
        # No weights at all: honest unavailable report (no fabricated metrics).
        report = build_unavailable_report(
            EvalBlocker(
                kind="NO_WEIGHTS",
                message=(
                    "no --weights (assignment-trained) or --stock-model provided; "
                    "cannot run a real evaluation. Metrics are unavailable, not "
                    "fabricated (R26.4)."
                ),
            )
        )
    else:
        detector = YoloPoseDetector(
            weights_path=args.weights, stock_model_id=args.stock_model
        )
        report = run_evaluation(
            manifest=manifest,
            annotations=annotations,
            detector=detector,
            images_dir=images_dir,
            iou_threshold=args.iou_threshold,
        )

    payload = json.dumps(report.to_dict(), indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Wrote evaluation report to {out_path}")
    else:
        print(payload)

    if report.blocker is not None:
        print(
            f"[EVAL {report.provenance.value.upper()}: {report.blocker.kind}] "
            f"{report.blocker.message}",
            file=sys.stderr,
        )
        return 2
    print(
        f"Evaluation complete over {report.sample_count_images} held-out images "
        f"(provenance: {report.provenance.value}).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
