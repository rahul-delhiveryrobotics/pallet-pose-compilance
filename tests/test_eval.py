"""Unit tests for the eval.py orchestration layer (task 4.6).

Covers ``scripts/eval.py`` without ultralytics/real weights:

- Ground-truth grouping and box extraction from an :class:`AnnotationSet`.
- Matched-pallet pairing that feeds localisation metrics.
- Honest degrade: an unavailable detector yields an ``unavailable`` report with
  a blocker, never fabricated measured metrics (R26.4).
- A full run over a fake detector produces both metric families with the
  correct provenance driven by the weights' assignment-trained flag (R5.5).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load scripts/eval.py as a module (it is a script, not an installed package).
_spec = importlib.util.spec_from_file_location(
    "pallet_eval_script", _REPO_ROOT / "scripts" / "eval.py"
)
assert _spec and _spec.loader
eval_mod = importlib.util.module_from_spec(_spec)
sys.modules["pallet_eval_script"] = eval_mod
_spec.loader.exec_module(eval_mod)

from pallet_pose_compliance.annotation import (  # noqa: E402
    AnnotationSet,
    ImageAnnotation,
    KeypointAnnotation,
    PalletAnnotation,
)
from pallet_pose_compliance.dataset import DatasetSample, SplitManifest  # noqa: E402
from pallet_pose_compliance.detection.detector import (  # noqa: E402
    DetectorUnavailableError,
    WeightsProvenance,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel  # noqa: E402
from pallet_pose_compliance.output.schema import Detection, Keypoint  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

_NAMES = [f"bottom_corner_{i}" for i in range(4)] + [f"top_corner_{i}" for i in range(4)]
_COORDS = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 2), (10, 2), (10, 12), (0, 12)]


def _gt_keypoints():
    return tuple(
        KeypointAnnotation(name=n, x=float(x), y=float(y), visibility="visible")
        for n, (x, y) in zip(_NAMES, _COORDS)
    )


def _annotation_set():
    images = [ImageAnnotation(image_id=1, file_name="img1.png", width=640, height=480)]
    anns = [PalletAnnotation(image_id=1, keypoints=_gt_keypoints(), annotation_id=1)]
    return AnnotationSet(images=images, annotations=anns)


def _manifest():
    sample = DatasetSample(
        image_id=1,
        file_name="img1.png",
        group_id="scene_a",
        class_counts={"pallet": 1},
        annotation_count=1,
    )
    return SplitManifest(split="held_out", split_by="scene", seed=42, samples=[sample])


def _pred_keypoints():
    return [Keypoint(name=n, u=float(x), v=float(y), visibility="visible")
            for n, (x, y) in zip(_NAMES, _COORDS)]


def _pred_detection():
    # GT bbox over corners is [0, 0, 10, 12]; match this so IoU=1.
    return Detection(cls="pallet", bbox=[0, 0, 10, 12], score=0.9, keypoints=_pred_keypoints())


class _FakeDetector:
    """A detector-shaped fake for orchestration tests (no ultralytics)."""

    def __init__(self, detections, *, is_assignment_trained=True, raises=False):
        self._detections = detections
        self.is_assignment_trained = is_assignment_trained
        self._raises = raises
        if is_assignment_trained:
            self.weights_provenance = WeightsProvenance.assignment_trained("weights/best.pt")
        else:
            self.weights_provenance = WeightsProvenance.stock("yolo11n-pose.pt")

    @property
    def provenance(self):
        return self.weights_provenance.provenance

    def detect(self, image):
        if self._raises:
            raise DetectorUnavailableError("ultralytics unavailable in test")
        return list(self._detections)


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------


def test_ground_truth_by_image_groups_annotations():
    gt = eval_mod.ground_truth_by_image(_annotation_set())
    assert set(gt.keys()) == {1}
    assert len(gt[1]) == 1


def test_gt_boxes_for_images_returns_boxes_in_order():
    gt = eval_mod.ground_truth_by_image(_annotation_set())
    boxes = eval_mod.gt_boxes_for_images([1], gt)
    assert boxes == [[[0.0, 0.0, 10.0, 12.0]]]


def test_match_pallets_for_localisation_pairs_matched_detection():
    gt = eval_mod.ground_truth_by_image(_annotation_set())
    pred_kpts, gt_kpts, gt_bboxes = eval_mod.match_pallets_for_localisation(
        [[_pred_detection()]], [1], gt
    )
    assert len(pred_kpts) == 1
    assert len(gt_kpts) == 1
    assert gt_bboxes == [[0.0, 0.0, 10.0, 12.0]]


def test_match_pallets_skips_unmatched_prediction():
    gt = eval_mod.ground_truth_by_image(_annotation_set())
    far = Detection(cls="pallet", bbox=[999, 999, 5, 5], score=0.9, keypoints=_pred_keypoints())
    pred_kpts, gt_kpts, gt_bboxes = eval_mod.match_pallets_for_localisation(
        [[far]], [1], gt
    )
    assert pred_kpts == [] and gt_kpts == [] and gt_bboxes == []


# ---------------------------------------------------------------------------
# Honest degrade
# ---------------------------------------------------------------------------


def test_unavailable_detector_yields_unavailable_report():
    detector = _FakeDetector([], raises=True)
    report = eval_mod.run_evaluation(
        manifest=_manifest(),
        annotations=_annotation_set(),
        detector=detector,
        images_dir=None,
    )
    assert report.provenance is ProvenanceLabel.UNAVAILABLE
    assert report.detection is None
    assert report.localisation is None
    assert report.blocker is not None
    assert report.blocker.kind == "DETECTOR_UNAVAILABLE"


def test_build_unavailable_report_has_no_metrics():
    report = eval_mod.build_unavailable_report(
        eval_mod.EvalBlocker(kind="NO_WEIGHTS", message="none")
    )
    assert report.provenance is ProvenanceLabel.UNAVAILABLE
    assert report.detection is None and report.localisation is None
    d = report.to_dict()
    assert d["detection"] is None and d["blocker"]["kind"] == "NO_WEIGHTS"


# ---------------------------------------------------------------------------
# Full run over a fake detector (no image loading)
# ---------------------------------------------------------------------------


def test_full_report_from_predictions_measured_on_trained_weights():
    report = eval_mod._report_from_predictions(
        predictions_per_image=[[_pred_detection()]],
        image_ids=[1],
        gt_by_image=eval_mod.ground_truth_by_image(_annotation_set()),
        provenance=eval_mod.resolve_metric_provenance(
            ran_eval=True, is_assignment_trained=True
        ),
        iou_threshold=0.5,
    )
    assert report.provenance is ProvenanceLabel.MEASURED
    assert report.detection is not None
    assert report.localisation is not None
    assert report.detection.precision.sample_count == 1
    assert report.detection.precision.values[0] == pytest.approx(1.0)
    # Perfect keypoint match -> zero pixel error.
    assert report.localisation.pixel_error.mean == pytest.approx(0.0)


def test_full_report_estimated_on_stock_weights():
    report = eval_mod._report_from_predictions(
        predictions_per_image=[[_pred_detection()]],
        image_ids=[1],
        gt_by_image=eval_mod.ground_truth_by_image(_annotation_set()),
        provenance=eval_mod.resolve_metric_provenance(
            ran_eval=True, is_assignment_trained=False
        ),
        iou_threshold=0.5,
    )
    assert report.provenance is ProvenanceLabel.ESTIMATED
    assert report.detection.provenance is ProvenanceLabel.ESTIMATED
    assert report.localisation.provenance is ProvenanceLabel.ESTIMATED


def test_report_to_dict_metrics_carry_sample_counts():
    report = eval_mod._report_from_predictions(
        predictions_per_image=[[_pred_detection()]],
        image_ids=[1],
        gt_by_image=eval_mod.ground_truth_by_image(_annotation_set()),
        provenance=eval_mod.resolve_metric_provenance(
            ran_eval=True, is_assignment_trained=True
        ),
        iou_threshold=0.5,
    )
    d = report.to_dict()
    assert d["detection"]["precision"]["sample_count"] == 1
    assert "sample_count" in d["localisation"]["pixel_error"]
