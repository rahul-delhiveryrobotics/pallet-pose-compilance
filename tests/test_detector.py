"""Unit tests for the YOLO-pose detector wrapper and training workflow (task 4.5).

Covers the parts that do NOT require ultralytics/torch:

- Output conversion: a fake YOLO ``Results`` object -> ``Detection`` schema
  (box format, score clamping, eight named keypoints with visibility + score).
- Weights provenance / honesty (R5.5, R26.3): assignment-trained weights are
  ``measured``-eligible; a stock checkpoint is ``estimated`` and flagged
  not-assignment-trained; a stock checkpoint can never be mislabelled as
  assignment-trained or ``measured``.
- Honest degrade: the wrapper raises ``DetectorUnavailableError`` (never
  fabricates detections) when ultralytics/weights are unavailable.
- Training workflow (scripts/train.py): documented fixed hyperparameters,
  explicit blocked-training fallback (missing data / missing ultralytics), and
  checkpoint selection by held-out localisation metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pallet_pose_compliance.detection import (
    KEYPOINT_NAMES,
    DetectorUnavailableError,
    WeightsProvenance,
    YoloPoseDetector,
    detections_from_yolo_result,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel

# Make scripts/ importable so we can exercise the training workflow helpers.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train  # noqa: E402  (imported after sys.path tweak)


# ---------------------------------------------------------------------------
# Fake YOLO result plumbing (no ultralytics/torch required)
# ---------------------------------------------------------------------------


class _FakeBoxes:
    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=float)


class _FakeKeypoints:
    def __init__(self, xy, conf=None):
        self.xy = np.asarray(xy, dtype=float)
        self.conf = None if conf is None else np.asarray(conf, dtype=float)


class _FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


def _make_result(num_objects=1, with_keypoints=True, kpt_conf=True):
    xywh = [[100.0 + 50 * i, 200.0, 40.0, 60.0] for i in range(num_objects)]
    conf = [0.9 for _ in range(num_objects)]
    cls = [0 for _ in range(num_objects)]  # pallet
    boxes = _FakeBoxes(xywh, conf, cls)

    keypoints = None
    if with_keypoints:
        # 8 corners per object arranged on a simple grid.
        xy = [
            [[10.0 + k, 20.0 + k] for k in range(len(KEYPOINT_NAMES))]
            for _ in range(num_objects)
        ]
        conf_arr = None
        if kpt_conf:
            # First four visible (>=0.5), last four occluded (<0.5).
            conf_arr = [[0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2] for _ in range(num_objects)]
        keypoints = _FakeKeypoints(xy, conf_arr)
    return _FakeResult(boxes=boxes, keypoints=keypoints)


# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------


class TestOutputConversion:
    def test_converts_box_score_and_class(self):
        result = _make_result(num_objects=1)
        detections = detections_from_yolo_result(result)
        assert len(detections) == 1
        det = detections[0]
        assert det.cls == "pallet"
        assert det.score == pytest.approx(0.9)
        # YOLO centre xywh (100,200,40,60) -> top-left [80, 170, 40, 60].
        assert det.bbox == pytest.approx([80.0, 170.0, 40.0, 60.0])

    def test_emits_eight_named_keypoints_in_order(self):
        det = detections_from_yolo_result(_make_result())[0]
        assert [kp.name for kp in det.keypoints] == list(KEYPOINT_NAMES)

    def test_visibility_derived_from_keypoint_score(self):
        det = detections_from_yolo_result(_make_result())[0]
        # First four >=0.5 -> visible; last four <0.5 -> occluded.
        vis = [kp.visibility for kp in det.keypoints]
        assert vis[:4] == ["visible"] * 4
        assert vis[4:] == ["occluded"] * 4

    def test_keypoint_scores_clamped_to_unit_interval(self):
        boxes = _FakeBoxes([[10, 10, 4, 4]], [1.5], [0])  # score > 1 clamps
        xy = [[[float(k), float(k)] for k in range(8)]]
        conf = [[1.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]
        det = detections_from_yolo_result(
            _FakeResult(boxes=boxes, keypoints=_FakeKeypoints(xy, conf))
        )[0]
        assert det.score == pytest.approx(1.0)
        assert det.keypoints[0].score == pytest.approx(1.0)

    def test_no_keypoints_when_result_has_none(self):
        det = detections_from_yolo_result(_make_result(with_keypoints=False))[0]
        assert det.keypoints == []

    def test_no_boxes_returns_empty(self):
        assert detections_from_yolo_result(_FakeResult(boxes=None)) == []

    def test_multiple_objects(self):
        detections = detections_from_yolo_result(_make_result(num_objects=3))
        assert len(detections) == 3

    def test_box_class_maps_to_box(self):
        boxes = _FakeBoxes([[10, 10, 4, 4]], [0.8], [1])  # cls 1 -> box
        det = detections_from_yolo_result(_FakeResult(boxes=boxes))[0]
        assert det.cls == "box"

    def test_keypoints_without_conf_all_visible(self):
        det = detections_from_yolo_result(_make_result(kpt_conf=False))[0]
        assert all(kp.visibility == "visible" for kp in det.keypoints)
        assert all(kp.score is None for kp in det.keypoints)


# ---------------------------------------------------------------------------
# Weights provenance / honesty (R5.5, R26.3)
# ---------------------------------------------------------------------------


class TestWeightsProvenance:
    def test_assignment_trained_is_measured_eligible(self):
        prov = WeightsProvenance.assignment_trained("weights/best.pt")
        assert prov.is_assignment_trained is True
        assert prov.provenance is ProvenanceLabel.MEASURED

    def test_stock_is_estimated_and_flagged(self):
        prov = WeightsProvenance.stock("yolo11n-pose.pt")
        assert prov.is_assignment_trained is False
        assert prov.provenance is ProvenanceLabel.ESTIMATED

    def test_stock_can_never_be_labelled_measured(self):
        # Honesty invariant: mislabelling a stock checkpoint as measured fails.
        with pytest.raises(ValueError):
            WeightsProvenance(
                is_assignment_trained=False,
                provenance=ProvenanceLabel.MEASURED,
                weights_ref="yolo11n-pose.pt",
                note="bogus",
            )

    def test_detector_with_weights_path_is_assignment_trained(self):
        det = YoloPoseDetector(weights_path="weights/best.pt")
        assert det.is_assignment_trained is True
        assert det.provenance is ProvenanceLabel.MEASURED

    def test_detector_with_stock_model_is_estimated_not_trained(self):
        det = YoloPoseDetector(stock_model_id="yolo11n-pose.pt")
        assert det.is_assignment_trained is False
        assert det.provenance is ProvenanceLabel.ESTIMATED
        # Never mislabelled as assignment-trained.
        assert det.weights_provenance.is_assignment_trained is False

    def test_detector_requires_some_weights_source(self):
        with pytest.raises(ValueError):
            YoloPoseDetector()


# ---------------------------------------------------------------------------
# Honest degrade when ultralytics/weights unavailable
# ---------------------------------------------------------------------------


class TestHonestDegrade:
    def test_missing_assignment_weights_raises_unavailable(self, monkeypatch):
        # Simulate ultralytics being importable so we reach the weights check.
        import types

        fake_mod = types.ModuleType("ultralytics")
        fake_mod.YOLO = lambda *a, **k: (_ for _ in ()).throw(  # pragma: no cover
            AssertionError("should not construct")
        )
        monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)

        det = YoloPoseDetector(weights_path="weights/does-not-exist-xyz.pt")
        with pytest.raises(DetectorUnavailableError):
            det.load()

    def test_load_without_ultralytics_raises_unavailable(self, monkeypatch):
        # Force the ultralytics import to fail.
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        det = YoloPoseDetector(stock_model_id="yolo11n-pose.pt")
        with pytest.raises(DetectorUnavailableError):
            det.load()

    def test_detect_rejects_bad_image(self):
        det = YoloPoseDetector(stock_model_id="yolo11n-pose.pt")
        with pytest.raises(ValueError):
            det.detect(np.zeros((5,)))  # 1-D, not an image


# ---------------------------------------------------------------------------
# Training workflow (scripts/train.py) - no ultralytics/torch required
# ---------------------------------------------------------------------------


class TestTrainingHyperparameters:
    def test_defaults_are_documented_and_fixed(self):
        hp = train.TrainingHyperparameters()
        assert hp.seed == 42
        assert hp.epochs > 0
        assert hp.learning_rate > 0
        assert hp.batch_size > 0
        assert hp.imgsz == 640
        # Every hyperparameter carries a documented reasoning string (R4.3).
        for key in ("epochs", "learning_rate", "batch_size", "imgsz", "base_model"):
            assert key in hp.reasoning and hp.reasoning[key]

    def test_checkpoint_selection_metric_is_localisation_not_loss(self):
        hp = train.TrainingHyperparameters()
        metric = hp.checkpoint_selection_metric.lower()
        assert "map" in metric  # a localisation mAP metric
        assert "loss" not in metric

    def test_resolve_hyperparameters_reads_seed_from_config(self, tmp_path):
        cfg = tmp_path / "pipeline.yaml"
        cfg.write_text("seed: 7\ninput_resolution: 512\n", encoding="utf-8")
        hp = train.resolve_hyperparameters(cfg)
        assert hp.seed == 7
        assert hp.imgsz == 512

    def test_train_args_fix_seed_and_deterministic(self):
        hp = train.TrainingHyperparameters(seed=123)
        args = hp.as_ultralytics_train_args(data_yaml="d.yaml", project="weights")
        assert args["seed"] == 123
        assert args["deterministic"] is True
        assert args["imgsz"] == hp.imgsz
        assert args["lr0"] == hp.learning_rate


class TestCheckpointSelection:
    def test_selects_best_held_out_metric(self):
        metrics = {"epoch10.pt": 0.41, "epoch20.pt": 0.55, "epoch30.pt": 0.52}
        assert train.select_best_checkpoint(metrics) == "epoch20.pt"

    def test_lower_is_better_selection(self):
        metrics = {"a.pt": 3.0, "b.pt": 1.0, "c.pt": 2.0}
        assert train.select_best_checkpoint(metrics, higher_is_better=False) == "b.pt"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            train.select_best_checkpoint({})


class TestBlockedTrainingFallback:
    def test_blocker_when_no_dataset(self):
        blocker = train.detect_blocker(data_yaml=None)
        assert blocker is not None
        assert blocker.kind == "NO_DATASET"

    def test_blocker_when_dataset_missing_file(self, tmp_path):
        blocker = train.detect_blocker(
            data_yaml=tmp_path / "missing.yaml", require_ultralytics=False
        )
        assert blocker is not None
        assert blocker.kind == "NO_DATASET"

    def test_no_blocker_when_data_present_and_deps_skipped(self, tmp_path):
        data = tmp_path / "data.yaml"
        data.write_text("train: x\nval: y\n", encoding="utf-8")
        assert train.detect_blocker(data_yaml=data, require_ultralytics=False) is None

    def test_run_training_raises_explicit_blocker(self):
        # No dataset -> explicit blocker, never a fabricated model.
        with pytest.raises(RuntimeError) as exc:
            train.run_training(data_yaml=None)
        assert "TRAINING BLOCKED" in str(exc.value)

    def test_main_returns_nonzero_on_blocker(self, capsys):
        # No --data -> blocked -> non-zero exit, explicit message, no model.
        code = train.main(["--data", "/nonexistent/path/data.yaml"])
        assert code != 0
        captured = capsys.readouterr()
        assert "TRAINING BLOCKED" in captured.err
