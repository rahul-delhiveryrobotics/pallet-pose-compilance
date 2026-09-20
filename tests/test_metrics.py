"""Unit tests for detection/localisation metric reporting (task 4.6).

Covers :mod:`pallet_pose_compliance.detection.metrics`:

- **Distribution summary** always carries a correct ``sample_count`` (R5.4,
  Property 4 shape) and honest empty-distribution handling (R26.4).
- **Detection metrics** (R5.1, R5.2): precision/recall/AP on toy data, reported
  as per-image distributions with sample counts.
- **Localisation metrics** (R5.3): per-keypoint pixel + normalised error,
  percentiles, visibility-conditioned splits, and failure rates.
- **Provenance resolution** (R5.5, R26): ``measured`` only from an actual eval
  run on assignment-trained weights; ``estimated`` for stock; ``simulated`` for
  synthetic data; ``unavailable`` when no eval ran.
"""

from __future__ import annotations

import math

import pytest

from pallet_pose_compliance.detection.metrics import (
    GROSS_OUTLIER_NORMALISED_THRESHOLD,
    DistributionSummary,
    compute_detection_metrics,
    compute_localisation_metrics,
    iou_xywh,
    keypoint_pixel_errors,
    match_detections_to_ground_truth,
    per_image_detection_scores,
    resolve_metric_provenance,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel
from pallet_pose_compliance.output.schema import Detection, Keypoint


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _pallet(bbox, score, keypoints=None):
    return Detection(cls="pallet", bbox=list(bbox), score=score, keypoints=keypoints or [])


def _kp(name, u, v, visibility="visible"):
    if visibility == "absent":
        return Keypoint(name=name, u=None, v=None, visibility="absent")
    return Keypoint(name=name, u=u, v=v, visibility=visibility)


def _gt_kp(name, x, y, visibility="visible"):
    return {"name": name, "x": x, "y": y, "visibility": visibility}


def _measured():
    return resolve_metric_provenance(ran_eval=True, is_assignment_trained=True)


# ---------------------------------------------------------------------------
# DistributionSummary: sample_count discipline (Property 4 shape)
# ---------------------------------------------------------------------------


def test_distribution_sample_count_equals_len_values():
    dist = DistributionSummary.from_values("x", [1.0, 2.0, 3.0, 4.0])
    assert dist.sample_count == 4
    assert dist.sample_count == len(dist.values)


def test_distribution_statistics_are_correct():
    dist = DistributionSummary.from_values("x", [0.0, 10.0])
    assert dist.mean == pytest.approx(5.0)
    assert dist.min == 0.0
    assert dist.max == 10.0
    assert dist.median == pytest.approx(5.0)


def test_distribution_percentiles_present():
    dist = DistributionSummary.from_values("x", list(range(101)), unit="px")
    assert dist.median == pytest.approx(50.0)
    assert dist.p90 == pytest.approx(90.0)
    assert dist.p95 == pytest.approx(95.0)


def test_empty_distribution_is_honest_not_zero():
    dist = DistributionSummary.from_values("x", [])
    assert dist.sample_count == 0
    assert dist.mean is None
    assert dist.median is None
    assert dist.values == ()


def test_distribution_rejects_mismatched_sample_count():
    with pytest.raises(ValueError):
        DistributionSummary(name="x", sample_count=3, values=(1.0, 2.0))


def test_distribution_to_dict_always_has_sample_count():
    d = DistributionSummary.from_values("x", []).to_dict()
    assert "sample_count" in d and d["sample_count"] == 0


# ---------------------------------------------------------------------------
# Provenance resolution (R5.5, R26)
# ---------------------------------------------------------------------------


def test_measured_only_from_eval_on_assignment_trained():
    prov = resolve_metric_provenance(ran_eval=True, is_assignment_trained=True)
    assert prov.label is ProvenanceLabel.MEASURED


def test_stock_checkpoint_is_estimated_never_measured():
    prov = resolve_metric_provenance(ran_eval=True, is_assignment_trained=False)
    assert prov.label is ProvenanceLabel.ESTIMATED


def test_no_eval_run_is_unavailable():
    prov = resolve_metric_provenance(ran_eval=False, is_assignment_trained=True)
    assert prov.label is ProvenanceLabel.UNAVAILABLE


def test_synthetic_data_is_simulated_even_if_trained():
    prov = resolve_metric_provenance(
        ran_eval=True, is_assignment_trained=True, is_synthetic_data=True
    )
    assert prov.label is ProvenanceLabel.SIMULATED


# ---------------------------------------------------------------------------
# IoU + matching
# ---------------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    assert iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou_xywh([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 region -> inter 50, union 150.
    assert iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50.0 / 150.0)


def test_matching_assigns_best_iou_greedily_by_score():
    pred_boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    pred_scores = [0.9, 0.4]
    gt_boxes = [[0, 0, 10, 10]]
    matches = match_detections_to_ground_truth(pred_boxes, pred_scores, gt_boxes)
    # Higher-scored prediction wins the single GT; the other is a false positive.
    assert matches[0] == 0
    assert matches[1] is None


# ---------------------------------------------------------------------------
# Detection metrics (R5.1, R5.2)
# ---------------------------------------------------------------------------


def test_perfect_detection_scores():
    score = per_image_detection_scores(
        [[0, 0, 10, 10]], [0.9], [[0, 0, 10, 10]]
    )
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.ap == pytest.approx(1.0)


def test_false_positive_lowers_precision():
    score = per_image_detection_scores(
        [[0, 0, 10, 10], [500, 500, 10, 10]], [0.9, 0.8], [[0, 0, 10, 10]]
    )
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(1.0)


def test_missed_gt_lowers_recall():
    score = per_image_detection_scores(
        [[0, 0, 10, 10]], [0.9], [[0, 0, 10, 10], [500, 500, 10, 10]]
    )
    assert score.recall == pytest.approx(0.5)
    assert score.precision == pytest.approx(1.0)


def test_empty_image_correct_is_perfect():
    score = per_image_detection_scores([], [], [])
    assert score.precision == 1.0 and score.recall == 1.0 and score.ap == 1.0


def test_detection_metrics_reported_as_distribution_with_sample_count():
    preds = [
        [_pallet([0, 0, 10, 10], 0.9)],
        [_pallet([0, 0, 10, 10], 0.8), _pallet([500, 500, 10, 10], 0.7)],
    ]
    gts = [
        [[0, 0, 10, 10]],
        [[0, 0, 10, 10]],
    ]
    metrics = compute_detection_metrics(preds, gts, provenance=_measured())
    assert metrics.precision.sample_count == 2
    assert metrics.recall.sample_count == 2
    assert metrics.average_precision.sample_count == 2
    assert metrics.provenance is ProvenanceLabel.MEASURED
    # image 0 precision 1.0, image 1 precision 0.5 -> mean 0.75
    assert metrics.precision.mean == pytest.approx(0.75)


def test_detection_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_detection_metrics([[]], [], provenance=_measured())


def test_detection_metrics_only_score_pallet_class():
    preds = [[_pallet([0, 0, 10, 10], 0.9), Detection(cls="box", bbox=[0, 0, 5, 5], score=0.9)]]
    gts = [[[0, 0, 10, 10]]]
    metrics = compute_detection_metrics(preds, gts, provenance=_measured())
    # The box detection must not count as a pallet false positive.
    assert metrics.precision.values[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Localisation metrics (R5.3)
# ---------------------------------------------------------------------------


def _eight_pred(offset=0.0):
    names = [f"bottom_corner_{i}" for i in range(4)] + [f"top_corner_{i}" for i in range(4)]
    coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 2), (10, 2), (10, 12), (0, 12)]
    return [_kp(n, x + offset, y + offset) for n, (x, y) in zip(names, coords)]


def _eight_gt(visibilities=None):
    names = [f"bottom_corner_{i}" for i in range(4)] + [f"top_corner_{i}" for i in range(4)]
    coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 2), (10, 2), (10, 12), (0, 12)]
    vis = visibilities or ["visible"] * 8
    return [_gt_kp(n, x, y, v) for n, (x, y), v in zip(names, coords, vis)]


def test_zero_pixel_error_when_prediction_matches_gt():
    errors, missing = keypoint_pixel_errors(_eight_pred(0.0), _eight_gt(), [0, 0, 10, 12])
    assert missing == 0
    assert all(e.pixel_error == pytest.approx(0.0) for e in errors)


def test_pixel_and_normalised_error_computed():
    # Shift every prediction by (3, 4) -> pixel error 5 for each.
    names = [f"bottom_corner_{i}" for i in range(4)] + [f"top_corner_{i}" for i in range(4)]
    coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 2), (10, 2), (10, 12), (0, 12)]
    pred = [_kp(n, x + 3.0, y + 4.0) for n, (x, y) in zip(names, coords)]
    bbox = [0, 0, 10, 12]  # diagonal sqrt(244)
    errors, missing = keypoint_pixel_errors(pred, _eight_gt(), bbox)
    diag = math.hypot(10, 12)
    assert all(e.pixel_error == pytest.approx(5.0) for e in errors)
    assert all(e.normalised_error == pytest.approx(5.0 / diag) for e in errors)


def test_missing_keypoint_counted_as_failure_not_error():
    pred = _eight_pred()[:-1]  # drop the last predicted keypoint
    errors, missing = keypoint_pixel_errors(pred, _eight_gt(), [0, 0, 10, 12])
    assert missing == 1
    assert len(errors) == 7


def test_absent_gt_keypoints_not_scored():
    vis = ["visible"] * 7 + ["absent"]
    gt = _eight_gt(vis)
    # The absent GT keypoint carries null coords per annotation discipline.
    gt[-1]["x"] = None
    gt[-1]["y"] = None
    errors, missing = keypoint_pixel_errors(_eight_pred(), gt, [0, 0, 10, 12])
    assert len(errors) == 7
    assert missing == 0


def test_localisation_visibility_conditioned_split():
    vis = ["visible"] * 4 + ["occluded"] * 4
    metrics = compute_localisation_metrics(
        [_eight_pred()], [_eight_gt(vis)], [[0, 0, 10, 12]], provenance=_measured()
    )
    assert metrics.pixel_error_visible.sample_count == 4
    assert metrics.pixel_error_occluded.sample_count == 4


def test_localisation_per_keypoint_distributions():
    metrics = compute_localisation_metrics(
        [_eight_pred(), _eight_pred()],
        [_eight_gt(), _eight_gt()],
        [[0, 0, 10, 12], [0, 0, 10, 12]],
        provenance=_measured(),
    )
    # 8 keypoint names, each seen twice.
    assert len(metrics.per_keypoint_pixel_error) == 8
    for dist in metrics.per_keypoint_pixel_error.values():
        assert dist.sample_count == 2


def test_localisation_failure_rates():
    # One pallet: 1 missing keypoint out of 8 labelled -> missing_rate 1/8.
    pred = _eight_pred()[:-1]
    metrics = compute_localisation_metrics(
        [pred], [_eight_gt()], [[0, 0, 10, 12]], provenance=_measured()
    )
    assert metrics.missing_rate.sample_count == 1
    assert metrics.missing_rate.values[0] == pytest.approx(1.0 / 8.0)


def test_gross_outlier_rate():
    # Move one prediction far away so its normalised error exceeds the threshold.
    names = [f"bottom_corner_{i}" for i in range(4)] + [f"top_corner_{i}" for i in range(4)]
    coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 2), (10, 2), (10, 12), (0, 12)]
    pred = [_kp(n, x, y) for n, (x, y) in zip(names, coords)]
    # Displace the first keypoint by a huge amount (normalised >> threshold).
    pred[0] = _kp("bottom_corner_0", 10000.0, 10000.0)
    metrics = compute_localisation_metrics(
        [pred], [_eight_gt()], [[0, 0, 10, 12]], provenance=_measured()
    )
    assert metrics.gross_outlier_rate.values[0] == pytest.approx(1.0 / 8.0)


def test_localisation_metrics_carry_provenance():
    metrics = compute_localisation_metrics(
        [_eight_pred()], [_eight_gt()], [[0, 0, 10, 12]],
        provenance=resolve_metric_provenance(ran_eval=True, is_assignment_trained=False),
    )
    assert metrics.provenance is ProvenanceLabel.ESTIMATED


def test_localisation_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_localisation_metrics([_eight_pred()], [], [], provenance=_measured())
