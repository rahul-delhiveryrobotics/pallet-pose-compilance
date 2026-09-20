"""Detection and localisation metric computation and reporting (R5).

This module computes the two **separate** metric families the assignment
requires and reports each as a **distribution over the held-out set with a
sample count**, never as a single point estimate (R5.1-R5.4):

- **Detection metrics** (R5.1, R5.2): per-image precision / recall and
  Average Precision (AP), summarised as distributions (their spread across the
  held-out images) with a ``sample_count``. Answers *"found the pallet?"*.
- **Localisation metrics** (R5.3): per-keypoint **pixel error** and
  **normalised error** (normalised by the ground-truth pallet bbox diagonal),
  **percentiles** (median / p90 / p95), **visibility-conditioned** errors
  (``visible`` vs ``occluded``-but-labelled corners), and **failure rates**
  (missing / gross-outlier keypoints). Each is a distribution with a
  ``sample_count``. Answers *"placed it correctly?"*.

Honesty discipline (R5.5, R26)
------------------------------
The metric *math* here is pure and provenance-agnostic: it computes numbers
from whatever predicted/ground-truth arrays it is given. The **provenance** of
a metric report is resolved separately by :func:`resolve_metric_provenance`,
which assigns ``measured`` **only** when the numbers come from an actual
evaluation run on **assignment-trained** weights. Metrics computed from a
stock/pretrained checkpoint are ``estimated``; metrics from synthetic/simulated
data are ``simulated``; and when no evaluation has run they are ``unavailable``
(explicit, never fabricated).

Every reported distribution is a :class:`DistributionSummary`, which *always*
carries a ``sample_count`` field equal to the number of underlying samples
(the invariant the tagged Property 4 test, task 4.7, will assert). This module
imports and unit-tests **without ultralytics**: it operates on plain
``Detection``/annotation objects and numeric arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..output.provenance import ProvenanceLabel
from ..output.schema import Detection, Keypoint

__all__ = [
    "DistributionSummary",
    "DetectionMetrics",
    "LocalisationMetrics",
    "MetricProvenance",
    "GROSS_OUTLIER_NORMALISED_THRESHOLD",
    "resolve_metric_provenance",
    "iou_xywh",
    "match_detections_to_ground_truth",
    "per_image_detection_scores",
    "compute_detection_metrics",
    "keypoint_pixel_errors",
    "compute_localisation_metrics",
]


# A predicted keypoint whose normalised pixel error (by GT bbox diagonal)
# exceeds this fraction of the pallet's diagonal is treated as a *gross
# outlier* (localisation failure), separate from a *missing* keypoint (the
# predictor did not place it). Documented so the failure-rate semantics are
# explicit; it is the default and can be overridden per call.
GROSS_OUTLIER_NORMALISED_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Distribution summary (always carries sample_count) — Property 4 target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributionSummary:
    """A reported distribution over the held-out set, with a sample count.

    This is the single reporting primitive for every accuracy/error metric in
    the system (R5.2, R5.3): a metric is never reported as a bare point
    estimate, but as the distribution of its per-sample values together with a
    ``sample_count`` (R5.4).

    Attributes
    ----------
    name:
        Human-readable metric name (e.g. ``"per_image_precision"``).
    sample_count:
        The number of underlying samples the distribution is computed from.
        **Always present** and equal to ``len(values)`` (Property 4).
    values:
        The raw per-sample values (kept so consumers can re-plot the
        distribution / recompute statistics honestly).
    mean, std, min, max, median, p90, p95:
        Summary statistics; ``None`` only when ``sample_count == 0`` (an empty
        distribution is reported honestly, not fabricated as zero).
    unit:
        The measurement unit (e.g. ``"px"``, ``"fraction"``, ``"AP"``).
    """

    name: str
    sample_count: int
    values: tuple[float, ...] = field(default_factory=tuple)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if self.sample_count != len(self.values):
            raise ValueError(
                "sample_count must equal the number of underlying values "
                f"(sample_count={self.sample_count}, len(values)={len(self.values)})"
            )

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Sequence[float],
        *,
        unit: Optional[str] = None,
    ) -> "DistributionSummary":
        """Summarise ``values`` into a distribution, always tagging the count.

        An empty ``values`` yields a distribution with ``sample_count == 0`` and
        ``None`` statistics — an honest empty distribution, never a fabricated
        zero (R26.4).
        """
        arr = np.asarray(list(values), dtype=float)
        n = int(arr.size)
        if n == 0:
            return cls(name=name, sample_count=0, values=(), unit=unit)
        return cls(
            name=name,
            sample_count=n,
            values=tuple(float(v) for v in arr),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=float(np.median(arr)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            unit=unit,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-ready dict; ``sample_count`` is always present."""
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
            "unit": self.unit,
            "values": list(self.values),
        }


# ---------------------------------------------------------------------------
# Provenance resolution (R5.5, R26) — measured ONLY from actual eval runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricProvenance:
    """The provenance decision for a metric report, with a reasoning note.

    ``label`` is the :class:`ProvenanceLabel` every metric in the report
    inherits; ``note`` records *why* (for the decision log / README).
    """

    label: ProvenanceLabel
    note: str


def resolve_metric_provenance(
    *,
    ran_eval: bool,
    is_assignment_trained: bool,
    is_synthetic_data: bool = False,
) -> MetricProvenance:
    """Resolve the provenance label for a metric report (R5.5, R26).

    Decision order (honesty first):

    1. **No eval run** -> ``UNAVAILABLE``. Metrics that were never computed are
       stated unavailable, never fabricated (R26.4).
    2. **Synthetic/simulated data** -> ``SIMULATED``. Numbers computed on
       synthetic renders are simulated, never a real-world measurement (R26.2).
    3. **Real eval on assignment-trained weights** -> ``MEASURED``. This is the
       *only* path to a ``measured`` label (R5.5).
    4. **Real eval on a stock/pretrained checkpoint** -> ``ESTIMATED``. A
       checkpoint not trained for this assignment can never back a ``measured``
       metric (R5.5, R26.3).

    Parameters
    ----------
    ran_eval:
        Whether an actual evaluation run produced the numbers.
    is_assignment_trained:
        Whether the detector weights were trained for *this assignment*
        (from :attr:`YoloPoseDetector.is_assignment_trained`).
    is_synthetic_data:
        Whether the evaluation inputs are synthetic/simulated rather than real
        held-out data.
    """
    if not ran_eval:
        return MetricProvenance(
            label=ProvenanceLabel.UNAVAILABLE,
            note=(
                "No evaluation run has been executed; metrics are unavailable "
                "(stated, not fabricated). Run scripts/eval.py on the held-out "
                "set with assignment-trained weights to obtain measured metrics."
            ),
        )
    if is_synthetic_data:
        return MetricProvenance(
            label=ProvenanceLabel.SIMULATED,
            note=(
                "Metrics computed on synthetic/simulated data; labelled "
                "'simulated' and never presented as a real-world measurement "
                "(R26.2)."
            ),
        )
    if is_assignment_trained:
        return MetricProvenance(
            label=ProvenanceLabel.MEASURED,
            note=(
                "Metrics computed from an actual evaluation run on "
                "assignment-trained weights over the held-out set (R5.5)."
            ),
        )
    return MetricProvenance(
        label=ProvenanceLabel.ESTIMATED,
        note=(
            "Metrics computed on a stock/pretrained checkpoint that was NOT "
            "trained for this assignment; labelled 'estimated' and flagged "
            "not-assignment-trained — never 'measured' (R5.5, R26.3)."
        ),
    )


# ---------------------------------------------------------------------------
# Detection metrics (R5.1, R5.2) — precision / recall / AP as distributions
# ---------------------------------------------------------------------------


def iou_xywh(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``[x, y, w, h]`` boxes (top-left origin).

    Returns ``0.0`` when either box has non-positive area or they do not
    overlap.
    """
    ax, ay, aw, ah = (float(v) for v in box_a)
    bx, by, bw, bh = (float(v) for v in box_b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_detections_to_ground_truth(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    *,
    iou_threshold: float = 0.5,
) -> list[Optional[int]]:
    """Greedily match predicted boxes to ground-truth boxes by IoU.

    Predictions are considered in descending score order; each is matched to
    the highest-IoU unmatched ground-truth box whose IoU meets
    ``iou_threshold``. Standard detection-eval matching (one GT per prediction).

    Returns
    -------
    list[Optional[int]]
        For each predicted box (in the *input* order), the index of the matched
        ground-truth box, or ``None`` if the prediction is a false positive.
    """
    n_pred = len(pred_boxes)
    matches: list[Optional[int]] = [None] * n_pred
    if n_pred == 0 or len(gt_boxes) == 0:
        return matches
    order = sorted(range(n_pred), key=lambda i: pred_scores[i], reverse=True)
    used_gt: set[int] = set()
    for pi in order:
        best_iou = iou_threshold
        best_gt: Optional[int] = None
        for gi, gt in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            iou = iou_xywh(pred_boxes[pi], gt)
            if iou >= best_iou:
                best_iou = iou
                best_gt = gi
        if best_gt is not None:
            matches[pi] = best_gt
            used_gt.add(best_gt)
    return matches


def _average_precision(
    scores: Sequence[float],
    is_true_positive: Sequence[bool],
    n_ground_truth: int,
) -> float:
    """Average precision (area under the PR curve) for one image.

    Uses the all-points (continuous) interpolation of the precision-recall
    curve. When there is no ground truth, AP is ``1.0`` if there are also no
    predictions (nothing to find, nothing wrongly reported) else ``0.0``.
    """
    if n_ground_truth == 0:
        return 1.0 if len(scores) == 0 else 0.0
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.asarray(is_true_positive, dtype=float)[order]
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / float(n_ground_truth)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    # All-points interpolation: prepend (recall=0, precision=1).
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Make precision monotonically decreasing (envelope).
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return max(0.0, min(1.0, ap))


@dataclass(frozen=True)
class _ImageDetectionScore:
    precision: float
    recall: float
    ap: float


def per_image_detection_scores(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    *,
    iou_threshold: float = 0.5,
) -> _ImageDetectionScore:
    """Compute precision, recall, and AP for a single image.

    Precision/recall use the ``iou_threshold`` matching; AP is the area under
    the per-image PR curve. Edge cases:

    - No GT, no predictions -> precision/recall/AP all ``1.0`` (correct empty).
    - No GT, some predictions -> precision ``0.0``, recall ``1.0`` (all FPs),
      AP ``0.0``.
    - Some GT, no predictions -> precision ``1.0``? No: precision is undefined
      with no predictions and is reported ``0.0`` recall, AP ``0.0``; precision
      is set to ``1.0`` only when there are also no GTs. Here precision ``0.0``.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    if n_pred == 0 and n_gt == 0:
        return _ImageDetectionScore(precision=1.0, recall=1.0, ap=1.0)
    matches = match_detections_to_ground_truth(
        pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold
    )
    tp_flags = [m is not None for m in matches]
    tp = sum(1 for f in tp_flags if f)
    fp = n_pred - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 1.0
    ap = _average_precision(pred_scores, tp_flags, n_gt)
    return _ImageDetectionScore(precision=precision, recall=recall, ap=ap)


@dataclass(frozen=True)
class DetectionMetrics:
    """Detection accuracy reported as distributions with sample counts (R5.2).

    Each field is a :class:`DistributionSummary` over the per-image scores of
    the held-out set (so detection accuracy is a *distribution*, not a point
    estimate). ``provenance`` is resolved by :func:`resolve_metric_provenance`.
    """

    precision: DistributionSummary
    recall: DistributionSummary
    average_precision: DistributionSummary
    provenance: ProvenanceLabel
    provenance_note: str
    iou_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "average_precision": self.average_precision.to_dict(),
            "provenance": self.provenance.value,
            "provenance_note": self.provenance_note,
            "iou_threshold": self.iou_threshold,
        }


def _pallet_boxes(detections: Sequence[Detection]) -> list[list[float]]:
    """Extract pallet-class boxes from detections (the localisation target)."""
    return [list(d.bbox) for d in detections if d.cls == "pallet"]


def _pallet_scores(detections: Sequence[Detection]) -> list[float]:
    return [float(d.score) for d in detections if d.cls == "pallet"]


def compute_detection_metrics(
    predictions_per_image: Sequence[Sequence[Detection]],
    ground_truth_boxes_per_image: Sequence[Sequence[Sequence[float]]],
    *,
    provenance: MetricProvenance,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Compute detection metrics over a held-out set as distributions (R5.2).

    Parameters
    ----------
    predictions_per_image:
        Predicted :class:`Detection`\\ s for each held-out image. Only
        pallet-class boxes are scored.
    ground_truth_boxes_per_image:
        Ground-truth pallet ``[x, y, w, h]`` boxes for each held-out image
        (same order/length as ``predictions_per_image``).
    provenance:
        The resolved metric provenance (see :func:`resolve_metric_provenance`).
    iou_threshold:
        IoU threshold for a true-positive match.

    Returns
    -------
    DetectionMetrics
        Precision / recall / AP as per-image distributions with sample counts.
    """
    if len(predictions_per_image) != len(ground_truth_boxes_per_image):
        raise ValueError(
            "predictions_per_image and ground_truth_boxes_per_image must have "
            "the same length (one entry per held-out image)"
        )
    precisions: list[float] = []
    recalls: list[float] = []
    aps: list[float] = []
    for preds, gts in zip(predictions_per_image, ground_truth_boxes_per_image):
        pred_boxes = _pallet_boxes(preds)
        pred_scores = _pallet_scores(preds)
        gt_boxes = [list(b) for b in gts]
        score = per_image_detection_scores(
            pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold
        )
        precisions.append(score.precision)
        recalls.append(score.recall)
        aps.append(score.ap)
    return DetectionMetrics(
        precision=DistributionSummary.from_values(
            "per_image_precision", precisions, unit="fraction"
        ),
        recall=DistributionSummary.from_values(
            "per_image_recall", recalls, unit="fraction"
        ),
        average_precision=DistributionSummary.from_values(
            "per_image_average_precision", aps, unit="AP"
        ),
        provenance=provenance.label,
        provenance_note=provenance.note,
        iou_threshold=iou_threshold,
    )


# ---------------------------------------------------------------------------
# Localisation metrics (R5.3) — keypoint pixel/normalised error, percentiles,
# visibility-conditioned splits, failure rates
# ---------------------------------------------------------------------------


def _bbox_diagonal(bbox: Sequence[float]) -> float:
    """Diagonal length of an ``[x, y, w, h]`` box (the normalisation scale)."""
    _, _, w, h = (float(v) for v in bbox)
    return float(np.hypot(w, h))


@dataclass(frozen=True)
class _KeypointError:
    name: str
    pixel_error: float
    normalised_error: float
    gt_visibility: str  # "visible" | "occluded"


def keypoint_pixel_errors(
    pred_keypoints: Sequence[Keypoint],
    gt_keypoints: Sequence[Mapping[str, Any]],
    gt_bbox: Sequence[float],
) -> tuple[list[_KeypointError], int]:
    """Per-keypoint pixel + normalised error for one matched pallet.

    A ground-truth keypoint is scored only when it is *labelled* (visibility
    ``visible`` or ``occluded`` — not ``absent``) and carries pixel coords. The
    matched prediction must place the keypoint (non-null ``u``/``v``); when it
    does not, the keypoint is counted as **missing** (a localisation failure)
    and contributes to the missing count rather than the error list.

    Parameters
    ----------
    pred_keypoints:
        Predicted :class:`Keypoint`\\ s (by ``name``).
    gt_keypoints:
        Ground-truth keypoints as mappings with ``name``, ``x``, ``y``,
        ``visibility`` (e.g. from :class:`KeypointAnnotation` via ``vars``), or
        objects exposing those attributes.
    gt_bbox:
        The ground-truth pallet ``[x, y, w, h]`` box used to normalise error by
        its diagonal.

    Returns
    -------
    tuple[list[_KeypointError], int]
        The per-keypoint errors for scored keypoints, and the count of
        **missing** keypoints (labelled in GT but not placed by the predictor).
    """
    pred_by_name = {kp.name: kp for kp in pred_keypoints}
    diag = _bbox_diagonal(gt_bbox)
    errors: list[_KeypointError] = []
    missing = 0
    for gt in gt_keypoints:
        name = _get(gt, "name")
        visibility = _get(gt, "visibility")
        gx = _get(gt, "x")
        gy = _get(gt, "y")
        if visibility == "absent" or gx is None or gy is None:
            continue  # not a labelled GT keypoint; nothing to score
        pred = pred_by_name.get(name)
        if pred is None or pred.u is None or pred.v is None:
            missing += 1
            continue
        pixel_error = float(np.hypot(float(pred.u) - float(gx), float(pred.v) - float(gy)))
        normalised = pixel_error / diag if diag > 0 else float("inf")
        errors.append(
            _KeypointError(
                name=str(name),
                pixel_error=pixel_error,
                normalised_error=normalised,
                gt_visibility=str(visibility),
            )
        )
    return errors, missing


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a mapping or an attribute-bearing object."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


@dataclass(frozen=True)
class LocalisationMetrics:
    """Localisation accuracy reported as distributions with sample counts (R5.3).

    Fields
    ------
    pixel_error / normalised_error:
        Distributions of per-keypoint pixel error and bbox-diagonal-normalised
        error over all scored keypoints (percentiles available via the
        distribution's ``median``/``p90``/``p95``, R5.3).
    per_keypoint_pixel_error:
        Per-keypoint-name pixel-error distributions (one per corner).
    pixel_error_visible / pixel_error_occluded:
        Visibility-conditioned pixel-error distributions (``visible`` vs
        ``occluded``-but-labelled GT corners).
    missing_rate / gross_outlier_rate:
        Failure-rate distributions: the per-pallet fraction of labelled GT
        keypoints that were missing (not placed) or gross outliers (normalised
        error beyond ``gross_outlier_threshold``).
    provenance / provenance_note:
        Resolved metric provenance (measured only from an actual eval run on
        assignment-trained weights, R5.5).
    """

    pixel_error: DistributionSummary
    normalised_error: DistributionSummary
    per_keypoint_pixel_error: Mapping[str, DistributionSummary]
    pixel_error_visible: DistributionSummary
    pixel_error_occluded: DistributionSummary
    missing_rate: DistributionSummary
    gross_outlier_rate: DistributionSummary
    provenance: ProvenanceLabel
    provenance_note: str
    gross_outlier_threshold: float = GROSS_OUTLIER_NORMALISED_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_error": self.pixel_error.to_dict(),
            "normalised_error": self.normalised_error.to_dict(),
            "per_keypoint_pixel_error": {
                k: v.to_dict() for k, v in self.per_keypoint_pixel_error.items()
            },
            "pixel_error_visible": self.pixel_error_visible.to_dict(),
            "pixel_error_occluded": self.pixel_error_occluded.to_dict(),
            "missing_rate": self.missing_rate.to_dict(),
            "gross_outlier_rate": self.gross_outlier_rate.to_dict(),
            "provenance": self.provenance.value,
            "provenance_note": self.provenance_note,
            "gross_outlier_threshold": self.gross_outlier_threshold,
        }


def compute_localisation_metrics(
    matched_predictions: Sequence[Sequence[Keypoint]],
    matched_ground_truth: Sequence[Sequence[Mapping[str, Any]]],
    matched_gt_bboxes: Sequence[Sequence[float]],
    *,
    provenance: MetricProvenance,
    gross_outlier_threshold: float = GROSS_OUTLIER_NORMALISED_THRESHOLD,
) -> LocalisationMetrics:
    """Compute localisation metrics over matched pallets as distributions (R5.3).

    Each element of the three parallel sequences corresponds to one *matched*
    predicted↔GT pallet: its predicted keypoints, its GT keypoints, and its GT
    bbox (for normalisation). Localisation is evaluated **separately** from
    detection (R5.1), so only matched pallets feed the keypoint errors here.

    Parameters
    ----------
    matched_predictions / matched_ground_truth / matched_gt_bboxes:
        Parallel sequences (same length) over matched pallets.
    provenance:
        Resolved metric provenance.
    gross_outlier_threshold:
        Normalised-error threshold above which a placed keypoint is a gross
        outlier (localisation failure).

    Returns
    -------
    LocalisationMetrics
        Pixel/normalised error, per-keypoint, visibility-conditioned, and
        failure-rate distributions, each with a sample count.
    """
    n = len(matched_predictions)
    if not (len(matched_ground_truth) == n == len(matched_gt_bboxes)):
        raise ValueError(
            "matched_predictions, matched_ground_truth, and matched_gt_bboxes "
            "must have the same length (one entry per matched pallet)"
        )

    pixel_errors: list[float] = []
    normalised_errors: list[float] = []
    per_kp_pixel: dict[str, list[float]] = {}
    visible_px: list[float] = []
    occluded_px: list[float] = []
    missing_rates: list[float] = []
    gross_rates: list[float] = []

    for preds, gts, bbox in zip(
        matched_predictions, matched_ground_truth, matched_gt_bboxes
    ):
        errors, missing = keypoint_pixel_errors(preds, gts, bbox)
        n_labelled = missing + len(errors)
        n_gross = 0
        for e in errors:
            pixel_errors.append(e.pixel_error)
            normalised_errors.append(e.normalised_error)
            per_kp_pixel.setdefault(e.name, []).append(e.pixel_error)
            if e.gt_visibility == "visible":
                visible_px.append(e.pixel_error)
            elif e.gt_visibility == "occluded":
                occluded_px.append(e.pixel_error)
            if e.normalised_error > gross_outlier_threshold:
                n_gross += 1
        if n_labelled > 0:
            missing_rates.append(missing / n_labelled)
            gross_rates.append(n_gross / n_labelled)

    per_keypoint = {
        name: DistributionSummary.from_values(
            f"pixel_error[{name}]", vals, unit="px"
        )
        for name, vals in sorted(per_kp_pixel.items())
    }

    return LocalisationMetrics(
        pixel_error=DistributionSummary.from_values(
            "keypoint_pixel_error", pixel_errors, unit="px"
        ),
        normalised_error=DistributionSummary.from_values(
            "keypoint_normalised_error", normalised_errors, unit="fraction_of_diag"
        ),
        per_keypoint_pixel_error=per_keypoint,
        pixel_error_visible=DistributionSummary.from_values(
            "keypoint_pixel_error[visible]", visible_px, unit="px"
        ),
        pixel_error_occluded=DistributionSummary.from_values(
            "keypoint_pixel_error[occluded]", occluded_px, unit="px"
        ),
        missing_rate=DistributionSummary.from_values(
            "per_pallet_missing_keypoint_rate", missing_rates, unit="fraction"
        ),
        gross_outlier_rate=DistributionSummary.from_values(
            "per_pallet_gross_outlier_rate", gross_rates, unit="fraction"
        ),
        provenance=provenance.label,
        provenance_note=provenance.note,
        gross_outlier_threshold=gross_outlier_threshold,
    )
