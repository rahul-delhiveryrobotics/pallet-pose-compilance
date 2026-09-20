"""Detector wrapper and detection/localisation metrics (R4, R5).

Exposes the real YOLO-pose detector wrapper and its weights-provenance honesty
record (R5.5, R26.3). The stub detector remains available for the early smoke
pipeline until it is swapped in task 4.8.
"""

from .detector import (
    KEYPOINT_NAMES,
    DetectorUnavailableError,
    WeightsProvenance,
    YoloPoseDetector,
    detections_from_yolo_result,
)
from .metrics import (
    GROSS_OUTLIER_NORMALISED_THRESHOLD,
    DetectionMetrics,
    DistributionSummary,
    LocalisationMetrics,
    MetricProvenance,
    compute_detection_metrics,
    compute_localisation_metrics,
    iou_xywh,
    keypoint_pixel_errors,
    match_detections_to_ground_truth,
    per_image_detection_scores,
    resolve_metric_provenance,
)

__all__ = [
    "KEYPOINT_NAMES",
    "DetectorUnavailableError",
    "WeightsProvenance",
    "YoloPoseDetector",
    "detections_from_yolo_result",
    # metrics (R5)
    "GROSS_OUTLIER_NORMALISED_THRESHOLD",
    "DetectionMetrics",
    "DistributionSummary",
    "LocalisationMetrics",
    "MetricProvenance",
    "compute_detection_metrics",
    "compute_localisation_metrics",
    "iou_xywh",
    "keypoint_pixel_errors",
    "match_detections_to_ground_truth",
    "per_image_detection_scores",
    "resolve_metric_provenance",
]
