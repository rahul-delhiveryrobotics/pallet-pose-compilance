"""Verifiable-subset SOP-PAL-03 checks with config-driven thresholds (task 9.3).

This module implements the *verifiable subset* of SOP-PAL-03 (design: SOP rules
row) — the rules that can be measured from the available single-side view once
detections and (where needed) a metric pose are available:

===== =============================== ============================ =============
Rule  Name                            Measurement                  Threshold key
===== =============================== ============================ =============
1     No box overhang > 3 cm          max box overhang past edge   overhang_max_cm
2     Load height <= 1.8 m            load height above floor      load_height_max_m
3     Aligned columns (<= 15 deg)     max box rotation vs pallet   column_tilt_max_deg
5     Load is stretch-wrapped         wrap presence (heuristic)    (threshold-free)
7     Load centroid within 10 cm      centroid offset from centre  centroid_offset_max_cm
===== =============================== ============================ =============

Scope of THIS task (9.3)
------------------------
The focus here is the *check machinery*: read thresholds from
:class:`~pallet_pose_compliance.sop.config.SopConfig` (R18.1/R18.2), compute a
measurement from the available geometry using documented, testable formulas,
and emit a :class:`~pallet_pose_compliance.output.schema.SopCheck` carrying
``measurement``, ``measurement_unit``, ``threshold_used``, ``threshold_source``
and a ``status`` of ``pass``/``fail`` derived by comparing the measurement to
the config threshold. Because editing the YAML changes ``threshold_used``, the
boundary moves with **no code change** (R18.2, the design's Property 19
behaviour).

Status & confidence discipline (task 9.7)
-----------------------------------------
Layered on top of the measurement machinery above:

- **not_verifiable rules (4, 6, 8)** are ALWAYS ``unresolved`` (R15.3) - never
  pass/fail/omitted - and carry no ``PROCESSING_FAILED`` reason (that would
  imply a runtime failure); the triage classification + ``hidden_regions``
  document why they are undecidable from a single view.
- **Pose-dependent checks (rules 1, 3, 7)** are ``invalidated`` with a
  ``Reason_Code`` when the pose is unavailable (R15.4); non-pose-dependent
  verifiable checks (2 load height, 5 wrap) are unaffected by pose loss.
- Every **implemented** check carries a ``confidence`` and an honest
  ``confidence_semantics`` label (R16.1-R16.4): geometry-driven checks report a
  ``raw_detector_score`` (never a calibrated pass probability), the stretch-wrap
  presence check reports ``heuristic_evidence_quality``, and - because no
  calibration mapping exists - ``calibrated_pass_probability`` is never emitted.

This module is wired into the pipeline in task 9.10.

Geometry conventions
--------------------
Detections carry ``bbox = [x, y, w, h]`` in image pixels (top-left origin,
``x`` right, ``y`` down). The verifiable-subset measurements are computed from
the pallet detection's bbox (the reference frame) and the box detections
(the load), converted to metric using the pallet's known width as a scale — a
clearly documented, unit-testable formula suitable for the current stub
detector. Where an input required for a measurement is absent, the measurement
is ``None`` and the check is left ``unresolved`` (real reason-coding is task
9.7).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..geometry.pallet_model import NOMINAL_PALLET_LENGTH_M
from ..output.provenance import ProvenanceLabel, ReasonCode
from ..output.schema import Detection, PoseResult, SopCheck
from .triage import RULE_IDS, get_entry, pose_dependent_rule_ids
from .config import SopConfig

__all__ = [
    "check_overhang",
    "check_load_height",
    "check_column_tilt",
    "check_wrap_presence",
    "check_centroid_offset",
    "analyze",
]

# Metres-per-pixel would come from calibration/pose downstream; for the pure,
# unit-testable measurement formulas here we scale using the pallet's known
# physical width against its detected pixel width.
_EPS = 1e-9

# ---------------------------------------------------------------------------
# Confidence semantics (task 9.7, R16.1-R16.4)
# ---------------------------------------------------------------------------
#
# Every *implemented* check attaches a ``confidence`` value AND a
# ``confidence_semantics`` label that HONESTLY names how that number was
# derived (R16.1/R16.2). The three allowed labels (schema/design) are:
#
#   * ``raw_detector_score``          - the confidence is a raw detector/
#       detection score with NO calibration to a probability. It reflects how
#       confident the detector is in the underlying detections that drive the
#       geometric measurement (overhang, load height, column tilt, centroid).
#       It is NOT a probability that the rule passes (R16.3/R16.4).
#   * ``heuristic_evidence_quality``  - the confidence is a heuristic quality-
#       of-evidence score for an appearance/presence check (stretch-wrap). It
#       is not a detector probability and not a calibrated pass probability.
#   * ``calibrated_pass_probability`` - reserved for a confidence produced by a
#       real calibration mapping (isotonic/Platt/etc.) from evidence to a
#       pass probability. NO such calibration exists in the current system, so
#       this label is DELIBERATELY NEVER emitted here — labelling an
#       uncalibrated score as calibrated would be dishonest (R16.3/R16.4).
#
# ``confidence_semantics`` string constants (must match the schema Literal).
SEMANTICS_RAW_DETECTOR_SCORE = "raw_detector_score"
SEMANTICS_HEURISTIC_EVIDENCE_QUALITY = "heuristic_evidence_quality"
SEMANTICS_CALIBRATED_PASS_PROBABILITY = "calibrated_pass_probability"

#: Fallback confidence when a detection score cannot be derived (e.g. no
#: contributing detections). Kept low + honest; never labelled as calibrated.
_DEFAULT_DETECTOR_CONFIDENCE = 0.0


def _detector_confidence(detections: Sequence[Detection]) -> float:
    """Raw detector-score confidence for a geometry-driven check.

    Derived as the minimum detection score across the detections that
    contribute to the measurement (pallet + boxes) - the measurement is only as
    trustworthy as its least-confident supporting detection. This is a *raw
    detector score*, NOT a calibrated pass probability (R16.3/R16.4).
    """
    scores = [det.score for det in detections]
    if not scores:
        return _DEFAULT_DETECTOR_CONFIDENCE
    return min(scores)


def _pose_available(pose: Optional[PoseResult]) -> bool:
    return pose is not None and pose.pose_status == "available"


def _pallet_detection(detections: Sequence[Detection]) -> Optional[Detection]:
    for det in detections:
        if det.cls == "pallet":
            return det
    return None


def _box_detections(detections: Sequence[Detection]) -> list[Detection]:
    return [det for det in detections if det.cls == "box"]


def _metres_per_pixel(pallet: Detection) -> Optional[float]:
    """Scale factor from the pallet's known physical width vs its pixel width.

    Uses ``bbox = [x, y, w, h]`` width in pixels mapped to the nominal pallet
    length (the long, typically camera-facing axis). Returns ``None`` when the
    pixel width is degenerate.
    """
    _x, _y, w, _h = pallet.bbox
    if w <= _EPS:
        return None
    return NOMINAL_PALLET_LENGTH_M / w


def _make_check(
    rule_id: int,
    *,
    status: str,
    measurement: Optional[float],
    threshold_value: Optional[float],
    threshold_unit: Optional[str],
    threshold_source: Optional[str],
    confidence: Optional[float] = None,
    confidence_semantics: Optional[str] = None,
    provenance: ProvenanceLabel = ProvenanceLabel.MEASURED,
    reason_code: Optional[ReasonCode] = None,
) -> SopCheck:
    """Assemble a :class:`SopCheck` for ``rule_id`` from the triage table.

    ``measurement_unit`` is only set when a ``measurement`` is present (the
    schema forbids a unit without a value). ``triage``/``rule_name``/
    ``assumptions``/``hidden_regions`` come from the authoritative triage table
    so they are defined in one place.
    """
    entry = get_entry(rule_id)
    measurement_unit = threshold_unit if measurement is not None else None
    return SopCheck(
        rule_id=rule_id,
        rule_name=entry.rule_name,
        triage=entry.classification,
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        confidence_semantics=confidence_semantics,  # type: ignore[arg-type]
        measurement=measurement,
        measurement_unit=measurement_unit,
        threshold_used=threshold_value,
        threshold_source=threshold_source,
        assumptions=list(entry.assumptions),
        hidden_regions=list(entry.hidden_regions),
        reason_code=reason_code,
        provenance=provenance,
    )


def _status_from_measurement(measurement: float, threshold: float) -> str:
    """``pass`` when ``measurement <= threshold`` (at-or-below the boundary)."""
    return "pass" if measurement <= threshold else "fail"


#: The pose-dependent rule ids (1, 3, 7) sourced from the authoritative triage
#: table so pose-dependence is defined in one place (task 9.7 consumes 15.4).
POSE_DEPENDENT_RULE_IDS = pose_dependent_rule_ids()


def _pose_invalidated_check(
    rule_id: int,
    pose_result: Optional[PoseResult],
    threshold,
) -> Optional[SopCheck]:
    """Return an ``invalidated`` SopCheck when a pose-dependent rule has no pose.

    Implements the pose-dependent invalidation discipline (R15.4): when the
    pose is unavailable (``None`` or ``pose_status != "available"``), every
    pose-dependent check (rules 1, 3, 7 per :func:`pose_dependent_rule_ids`)
    MUST be ``invalidated`` and carry a ``Reason_Code`` explaining why. The
    pose's own ``reason_code`` is propagated when present (so the SOP output
    echoes *why* the pose was unavailable); otherwise a sensible default
    (:attr:`ReasonCode.POSE_INSUFFICIENT_KEYPOINTS`) is used.

    Returns ``None`` for rules that are not pose-dependent or when the pose is
    available, letting the caller proceed to measure normally. An invalidated
    check carries no measurement (``measurement=None``) and no calibrated
    confidence (``confidence=None``) - there is no defensible per-check
    confidence when the measurement itself could not be produced.
    """
    if rule_id not in POSE_DEPENDENT_RULE_IDS:
        return None
    if _pose_available(pose_result):
        return None
    reason_code = ReasonCode.POSE_INSUFFICIENT_KEYPOINTS
    if pose_result is not None and pose_result.reason_code is not None:
        reason_code = pose_result.reason_code
    return _make_check(
        rule_id,
        status="invalidated",
        measurement=None,
        threshold_value=threshold.value if threshold is not None else None,
        threshold_unit=threshold.unit if threshold is not None else None,
        threshold_source=threshold.source if threshold is not None else None,
        confidence=None,
        confidence_semantics=None,
        provenance=ProvenanceLabel.UNAVAILABLE,
        reason_code=reason_code,
    )


# ---------------------------------------------------------------------------
# Rule 1 - overhang (partially verifiable, pose-dependent)
# ---------------------------------------------------------------------------


def check_overhang(
    detections: Sequence[Detection],
    pose_result: Optional[PoseResult],
    sop_config: SopConfig,
) -> SopCheck:
    """Rule 1: max box overhang past the pallet edge vs ``overhang_max_cm``.

    Measures how far the widest box extends past the pallet's left/right edges
    (in pixels), converts to centimetres via the pallet-width scale, and
    compares to the configured maximum. This rule is pose-dependent: when the
    pose is unavailable the check is ``invalidated`` with a reason (R15.4);
    otherwise the measurement drives a ``pass``/``fail`` against the config
    threshold and carries a raw-detector-score confidence (R16).
    """
    threshold = sop_config.overhang_max_cm

    # Pose-dependent invalidation (R15.4): with no reliable pose the pixel->cm
    # scaling in the pallet frame is not defensible, so the check is invalidated.
    invalidated = _pose_invalidated_check(1, pose_result, threshold)
    if invalidated is not None:
        return invalidated

    pallet = _pallet_detection(detections)
    boxes = _box_detections(detections)
    mpp = _metres_per_pixel(pallet) if pallet is not None else None

    if pallet is None or not boxes or mpp is None:
        return _make_check(
            1,
            status="unresolved",
            measurement=None,
            threshold_value=threshold.value,
            threshold_unit=threshold.unit,
            threshold_source=threshold.source,
        )

    px, _py, pw, _ph = pallet.bbox
    pallet_left = px
    pallet_right = px + pw
    max_overhang_px = 0.0
    for box in boxes:
        bx, _by, bw, _bh = box.bbox
        left_over = pallet_left - bx
        right_over = (bx + bw) - pallet_right
        max_overhang_px = max(max_overhang_px, left_over, right_over)

    overhang_cm = max(0.0, max_overhang_px) * mpp * 100.0
    status = _status_from_measurement(overhang_cm, threshold.value)
    return _make_check(
        1,
        status=status,
        measurement=overhang_cm,
        threshold_value=threshold.value,
        threshold_unit=threshold.unit,
        threshold_source=threshold.source,
        confidence=_detector_confidence([pallet, *boxes]),
        confidence_semantics=SEMANTICS_RAW_DETECTOR_SCORE,
    )


# ---------------------------------------------------------------------------
# Rule 2 - load height (verifiable)
# ---------------------------------------------------------------------------


def check_load_height(
    detections: Sequence[Detection],
    pose_result: Optional[PoseResult],
    sop_config: SopConfig,
) -> SopCheck:
    """Rule 2: load height above the floor vs ``load_height_max_m``.

    Measures the vertical extent from the bottom of the pallet detection to the
    top of the tallest box detection (in pixels), converts to metres via the
    pallet-width scale, and compares to the configured maximum height.
    """
    threshold = sop_config.load_height_max_m
    pallet = _pallet_detection(detections)
    boxes = _box_detections(detections)
    mpp = _metres_per_pixel(pallet) if pallet is not None else None

    if pallet is None or not boxes or mpp is None:
        return _make_check(
            2,
            status="unresolved",
            measurement=None,
            threshold_value=threshold.value,
            threshold_unit=threshold.unit,
            threshold_source=threshold.source,
        )

    _px, py, _pw, ph = pallet.bbox
    pallet_bottom = py + ph  # larger v is lower in the image
    # Top of the tallest box = smallest v (highest in image).
    load_top = min(by for _bx, by, _bw, _bh in (b.bbox for b in boxes))
    height_px = max(0.0, pallet_bottom - load_top)
    height_m = height_px * mpp
    status = _status_from_measurement(height_m, threshold.value)
    return _make_check(
        2,
        status=status,
        measurement=height_m,
        threshold_value=threshold.value,
        threshold_unit=threshold.unit,
        threshold_source=threshold.source,
        confidence=_detector_confidence([pallet, *boxes]),
        confidence_semantics=SEMANTICS_RAW_DETECTOR_SCORE,
    )


# ---------------------------------------------------------------------------
# Rule 3 - column tilt (partially verifiable, pose-dependent)
# ---------------------------------------------------------------------------


def check_column_tilt(
    detections: Sequence[Detection],
    pose_result: Optional[PoseResult],
    sop_config: SopConfig,
) -> SopCheck:
    """Rule 3: max box rotation relative to the pallet axes vs ``column_tilt_max_deg``.

    Estimates each box's rotation relative to the pallet's orientation. When a
    pose is available its ``orientation_deg`` defines the pallet axis; otherwise
    the axis-aligned pallet bbox (0 deg) is used as the reference. Box
    orientation is estimated from the first two of its keypoints when present.
    The maximum absolute tilt is compared to the configured maximum.
    """
    threshold = sop_config.column_tilt_max_deg

    # Pose-dependent invalidation (R15.4): the pallet axes come from the pose,
    # so without a reliable pose the tilt cannot be defensibly measured.
    invalidated = _pose_invalidated_check(3, pose_result, threshold)
    if invalidated is not None:
        return invalidated

    pallet = _pallet_detection(detections)
    boxes = _box_detections(detections)

    box_angles = [_box_orientation_deg(box) for box in boxes]
    measurable = [a for a in box_angles if a is not None]
    if pallet is None or not measurable:
        return _make_check(
            3,
            status="unresolved",
            measurement=None,
            threshold_value=threshold.value,
            threshold_unit=threshold.unit,
            threshold_source=threshold.source,
        )

    pallet_axis_deg = 0.0
    if _pose_available(pose_result) and pose_result.orientation_deg is not None:
        pallet_axis_deg = pose_result.orientation_deg

    max_tilt_deg = max(
        abs(_wrap_deg(angle - pallet_axis_deg)) for angle in measurable
    )
    status = _status_from_measurement(max_tilt_deg, threshold.value)
    return _make_check(
        3,
        status=status,
        measurement=max_tilt_deg,
        threshold_value=threshold.value,
        threshold_unit=threshold.unit,
        threshold_source=threshold.source,
        confidence=_detector_confidence([pallet, *boxes]),
        confidence_semantics=SEMANTICS_RAW_DETECTOR_SCORE,
    )


def _box_orientation_deg(box: Detection) -> Optional[float]:
    """Estimate a box's in-image orientation from its first two keypoints.

    Returns the angle (deg) of the vector between the first two keypoints that
    both carry pixel coordinates, or ``None`` when fewer than two are usable.
    """
    pts = [
        (kp.u, kp.v)
        for kp in box.keypoints
        if kp.u is not None and kp.v is not None
    ]
    if len(pts) < 2:
        return None
    (u0, v0), (u1, v1) = pts[0], pts[1]
    return math.degrees(math.atan2(v1 - v0, u1 - u0))


def _wrap_deg(angle_deg: float) -> float:
    """Wrap an angle to (-180, 180]."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        wrapped = 180.0
    return wrapped


# ---------------------------------------------------------------------------
# Rule 5 - stretch-wrap presence (verifiable, threshold-free heuristic)
# ---------------------------------------------------------------------------


def check_wrap_presence(
    detections: Sequence[Detection],
    pose_result: Optional[PoseResult],
    sop_config: SopConfig,
) -> SopCheck:
    """Rule 5: presence of stretch-wrap on the observed faces (heuristic).

    This is a presence/appearance check with no numeric config threshold. As a
    testable proxy for the current stub detector, presence is asserted when a
    pallet detection exists (a real wrap classifier arrives with the model).
    The check is threshold-free: ``threshold_used``/``threshold_source`` are
    left null and the confidence is reported as ``heuristic_evidence_quality``
    (R16.2) - not a detector class score and not a calibrated pass probability.
    """
    pallet = _pallet_detection(detections)
    if pallet is None:
        return _make_check(
            5,
            status="unresolved",
            measurement=None,
            threshold_value=None,
            threshold_unit=None,
            threshold_source=None,
        )
    # Presence proxy: a detected pallet load is treated as wrapped-present here.
    # This is an appearance/heuristic check, so its confidence is a HEURISTIC
    # EVIDENCE-QUALITY score (the pallet detection score as a proxy for how
    # strong the visual evidence is) - explicitly NOT a raw detector class
    # score for a wrap classifier and NOT a calibrated pass probability
    # (R16.2/R16.3/R16.4). A real wrap classifier arrives with the model.
    return _make_check(
        5,
        status="pass",
        measurement=None,
        threshold_value=None,
        threshold_unit=None,
        threshold_source=None,
        confidence=pallet.score,
        confidence_semantics=SEMANTICS_HEURISTIC_EVIDENCE_QUALITY,
    )


# ---------------------------------------------------------------------------
# Rule 7 - centroid offset (partially verifiable, pose-dependent)
# ---------------------------------------------------------------------------


def check_centroid_offset(
    detections: Sequence[Detection],
    pose_result: Optional[PoseResult],
    sop_config: SopConfig,
) -> SopCheck:
    """Rule 7: load-centroid lateral offset from the pallet centre vs ``centroid_offset_max_cm``.

    Measures the horizontal distance between the centroid of the box
    detections (the visible load mass proxy) and the pallet-detection centre,
    converts to centimetres via the pallet-width scale, and compares to the
    configured maximum offset.
    """
    threshold = sop_config.centroid_offset_max_cm

    # Pose-dependent invalidation (R15.4): locating the pallet centre in the
    # floor frame needs the pose, so without it the offset is invalidated.
    invalidated = _pose_invalidated_check(7, pose_result, threshold)
    if invalidated is not None:
        return invalidated

    pallet = _pallet_detection(detections)
    boxes = _box_detections(detections)
    mpp = _metres_per_pixel(pallet) if pallet is not None else None

    if pallet is None or not boxes or mpp is None:
        return _make_check(
            7,
            status="unresolved",
            measurement=None,
            threshold_value=threshold.value,
            threshold_unit=threshold.unit,
            threshold_source=threshold.source,
        )

    px, _py, pw, _ph = pallet.bbox
    pallet_centre_u = px + pw / 2.0
    box_centres = [bx + bw / 2.0 for bx, _by, bw, _bh in (b.bbox for b in boxes)]
    load_centroid_u = sum(box_centres) / len(box_centres)
    offset_cm = abs(load_centroid_u - pallet_centre_u) * mpp * 100.0
    status = _status_from_measurement(offset_cm, threshold.value)
    return _make_check(
        7,
        status=status,
        measurement=offset_cm,
        threshold_value=threshold.value,
        threshold_unit=threshold.unit,
        threshold_source=threshold.source,
        confidence=_detector_confidence([pallet, *boxes]),
        confidence_semantics=SEMANTICS_RAW_DETECTOR_SCORE,
    )


# ---------------------------------------------------------------------------
# analyze() - assemble all eight checks (verifiable subset implemented here)
# ---------------------------------------------------------------------------

#: The verifiable-subset rule ids implemented in this task.
_IMPLEMENTED_CHECKS = {
    1: check_overhang,
    2: check_load_height,
    3: check_column_tilt,
    5: check_wrap_presence,
    7: check_centroid_offset,
}


def analyze(
    detections: Sequence[Detection],
    pose_result: Optional[PoseResult],
    sop_config: SopConfig,
) -> list[SopCheck]:
    """Run the verifiable-subset checks and return all eight SOP checks.

    The implemented verifiable subset (rules 1, 2, 3, 5, 7) is measured from
    the detections/pose and compared against the config thresholds (R15.1/R15.2,
    R18.1/R18.2), each carrying an honest ``confidence``/``confidence_semantics``
    (R16). Pose-dependent checks (1, 3, 7) are ``invalidated`` with a reason
    when the pose is unavailable (R15.4). The remaining rules (4, 6, 8) are
    ``not_verifiable`` and are ALWAYS ``unresolved`` (R15.3) so the full
    eight-rule set is present (Property 14 shape).

    Parameters
    ----------
    detections:
        The pallet/box detections for one pallet.
    pose_result:
        The pallet's pose (may be ``None`` or unavailable).
    sop_config:
        The loaded SOP thresholds (source of every ``threshold_used``).

    Returns
    -------
    list[SopCheck]
        Exactly eight checks, rule_id 1..8 in ascending order.
    """
    checks: list[SopCheck] = []
    for rule_id in RULE_IDS:
        fn = _IMPLEMENTED_CHECKS.get(rule_id)
        if fn is not None:
            checks.append(fn(detections, pose_result, sop_config))
        else:
            # not_verifiable rule (4/6/8): ALWAYS unresolved (R15.3) - never
            # pass/fail/omitted, regardless of inputs. This is the *honest*
            # not-verifiable case (the single-side view fundamentally cannot
            # decide the rule), NOT a processing failure, so we do NOT emit
            # ReasonCode.PROCESSING_FAILED (which would imply a runtime error).
            # There is no NOT_VERIFIABLE reason code in the enum; the triage
            # classification + hidden_regions (attached by _make_check) already
            # document *why* the rule is unresolved, so reason_code stays None.
            # No calibrated confidence is claimed for an undecidable rule.
            checks.append(
                _make_check(
                    rule_id,
                    status="unresolved",
                    measurement=None,
                    threshold_value=None,
                    threshold_unit=None,
                    threshold_source=None,
                    confidence=None,
                    confidence_semantics=None,
                    provenance=ProvenanceLabel.UNAVAILABLE,
                    reason_code=None,
                )
            )
    return checks
