"""Pose-error sensitivity analysis and usable-envelope classification (task 7.4, R12).

The evaluator needs to know **when to trust the pose and when not to** (R12): how
pose error grows as the camera is mis-set (height error R12.1, tilt error R12.2),
at both **short** and **long** range, and — crucially — a *declared* usable
envelope (the conditions in which measured error meets the ±2 cm / ±3° tolerance,
R12.3) and the non-usable region (where it does not, R12.4). This module builds
that analysis on top of the self-constructed pose evaluation harness
(:mod:`~pallet_pose_compliance.geometry.pose_eval`) and the error/tolerance report
(:mod:`~pallet_pose_compliance.geometry.pose_error`).

How camera height / tilt *error* is modelled (design: sensitivity, R12.4)
-------------------------------------------------------------------------
There is no real camera, so a *mis-set* camera is modelled honestly with two
calibrations:

- The **true** calibration renders the ground truth (the pallet is where it
  actually is, projected through the real mounting).
- The **assumed** calibration — the one the estimator is handed — has its camera
  height (R12.1) or down-tilt (R12.2) perturbed by the *error amount*. This is
  exactly the operational failure being studied: the operator believes the
  camera sits at some height/tilt, but it is off by ``delta``. The estimator
  therefore solves in a slightly wrong world model, and the resulting pose error
  is the sensitivity we report.

Because both calibrations are synthetic, every result here is ``SIMULATED``
(R26): this is an evaluation against a target, never a real-world guarantee.

Range definitions (documented constants)
-----------------------------------------
Short vs long range is expressed via the ground-truth sampling depth
(``y_range`` in the Floor_Frame, i.e. distance from the camera):

- :data:`SHORT_RANGE_Y_M` ``= (2.5, 3.5)`` m — near the camera.
- :data:`LONG_RANGE_Y_M` ``= (6.0, 8.0)`` m — far from the camera.

These are constants so the "short/long range" claim is pinned and reproducible;
they are the *only* ranges evaluated (no extrapolation, R12.4).

Error grids (documented constants)
----------------------------------
- :data:`HEIGHT_ERROR_GRID_M` — camera height errors in metres (R12.1),
  including ``0.0`` (a correctly-set camera) so the grid contains a
  known-good baseline.
- :data:`TILT_ERROR_GRID_DEG` — camera tilt errors in degrees (R12.2), likewise
  including ``0.0``.

Per-condition classification (strict partition, R12.4)
------------------------------------------------------
Each evaluated condition is classified into **exactly one** of
:class:`EnvelopeLabel` ``{MEETS_TOLERANCE, FAILS_TOLERANCE, INSUFFICIENT_EVIDENCE}``:

1. **Insufficient evidence first.** If too few samples were *evaluated at all*
   at the condition (``total_samples < MIN_EVALUABLE_SAMPLES``), the condition
   is ``INSUFFICIENT_EVIDENCE`` — we never decide from too little data.
2. **Meets tolerance.** Otherwise, if the combined success-fraction over *all*
   evaluated samples (radial translation within ±2 cm *and* orientation within
   ±3°) is at least :data:`MEETS_PASS_FRACTION`, the condition
   ``MEETS_TOLERANCE``.
3. **Fails tolerance.** Otherwise the condition ``FAILS_TOLERANCE``.

The success-fraction is computed over *total* evaluated samples, not just the
available ones: a mis-set camera that makes the estimator *reject* the pose
(degrade-to-unavailable) is a condition where the pose is **not usable**, so a
rejected sample counts against the tolerance rather than being ignored. This is
why a large height/tilt error lands in ``FAILS_TOLERANCE`` (many rejected or
wrong poses) while a genuinely under-sampled condition — too few samples run at
all — is the only thing that lands in ``INSUFFICIENT_EVIDENCE``.

Steps 1-3 are mutually exclusive and exhaustive over evaluated conditions, so
the classification is a strict partition: every evaluated condition receives
exactly one label, and nothing unevaluated is ever inferred.

Usable-envelope declaration (R12.3, R12.4)
------------------------------------------
:class:`UsableEnvelope` aggregates the classified conditions into the declared
usable region (the ``MEETS_TOLERANCE`` conditions, R12.3), the non-usable region
(``FAILS_TOLERANCE``, R12.4), and a separately-listed ``INSUFFICIENT_EVIDENCE``
set. It carries **only conditions actually evaluated** — there is no
interpolation or extrapolation to unevaluated height/tilt/range combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

from ..calibration.calibration import Calibration, build_synthetic_calibration
from ..output.provenance import ProvenanceLabel
from ..output.schema import PalletModel
from .pallet_model import build_nominal_pallet_model
from .pose_error import (
    ORIENTATION_TOLERANCE_DEG,
    POSITION_TOLERANCE_M,
    PoseErrorReport,
    compute_pose_error_report,
)
from .pose_eval import (
    DEFAULT_EVAL_SEED,
    PoseEvalSample,
    generate_simulated_gt_samples,
    run_estimator_on_gt,
)

__all__ = [
    "SHORT_RANGE_Y_M",
    "LONG_RANGE_Y_M",
    "RANGE_DEFINITIONS",
    "HEIGHT_ERROR_GRID_M",
    "TILT_ERROR_GRID_DEG",
    "NOMINAL_CAMERA_HEIGHT_M",
    "NOMINAL_TILT_DOWN_DEG",
    "MEETS_PASS_FRACTION",
    "MIN_EVALUABLE_SAMPLES",
    "DEFAULT_SAMPLES_PER_CONDITION",
    "EnvelopeLabel",
    "SensitivityCondition",
    "SensitivityReport",
    "UsableEnvelope",
    "combined_success_fraction_over_total",
    "classify_condition",
    "run_sensitivity_analysis",
    "declare_usable_envelope",
]


# ---------------------------------------------------------------------------
# Documented range definitions (short vs long), expressed as GT sampling depth
# ---------------------------------------------------------------------------

#: Short-range ground-truth depth window (Floor_Frame ``y``, metres): near the
#: camera. The pallet is sampled with ``y`` drawn from this window.
SHORT_RANGE_Y_M: tuple[float, float] = (2.5, 3.5)

#: Long-range ground-truth depth window (Floor_Frame ``y``, metres): far from
#: the camera.
LONG_RANGE_Y_M: tuple[float, float] = (6.0, 8.0)

#: Ordered mapping of range-name -> depth window. These are the *only* ranges
#: evaluated; the analysis never extrapolates outside them (R12.4).
RANGE_DEFINITIONS: dict[str, tuple[float, float]] = {
    "short": SHORT_RANGE_Y_M,
    "long": LONG_RANGE_Y_M,
}


# ---------------------------------------------------------------------------
# Documented error grids (camera height error R12.1, tilt error R12.2)
# ---------------------------------------------------------------------------

#: Camera height *error* grid in metres (R12.1). ``0.0`` is included so the grid
#: carries a correctly-set-camera baseline; positive values mean the assumed
#: camera height is above the true height.
HEIGHT_ERROR_GRID_M: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20)

#: Camera tilt *error* grid in degrees (R12.2). ``0.0`` is the correctly-set
#: baseline; positive values mean the assumed down-tilt exceeds the true tilt.
TILT_ERROR_GRID_DEG: tuple[float, ...] = (0.0, 0.5, 1.0, 3.0, 6.0)

#: Nominal (true) mounting the synthetic camera is built at, matching
#: ``build_synthetic_calibration`` defaults and the ~1.2 m / ~20 deg brief.
NOMINAL_CAMERA_HEIGHT_M: float = 1.2
NOMINAL_TILT_DOWN_DEG: float = 20.0


# ---------------------------------------------------------------------------
# Documented classification thresholds (strict partition, R12.4)
# ---------------------------------------------------------------------------

#: A condition ``MEETS_TOLERANCE`` when its combined (radial AND orientation)
#: pass-fraction is at least this value. Documented high bar: the great majority
#: of samples must be within ±2 cm / ±3°.
MEETS_PASS_FRACTION: float = 0.95

#: Minimum number of samples that must be *evaluated at all* at a condition to
#: decide it. Below this the condition is ``INSUFFICIENT_EVIDENCE`` — we do not
#: classify a condition we barely sampled.
MIN_EVALUABLE_SAMPLES: int = 8

#: Default number of GT samples generated per evaluated condition.
DEFAULT_SAMPLES_PER_CONDITION: int = 24


# ---------------------------------------------------------------------------
# Classification label (exactly one per evaluated condition, R12.4)
# ---------------------------------------------------------------------------


class EnvelopeLabel(str, Enum):
    """The strict-partition classification of an evaluated condition (R12.4).

    Exactly one of these is assigned to every evaluated condition:

    - ``MEETS_TOLERANCE``: adequate evidence the condition meets ±2 cm / ±3°.
    - ``FAILS_TOLERANCE``: adequate evidence the condition does *not* meet it.
    - ``INSUFFICIENT_EVIDENCE``: too little available data to decide (never a
      guess).
    """

    MEETS_TOLERANCE = "meets_tolerance"
    FAILS_TOLERANCE = "fails_tolerance"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ---------------------------------------------------------------------------
# Per-condition result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityCondition:
    """One evaluated (error-type, error-amount, range) condition and its result.

    Attributes
    ----------
    error_type:
        ``"camera_height"`` (R12.1) or ``"camera_tilt"`` (R12.2).
    error_amount:
        The perturbation applied to the assumed calibration (metres for height,
        degrees for tilt). ``0.0`` is the correctly-set baseline.
    error_unit:
        ``"m"`` for height error, ``"deg"`` for tilt error.
    range_name:
        ``"short"`` or ``"long"``.
    range_y_m:
        The depth window (Floor_Frame ``y``, metres) used for this range.
    label:
        The :class:`EnvelopeLabel` assigned by :func:`classify_condition`.
    error_report:
        The full :class:`PoseErrorReport` for the condition (separate
        translation/rotation distributions, tolerance evaluation, coverage).
    provenance:
        Always :attr:`ProvenanceLabel.SIMULATED` (synthetic calibrations).
    """

    error_type: str
    error_amount: float
    error_unit: str
    range_name: str
    range_y_m: tuple[float, float]
    label: EnvelopeLabel
    error_report: PoseErrorReport
    provenance: ProvenanceLabel = ProvenanceLabel.SIMULATED

    @property
    def key(self) -> tuple[str, float, str]:
        """A hashable identity for the evaluated condition (no extrapolation)."""
        return (self.error_type, float(self.error_amount), self.range_name)

    def to_dict(self) -> dict[str, Any]:
        tol = self.error_report.tolerance
        return {
            "error_type": self.error_type,
            "error_amount": self.error_amount,
            "error_unit": self.error_unit,
            "range_name": self.range_name,
            "range_y_m": list(self.range_y_m),
            "label": self.label.value,
            "pass_fraction_combined": tol.pass_fraction_combined,
            "pass_fraction_radial": tol.pass_fraction_radial,
            "pass_fraction_orientation": tol.pass_fraction_orientation,
            "radial_error_m": {
                "median": self.error_report.error_radial_m.median,
                "p95": self.error_report.error_radial_m.p95,
                "max": self.error_report.error_radial_m.max,
            },
            "orientation_error_deg": {
                "median": self.error_report.error_orientation_deg.median,
                "p95": self.error_report.error_orientation_deg.p95,
                "max": self.error_report.error_orientation_deg.max,
            },
            "coverage_fraction": self.error_report.coverage_fraction,
            "available_samples": self.error_report.available_samples,
            "total_samples": self.error_report.total_samples,
            "provenance": self.provenance.value,
        }


# ---------------------------------------------------------------------------
# Classification (strict partition into exactly one label, R12.4)
# ---------------------------------------------------------------------------


def combined_success_fraction_over_total(report: PoseErrorReport) -> Optional[float]:
    """Fraction of *all* evaluated samples that met both tolerances.

    The tolerance evaluation's ``pass_fraction_combined`` is computed over the
    *available* predictions only. Here we spread it over the *total* evaluated
    samples so that predictions the estimator rejected (degrade-to-unavailable)
    count as tolerance failures rather than being ignored: a rejected pose is
    not a usable pose meeting ±2 cm / ±3°.

    Returns ``None`` when no samples were evaluated (honest empty, never a
    fabricated fraction).
    """
    total = report.total_samples
    if total <= 0:
        return None
    combined_available = report.tolerance.pass_fraction_combined
    if combined_available is None:
        # No available predictions contributed; every sample failed/was rejected.
        return 0.0
    # available * fraction = number that passed both; divide by total.
    n_pass = combined_available * report.available_samples
    return n_pass / total


def classify_condition(
    report: PoseErrorReport,
    *,
    meets_pass_fraction: float = MEETS_PASS_FRACTION,
    min_evaluable_samples: int = MIN_EVALUABLE_SAMPLES,
) -> EnvelopeLabel:
    """Classify one condition's error report into exactly one :class:`EnvelopeLabel`.

    The rule is a strict, ordered partition (R12.4):

    1. **Insufficient evidence** if too few samples were evaluated at all
       (``total_samples < min_evaluable_samples``). We never classify a
       condition we barely sampled.
    2. **Meets tolerance** if the combined success-fraction over *all* evaluated
       samples (radial translation AND orientation within tolerance, rejected
       poses counting as failures) is ``>= meets_pass_fraction``.
    3. **Fails tolerance** otherwise.

    Because the branches are ordered and exhaustive, exactly one label is
    returned for any report — no condition is left unlabelled or double-labelled.

    Parameters
    ----------
    report:
        The condition's :class:`PoseErrorReport`.
    meets_pass_fraction:
        Documented high threshold for ``MEETS_TOLERANCE``.
    min_evaluable_samples:
        Documented minimum number of evaluated samples required to decide.

    Returns
    -------
    EnvelopeLabel
        Exactly one of the three labels.
    """
    if report.total_samples < min_evaluable_samples:
        return EnvelopeLabel.INSUFFICIENT_EVIDENCE

    success = combined_success_fraction_over_total(report)
    if success is not None and success >= meets_pass_fraction:
        return EnvelopeLabel.MEETS_TOLERANCE
    return EnvelopeLabel.FAILS_TOLERANCE


# ---------------------------------------------------------------------------
# Sensitivity analysis: build one condition, then the whole grid
# ---------------------------------------------------------------------------


def _perturbed_calibrations(
    error_type: str,
    error_amount: float,
    *,
    calibration_id_prefix: str,
    image_size: tuple[int, int],
    horizontal_fov_deg: float,
) -> tuple[Calibration, Calibration]:
    """Return ``(true_calibration, assumed_calibration)`` for a mis-set camera.

    The true calibration is built at the nominal mounting; the assumed
    calibration perturbs *only* the height (``camera_height`` error, R12.1) or
    the tilt (``camera_tilt`` error, R12.2) by ``error_amount``. Feeding GT
    rendered with the true calibration to an estimator handed the assumed
    calibration models the operational error under study.
    """
    true_cal = build_synthetic_calibration(
        f"{calibration_id_prefix}-true",
        image_size=image_size,
        horizontal_fov_deg=horizontal_fov_deg,
        camera_height_m=NOMINAL_CAMERA_HEIGHT_M,
        tilt_down_deg=NOMINAL_TILT_DOWN_DEG,
    )
    if error_type == "camera_height":
        assumed_height = NOMINAL_CAMERA_HEIGHT_M + error_amount
        assumed_tilt = NOMINAL_TILT_DOWN_DEG
    elif error_type == "camera_tilt":
        assumed_height = NOMINAL_CAMERA_HEIGHT_M
        assumed_tilt = NOMINAL_TILT_DOWN_DEG + error_amount
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown error_type {error_type!r}")

    assumed_cal = build_synthetic_calibration(
        f"{calibration_id_prefix}-assumed",
        image_size=image_size,
        horizontal_fov_deg=horizontal_fov_deg,
        camera_height_m=assumed_height,
        tilt_down_deg=assumed_tilt,
    )
    return true_cal, assumed_cal


def _evaluate_condition_samples(
    true_cal: Calibration,
    assumed_cal: Calibration,
    pallet_model: PalletModel,
    *,
    n_samples: int,
    range_y_m: tuple[float, float],
    seed: int,
) -> list[PoseEvalSample]:
    """Render GT with ``true_cal`` and estimate with ``assumed_cal``.

    ``evaluate_samples`` generates the GT and runs the estimator with the *same*
    calibration it is passed, so to model a mis-set camera we generate the GT
    using ``true_cal`` and then re-run the estimator against ``assumed_cal`` on
    those very keypoints, re-pairing the results.
    """
    ground_truths = generate_simulated_gt_samples(
        n_samples,
        calibration=true_cal,
        pallet_model=pallet_model,
        seed=seed,
        y_range=range_y_m,
    )
    samples: list[PoseEvalSample] = []
    for gt in ground_truths:
        prediction = run_estimator_on_gt(gt, assumed_cal, pallet_model)
        samples.append(PoseEvalSample(ground_truth=gt, prediction=prediction))
    return samples


@dataclass(frozen=True)
class SensitivityReport:
    """The full sensitivity analysis over height/tilt error x short/long range.

    Attributes
    ----------
    conditions:
        Every evaluated :class:`SensitivityCondition`. These are the *only*
        conditions the report knows about (no extrapolation, R12.4).
    height_error_grid_m / tilt_error_grid_deg:
        The documented error grids evaluated (R12.1, R12.2).
    range_definitions:
        The documented short/long depth windows evaluated.
    position_tolerance_m / orientation_tolerance_deg:
        The ±2 cm / ±3° bar the conditions were classified against.
    provenance:
        Always :attr:`ProvenanceLabel.SIMULATED`.
    """

    conditions: tuple[SensitivityCondition, ...]
    height_error_grid_m: tuple[float, ...] = HEIGHT_ERROR_GRID_M
    tilt_error_grid_deg: tuple[float, ...] = TILT_ERROR_GRID_DEG
    range_definitions: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(RANGE_DEFINITIONS)
    )
    position_tolerance_m: float = POSITION_TOLERANCE_M
    orientation_tolerance_deg: float = ORIENTATION_TOLERANCE_DEG
    provenance: ProvenanceLabel = ProvenanceLabel.SIMULATED

    def conditions_for(
        self, error_type: str, range_name: str
    ) -> list[SensitivityCondition]:
        """Return evaluated conditions of an error type at a range, by amount."""
        matched = [
            c
            for c in self.conditions
            if c.error_type == error_type and c.range_name == range_name
        ]
        return sorted(matched, key=lambda c: c.error_amount)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": [c.to_dict() for c in self.conditions],
            "grids": {
                "height_error_grid_m": list(self.height_error_grid_m),
                "tilt_error_grid_deg": list(self.tilt_error_grid_deg),
            },
            "range_definitions": {
                k: list(v) for k, v in self.range_definitions.items()
            },
            "tolerance": {
                "position_tolerance_m": self.position_tolerance_m,
                "orientation_tolerance_deg": self.orientation_tolerance_deg,
            },
            "provenance": self.provenance.value,
            "note": (
                "Only the conditions listed here were evaluated; the report does "
                "not extrapolate to unevaluated height/tilt/range combinations "
                "(R12.4)."
            ),
        }


def run_sensitivity_analysis(
    *,
    pallet_model: Optional[PalletModel] = None,
    height_error_grid_m: Sequence[float] = HEIGHT_ERROR_GRID_M,
    tilt_error_grid_deg: Sequence[float] = TILT_ERROR_GRID_DEG,
    range_definitions: Optional[dict[str, tuple[float, float]]] = None,
    n_samples: int = DEFAULT_SAMPLES_PER_CONDITION,
    seed: int = DEFAULT_EVAL_SEED,
    image_size: tuple[int, int] = (1280, 720),
    horizontal_fov_deg: float = 70.0,
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    orientation_tolerance_deg: float = ORIENTATION_TOLERANCE_DEG,
    meets_pass_fraction: float = MEETS_PASS_FRACTION,
    min_evaluable_samples: int = MIN_EVALUABLE_SAMPLES,
) -> SensitivityReport:
    """Run the pose-error sensitivity analysis over the documented grids (R12.1-R12.4).

    For every combination of {camera-height error (R12.1), camera-tilt error
    (R12.2)} x {short, long range}, this renders GT with the true calibration,
    estimates with a calibration perturbed by the error amount, computes the
    per-condition :class:`PoseErrorReport` (translation + rotation), and
    classifies the condition into exactly one :class:`EnvelopeLabel` (R12.4).

    Every condition is a genuinely evaluated point; nothing is extrapolated.

    Parameters
    ----------
    pallet_model:
        Pallet model to place. Defaults to the nominal model.
    height_error_grid_m, tilt_error_grid_deg:
        Documented error grids to sweep (R12.1, R12.2).
    range_definitions:
        Short/long depth windows. Defaults to :data:`RANGE_DEFINITIONS`.
    n_samples:
        GT samples generated per condition.
    seed:
        Base seed for reproducible GT generation (R28.3). A distinct per-
        condition seed is derived so different conditions are not identical
        draws while remaining reproducible.
    image_size, horizontal_fov_deg:
        Synthetic camera intrinsics parameters shared by both calibrations.
    position_tolerance_m, orientation_tolerance_deg:
        The tolerance bar to evaluate against (default ±2 cm / ±3°).
    meets_pass_fraction, min_evaluable_samples:
        Documented classification thresholds (see :func:`classify_condition`).

    Returns
    -------
    SensitivityReport
        All evaluated conditions with their per-condition error reports and
        classification labels. Provenance ``SIMULATED``.
    """
    model = pallet_model if pallet_model is not None else build_nominal_pallet_model()
    ranges = range_definitions if range_definitions is not None else dict(RANGE_DEFINITIONS)

    error_axes: list[tuple[str, str, Sequence[float]]] = [
        ("camera_height", "m", list(height_error_grid_m)),
        ("camera_tilt", "deg", list(tilt_error_grid_deg)),
    ]

    conditions: list[SensitivityCondition] = []
    condition_index = 0
    for error_type, unit, grid in error_axes:
        for amount in grid:
            for range_name, range_y in ranges.items():
                # Derive a distinct-but-reproducible seed per condition.
                cond_seed = seed + condition_index * 1000 + 1
                condition_index += 1

                true_cal, assumed_cal = _perturbed_calibrations(
                    error_type,
                    float(amount),
                    calibration_id_prefix=(
                        f"sens-{error_type}-{amount}-{range_name}"
                    ),
                    image_size=image_size,
                    horizontal_fov_deg=horizontal_fov_deg,
                )
                samples = _evaluate_condition_samples(
                    true_cal,
                    assumed_cal,
                    model,
                    n_samples=n_samples,
                    range_y_m=range_y,
                    seed=cond_seed,
                )
                report = compute_pose_error_report(
                    samples,
                    position_tolerance_m=position_tolerance_m,
                    orientation_tolerance_deg=orientation_tolerance_deg,
                )
                label = classify_condition(
                    report,
                    meets_pass_fraction=meets_pass_fraction,
                    min_evaluable_samples=min_evaluable_samples,
                )
                conditions.append(
                    SensitivityCondition(
                        error_type=error_type,
                        error_amount=float(amount),
                        error_unit=unit,
                        range_name=range_name,
                        range_y_m=tuple(range_y),
                        label=label,
                        error_report=report,
                    )
                )

    return SensitivityReport(
        conditions=tuple(conditions),
        height_error_grid_m=tuple(height_error_grid_m),
        tilt_error_grid_deg=tuple(tilt_error_grid_deg),
        range_definitions=dict(ranges),
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_deg=orientation_tolerance_deg,
    )


# ---------------------------------------------------------------------------
# Usable-envelope declaration (R12.3, R12.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsableEnvelope:
    """The declared usable / non-usable envelope, from evaluated conditions only.

    Attributes
    ----------
    usable:
        The declared usable envelope: conditions labelled ``MEETS_TOLERANCE``
        (measured error meets ±2 cm / ±3°, R12.3).
    non_usable:
        The declared non-usable region: conditions labelled ``FAILS_TOLERANCE``
        (measured error does not meet the tolerance, R12.4).
    insufficient_evidence:
        Conditions with too little available data to decide, listed separately
        so they are never silently folded into usable or non-usable.
    provenance:
        Always :attr:`ProvenanceLabel.SIMULATED`.

    The three lists partition exactly the evaluated conditions of the source
    :class:`SensitivityReport` — no condition appears twice and none is invented
    (no extrapolation, R12.4).
    """

    usable: tuple[SensitivityCondition, ...]
    non_usable: tuple[SensitivityCondition, ...]
    insufficient_evidence: tuple[SensitivityCondition, ...]
    provenance: ProvenanceLabel = ProvenanceLabel.SIMULATED

    @property
    def evaluated_condition_keys(self) -> set[tuple[str, float, str]]:
        """The set of evaluated-condition keys covered by this envelope."""
        return {
            c.key
            for group in (self.usable, self.non_usable, self.insufficient_evidence)
            for c in group
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable_envelope": [c.to_dict() for c in self.usable],
            "non_usable_region": [c.to_dict() for c in self.non_usable],
            "insufficient_evidence": [
                c.to_dict() for c in self.insufficient_evidence
            ],
            "provenance": self.provenance.value,
            "note": (
                "Envelope declared strictly from evaluated conditions; usable = "
                "conditions meeting the ±2 cm / ±3° tolerance (R12.3), "
                "non-usable = conditions failing it (R12.4). No extrapolation to "
                "unevaluated conditions."
            ),
        }


def declare_usable_envelope(report: SensitivityReport) -> UsableEnvelope:
    """Aggregate a :class:`SensitivityReport` into the declared envelope (R12.3/R12.4).

    Partitions the report's evaluated conditions by their :class:`EnvelopeLabel`
    into the usable envelope (``MEETS_TOLERANCE``), the non-usable region
    (``FAILS_TOLERANCE``), and the separately-listed ``INSUFFICIENT_EVIDENCE``
    set. Only evaluated conditions are included — nothing is extrapolated.

    Parameters
    ----------
    report:
        The sensitivity report to aggregate.

    Returns
    -------
    UsableEnvelope
        The declared usable / non-usable / insufficient-evidence partition.
    """
    usable = tuple(
        c for c in report.conditions if c.label is EnvelopeLabel.MEETS_TOLERANCE
    )
    non_usable = tuple(
        c for c in report.conditions if c.label is EnvelopeLabel.FAILS_TOLERANCE
    )
    insufficient = tuple(
        c
        for c in report.conditions
        if c.label is EnvelopeLabel.INSUFFICIENT_EVIDENCE
    )
    return UsableEnvelope(
        usable=usable,
        non_usable=non_usable,
        insufficient_evidence=insufficient,
    )
