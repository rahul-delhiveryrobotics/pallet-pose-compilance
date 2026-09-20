"""Translation/rotation error distributions and tolerance evaluation (task 7.2, R11).

Given the self-constructed pose evaluation samples from
:mod:`~pallet_pose_compliance.geometry.pose_eval` (each a
:class:`~pallet_pose_compliance.geometry.pose_eval.PoseEvalSample` pairing a
``SIMULATED`` :class:`GroundTruthPose` with the estimator's
:class:`~pallet_pose_compliance.output.schema.PoseResult`), this module reports:

- **Translation error**, *separately* from rotation (R11.1): per-axis
  ``dx = pred.x - gt.x`` and ``dy = pred.y - gt.y`` and the radial error
  ``sqrt(dx^2 + dy^2)``, in metres (a centimetre view is also exposed).
- **Rotation error**, *separately* from translation (R11.2): the orientation
  error in degrees via
  :func:`~pallet_pose_compliance.geometry.pose_eval.orientation_error_deg`,
  which folds the pallet's symmetry / 180-degree long-axis ambiguity (R9.7).

Each is reported as a
:class:`~pallet_pose_compliance.detection.metrics.DistributionSummary`, reusing
the one distribution-summary primitive so a ``sample_count`` is *always* present
(R11.3, Property 4). Only *available* predictions with a position/orientation
contribute an error value; unavailable predictions are counted and surfaced as a
**coverage** figure (available vs total) so they are never silently dropped in a
misleading way.

Tolerance evaluation (R11.4, R11.5)
-----------------------------------
The measured error distributions are evaluated against the ``Pose_Tolerance`` of
**±2 cm** position and **±3°** orientation. We report pass-fractions:

- **per-axis** translation (``|dx| <= 2 cm``, ``|dy| <= 2 cm``),
- **radial** translation (``sqrt(dx^2 + dy^2) <= 2 cm``),
- **rotation** (``orientation_error <= 3°``), and
- a **combined** pass-fraction (radial *and* rotation within tolerance).

These are returned as an evaluation-result structure. The result makes clear,
in its labels and provenance, that the tolerance is a **target being evaluated
against**, not a guaranteed accuracy level (R11.5): ``tolerance_semantics`` is
fixed to ``"evaluation_target_not_guarantee"`` and the whole evaluation carries
``SIMULATED`` provenance (the ground truth is simulated). Nothing here upgrades
the result to ``measured`` or claims the system *meets* the tolerance — it only
reports the measured-against-target fraction.

Tolerance thresholds are documented module constants and can be overridden per
call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..detection.metrics import DistributionSummary
from ..output.provenance import ProvenanceLabel
from .pose_eval import PoseEvalSample, evaluate_samples, orientation_error_deg

__all__ = [
    "POSITION_TOLERANCE_M",
    "ORIENTATION_TOLERANCE_DEG",
    "TOLERANCE_SEMANTICS",
    "ToleranceEvaluation",
    "PoseErrorReport",
    "compute_pose_error_report",
    "evaluate_pose_error",
]

#: Position tolerance of the ``Pose_Tolerance`` bar: ±2 cm expressed in metres.
#: A *target to evaluate against*, never a guaranteed accuracy level (R11.5).
POSITION_TOLERANCE_M = 0.02

#: Orientation tolerance of the ``Pose_Tolerance`` bar: ±3 degrees. A *target to
#: evaluate against*, never a guaranteed accuracy level (R11.5).
ORIENTATION_TOLERANCE_DEG = 3.0

#: Fixed label recorded on every tolerance evaluation so the report can never be
#: read as a guarantee: the tolerance is a target the system measures itself
#: against and reports against (R11.5).
TOLERANCE_SEMANTICS = "evaluation_target_not_guarantee"


# ---------------------------------------------------------------------------
# Tolerance evaluation result (R11.4, R11.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToleranceEvaluation:
    """Pass-fractions of measured error against the ±2 cm / ±3° target (R11.4).

    Every field is a *measured-against-target* fraction, not a guarantee. The
    :attr:`tolerance_semantics` label and the ``SIMULATED`` provenance of the
    enclosing :class:`PoseErrorReport` make this explicit (R11.5).

    Attributes
    ----------
    sample_count:
        Number of evaluated (available) prediction/GT pairs the fractions are
        computed over. Equals the translation/rotation distribution counts.
    position_tolerance_m:
        The position tolerance used (metres); default :data:`POSITION_TOLERANCE_M`.
    orientation_tolerance_deg:
        The orientation tolerance used (degrees); default
        :data:`ORIENTATION_TOLERANCE_DEG`.
    pass_fraction_dx / pass_fraction_dy:
        Per-axis translation pass-fractions (``|dx| <= tol`` / ``|dy| <= tol``).
    pass_fraction_radial:
        Radial translation pass-fraction (``sqrt(dx^2+dy^2) <= tol``).
    pass_fraction_orientation:
        Rotation pass-fraction (``orientation_error <= tol``).
    pass_fraction_combined:
        Combined pass-fraction (radial translation *and* rotation within
        tolerance for the same sample).
    tolerance_semantics:
        Always :data:`TOLERANCE_SEMANTICS` — the tolerance is a target, not a
        guarantee (R11.5).
    """

    sample_count: int
    position_tolerance_m: float
    orientation_tolerance_deg: float
    pass_fraction_dx: Optional[float]
    pass_fraction_dy: Optional[float]
    pass_fraction_radial: Optional[float]
    pass_fraction_orientation: Optional[float]
    pass_fraction_combined: Optional[float]
    tolerance_semantics: str = TOLERANCE_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "pass_fraction_dx": self.pass_fraction_dx,
            "pass_fraction_dy": self.pass_fraction_dy,
            "pass_fraction_radial": self.pass_fraction_radial,
            "pass_fraction_orientation": self.pass_fraction_orientation,
            "pass_fraction_combined": self.pass_fraction_combined,
            "tolerance_semantics": self.tolerance_semantics,
        }


# ---------------------------------------------------------------------------
# Full pose error report (R11.1, R11.2, R11.3, R11.4, R11.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseErrorReport:
    """Translation/rotation error distributions + tolerance evaluation (R11).

    Translation and rotation error are reported as *separate* distributions
    (R11.1, R11.2), each a :class:`DistributionSummary` carrying a
    ``sample_count`` (R11.3). Coverage records how many predictions were
    available versus the total number of samples, so unavailable poses are
    surfaced honestly rather than silently dropped.

    The whole report is ``SIMULATED``: the ground truth is a synthetic render,
    so this is an evaluation against a target, never a real-world guarantee that
    the system meets the tolerance (R11.5, R26.2).

    Attributes
    ----------
    error_dx_m / error_dy_m:
        Per-axis translation error distributions (metres).
    error_radial_m:
        Radial translation error distribution (metres).
    error_radial_cm:
        Radial translation error distribution (centimetres) — the same samples
        rescaled for the ±2 cm view.
    error_orientation_deg:
        Rotation error distribution (degrees), folded modulo pallet symmetry.
    tolerance:
        The :class:`ToleranceEvaluation` against ±2 cm / ±3°.
    total_samples:
        Total number of GT/prediction pairs supplied.
    available_samples:
        Number of pairs whose prediction was ``available`` and contributed an
        error value.
    unavailable_samples:
        ``total_samples - available_samples`` (pairs excluded from the error
        distributions, reported honestly, not dropped).
    coverage_fraction:
        ``available_samples / total_samples`` (``None`` when no samples).
    provenance:
        Always :attr:`ProvenanceLabel.SIMULATED` for a synthetic-GT evaluation.
    """

    error_dx_m: DistributionSummary
    error_dy_m: DistributionSummary
    error_radial_m: DistributionSummary
    error_radial_cm: DistributionSummary
    error_orientation_deg: DistributionSummary
    tolerance: ToleranceEvaluation
    total_samples: int
    available_samples: int
    unavailable_samples: int
    coverage_fraction: Optional[float]
    provenance: ProvenanceLabel = ProvenanceLabel.SIMULATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation_error": {
                "dx_m": self.error_dx_m.to_dict(),
                "dy_m": self.error_dy_m.to_dict(),
                "radial_m": self.error_radial_m.to_dict(),
                "radial_cm": self.error_radial_cm.to_dict(),
            },
            "rotation_error": {
                "orientation_deg": self.error_orientation_deg.to_dict(),
            },
            "tolerance_evaluation": self.tolerance.to_dict(),
            "coverage": {
                "total_samples": self.total_samples,
                "available_samples": self.available_samples,
                "unavailable_samples": self.unavailable_samples,
                "coverage_fraction": self.coverage_fraction,
            },
            "provenance": self.provenance.value,
        }


def _pass_fraction(
    values: Sequence[float], threshold: float
) -> Optional[float]:
    """Fraction of ``values`` whose magnitude is within ``threshold`` (inclusive).

    Returns ``None`` for an empty input (an honest empty result, never a
    fabricated ``1.0``/``0.0``). Values are compared by absolute magnitude so
    the same helper works for signed per-axis error and non-negative
    radial/rotation error.
    """
    n = len(values)
    if n == 0:
        return None
    within = sum(1 for v in values if abs(v) <= threshold)
    return within / n


def compute_pose_error_report(
    samples: Sequence[PoseEvalSample],
    *,
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    orientation_tolerance_deg: float = ORIENTATION_TOLERANCE_DEG,
) -> PoseErrorReport:
    """Build the translation/rotation error report from evaluation samples (R11).

    For each :class:`PoseEvalSample`, if the prediction is ``available`` with a
    position and orientation, its per-axis / radial translation error and
    orientation error (folded modulo the GT's symmetry) are collected. Samples
    whose prediction is ``unavailable`` (or missing metric fields) are *not*
    turned into fabricated zero errors — they are excluded from the error
    distributions and counted in the coverage figure instead (R11.3 honesty).

    Parameters
    ----------
    samples:
        The GT/prediction pairs from
        :func:`~pallet_pose_compliance.geometry.pose_eval.evaluate_samples`.
    position_tolerance_m:
        Position tolerance to evaluate against (metres); default ±2 cm.
    orientation_tolerance_deg:
        Orientation tolerance to evaluate against (degrees); default ±3°.

    Returns
    -------
    PoseErrorReport
        Separate translation/rotation distributions (with sample counts), a
        tolerance evaluation, and a coverage figure. Provenance ``SIMULATED``.
    """
    dx: list[float] = []
    dy: list[float] = []
    radial: list[float] = []
    orient_err: list[float] = []
    # Per-sample combined tolerance flags (radial AND rotation), collected only
    # for samples that contributed both a translation and a rotation error.
    combined_pass: list[float] = []

    total = len(samples)
    available = 0
    for s in samples:
        pred = s.prediction
        gt = s.ground_truth
        if (
            pred.pose_status != "available"
            or pred.position_m is None
            or pred.orientation_deg is None
        ):
            continue  # excluded honestly; counted via coverage below
        available += 1
        edx = float(pred.position_m.x) - float(gt.x_m)
        edy = float(pred.position_m.y) - float(gt.y_m)
        er = math.hypot(edx, edy)
        eo = orientation_error_deg(
            pred.orientation_deg,
            gt.theta_deg,
            symmetry_order=gt.symmetry_order,
        )
        dx.append(edx)
        dy.append(edy)
        radial.append(er)
        orient_err.append(eo)
        combined_pass.append(
            1.0
            if (er <= position_tolerance_m and eo <= orientation_tolerance_deg)
            else 0.0
        )

    unavailable = total - available
    coverage = (available / total) if total > 0 else None

    tolerance = ToleranceEvaluation(
        sample_count=available,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_deg=orientation_tolerance_deg,
        pass_fraction_dx=_pass_fraction(dx, position_tolerance_m),
        pass_fraction_dy=_pass_fraction(dy, position_tolerance_m),
        pass_fraction_radial=_pass_fraction(radial, position_tolerance_m),
        pass_fraction_orientation=_pass_fraction(
            orient_err, orientation_tolerance_deg
        ),
        # combined_pass entries are already 0/1 flags; the fraction of 1s is
        # their mean, which _pass_fraction(..., threshold=0.0) does NOT give
        # (0.0 is within threshold), so compute directly here.
        pass_fraction_combined=(
            (sum(combined_pass) / len(combined_pass)) if combined_pass else None
        ),
    )

    return PoseErrorReport(
        error_dx_m=DistributionSummary.from_values(
            "translation_error_dx", dx, unit="m"
        ),
        error_dy_m=DistributionSummary.from_values(
            "translation_error_dy", dy, unit="m"
        ),
        error_radial_m=DistributionSummary.from_values(
            "translation_error_radial", radial, unit="m"
        ),
        error_radial_cm=DistributionSummary.from_values(
            "translation_error_radial_cm", [r * 100.0 for r in radial], unit="cm"
        ),
        error_orientation_deg=DistributionSummary.from_values(
            "rotation_error", orient_err, unit="deg"
        ),
        tolerance=tolerance,
        total_samples=total,
        available_samples=available,
        unavailable_samples=unavailable,
        coverage_fraction=coverage,
    )


def evaluate_pose_error(
    n_samples: int,
    *,
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    orientation_tolerance_deg: float = ORIENTATION_TOLERANCE_DEG,
    **eval_kwargs: Any,
) -> PoseErrorReport:
    """Run the self-constructed evaluation and produce the error report (R11).

    Convenience top-level: runs
    :func:`~pallet_pose_compliance.geometry.pose_eval.evaluate_samples` to obtain
    ``n_samples`` SIMULATED GT/prediction pairs, then summarises them into a
    :class:`PoseErrorReport` (separate translation/rotation distributions with
    sample counts + tolerance evaluation + coverage). Since the ground truth is
    simulated, the report is ``SIMULATED`` and the tolerance is treated strictly
    as a target (R11.5).

    Parameters
    ----------
    n_samples:
        Number of simulated GT/prediction pairs to evaluate.
    position_tolerance_m, orientation_tolerance_deg:
        Tolerance thresholds to evaluate against (default ±2 cm / ±3°).
    **eval_kwargs:
        Forwarded to
        :func:`~pallet_pose_compliance.geometry.pose_eval.evaluate_samples`
        (e.g. ``calibration``, ``pallet_model``, ``seed``, ``with_uncertainty``,
        sampling ranges, ``symmetry_order``).

    Returns
    -------
    PoseErrorReport
        The error/tolerance report over the self-constructed evaluation.
    """
    samples = evaluate_samples(n_samples, **eval_kwargs)
    return compute_pose_error_report(
        samples,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_deg=orientation_tolerance_deg,
    )
