"""Tests for translation/rotation error distributions + tolerance eval (task 7.2, R11).

These tests pin the honesty and correctness properties of the pose error
reporting module:

- Translation and rotation errors are reported as **separate** distributions,
  each carrying a correct ``sample_count`` (R11.1, R11.2, R11.3).
- On noise-free synthetic samples the errors are ~0 and all pass-fractions are
  ~1.0 (all within ±2 cm / ±3°) (R11.4).
- Pass-fraction computation is correct on hand-built errors — per-axis and
  radial, with a mix of within/over tolerance (R11.4).
- The tolerance is treated as a target: the result is labelled
  ``evaluation_target_not_guarantee`` and the report provenance is
  ``SIMULATED``, never a guarantee (R11.5).
- Coverage counts available vs total when some predictions are unavailable
  (unavailable poses are surfaced, not silently dropped) (R11.3).
"""

from __future__ import annotations

import math

import pytest

from pallet_pose_compliance.calibration.calibration import build_synthetic_calibration
from pallet_pose_compliance.geometry.pallet_model import build_nominal_pallet_model
from pallet_pose_compliance.geometry.pose_error import (
    ORIENTATION_TOLERANCE_DEG,
    POSITION_TOLERANCE_M,
    TOLERANCE_SEMANTICS,
    PoseErrorReport,
    compute_pose_error_report,
    evaluate_pose_error,
)
from pallet_pose_compliance.geometry.pose_eval import (
    GroundTruthPose,
    PoseEvalSample,
    place_and_project,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel, ReasonCode
from pallet_pose_compliance.output.schema import PositionM, PoseResult


@pytest.fixture
def calibration():
    return build_synthetic_calibration("pose-error-test-cal")


@pytest.fixture
def pallet_model():
    return build_nominal_pallet_model()


# ---------------------------------------------------------------------------
# Test helpers: build GT + prediction pairs with controlled errors
# ---------------------------------------------------------------------------


def _make_gt(calibration, pallet_model, x=0.0, y=3.0, theta=0.0, symmetry_order=1):
    keypoints, _ = place_and_project(calibration, pallet_model, x, y, theta)
    return GroundTruthPose(
        x_m=x,
        y_m=y,
        theta_deg=theta,
        keypoints=tuple(keypoints),
        symmetry_order=symmetry_order,
    )


def _available_prediction(x, y, theta):
    return PoseResult(
        pose_status="available",
        position_m=PositionM(x=x, y=y),
        orientation_deg=theta,
        provenance=ProvenanceLabel.SIMULATED,
    )


def _unavailable_prediction():
    return PoseResult(
        pose_status="unavailable",
        reason_code=ReasonCode.ILL_CONDITIONED,
        provenance=ProvenanceLabel.UNAVAILABLE,
    )


def _sample_with_error(calibration, pallet_model, *, gt_xytheta, pred_xytheta):
    gx, gy, gtheta = gt_xytheta
    px, py, ptheta = pred_xytheta
    gt = _make_gt(calibration, pallet_model, x=gx, y=gy, theta=gtheta)
    pred = _available_prediction(px, py, ptheta)
    return PoseEvalSample(ground_truth=gt, prediction=pred)


# ---------------------------------------------------------------------------
# Separate distributions carry correct sample counts (R11.1, R11.2, R11.3)
# ---------------------------------------------------------------------------


class TestSampleCounts:
    def test_translation_and_rotation_distributions_are_separate_with_counts(
        self, calibration, pallet_model
    ):
        samples = [
            _sample_with_error(
                calibration,
                pallet_model,
                gt_xytheta=(0.0, 3.0, 0.0),
                pred_xytheta=(0.005, 3.0, 1.0),
            ),
            _sample_with_error(
                calibration,
                pallet_model,
                gt_xytheta=(0.2, 3.5, 10.0),
                pred_xytheta=(0.21, 3.49, 11.0),
            ),
            _sample_with_error(
                calibration,
                pallet_model,
                gt_xytheta=(-0.3, 4.0, -20.0),
                pred_xytheta=(-0.3, 4.0, -20.0),
            ),
        ]
        report = compute_pose_error_report(samples)

        # Translation distributions (separate per-axis + radial), R11.1.
        assert report.error_dx_m.sample_count == 3
        assert report.error_dy_m.sample_count == 3
        assert report.error_radial_m.sample_count == 3
        assert report.error_radial_cm.sample_count == 3
        # Rotation distribution is separate, R11.2/R11.3.
        assert report.error_orientation_deg.sample_count == 3
        # sample_count equals len(values) — the distribution invariant.
        assert report.error_dx_m.sample_count == len(report.error_dx_m.values)
        assert report.error_orientation_deg.sample_count == len(
            report.error_orientation_deg.values
        )
        # Translation is metres; rotation is degrees — separately reported.
        assert report.error_radial_m.unit == "m"
        assert report.error_radial_cm.unit == "cm"
        assert report.error_orientation_deg.unit == "deg"

    def test_radial_error_matches_hand_computed(self, calibration, pallet_model):
        # dx = 0.03, dy = 0.04 -> radial = 0.05 m = 5 cm.
        sample = _sample_with_error(
            calibration,
            pallet_model,
            gt_xytheta=(0.0, 3.0, 0.0),
            pred_xytheta=(0.03, 3.04, 0.0),
        )
        report = compute_pose_error_report([sample])
        assert report.error_radial_m.values[0] == pytest.approx(0.05, abs=1e-9)
        assert report.error_radial_cm.values[0] == pytest.approx(5.0, abs=1e-6)
        assert report.error_dx_m.values[0] == pytest.approx(0.03, abs=1e-9)
        assert report.error_dy_m.values[0] == pytest.approx(0.04, abs=1e-9)


# ---------------------------------------------------------------------------
# Noise-free synthetic samples -> ~0 error, pass-fraction ~1.0 (R11.4)
# ---------------------------------------------------------------------------


class TestNoiseFreeSamples:
    def test_noise_free_errors_near_zero_and_all_pass(self, calibration, pallet_model):
        report = evaluate_pose_error(
            12, calibration=calibration, pallet_model=pallet_model
        )
        # Errors are essentially zero for a noise-free synthetic round-trip.
        assert report.error_radial_m.max == pytest.approx(0.0, abs=2e-2)
        assert report.error_orientation_deg.max == pytest.approx(0.0, abs=1e-1)
        # All within ±2 cm / ±3°: every pass-fraction is 1.0.
        assert report.tolerance.pass_fraction_dx == pytest.approx(1.0)
        assert report.tolerance.pass_fraction_dy == pytest.approx(1.0)
        assert report.tolerance.pass_fraction_radial == pytest.approx(1.0)
        assert report.tolerance.pass_fraction_orientation == pytest.approx(1.0)
        assert report.tolerance.pass_fraction_combined == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pass-fraction correctness on hand-built mixed errors (R11.4)
# ---------------------------------------------------------------------------


class TestPassFractionCorrectness:
    def test_mixed_within_and_over_tolerance(self, calibration, pallet_model):
        # Four samples with controlled errors relative to ±2 cm / ±3°:
        #  A: dx=+1cm dy=0    theta+1deg     -> dx pass, dy pass, radial pass, rot pass
        #  B: dx=0    dy=+3cm theta+0        -> dx pass, dy FAIL, radial FAIL, rot pass
        #  C: dx=+5cm dy=0    theta+0        -> dx FAIL, dy pass, radial FAIL, rot pass
        #  D: dx=0    dy=0    theta+10deg    -> dx pass, dy pass, radial pass, rot FAIL
        specs = [
            ((0.0, 3.0, 0.0), (0.01, 3.0, 1.0)),
            ((0.0, 3.0, 0.0), (0.0, 3.03, 0.0)),
            ((0.0, 3.0, 0.0), (0.05, 3.0, 0.0)),
            ((0.0, 3.0, 0.0), (0.0, 3.0, 10.0)),
        ]
        samples = [
            _sample_with_error(
                calibration, pallet_model, gt_xytheta=g, pred_xytheta=p
            )
            for g, p in specs
        ]
        report = compute_pose_error_report(samples)

        assert report.tolerance.sample_count == 4
        # dx within 2cm: A(1cm),B(0),D(0) pass; C(5cm) fails -> 3/4.
        assert report.tolerance.pass_fraction_dx == pytest.approx(3 / 4)
        # dy within 2cm: A(0),C(0),D(0) pass; B(3cm) fails -> 3/4.
        assert report.tolerance.pass_fraction_dy == pytest.approx(3 / 4)
        # radial within 2cm: A(~1cm),D(0) pass; B(3cm),C(5cm) fail -> 2/4.
        assert report.tolerance.pass_fraction_radial == pytest.approx(2 / 4)
        # rotation within 3deg: A,B,C pass; D(10deg) fails -> 3/4.
        assert report.tolerance.pass_fraction_orientation == pytest.approx(3 / 4)
        # combined (radial AND rotation): only A passes both -> 1/4.
        assert report.tolerance.pass_fraction_combined == pytest.approx(1 / 4)

    def test_boundary_is_inclusive(self, calibration, pallet_model):
        # Exactly at ±2 cm radial and ±3 deg counts as within tolerance.
        sample = _sample_with_error(
            calibration,
            pallet_model,
            gt_xytheta=(0.0, 3.0, 0.0),
            pred_xytheta=(POSITION_TOLERANCE_M, 3.0, ORIENTATION_TOLERANCE_DEG),
        )
        report = compute_pose_error_report([sample])
        assert report.tolerance.pass_fraction_radial == pytest.approx(1.0)
        assert report.tolerance.pass_fraction_orientation == pytest.approx(1.0)
        assert report.tolerance.pass_fraction_combined == pytest.approx(1.0)

    def test_override_tolerance_thresholds(self, calibration, pallet_model):
        # A 3 cm radial error fails the default 2 cm bar but passes a 4 cm bar.
        sample = _sample_with_error(
            calibration,
            pallet_model,
            gt_xytheta=(0.0, 3.0, 0.0),
            pred_xytheta=(0.03, 3.0, 0.0),
        )
        default = compute_pose_error_report([sample])
        assert default.tolerance.pass_fraction_radial == pytest.approx(0.0)

        relaxed = compute_pose_error_report(
            [sample], position_tolerance_m=0.04
        )
        assert relaxed.tolerance.pass_fraction_radial == pytest.approx(1.0)
        assert relaxed.tolerance.position_tolerance_m == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Tolerance treated as a target, not a guarantee (R11.5)
# ---------------------------------------------------------------------------


class TestToleranceIsTargetNotGuarantee:
    def test_result_labelled_as_evaluation_and_simulated(
        self, calibration, pallet_model
    ):
        report = evaluate_pose_error(
            5, calibration=calibration, pallet_model=pallet_model
        )
        assert isinstance(report, PoseErrorReport)
        # Provenance is SIMULATED (GT is a synthetic render), never MEASURED.
        assert report.provenance is ProvenanceLabel.SIMULATED
        # The tolerance semantics label makes the target-not-guarantee explicit.
        assert report.tolerance.tolerance_semantics == TOLERANCE_SEMANTICS
        assert report.tolerance.tolerance_semantics == "evaluation_target_not_guarantee"
        # Serialised form carries both signals for downstream JSON reporting.
        d = report.to_dict()
        assert d["provenance"] == "simulated"
        assert (
            d["tolerance_evaluation"]["tolerance_semantics"]
            == "evaluation_target_not_guarantee"
        )

    def test_default_tolerances_are_the_pose_tolerance_bar(self):
        assert POSITION_TOLERANCE_M == pytest.approx(0.02)
        assert ORIENTATION_TOLERANCE_DEG == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Coverage: available vs total when some predictions are unavailable (R11.3)
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_unavailable_predictions_excluded_and_counted(
        self, calibration, pallet_model
    ):
        gt_ok = _make_gt(calibration, pallet_model, x=0.0, y=3.0, theta=0.0)
        gt_bad = _make_gt(calibration, pallet_model, x=0.1, y=3.2, theta=5.0)
        samples = [
            PoseEvalSample(
                ground_truth=gt_ok, prediction=_available_prediction(0.0, 3.0, 0.0)
            ),
            PoseEvalSample(
                ground_truth=gt_bad, prediction=_unavailable_prediction()
            ),
            PoseEvalSample(
                ground_truth=gt_ok, prediction=_available_prediction(0.005, 3.0, 1.0)
            ),
        ]
        report = compute_pose_error_report(samples)

        assert report.total_samples == 3
        assert report.available_samples == 2
        assert report.unavailable_samples == 1
        assert report.coverage_fraction == pytest.approx(2 / 3)
        # Only the 2 available predictions contribute to the distributions.
        assert report.error_radial_m.sample_count == 2
        assert report.error_orientation_deg.sample_count == 2
        assert report.tolerance.sample_count == 2

    def test_all_unavailable_yields_empty_distributions_not_fabricated(
        self, calibration, pallet_model
    ):
        gt = _make_gt(calibration, pallet_model)
        samples = [
            PoseEvalSample(ground_truth=gt, prediction=_unavailable_prediction())
            for _ in range(3)
        ]
        report = compute_pose_error_report(samples)

        assert report.total_samples == 3
        assert report.available_samples == 0
        assert report.unavailable_samples == 3
        assert report.coverage_fraction == pytest.approx(0.0)
        # Empty distributions: sample_count 0, no fabricated zero stats.
        assert report.error_radial_m.sample_count == 0
        assert report.error_radial_m.mean is None
        # Pass-fractions are None (honest empty), never a fabricated 1.0/0.0.
        assert report.tolerance.pass_fraction_radial is None
        assert report.tolerance.pass_fraction_orientation is None
        assert report.tolerance.pass_fraction_combined is None
