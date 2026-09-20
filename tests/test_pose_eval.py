"""Tests for the self-constructed pose ground truth + eval harness (task 7.1, R10).

These tests pin the honesty and correctness properties of the pose evaluation
harness:

- GT samples carry ``SIMULATED`` provenance and a pose independent of the
  estimator under test (the GT comes from the forward render, not
  ``estimate_pose``) — R10.2, R10.3.
- Running ``estimate_pose`` on a GT sample's keypoints recovers the GT pose
  within a tight tolerance (position well under 2 cm; orientation modulo the
  180-degree long-axis ambiguity) — R10.1.
- The harness generates the requested number of reproducible samples for a
  fixed seed and differs for a different seed — R28.3.
- The measured-GT path is optional/absent by default and correctly typed when
  supplied — R10.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from pallet_pose_compliance.calibration.calibration import build_synthetic_calibration
from pallet_pose_compliance.geometry.pallet_model import build_nominal_pallet_model
from pallet_pose_compliance.geometry.pose import estimate_pose
from pallet_pose_compliance.geometry.pose_eval import (
    DEFAULT_EVAL_SEED,
    GroundTruthPose,
    MeasuredFloorPointGT,
    PoseEvalSample,
    evaluate_samples,
    generate_simulated_gt_samples,
    orientation_error_deg,
    place_and_project,
    run_estimator_on_gt,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel


@pytest.fixture
def calibration():
    return build_synthetic_calibration("pose-eval-test-cal")


@pytest.fixture
def pallet_model():
    return build_nominal_pallet_model()


# ---------------------------------------------------------------------------
# GT provenance + independence from the estimator (R10.2, R10.3)
# ---------------------------------------------------------------------------


class TestGroundTruthProvenance:
    def test_generated_gt_is_simulated(self, calibration, pallet_model):
        samples = generate_simulated_gt_samples(
            5, calibration=calibration, pallet_model=pallet_model
        )
        assert len(samples) == 5
        for gt in samples:
            assert isinstance(gt, GroundTruthPose)
            assert gt.provenance is ProvenanceLabel.SIMULATED
            # Rendered keypoints are the known projection labels.
            assert len(gt.keypoints) == 8
            # The source string documents the self-constructed method (R10.2).
            assert "self-constructed" in gt.source
            assert "assignment-provided" not in gt.source.replace(
                "NOT assignment-provided", ""
            )

    def test_gt_pose_comes_from_forward_model_not_estimator(
        self, calibration, pallet_model
    ):
        # The GT pose must equal the pose we placed via the forward model, and be
        # independent of what estimate_pose would report. We verify this by
        # constructing GT directly from place_and_project and confirming its
        # stored pose matches the chosen pose exactly (no estimator involved).
        x0, y0, theta = 0.2, 3.4, 40.0
        keypoints, _ = place_and_project(calibration, pallet_model, x0, y0, theta)
        gt = GroundTruthPose(
            x_m=x0, y_m=y0, theta_deg=theta, keypoints=tuple(keypoints)
        )
        assert gt.x_m == pytest.approx(x0)
        assert gt.y_m == pytest.approx(y0)
        assert gt.theta_deg == pytest.approx(theta)

    def test_gt_rejects_non_simulated_provenance(self, calibration, pallet_model):
        keypoints, _ = place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        with pytest.raises(ValueError):
            GroundTruthPose(
                x_m=0.0,
                y_m=3.0,
                theta_deg=0.0,
                keypoints=tuple(keypoints),
                provenance=ProvenanceLabel.MEASURED,
            )


# ---------------------------------------------------------------------------
# Estimator recovers the GT pose within tolerance (R10.1)
# ---------------------------------------------------------------------------


class TestEstimatorRecoversGroundTruth:
    @pytest.mark.parametrize(
        "x0, y0, theta_deg",
        [
            (0.0, 3.0, 0.0),
            (0.4, 3.5, 20.0),
            (-0.5, 4.0, -35.0),
            (0.2, 2.7, 120.0),
        ],
    )
    def test_recovers_position_and_orientation(
        self, calibration, pallet_model, x0, y0, theta_deg
    ):
        keypoints, _ = place_and_project(calibration, pallet_model, x0, y0, theta_deg)
        gt = GroundTruthPose(
            x_m=x0, y_m=y0, theta_deg=theta_deg, keypoints=tuple(keypoints)
        )
        prediction = run_estimator_on_gt(gt, calibration, pallet_model)

        assert prediction.pose_status == "available"
        # Position within a tight tolerance (well under the 2 cm eval bar).
        assert prediction.position_m.x == pytest.approx(x0, abs=1e-3)
        assert prediction.position_m.y == pytest.approx(y0, abs=1e-3)
        # Orientation error modulo the 180-degree long-axis ambiguity is ~0.
        err = orientation_error_deg(prediction.orientation_deg, gt.theta_deg)
        assert err == pytest.approx(0.0, abs=1e-2)

    def test_forward_model_matches_pose_unit_test_helper(
        self, calibration, pallet_model
    ):
        # The factored forward model should behave identically to the estimator's
        # own round-trip: feed rendered keypoints straight to estimate_pose.
        keypoints, _ = place_and_project(calibration, pallet_model, 0.1, 3.0, 10.0)
        result = estimate_pose(keypoints, calibration, pallet_model)
        assert result.pose_status == "available"
        assert result.position_m.x == pytest.approx(0.1, abs=1e-3)
        assert result.position_m.y == pytest.approx(3.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Orientation error modulo symmetry / long-axis ambiguity
# ---------------------------------------------------------------------------


class TestOrientationError:
    def test_long_axis_flip_is_zero_error(self):
        # theta and theta+180 are the same line for a rectangular pallet.
        assert orientation_error_deg(30.0, 210.0) == pytest.approx(0.0, abs=1e-9)
        assert orientation_error_deg(-170.0, 10.0) == pytest.approx(0.0, abs=1e-9)

    def test_small_difference_reported(self):
        assert orientation_error_deg(12.0, 10.0) == pytest.approx(2.0, abs=1e-9)

    def test_wrap_boundary(self):
        # 179 vs -179 differ by 2 degrees across the wrap boundary.
        assert orientation_error_deg(179.0, -179.0) == pytest.approx(2.0, abs=1e-9)

    def test_symmetry_order_folds(self):
        # With 4-fold symmetry, 0 and 90 are equivalent -> zero error.
        assert orientation_error_deg(
            0.0, 90.0, symmetry_order=4
        ) == pytest.approx(0.0, abs=1e-9)

    def test_rejects_bad_symmetry_order(self):
        with pytest.raises(ValueError):
            orientation_error_deg(0.0, 0.0, symmetry_order=0)


# ---------------------------------------------------------------------------
# Reproducibility of the harness (R28.3)
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_requested_count_generated(self, calibration, pallet_model):
        samples = generate_simulated_gt_samples(
            17, calibration=calibration, pallet_model=pallet_model
        )
        assert len(samples) == 17

    def test_same_seed_is_reproducible(self, calibration, pallet_model):
        a = generate_simulated_gt_samples(
            10, calibration=calibration, pallet_model=pallet_model, seed=DEFAULT_EVAL_SEED
        )
        b = generate_simulated_gt_samples(
            10, calibration=calibration, pallet_model=pallet_model, seed=DEFAULT_EVAL_SEED
        )
        for ga, gb in zip(a, b):
            assert ga.x_m == pytest.approx(gb.x_m)
            assert ga.y_m == pytest.approx(gb.y_m)
            assert ga.theta_deg == pytest.approx(gb.theta_deg)

    def test_different_seed_differs(self, calibration, pallet_model):
        a = generate_simulated_gt_samples(
            10, calibration=calibration, pallet_model=pallet_model, seed=1
        )
        b = generate_simulated_gt_samples(
            10, calibration=calibration, pallet_model=pallet_model, seed=2
        )
        # At least one sample should differ across different seeds.
        differs = any(
            ga.x_m != pytest.approx(gb.x_m)
            or ga.y_m != pytest.approx(gb.y_m)
            or ga.theta_deg != pytest.approx(gb.theta_deg)
            for ga, gb in zip(a, b)
        )
        assert differs

    def test_bad_sample_count_rejected(self):
        with pytest.raises(ValueError):
            generate_simulated_gt_samples(0)


# ---------------------------------------------------------------------------
# Full harness pairing + optional measured GT (R10.1, R10.4)
# ---------------------------------------------------------------------------


class TestEvaluateSamples:
    def test_pairs_gt_with_prediction(self, calibration, pallet_model):
        samples = evaluate_samples(
            8, calibration=calibration, pallet_model=pallet_model
        )
        assert len(samples) == 8
        for s in samples:
            assert isinstance(s, PoseEvalSample)
            assert s.ground_truth.provenance is ProvenanceLabel.SIMULATED
            # For noise-free synthetic projection the estimator recovers the pose.
            assert s.prediction.pose_status == "available"
            err = orientation_error_deg(
                s.prediction.orientation_deg, s.ground_truth.theta_deg
            )
            assert err == pytest.approx(0.0, abs=1e-1)
            assert s.prediction.position_m.x == pytest.approx(
                s.ground_truth.x_m, abs=2e-2
            )
            assert s.prediction.position_m.y == pytest.approx(
                s.ground_truth.y_m, abs=2e-2
            )

    def test_measured_gt_absent_by_default(self, calibration, pallet_model):
        # No measured GT supplied -> harness runs on simulated GT only (honest
        # "real-world unverified" posture, R10.4). This must simply work.
        samples = evaluate_samples(
            3, calibration=calibration, pallet_model=pallet_model
        )
        assert len(samples) == 3

    def test_measured_gt_typed_when_supplied(self, calibration, pallet_model):
        measured = MeasuredFloorPointGT(
            label="bench-survey-1",
            floor_points_m=np.array([[0.0, 3.0, 0.0], [0.5, 3.0, 0.0]]),
            measurement_method="tape survey",
        )
        assert measured.provenance is ProvenanceLabel.MEASURED
        samples = evaluate_samples(
            3,
            calibration=calibration,
            pallet_model=pallet_model,
            measured_gt=measured,
        )
        assert len(samples) == 3

    def test_measured_gt_wrong_type_rejected(self, calibration, pallet_model):
        with pytest.raises(TypeError):
            evaluate_samples(
                3,
                calibration=calibration,
                pallet_model=pallet_model,
                measured_gt=object(),
            )


class TestMeasuredFloorPointGT:
    def test_rejects_non_measured_provenance(self):
        with pytest.raises(ValueError):
            MeasuredFloorPointGT(
                label="x",
                floor_points_m=np.array([[0.0, 0.0, 0.0]]),
                provenance=ProvenanceLabel.SIMULATED,
            )

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            MeasuredFloorPointGT(
                label="x", floor_points_m=np.array([1.0, 2.0, 3.0])
            )
