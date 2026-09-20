"""Unit tests for Monte Carlo pose uncertainty propagation (task 6.8, R13, R27).

Method under test (design: "Uncertainty via Monte Carlo propagation"):
- Build a synthetic (SIMULATED) calibration and the nominal pallet 3D model.
- Place the pallet at a known Floor_Frame pose and project its true 3D corners
  to pixels (the same forward model used by ``test_pose.py``).
- Feed the projected keypoints to the uncertainty machinery and assert the
  documented behaviour: zero input noise -> ~zero spread, spread grows
  monotonically with input noise, the uncertainty is ``estimated`` +
  assumption-documented, excessive uncertainty degrades to unavailable, and the
  propagation is reproducible under a fixed seed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pallet_pose_compliance.calibration.calibration import build_synthetic_calibration
from pallet_pose_compliance.geometry.pallet_model import build_nominal_pallet_model
from pallet_pose_compliance.geometry.pose import (
    MAX_ORIENTATION_STD_DEG,
    MAX_POSITION_STD_M,
    PoseNoiseModel,
    estimate_pose_with_uncertainty,
    propagate_pose_uncertainty,
)
from pallet_pose_compliance.output.provenance import (
    ProvenanceLabel,
    QualityFlag,
    ReasonCode,
)
from pallet_pose_compliance.output.schema import Keypoint

# Reuse the forward-model projection helper from the core pose tests.
from tests.test_pose import _place_and_project


@pytest.fixture
def calibration():
    return build_synthetic_calibration("pose-unc-test-cal")


@pytest.fixture
def pallet_model():
    return build_nominal_pallet_model()


@pytest.fixture
def keypoints(calibration, pallet_model):
    kps, _ = _place_and_project(calibration, pallet_model, 0.1, 3.0, 15.0)
    return kps


_ZERO_NOISE = PoseNoiseModel(
    keypoint_sigma_px=0.0,
    focal_rel_sigma=0.0,
    principal_point_sigma_px=0.0,
    extrinsic_rot_sigma_deg=0.0,
    extrinsic_trans_sigma_m=0.0,
    dim_sigma_m=0.0,
)


def _position_std(uncertainty) -> float:
    """Larger principal 1-sigma position std (metres) from the 2x2 covariance."""
    cov = np.asarray(uncertainty.position_cov_m2, dtype=float)
    return math.sqrt(float(np.max(np.linalg.eigvalsh(cov))))


# ---------------------------------------------------------------------------
# Zero input noise -> ~zero spread
# ---------------------------------------------------------------------------


class TestZeroNoise:
    def test_zero_input_noise_gives_zero_spread(self, keypoints, calibration, pallet_model):
        unc = propagate_pose_uncertainty(
            keypoints,
            calibration,
            pallet_model,
            n_samples=32,
            seed=42,
            noise_model=_ZERO_NOISE,
        )
        assert unc.position_cov_m2 is not None
        assert unc.orientation_std_deg is not None
        # With no input noise every sample re-solves to the same pose, so the
        # covariance and orientation std collapse to ~0.
        assert _position_std(unc) == pytest.approx(0.0, abs=1e-9)
        assert unc.orientation_std_deg == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Monotonic growth of spread with input noise
# ---------------------------------------------------------------------------


class TestMonotonicGrowth:
    def test_position_spread_grows_with_keypoint_noise(
        self, keypoints, calibration, pallet_model
    ):
        stds = []
        for sigma in (0.0, 0.5, 1.5, 3.0):
            noise = PoseNoiseModel(
                keypoint_sigma_px=sigma,
                focal_rel_sigma=0.0,
                principal_point_sigma_px=0.0,
                extrinsic_rot_sigma_deg=0.0,
                extrinsic_trans_sigma_m=0.0,
                dim_sigma_m=0.0,
            )
            unc = propagate_pose_uncertainty(
                keypoints, calibration, pallet_model, n_samples=200, seed=7, noise_model=noise
            )
            stds.append(_position_std(unc))
        # Strictly non-decreasing, and clearly increasing overall.
        for smaller, larger in zip(stds, stds[1:]):
            assert larger >= smaller - 1e-9
        assert stds[-1] > stds[0] + 1e-4

    def test_orientation_spread_grows_with_keypoint_noise(
        self, keypoints, calibration, pallet_model
    ):
        stds = []
        for sigma in (0.0, 0.5, 1.5, 3.0):
            noise = PoseNoiseModel(
                keypoint_sigma_px=sigma,
                focal_rel_sigma=0.0,
                principal_point_sigma_px=0.0,
                extrinsic_rot_sigma_deg=0.0,
                extrinsic_trans_sigma_m=0.0,
                dim_sigma_m=0.0,
            )
            unc = propagate_pose_uncertainty(
                keypoints, calibration, pallet_model, n_samples=200, seed=7, noise_model=noise
            )
            stds.append(unc.orientation_std_deg)
        for smaller, larger in zip(stds, stds[1:]):
            assert larger >= smaller - 1e-9
        assert stds[-1] > stds[0] + 1e-3


# ---------------------------------------------------------------------------
# Provenance + assumptions discipline (R13.2, R13.3, R26.2)
# ---------------------------------------------------------------------------


class TestProvenanceAndAssumptions:
    def test_uncertainty_is_estimated_with_documented_assumptions(
        self, keypoints, calibration, pallet_model
    ):
        result = estimate_pose_with_uncertainty(
            keypoints, calibration, pallet_model, n_samples=64, seed=42
        )
        assert result.pose_status == "available"
        assert result.uncertainty is not None
        # Assumption-based -> ESTIMATED, never MEASURED (R26.2).
        assert result.uncertainty.provenance is ProvenanceLabel.ESTIMATED
        assert result.uncertainty.method == "monte_carlo"
        assert len(result.uncertainty.assumptions) > 0
        # The assumptions document each of the modelled noise sources.
        joined = " ".join(result.uncertainty.assumptions).lower()
        assert "keypoint" in joined
        assert "intrinsics" in joined or "focal" in joined
        assert "extrinsic" in joined
        assert "dimension" in joined or "dimensions" in joined
        # Real, quantified metrics attached to an available pose (R13.1, R27.1).
        assert result.uncertainty.position_cov_m2 is not None
        assert result.uncertainty.orientation_std_deg is not None
        assert result.uncertainty.orientation_std_deg >= 0.0

    def test_dim_uncertainty_defaults_from_pallet_model(
        self, keypoints, calibration, pallet_model
    ):
        # No explicit noise model -> dim_sigma taken from the model (R13.3).
        unc = propagate_pose_uncertainty(
            keypoints, calibration, pallet_model, n_samples=8, seed=1
        )
        joined = " ".join(unc.assumptions)
        assert f"{pallet_model.dim_uncertainty_m} m" in joined


# ---------------------------------------------------------------------------
# Excessive uncertainty degrades to unavailable (R27.2)
# ---------------------------------------------------------------------------


class TestDegradeOnExcessiveUncertainty:
    def test_excessive_noise_degrades_to_unavailable(
        self, keypoints, calibration, pallet_model
    ):
        # Very large keypoint noise blows up the propagated spread well past the
        # documented thresholds -> defined unavailable with the right reason.
        huge_noise = PoseNoiseModel(
            keypoint_sigma_px=40.0,
            focal_rel_sigma=0.05,
            principal_point_sigma_px=20.0,
            extrinsic_rot_sigma_deg=10.0,
            extrinsic_trans_sigma_m=0.2,
            dim_sigma_m=0.1,
        )
        result = estimate_pose_with_uncertainty(
            keypoints,
            calibration,
            pallet_model,
            n_samples=128,
            seed=42,
            noise_model=huge_noise,
        )
        assert result.pose_status == "unavailable"
        assert result.reason_code is ReasonCode.POSE_UNCERTAINTY_EXCEEDED
        assert QualityFlag.POSE_UNCERTAINTY_EXCEEDED in result.quality_flags
        # Null discipline: no fabricated metrics on the degraded pose.
        assert result.position_m is None
        assert result.orientation_deg is None
        assert result.provenance is ProvenanceLabel.UNAVAILABLE

    def test_small_noise_stays_available_with_uncertainty(
        self, keypoints, calibration, pallet_model
    ):
        result = estimate_pose_with_uncertainty(
            keypoints, calibration, pallet_model, n_samples=128, seed=42
        )
        assert result.pose_status == "available"
        # Sanity: the propagated spread is within the documented thresholds.
        assert _position_std(result.uncertainty) <= MAX_POSITION_STD_M
        assert result.uncertainty.orientation_std_deg <= MAX_ORIENTATION_STD_DEG

    def test_threshold_boundary_low_position_std_available(
        self, keypoints, calibration, pallet_model
    ):
        # Set an absurdly generous threshold: even moderate noise stays available.
        moderate = PoseNoiseModel(keypoint_sigma_px=2.0)
        result = estimate_pose_with_uncertainty(
            keypoints,
            calibration,
            pallet_model,
            n_samples=128,
            seed=42,
            noise_model=moderate,
            max_position_std_m=10.0,
            max_orientation_std_deg=360.0,
        )
        assert result.pose_status == "available"
        assert result.uncertainty is not None


# ---------------------------------------------------------------------------
# Reproducibility under a fixed seed (R28.3)
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self, keypoints, calibration, pallet_model):
        kwargs = dict(n_samples=64, seed=123)
        a = propagate_pose_uncertainty(keypoints, calibration, pallet_model, **kwargs)
        b = propagate_pose_uncertainty(keypoints, calibration, pallet_model, **kwargs)
        assert a.position_cov_m2 == b.position_cov_m2
        assert a.orientation_std_deg == b.orientation_std_deg

    def test_different_seed_differs(self, keypoints, calibration, pallet_model):
        a = propagate_pose_uncertainty(
            keypoints, calibration, pallet_model, n_samples=64, seed=1
        )
        b = propagate_pose_uncertainty(
            keypoints, calibration, pallet_model, n_samples=64, seed=2
        )
        # Different seeds should generally give different covariances.
        assert a.position_cov_m2 != b.position_cov_m2


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_too_few_samples_raises(self, keypoints, calibration, pallet_model):
        with pytest.raises(ValueError):
            propagate_pose_uncertainty(
                keypoints, calibration, pallet_model, n_samples=1, seed=42
            )

    def test_unavailable_base_pose_passes_through(self, calibration, pallet_model):
        # Too few keypoints -> the base estimator is unavailable; the
        # uncertainty-aware wrapper returns it unchanged (no fabricated spread).
        kps, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        result = estimate_pose_with_uncertainty(
            kps[:3], calibration, pallet_model, n_samples=8, seed=42
        )
        assert result.pose_status == "unavailable"
        assert result.uncertainty is None
        assert result.reason_code is not None
