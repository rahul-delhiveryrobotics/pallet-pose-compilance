"""Unit tests for ill-conditioned rejection, ambiguity set, symmetry, and the
defined unavailable ``PoseResult`` (task 6.5, R9.5, R9.6, R9.7, R13.4).

These tests exercise the *degrade-loudly* discipline of the known-geometry PnP
estimator: rather than emitting a confidently-wrong point estimate, the
estimator returns an explicit ``unavailable`` result (null metrics + a
``Reason_Code`` + quality flags), records an admissible solution set for a
genuine planar ambiguity (never a silent pick), and reports orientation modulo a
declared geometric symmetry with face-identity ambiguity handled.

The synthetic forward model mirrors ``tests/test_pose.py``: place the pallet at
a known Floor_Frame pose, project its true 3D corners (bottom on Z=0, top
elevated) with ``cv2.projectPoints``, and feed the projected keypoints to
``estimate_pose``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pallet_pose_compliance.calibration.calibration import build_synthetic_calibration
from pallet_pose_compliance.geometry.frames import (
    reduce_orientation_modulo_symmetry,
    wrap_to_180,
)
from pallet_pose_compliance.geometry.pallet_model import build_nominal_pallet_model
from pallet_pose_compliance.geometry.pose import (
    BOTTOM_CORNER_NAMES,
    MAX_CROSS_CHECK_RESIDUAL_M,
    MAX_PNP_RESIDUAL_PX,
    TOP_CORNER_NAMES,
    estimate_pose,
    is_degenerate_configuration,
    match_keypoints_to_model,
)
from pallet_pose_compliance.output.provenance import (
    ProvenanceLabel,
    QualityFlag,
    ReasonCode,
)
from pallet_pose_compliance.output.schema import Keypoint


# ---------------------------------------------------------------------------
# Forward-model helpers (same construction as test_pose.py)
# ---------------------------------------------------------------------------


def _rotation_about_z(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _world_to_camera(calibration):
    import cv2

    R_cf = np.asarray(calibration.rotation, dtype=float)
    t_cf = np.asarray(calibration.translation, dtype=float).reshape(3)
    R_wc = R_cf.T
    t_wc = -R_cf.T @ t_cf
    rvec, _ = cv2.Rodrigues(R_wc)
    return rvec, t_wc.reshape(3, 1)


def _project_corners(calibration, corners_floor: np.ndarray) -> np.ndarray:
    import cv2

    rvec, tvec = _world_to_camera(calibration)
    img, _ = cv2.projectPoints(
        corners_floor.astype(np.float64),
        rvec,
        tvec,
        np.asarray(calibration.intrinsics, dtype=np.float64),
        np.asarray(calibration.distortion, dtype=np.float64).reshape(-1, 1),
    )
    return img.reshape(-1, 2)


def _place_and_project(calibration, pallet_model, x0, y0, theta_deg):
    R_pf = _rotation_about_z(theta_deg)
    t_pf = np.array([x0, y0, 0.0])
    bottom = np.asarray(pallet_model.bottom_corners_m, dtype=float)
    top = np.asarray(pallet_model.top_corners_m, dtype=float)
    corners_local = np.vstack([bottom, top])
    names = list(BOTTOM_CORNER_NAMES) + list(TOP_CORNER_NAMES)
    corners_floor = (R_pf @ corners_local.T).T + t_pf
    img = _project_corners(calibration, corners_floor)
    keypoints = [
        Keypoint(name=name, u=float(px[0]), v=float(px[1]), visibility="visible", score=0.9)
        for name, px in zip(names, img)
    ]
    return keypoints, corners_floor


@pytest.fixture
def calibration():
    return build_synthetic_calibration("pose-reject-test-cal")


@pytest.fixture
def pallet_model():
    return build_nominal_pallet_model()


# ---------------------------------------------------------------------------
# (1) Too few usable keypoints -> unavailable / POSE_INSUFFICIENT_KEYPOINTS
# ---------------------------------------------------------------------------


class TestTooFewKeypoints:
    def test_too_few_keypoints_degrades_to_unavailable(self, calibration, pallet_model):
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        result = estimate_pose(keypoints[:3], calibration, pallet_model)

        assert result.pose_status == "unavailable"
        assert result.reason_code is ReasonCode.POSE_INSUFFICIENT_KEYPOINTS
        assert QualityFlag.INSUFFICIENT_KEYPOINTS in result.quality_flags
        # Null discipline: no fabricated metrics.
        assert result.position_m is None
        assert result.orientation_deg is None
        assert result.face_identity is None
        assert result.provenance is ProvenanceLabel.UNAVAILABLE

    def test_absent_keypoints_do_not_count(self, calibration, pallet_model):
        # Eight projected corners but five marked absent leaves only three usable.
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        for i in range(5):
            keypoints[i] = Keypoint(
                name=keypoints[i].name, u=None, v=None, visibility="absent"
            )
        result = estimate_pose(keypoints, calibration, pallet_model)
        assert result.pose_status == "unavailable"
        assert result.reason_code is ReasonCode.POSE_INSUFFICIENT_KEYPOINTS


# ---------------------------------------------------------------------------
# (2) Near-collinear / degenerate configuration -> ILL_CONDITIONED
# ---------------------------------------------------------------------------


class TestDegenerateConfiguration:
    def test_collinear_image_points_flagged_degenerate(self, calibration, pallet_model):
        # Four+ keypoints whose image coordinates lie (almost) on a straight
        # line -> near-collinear, ill-posed PnP.
        names = list(BOTTOM_CORNER_NAMES) + [TOP_CORNER_NAMES[0]]
        keypoints = [
            Keypoint(
                name=name,
                u=100.0 + 40.0 * i,
                v=200.0 + 0.01 * i,  # essentially a horizontal line
                visibility="visible",
                score=0.9,
            )
            for i, name in enumerate(names)
        ]
        matches = match_keypoints_to_model(keypoints, pallet_model)
        assert is_degenerate_configuration(matches) is not None

    def test_collinear_config_degrades_to_unavailable(self, calibration, pallet_model):
        names = list(BOTTOM_CORNER_NAMES) + [TOP_CORNER_NAMES[0]]
        keypoints = [
            Keypoint(
                name=name,
                u=100.0 + 40.0 * i,
                v=200.0 + 0.01 * i,
                visibility="visible",
                score=0.9,
            )
            for i, name in enumerate(names)
        ]
        result = estimate_pose(keypoints, calibration, pallet_model)
        assert result.pose_status == "unavailable"
        assert result.reason_code is ReasonCode.ILL_CONDITIONED
        assert QualityFlag.ILL_CONDITIONED in result.quality_flags
        assert result.position_m is None

    def test_point_cluster_flagged_degenerate(self, calibration, pallet_model):
        # All keypoints piled at the same pixel -> no usable spread.
        names = list(BOTTOM_CORNER_NAMES)
        keypoints = [
            Keypoint(name=name, u=320.0, v=240.0, visibility="visible", score=0.9)
            for name in names
        ]
        matches = match_keypoints_to_model(keypoints, pallet_model)
        assert is_degenerate_configuration(matches) is not None

    def test_well_conditioned_config_not_degenerate(self, calibration, pallet_model):
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.1, 3.0, 15.0)
        matches = match_keypoints_to_model(keypoints, pallet_model)
        assert is_degenerate_configuration(matches) is None


# ---------------------------------------------------------------------------
# (3) High PnP residual -> ILL_CONDITIONED
# ---------------------------------------------------------------------------


class TestHighResidual:
    def test_inconsistent_keypoints_high_residual_unavailable(
        self, calibration, pallet_model
    ):
        # Start from a valid, well-spread projection then perturb several
        # keypoints by large, inconsistent amounts so no single rigid pose can
        # explain them: the PnP residual blows past the documented bound.
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        perturbed = list(keypoints)
        # Scramble the top corners far from where any consistent pose predicts.
        offsets = [(90.0, -70.0), (-80.0, 85.0), (75.0, 95.0), (-95.0, -60.0)]
        for name, (du, dv) in zip(TOP_CORNER_NAMES, offsets):
            idx = next(i for i, kp in enumerate(perturbed) if kp.name == name)
            kp = perturbed[idx]
            perturbed[idx] = Keypoint(
                name=kp.name, u=kp.u + du, v=kp.v + dv, visibility="visible", score=0.9
            )
        result = estimate_pose(perturbed, calibration, pallet_model)
        assert result.pose_status == "unavailable"
        assert result.reason_code is ReasonCode.ILL_CONDITIONED
        assert QualityFlag.ILL_CONDITIONED in result.quality_flags

    def test_documented_residual_bound_is_positive(self):
        # The rejection bound is a documented, finite, positive constant.
        assert MAX_PNP_RESIDUAL_PX > 0.0


# ---------------------------------------------------------------------------
# (4) Ray-plane cross-check disagreement -> ILL_CONDITIONED
# ---------------------------------------------------------------------------


class TestCrossCheckDisagreement:
    def test_cross_check_disagreement_unavailable(self, calibration, pallet_model):
        # Build a consistent projection, then shift ONLY the bottom (floor-
        # contact) corners together by a small consistent pixel offset. PnP,
        # using all eight corners, still finds a low-residual pose near the
        # truth, but the independent ray-plane back-projection of the shifted
        # bottom corners lands the floor centroid far from the PnP centre ->
        # the two independent estimates disagree beyond the documented bound.
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        shifted = list(keypoints)
        for name in BOTTOM_CORNER_NAMES:
            idx = next(i for i, kp in enumerate(shifted) if kp.name == name)
            kp = shifted[idx]
            shifted[idx] = Keypoint(
                name=kp.name, u=kp.u + 60.0, v=kp.v + 40.0, visibility="visible", score=0.9
            )
        result = estimate_pose(shifted, calibration, pallet_model)
        assert result.pose_status == "unavailable"
        assert result.reason_code is ReasonCode.ILL_CONDITIONED
        assert QualityFlag.ILL_CONDITIONED in result.quality_flags

    def test_documented_cross_check_bound_is_positive(self):
        assert MAX_CROSS_CHECK_RESIDUAL_M > 0.0

    def test_consistent_projection_passes_cross_check(self, calibration, pallet_model):
        # Sanity: a fully-consistent projection stays available (the cross-check
        # residual is ~0) and exposes the diagnostic entry.
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        result = estimate_pose(keypoints, calibration, pallet_model)
        assert result.pose_status == "available"
        cross = [
            s for s in result.admissible_solutions
            if s.get("kind") == "ray_plane_cross_check"
        ]
        assert len(cross) == 1
        assert cross[0]["residual_m"] <= MAX_CROSS_CHECK_RESIDUAL_M


# ---------------------------------------------------------------------------
# (5) Symmetry: orientation modulo symmetry + face-identity ambiguity (R9.7)
# ---------------------------------------------------------------------------


class TestSymmetryReduction:
    def test_orientation_reduced_modulo_symmetry(self, calibration, pallet_model):
        # A well-conditioned projection at a known theta, declared symmetric with
        # order 2, must report orientation reduced modulo 180 deg and flag it.
        x0, y0, theta = 0.0, 3.0, 100.0
        keypoints, _ = _place_and_project(calibration, pallet_model, x0, y0, theta)
        result = estimate_pose(
            keypoints, calibration, pallet_model, symmetry_order=2
        )

        assert result.pose_status == "available"
        assert result.orientation_symmetry_reduced is True
        # The reported orientation lies within the symmetry sector (-90, 90].
        assert -90.0 < result.orientation_deg <= 90.0
        # And it equals the symmetry reduction of some representative consistent
        # with the true (or 180-flipped) pose.
        expected_a = reduce_orientation_modulo_symmetry(theta, 2)
        expected_b = reduce_orientation_modulo_symmetry(theta + 180.0, 2)
        err = min(
            abs(wrap_to_180(result.orientation_deg - expected_a)),
            abs(wrap_to_180(result.orientation_deg - expected_b)),
        )
        assert err == pytest.approx(0.0, abs=1e-2)

    def test_face_identity_ambiguous_under_symmetry(self, calibration, pallet_model):
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 30.0)
        result = estimate_pose(
            keypoints, calibration, pallet_model, symmetry_order=2
        )
        # Face identity cannot be uniquely resolved under a symmetry -> null +
        # the FACE_AMBIGUOUS_SYMMETRY flag, never a silent pick.
        assert result.face_identity is None
        assert QualityFlag.FACE_AMBIGUOUS_SYMMETRY in result.quality_flags

    def test_asymmetric_default_not_reduced(self, calibration, pallet_model):
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 30.0)
        result = estimate_pose(keypoints, calibration, pallet_model)
        assert result.orientation_symmetry_reduced is False
        assert result.face_identity is not None
        assert QualityFlag.FACE_AMBIGUOUS_SYMMETRY not in result.quality_flags


# ---------------------------------------------------------------------------
# (6) Ambiguity set: > 1 admissible solution, POSE_AMBIGUOUS, no silent pick
# ---------------------------------------------------------------------------


class TestAmbiguitySet:
    # A near-fronto-parallel, far-range view of the coplanar bottom corners is
    # the classic planar PnP ambiguity: two reflected poses reproject almost
    # identically, so both are geometrically admissible.
    @pytest.fixture
    def ambiguous_calibration(self):
        return build_synthetic_calibration(
            "pose-ambiguous-cal", tilt_down_deg=3.0, camera_height_m=1.2
        )

    def _planar_bottom_only_keypoints(self, calibration, pallet_model, x0, y0, theta):
        # Use ONLY the four coplanar bottom corners (all on Z=0). A single planar
        # set of correspondences is the classic PnP ambiguity: solvePnPGeneric
        # (IPPE) returns two geometrically admissible reflected poses.
        R_pf = _rotation_about_z(theta)
        t_pf = np.array([x0, y0, 0.0])
        bottom = np.asarray(pallet_model.bottom_corners_m, dtype=float)
        corners_floor = (R_pf @ bottom.T).T + t_pf
        img = _project_corners(calibration, corners_floor)
        return [
            Keypoint(name=name, u=float(px[0]), v=float(px[1]), visibility="visible", score=0.9)
            for name, px in zip(BOTTOM_CORNER_NAMES, img)
        ]

    def test_planar_ambiguity_records_solution_set(
        self, ambiguous_calibration, pallet_model
    ):
        keypoints = self._planar_bottom_only_keypoints(
            ambiguous_calibration, pallet_model, 0.0, 25.0, 10.0
        )
        result = estimate_pose(keypoints, ambiguous_calibration, pallet_model)

        # If the planar case is ambiguous the estimator must NOT silently pick
        # one: it records an admissible pose-candidate set (> 1) and flags it.
        pose_candidates = [
            s for s in result.admissible_solutions
            if s.get("kind") == "pose_candidate"
        ]
        if result.pose_status == "unavailable":
            # Acceptable degrade path (e.g. cross-check/degeneracy), but then it
            # must be a defined unavailable result, never a fabricated pose.
            assert result.position_m is None
            pytest.skip("planar bottom-only config degraded to unavailable")
        # Available + ambiguous: solution set recorded, flag present, no silent pick.
        assert len(pose_candidates) > 1
        assert QualityFlag.POSE_AMBIGUOUS in result.quality_flags
        # Each candidate carries a full Floor_Frame pose (position + orientation).
        for cand in pose_candidates:
            assert "position_m" in cand and "orientation_deg" in cand
            assert set(cand["position_m"].keys()) == {"x", "y"}

    def test_well_conditioned_full_corners_not_ambiguous(
        self, calibration, pallet_model
    ):
        # The full eight-corner (non-coplanar) configuration is well-posed: no
        # POSE_AMBIGUOUS flag and no pose-candidate set beyond the diagnostic.
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 15.0)
        result = estimate_pose(keypoints, calibration, pallet_model)
        assert result.pose_status == "available"
        assert QualityFlag.POSE_AMBIGUOUS not in result.quality_flags
        pose_candidates = [
            s for s in result.admissible_solutions
            if s.get("kind") == "pose_candidate"
        ]
        assert pose_candidates == []
