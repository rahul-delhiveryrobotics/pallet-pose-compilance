"""Unit tests for the known-geometry PnP pose estimator core (task 6.2, R7, R9).

Method under test (design: "Pose method (known-geometry PnP)"):
- Build a synthetic (SIMULATED) calibration and the nominal pallet 3D model.
- Place the pallet at a *known* Floor_Frame pose (x, y, theta).
- Project the model's true 3D corners (bottom on Z=0, top elevated) into the
  image with ``cv2.projectPoints`` (the *forward* model).
- Feed the projected keypoints to ``estimate_pose`` and assert the recovered
  position and orientation match the known pose within tolerance.

These tests pin the soundness requirements by construction:
- The top corners are projected at their **true elevated height**, so the solve
  is genuine known-geometry PnP, not a floor homography on elevated corners
  (R9.2) and not a bbox+tilt heuristic (R9.3).
- The ray-plane (Z=0) back-projection recovers the known bottom-corner floor
  points, cross-checking the PnP centre (R9.4).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pallet_pose_compliance.calibration.calibration import build_synthetic_calibration
from pallet_pose_compliance.geometry.frames import wrap_to_180
from pallet_pose_compliance.geometry.pallet_model import build_nominal_pallet_model
from pallet_pose_compliance.geometry.pose import (
    BOTTOM_CORNER_NAMES,
    TOP_CORNER_NAMES,
    PoseEstimationError,
    backproject_bottom_corners_to_floor,
    estimate_pose,
    match_keypoints_to_model,
    ray_plane_intersection,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel
from pallet_pose_compliance.output.schema import Keypoint


# ---------------------------------------------------------------------------
# Synthetic projection helpers (the forward model)
# ---------------------------------------------------------------------------


def _rotation_about_z(theta_deg: float) -> np.ndarray:
    """Rotation matrix about Floor_Frame +Z by ``theta_deg`` (CCW from above)."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _world_to_camera(calibration):
    """Return ``(rvec, tvec)`` mapping Floor_Frame -> camera for projectPoints.

    The calibration stores camera->floor (``X_floor = R_cf @ X_cam + t_cf``);
    the inverse floor->camera pose used by ``cv2.projectPoints`` is
    ``R_wc = R_cf^T`` and ``t_wc = -R_cf^T @ t_cf``.
    """
    import cv2

    R_cf = np.asarray(calibration.rotation, dtype=float)
    t_cf = np.asarray(calibration.translation, dtype=float).reshape(3)
    R_wc = R_cf.T
    t_wc = -R_cf.T @ t_cf
    rvec, _ = cv2.Rodrigues(R_wc)
    return rvec, t_wc.reshape(3, 1)


def _place_and_project(calibration, pallet_model, x0, y0, theta_deg):
    """Place the pallet at ``(x0, y0, theta)`` and project its corners to pixels.

    Returns a list of :class:`Keypoint` (four bottom + four top) with the
    projected pixel coordinates, all marked ``visible``.
    """
    import cv2

    R_pf = _rotation_about_z(theta_deg)
    t_pf = np.array([x0, y0, 0.0])

    bottom = np.asarray(pallet_model.bottom_corners_m, dtype=float)
    top = np.asarray(pallet_model.top_corners_m, dtype=float)
    corners_local = np.vstack([bottom, top])  # (8, 3) pallet-local
    names = list(BOTTOM_CORNER_NAMES) + list(TOP_CORNER_NAMES)

    # Pallet-local -> Floor_Frame.
    corners_floor = (R_pf @ corners_local.T).T + t_pf

    rvec, tvec = _world_to_camera(calibration)
    img, _ = cv2.projectPoints(
        corners_floor.astype(np.float64),
        rvec,
        tvec,
        np.asarray(calibration.intrinsics, dtype=np.float64),
        np.asarray(calibration.distortion, dtype=np.float64).reshape(-1, 1),
    )
    img = img.reshape(-1, 2)

    keypoints = [
        Keypoint(name=name, u=float(px[0]), v=float(px[1]), visibility="visible", score=0.9)
        for name, px in zip(names, img)
    ]
    return keypoints, corners_floor


@pytest.fixture
def calibration():
    # A synthetic camera looking down the +Y depth axis, tilted down 20 deg.
    return build_synthetic_calibration("pose-test-cal")


@pytest.fixture
def pallet_model():
    return build_nominal_pallet_model()


# ---------------------------------------------------------------------------
# Round-trip: recover a known pose
# ---------------------------------------------------------------------------


class TestKnownPoseRoundTrip:
    @pytest.mark.parametrize(
        "x0, y0, theta_deg",
        [
            (0.0, 3.0, 0.0),
            (0.4, 3.5, 20.0),
            (-0.6, 4.0, -35.0),
            (0.2, 2.5, 90.0),
            (0.0, 5.0, 175.0),
        ],
    )
    def test_recovers_position_and_orientation(
        self, calibration, pallet_model, x0, y0, theta_deg
    ):
        keypoints, _ = _place_and_project(calibration, pallet_model, x0, y0, theta_deg)
        result = estimate_pose(keypoints, calibration, pallet_model)

        assert result.pose_status == "available"
        assert result.position_m is not None
        # Position within a tight tolerance (well under the 2 cm eval bar) for a
        # noise-free synthetic projection.
        assert result.position_m.x == pytest.approx(x0, abs=1e-3)
        assert result.position_m.y == pytest.approx(y0, abs=1e-3)

        # Orientation matches modulo the (-180,180] wrap and the pallet's own
        # 180-degree rectangular ambiguity (long axis is a line, not a ray):
        # accept theta or theta+180.
        err = abs(wrap_to_180(result.orientation_deg - theta_deg))
        err_flip = abs(wrap_to_180(result.orientation_deg - theta_deg - 180.0))
        assert min(err, err_flip) == pytest.approx(0.0, abs=1e-2)

    def test_available_pose_fields_and_provenance(
        self, calibration, pallet_model
    ):
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.1, 3.0, 10.0)
        result = estimate_pose(keypoints, calibration, pallet_model)

        # Orientation wrapped to (-180, 180].
        assert -180.0 < result.orientation_deg <= 180.0
        # Face identity resolved for asymmetric geometry (symmetry_order=1).
        assert result.face_identity is not None
        # Range/viewing angle populated for envelope/sensitivity analysis.
        assert result.range_m is not None and result.range_m > 0.0
        assert result.viewing_angle_deg is not None
        assert 0.0 <= result.viewing_angle_deg <= 90.0
        # Provenance never claims measured off a simulated calibration (R26.2).
        assert calibration.provenance is ProvenanceLabel.SIMULATED
        assert result.provenance is ProvenanceLabel.SIMULATED
        # Core estimator leaves uncertainty for task 6.8.
        assert result.uncertainty is None

    def test_range_matches_camera_to_pallet_distance(self, calibration, pallet_model):
        x0, y0 = 0.0, 3.0
        keypoints, _ = _place_and_project(calibration, pallet_model, x0, y0, 0.0)
        result = estimate_pose(keypoints, calibration, pallet_model)
        # Camera centre is at (0, 0, 1.2); pallet centre ~(0, 3, 0).
        expected = math.sqrt(x0**2 + y0**2 + 1.2**2)
        assert result.range_m == pytest.approx(expected, abs=2e-2)


# ---------------------------------------------------------------------------
# Known-geometry: uses true heights, NOT a floor homography / bbox+tilt
# ---------------------------------------------------------------------------


class TestUsesTrueHeights:
    def test_top_corners_projected_above_bottom_corners(
        self, calibration, pallet_model
    ):
        # For a camera looking down, elevated top corners project to *different*
        # pixels than the bottom corners; if the method treated them as coplanar
        # (a floor homography), it would be geometrically wrong. Here they differ
        # and the solve still recovers the true pose using their true Z.
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        by_name = {kp.name: (kp.u, kp.v) for kp in keypoints}
        for b_name, t_name in zip(BOTTOM_CORNER_NAMES, TOP_CORNER_NAMES):
            bu, bv = by_name[b_name]
            tu, tv = by_name[t_name]
            assert (abs(bu - tu) > 1.0) or (abs(bv - tv) > 1.0)

    def test_solves_with_only_top_and_bottom_mixed(self, calibration, pallet_model):
        # Provide a mix that includes elevated top corners; a floor-homography
        # method could not use them, but known-geometry PnP does (R9.2).
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.3, 3.2, 15.0)
        # Keep 2 bottom + 2 top (still 4 correspondences, mixed heights).
        subset = [
            kp
            for kp in keypoints
            if kp.name
            in {
                BOTTOM_CORNER_NAMES[0],
                BOTTOM_CORNER_NAMES[2],
                TOP_CORNER_NAMES[1],
                TOP_CORNER_NAMES[3],
            }
        ]
        result = estimate_pose(subset, calibration, pallet_model)
        assert result.pose_status == "available"
        assert result.position_m.x == pytest.approx(0.3, abs=5e-2)
        assert result.position_m.y == pytest.approx(3.2, abs=5e-2)


# ---------------------------------------------------------------------------
# Ray-plane (Z = 0) intersection recovers known floor points (R9.4)
# ---------------------------------------------------------------------------


class TestRayPlaneCrossCheck:
    def test_backprojection_recovers_known_bottom_corners(
        self, calibration, pallet_model
    ):
        x0, y0, theta = 0.2, 3.0, 25.0
        keypoints, corners_floor = _place_and_project(
            calibration, pallet_model, x0, y0, theta
        )
        matches = match_keypoints_to_model(keypoints, pallet_model)
        floor_pts = backproject_bottom_corners_to_floor(matches, calibration)

        # The four known bottom-corner floor points (first four rows).
        known_bottom = corners_floor[:4]
        # Ray-plane intersection should recover them (Z=0) within tight tol.
        assert floor_pts.shape == (4, 3)
        for recovered, known in zip(floor_pts, known_bottom):
            assert recovered[0] == pytest.approx(known[0], abs=1e-3)
            assert recovered[1] == pytest.approx(known[1], abs=1e-3)
            assert recovered[2] == pytest.approx(0.0, abs=1e-9)

    def test_cross_check_residual_exposed(self, calibration, pallet_model):
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        result = estimate_pose(keypoints, calibration, pallet_model)
        # The core exposes the ray-plane cross-check as diagnostic metadata.
        cross = [
            s
            for s in result.admissible_solutions
            if s.get("kind") == "ray_plane_cross_check"
        ]
        assert len(cross) == 1
        # For a consistent synthetic projection the residual is ~0.
        assert cross[0]["residual_m"] == pytest.approx(0.0, abs=1e-3)

    def test_ray_plane_intersection_basic(self):
        # Camera at height 2 looking straight down: ray (0,0,-1) hits (0,0,0).
        pt = ray_plane_intersection(
            np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0])
        )
        assert np.allclose(pt, [0.0, 0.0, 0.0])

    def test_ray_parallel_to_plane_raises(self):
        with pytest.raises(PoseEstimationError):
            ray_plane_intersection(
                np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0])
            )

    def test_ray_pointing_away_raises(self):
        with pytest.raises(PoseEstimationError):
            ray_plane_intersection(
                np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0])
            )


# ---------------------------------------------------------------------------
# Matching / insufficient-keypoint handling
# ---------------------------------------------------------------------------


class TestMatching:
    def test_absent_and_missing_coords_excluded(self, calibration, pallet_model):
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        # Mark one absent (drops coords) and one occluded-but-labelled (kept).
        keypoints[0] = Keypoint(
            name=keypoints[0].name, u=None, v=None, visibility="absent"
        )
        keypoints[1] = Keypoint(
            name=keypoints[1].name,
            u=keypoints[1].u,
            v=keypoints[1].v,
            visibility="occluded",
            score=0.5,
        )
        matches = match_keypoints_to_model(keypoints, pallet_model)
        # 8 total minus 1 absent = 7 usable; occluded one retained.
        assert len(matches) == 7
        assert keypoints[0].name not in matches.names
        assert keypoints[1].name in matches.names

    def test_too_few_keypoints_degrades_to_unavailable(
        self, calibration, pallet_model
    ):
        # Per R13.4 the estimator returns a *defined* unavailable PoseResult
        # (never a fabricated pose) when too few usable keypoints remain; the
        # matcher helper still raises, but estimate_pose degrades loudly.
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        # Keep only 3 -> below the PnP minimum.
        matches_input = keypoints[:3]
        # The low-level matcher still raises for callers that want the hard path.
        with pytest.raises(PoseEstimationError):
            match_keypoints_to_model(matches_input, pallet_model)
        # The top-level estimator degrades to a defined unavailable result.
        result = estimate_pose(matches_input, calibration, pallet_model)
        assert result.pose_status == "unavailable"
        assert result.reason_code is not None
        assert result.position_m is None
        assert result.orientation_deg is None

    def test_unknown_names_ignored(self, calibration, pallet_model):
        keypoints, _ = _place_and_project(calibration, pallet_model, 0.0, 3.0, 0.0)
        keypoints.append(
            Keypoint(name="not_a_corner", u=10.0, v=10.0, visibility="visible", score=0.5)
        )
        matches = match_keypoints_to_model(keypoints, pallet_model)
        assert "not_a_corner" not in matches.names
        assert len(matches) == 8
