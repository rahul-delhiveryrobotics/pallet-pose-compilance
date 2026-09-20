"""Known-geometry PnP pose estimator (design: Pose method, R7, R9).

This module implements the *core happy-path* metric pose estimator for a
detected pallet. It follows the design's **known-geometry PnP** method and is
deliberately built on the pallet's *true 3D model* — bottom deck corners on the
floor (``Z = 0``) and top deck corners **elevated** by the deck height — so the
solver respects keypoint height above the floor.

This is emphatically **not**:

- a floor-plane homography applied to elevated corners (R9.2) — the elevated
  top corners keep their true ``Z`` and are fed to ``cv2.solvePnP`` with their
  real 3D coordinates; and
- a bounding-box + approximate-tilt inference (R9.3) — no bbox or tilt heuristic
  is used anywhere; the pose comes from 2D<->3D point correspondences.

Method (design: "Pose method (known-geometry PnP)")
---------------------------------------------------
1. Match observed image keypoints (``visible``/``occluded``, with ``u``/``v``)
   to the pallet model's named 3D corners (bottom + top). ``absent`` keypoints
   and keypoints missing coordinates are excluded (R9.4).
2. Undistort the matched image points with the calibration distortion model,
   then solve PnP for the **pallet-in-camera** pose using the corners' true 3D
   coordinates (including elevated top corners).
3. Compose the calibration's **camera -> Floor_Frame** extrinsics with the
   PnP pallet-in-camera pose to express the pallet pose in Floor_Frame. The
   reported position is the floor-projected pallet centre; the orientation is
   the pallet long axis about ``+Z`` wrapped to ``(-180, 180]``.
4. **Ray-plane (Z = 0) cross-check**: back-project the undistorted rays of the
   floor-contact bottom corners through the calibrated camera and intersect them
   with the floor plane; compare the centroid of those metric floor points with
   the PnP floor-projected centre and expose the residual (R9.4). (Rejection on
   disagreement is task 6.5; here the residual is only computed/exposed.)
5. Derive :class:`~..geometry.frames.FaceIdentity`, and compute ``range_m`` and
   ``viewing_angle_deg`` for the envelope/sensitivity analysis.

Degrade-loudly discipline (task 6.5, R9.5/R9.6/R9.7/R13.4)
----------------------------------------------------------
Rather than emitting a confidently-wrong point estimate, :func:`estimate_pose`
returns a *defined* ``unavailable`` :class:`PoseResult` (null metrics + a
``Reason_Code`` + quality flags) when the solve is not trustworthy:

- **Too few usable keypoints** -> ``POSE_INSUFFICIENT_KEYPOINTS`` +
  ``INSUFFICIENT_KEYPOINTS``.
- **Near-collinear/coplanar-degenerate configuration**, **high PnP residual**,
  or **ray-plane cross-check disagreement** beyond documented bounds ->
  ``ILL_CONDITIONED``.

When PnP admits multiple geometrically admissible solutions (planar/near-planar
ambiguity, via :func:`cv2.solvePnPGeneric`), the result records the **admissible
solution set** in ``admissible_solutions`` and flags ``POSE_AMBIGUOUS`` rather
than silently picking one (R9.6). When the visible geometry is symmetric
(``symmetry_order > 1``) the orientation is reported reduced modulo the symmetry
with ``orientation_symmetry_reduced=True``, and face identity degrades to
``None`` with ``FACE_AMBIGUOUS_SYMMETRY`` and the admissible faces recorded
(R9.7). The documented rejection/ambiguity thresholds are module constants.

Provenance discipline (R26)
---------------------------
The geometric pose is at best as trustworthy as the calibration it rests on. A
``SIMULATED`` calibration yields a ``SIMULATED`` pose; an ``ESTIMATED``
calibration yields an ``ESTIMATED`` pose. We never upgrade to ``MEASURED`` off a
non-measured calibration (R26.2). An unavailable pose carries provenance
``UNAVAILABLE``. Monte Carlo uncertainty (task 6.8) is layered on later; this
estimator returns an ``available`` pose with ``uncertainty=None`` (the schema
permits it) so that task can fill in the real uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..calibration.calibration import Calibration
from ..output.provenance import ProvenanceLabel, QualityFlag, ReasonCode
from ..output.schema import (
    Keypoint,
    PalletModel,
    PositionM,
    PoseResult,
    PoseUncertainty,
)
from .frames import (
    FLOOR_PLANE_Z_M,
    circular_mean_deg,
    circular_std_deg,
    derive_face_identity,
    floor_projected_centre,
    rad_to_deg,
    reduce_orientation_modulo_symmetry,
    wrap_to_180,
)

__all__ = [
    "PoseEstimationError",
    "MatchedCorrespondences",
    "match_keypoints_to_model",
    "undistort_image_points",
    "solve_pnp_pallet_in_camera",
    "compose_pallet_in_floor",
    "pallet_orientation_deg",
    "ray_plane_intersection",
    "backproject_bottom_corners_to_floor",
    "estimate_pose",
    "estimate_pose_with_uncertainty",
    "propagate_pose_uncertainty",
    "PoseNoiseModel",
    "is_degenerate_configuration",
    "MIN_PNP_CORRESPONDENCES",
    "MIN_IMAGE_SPREAD_PX",
    "MIN_COLLINEARITY_RATIO",
    "MAX_PNP_RESIDUAL_PX",
    "MAX_CROSS_CHECK_RESIDUAL_M",
    "AMBIGUITY_REPROJECTION_RATIO",
    "MC_DEFAULT_SAMPLES",
    "MC_KEYPOINT_SIGMA_PX",
    "MC_FOCAL_REL_SIGMA",
    "MC_PRINCIPAL_POINT_SIGMA_PX",
    "MC_EXTRINSIC_ROT_SIGMA_DEG",
    "MC_EXTRINSIC_TRANS_SIGMA_M",
    "MC_DEFAULT_DIM_SIGMA_M",
    "MAX_POSITION_STD_M",
    "MAX_ORIENTATION_STD_DEG",
]

#: Minimum number of matched 2D<->3D correspondences PnP needs to be solvable.
#: PnP needs >= 4 non-degenerate points for a well-posed iterative solve.
MIN_PNP_CORRESPONDENCES = 4

# ---------------------------------------------------------------------------
# Documented rejection thresholds (task 6.5, R9.5/R9.6/R13.4)
# ---------------------------------------------------------------------------
# These are the *documented bounds* referenced by the design's
# "Ill-conditioned / degenerate rejection" and "Ambiguity representation"
# clauses. They are deliberately conservative: a well-conditioned, noise-free
# synthetic projection (task 6.2's happy path) passes them with wide margin,
# while genuinely degenerate/inconsistent inputs are rejected loudly rather
# than yielding a confidently-wrong point estimate.

#: Minimum spatial spread (largest singular value of the mean-centred image
#: points, in pixels) required for a well-posed solve. Below this the observed
#: keypoints occupy a near-point cluster with no usable geometry.
MIN_IMAGE_SPREAD_PX = 5.0

#: Minimum ratio of the smaller to the larger singular value of the
#: mean-centred point cloud (image or object). Below this the points are
#: near-collinear (2D) or near-coplanar-degenerate (3D), so PnP is
#: ill-conditioned. E.g. 0.02 means the minor spread must be >= 2% of the major
#: spread.
MIN_COLLINEARITY_RATIO = 0.02

#: Maximum acceptable RMS PnP reprojection residual (pixels) of the solved pose
#: over the correspondences. Beyond this the solve does not explain the observed
#: keypoints and is rejected as ill-conditioned.
MAX_PNP_RESIDUAL_PX = 8.0

#: Maximum acceptable disagreement (metres) between the independent ray-plane
#: (Z = 0) floor centroid and the PnP floor-projected centre. Beyond this the
#: two independent geometric estimates disagree and the pose is rejected.
MAX_CROSS_CHECK_RESIDUAL_M = 0.10

#: When PnP returns multiple candidate solutions, a second candidate is treated
#: as *geometrically admissible* (a genuine ambiguity, not a clear runner-up)
#: when its reprojection error is within this factor of the best candidate's.
#: E.g. 3.0 means "within 3x the best reprojection error".
AMBIGUITY_REPROJECTION_RATIO = 3.0

#: Absolute reprojection tolerance (pixels) for admissibility. A second
#: candidate whose residual is within this many pixels of the best is treated as
#: admissible even when the best residual is ~0 (a noise-free synthetic solve),
#: because keypoint localisation noise of this magnitude would make the two
#: solutions indistinguishable in practice. This is what surfaces the classic
#: planar reflection ambiguity that the pure ratio test would miss at zero
#: residual.
AMBIGUITY_ABS_TOL_PX = 2.0

# ---------------------------------------------------------------------------
# Documented Monte Carlo uncertainty-propagation parameters (task 6.8, R13)
# ---------------------------------------------------------------------------
# The pose is only as trustworthy as the noisy inputs it rests on: keypoint
# localisation is noisy, the calibration is imperfect, and the pallet model is a
# nominal spec. We quantify how that input noise propagates into the reported
# Floor_Frame pose by Monte Carlo: draw many samples of perturbed inputs from
# *documented* distributions, re-solve the pose per sample, and summarise the
# spread of the resulting (x, y, orientation). The summary (2x2 position
# covariance + orientation circular std) is an **assumption-based `estimated`**
# quantity — never `measured` — because the noise magnitudes below are modelled
# assumptions, not empirically calibrated against measured ground truth (R13.2,
# R13.3, R26.2). Each assumption is also recorded in the returned
# ``PoseUncertainty.assumptions`` list so the report is self-documenting.

#: Default number of Monte Carlo samples. Enough to stabilise a 2x2 covariance
#: and a scalar circular std without making the per-pose solve prohibitively
#: slow (each sample re-runs PnP).
MC_DEFAULT_SAMPLES = 128

#: Keypoint localisation noise: 1-sigma Gaussian pixel noise added independently
#: to each observed keypoint's (u, v). Represents sub-detector localisation
#: jitter of a well-trained keypoint regressor.
MC_KEYPOINT_SIGMA_PX = 1.0

#: Calibration intrinsics noise: relative 1-sigma on the focal lengths
#: (fx, fy), i.e. sigma_f = MC_FOCAL_REL_SIGMA * f. A small fractional focal
#: uncertainty models residual intrinsics-calibration error.
MC_FOCAL_REL_SIGMA = 0.002

#: Calibration intrinsics noise: 1-sigma Gaussian pixel noise on the principal
#: point (cx, cy).
MC_PRINCIPAL_POINT_SIGMA_PX = 1.0

#: Calibration extrinsics noise: 1-sigma of a small-angle random rotation
#: (degrees) applied about a random axis to the camera->floor rotation, modelling
#: residual extrinsic-orientation error (and mild suspected mounting drift).
MC_EXTRINSIC_ROT_SIGMA_DEG = 0.25

#: Calibration extrinsics noise: 1-sigma Gaussian noise (metres) added
#: independently to each component of the camera->floor translation.
MC_EXTRINSIC_TRANS_SIGMA_M = 0.005

#: Fallback pallet-dimension uncertainty (metres, 1-sigma) used when the pallet
#: model does not carry its own ``dim_uncertainty_m``. When the model provides a
#: value it takes precedence (R13.3).
MC_DEFAULT_DIM_SIGMA_M = 0.010

# ---------------------------------------------------------------------------
# Documented uncertainty degrade thresholds (task 6.8, R27.2)
# ---------------------------------------------------------------------------
# When the *propagated* uncertainty exceeds these documented bounds the pose is
# no longer trustworthy for pose-dependent SOP checks, so the pose degrades to a
# defined ``unavailable`` result (POSE_UNCERTAINTY_EXCEEDED) rather than
# reporting a confidently-wrong point estimate (R27.2, design Error Handling).
# These are pinned to the verdict engine's pose-quality gate
# (``configs/verdict.yaml`` -> ``pose_quality.max_position_std_cm`` = 5.0 cm and
# ``max_orientation_std_deg`` = 5.0 deg) so the pose stage and the verdict stage
# agree on what "too uncertain" means; they are module constants here so the
# pose estimator is self-contained.

#: Maximum tolerated 1-sigma position std (metres) before the pose degrades to
#: unavailable. The position std is the larger principal std of the propagated
#: 2x2 position covariance. 0.05 m == verdict.yaml ``max_position_std_cm`` 5.0.
MAX_POSITION_STD_M = 0.05

#: Maximum tolerated 1-sigma orientation std (degrees) before the pose degrades
#: to unavailable. Matches verdict.yaml ``max_orientation_std_deg`` 5.0.
MAX_ORIENTATION_STD_DEG = 5.0

#: The pallet model's bottom-corner keypoint names (floor-contact, Z = 0).
BOTTOM_CORNER_NAMES = ("bottom_corner_0", "bottom_corner_1", "bottom_corner_2", "bottom_corner_3")
#: The pallet model's top-corner keypoint names (elevated by deck height).
TOP_CORNER_NAMES = ("top_corner_0", "top_corner_1", "top_corner_2", "top_corner_3")


class PoseEstimationError(ValueError):
    """Raised when the core estimator cannot solve a well-posed pose.

    This is the *hard* error path for the happy-path core (e.g. too few usable
    keypoints or a PnP failure). The richer degrade-to-``unavailable`` policy
    (ill-conditioned/ambiguity thresholds) is layered on in task 6.5; callers of
    the core may catch this to emit an ``unavailable`` :class:`PoseResult`.
    """


@dataclass(frozen=True)
class MatchedCorrespondences:
    """A set of matched 2D image points and their true 3D pallet-local points.

    Attributes
    ----------
    names:
        Keypoint names, aligned with the rows of ``object_points`` /
        ``image_points``.
    object_points:
        ``(N, 3)`` true 3D corner coordinates in the pallet-local frame (bottom
        corners on ``Z = 0``, top corners elevated). This is what makes the
        method *known-geometry* PnP rather than a floor homography (R9.2).
    image_points:
        ``(N, 2)`` observed (distorted) pixel coordinates.
    bottom_mask:
        Boolean ``(N,)`` mask marking the floor-contact bottom corners (used for
        the ray-plane cross-check).
    """

    names: tuple[str, ...]
    object_points: np.ndarray
    image_points: np.ndarray
    bottom_mask: np.ndarray

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.object_points.shape[0])


# ---------------------------------------------------------------------------
# Step 1: match keypoints to the 3D model
# ---------------------------------------------------------------------------


def match_keypoints_to_model(
    keypoints: Sequence[Keypoint],
    pallet_model: PalletModel,
) -> MatchedCorrespondences:
    """Match observed keypoints to the pallet model's true 3D corners (R9.4).

    Only keypoints with visibility ``visible`` or ``occluded`` **and** with
    non-null ``u``/``v`` are used; ``absent`` keypoints (and any keypoint
    missing coordinates) are excluded. Names are matched to the model's
    bottom/top corner names, and each matched point keeps its **true 3D
    coordinate including height** (top corners are elevated) so PnP accounts for
    keypoint height above the floor.

    Parameters
    ----------
    keypoints:
        Observed pallet keypoints (typically from the detector/localiser).
    pallet_model:
        The known 3D pallet model (bottom corners on ``Z = 0``, top elevated).

    Returns
    -------
    MatchedCorrespondences
        The aligned 3D object points, 2D image points, names, and bottom-corner
        mask.

    Raises
    ------
    PoseEstimationError
        If fewer than :data:`MIN_PNP_CORRESPONDENCES` usable correspondences
        remain after filtering.
    """
    model_points: dict[str, tuple[list[float], bool]] = {}
    for name, corner in zip(BOTTOM_CORNER_NAMES, pallet_model.bottom_corners_m):
        model_points[name] = ([float(c) for c in corner], True)
    for name, corner in zip(TOP_CORNER_NAMES, pallet_model.top_corners_m):
        model_points[name] = ([float(c) for c in corner], False)

    names: list[str] = []
    obj: list[list[float]] = []
    img: list[list[float]] = []
    bottom: list[bool] = []
    for kp in keypoints:
        if kp.visibility == "absent":
            continue
        if kp.u is None or kp.v is None:
            continue
        entry = model_points.get(kp.name)
        if entry is None:
            continue
        point3d, is_bottom = entry
        names.append(kp.name)
        obj.append(point3d)
        img.append([float(kp.u), float(kp.v)])
        bottom.append(is_bottom)

    if len(obj) < MIN_PNP_CORRESPONDENCES:
        raise PoseEstimationError(
            f"need >= {MIN_PNP_CORRESPONDENCES} usable keypoints matched to the "
            f"pallet model, got {len(obj)}"
        )

    return MatchedCorrespondences(
        names=tuple(names),
        object_points=np.asarray(obj, dtype=float),
        image_points=np.asarray(img, dtype=float),
        bottom_mask=np.asarray(bottom, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Step 2a: undistort image points
# ---------------------------------------------------------------------------


def undistort_image_points(
    image_points: np.ndarray,
    calibration: Calibration,
) -> np.ndarray:
    """Undistort pixel points to *ideal* pixel coordinates (R9.4).

    Applies the calibration's lens-distortion model via
    ``cv2.undistortPoints`` and re-projects the result through the intrinsics so
    the output is again in **pixel** coordinates but corresponds to an ideal
    pinhole camera (zero distortion). This is used before ray back-projection;
    the PnP solve itself takes the distorted points plus the distortion model.

    Parameters
    ----------
    image_points:
        ``(N, 2)`` distorted pixel coordinates.
    calibration:
        The camera calibration (intrinsics + distortion).

    Returns
    -------
    np.ndarray
        ``(N, 2)`` undistorted pixel coordinates.
    """
    import cv2

    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(calibration.intrinsics, dtype=np.float64)
    dist = np.asarray(calibration.distortion, dtype=np.float64).reshape(-1, 1)
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    return undistorted.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Step 2b: solve PnP for the pallet pose in the camera frame
# ---------------------------------------------------------------------------


def solve_pnp_pallet_in_camera(
    matches: MatchedCorrespondences,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve PnP for the pallet-in-camera pose ``(R_pc, t_pc)``.

    Uses ``cv2.solvePnP`` with the matched 3D pallet-local points and the
    observed **distorted** image points plus the calibration distortion model,
    so distortion is handled inside the solver (R9.4). The returned rotation and
    translation satisfy ``X_cam = R_pc @ X_pallet + t_pc``.

    Parameters
    ----------
    matches:
        The matched 2D<->3D correspondences (true 3D heights preserved).
    calibration:
        The camera calibration (intrinsics + distortion).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(R_pc, t_pc)`` — a 3x3 rotation matrix and a 3-vector (metres).

    Raises
    ------
    PoseEstimationError
        If ``cv2.solvePnP`` fails to produce a solution.
    """
    import cv2

    obj = np.asarray(matches.object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(matches.image_points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(calibration.intrinsics, dtype=np.float64)
    dist = np.asarray(calibration.distortion, dtype=np.float64).reshape(-1, 1)

    # Flag selection: the iterative (DLT-initialised) solver requires >= 6
    # points for a general 3D configuration; SQPNP is a stable global solver
    # that works from 3 points upward and handles both planar and non-planar
    # configurations, so it is the sound choice for the minimal 4-corner case.
    n_points = obj.shape[0]
    flag = cv2.SOLVEPNP_ITERATIVE if n_points >= 6 else cv2.SOLVEPNP_SQPNP

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flag)
    if not ok:
        raise PoseEstimationError("cv2.solvePnP failed to estimate the pallet pose")

    R_pc, _ = cv2.Rodrigues(rvec)
    t_pc = np.asarray(tvec, dtype=float).reshape(3)
    return R_pc, t_pc


# ---------------------------------------------------------------------------
# Step 3: compose extrinsics to Floor_Frame + orientation
# ---------------------------------------------------------------------------


def compose_pallet_in_floor(
    R_pc: np.ndarray,
    t_pc: np.ndarray,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose PnP pallet-in-camera pose with camera->floor extrinsics.

    Given ``X_cam = R_pc @ X_pallet + t_pc`` (from PnP) and the calibration's
    ``X_floor = R_cf @ X_cam + t_cf`` (camera -> Floor_Frame), the pallet pose in
    Floor_Frame is ``X_floor = R_pf @ X_pallet + t_pf`` with
    ``R_pf = R_cf @ R_pc`` and ``t_pf = R_cf @ t_pc + t_cf``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(R_pf, t_pf)`` — pallet orientation (3x3) and origin (3-vector) in
        Floor_Frame.
    """
    R_cf = np.asarray(calibration.rotation, dtype=float)
    t_cf = np.asarray(calibration.translation, dtype=float).reshape(3)
    R_pf = R_cf @ np.asarray(R_pc, dtype=float)
    t_pf = R_cf @ np.asarray(t_pc, dtype=float).reshape(3) + t_cf
    return R_pf, t_pf


def pallet_orientation_deg(R_pf: np.ndarray) -> float:
    """Extract the pallet long-axis orientation about ``+Z`` (degrees).

    The pallet-local ``+X`` axis (the long axis) expressed in Floor_Frame is the
    first column of ``R_pf``. Its projection onto the floor plane gives the
    orientation ``theta = atan2(y, x)`` about ``+Z``, wrapped to ``(-180, 180]``
    (design: Orientation, ``theta = 0`` when the long axis is along ``+X``).

    Parameters
    ----------
    R_pf:
        The 3x3 pallet-in-floor rotation matrix.

    Returns
    -------
    float
        Orientation in degrees, wrapped to ``(-180, 180]``.
    """
    long_axis_floor = np.asarray(R_pf, dtype=float)[:, 0]
    theta_rad = math.atan2(float(long_axis_floor[1]), float(long_axis_floor[0]))
    return wrap_to_180(rad_to_deg(theta_rad))


# ---------------------------------------------------------------------------
# Step 4: ray-plane (Z = 0) intersection cross-check
# ---------------------------------------------------------------------------


def ray_plane_intersection(
    camera_centre_floor: np.ndarray,
    ray_dir_floor: np.ndarray,
    plane_z: float = FLOOR_PLANE_Z_M,
) -> np.ndarray:
    """Intersect a ray (in Floor_Frame) with the horizontal plane ``Z = plane_z``.

    The ray is ``P(s) = camera_centre_floor + s * ray_dir_floor``. Solving for
    the ``Z = plane_z`` crossing gives ``s = (plane_z - c_z) / d_z``; the metric
    floor point is ``P(s)``.

    Parameters
    ----------
    camera_centre_floor:
        The camera centre in Floor_Frame (3-vector, metres).
    ray_dir_floor:
        The ray direction in Floor_Frame (3-vector).
    plane_z:
        The plane height (defaults to the floor ``Z = 0``).

    Returns
    -------
    np.ndarray
        The ``(x, y, z)`` intersection point in Floor_Frame.

    Raises
    ------
    PoseEstimationError
        If the ray is parallel to the plane (no unique intersection) or points
        away from it (``s <= 0``, i.e. the floor is behind the camera).
    """
    c = np.asarray(camera_centre_floor, dtype=float).reshape(3)
    d = np.asarray(ray_dir_floor, dtype=float).reshape(3)
    dz = float(d[2])
    if math.isclose(dz, 0.0, abs_tol=1e-12):
        raise PoseEstimationError(
            "ray is parallel to the floor plane; no unique intersection"
        )
    s = (float(plane_z) - float(c[2])) / dz
    if s <= 0.0:
        raise PoseEstimationError(
            "floor-plane intersection lies behind the camera (s <= 0)"
        )
    return c + s * d


def backproject_bottom_corners_to_floor(
    matches: MatchedCorrespondences,
    calibration: Calibration,
) -> np.ndarray:
    """Back-project bottom-corner rays to metric floor points (R9.4).

    For each floor-contact bottom corner, undistort its pixel, form the camera
    ray in the camera frame, rotate it into Floor_Frame using the calibration
    extrinsics, and intersect it with the floor plane ``Z = 0``. These metric
    floor points are used to cross-check the PnP floor-projected centre — a
    genuinely independent geometric estimate that does *not* rely on the pallet
    model's rigid solve.

    Parameters
    ----------
    matches:
        The matched correspondences (bottom corners identified by the mask).
    calibration:
        The camera calibration (intrinsics + distortion + extrinsics).

    Returns
    -------
    np.ndarray
        ``(M, 3)`` metric floor points (one per bottom corner) in Floor_Frame.

    Raises
    ------
    PoseEstimationError
        If there are no bottom corners to back-project.
    """
    bottom_mask = np.asarray(matches.bottom_mask, dtype=bool)
    if not bottom_mask.any():
        raise PoseEstimationError(
            "no floor-contact bottom corners available for ray-plane cross-check"
        )

    bottom_px = matches.image_points[bottom_mask]
    undistorted = undistort_image_points(bottom_px, calibration)

    K = np.asarray(calibration.intrinsics, dtype=float)
    K_inv = np.linalg.inv(K)
    R_cf = np.asarray(calibration.rotation, dtype=float)
    # Camera centre in Floor_Frame: X_floor = R_cf @ 0 + t_cf.
    camera_centre_floor = np.asarray(calibration.translation, dtype=float).reshape(3)

    floor_points = []
    for u, v in undistorted:
        # Undistorted pixel -> normalised camera ray (camera frame, +Z forward).
        ray_cam = K_inv @ np.array([u, v, 1.0], dtype=float)
        # Rotate direction into Floor_Frame (translation does not affect a dir).
        ray_floor = R_cf @ ray_cam
        floor_points.append(
            ray_plane_intersection(camera_centre_floor, ray_floor, FLOOR_PLANE_Z_M)
        )
    return np.asarray(floor_points, dtype=float)


# ---------------------------------------------------------------------------
# Degeneracy / conditioning checks (task 6.5, R9.5)
# ---------------------------------------------------------------------------


def _spread_singular_values(points: np.ndarray) -> np.ndarray:
    """Return the singular values of the mean-centred point cloud.

    The singular values describe the point cloud's spread along its principal
    axes. A tiny trailing singular value means the points collapse onto a lower
    dimensional subspace (a line for 2D image points, a plane/line for 3D object
    points) — the hallmark of a degenerate PnP configuration.
    """
    pts = np.asarray(points, dtype=float)
    centred = pts - pts.mean(axis=0, keepdims=True)
    # full_matrices=False keeps the singular values only; robust to N >= dim.
    return np.linalg.svd(centred, compute_uv=False)


def is_degenerate_configuration(matches: MatchedCorrespondences) -> Optional[str]:
    """Detect a near-collinear/coplanar-degenerate configuration (R9.5).

    Returns a human-readable reason string when the matched image points and/or
    object points are too poorly spread for a well-posed PnP solve, or ``None``
    when the configuration is well-conditioned.

    The checks use documented thresholds:

    - **Image spread** (:data:`MIN_IMAGE_SPREAD_PX`): the largest singular value
      of the mean-centred image points must exceed a few pixels — otherwise the
      keypoints form a near-point cluster.
    - **Image collinearity** (:data:`MIN_COLLINEARITY_RATIO`): the ratio of the
      smallest to the largest image singular value must exceed the bound —
      otherwise the image points are near-collinear.
    - **Object collinearity** (:data:`MIN_COLLINEARITY_RATIO`): the two largest
      object-point singular values must have a ratio above the bound — otherwise
      the 3D model points selected are near-collinear (an ill-posed reference).

    Parameters
    ----------
    matches:
        The matched 2D<->3D correspondences.

    Returns
    -------
    Optional[str]
        A reason string when degenerate, otherwise ``None``.
    """
    img = np.asarray(matches.image_points, dtype=float)
    obj = np.asarray(matches.object_points, dtype=float)

    img_sv = _spread_singular_values(img)  # up to 2 values
    if img_sv.size == 0 or float(img_sv[0]) < MIN_IMAGE_SPREAD_PX:
        return (
            f"image points span only {float(img_sv[0]) if img_sv.size else 0.0:.3f}px "
            f"(< {MIN_IMAGE_SPREAD_PX}px); near-point-cluster, no usable geometry"
        )
    # Ratio of minor to major image spread; near-zero => near-collinear.
    img_ratio = float(img_sv[-1]) / float(img_sv[0])
    if img_ratio < MIN_COLLINEARITY_RATIO:
        return (
            f"image points near-collinear (minor/major spread ratio {img_ratio:.4f} "
            f"< {MIN_COLLINEARITY_RATIO})"
        )

    # Object points: the top-two principal spreads define the reference plane;
    # if the second is negligible relative to the first the 3D reference points
    # are near-collinear (degenerate regardless of image spread).
    obj_sv = _spread_singular_values(obj)
    if obj_sv.size >= 2 and float(obj_sv[0]) > 0.0:
        obj_ratio = float(obj_sv[1]) / float(obj_sv[0])
        if obj_ratio < MIN_COLLINEARITY_RATIO:
            return (
                f"object (3D model) reference points near-collinear "
                f"(second/first spread ratio {obj_ratio:.4f} < {MIN_COLLINEARITY_RATIO})"
            )
    return None


def _reprojection_residual_px(
    R_pc: np.ndarray,
    t_pc: np.ndarray,
    matches: MatchedCorrespondences,
    calibration: Calibration,
) -> float:
    """RMS reprojection residual (pixels) of a solved pose over the matches.

    Projects the matched object points through the solved pallet-in-camera pose
    (with the calibration intrinsics + distortion) and compares against the
    observed image points. A large residual means the pose does not explain the
    observations (ill-conditioned / high-residual solve).
    """
    import cv2

    rvec, _ = cv2.Rodrigues(np.asarray(R_pc, dtype=np.float64))
    tvec = np.asarray(t_pc, dtype=np.float64).reshape(3, 1)
    obj = np.asarray(matches.object_points, dtype=np.float64).reshape(-1, 1, 3)
    K = np.asarray(calibration.intrinsics, dtype=np.float64)
    dist = np.asarray(calibration.distortion, dtype=np.float64).reshape(-1, 1)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    observed = np.asarray(matches.image_points, dtype=float).reshape(-1, 2)
    diffs = projected - observed
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))


def _solve_pnp_candidates(
    matches: MatchedCorrespondences,
    calibration: Calibration,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Solve PnP and return all candidate solutions with their residuals.

    Uses ``cv2.solvePnPGeneric``, which returns *every* solution the solver
    finds — for planar/near-planar configurations PnP is genuinely ambiguous and
    returns multiple candidates. Each candidate is returned as
    ``(R_pc, t_pc, rms_reprojection_residual_px)`` sorted by ascending residual,
    so the caller can (a) pick the best for the reported pose and (b) inspect
    whether more than one candidate is geometrically admissible (R9.6).

    Raises
    ------
    PoseEstimationError
        If the solver produces no solution at all.
    """
    import cv2

    obj = np.asarray(matches.object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(matches.image_points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(calibration.intrinsics, dtype=np.float64)
    dist = np.asarray(calibration.distortion, dtype=np.float64).reshape(-1, 1)

    # Flag selection. When the selected object points are (near-)coplanar the
    # solve is a planar PnP with a well-known two-fold reflection ambiguity; the
    # IPPE solver is designed for the planar case and returns *both* admissible
    # solutions, which is exactly what R9.6 needs to represent the ambiguity set.
    # For a genuinely non-planar (full bottom+top) configuration we use the
    # general solvers (iterative for >= 6 points, SQPNP otherwise).
    obj_pts = obj.reshape(-1, 3)
    obj_sv = _spread_singular_values(obj_pts)
    is_planar_object = obj_sv.size >= 3 and float(obj_sv[0]) > 0.0 and (
        float(obj_sv[2]) / float(obj_sv[0]) < MIN_COLLINEARITY_RATIO
    )

    n_points = obj.shape[0]
    if is_planar_object:
        flag = cv2.SOLVEPNP_IPPE
    elif n_points >= 6:
        flag = cv2.SOLVEPNP_ITERATIVE
    else:
        flag = cv2.SOLVEPNP_SQPNP

    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=flag)
    if not retval or rvecs is None or len(rvecs) == 0:
        raise PoseEstimationError("cv2.solvePnPGeneric produced no pose solution")

    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    for rvec, tvec in zip(rvecs, tvecs):
        R_pc, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        t_pc = np.asarray(tvec, dtype=float).reshape(3)
        residual = _reprojection_residual_px(R_pc, t_pc, matches, calibration)
        candidates.append((R_pc, t_pc, residual))

    candidates.sort(key=lambda c: c[2])
    return candidates


# ---------------------------------------------------------------------------
# Top-level estimator
# ---------------------------------------------------------------------------


def _unavailable_pose(
    reason_code: ReasonCode,
    quality_flags: Sequence[QualityFlag],
) -> PoseResult:
    """Build a *defined* unavailable :class:`PoseResult` (R13.4, R24 null discipline).

    All metric fields are null, the provenance is ``UNAVAILABLE``, and the
    ``reason_code`` plus ``quality_flags`` explain the degradation — never a
    fabricated or zero pose.
    """
    return PoseResult(
        pose_status="unavailable",
        reason_code=reason_code,
        position_m=None,
        orientation_deg=None,
        orientation_symmetry_reduced=False,
        face_identity=None,
        admissible_solutions=[],
        uncertainty=None,
        range_m=None,
        viewing_angle_deg=None,
        quality_flags=list(quality_flags),
        provenance=ProvenanceLabel.UNAVAILABLE,
    )


def _resolve_pose_provenance(calibration: Calibration) -> ProvenanceLabel:
    """Pick the pose provenance, never upgrading past the calibration's (R26.2).

    The metric pose can be no more trustworthy than the calibration it rests on.
    A ``SIMULATED`` calibration gives a ``SIMULATED`` pose; an ``ESTIMATED`` one
    gives an ``ESTIMATED`` pose; a ``MEASURED`` calibration gives an
    ``ESTIMATED`` pose here because the geometric solve is still an estimate
    (empirical calibration of pose error would be needed to claim ``MEASURED``).
    We never claim ``MEASURED`` off a non-measured calibration.
    """
    if calibration.provenance is ProvenanceLabel.SIMULATED:
        return ProvenanceLabel.SIMULATED
    # Measured or estimated calibration -> the geometric pose is an estimate.
    return ProvenanceLabel.ESTIMATED


def estimate_pose(
    keypoints: Sequence[Keypoint],
    calibration: Calibration,
    pallet_model: PalletModel,
    *,
    symmetry_order: int = 1,
) -> PoseResult:
    """Estimate a pallet's metric Floor_Frame pose via known-geometry PnP.

    It matches observed keypoints to the pallet's true 3D corners (bottom on
    ``Z = 0``, top elevated), undistorts and solves PnP for the pallet-in-camera
    pose, composes the camera->Floor_Frame extrinsics, and reports the
    floor-projected pallet centre, wrapped orientation, face identity, range,
    and viewing angle for well-conditioned input. A ray-plane (``Z = 0``)
    cross-check of the bottom corners guards the PnP centre.

    Degrade-loudly (task 6.5): too-few keypoints, a near-collinear/coplanar
    configuration, a high PnP residual, or a cross-check disagreement beyond the
    documented bounds return a *defined* ``unavailable`` result (null metrics +
    ``Reason_Code`` + quality flags) instead of a fabricated pose (R9.5/R13.4).
    Multiple geometrically admissible PnP solutions are recorded as an admissible
    set with ``POSE_AMBIGUOUS`` (R9.6); symmetric geometry reports orientation
    modulo the symmetry with face-identity ambiguity handled (R9.7). Monte Carlo
    uncertainty is task 6.8 (``uncertainty=None`` here).

    Explicitly **not** a floor homography on elevated corners (R9.2) and **not**
    bbox+tilt inference (R9.3): the pose is solved from true-height 2D<->3D
    correspondences.

    Parameters
    ----------
    keypoints:
        Observed pallet keypoints matched by name to the model corners.
    calibration:
        The camera calibration (intrinsics, distortion, camera->floor
        extrinsics, provenance).
    pallet_model:
        The known 3D pallet model.
    symmetry_order:
        Rotational symmetry order of the visible geometry. ``1`` (default) means
        asymmetric; ``> 1`` triggers orientation-modulo-symmetry reduction and
        face-identity ambiguity handling (R9.7).

    Returns
    -------
    PoseResult
        For well-conditioned input, an ``available`` pose result with position,
        wrapped orientation, face identity, range and viewing angle
        (``uncertainty`` is ``None``, filled by task 6.8). For a degenerate,
        high-residual, or cross-check-inconsistent solve — or too few usable
        keypoints — a *defined* ``unavailable`` result with a ``Reason_Code``,
        null metrics, and ``ILL_CONDITIONED``/``INSUFFICIENT_KEYPOINTS`` quality
        flags (task 6.5, R9.5/R13.4). When PnP admits multiple geometrically
        admissible solutions the result carries the admissible solution set and
        a ``POSE_AMBIGUOUS`` flag rather than silently picking one (R9.6).
    """
    # Step 1: match keypoints to the true 3D corners. Too few usable keypoints
    # degrades to a defined unavailable result rather than raising (R13.4).
    try:
        matches = match_keypoints_to_model(keypoints, pallet_model)
    except PoseEstimationError:
        return _unavailable_pose(
            ReasonCode.POSE_INSUFFICIENT_KEYPOINTS,
            [QualityFlag.INSUFFICIENT_KEYPOINTS],
        )

    # Rejection (b): near-collinear/coplanar-degenerate configuration (R9.5).
    degeneracy = is_degenerate_configuration(matches)
    if degeneracy is not None:
        return _unavailable_pose(
            ReasonCode.ILL_CONDITIONED, [QualityFlag.ILL_CONDITIONED]
        )

    # Step 2: solve PnP for the pallet-in-camera pose, keeping *all* candidate
    # solutions so planar/near-planar ambiguity can be represented (R9.6).
    try:
        candidates = _solve_pnp_candidates(matches, calibration)
    except PoseEstimationError:
        return _unavailable_pose(
            ReasonCode.ILL_CONDITIONED, [QualityFlag.ILL_CONDITIONED]
        )
    R_pc, t_pc, best_residual_px = candidates[0]

    # Rejection (c): high PnP residual — the solve does not explain the observed
    # keypoints beyond the documented bound (R9.5).
    if best_residual_px > MAX_PNP_RESIDUAL_PX:
        return _unavailable_pose(
            ReasonCode.ILL_CONDITIONED, [QualityFlag.ILL_CONDITIONED]
        )

    # Step 3: compose extrinsics to Floor_Frame.
    R_pf, t_pf = compose_pallet_in_floor(R_pc, t_pc, calibration)

    # Position: floor-projected pallet centre. The pallet model is centred at
    # its floor-projected centre, so the pallet-local origin maps to t_pf; its
    # (x, y) is the floor-projected centre. We also compute it from the
    # transformed bottom corners for robustness/consistency.
    R_pf_arr = np.asarray(R_pf, dtype=float)
    bottom_corners_floor = (
        R_pf_arr @ np.asarray(pallet_model.bottom_corners_m, dtype=float).T
    ).T + t_pf
    centre_x, centre_y = floor_projected_centre(bottom_corners_floor.tolist())

    # Orientation about +Z from the pallet long axis.
    orientation_deg = pallet_orientation_deg(R_pf_arr)

    # Orientation modulo symmetry (R9.7): when the visible geometry is symmetric
    # (symmetry_order > 1) orientation is only determinable modulo that symmetry,
    # so report the symmetry-reduced representative and flag it.
    orientation_symmetry_reduced = symmetry_order > 1
    if orientation_symmetry_reduced:
        orientation_deg = reduce_orientation_modulo_symmetry(
            orientation_deg, symmetry_order
        )

    # Face identity from the orientation/symmetry configuration. Under a
    # symmetry the face cannot be uniquely resolved: face_identity is null, the
    # admissible faces are recorded, and FACE_AMBIGUOUS_SYMMETRY is flagged —
    # never a silent pick (R9.6, R9.7).
    face_result = derive_face_identity(orientation_deg, symmetry_order=symmetry_order)
    face_identity = (
        face_result.face_identity.value
        if face_result.face_identity is not None
        else None
    )

    # Range: distance from the camera centre (Floor_Frame) to the pallet centre.
    camera_centre_floor = np.asarray(calibration.translation, dtype=float).reshape(3)
    pallet_centre_floor = np.array([centre_x, centre_y, float(t_pf[2])])
    range_m = float(np.linalg.norm(pallet_centre_floor - camera_centre_floor))

    # Viewing angle: angle between the camera->pallet line of sight and the
    # floor plane (0 deg = horizontal grazing, 90 deg = straight down).
    los = pallet_centre_floor - camera_centre_floor
    horizontal = float(math.hypot(los[0], los[1]))
    viewing_angle_deg = float(rad_to_deg(math.atan2(abs(float(los[2])), horizontal)))

    # Step 4: ray-plane (Z = 0) cross-check of the PnP floor centre.
    quality_flags: list[QualityFlag] = []
    cross_check_residual_m: Optional[float] = None
    try:
        floor_points = backproject_bottom_corners_to_floor(matches, calibration)
        rp_centre = floor_points[:, :2].mean(axis=0)
        cross_check_residual_m = float(
            math.hypot(rp_centre[0] - centre_x, rp_centre[1] - centre_y)
        )
    except PoseEstimationError:
        # A cross-check that cannot even be computed (e.g. degenerate rays) is
        # itself a sign the configuration is ill-posed; reject loudly.
        return _unavailable_pose(
            ReasonCode.ILL_CONDITIONED, [QualityFlag.ILL_CONDITIONED]
        )

    # Rejection (d): reprojection cross-check disagreement beyond the documented
    # bound — two independent geometric estimates disagree (R9.5).
    if cross_check_residual_m > MAX_CROSS_CHECK_RESIDUAL_M:
        return _unavailable_pose(
            ReasonCode.ILL_CONDITIONED, [QualityFlag.ILL_CONDITIONED]
        )

    provenance = _resolve_pose_provenance(calibration)

    # Assemble the admissible_solutions list. This holds two *distinct* kinds of
    # entry, disambiguated by "kind":
    #   - "ray_plane_cross_check": a labelled diagnostic entry (not a pose).
    #   - "pose_candidate": a geometrically admissible PnP solution.
    admissible_solutions: list[dict] = [
        {
            "kind": "ray_plane_cross_check",
            "residual_m": cross_check_residual_m,
            "note": (
                "floor-contact ray-plane centroid vs PnP floor centre; "
                "diagnostic cross-check (rejects beyond "
                f"{MAX_CROSS_CHECK_RESIDUAL_M} m)"
            ),
        }
    ]

    # Ambiguity representation (R9.6): if PnP returned more than one candidate
    # and a second candidate is geometrically admissible (its reprojection error
    # is within a documented factor of the best), record the admissible solution
    # set and flag POSE_AMBIGUOUS rather than silently picking one. Each pose
    # candidate carries its Floor_Frame position/orientation so a consumer sees
    # the full set of physically plausible poses.
    admissible_pose_candidates: list[dict] = []
    # A candidate is admissible if it explains the observations about as well as
    # the best solve: within a relative factor OR within an absolute pixel
    # tolerance of the best residual (the latter surfaces near-planar ambiguity
    # even when the best residual is ~0). It must also itself be an acceptable
    # solve (residual within the documented PnP bound).
    admissibility_bound = max(
        AMBIGUITY_REPROJECTION_RATIO * best_residual_px,
        best_residual_px + AMBIGUITY_ABS_TOL_PX,
    )
    for cand_R_pc, cand_t_pc, cand_residual in candidates:
        if cand_residual > admissibility_bound or cand_residual > MAX_PNP_RESIDUAL_PX:
            continue
        cand_R_pf, cand_t_pf = compose_pallet_in_floor(
            cand_R_pc, cand_t_pc, calibration
        )
        cand_R_pf_arr = np.asarray(cand_R_pf, dtype=float)
        cand_bottom_floor = (
            cand_R_pf_arr
            @ np.asarray(pallet_model.bottom_corners_m, dtype=float).T
        ).T + cand_t_pf
        cand_x, cand_y = floor_projected_centre(cand_bottom_floor.tolist())
        cand_orientation = pallet_orientation_deg(cand_R_pf_arr)
        if orientation_symmetry_reduced:
            cand_orientation = reduce_orientation_modulo_symmetry(
                cand_orientation, symmetry_order
            )
        admissible_pose_candidates.append(
            {
                "kind": "pose_candidate",
                "position_m": {"x": cand_x, "y": cand_y},
                "orientation_deg": cand_orientation,
                "reprojection_residual_px": cand_residual,
            }
        )

    is_ambiguous = len(admissible_pose_candidates) > 1
    if is_ambiguous:
        # Record the full admissible set (never a silent pick) and flag it. The
        # reported point estimate remains the best-residual candidate, but the
        # POSE_AMBIGUOUS flag + admissible set make the ambiguity explicit.
        admissible_solutions.extend(admissible_pose_candidates)
        quality_flags.append(QualityFlag.POSE_AMBIGUOUS)

    if face_result.is_ambiguous:
        quality_flags.append(QualityFlag.FACE_AMBIGUOUS_SYMMETRY)

    return PoseResult(
        pose_status="available",
        reason_code=None,
        position_m=PositionM(x=centre_x, y=centre_y),
        orientation_deg=orientation_deg,
        orientation_symmetry_reduced=orientation_symmetry_reduced,
        face_identity=face_identity,
        admissible_solutions=admissible_solutions,
        uncertainty=None,  # task 6.8 fills real Monte Carlo uncertainty
        range_m=range_m,
        viewing_angle_deg=viewing_angle_deg,
        quality_flags=quality_flags,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Monte Carlo uncertainty propagation (task 6.8, R13, R27.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseNoiseModel:
    """Documented input-noise distributions for Monte Carlo pose propagation.

    Each field is a **1-sigma** parameter of an assumed distribution; the
    defaults come from the module-level ``MC_*`` constants. The noise model is a
    modelling *assumption* (not an empirically calibrated error budget), so the
    resulting uncertainty is labelled ``ESTIMATED`` (R13.2, R26.2). The fields
    are, in order of the four documented noise sources:

    - **Keypoint localisation** (``keypoint_sigma_px``): independent Gaussian
      pixel noise on each observed keypoint ``(u, v)``.
    - **Calibration intrinsics** (``focal_rel_sigma``,
      ``principal_point_sigma_px``): relative focal-length noise and pixel noise
      on the principal point.
    - **Calibration extrinsics** (``extrinsic_rot_sigma_deg``,
      ``extrinsic_trans_sigma_m``): small-angle rotation noise about a random
      axis and per-component translation noise on the camera->floor transform.
    - **Pallet dimensions** (``dim_sigma_m``): isotropic Gaussian noise on the
      nominal pallet length/width/deck-height, defaulting to the pallet model's
      own ``dim_uncertainty_m`` when available (R13.3).
    """

    keypoint_sigma_px: float = MC_KEYPOINT_SIGMA_PX
    focal_rel_sigma: float = MC_FOCAL_REL_SIGMA
    principal_point_sigma_px: float = MC_PRINCIPAL_POINT_SIGMA_PX
    extrinsic_rot_sigma_deg: float = MC_EXTRINSIC_ROT_SIGMA_DEG
    extrinsic_trans_sigma_m: float = MC_EXTRINSIC_TRANS_SIGMA_M
    dim_sigma_m: float = MC_DEFAULT_DIM_SIGMA_M

    def assumptions(self) -> list[str]:
        """Return a human-readable list of the documented noise assumptions.

        This is attached verbatim to the ``PoseUncertainty.assumptions`` field so
        the reported interval is self-documenting (R13.2, R13.3).
        """
        return [
            "method: Monte Carlo propagation of input noise through the "
            "known-geometry PnP solve; position summarised as a 2x2 covariance "
            "(m^2) and orientation as a circular standard deviation (deg).",
            "distributions are modelled assumptions, not empirically calibrated "
            "against measured ground truth; the interval is therefore "
            "'estimated'/assumption-based, never 'measured' (R13.2, R26.2).",
            "input noise sources assumed independent.",
            f"keypoint localisation: i.i.d. Gaussian pixel noise, "
            f"sigma = {self.keypoint_sigma_px} px on each (u, v).",
            f"calibration intrinsics: relative Gaussian focal-length noise "
            f"sigma_f = {self.focal_rel_sigma} * f; Gaussian principal-point "
            f"noise sigma = {self.principal_point_sigma_px} px.",
            f"calibration extrinsics: small-angle rotation about a random axis "
            f"with sigma = {self.extrinsic_rot_sigma_deg} deg; per-component "
            f"translation Gaussian noise sigma = {self.extrinsic_trans_sigma_m} m.",
            f"pallet dimensions: isotropic Gaussian noise sigma = "
            f"{self.dim_sigma_m} m on length/width/deck-height (R13.3).",
        ]


def _small_angle_rotation(sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Return a random 3x3 rotation drawn from a small-angle distribution.

    A rotation axis is drawn uniformly on the unit sphere and the rotation angle
    is Gaussian with the given 1-sigma (degrees). Uses Rodrigues' formula.
    """
    if sigma_deg <= 0.0:
        return np.eye(3)
    axis = rng.normal(size=3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    angle = math.radians(float(rng.normal(0.0, sigma_deg)))
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _perturb_calibration(
    calibration: Calibration,
    noise: PoseNoiseModel,
    rng: np.random.Generator,
) -> Calibration:
    """Return a calibration with intrinsics/extrinsics perturbed per the model."""
    K = np.array(calibration.intrinsics, dtype=float, copy=True)
    K[0, 0] *= 1.0 + float(rng.normal(0.0, noise.focal_rel_sigma))
    K[1, 1] *= 1.0 + float(rng.normal(0.0, noise.focal_rel_sigma))
    K[0, 2] += float(rng.normal(0.0, noise.principal_point_sigma_px))
    K[1, 2] += float(rng.normal(0.0, noise.principal_point_sigma_px))

    dR = _small_angle_rotation(noise.extrinsic_rot_sigma_deg, rng)
    R = dR @ np.asarray(calibration.rotation, dtype=float)
    t = np.asarray(calibration.translation, dtype=float).reshape(3) + rng.normal(
        0.0, noise.extrinsic_trans_sigma_m, size=3
    )

    return Calibration(
        id=calibration.id,
        intrinsics=K,
        distortion=np.array(calibration.distortion, dtype=float, copy=True),
        rotation=R,
        translation=t,
        reprojection_error=calibration.reprojection_error,
        provenance=calibration.provenance,
        image_size=calibration.image_size,
        floor_origin_note=calibration.floor_origin_note,
        source=calibration.source,
    )


def _perturb_pallet_model(
    pallet_model: PalletModel,
    dim_sigma_m: float,
    rng: np.random.Generator,
) -> PalletModel:
    """Return a pallet model with length/width/deck-height jittered (R13.3).

    The corner arrays are rebuilt from the perturbed dimensions so the model
    stays a self-consistent rectangle (bottom on Z=0, top elevated).
    """
    from .pallet_model import build_nominal_pallet_model

    dims = pallet_model.dimensions_m
    if dim_sigma_m <= 0.0:
        return pallet_model
    length = max(1e-3, float(dims.length) + float(rng.normal(0.0, dim_sigma_m)))
    width = max(1e-3, float(dims.width) + float(rng.normal(0.0, dim_sigma_m)))
    deck = max(1e-3, float(dims.deck_height) + float(rng.normal(0.0, dim_sigma_m)))
    return build_nominal_pallet_model(
        length_m=length,
        width_m=width,
        deck_height_m=deck,
        dim_uncertainty_m=pallet_model.dim_uncertainty_m or dim_sigma_m,
        source=pallet_model.source,
        provenance=pallet_model.provenance,
    )


def _solve_xy_theta(
    keypoints: Sequence[Keypoint],
    calibration: Calibration,
    pallet_model: PalletModel,
    symmetry_order: int,
) -> Optional[tuple[float, float, float]]:
    """Solve one pose sample, returning ``(x, y, orientation_deg)`` or ``None``.

    A ``None`` return means this perturbed sample was not solvable (too few
    keypoints, degenerate, or PnP failure); the caller skips it. This reuses the
    exact same solve path (``_solve_pnp_candidates`` + ``compose_pallet_in_floor``)
    as :func:`estimate_pose` so the Monte Carlo samples are consistent with the
    reported point estimate.
    """
    try:
        matches = match_keypoints_to_model(keypoints, pallet_model)
        candidates = _solve_pnp_candidates(matches, calibration)
    except Exception:  # noqa: BLE001 - PoseEstimationError or cv2.error on a bad sample
        return None
    R_pc, t_pc, _ = candidates[0]
    R_pf, t_pf = compose_pallet_in_floor(R_pc, t_pc, calibration)
    R_pf_arr = np.asarray(R_pf, dtype=float)
    bottom_corners_floor = (
        R_pf_arr @ np.asarray(pallet_model.bottom_corners_m, dtype=float).T
    ).T + t_pf
    centre_x, centre_y = floor_projected_centre(bottom_corners_floor.tolist())
    orientation_deg = pallet_orientation_deg(R_pf_arr)
    if symmetry_order > 1:
        orientation_deg = reduce_orientation_modulo_symmetry(
            orientation_deg, symmetry_order
        )
    return float(centre_x), float(centre_y), float(orientation_deg)


def propagate_pose_uncertainty(
    keypoints: Sequence[Keypoint],
    calibration: Calibration,
    pallet_model: PalletModel,
    *,
    n_samples: int = MC_DEFAULT_SAMPLES,
    seed: int = 42,
    noise_model: Optional[PoseNoiseModel] = None,
    symmetry_order: int = 1,
) -> PoseUncertainty:
    """Monte Carlo propagation of input noise into pose uncertainty (R13.1-13.3).

    Draws ``n_samples`` perturbations of the keypoints, calibration, and pallet
    dimensions from the documented distributions in ``noise_model``, re-solves
    the pose per sample, and summarises the spread of the recovered Floor_Frame
    ``(x, y, orientation)``:

    - **position** as a 2x2 covariance in ``m^2`` (row-major), and
    - **orientation** as a circular standard deviation in degrees (orientation
      wraps, so :func:`~..geometry.frames.circular_std_deg` is used).

    Orientation samples are first aligned to a common circular-mean
    representative modulo the pallet's 180-degree long-axis ambiguity so the two
    equivalent solutions (``theta`` and ``theta + 180``) do not inflate the
    spread. The result is labelled :attr:`ProvenanceLabel.ESTIMATED` (never
    ``MEASURED`` — the noise magnitudes are modelling assumptions, R13.2, R26.2)
    and carries the documented assumptions list.

    Parameters
    ----------
    keypoints, calibration, pallet_model:
        The same inputs as :func:`estimate_pose`; the nominal (unperturbed) solve
        should be available for a meaningful uncertainty.
    n_samples:
        Number of Monte Carlo samples (>= 2).
    seed:
        Seed for the ``numpy`` RNG so the propagation is reproducible (R28.3).
    noise_model:
        The documented noise distributions; defaults to a :class:`PoseNoiseModel`
        whose ``dim_sigma_m`` is taken from ``pallet_model.dim_uncertainty_m``
        when available (R13.3).
    symmetry_order:
        Rotational symmetry order, matching :func:`estimate_pose`.

    Returns
    -------
    PoseUncertainty
        A quantified uncertainty with ``position_cov_m2`` (2x2), a non-negative
        ``orientation_std_deg``, ``method='monte_carlo'``, the documented
        ``assumptions``, and ``ESTIMATED`` provenance. When too few samples solve
        (heavily perturbed / near-degenerate input) the metrics degrade to
        ``None`` while still documenting the method and assumptions.

    Raises
    ------
    ValueError
        If ``n_samples < 2``.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2 for a covariance, got {n_samples}")

    if noise_model is None:
        dim_sigma = (
            float(pallet_model.dim_uncertainty_m)
            if pallet_model.dim_uncertainty_m is not None
            else MC_DEFAULT_DIM_SIGMA_M
        )
        noise_model = PoseNoiseModel(dim_sigma_m=dim_sigma)

    rng = np.random.default_rng(seed)
    base_img = {
        kp.name: (kp.u, kp.v)
        for kp in keypoints
        if kp.u is not None and kp.v is not None
    }

    xs: list[float] = []
    ys: list[float] = []
    thetas: list[float] = []
    for _ in range(n_samples):
        # Perturb the keypoints (independent Gaussian pixel noise).
        perturbed_kps: list[Keypoint] = []
        for kp in keypoints:
            if kp.u is None or kp.v is None:
                perturbed_kps.append(kp)
                continue
            du = float(rng.normal(0.0, noise_model.keypoint_sigma_px))
            dv = float(rng.normal(0.0, noise_model.keypoint_sigma_px))
            perturbed_kps.append(
                Keypoint(
                    name=kp.name,
                    u=float(kp.u) + du,
                    v=float(kp.v) + dv,
                    visibility=kp.visibility,
                    score=kp.score,
                )
            )
        perturbed_cal = _perturb_calibration(calibration, noise_model, rng)
        perturbed_model = _perturb_pallet_model(
            pallet_model, noise_model.dim_sigma_m, rng
        )
        solved = _solve_xy_theta(
            perturbed_kps, perturbed_cal, perturbed_model, symmetry_order
        )
        if solved is None:
            continue
        xs.append(solved[0])
        ys.append(solved[1])
        thetas.append(solved[2])

    assumptions = noise_model.assumptions()

    # If too few samples solved we cannot summarise a covariance; degrade the
    # metrics to null (still documenting method + assumptions) rather than
    # fabricating a spread.
    if len(xs) < 2:
        return PoseUncertainty(
            position_cov_m2=None,
            orientation_std_deg=None,
            method="monte_carlo",
            assumptions=assumptions,
            provenance=ProvenanceLabel.ESTIMATED,
        )

    positions = np.column_stack([np.asarray(xs), np.asarray(ys)])
    # ddof=1 sample covariance; row-major 2x2.
    cov = np.cov(positions, rowvar=False, ddof=1)
    cov_list = [[float(cov[0, 0]), float(cov[0, 1])], [float(cov[1, 0]), float(cov[1, 1])]]

    # Orientation: reduce the 180-degree long-axis ambiguity by folding every
    # sample to within +/-90 deg of the circular mean, then take the circular
    # std of the aligned samples.
    aligned = _align_orientation_samples(thetas)
    orientation_std = circular_std_deg(aligned)
    if not math.isfinite(orientation_std):
        orientation_std = None

    return PoseUncertainty(
        position_cov_m2=cov_list,
        orientation_std_deg=orientation_std,
        method="monte_carlo",
        assumptions=assumptions,
        provenance=ProvenanceLabel.ESTIMATED,
    )


def _align_orientation_samples(thetas: Sequence[float]) -> list[float]:
    """Fold orientation samples modulo the pallet's 180-degree long-axis flip.

    The pallet long axis is a line, not a ray, so ``theta`` and ``theta + 180``
    describe the same physical orientation. Left unaligned, a sample cluster
    straddling that flip would report a spuriously huge spread. We pick a
    reference (the circular mean over the doubled angles, halved) and rotate each
    sample by 180 deg when that brings it closer to the reference.
    """
    thetas = list(thetas)
    if len(thetas) == 0:
        return thetas
    # Circular mean on doubled angles collapses the 180-deg ambiguity; halving
    # gives an axis reference in (-90, 90].
    try:
        doubled_mean = circular_mean_deg([2.0 * t for t in thetas])
    except ValueError:
        doubled_mean = 0.0
    ref = doubled_mean / 2.0
    aligned: list[float] = []
    for t in thetas:
        d = wrap_to_180(t - ref)
        if d > 90.0:
            t = t - 180.0
        elif d <= -90.0:
            t = t + 180.0
        aligned.append(wrap_to_180(t))
    return aligned


def _position_std_from_cov(position_cov_m2: Optional[list[list[float]]]) -> Optional[float]:
    """Return the larger principal position std (metres) from a 2x2 covariance.

    Uses the largest eigenvalue of the covariance so the reported position std is
    the worst-case 1-sigma spread over any direction in the floor plane. Returns
    ``None`` when the covariance is unavailable.
    """
    if position_cov_m2 is None:
        return None
    cov = np.asarray(position_cov_m2, dtype=float)
    eigvals = np.linalg.eigvalsh(cov)
    max_var = float(np.max(eigvals))
    if max_var < 0.0:  # numerical guard
        max_var = 0.0
    return math.sqrt(max_var)


def estimate_pose_with_uncertainty(
    keypoints: Sequence[Keypoint],
    calibration: Calibration,
    pallet_model: PalletModel,
    *,
    symmetry_order: int = 1,
    n_samples: int = MC_DEFAULT_SAMPLES,
    seed: int = 42,
    noise_model: Optional[PoseNoiseModel] = None,
    max_position_std_m: float = MAX_POSITION_STD_M,
    max_orientation_std_deg: float = MAX_ORIENTATION_STD_DEG,
) -> PoseResult:
    """Estimate a pose *and* attach Monte Carlo uncertainty, degrading loudly.

    This is the uncertainty-aware entry point layered on :func:`estimate_pose`
    (task 6.8). It first runs the deterministic estimator; if that already
    degraded to ``unavailable`` (too few keypoints, ill-conditioned, ...), the
    unavailable result is returned unchanged. Otherwise it runs
    :func:`propagate_pose_uncertainty` and:

    - attaches the resulting :class:`PoseUncertainty` (``ESTIMATED`` provenance,
      documented assumptions) to the available pose (R13.1, R27.1); and
    - **degrades to a defined ``unavailable`` result** with
      ``POSE_UNCERTAINTY_EXCEEDED`` + a matching quality flag when the propagated
      position std exceeds ``max_position_std_m`` or the orientation std exceeds
      ``max_orientation_std_deg`` — never a confidently-wrong point estimate
      (R27.2, R27.3, design Error Handling table).

    The default estimator :func:`estimate_pose` is intentionally left unchanged
    (``uncertainty=None``); callers opt in to the more expensive Monte Carlo path
    through this function.

    Parameters
    ----------
    keypoints, calibration, pallet_model, symmetry_order:
        As for :func:`estimate_pose`.
    n_samples, seed, noise_model:
        Monte Carlo controls (see :func:`propagate_pose_uncertainty`). ``seed``
        makes the propagation reproducible (R28.3).
    max_position_std_m, max_orientation_std_deg:
        Documented degrade thresholds (default to the module constants pinned to
        ``configs/verdict.yaml``); exceeding either degrades the pose to
        unavailable (R27.2).

    Returns
    -------
    PoseResult
        An ``available`` pose carrying a real :class:`PoseUncertainty`, or a
        defined ``unavailable`` result when the pose could not be solved or its
        propagated uncertainty exceeds the documented thresholds.
    """
    base = estimate_pose(
        keypoints, calibration, pallet_model, symmetry_order=symmetry_order
    )
    if base.pose_status != "available":
        return base

    uncertainty = propagate_pose_uncertainty(
        keypoints,
        calibration,
        pallet_model,
        n_samples=n_samples,
        seed=seed,
        noise_model=noise_model,
        symmetry_order=symmetry_order,
    )

    position_std = _position_std_from_cov(uncertainty.position_cov_m2)
    orientation_std = uncertainty.orientation_std_deg

    # Degrade-loudly (R27.2): uncertainty unquantifiable (metrics degraded to
    # null) or beyond the documented bounds -> defined unavailable pose.
    exceeded = False
    if position_std is None or orientation_std is None:
        exceeded = True
    elif position_std > max_position_std_m or orientation_std > max_orientation_std_deg:
        exceeded = True

    if exceeded:
        return _unavailable_pose(
            ReasonCode.POSE_UNCERTAINTY_EXCEEDED,
            [QualityFlag.POSE_UNCERTAINTY_EXCEEDED],
        )

    # Attach the quantified uncertainty to the available pose. Reconstruct the
    # PoseResult via model_copy so all other fields (position, orientation,
    # admissible set, flags, provenance) are preserved.
    return base.model_copy(update={"uncertainty": uncertainty})
