"""Self-constructed pose ground truth and evaluation harness (task 7.1, R10).

The assignment provides **no** pose ground truth (R10.4), so pose accuracy
claims can only be credible if the ground truth is *self-constructed* with a
justified, transparent method (R10.1, R10.2). This module does exactly that and
nothing more: it builds pose ground truth by a **synthetic forward render** and
provides a reproducible harness that pairs each synthetic GT with the estimator
under test so downstream error distributions (task 7.2) can be computed.

Method: synthetic render with known projection labels (R10.2)
-------------------------------------------------------------
Ground truth is constructed by placing the known 3D pallet model at a *chosen*
Floor_Frame pose ``(x, y, theta)`` and projecting its true 3D corners (bottom on
``Z = 0``, top elevated by the deck height) through a calibration with
``cv2.projectPoints`` — the classic forward camera model. The resulting pixel
keypoints are the *known projection labels*, and the chosen ``(x, y, theta)`` is
the ground-truth pose. This is the same forward model the pose unit tests use to
pin the round-trip; factoring it here lets tests and the evaluation harness reuse
one authoritative implementation rather than importing from the test suite.

Crucially, the ground truth pose comes **only** from the pose we chose and fed
into the forward projection — it is *never* read back from
:func:`~pallet_pose_compliance.geometry.pose.estimate_pose` or any estimate of
the estimator under test (R10.2, R10.3). The forward model (place + project) and
the inverse model (``estimate_pose``: keypoints -> pose) are independent code
paths, so evaluating one against the other is a genuine test, not a tautology.

Provenance discipline (R10.3, R26)
----------------------------------
Every synthetic GT sample carries provenance
:attr:`~pallet_pose_compliance.output.provenance.ProvenanceLabel.SIMULATED` — it
is a rendered pose, never a real-world measurement, and is never relabelled
``measured`` downstream (R26.2). An **optional** measured floor-point GT path is
provided (:class:`MeasuredFloorPointGT`, labelled ``MEASURED``) so that, if real
hardware measurements are collected later (the separate stretch task 7.6), they
can be supplied and reported *separately*. This module never fabricates measured
data: when no measured GT is supplied it is simply absent (defaults to ``None``),
which is the honest "real-world compliance unverified" posture (R10.4, R26.4).

Honesty note (R10.4)
--------------------
Nothing here is presented as assignment-provided ground truth. The ground truth
is self-constructed by this system's own forward model and is labelled
``SIMULATED`` at every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..calibration.calibration import Calibration, build_synthetic_calibration
from ..output.provenance import ProvenanceLabel
from ..output.schema import Keypoint, PalletModel, PoseResult
from .frames import reduce_orientation_modulo_symmetry, wrap_to_180
from .pallet_model import build_nominal_pallet_model
from .pose import (
    BOTTOM_CORNER_NAMES,
    TOP_CORNER_NAMES,
    estimate_pose,
    estimate_pose_with_uncertainty,
)

__all__ = [
    "DEFAULT_EVAL_SEED",
    "rotation_about_z",
    "world_to_camera_rt",
    "place_and_project",
    "GroundTruthPose",
    "MeasuredFloorPointGT",
    "orientation_error_deg",
    "PoseEvalSample",
    "generate_simulated_gt_samples",
    "run_estimator_on_gt",
    "evaluate_samples",
]

#: Default seed for the evaluation harness, pinned to ``configs/pipeline.yaml``
#: (``seed: 42``) so self-constructed evaluation runs are reproducible (R28.3).
DEFAULT_EVAL_SEED = 42


# ---------------------------------------------------------------------------
# Forward model: synthetic render with known projection labels (R10.2)
# ---------------------------------------------------------------------------


def rotation_about_z(theta_deg: float) -> np.ndarray:
    """Rotation matrix about Floor_Frame ``+Z`` by ``theta_deg`` (CCW from above).

    Matches the orientation convention pinned in
    :mod:`~pallet_pose_compliance.geometry.frames` (``theta = 0`` when the long
    axis is along ``+X``; positive is counter-clockwise viewed from ``+Z``).
    """
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def world_to_camera_rt(calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rvec, tvec)`` mapping Floor_Frame -> camera for ``projectPoints``.

    The calibration stores camera -> Floor_Frame (``X_floor = R_cf @ X_cam +
    t_cf``); the inverse floor -> camera pose used by ``cv2.projectPoints`` is
    ``R_wc = R_cf^T`` and ``t_wc = -R_cf^T @ t_cf``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(rvec, tvec)`` — a Rodrigues 3-vector and a ``(3, 1)`` translation.
    """
    import cv2

    R_cf = np.asarray(calibration.rotation, dtype=float)
    t_cf = np.asarray(calibration.translation, dtype=float).reshape(3)
    R_wc = R_cf.T
    t_wc = -R_cf.T @ t_cf
    rvec, _ = cv2.Rodrigues(R_wc)
    return rvec, t_wc.reshape(3, 1)


def place_and_project(
    calibration: Calibration,
    pallet_model: PalletModel,
    x0: float,
    y0: float,
    theta_deg: float,
) -> tuple[list[Keypoint], np.ndarray]:
    """Place the pallet at ``(x0, y0, theta)`` and project its corners to pixels.

    This is the synthetic **forward model** used to self-construct pose ground
    truth (R10.2): the pallet model is placed at the chosen Floor_Frame pose,
    its true 3D corners (four bottom on ``Z = 0``, four top elevated by the deck
    height) are transformed into Floor_Frame and projected through the
    calibration with ``cv2.projectPoints`` (respecting keypoint height above the
    floor and the lens-distortion model). The returned pixel coordinates are the
    *known projection labels*; the chosen ``(x0, y0, theta_deg)`` is the ground
    truth pose. It is emphatically independent of the estimator under test.

    Parameters
    ----------
    calibration:
        The camera calibration used to render (intrinsics, distortion,
        camera->floor extrinsics).
    pallet_model:
        The known 3D pallet model to place and render.
    x0, y0:
        The ground-truth floor-projected pallet centre in metres (Floor_Frame).
    theta_deg:
        The ground-truth orientation in degrees (``theta = 0`` = long axis along
        Floor_Frame ``+X``; positive CCW from ``+Z``).

    Returns
    -------
    tuple[list[Keypoint], np.ndarray]
        A list of eight :class:`Keypoint` (four bottom + four top, all marked
        ``visible``) carrying the projected pixel coordinates, and the ``(8, 3)``
        Floor_Frame corner coordinates (bottom rows first) for cross-checking.
    """
    import cv2

    R_pf = rotation_about_z(theta_deg)
    t_pf = np.array([float(x0), float(y0), 0.0], dtype=float)

    bottom = np.asarray(pallet_model.bottom_corners_m, dtype=float)
    top = np.asarray(pallet_model.top_corners_m, dtype=float)
    corners_local = np.vstack([bottom, top])  # (8, 3) pallet-local
    names = list(BOTTOM_CORNER_NAMES) + list(TOP_CORNER_NAMES)

    # Pallet-local -> Floor_Frame.
    corners_floor = (R_pf @ corners_local.T).T + t_pf

    rvec, tvec = world_to_camera_rt(calibration)
    img, _ = cv2.projectPoints(
        corners_floor.astype(np.float64),
        rvec,
        tvec,
        np.asarray(calibration.intrinsics, dtype=np.float64),
        np.asarray(calibration.distortion, dtype=np.float64).reshape(-1, 1),
    )
    img = img.reshape(-1, 2)

    keypoints = [
        Keypoint(
            name=name,
            u=float(px[0]),
            v=float(px[1]),
            visibility="visible",
            score=0.9,
        )
        for name, px in zip(names, img)
    ]
    return keypoints, corners_floor


# ---------------------------------------------------------------------------
# Ground-truth data models (R10.3 provenance discipline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthPose:
    """A self-constructed (SIMULATED) pose ground-truth sample (R10.1-R10.3).

    Carries the *known* Floor_Frame pose ``(x, y, theta)`` we chose and the
    keypoints the forward model produced by projecting the pallet model at that
    pose. The provenance is fixed to
    :attr:`~pallet_pose_compliance.output.provenance.ProvenanceLabel.SIMULATED`:
    this is a rendered pose, never a real-world measurement, and is never
    presented as assignment-provided ground truth (R10.4).

    The keypoints and the pose come *only* from the forward projection model —
    they are independent of :func:`estimate_pose` / the estimator under test, so
    comparing an estimate against this GT is a genuine test (R10.2).

    Attributes
    ----------
    x_m, y_m:
        Ground-truth floor-projected pallet centre (metres, Floor_Frame).
    theta_deg:
        Ground-truth orientation (degrees), wrapped to ``(-180, 180]``.
    keypoints:
        The rendered known projection labels (eight visible corners).
    symmetry_order:
        Rotational symmetry order of the pallet geometry the sample was rendered
        for. Used when comparing orientation error modulo the symmetry (R9.7).
    provenance:
        Always :attr:`ProvenanceLabel.SIMULATED` (R10.3); construction rejects
        any other label so a rendered GT can never masquerade as measured.
    source:
        Human-readable note documenting the self-constructed method (R10.2).
    """

    x_m: float
    y_m: float
    theta_deg: float
    keypoints: tuple[Keypoint, ...]
    symmetry_order: int = 1
    provenance: ProvenanceLabel = ProvenanceLabel.SIMULATED
    source: str = (
        "self-constructed synthetic render: pallet model placed at a known "
        "Floor_Frame pose and projected via cv2.projectPoints (known projection "
        "labels); NOT assignment-provided, NOT derived from the estimator"
    )

    def __post_init__(self) -> None:
        if self.provenance is not ProvenanceLabel.SIMULATED:
            raise ValueError(
                "a synthetic-render GroundTruthPose must be labelled SIMULATED "
                f"(got {self.provenance!r}); rendered GT is never measured (R26.2)"
            )
        if not isinstance(self.symmetry_order, int) or self.symmetry_order < 1:
            raise ValueError(
                f"symmetry_order must be a positive integer, got "
                f"{self.symmetry_order!r}"
            )
        if len(self.keypoints) == 0:
            raise ValueError("a GroundTruthPose must carry rendered keypoints")
        # Pin the orientation to the canonical wrap range so downstream error
        # math is unambiguous.
        object.__setattr__(self, "theta_deg", wrap_to_180(self.theta_deg))


@dataclass(frozen=True)
class MeasuredFloorPointGT:
    """Optional, real *measured* floor-point ground truth, reported separately.

    This is the honest entry point for real-world ground truth collected on
    hardware (the separate stretch task 7.6). It is labelled
    :attr:`~pallet_pose_compliance.output.provenance.ProvenanceLabel.MEASURED`
    because — and only because — the points are actual physical measurements.
    This module **never fabricates** such data: the harness defaults to *no*
    measured GT (``None``), which is the honest "real-world compliance
    unverified" posture (R10.4, R26.4). The type exists so measured GT can be
    supplied later without changing the harness API.

    Attributes
    ----------
    label:
        A human-readable identifier for the measurement session/point set.
    floor_points_m:
        ``(N, 3)`` surveyed floor points in Floor_Frame (metres). Real measured
        coordinates only.
    pose:
        Optional measured pose ``(x, y, theta)`` in metres/degrees, when a full
        pose (not just floor points) was physically measured.
    measurement_method:
        Documented how the points/pose were physically measured (e.g. tape/laser
        survey), for the report.
    provenance:
        Always :attr:`ProvenanceLabel.MEASURED`; construction rejects any other
        label so this path cannot be used to smuggle in non-measured data.
    """

    label: str
    floor_points_m: np.ndarray
    pose: Optional[tuple[float, float, float]] = None
    measurement_method: str = ""
    provenance: ProvenanceLabel = ProvenanceLabel.MEASURED

    def __post_init__(self) -> None:
        if self.provenance is not ProvenanceLabel.MEASURED:
            raise ValueError(
                "MeasuredFloorPointGT must be labelled MEASURED (got "
                f"{self.provenance!r}); this path is for real measured data only"
            )
        pts = np.asarray(self.floor_points_m, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] != 3:
            raise ValueError(
                "floor_points_m must be a non-empty (N, 3) array of Floor_Frame "
                f"points, got shape {pts.shape}"
            )
        if not np.all(np.isfinite(pts)):
            raise ValueError("floor_points_m must contain only finite values")
        object.__setattr__(self, "floor_points_m", pts)


# ---------------------------------------------------------------------------
# Orientation error modulo symmetry + long-axis (180-degree) ambiguity
# ---------------------------------------------------------------------------


def orientation_error_deg(
    predicted_deg: float,
    ground_truth_deg: float,
    *,
    symmetry_order: int = 1,
    long_axis_ambiguity: bool = True,
) -> float:
    """Absolute orientation error (degrees) accounting for pallet symmetry.

    A rectangular pallet's *long axis* is a line, not a ray: the pose that puts
    the long axis at ``theta`` is geometrically indistinguishable from the one at
    ``theta + 180`` (the classic 180-degree long-axis ambiguity). Comparing a raw
    signed difference would therefore over-report error whenever the estimator
    picks the equally-valid flipped solution. This helper compares the predicted
    and ground-truth orientations *modulo* the pallet's symmetry so the reported
    error reflects genuine disagreement, not an admissible ambiguity (R9.7).

    The comparison reduces both angles modulo the effective symmetry sector and
    returns the smallest wrapped magnitude:

    - ``symmetry_order`` folds by ``360 / symmetry_order`` (the pallet's stated
      rotational symmetry).
    - ``long_axis_ambiguity`` additionally folds by ``180`` (the long-axis
      line ambiguity), which is the common case for a rectangular pallet.

    The two effects combine multiplicatively into an effective symmetry order of
    ``lcm``-like folding; in practice we fold by the *finest* applicable sector,
    i.e. the larger of ``symmetry_order`` and (when the long-axis ambiguity
    applies) ``2``.

    Parameters
    ----------
    predicted_deg:
        The estimator's reported orientation in degrees.
    ground_truth_deg:
        The self-constructed ground-truth orientation in degrees.
    symmetry_order:
        The pallet's rotational symmetry order (``1`` = asymmetric).
    long_axis_ambiguity:
        When ``True`` (default) also fold by the 180-degree long-axis ambiguity,
        appropriate for a rectangular pallet whose long axis is a line.

    Returns
    -------
    float
        The absolute orientation error in degrees, in ``[0, sector/2]`` where
        ``sector`` is the effective symmetry sector.

    Raises
    ------
    ValueError
        If ``symmetry_order`` is not a positive integer.
    """
    if not isinstance(symmetry_order, int) or symmetry_order < 1:
        raise ValueError(
            f"symmetry_order must be a positive integer, got {symmetry_order!r}"
        )

    effective_order = symmetry_order
    if long_axis_ambiguity:
        # The long-axis line ambiguity is a 180-degree (order-2) symmetry; fold
        # by the finer of the two so both effects are respected.
        effective_order = max(symmetry_order, 2)

    pred = reduce_orientation_modulo_symmetry(predicted_deg, effective_order)
    gt = reduce_orientation_modulo_symmetry(ground_truth_deg, effective_order)
    sector = 360.0 / effective_order
    diff = pred - gt
    # Reduce the difference into the symmetry sector, then take the magnitude of
    # the shortest representative within (-sector/2, sector/2].
    diff = math.remainder(diff, sector)
    return abs(diff)


# ---------------------------------------------------------------------------
# Evaluation harness (R10.1): generate GT, run estimator, pair the two
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseEvalSample:
    """A ground-truth sample paired with the estimator's prediction (R10.1).

    Holds the self-constructed :class:`GroundTruthPose` and the
    :class:`PoseResult` the estimator produced from that GT's keypoints. This is
    the unit downstream error-distribution reporting (task 7.2) consumes: it can
    compute translation/rotation errors from ``(ground_truth, prediction)`` pairs
    and evaluate them against the ±2 cm / ±3 deg tolerance.

    Attributes
    ----------
    ground_truth:
        The self-constructed, SIMULATED ground-truth pose.
    prediction:
        The estimator's :class:`PoseResult` (``available`` or a defined
        ``unavailable`` result — the harness does not hide unavailable poses).
    """

    ground_truth: GroundTruthPose
    prediction: PoseResult


def _default_pose_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    theta_range: tuple[float, float],
    n_samples: int,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    """Draw ``n_samples`` random ``(x, y, theta)`` poses from the given ranges."""
    xs = rng.uniform(x_range[0], x_range[1], size=n_samples)
    ys = rng.uniform(y_range[0], y_range[1], size=n_samples)
    thetas = rng.uniform(theta_range[0], theta_range[1], size=n_samples)
    return [
        (float(x), float(y), float(theta))
        for x, y, theta in zip(xs, ys, thetas)
    ]


def generate_simulated_gt_samples(
    n_samples: int,
    *,
    calibration: Optional[Calibration] = None,
    pallet_model: Optional[PalletModel] = None,
    seed: int = DEFAULT_EVAL_SEED,
    x_range: tuple[float, float] = (-0.6, 0.6),
    y_range: tuple[float, float] = (2.5, 5.0),
    theta_range: tuple[float, float] = (-180.0, 180.0),
    symmetry_order: int = 1,
) -> list[GroundTruthPose]:
    """Generate ``n_samples`` reproducible SIMULATED pose GT samples (R10.1-R10.3).

    Draws ``n_samples`` random Floor_Frame poses from the documented ranges using
    a seeded NumPy generator (``seed`` defaults to the pipeline seed 42), renders
    each via the :func:`place_and_project` forward model, and wraps the result in
    a :class:`GroundTruthPose` labelled ``SIMULATED``. Because the RNG is seeded,
    repeated calls with the same arguments produce identical samples (R28.3).

    The ground truth is entirely self-constructed by the forward model and never
    read back from the estimator under test (R10.2, R10.3), and it is never
    presented as assignment-provided (R10.4).

    Parameters
    ----------
    n_samples:
        Number of GT samples to generate (must be >= 1).
    calibration:
        Calibration used to render. Defaults to a SIMULATED synthetic camera.
    pallet_model:
        Pallet model to place. Defaults to the nominal pallet model.
    seed:
        Seed for the reproducible RNG (R28.3).
    x_range, y_range, theta_range:
        Sampling ranges for the pose. ``y_range`` is kept in front of the camera
        (positive depth) so the pallet is visible; ``theta_range`` spans a full
        turn by default.
    symmetry_order:
        Symmetry order recorded on each sample for orientation-error comparison.

    Returns
    -------
    list[GroundTruthPose]
        ``n_samples`` self-constructed, SIMULATED ground-truth poses.

    Raises
    ------
    ValueError
        If ``n_samples`` is not a positive integer.
    """
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")

    cal = calibration if calibration is not None else build_synthetic_calibration(
        "pose-eval-sim-cal"
    )
    model = pallet_model if pallet_model is not None else build_nominal_pallet_model()

    rng = np.random.default_rng(seed)
    poses = _default_pose_grid(x_range, y_range, theta_range, n_samples, rng)

    samples: list[GroundTruthPose] = []
    for x0, y0, theta_deg in poses:
        keypoints, _ = place_and_project(cal, model, x0, y0, theta_deg)
        samples.append(
            GroundTruthPose(
                x_m=x0,
                y_m=y0,
                theta_deg=theta_deg,
                keypoints=tuple(keypoints),
                symmetry_order=symmetry_order,
            )
        )
    return samples


def run_estimator_on_gt(
    ground_truth: GroundTruthPose,
    calibration: Calibration,
    pallet_model: PalletModel,
    *,
    with_uncertainty: bool = False,
    estimator: Optional[Callable[..., PoseResult]] = None,
    **estimator_kwargs,
) -> PoseResult:
    """Run the estimator under test on a GT sample's rendered keypoints (R10.1).

    Feeds the ground truth's *known projection labels* (its rendered keypoints)
    to the estimator and returns the :class:`PoseResult`. The estimator is the
    inverse model (keypoints -> pose); the GT came from the independent forward
    model, so this is a genuine evaluation, not a round-trip of the estimator's
    own output (R10.2).

    Parameters
    ----------
    ground_truth:
        The self-constructed GT sample whose keypoints drive the estimator.
    calibration, pallet_model:
        The calibration and pallet model handed to the estimator.
    with_uncertainty:
        When ``True`` use :func:`estimate_pose_with_uncertainty` (Monte Carlo);
        otherwise the deterministic :func:`estimate_pose`.
    estimator:
        Optional explicit estimator callable (advanced/testing use). Defaults to
        the module's ``estimate_pose`` / ``estimate_pose_with_uncertainty``.
    **estimator_kwargs:
        Extra keyword arguments forwarded to the estimator (e.g. ``seed``,
        ``n_samples`` for the uncertainty path).

    Returns
    -------
    PoseResult
        The estimator's prediction for this GT sample.
    """
    if estimator is None:
        estimator = (
            estimate_pose_with_uncertainty if with_uncertainty else estimate_pose
        )
    return estimator(
        list(ground_truth.keypoints),
        calibration,
        pallet_model,
        symmetry_order=ground_truth.symmetry_order,
        **estimator_kwargs,
    )


def evaluate_samples(
    n_samples: int,
    *,
    calibration: Optional[Calibration] = None,
    pallet_model: Optional[PalletModel] = None,
    seed: int = DEFAULT_EVAL_SEED,
    with_uncertainty: bool = False,
    measured_gt: Optional[MeasuredFloorPointGT] = None,
    **gt_kwargs,
) -> list[PoseEvalSample]:
    """Run the full self-constructed pose evaluation harness (R10.1).

    Generates ``n_samples`` reproducible SIMULATED ground-truth poses, runs the
    estimator under test on each GT's rendered keypoints, and pairs each
    prediction with its ground truth in a :class:`PoseEvalSample`. The returned
    list is the input to downstream translation/rotation error-distribution and
    tolerance reporting (task 7.2).

    The ``measured_gt`` argument is the **optional** real-measured-GT hook
    (R10.4): it defaults to ``None`` (no measured data — honest "unverified"
    posture) and, when supplied, must be a :class:`MeasuredFloorPointGT`
    (labelled ``MEASURED``). This function does not fold measured GT into the
    simulated distribution; measured GT is validated here and reported
    separately by the caller so simulated and measured results are never
    conflated (R26.2).

    Parameters
    ----------
    n_samples:
        Number of simulated GT samples to evaluate.
    calibration, pallet_model:
        Calibration and pallet model shared by GT generation and the estimator.
        Default to a SIMULATED synthetic camera and the nominal pallet model.
    seed:
        Seed for reproducible GT generation (R28.3).
    with_uncertainty:
        Whether to run the Monte Carlo uncertainty-aware estimator.
    measured_gt:
        Optional real measured floor-point GT (defaults to ``None``). Validated
        to be a :class:`MeasuredFloorPointGT` when provided.
    **gt_kwargs:
        Extra keyword arguments forwarded to
        :func:`generate_simulated_gt_samples` (e.g. sampling ranges,
        ``symmetry_order``).

    Returns
    -------
    list[PoseEvalSample]
        ``n_samples`` ground-truth/prediction pairs.

    Raises
    ------
    TypeError
        If ``measured_gt`` is supplied but is not a :class:`MeasuredFloorPointGT`.
    """
    if measured_gt is not None and not isinstance(measured_gt, MeasuredFloorPointGT):
        raise TypeError(
            "measured_gt must be a MeasuredFloorPointGT (labelled MEASURED) or "
            f"None; got {type(measured_gt).__name__}"
        )

    cal = calibration if calibration is not None else build_synthetic_calibration(
        "pose-eval-sim-cal"
    )
    model = pallet_model if pallet_model is not None else build_nominal_pallet_model()

    ground_truths = generate_simulated_gt_samples(
        n_samples,
        calibration=cal,
        pallet_model=model,
        seed=seed,
        **gt_kwargs,
    )

    samples: list[PoseEvalSample] = []
    for gt in ground_truths:
        prediction = run_estimator_on_gt(
            gt, cal, model, with_uncertainty=with_uncertainty
        )
        samples.append(PoseEvalSample(ground_truth=gt, prediction=prediction))
    return samples
