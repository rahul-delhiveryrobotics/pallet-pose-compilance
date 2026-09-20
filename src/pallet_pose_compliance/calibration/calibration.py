"""Camera calibration loading, reprojection error, and provenance (R8).

This module owns the ``Calibrator`` component from the design: it loads a
per-id calibration artefact describing camera intrinsics, lens distortion, and
extrinsics (camera -> Floor_Frame), together with the calibration's
``reprojection_error`` and a :class:`ProvenanceLabel`. Every ``Assessment``
references the calibration ``id`` used to produce it (R8.4, R24.7, R28.2).

Design anchors (design: Calibrator)
------------------------------------
- Interface ``load_calibration(id) -> Calibration`` returning
  ``{intrinsics K, distortion, extrinsics R|t (camera->Floor_Frame),
  reprojection_error, provenance}`` with an id referenced by every Assessment.
- Reprojection error is reported with a provenance label (R8.2, R8.3).
- **Real path**: checkerboard intrinsics (``cv2.calibrateCamera``) + floor
  extrinsics (``cv2.solvePnP`` against known floor points) yields a *measured*
  reprojection error.
- **No-hardware path**: a synthetic/simulated calibration is clearly labelled
  ``simulated`` (design scope posture, R26.4) so it is never mistaken for a
  real-world measurement.

Honesty discipline (R26)
------------------------
- The synthetic path is labelled ``SIMULATED`` and its stated reprojection
  error is labelled ``SIMULATED`` — never ``MEASURED``.
- The real path attaches ``MEASURED`` only because the number is computed from
  actual image<->world correspondences.
- A missing artefact degrades honestly (raises :class:`CalibrationNotFoundError`)
  rather than fabricating a plausible calibration.

Coordinate convention
----------------------
Extrinsics map a point ``X_cam`` in the camera frame to Floor_Frame via
``X_floor = R @ X_cam + t`` where ``R`` is 3x3 and ``t`` is a 3-vector, and the
Floor_Frame is the one pinned in :mod:`..geometry.frames` (X camera-right,
Y depth away, Z up, floor at Z = 0, right-handed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

from ..output.provenance import ProvenanceLabel

__all__ = [
    "DEFAULT_CALIBRATION_DIR",
    "Calibration",
    "CalibrationError",
    "CalibrationNotFoundError",
    "compute_reprojection_error",
    "build_synthetic_calibration",
    "build_calibration_from_checkerboard_and_floor",
    "load_calibration",
]

#: Default directory holding per-id calibration artefacts (matches
#: ``paths.calibration_dir`` in ``configs/pipeline.yaml``).
DEFAULT_CALIBRATION_DIR = Path("configs/calibration")


class CalibrationError(ValueError):
    """Base error for calibration loading/validation failures."""


class CalibrationNotFoundError(CalibrationError, FileNotFoundError):
    """Raised when a calibration artefact for the requested id does not exist.

    Degrading honestly: rather than fabricating a plausible calibration when the
    artefact is missing, the loader raises this error so the caller can surface
    ``CALIBRATION_INVALID`` (R26.3, R26.4).
    """


@dataclass(frozen=True)
class Calibration:
    """A loaded calibration artefact set (design: Calibrator, R8).

    Attributes
    ----------
    id:
        The calibration identifier; every Assessment references it (R8.4).
    intrinsics:
        Camera intrinsic matrix ``K`` (3x3): ``[[fx, 0, cx], [0, fy, cy],
        [0, 0, 1]]`` in pixels.
    distortion:
        Lens distortion coefficients (OpenCV order ``[k1, k2, p1, p2, k3, ...]``).
    rotation:
        Extrinsic rotation ``R`` (3x3) mapping camera -> Floor_Frame.
    translation:
        Extrinsic translation ``t`` (3-vector, metres) mapping camera ->
        Floor_Frame, applied as ``X_floor = R @ X_cam + t``.
    reprojection_error:
        RMS reprojection error in pixels for the calibration.
    provenance:
        Provenance of the artefact (``MEASURED`` for a real checkerboard/floor
        calibration; ``SIMULATED`` for the synthetic no-hardware calibration).
    image_size:
        Optional ``(width, height)`` in pixels the intrinsics were solved at.
    floor_origin_note:
        Human-readable note recording the physical Floor_Frame origin (R7.4).
    source:
        Human-readable note describing how the artefact was produced.

    Invariants (validated at construction)
    --------------------------------------
    - ``intrinsics`` is 3x3, ``rotation`` is 3x3, ``translation`` is length-3.
    - ``reprojection_error`` is finite and non-negative.
    - ``provenance`` is a :class:`ProvenanceLabel` and is never ``UNAVAILABLE``
      for a *loaded* calibration (an unavailable calibration is represented by
      not returning a ``Calibration`` at all — the loader raises instead).
    """

    id: str
    intrinsics: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    reprojection_error: float
    provenance: ProvenanceLabel
    image_size: Optional[tuple[int, int]] = None
    floor_origin_note: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "intrinsics", _as_float_array(self.intrinsics))
        object.__setattr__(self, "distortion", _as_float_array(self.distortion).ravel())
        object.__setattr__(self, "rotation", _as_float_array(self.rotation))
        object.__setattr__(
            self, "translation", _as_float_array(self.translation).ravel()
        )

        if self.intrinsics.shape != (3, 3):
            raise CalibrationError(
                f"intrinsics K must be 3x3, got shape {self.intrinsics.shape}"
            )
        if self.rotation.shape != (3, 3):
            raise CalibrationError(
                f"extrinsic rotation R must be 3x3, got shape {self.rotation.shape}"
            )
        if self.translation.shape != (3,):
            raise CalibrationError(
                "extrinsic translation t must be a 3-vector, got shape "
                f"{self.translation.shape}"
            )
        if not np.all(np.isfinite(self.intrinsics)):
            raise CalibrationError("intrinsics K must contain only finite values")
        if not np.all(np.isfinite(self.rotation)) or not np.all(
            np.isfinite(self.translation)
        ):
            raise CalibrationError("extrinsics R|t must contain only finite values")
        if not np.all(np.isfinite(self.distortion)):
            raise CalibrationError("distortion coeffs must be finite")
        if not np.isfinite(self.reprojection_error) or self.reprojection_error < 0.0:
            raise CalibrationError(
                "reprojection_error must be finite and non-negative, got "
                f"{self.reprojection_error!r}"
            )
        if not isinstance(self.provenance, ProvenanceLabel):
            raise CalibrationError(
                f"provenance must be a ProvenanceLabel, got {self.provenance!r}"
            )
        if self.provenance is ProvenanceLabel.UNAVAILABLE:
            raise CalibrationError(
                "a loaded Calibration must not be labelled 'unavailable'; a "
                "missing calibration is surfaced by raising, not fabricating"
            )

    def to_dict(self) -> dict:
        """Serialise to a plain, YAML/JSON-friendly dict (lists, not ndarrays)."""
        return {
            "id": self.id,
            "provenance": self.provenance.value,
            "image_size": list(self.image_size) if self.image_size else None,
            "intrinsics": self.intrinsics.tolist(),
            "distortion": self.distortion.tolist(),
            "extrinsics": {
                "rotation": self.rotation.tolist(),
                "translation": self.translation.tolist(),
                "note": "maps camera -> Floor_Frame: X_floor = R @ X_cam + t",
            },
            "reprojection_error": {
                "value": self.reprojection_error,
                "unit": "pixels",
                "provenance": self.provenance.value,
            },
            "floor_origin_note": self.floor_origin_note,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------


def compute_reprojection_error(
    object_points: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """Return the RMS reprojection error (pixels) for a set of correspondences.

    Projects the 3D ``object_points`` through the camera model
    (``intrinsics``, ``distortion``, pose ``rvec``/``tvec``) and compares the
    result to the observed ``image_points``, returning the root-mean-square of
    the per-point Euclidean pixel residuals. This is the honest ``measured``
    number when computed from *actual* correspondences (R8.2).

    ``rvec``/``tvec`` follow the OpenCV convention (world -> camera): they are
    the pose used by ``cv2.projectPoints``. For a synthetic self-consistent set
    of correspondences the returned error is ~0; perturbing the image points
    increases it monotonically.

    Parameters
    ----------
    object_points:
        ``(N, 3)`` 3D points in the reference (world/floor) frame.
    image_points:
        ``(N, 2)`` observed pixel coordinates for the same points.
    intrinsics:
        Camera matrix ``K`` (3x3).
    distortion:
        Distortion coefficients (OpenCV order).
    rvec, tvec:
        Rotation (Rodrigues 3-vector) and translation (3-vector) of the world
        points in the camera frame (``cv2.projectPoints`` convention).

    Returns
    -------
    float
        RMS reprojection error in pixels (>= 0).

    Raises
    ------
    CalibrationError
        If the point arrays are empty or shape-mismatched.
    """
    import cv2

    obj = _as_float_array(object_points).reshape(-1, 3)
    img = _as_float_array(image_points).reshape(-1, 2)
    if obj.shape[0] == 0:
        raise CalibrationError("compute_reprojection_error needs >= 1 point")
    if obj.shape[0] != img.shape[0]:
        raise CalibrationError(
            "object_points and image_points must have the same length "
            f"({obj.shape[0]} vs {img.shape[0]})"
        )

    K = _as_float_array(intrinsics).reshape(3, 3)
    dist = _as_float_array(distortion).reshape(-1, 1)
    rvec = _as_float_array(rvec).reshape(3, 1)
    tvec = _as_float_array(tvec).reshape(3, 1)

    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    residuals = projected - img
    # RMS over all points and both axes.
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return rms


# ---------------------------------------------------------------------------
# Builders: synthetic (simulated) and real (checkerboard + floor)
# ---------------------------------------------------------------------------


def build_synthetic_calibration(
    calibration_id: str,
    *,
    image_size: tuple[int, int] = (1280, 720),
    horizontal_fov_deg: float = 70.0,
    camera_height_m: float = 1.2,
    tilt_down_deg: float = 20.0,
    stated_reprojection_error_px: float = 0.35,
    floor_origin_note: str = (
        "simulated Floor_Frame origin at the camera's ground-projected nadir; "
        "no physical fiducial (synthetic calibration)"
    ),
    source: str = (
        "synthetic/simulated calibration for the no-hardware case; intrinsics "
        "from a pinhole model at the stated FOV, extrinsics from the nominal "
        "~1.2 m height / ~20 deg down-tilt mounting"
    ),
) -> Calibration:
    """Build a clearly-labelled ``SIMULATED`` calibration for the no-hardware case.

    When there is no physical camera access, the system still needs a resolvable
    calibration id so the metric pipeline can run and Assessments can reference
    it (design scope posture, R26.4). This constructs a plausible pinhole
    intrinsics matrix for the given field of view and image size, zero lens
    distortion, and extrinsics for a camera mounted ``camera_height_m`` above the
    floor tilted ``tilt_down_deg`` below horizontal. The result is labelled
    :attr:`ProvenanceLabel.SIMULATED` and carries a *stated* (simulated)
    reprojection error — it is never presented as a measured value (R26.2).

    Camera axes follow the OpenCV convention (``+Z`` forward along the optical
    axis, ``+X`` right, ``+Y`` down). The returned ``R``/``t`` map a camera-frame
    point to Floor_Frame (``X_floor = R @ X_cam + t``); with zero tilt the camera
    looks along Floor_Frame ``+Y`` (depth away) and with positive tilt it looks
    down toward the floor.

    Parameters
    ----------
    calibration_id:
        The id assigned to this artefact (referenced by Assessments, R8.4).
    image_size:
        ``(width, height)`` in pixels the intrinsics are defined at.
    horizontal_fov_deg:
        Horizontal field of view used to derive the focal length.
    camera_height_m:
        Camera height above the floor (metres).
    tilt_down_deg:
        Downward tilt of the optical axis below horizontal (degrees).
    stated_reprojection_error_px:
        A plausible, clearly-simulated reprojection error (pixels).

    Returns
    -------
    Calibration
        A ``SIMULATED``-labelled calibration.
    """
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise CalibrationError(f"image_size must be positive, got {image_size!r}")
    if not (0.0 < horizontal_fov_deg < 180.0):
        raise CalibrationError(
            f"horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}"
        )

    # Pinhole focal length from horizontal FOV; square pixels (fx == fy).
    fx = (width / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float
    )
    distortion = np.zeros(5, dtype=float)  # synthetic: no lens distortion

    R, t = _mounting_extrinsics(
        camera_height_m=camera_height_m, tilt_down_deg=tilt_down_deg
    )

    return Calibration(
        id=calibration_id,
        intrinsics=K,
        distortion=distortion,
        rotation=R,
        translation=t,
        reprojection_error=float(stated_reprojection_error_px),
        provenance=ProvenanceLabel.SIMULATED,
        image_size=(width, height),
        floor_origin_note=floor_origin_note,
        source=source,
    )


def build_calibration_from_checkerboard_and_floor(
    calibration_id: str,
    *,
    checkerboard_object_points: Sequence[Sequence[Sequence[float]]],
    checkerboard_image_points: Sequence[Sequence[Sequence[float]]],
    image_size: tuple[int, int],
    floor_object_points: Sequence[Sequence[float]],
    floor_image_points: Sequence[Sequence[float]],
    floor_origin_note: str = "physical floor origin per checkerboard survey",
    source: str = (
        "checkerboard intrinsics (cv2.calibrateCamera) + floor extrinsics "
        "(cv2.solvePnP against surveyed floor points)"
    ),
) -> Calibration:
    """Build a ``MEASURED`` calibration from checkerboard + floor correspondences.

    This is the real calibration path (design: Calibrator): intrinsics and
    distortion are estimated from multiple checkerboard views via
    ``cv2.calibrateCamera``; the camera->floor extrinsics are then estimated
    from known floor points via ``cv2.solvePnP``. The reprojection error is
    computed from the *actual* floor correspondences and is therefore honestly
    labelled :attr:`ProvenanceLabel.MEASURED` (R8.2, R8.3).

    The returned extrinsics map camera -> Floor_Frame: ``cv2.solvePnP`` yields
    the floor-in-camera pose ``(R_wc, t_wc)`` (``X_cam = R_wc @ X_floor +
    t_wc``); we invert it to store ``X_floor = R @ X_cam + t`` with
    ``R = R_wc^T`` and ``t = -R_wc^T @ t_wc``.

    Parameters
    ----------
    calibration_id:
        The id assigned to this artefact.
    checkerboard_object_points:
        List over views of ``(M, 3)`` planar checkerboard corner coordinates.
    checkerboard_image_points:
        List over views of ``(M, 2)`` detected corner pixels (same M per view).
    image_size:
        ``(width, height)`` in pixels used for ``cv2.calibrateCamera``.
    floor_object_points:
        ``(N, 3)`` surveyed floor points in Floor_Frame (metres).
    floor_image_points:
        ``(N, 2)`` pixel coordinates of the floor points.

    Returns
    -------
    Calibration
        A ``MEASURED``-labelled calibration.

    Raises
    ------
    CalibrationError
        If OpenCV calibration/PnP fails or inputs are malformed.
    """
    import cv2

    obj_views = [
        _as_float_array(v).reshape(-1, 3).astype(np.float32)
        for v in checkerboard_object_points
    ]
    img_views = [
        _as_float_array(v).reshape(-1, 2).astype(np.float32)
        for v in checkerboard_image_points
    ]
    if len(obj_views) == 0 or len(obj_views) != len(img_views):
        raise CalibrationError(
            "checkerboard object/image point view counts must match and be >= 1"
        )

    width, height = int(image_size[0]), int(image_size[1])
    ok, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_views, img_views, (width, height), None, None
    )
    if not ok:
        raise CalibrationError("cv2.calibrateCamera failed to converge")

    floor_obj = _as_float_array(floor_object_points).reshape(-1, 3)
    floor_img = _as_float_array(floor_image_points).reshape(-1, 2)
    if floor_obj.shape[0] < 4 or floor_obj.shape[0] != floor_img.shape[0]:
        raise CalibrationError(
            "floor extrinsics need >= 4 matching object/image points"
        )

    ok, rvec, tvec = cv2.solvePnP(
        floor_obj.astype(np.float32),
        floor_img.astype(np.float32),
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise CalibrationError("cv2.solvePnP failed to estimate floor extrinsics")

    # Measured reprojection error from the actual floor correspondences.
    rms = compute_reprojection_error(floor_obj, floor_img, K, dist, rvec, tvec)

    # Invert floor-in-camera (R_wc, t_wc) to camera->floor (R, t).
    R_wc, _ = cv2.Rodrigues(rvec)
    t_wc = tvec.reshape(3)
    R = R_wc.T
    t = -R_wc.T @ t_wc

    return Calibration(
        id=calibration_id,
        intrinsics=K,
        distortion=np.asarray(dist, dtype=float).ravel(),
        rotation=R,
        translation=t,
        reprojection_error=rms,
        provenance=ProvenanceLabel.MEASURED,
        image_size=(width, height),
        floor_origin_note=floor_origin_note,
        source=source,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_calibration(
    calibration_id: str,
    *,
    calibration_dir: "str | Path" = DEFAULT_CALIBRATION_DIR,
) -> Calibration:
    """Load the calibration artefact for ``calibration_id`` (design: Calibrator).

    Reads ``<calibration_dir>/<id>.yaml`` and returns a :class:`Calibration`
    carrying intrinsics ``K``, distortion, extrinsics ``R|t`` (camera ->
    Floor_Frame), the ``reprojection_error``, its provenance, and the id
    (R8.1-R8.4). If the artefact is missing, this degrades honestly by raising
    :class:`CalibrationNotFoundError` rather than fabricating a calibration
    (R26.3, R26.4).

    Expected YAML shape (see ``configs/calibration/sim-cal-v0.yaml``)::

        id: sim-cal-v0
        provenance: simulated
        image_size: [1280, 720]
        intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        distortion: [k1, k2, p1, p2, k3]
        extrinsics:
          rotation: [[...], [...], [...]]   # 3x3, camera -> Floor_Frame
          translation: [tx, ty, tz]          # metres
        reprojection_error:
          value: 0.35
          unit: pixels
          provenance: simulated
        floor_origin_note: "..."
        source: "..."

    Parameters
    ----------
    calibration_id:
        The id whose artefact should be loaded.
    calibration_dir:
        Directory holding per-id YAML artefacts (defaults to
        ``configs/calibration``).

    Returns
    -------
    Calibration
        The loaded, validated calibration.

    Raises
    ------
    CalibrationNotFoundError
        If no artefact file exists for the id.
    CalibrationError
        If the artefact exists but is malformed or internally inconsistent.
    """
    if not calibration_id or not str(calibration_id).strip():
        raise CalibrationError("calibration_id must be a non-empty string")

    path = Path(calibration_dir) / f"{calibration_id}.yaml"
    if not path.is_file():
        raise CalibrationNotFoundError(
            f"no calibration artefact for id {calibration_id!r} at {path}; "
            "not fabricating a calibration (honesty discipline, R26.4)"
        )

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise CalibrationError(f"calibration artefact {path} must be a mapping")

    file_id = data.get("id", calibration_id)
    if file_id != calibration_id:
        raise CalibrationError(
            f"calibration artefact id {file_id!r} does not match requested "
            f"id {calibration_id!r} (file: {path})"
        )

    provenance = _parse_provenance(data.get("provenance"), path)

    try:
        intrinsics = np.asarray(data["intrinsics"], dtype=float)
        distortion = np.asarray(data.get("distortion", []), dtype=float)
        extrinsics = data["extrinsics"]
        rotation = np.asarray(extrinsics["rotation"], dtype=float)
        translation = np.asarray(extrinsics["translation"], dtype=float)
        reproj = data["reprojection_error"]
        reproj_value = float(reproj["value"] if isinstance(reproj, dict) else reproj)
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(
            f"calibration artefact {path} is missing/invalid required fields: {exc}"
        ) from exc

    # If the reprojection error carries its own provenance, it must not claim a
    # stronger label than the artefact (no silent upgrade to measured; R26.2).
    if isinstance(reproj, dict) and reproj.get("provenance") is not None:
        reproj_prov = _parse_provenance(reproj.get("provenance"), path)
        if reproj_prov is not provenance:
            raise CalibrationError(
                f"reprojection_error provenance {reproj_prov.value!r} disagrees "
                f"with artefact provenance {provenance.value!r} (file: {path})"
            )

    image_size = data.get("image_size")
    image_size_tuple = (
        (int(image_size[0]), int(image_size[1]))
        if image_size and len(image_size) == 2
        else None
    )

    return Calibration(
        id=calibration_id,
        intrinsics=intrinsics,
        distortion=distortion,
        rotation=rotation,
        translation=translation,
        reprojection_error=reproj_value,
        provenance=provenance,
        image_size=image_size_tuple,
        floor_origin_note=str(data.get("floor_origin_note", "")),
        source=str(data.get("source", "")),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_float_array(value) -> np.ndarray:
    """Coerce ``value`` to a float ndarray, raising a CalibrationError on failure."""
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"expected numeric array, got {value!r}") from exc


def _parse_provenance(raw, path: Path) -> ProvenanceLabel:
    """Parse a provenance string into a :class:`ProvenanceLabel`."""
    if raw is None:
        raise CalibrationError(
            f"calibration artefact {path} is missing a 'provenance' label "
            "(honesty discipline, R26.1)"
        )
    try:
        return ProvenanceLabel(str(raw))
    except ValueError as exc:
        raise CalibrationError(
            f"invalid provenance {raw!r} in {path}; must be one of "
            f"{[p.value for p in ProvenanceLabel]}"
        ) from exc


def _mounting_extrinsics(
    *, camera_height_m: float, tilt_down_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera->Floor_Frame ``(R, t)`` for a height/down-tilt mounting.

    Camera axes (OpenCV): ``+X`` right, ``+Y`` down, ``+Z`` forward (optical
    axis). Floor_Frame: ``+X`` camera-right, ``+Y`` depth away, ``+Z`` up.

    With zero tilt the optical axis (camera ``+Z``) should point along
    Floor_Frame ``+Y`` (depth away) and be horizontal; a positive
    ``tilt_down_deg`` rotates the optical axis downward toward the floor. The
    camera sits at height ``camera_height_m`` above the floor at the origin's
    ``(x, y) = (0, 0)`` (its ground-projected nadir), so ``t = (0, 0, h)``.
    """
    if camera_height_m <= 0.0:
        raise CalibrationError(
            f"camera_height_m must be positive, got {camera_height_m}"
        )
    tilt = np.radians(tilt_down_deg)

    # Base orientation (zero tilt): the camera basis expressed in Floor_Frame is
    #   cam_x (right)   -> +X_floor  = [1, 0, 0]
    #   cam_y (down)    -> -Z_floor  = [0, 0, -1]
    #   cam_z (forward) -> +Y_floor  = [0, 1, 0]
    # i.e. R0 = [[1,0,0], [0,0,1], [0,-1,0]] (columns are the camera axes).
    R0 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    # A downward tilt rotates the (forward, down) axes about the camera x-axis
    # (Floor_Frame +X). Right-multiplying R0 by a rotation about the camera x
    # keeps R orthonormal by construction. A downward tilt of the optical axis
    # (giving it a negative Floor_Frame Z component) corresponds to -tilt here.
    c, s = np.cos(-tilt), np.sin(-tilt)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )
    R = R0 @ Rx
    t = np.array([0.0, 0.0, float(camera_height_m)])
    return R, t
