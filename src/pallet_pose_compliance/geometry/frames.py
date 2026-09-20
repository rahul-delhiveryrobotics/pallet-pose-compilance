"""Floor_Frame conventions, angle wrapping, circular statistics, symmetry
reduction, and Face_Identity derivation (R7.1-R7.4, R9.7).

This module pins the metric-geometry conventions from the design's *Coordinate
Systems and Conventions* section so unit/property tests can assert them
explicitly. Everything here is pure Python + numpy and free of any dependency
on the detector, calibration, or pose solver, so the conventions have a single
authoritative source.

Floor_Frame (design: Floor_Frame)
---------------------------------
- **Origin**: a documented, physically identifiable floor point (recorded per
  calibration id). Represented here as the abstract origin ``(0, 0, 0)``.
- **Axes**: right-handed. ``+X`` runs along the slot rows increasing to
  camera-right; ``+Y`` runs along the depth away from the camera; ``+Z`` is up,
  out of the floor. The floor plane is ``Z = 0``.
- **Units**: metres for position; the floor is treated as planar.

Orientation (design: Orientation)
----------------------------------
- ``theta = 0`` when the pallet long axis is aligned with Floor_Frame ``+X``.
- Positive is counter-clockwise viewed from ``+Z`` (right-handed).
- Units: degrees. Canonical wrap range ``(-180, 180]`` via :func:`wrap_to_180`.
- When the pallet geometry is symmetric such that orientation is only
  determinable modulo a symmetry, :func:`reduce_orientation_modulo_symmetry`
  reports the orientation modulo that symmetry (R9.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "FLOOR_FRAME_ORIGIN_M",
    "FLOOR_FRAME_X_AXIS",
    "FLOOR_FRAME_Y_AXIS",
    "FLOOR_FRAME_Z_AXIS",
    "FLOOR_PLANE_Z_M",
    "wrap_to_180",
    "deg_to_rad",
    "rad_to_deg",
    "circular_mean_deg",
    "circular_std_deg",
    "angular_difference_deg",
    "reduce_orientation_modulo_symmetry",
    "floor_projected_centre",
    "FaceIdentity",
    "FaceIdentityResult",
    "derive_face_identity",
]


# ---------------------------------------------------------------------------
# Floor_Frame definition (right-handed; floor at Z = 0)
# ---------------------------------------------------------------------------

#: Floor_Frame origin expressed in Floor_Frame coordinates (metres). The
#: physical anchor point is recorded per calibration id; in-frame it is (0,0,0).
FLOOR_FRAME_ORIGIN_M: tuple[float, float, float] = (0.0, 0.0, 0.0)

#: +X along the slot rows, increasing to camera-right.
FLOOR_FRAME_X_AXIS: tuple[float, float, float] = (1.0, 0.0, 0.0)

#: +Y along the depth, away from the camera.
FLOOR_FRAME_Y_AXIS: tuple[float, float, float] = (0.0, 1.0, 0.0)

#: +Z up, out of the floor. Right-handed: X x Y = Z.
FLOOR_FRAME_Z_AXIS: tuple[float, float, float] = (0.0, 0.0, 1.0)

#: The floor plane in Floor_Frame is Z = 0.
FLOOR_PLANE_Z_M: float = 0.0


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------


def wrap_to_180(angle_deg: float) -> float:
    """Wrap an angle in degrees to the canonical range ``(-180, 180]``.

    This is the single wrap convention used by all orientation math in the
    system (design: Orientation, wrap convention). Any real input maps into
    ``(-180, 180]``: ``-180`` maps to ``+180``, and ``+180`` stays ``+180``.

    Parameters
    ----------
    angle_deg:
        Any finite angle in degrees.

    Returns
    -------
    float
        The equivalent angle in ``(-180, 180]``.

    Raises
    ------
    ValueError
        If ``angle_deg`` is not finite (NaN/inf), since wrapping is undefined.
    """
    if not math.isfinite(angle_deg):
        raise ValueError(f"cannot wrap a non-finite angle: {angle_deg!r}")
    # Map to [0, 360): use fmod then shift the (-180, 180] half-open interval.
    # (angle + 180) mod 360 lands in [0, 360); subtract 180 -> [-180, 180).
    # We want (-180, 180], so map the -180 endpoint to +180.
    wrapped = math.remainder(angle_deg, 360.0)  # -> [-180, 180]
    # math.remainder returns values in [-180, 180]; normalise the -180 endpoint.
    if wrapped == -180.0:
        wrapped = 180.0
    return wrapped


def deg_to_rad(angle_deg: float) -> float:
    """Convert degrees to radians (exact scaling by pi/180)."""
    return angle_deg * math.pi / 180.0


def rad_to_deg(angle_rad: float) -> float:
    """Convert radians to degrees (exact scaling by 180/pi)."""
    return angle_rad * 180.0 / math.pi


# ---------------------------------------------------------------------------
# Circular statistics (foundations for temporal aggregation, Property 22)
# ---------------------------------------------------------------------------


def circular_mean_deg(angles_deg: Sequence[float]) -> float:
    """Return the circular mean of angles (degrees), wrapped to ``(-180, 180]``.

    Uses the standard mean-of-unit-vectors definition so the mean is correct
    across the wrap boundary (e.g. the mean of ``179`` and ``-179`` is ``180``,
    not ``0``). This is the foundation used by the tracker's temporal
    orientation aggregation (design: Orientation, Property 22).

    Parameters
    ----------
    angles_deg:
        A non-empty sequence of angles in degrees.

    Returns
    -------
    float
        The circular mean in ``(-180, 180]``.

    Raises
    ------
    ValueError
        If ``angles_deg`` is empty, or if the resultant vector length is zero
        (angles cancel out, e.g. antipodal pairs) so the mean is undefined.
    """
    if len(angles_deg) == 0:
        raise ValueError("circular_mean_deg requires at least one angle")
    radians = np.radians(np.asarray(angles_deg, dtype=float))
    sin_sum = float(np.sin(radians).sum())
    cos_sum = float(np.cos(radians).sum())
    if math.isclose(sin_sum, 0.0, abs_tol=1e-12) and math.isclose(
        cos_sum, 0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "circular mean is undefined: resultant vector length is zero "
            "(angles cancel out)"
        )
    mean_rad = math.atan2(sin_sum, cos_sum)
    return wrap_to_180(rad_to_deg(mean_rad))


def circular_std_deg(angles_deg: Sequence[float]) -> float:
    """Return the circular standard deviation of angles in degrees.

    Uses the circular standard deviation ``sqrt(-2 ln R)`` (in radians, then
    converted to degrees), where ``R`` is the mean resultant length in
    ``[0, 1]``. Small spread -> small std; fully dispersed -> large std.

    Parameters
    ----------
    angles_deg:
        A non-empty sequence of angles in degrees.

    Returns
    -------
    float
        The circular standard deviation in degrees (>= 0).

    Raises
    ------
    ValueError
        If ``angles_deg`` is empty.
    """
    if len(angles_deg) == 0:
        raise ValueError("circular_std_deg requires at least one angle")
    radians = np.radians(np.asarray(angles_deg, dtype=float))
    n = radians.size
    resultant = math.hypot(float(np.sin(radians).sum()), float(np.cos(radians).sum()))
    r_mean = resultant / n
    r_mean = min(1.0, max(0.0, r_mean))
    if r_mean <= 0.0:
        # Maximal dispersion; std diverges. Return a large but finite value.
        return float("inf")
    std_rad = math.sqrt(-2.0 * math.log(r_mean))
    return rad_to_deg(std_rad)


def angular_difference_deg(a_deg: float, b_deg: float) -> float:
    """Return the signed smallest angular difference ``a - b`` in ``(-180, 180]``.

    The result is the shortest rotation from ``b`` to ``a``, wrapped so the
    magnitude never exceeds ``180`` degrees.
    """
    return wrap_to_180(a_deg - b_deg)


# ---------------------------------------------------------------------------
# Symmetry-modulo orientation reduction (R9.7)
# ---------------------------------------------------------------------------


def reduce_orientation_modulo_symmetry(
    orientation_deg: float, symmetry_order: int
) -> float:
    """Reduce an orientation modulo a rotational symmetry (R9.7).

    When the visible pallet geometry is symmetric under a rotation of
    ``360 / symmetry_order`` degrees, orientation is only determinable modulo
    that symmetry. This returns a canonical representative of the equivalence
    class, wrapped into the half-open range ``(-sector/2, sector/2]`` where
    ``sector = 360 / symmetry_order``.

    Examples
    --------
    - ``symmetry_order == 1`` (no symmetry): returns ``wrap_to_180(orientation)``.
    - ``symmetry_order == 2`` (180-degree symmetry): folds into ``(-90, 90]``,
      so ``100`` -> ``-80`` and both ``30`` and ``210`` -> ``30``.
    - ``symmetry_order == 4`` (90-degree symmetry): folds into ``(-45, 45]``.

    Parameters
    ----------
    orientation_deg:
        The (possibly unreduced) orientation in degrees.
    symmetry_order:
        The order of the rotational symmetry (positive integer). ``1`` means no
        symmetry.

    Returns
    -------
    float
        The symmetry-reduced orientation, a canonical class representative.

    Raises
    ------
    ValueError
        If ``symmetry_order`` is not a positive integer, or the orientation is
        non-finite.
    """
    if not isinstance(symmetry_order, int) or symmetry_order < 1:
        raise ValueError(
            f"symmetry_order must be a positive integer, got {symmetry_order!r}"
        )
    if not math.isfinite(orientation_deg):
        raise ValueError(f"orientation must be finite, got {orientation_deg!r}")
    if symmetry_order == 1:
        return wrap_to_180(orientation_deg)
    sector = 360.0 / symmetry_order
    # Fold into (-sector/2, sector/2] using the same convention as wrap_to_180.
    reduced = math.remainder(orientation_deg, sector)  # -> [-sector/2, sector/2]
    half = sector / 2.0
    if reduced <= -half:
        reduced += sector
    return reduced


# ---------------------------------------------------------------------------
# Floor-projected pallet centre (design: Pallet reference point)
# ---------------------------------------------------------------------------


def floor_projected_centre(
    bottom_corners_m: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """Compute the floor-projected pallet centre from the four bottom corners.

    The reported pallet position ``(x, y)`` is the centroid of the pallet's
    four bottom deckboard corners projected onto the floor plane ``Z = 0``
    (design: Pallet reference point). The ``Z`` component of each corner is
    ignored (projection onto the floor), so this is a floor-contact reference
    distinct from the load's 3D centre of mass.

    Parameters
    ----------
    bottom_corners_m:
        Exactly four ``[x, y, z]`` points in metres (Floor_Frame). ``z`` is
        expected to be ~0 for floor-contact corners and is ignored.

    Returns
    -------
    tuple[float, float]
        The ``(x, y)`` centroid in metres.

    Raises
    ------
    ValueError
        If not exactly four corners are supplied, or any corner is malformed.
    """
    corners = np.asarray(bottom_corners_m, dtype=float)
    if corners.shape != (4, 3):
        raise ValueError(
            "floor_projected_centre requires exactly four [x, y, z] corners, "
            f"got array of shape {corners.shape}"
        )
    if not np.all(np.isfinite(corners)):
        raise ValueError("bottom_corners_m must contain only finite values")
    centroid_xy = corners[:, :2].mean(axis=0)
    return float(centroid_xy[0]), float(centroid_xy[1])


# ---------------------------------------------------------------------------
# Face identity (design: Face identity)
# ---------------------------------------------------------------------------


class FaceIdentity(str, Enum):
    """The physical pallet face pointing toward the camera (design: Face identity).

    Pallets are not rotationally symmetric in use; the face identity labels
    which physical face is presented to the camera, derived from the asymmetric
    keypoint configuration. When the visible geometry is symmetric under a
    rotation, face identity cannot be uniquely resolved and is represented as
    ``None`` with ``Reason_Code = FACE_AMBIGUOUS_SYMMETRY`` (see
    :func:`derive_face_identity`).
    """

    STRINGER_FRONT = "stringer_front"
    BLOCK_SIDE = "block_side"
    FORK_ENTRY = "fork_entry"


@dataclass(frozen=True)
class FaceIdentityResult:
    """Result of a face-identity derivation.

    Attributes
    ----------
    face_identity:
        The resolved :class:`FaceIdentity`, or ``None`` when the geometry is
        symmetric and the face cannot be uniquely resolved.
    is_ambiguous:
        True when the configuration is symmetric so no unique face exists.
    admissible_faces:
        The set of admissible faces recorded for an ambiguous configuration
        (never a silent single pick). Empty when unresolved for other reasons.
    """

    face_identity: Optional[FaceIdentity]
    is_ambiguous: bool
    admissible_faces: tuple[FaceIdentity, ...] = ()


def derive_face_identity(
    orientation_deg: float,
    *,
    symmetry_order: int = 1,
    admissible_faces: Sequence[FaceIdentity] = (
        FaceIdentity.STRINGER_FRONT,
        FaceIdentity.BLOCK_SIDE,
        FaceIdentity.FORK_ENTRY,
        FaceIdentity.STRINGER_FRONT,
    ),
) -> FaceIdentityResult:
    """Derive the pallet's :class:`FaceIdentity` from its orientation.

    The face presented to the camera depends on which quadrant of orientation
    the pallet sits in; ``admissible_faces`` maps the four orientation quadrants
    (each 90 degrees, starting at the ``(-45, 45]`` sector around ``theta = 0``)
    to a physical face. When the geometry is symmetric such that distinct
    quadrants map to the *same* physical configuration (``symmetry_order > 1``),
    the face cannot be uniquely resolved: the result is flagged ambiguous with
    ``face_identity = None`` and the admissible set recorded — never a silent
    pick (design: Face identity, R9.6, R9.7).

    Parameters
    ----------
    orientation_deg:
        Pallet orientation in degrees (any range; wrapped internally).
    symmetry_order:
        Rotational symmetry order of the visible geometry. ``1`` means the
        geometry is asymmetric and a unique face can be resolved. ``> 1`` means
        the face is ambiguous under the symmetry.
    admissible_faces:
        Four faces mapped to the four orientation quadrants, in
        counter-clockwise order starting from the ``theta = 0`` sector.

    Returns
    -------
    FaceIdentityResult
        The resolved face, or an ambiguous result carrying the admissible set.

    Raises
    ------
    ValueError
        If ``symmetry_order`` is not a positive integer, ``admissible_faces``
        does not have exactly four entries, or the orientation is non-finite.
    """
    if not isinstance(symmetry_order, int) or symmetry_order < 1:
        raise ValueError(
            f"symmetry_order must be a positive integer, got {symmetry_order!r}"
        )
    if len(admissible_faces) != 4:
        raise ValueError(
            "admissible_faces must map exactly four orientation quadrants, "
            f"got {len(admissible_faces)}"
        )
    if not math.isfinite(orientation_deg):
        raise ValueError(f"orientation must be finite, got {orientation_deg!r}")

    if symmetry_order > 1:
        # Symmetric geometry: face identity cannot be uniquely resolved. Record
        # the distinct admissible faces (dedup, preserve order) — never pick one.
        distinct: list[FaceIdentity] = []
        for face in admissible_faces:
            if face not in distinct:
                distinct.append(face)
        return FaceIdentityResult(
            face_identity=None,
            is_ambiguous=True,
            admissible_faces=tuple(distinct),
        )

    # Asymmetric geometry: map the orientation to one of four quadrants. Quadrant
    # boundaries are centred on theta = 0, 90, 180, -90 (each spanning 90 deg).
    theta = wrap_to_180(orientation_deg)
    # Shift by +45 so quadrant 0 covers (-45, 45], then floor-divide by 90.
    quadrant = int(math.floor((theta + 45.0) / 90.0)) % 4
    return FaceIdentityResult(
        face_identity=admissible_faces[quadrant],
        is_ambiguous=False,
        admissible_faces=(),
    )
