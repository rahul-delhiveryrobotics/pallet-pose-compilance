"""Keypoint schema, coordinate frames, PnP, and ray-plane pose (R7, R9).

Exposes the pinned Floor_Frame conventions, angle/circular helpers, symmetry
reduction, floor-projected pallet-centre computation, and Face_Identity
derivation (design: Coordinate Systems and Conventions), plus the nominal
pallet 3D model factory.
"""

from .frames import (
    FLOOR_FRAME_ORIGIN_M,
    FLOOR_FRAME_X_AXIS,
    FLOOR_FRAME_Y_AXIS,
    FLOOR_FRAME_Z_AXIS,
    FLOOR_PLANE_Z_M,
    FaceIdentity,
    FaceIdentityResult,
    angular_difference_deg,
    circular_mean_deg,
    circular_std_deg,
    deg_to_rad,
    derive_face_identity,
    floor_projected_centre,
    rad_to_deg,
    reduce_orientation_modulo_symmetry,
    wrap_to_180,
)
from .pallet_model import (
    NOMINAL_DIM_UNCERTAINTY_M,
    NOMINAL_PALLET_DECK_HEIGHT_M,
    NOMINAL_PALLET_LENGTH_M,
    NOMINAL_PALLET_SOURCE,
    NOMINAL_PALLET_WIDTH_M,
    build_nominal_pallet_model,
)

__all__ = [
    # Floor_Frame conventions
    "FLOOR_FRAME_ORIGIN_M",
    "FLOOR_FRAME_X_AXIS",
    "FLOOR_FRAME_Y_AXIS",
    "FLOOR_FRAME_Z_AXIS",
    "FLOOR_PLANE_Z_M",
    # angle + circular helpers
    "wrap_to_180",
    "deg_to_rad",
    "rad_to_deg",
    "circular_mean_deg",
    "circular_std_deg",
    "angular_difference_deg",
    "reduce_orientation_modulo_symmetry",
    # floor-projected centre
    "floor_projected_centre",
    # face identity
    "FaceIdentity",
    "FaceIdentityResult",
    "derive_face_identity",
    # nominal pallet model
    "build_nominal_pallet_model",
    "NOMINAL_PALLET_LENGTH_M",
    "NOMINAL_PALLET_WIDTH_M",
    "NOMINAL_PALLET_DECK_HEIGHT_M",
    "NOMINAL_DIM_UNCERTAINTY_M",
    "NOMINAL_PALLET_SOURCE",
]
