"""Intrinsics/extrinsics loading and reprojection error (R8).

Public API re-exported for convenience so callers can do
``from pallet_pose_compliance.calibration import load_calibration``.
"""

from .calibration import (
    DEFAULT_CALIBRATION_DIR,
    Calibration,
    CalibrationError,
    CalibrationNotFoundError,
    build_calibration_from_checkerboard_and_floor,
    build_synthetic_calibration,
    compute_reprojection_error,
    load_calibration,
)

__all__ = [
    "DEFAULT_CALIBRATION_DIR",
    "Calibration",
    "CalibrationError",
    "CalibrationNotFoundError",
    "build_calibration_from_checkerboard_and_floor",
    "build_synthetic_calibration",
    "compute_reprojection_error",
    "load_calibration",
]
