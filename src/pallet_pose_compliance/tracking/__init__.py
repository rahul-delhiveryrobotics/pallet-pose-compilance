"""Temporal persistence and circular statistics (R23).

Public API re-exported from :mod:`.tracker`: the per-slot :class:`TrackState`,
its :class:`TrackConfig`, the :func:`update` operation that folds one frame into
a track, the :func:`effective_sample_size` helper for correlated frames, and the
documented default thresholds.
"""

from .tracker import (
    DEFAULT_FRAME_CORRELATION,
    DEFAULT_ORIENTATION_MOVE_THRESHOLD_DEG,
    DEFAULT_POSITION_MOVE_THRESHOLD_M,
    TrackConfig,
    TrackState,
    effective_sample_size,
    update,
)

__all__ = [
    "DEFAULT_FRAME_CORRELATION",
    "DEFAULT_ORIENTATION_MOVE_THRESHOLD_DEG",
    "DEFAULT_POSITION_MOVE_THRESHOLD_M",
    "TrackConfig",
    "TrackState",
    "effective_sample_size",
    "update",
]
