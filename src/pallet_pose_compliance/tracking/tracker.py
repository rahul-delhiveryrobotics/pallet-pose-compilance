"""Per-slot temporal persistence with circular-statistics aggregation (R23).

A camera fixed above numbered floor slots sees the *same* pallet in the *same*
slot across many consecutive frames. This module exploits that persistence
(R23.1) by maintaining a per-slot :class:`TrackState` and aggregating repeated
pose measurements over time, using **circular statistics** for orientation so
the aggregate is correct across the wrap boundary (Property 22, R23.2).

What temporal persistence buys us, and what it does *not* (R23.2)
-----------------------------------------------------------------
- **Stability**: averaging repeated measurements of a *static* pallet reduces
  the variance of zero-mean measurement noise, so the reported pose is steadier
  frame-to-frame. The reduction is proportional to ``1 / N_eff`` (see below).
- **Throughput**: once a slot's pose has converged, a downstream consumer can
  reuse the aggregate rather than re-solving at full effort every frame; the
  tracker exposes the running aggregate so that reuse is possible.
- **It does NOT remove systematic calibration bias.** Averaging only attenuates
  *zero-mean* (random) noise. A systematic error — a mis-measured camera height,
  a tilt-angle bias, a lens-distortion residual — is present in *every* frame
  identically, so its contribution to the mean is unchanged no matter how many
  frames are aggregated. Temporal aggregation therefore improves *precision*,
  never *accuracy* against a biased calibration. Callers must not treat a tight
  aggregated spread as evidence of low systematic error.

Effective sample size (correlated frames are not independent)
-------------------------------------------------------------
Consecutive frames of a static pallet are *highly correlated* (the same scene,
similar noise realisation, shared calibration). Treating ``N`` such frames as
``N`` independent samples would badly overstate the precision gained. We model
the frames as a first-order-correlated stream and discount the raw count to an
**effective sample size**::

    N_eff = N * (1 - rho) / (1 + rho)      (clamped to [1, N])

where ``rho`` in ``[0, 1)`` is a documented correlation coefficient
(:data:`DEFAULT_FRAME_CORRELATION`). With ``rho = 0`` frames are independent and
``N_eff == N``; as ``rho -> 1`` the frames collapse toward a single effective
sample. ``N_eff`` — not the naive frame count — is what should be used when
reasoning about the variance of the aggregate. The circular *mean* itself is
still computed over all frames (more samples never hurt the point estimate); it
is the *confidence* in that mean that must use ``N_eff``.

Movement / stale detection with state reset
-------------------------------------------
A slot's track is only meaningful while the *same* pallet sits there. When a new
measurement moves beyond a documented threshold in position
(:data:`DEFAULT_POSITION_MOVE_THRESHOLD_M`) or orientation
(:data:`DEFAULT_ORIENTATION_MOVE_THRESHOLD_DEG`) relative to the running
aggregate, we treat that as a *different* pallet (or the same pallet moved) and
**reset** the accumulated state before starting a fresh aggregate — otherwise we
would average two physically distinct poses into a meaningless blend. When the
measurement is *stale* (within threshold, i.e. unchanged), we aggregate it into
the running estimate.

Honesty discipline (degrade honestly)
-------------------------------------
An *unavailable* :class:`~pallet_pose_compliance.output.schema.PoseResult` is not
a measurement and is **never** aggregated as if it were one: the running
aggregate is left untouched (its sample count does not grow) and the frame is
recorded only as a skipped/unavailable observation. This keeps the aggregate
from being silently biased by "no data" frames.

All circular math is delegated to
:mod:`pallet_pose_compliance.geometry.frames` (``circular_mean_deg``,
``circular_std_deg``, ``angular_difference_deg``, ``wrap_to_180``) — there is a
single authoritative source for angle handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from ..geometry.frames import (
    angular_difference_deg,
    circular_mean_deg,
    circular_std_deg,
    wrap_to_180,
)
from ..output.schema import Detection, PoseResult, PositionM

__all__ = [
    "DEFAULT_POSITION_MOVE_THRESHOLD_M",
    "DEFAULT_ORIENTATION_MOVE_THRESHOLD_DEG",
    "DEFAULT_FRAME_CORRELATION",
    "TrackConfig",
    "TrackState",
    "effective_sample_size",
    "update",
]


# ---------------------------------------------------------------------------
# Documented default thresholds (design: Tracker, R23)
# ---------------------------------------------------------------------------

#: Position move threshold in metres. A new measurement whose floor-projected
#: centre is farther than this from the running aggregate is treated as a
#: different pallet, resetting the track. Chosen at 5 cm — comfortably above the
#: ±2 cm pose evaluation bar so ordinary measurement jitter does not trip a
#: spurious reset, while a genuine slot change (tens of cm) clearly does.
DEFAULT_POSITION_MOVE_THRESHOLD_M: float = 0.05

#: Orientation move threshold in degrees. A new measurement whose orientation
#: differs (circularly) from the running aggregate by more than this is treated
#: as a movement, resetting the track. Chosen at 10° — above the ±3° orientation
#: evaluation bar so noise does not trip it, below a meaningful re-placement.
DEFAULT_ORIENTATION_MOVE_THRESHOLD_DEG: float = 10.0

#: Frame-to-frame correlation coefficient ``rho`` in ``[0, 1)`` used to discount
#: correlated frames to an effective sample size. Consecutive frames of a static
#: pallet are strongly correlated; 0.5 is a documented, deliberately
#: conservative default (it roughly *thirds* the effective count relative to the
#: raw count). This is a modelling assumption, not a measured value.
DEFAULT_FRAME_CORRELATION: float = 0.5


@dataclass(frozen=True)
class TrackConfig:
    """Immutable configuration for the tracker's thresholds and correlation.

    Attributes
    ----------
    position_move_threshold_m:
        Position change (metres) beyond which the track is reset.
    orientation_move_threshold_deg:
        Orientation change (degrees, circular) beyond which the track is reset.
    frame_correlation:
        Correlation coefficient ``rho`` in ``[0, 1)`` for the effective-sample
        discount of correlated frames.
    """

    position_move_threshold_m: float = DEFAULT_POSITION_MOVE_THRESHOLD_M
    orientation_move_threshold_deg: float = DEFAULT_ORIENTATION_MOVE_THRESHOLD_DEG
    frame_correlation: float = DEFAULT_FRAME_CORRELATION

    def __post_init__(self) -> None:
        if self.position_move_threshold_m <= 0.0:
            raise ValueError("position_move_threshold_m must be positive")
        if self.orientation_move_threshold_deg <= 0.0:
            raise ValueError("orientation_move_threshold_deg must be positive")
        if not (0.0 <= self.frame_correlation < 1.0):
            raise ValueError("frame_correlation must be in [0, 1)")


def effective_sample_size(n: int, correlation: float) -> float:
    """Return the effective sample size for ``n`` correlated frames.

    Models a first-order-correlated stream::

        N_eff = n * (1 - rho) / (1 + rho)

    clamped to ``[1, n]`` for ``n >= 1`` (a single observation is always exactly
    one effective sample; correlation can never *increase* the effective count).

    Parameters
    ----------
    n:
        Raw number of aggregated (available) frames, ``>= 0``.
    correlation:
        ``rho`` in ``[0, 1)``.

    Returns
    -------
    float
        The effective sample size. ``0.0`` when ``n == 0``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not (0.0 <= correlation < 1.0):
        raise ValueError("correlation must be in [0, 1)")
    if n == 0:
        return 0.0
    n_eff = n * (1.0 - correlation) / (1.0 + correlation)
    # A single frame is exactly one effective sample; never exceed the raw count
    # and never drop below one once we have at least one observation.
    return max(1.0, min(float(n), n_eff))


@dataclass(frozen=True)
class TrackState:
    """Immutable per-slot track state (design: Tracker).

    A track accumulates the *available* pose measurements observed for one
    numbered floor slot while the pallet there is unchanged. It is immutable:
    :func:`update` returns a new ``TrackState`` rather than mutating in place.

    Attributes
    ----------
    slot_id:
        The numbered floor slot this track belongs to (per-slot identity).
    orientations_deg:
        The wrapped per-frame orientation measurements aggregated so far, in
        chronological order. Used to compute the circular mean/std.
    positions_m:
        The per-frame floor-projected centres aggregated so far, in order.
    sample_count:
        Number of aggregated *available* measurements (``len(orientations_deg)``).
    skipped_count:
        Number of frames observed whose pose was *unavailable* and therefore
        (honestly) not aggregated. Recorded for audit; does not affect the mean.
    frame_correlation:
        The ``rho`` carried from :class:`TrackConfig`, so consumers can recover
        the effective sample size without re-supplying config.
    """

    slot_id: str
    orientations_deg: tuple[float, ...] = ()
    positions_m: tuple[tuple[float, float], ...] = ()
    sample_count: int = 0
    skipped_count: int = 0
    frame_correlation: float = DEFAULT_FRAME_CORRELATION

    @property
    def has_estimate(self) -> bool:
        """True when at least one available measurement has been aggregated."""
        return self.sample_count > 0

    @property
    def effective_sample_size(self) -> float:
        """The correlation-discounted effective sample size of this track."""
        return effective_sample_size(self.sample_count, self.frame_correlation)

    def aggregated_orientation_deg(self) -> Optional[float]:
        """The circular-mean orientation over all aggregated frames.

        Returns ``None`` when no available measurement has been aggregated yet
        (degrade honestly — no fabricated zero). Correct across the wrap
        boundary (e.g. mean of 179 and -179 is ~180, not 0).
        """
        if self.sample_count == 0:
            return None
        return circular_mean_deg(self.orientations_deg)

    def orientation_std_deg(self) -> Optional[float]:
        """Circular standard deviation of the aggregated orientations.

        Returns ``None`` when no measurement has been aggregated. Note this is
        the *dispersion of the samples*; the precision of the mean improves with
        the **effective** sample size, not the raw count.
        """
        if self.sample_count == 0:
            return None
        return circular_std_deg(self.orientations_deg)

    def aggregated_position_m(self) -> Optional[PositionM]:
        """The arithmetic-mean floor-projected centre over aggregated frames.

        Returns ``None`` when no available measurement has been aggregated.
        Position is a plain Euclidean quantity (no wrap), so an ordinary mean is
        correct.
        """
        if self.sample_count == 0:
            return None
        n = len(self.positions_m)
        mean_x = sum(p[0] for p in self.positions_m) / n
        mean_y = sum(p[1] for p in self.positions_m) / n
        return PositionM(x=mean_x, y=mean_y)


def _reset_state(slot_id: str, correlation: float) -> TrackState:
    """Return a fresh, empty track state for ``slot_id``."""
    return TrackState(slot_id=slot_id, frame_correlation=correlation)


def _is_movement(
    state: TrackState,
    position: PositionM,
    orientation_deg: float,
    config: TrackConfig,
) -> bool:
    """Return True when the new measurement moved beyond the reset thresholds.

    Compares the incoming measurement to the running aggregate. When the track
    has no estimate yet there is nothing to move relative to, so this is False.
    """
    if not state.has_estimate:
        return False
    agg_pos = state.aggregated_position_m()
    assert agg_pos is not None  # has_estimate guarantees this
    dx = position.x - agg_pos.x
    dy = position.y - agg_pos.y
    position_delta_m = (dx * dx + dy * dy) ** 0.5
    if position_delta_m > config.position_move_threshold_m:
        return True
    agg_theta = state.aggregated_orientation_deg()
    assert agg_theta is not None
    orientation_delta_deg = abs(angular_difference_deg(orientation_deg, agg_theta))
    return orientation_delta_deg > config.orientation_move_threshold_deg


def update(
    track_state: Optional[TrackState],
    detection: Detection,
    pose_result: PoseResult,
    *,
    slot_id: Optional[str] = None,
    config: TrackConfig = TrackConfig(),
) -> TrackState:
    """Fold one frame's observation into a slot's :class:`TrackState` (R23).

    This is the tracker's single public operation. Given the current
    ``track_state`` for a slot (or ``None`` to start a new track), the frame's
    ``detection`` and the solved ``pose_result``, it returns the updated,
    immutable track state.

    Behaviour
    ---------
    - **Per-slot identity**: the returned state keeps the same ``slot_id``. When
      ``track_state`` is ``None`` a new track is started; ``slot_id`` must then
      be supplied. When ``track_state`` is given, its ``slot_id`` is retained and
      any conflicting ``slot_id`` argument is rejected.
    - **Unavailable pose (degrade honestly)**: if ``pose_result.pose_status`` is
      ``"unavailable"`` (or the pose lacks position/orientation), the running
      aggregate is left untouched and only ``skipped_count`` is incremented — an
      absent measurement is never aggregated as if it were a real one.
    - **Movement -> reset**: if the available measurement moved beyond the
      configured position/orientation thresholds relative to the running
      aggregate, the accumulated state is reset and the new measurement seeds a
      fresh aggregate (we never blend two physically distinct poses).
    - **Stale -> aggregate**: otherwise the measurement is appended and the
      circular-mean orientation / mean position are updated over all frames.

    Parameters
    ----------
    track_state:
        The prior state for this slot, or ``None`` to begin a new track.
    detection:
        The frame's detection (carried through for interface completeness;
        the aggregation uses the solved pose, not raw pixels).
    pose_result:
        The solved pose for this frame.
    slot_id:
        Required when ``track_state`` is ``None`` (identifies the new track);
        must match ``track_state.slot_id`` when both are provided.
    config:
        Thresholds and correlation coefficient (documented defaults).

    Returns
    -------
    TrackState
        A new, immutable track state reflecting this frame.

    Raises
    ------
    ValueError
        If no ``slot_id`` can be determined, or a supplied ``slot_id``
        contradicts the existing track's slot.
    """
    # ---- Resolve per-slot identity --------------------------------------
    if track_state is None:
        if slot_id is None:
            raise ValueError(
                "slot_id is required when starting a new track (track_state=None)"
            )
        state = _reset_state(slot_id, config.frame_correlation)
    else:
        if slot_id is not None and slot_id != track_state.slot_id:
            raise ValueError(
                f"slot_id {slot_id!r} conflicts with the track's slot "
                f"{track_state.slot_id!r}; a track belongs to one slot"
            )
        # Keep the track's slot; adopt the current config's correlation so the
        # effective-sample computation stays consistent with the caller's model.
        state = replace(track_state, frame_correlation=config.frame_correlation)

    # ---- Degrade honestly: never aggregate an unavailable pose ----------
    if (
        pose_result.pose_status != "available"
        or pose_result.position_m is None
        or pose_result.orientation_deg is None
    ):
        return replace(state, skipped_count=state.skipped_count + 1)

    position = pose_result.position_m
    orientation_deg = wrap_to_180(pose_result.orientation_deg)

    # ---- Movement detection -> reset before seeding a fresh aggregate ----
    if _is_movement(state, position, orientation_deg, config):
        state = _reset_state(state.slot_id, config.frame_correlation)

    # ---- Stale / first observation -> aggregate -------------------------
    new_orientations = state.orientations_deg + (orientation_deg,)
    new_positions = state.positions_m + ((position.x, position.y),)
    return replace(
        state,
        orientations_deg=new_orientations,
        positions_m=new_positions,
        sample_count=state.sample_count + 1,
    )
