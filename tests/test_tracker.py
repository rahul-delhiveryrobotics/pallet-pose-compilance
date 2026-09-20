"""Tests for per-slot temporal persistence + circular aggregation (task 11.1, R23).

These tests pin the tracker's behaviour from the design (Tracker component,
R23.1, R23.2, Property 22):

- **Per-slot identity**: a track carries a slot id; a new track needs one; a
  conflicting slot id is rejected.
- **Circular-mean orientation aggregation**, correct across the wrap boundary:
  the aggregate of 179 and -179 is ~180, not 0 (Property 22).
- **Movement detection resets state**: a measurement beyond the position or
  orientation threshold starts a fresh aggregate rather than blending poses.
- **Stale aggregation**: repeated within-threshold measurements accumulate and
  reduce jitter.
- **Degrade honestly**: an unavailable pose is not aggregated as a measurement;
  only the skipped count grows.
- **Effective sample size**: correlated frames are discounted below the raw
  count, never above.
"""

from __future__ import annotations

import math

import pytest

from pallet_pose_compliance.output.provenance import ProvenanceLabel, ReasonCode
from pallet_pose_compliance.output.schema import Detection, PoseResult, PositionM
from pallet_pose_compliance.tracking import (
    TrackConfig,
    TrackState,
    effective_sample_size,
    update,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detection() -> Detection:
    """A minimal valid pallet detection (raw pixels are irrelevant to tracking)."""
    return Detection(cls="pallet", bbox=[0.0, 0.0, 10.0, 10.0], score=0.9)


def _available_pose(x: float, y: float, theta_deg: float) -> PoseResult:
    """An available PoseResult at floor position (x, y) with orientation theta."""
    return PoseResult(
        pose_status="available",
        position_m=PositionM(x=x, y=y),
        orientation_deg=theta_deg,
        provenance=ProvenanceLabel.SIMULATED,
    )


def _unavailable_pose() -> PoseResult:
    """An unavailable PoseResult carrying a reason code (null discipline)."""
    return PoseResult(
        pose_status="unavailable",
        reason_code=ReasonCode.POSE_INSUFFICIENT_KEYPOINTS,
        provenance=ProvenanceLabel.UNAVAILABLE,
    )


# ---------------------------------------------------------------------------
# Per-slot identity
# ---------------------------------------------------------------------------


def test_new_track_requires_slot_id():
    with pytest.raises(ValueError, match="slot_id is required"):
        update(None, _detection(), _available_pose(1.0, 2.0, 10.0))


def test_new_track_seeds_first_measurement_with_slot_identity():
    state = update(None, _detection(), _available_pose(1.0, 2.0, 10.0), slot_id="A3")
    assert state.slot_id == "A3"
    assert state.sample_count == 1
    assert state.has_estimate
    assert state.aggregated_orientation_deg() == pytest.approx(10.0)
    pos = state.aggregated_position_m()
    assert pos is not None
    assert (pos.x, pos.y) == pytest.approx((1.0, 2.0))


def test_conflicting_slot_id_is_rejected():
    state = update(None, _detection(), _available_pose(1.0, 2.0, 10.0), slot_id="A3")
    with pytest.raises(ValueError, match="conflicts with the track's slot"):
        update(state, _detection(), _available_pose(1.0, 2.0, 10.0), slot_id="B7")


def test_slot_id_preserved_across_updates():
    state = update(None, _detection(), _available_pose(1.0, 2.0, 10.0), slot_id="A3")
    state = update(state, _detection(), _available_pose(1.0, 2.0, 11.0))
    assert state.slot_id == "A3"


# ---------------------------------------------------------------------------
# Circular-mean aggregation across the wrap boundary (Property 22)
# ---------------------------------------------------------------------------


def test_circular_mean_across_wrap_boundary_is_180_not_zero():
    # Same slot, unchanged pallet near the +/-180 boundary. A naive arithmetic
    # mean of 179 and -179 would be 0; the circular mean must be ~180.
    state = update(None, _detection(), _available_pose(1.0, 2.0, 179.0), slot_id="S1")
    state = update(state, _detection(), _available_pose(1.0, 2.0, -179.0))
    agg = state.aggregated_orientation_deg()
    assert agg is not None
    assert abs(agg) == pytest.approx(180.0)
    # Definitely not the naive-average zero.
    assert not math.isclose(agg, 0.0, abs_tol=1.0)


def test_stale_aggregation_averages_small_jitter():
    # Repeated near-identical measurements aggregate; the mean sits between them.
    state = update(None, _detection(), _available_pose(1.0, 1.0, 30.0), slot_id="S2")
    state = update(state, _detection(), _available_pose(1.005, 1.0, 32.0))
    state = update(state, _detection(), _available_pose(0.995, 1.0, 28.0))
    assert state.sample_count == 3
    assert state.aggregated_orientation_deg() == pytest.approx(30.0, abs=0.5)
    pos = state.aggregated_position_m()
    assert pos is not None
    assert pos.x == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Movement detection resets state
# ---------------------------------------------------------------------------


def test_position_movement_resets_track():
    state = update(None, _detection(), _available_pose(1.0, 1.0, 10.0), slot_id="S3")
    state = update(state, _detection(), _available_pose(1.005, 1.0, 10.0))
    assert state.sample_count == 2
    # Move well beyond the 5 cm default threshold -> reset, seed fresh aggregate.
    state = update(state, _detection(), _available_pose(2.0, 1.0, 10.0))
    assert state.sample_count == 1
    pos = state.aggregated_position_m()
    assert pos is not None
    assert pos.x == pytest.approx(2.0)


def test_orientation_movement_resets_track():
    state = update(None, _detection(), _available_pose(1.0, 1.0, 10.0), slot_id="S4")
    state = update(state, _detection(), _available_pose(1.0, 1.0, 12.0))
    assert state.sample_count == 2
    # Rotate well beyond the 10 deg default threshold -> reset.
    state = update(state, _detection(), _available_pose(1.0, 1.0, 60.0))
    assert state.sample_count == 1
    assert state.aggregated_orientation_deg() == pytest.approx(60.0)


def test_within_threshold_does_not_reset():
    cfg = TrackConfig()
    state = update(None, _detection(), _available_pose(1.0, 1.0, 10.0), slot_id="S5")
    # Just under the thresholds: 4 cm and 9 deg.
    state = update(state, _detection(), _available_pose(1.04, 1.0, 19.0), config=cfg)
    assert state.sample_count == 2


# ---------------------------------------------------------------------------
# Degrade honestly: unavailable poses are not aggregated
# ---------------------------------------------------------------------------


def test_unavailable_pose_is_not_aggregated():
    state = update(None, _detection(), _available_pose(1.0, 1.0, 10.0), slot_id="S6")
    assert state.sample_count == 1
    state = update(state, _detection(), _unavailable_pose())
    # The aggregate is untouched; only the skipped count grows.
    assert state.sample_count == 1
    assert state.skipped_count == 1
    assert state.aggregated_orientation_deg() == pytest.approx(10.0)


def test_new_track_starting_with_unavailable_pose_has_no_estimate():
    state = update(None, _detection(), _unavailable_pose(), slot_id="S7")
    assert state.slot_id == "S7"
    assert state.sample_count == 0
    assert not state.has_estimate
    assert state.skipped_count == 1
    assert state.aggregated_orientation_deg() is None
    assert state.aggregated_position_m() is None
    assert state.effective_sample_size == 0.0


# ---------------------------------------------------------------------------
# Effective sample size (correlated frames are non-independent)
# ---------------------------------------------------------------------------


def test_effective_sample_size_never_exceeds_raw_count():
    for n in range(1, 20):
        n_eff = effective_sample_size(n, correlation=0.5)
        assert 1.0 <= n_eff <= n


def test_effective_sample_size_independent_frames_equal_count():
    assert effective_sample_size(10, correlation=0.0) == pytest.approx(10.0)


def test_effective_sample_size_discounts_correlated_frames():
    # rho=0.5 => factor (1-0.5)/(1+0.5) = 1/3.
    assert effective_sample_size(9, correlation=0.5) == pytest.approx(3.0)


def test_effective_sample_size_zero_frames():
    assert effective_sample_size(0, correlation=0.5) == 0.0


def test_track_state_effective_sample_size_matches_helper():
    state = update(None, _detection(), _available_pose(1.0, 1.0, 10.0), slot_id="S8")
    state = update(state, _detection(), _available_pose(1.0, 1.0, 10.5))
    state = update(state, _detection(), _available_pose(1.0, 1.0, 9.5))
    expected = effective_sample_size(3, TrackConfig().frame_correlation)
    assert state.effective_sample_size == pytest.approx(expected)
    # Correlated frames must give fewer effective samples than the raw count.
    assert state.effective_sample_size < state.sample_count


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_track_config_rejects_bad_correlation():
    with pytest.raises(ValueError, match="frame_correlation"):
        TrackConfig(frame_correlation=1.0)


def test_track_config_rejects_nonpositive_thresholds():
    with pytest.raises(ValueError, match="position_move_threshold_m"):
        TrackConfig(position_move_threshold_m=0.0)
    with pytest.raises(ValueError, match="orientation_move_threshold_deg"):
        TrackConfig(orientation_move_threshold_deg=-1.0)


def test_immutability_update_returns_new_state():
    s0 = update(None, _detection(), _available_pose(1.0, 1.0, 10.0), slot_id="S9")
    s1 = update(s0, _detection(), _available_pose(1.0, 1.0, 11.0))
    assert s0 is not s1
    assert s0.sample_count == 1
    assert s1.sample_count == 2
    with pytest.raises(Exception):
        s0.sample_count = 5  # frozen dataclass
