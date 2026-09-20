"""Tests for sensitivity analysis + usable-envelope classification (task 7.4, R12).

These tests pin the honesty and correctness properties of the pose sensitivity
and usable-envelope module:

- The sensitivity report contains entries for **camera height error** (R12.1)
  and **camera tilt error** (R12.2) at **both short and long range**, each with
  per-condition translation/rotation error metrics.
- Every evaluated condition receives **exactly one** of the three labels
  ``{meets_tolerance, fails_tolerance, insufficient_evidence}`` — a strict
  partition, no condition unlabelled or double-labelled (R12.4).
- Zero height/tilt error at short range classifies ``meets_tolerance`` (errors
  ~0); a large height/tilt error classifies ``fails_tolerance``.
- An insufficient-coverage / under-sampled condition classifies
  ``insufficient_evidence``.
- The declared usable envelope contains **only** ``meets_tolerance`` conditions,
  the non-usable region contains **only** ``fails_tolerance`` conditions, and no
  extrapolated / unevaluated conditions ever appear (R12.3, R12.4).
- Everything is ``SIMULATED`` provenance (synthetic calibrations).
"""

from __future__ import annotations

import pytest

from pallet_pose_compliance.detection.metrics import DistributionSummary
from pallet_pose_compliance.geometry.pose_envelope import (
    HEIGHT_ERROR_GRID_M,
    MEETS_PASS_FRACTION,
    MIN_EVALUABLE_SAMPLES,
    RANGE_DEFINITIONS,
    TILT_ERROR_GRID_DEG,
    EnvelopeLabel,
    SensitivityReport,
    UsableEnvelope,
    classify_condition,
    combined_success_fraction_over_total,
    declare_usable_envelope,
    run_sensitivity_analysis,
)
from pallet_pose_compliance.geometry.pose_error import (
    PoseErrorReport,
    ToleranceEvaluation,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel


# A single, moderately-sized sensitivity run reused across tests. Kept module
# scope so the (relatively expensive) grid sweep runs once.
@pytest.fixture(scope="module")
def report() -> SensitivityReport:
    return run_sensitivity_analysis(n_samples=12)


# ---------------------------------------------------------------------------
# Helpers to build a synthetic PoseErrorReport for direct classifier tests
# ---------------------------------------------------------------------------


def _make_report(
    *,
    total: int,
    available: int,
    combined_available,
) -> PoseErrorReport:
    """Build a minimal PoseErrorReport with the fields the classifier reads."""
    coverage = (available / total) if total > 0 else None
    tol = ToleranceEvaluation(
        sample_count=available,
        position_tolerance_m=0.02,
        orientation_tolerance_deg=3.0,
        pass_fraction_dx=None,
        pass_fraction_dy=None,
        pass_fraction_radial=combined_available,
        pass_fraction_orientation=combined_available,
        pass_fraction_combined=combined_available,
    )
    empty = DistributionSummary.from_values("x", [], unit="m")
    return PoseErrorReport(
        error_dx_m=empty,
        error_dy_m=empty,
        error_radial_m=empty,
        error_radial_cm=empty,
        error_orientation_deg=empty,
        tolerance=tol,
        total_samples=total,
        available_samples=available,
        unavailable_samples=total - available,
        coverage_fraction=coverage,
    )


# ---------------------------------------------------------------------------
# Sensitivity report: entries for height + tilt at short + long range (R12.1/2)
# ---------------------------------------------------------------------------


class TestSensitivityCoverage:
    def test_contains_height_and_tilt_at_short_and_long_range(self, report):
        for error_type in ("camera_height", "camera_tilt"):
            for range_name in ("short", "long"):
                conds = report.conditions_for(error_type, range_name)
                assert conds, (
                    f"expected conditions for {error_type} at {range_name} range"
                )
                # Every documented grid amount is present exactly once.
                grid = (
                    HEIGHT_ERROR_GRID_M
                    if error_type == "camera_height"
                    else TILT_ERROR_GRID_DEG
                )
                amounts = sorted(c.error_amount for c in conds)
                assert amounts == sorted(grid)

    def test_conditions_carry_translation_and_rotation_metrics(self, report):
        # Every condition exposes separate translation + rotation error metrics.
        for c in report.conditions:
            er = c.error_report
            assert er.error_radial_m.unit == "m"
            assert er.error_orientation_deg.unit == "deg"
            d = c.to_dict()
            assert "radial_error_m" in d
            assert "orientation_error_deg" in d
            assert set(d["radial_error_m"]) >= {"median", "p95", "max"}

    def test_report_is_simulated_and_documents_no_extrapolation(self, report):
        assert report.provenance is ProvenanceLabel.SIMULATED
        d = report.to_dict()
        assert d["provenance"] == "simulated"
        assert "extrapolat" in d["note"].lower()
        # Grids and range definitions are documented in the serialised form.
        assert d["grids"]["height_error_grid_m"] == list(HEIGHT_ERROR_GRID_M)
        assert d["grids"]["tilt_error_grid_deg"] == list(TILT_ERROR_GRID_DEG)
        assert set(d["range_definitions"]) == set(RANGE_DEFINITIONS)

    def test_sensitivity_grows_with_error_and_with_range(self, report):
        # Report the sensitivity: at zero error the pose error is ~0; as the
        # error grows the (over-total) success fraction is non-increasing, and
        # long range is no more usable than short range at the same error.
        for error_type in ("camera_height", "camera_tilt"):
            short = report.conditions_for(error_type, "short")
            long = report.conditions_for(error_type, "long")
            short_success = [
                combined_success_fraction_over_total(c.error_report) or 0.0
                for c in short
            ]
            # Baseline (zero error) is the most usable point.
            assert short_success[0] == pytest.approx(1.0)
            # Non-increasing with growing error amount.
            for a, b in zip(short_success, short_success[1:]):
                assert b <= a + 1e-9
            # Long range is never more usable than short at the same amount.
            long_success = [
                combined_success_fraction_over_total(c.error_report) or 0.0
                for c in long
            ]
            for s, l in zip(short_success, long_success):
                assert l <= s + 1e-9


# ---------------------------------------------------------------------------
# Strict partition: exactly one label per evaluated condition (R12.4)
# ---------------------------------------------------------------------------


class TestStrictPartition:
    def test_every_condition_has_exactly_one_valid_label(self, report):
        valid = set(EnvelopeLabel)
        for c in report.conditions:
            assert c.label in valid
            # The label is a single EnvelopeLabel value (not a set/list).
            assert isinstance(c.label, EnvelopeLabel)

    def test_envelope_partitions_evaluated_conditions_without_overlap(self, report):
        env = declare_usable_envelope(report)
        usable_keys = {c.key for c in env.usable}
        non_keys = {c.key for c in env.non_usable}
        insuff_keys = {c.key for c in env.insufficient_evidence}

        # No condition appears in more than one group (no double-labelling).
        assert usable_keys.isdisjoint(non_keys)
        assert usable_keys.isdisjoint(insuff_keys)
        assert non_keys.isdisjoint(insuff_keys)

        # The three groups together cover exactly the evaluated conditions
        # (nothing unlabelled, nothing invented).
        all_keys = {c.key for c in report.conditions}
        assert usable_keys | non_keys | insuff_keys == all_keys
        assert (
            len(env.usable) + len(env.non_usable) + len(env.insufficient_evidence)
            == len(report.conditions)
        )


# ---------------------------------------------------------------------------
# Zero error -> meets; large error -> fails
# ---------------------------------------------------------------------------


class TestZeroAndLargeError:
    def test_zero_height_error_short_range_meets(self, report):
        conds = report.conditions_for("camera_height", "short")
        zero = next(c for c in conds if c.error_amount == 0.0)
        assert zero.label is EnvelopeLabel.MEETS_TOLERANCE
        # Errors are essentially zero for a correctly-set camera.
        assert zero.error_report.error_radial_m.max == pytest.approx(0.0, abs=1e-3)
        assert zero.error_report.error_orientation_deg.max == pytest.approx(
            0.0, abs=1e-1
        )

    def test_zero_tilt_error_short_range_meets(self, report):
        conds = report.conditions_for("camera_tilt", "short")
        zero = next(c for c in conds if c.error_amount == 0.0)
        assert zero.label is EnvelopeLabel.MEETS_TOLERANCE

    def test_large_height_error_fails(self, report):
        conds = report.conditions_for("camera_height", "short")
        largest = max(conds, key=lambda c: c.error_amount)
        assert largest.error_amount == max(HEIGHT_ERROR_GRID_M)
        assert largest.label is EnvelopeLabel.FAILS_TOLERANCE

    def test_large_tilt_error_fails(self, report):
        conds = report.conditions_for("camera_tilt", "long")
        largest = max(conds, key=lambda c: c.error_amount)
        assert largest.error_amount == max(TILT_ERROR_GRID_DEG)
        assert largest.label is EnvelopeLabel.FAILS_TOLERANCE


# ---------------------------------------------------------------------------
# Insufficient-evidence classification (direct classifier + small-sample run)
# ---------------------------------------------------------------------------


class TestInsufficientEvidence:
    def test_too_few_samples_is_insufficient_evidence(self):
        # Fewer evaluated samples than the documented minimum -> undecidable.
        rep = _make_report(
            total=MIN_EVALUABLE_SAMPLES - 1,
            available=MIN_EVALUABLE_SAMPLES - 1,
            combined_available=1.0,
        )
        assert classify_condition(rep) is EnvelopeLabel.INSUFFICIENT_EVIDENCE

    def test_small_sample_run_yields_insufficient_evidence(self):
        # A sensitivity run below the min-evaluable-samples bar leaves every
        # condition undecidable -> all insufficient_evidence.
        small = run_sensitivity_analysis(n_samples=MIN_EVALUABLE_SAMPLES - 2)
        assert small.conditions
        assert all(
            c.label is EnvelopeLabel.INSUFFICIENT_EVIDENCE
            for c in small.conditions
        )
        env = declare_usable_envelope(small)
        assert not env.usable
        assert not env.non_usable
        assert len(env.insufficient_evidence) == len(small.conditions)

    def test_classifier_meets_and_fails_boundary(self):
        # At/above the meets threshold with adequate samples -> meets.
        meets = _make_report(
            total=MIN_EVALUABLE_SAMPLES,
            available=MIN_EVALUABLE_SAMPLES,
            combined_available=MEETS_PASS_FRACTION,
        )
        assert classify_condition(meets) is EnvelopeLabel.MEETS_TOLERANCE
        # Just below the threshold with adequate samples -> fails.
        fails = _make_report(
            total=MIN_EVALUABLE_SAMPLES,
            available=MIN_EVALUABLE_SAMPLES,
            combined_available=MEETS_PASS_FRACTION - 0.2,
        )
        assert classify_condition(fails) is EnvelopeLabel.FAILS_TOLERANCE

    def test_rejected_poses_count_against_tolerance_not_as_insufficient(self):
        # Adequate samples but the estimator rejected them all (available 0):
        # this is a tolerance FAILURE (unusable pose), not insufficient evidence.
        rejected = _make_report(
            total=MIN_EVALUABLE_SAMPLES,
            available=0,
            combined_available=None,
        )
        assert classify_condition(rejected) is EnvelopeLabel.FAILS_TOLERANCE
        assert combined_success_fraction_over_total(rejected) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Usable envelope: only meets_tolerance in usable, no extrapolation (R12.3/4)
# ---------------------------------------------------------------------------


class TestUsableEnvelope:
    def test_usable_contains_only_meets_conditions(self, report):
        env = declare_usable_envelope(report)
        assert isinstance(env, UsableEnvelope)
        assert env.usable, "expected a non-empty usable envelope in this run"
        for c in env.usable:
            assert c.label is EnvelopeLabel.MEETS_TOLERANCE

    def test_non_usable_contains_only_fails_conditions(self, report):
        env = declare_usable_envelope(report)
        for c in env.non_usable:
            assert c.label is EnvelopeLabel.FAILS_TOLERANCE

    def test_insufficient_group_contains_only_insufficient_conditions(self, report):
        env = declare_usable_envelope(report)
        for c in env.insufficient_evidence:
            assert c.label is EnvelopeLabel.INSUFFICIENT_EVIDENCE

    def test_no_extrapolated_or_unevaluated_conditions_appear(self, report):
        env = declare_usable_envelope(report)
        evaluated_keys = {c.key for c in report.conditions}
        # Every key in the envelope was actually evaluated (no invention).
        assert env.evaluated_condition_keys <= evaluated_keys
        assert env.evaluated_condition_keys == evaluated_keys
        # Every declared condition's error amount is drawn from the documented
        # grid (never an interpolated in-between value).
        for c in list(env.usable) + list(env.non_usable) + list(
            env.insufficient_evidence
        ):
            grid = (
                HEIGHT_ERROR_GRID_M
                if c.error_type == "camera_height"
                else TILT_ERROR_GRID_DEG
            )
            assert c.error_amount in grid
            assert c.range_name in RANGE_DEFINITIONS

    def test_envelope_is_simulated_and_serialisable(self, report):
        env = declare_usable_envelope(report)
        assert env.provenance is ProvenanceLabel.SIMULATED
        d = env.to_dict()
        assert d["provenance"] == "simulated"
        assert "usable_envelope" in d
        assert "non_usable_region" in d
        assert "insufficient_evidence" in d
        assert "extrapolation" in d["note"].lower()
