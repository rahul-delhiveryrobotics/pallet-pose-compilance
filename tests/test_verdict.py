"""Unit tests for the verdict aggregation engine (task 10.1, covers 10.3).

Hand-built PASS / FAIL / MANUAL_INSPECTION / critical-failure-dominance cases
exercising :func:`pallet_pose_compliance.verdict.engine.aggregate` and the
config loader :func:`pallet_pose_compliance.verdict.config.load_verdict_config`.

These validate the R17 verdict rules directly:

- R17.1  verdict is one of PASS/FAIL/MANUAL_INSPECTION
- R17.2  confirmed mandatory violation -> FAIL
- R17.3  adequate evidence all mandatory pass -> PASS
- R17.4  inadequate evidence + no confirmed violation -> MANUAL_INSPECTION
- R17.5  a confirmed critical failure is never averaged away by passing checks
- R17.6  thresholds come from config (no hardcoding)
- R17.7  pose quality is a gate, not a weighted average
- R17.8  reason codes + human-readable reasoning recorded
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pallet_pose_compliance.output.provenance import (
    ProvenanceLabel,
    QualityFlag,
    ReasonCode,
)
from pallet_pose_compliance.output.schema import (
    PositionM,
    PoseResult,
    PoseUncertainty,
    SopCheck,
)
from pallet_pose_compliance.verdict.config import (
    PoseQualityGate,
    VerdictConfig,
    load_verdict_config,
)
from pallet_pose_compliance.verdict.engine import aggregate

_VERDICT_CONFIG_PATH = "configs/verdict.yaml"

# Mandatory rule ids used across the hand-built cases (mirrors configs/verdict.yaml).
_MANDATORY = (1, 2, 3, 5, 7)
_ALL_RULE_IDS = (1, 2, 3, 4, 5, 6, 7, 8)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _config(
    *,
    mandatory=_MANDATORY,
    confirmed_min=0.70,
    adequate_min=0.60,
    max_position_std_cm=5.0,
    max_orientation_std_deg=5.0,
) -> VerdictConfig:
    """Build a VerdictConfig in-memory for a hand-built case."""
    return VerdictConfig(
        mandatory_rule_ids=tuple(mandatory),
        confirmed_violation_min_confidence=confirmed_min,
        adequate_pass_min_confidence=adequate_min,
        pose_quality=PoseQualityGate(
            mode="gate",
            max_position_std_cm=max_position_std_cm,
            max_orientation_std_deg=max_orientation_std_deg,
            source="test#pose_quality",
        ),
        source="test",
    )


def _check(
    rule_id: int,
    status: str,
    confidence: float | None = None,
    *,
    triage: str = "verifiable",
    confidence_semantics: str | None = "raw_detector_score",
    reason_code: ReasonCode | None = None,
) -> SopCheck:
    """Build a minimal schema-valid SopCheck for a hand-built case."""
    prov = (
        ProvenanceLabel.UNAVAILABLE
        if status in ("unresolved", "invalidated")
        else ProvenanceLabel.MEASURED
    )
    # An invalidated/unresolved check carries no confidence semantics.
    if status in ("unresolved", "invalidated"):
        confidence = None
        confidence_semantics = None
    return SopCheck(
        rule_id=rule_id,
        rule_name=f"rule-{rule_id}",
        triage=triage,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        confidence_semantics=confidence_semantics,  # type: ignore[arg-type]
        reason_code=reason_code,
        provenance=prov,
    )


def _all_mandatory_pass(confidence: float = 0.9) -> list[SopCheck]:
    """Eight checks: mandatory rules pass (adequate), non-mandatory unresolved."""
    checks: list[SopCheck] = []
    for rid in _ALL_RULE_IDS:
        if rid in _MANDATORY:
            checks.append(_check(rid, "pass", confidence))
        else:
            checks.append(_check(rid, "unresolved", triage="not_verifiable"))
    return checks


def _good_pose() -> PoseResult:
    """An available pose whose uncertainty is comfortably within the gate."""
    # position std ~ sqrt(1e-4) m = 1 cm; orientation std 1 deg.
    return PoseResult(
        pose_status="available",
        position_m=PositionM(x=1.0, y=2.0),
        orientation_deg=10.0,
        provenance=ProvenanceLabel.SIMULATED,
        uncertainty=PoseUncertainty(
            position_cov_m2=[[1e-4, 0.0], [0.0, 1e-4]],
            orientation_std_deg=1.0,
            provenance=ProvenanceLabel.ESTIMATED,
        ),
    )


def _uncertain_pose() -> PoseResult:
    """An available pose whose position uncertainty exceeds the gate (>5 cm)."""
    # position std sqrt(0.01) m = 10 cm > 5 cm gate.
    return PoseResult(
        pose_status="available",
        position_m=PositionM(x=1.0, y=2.0),
        orientation_deg=10.0,
        provenance=ProvenanceLabel.SIMULATED,
        uncertainty=PoseUncertainty(
            position_cov_m2=[[0.01, 0.0], [0.0, 0.01]],
            orientation_std_deg=1.0,
            provenance=ProvenanceLabel.ESTIMATED,
        ),
    )


def _unavailable_pose() -> PoseResult:
    return PoseResult(
        pose_status="unavailable",
        reason_code=ReasonCode.POSE_INSUFFICIENT_KEYPOINTS,
        provenance=ProvenanceLabel.UNAVAILABLE,
        quality_flags=[QualityFlag.INSUFFICIENT_KEYPOINTS],
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_repo_config_loads_documented_thresholds(self):
        cfg = load_verdict_config(_VERDICT_CONFIG_PATH)
        assert cfg.mandatory_rule_ids == (1, 2, 3, 5, 7)
        assert cfg.confirmed_violation_min_confidence == pytest.approx(0.70)
        assert cfg.adequate_pass_min_confidence == pytest.approx(0.60)
        assert cfg.pose_quality.mode == "gate"
        assert cfg.pose_quality.max_position_std_cm == pytest.approx(5.0)
        assert cfg.pose_quality.max_orientation_std_deg == pytest.approx(5.0)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_verdict_config("configs/does-not-exist.yaml")

    def test_malformed_config_fails_loudly(self, tmp_path: Path):
        bad = tmp_path / "verdict.yaml"
        bad.write_text("mandatory_rule_ids: []\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_verdict_config(bad)

    def test_weighted_average_mode_rejected(self, tmp_path: Path):
        bad = tmp_path / "verdict.yaml"
        bad.write_text(
            "\n".join(
                [
                    "mandatory_rule_ids: [1]",
                    "confirmed_violation_min_confidence: 0.7",
                    "adequate_pass_min_confidence: 0.6",
                    "pose_quality:",
                    '  mode: "weighted_average"',
                    "  max_position_std_cm: 5.0",
                    "  max_orientation_std_deg: 5.0",
                ]
            ),
            encoding="utf-8",
        )
        # A weighted average could offset a confirmed critical failure (R17.5),
        # so the loader refuses it.
        with pytest.raises(ValueError):
            load_verdict_config(bad)


# ---------------------------------------------------------------------------
# FAIL — confirmed mandatory violation (R17.2)
# ---------------------------------------------------------------------------


class TestFail:
    def test_confirmed_mandatory_violation_is_fail(self):
        cfg = _config()
        checks = _all_mandatory_pass()
        # Rule 2 becomes a confirmed fail (confidence >= 0.70).
        checks[1] = _check(2, "fail", 0.95)
        result = aggregate(_good_pose(), checks, cfg)
        assert result.verdict == "FAIL"
        assert 2 in result.contributing_checks
        assert result.reasoning_text  # human-readable reasoning (R17.8)

    def test_low_confidence_fail_is_not_confirmed(self):
        cfg = _config()
        checks = _all_mandatory_pass()
        # Rule 2 fails but below the confirmed threshold -> not a FAIL.
        checks[1] = _check(2, "fail", 0.50)
        result = aggregate(_good_pose(), checks, cfg)
        assert result.verdict == "MANUAL_INSPECTION"

    def test_non_mandatory_fail_does_not_force_fail(self):
        cfg = _config()
        checks = _all_mandatory_pass()
        # Rule 4 is NOT mandatory; a confirmed fail there must not force FAIL.
        checks[3] = _check(4, "fail", 0.99, triage="not_verifiable")
        result = aggregate(_good_pose(), checks, cfg)
        assert result.verdict == "PASS"


# ---------------------------------------------------------------------------
# Critical-failure dominance (R17.5)
# ---------------------------------------------------------------------------


class TestCriticalFailureDominance:
    def test_many_passes_never_average_away_a_confirmed_failure(self):
        cfg = _config()
        # Every mandatory rule passes with very high confidence EXCEPT one
        # confirmed failure. Adding passes must never offset the failure.
        checks = _all_mandatory_pass(confidence=1.0)
        checks[0] = _check(1, "fail", 0.99)  # rule 1 confirmed violation
        result = aggregate(_good_pose(), checks, cfg)
        assert result.verdict == "FAIL"
        assert result.contributing_checks == [1]
        # Pose quality is gated to 0 weight in a FAIL — never offsetting.
        assert result.pose_quality_weighting == 0.0

    def test_fail_dominates_even_with_bad_pose(self):
        cfg = _config()
        checks = _all_mandatory_pass(confidence=1.0)
        checks[0] = _check(1, "fail", 0.99)
        # Even an unavailable pose does not change a confirmed FAIL.
        result = aggregate(_unavailable_pose(), checks, cfg)
        assert result.verdict == "FAIL"


# ---------------------------------------------------------------------------
# PASS — adequate evidence all mandatory pass (R17.3)
# ---------------------------------------------------------------------------


class TestPass:
    def test_all_mandatory_adequate_pass_and_good_pose_is_pass(self):
        cfg = _config()
        result = aggregate(_good_pose(), _all_mandatory_pass(0.8), cfg)
        assert result.verdict == "PASS"
        assert result.contributing_checks == list(_MANDATORY)
        assert result.pose_quality_weighting == 1.0

    def test_low_confidence_pass_is_not_adequate(self):
        cfg = _config()
        checks = _all_mandatory_pass(0.8)
        # Rule 5 passes but below the adequate threshold (0.60).
        checks[4] = _check(5, "pass", 0.40)
        result = aggregate(_good_pose(), checks, cfg)
        assert result.verdict == "MANUAL_INSPECTION"
        assert 5 in result.contributing_checks


# ---------------------------------------------------------------------------
# MANUAL_INSPECTION — inadequate evidence, no confirmed violation (R17.4)
# ---------------------------------------------------------------------------


class TestManualInspection:
    def test_unresolved_mandatory_is_manual(self):
        cfg = _config()
        checks = _all_mandatory_pass(0.8)
        checks[2] = _check(3, "unresolved")  # rule 3 unresolved
        result = aggregate(_good_pose(), checks, cfg)
        assert result.verdict == "MANUAL_INSPECTION"
        assert 3 in result.contributing_checks

    def test_invalidated_pose_dependent_mandatory_is_manual(self):
        cfg = _config()
        checks = _all_mandatory_pass(0.8)
        checks[0] = _check(
            1, "invalidated", reason_code=ReasonCode.POSE_INSUFFICIENT_KEYPOINTS
        )
        result = aggregate(_unavailable_pose(), checks, cfg)
        assert result.verdict == "MANUAL_INSPECTION"

    def test_stub_like_all_unresolved_is_manual(self):
        """All mandatory unresolved + unavailable pose -> MANUAL_INSPECTION."""
        cfg = _config()
        checks = [
            _check(rid, "unresolved", triage="not_verifiable")
            for rid in _ALL_RULE_IDS
        ]
        result = aggregate(_unavailable_pose(), checks, cfg)
        assert result.verdict == "MANUAL_INSPECTION"


# ---------------------------------------------------------------------------
# Pose-quality GATE (R17.7) — not a weighted average
# ---------------------------------------------------------------------------


class TestPoseQualityGate:
    def test_good_load_but_uncertain_pose_degrades_pass_to_manual(self):
        cfg = _config()
        # All mandatory load checks pass adequately, but the pose uncertainty
        # exceeds the gate -> degrade to MANUAL_INSPECTION (never a PASS by
        # averaging pose quality against load compliance).
        result = aggregate(_uncertain_pose(), _all_mandatory_pass(0.9), cfg)
        assert result.verdict == "MANUAL_INSPECTION"
        assert result.pose_quality_weighting == 0.0
        assert "pose" in result.reasoning_text.lower()

    def test_unavailable_pose_degrades_pass_to_manual_with_reason(self):
        cfg = _config()
        result = aggregate(_unavailable_pose(), _all_mandatory_pass(0.9), cfg)
        assert result.verdict == "MANUAL_INSPECTION"
        assert ReasonCode.POSE_INSUFFICIENT_KEYPOINTS in result.reason_codes

    def test_pose_gate_weight_is_binary_not_fractional(self):
        cfg = _config()
        good = aggregate(_good_pose(), _all_mandatory_pass(0.9), cfg)
        bad = aggregate(_uncertain_pose(), _all_mandatory_pass(0.9), cfg)
        # The weight is a gate (1.0 open, 0.0 closed), never a fractional blend.
        assert good.pose_quality_weighting in (0.0, 1.0)
        assert bad.pose_quality_weighting in (0.0, 1.0)
        assert good.pose_quality_weighting == 1.0
        assert bad.pose_quality_weighting == 0.0


# ---------------------------------------------------------------------------
# R17.1 — verdict is always one of the three labels
# ---------------------------------------------------------------------------


def test_verdict_is_always_valid_label():
    cfg = _config()
    for pose in (_good_pose(), _uncertain_pose(), _unavailable_pose()):
        result = aggregate(pose, _all_mandatory_pass(0.8), cfg)
        assert result.verdict in ("PASS", "FAIL", "MANUAL_INSPECTION")
        assert isinstance(result.reasoning_text, str) and result.reasoning_text
