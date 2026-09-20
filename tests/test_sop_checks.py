"""Unit tests for the verifiable-subset SOP checks + config loader (task 9.3).

Covers:
- the SOP config loader reads the four thresholds + units + sources from the
  YAML (R18.1/R18.3);
- each implemented check computes a measurement and derives pass/fail against
  the config threshold (R15.1/R15.2);
- CHANGING the config threshold changes the check boundary WITHOUT any source
  change (R18.2, the Property-19 behaviour; the property test itself is 9.5,
  out of scope here);
- each SopCheck carries measurement, unit, threshold_used, threshold_source;
- analyze() returns all eight rules with the verifiable subset implemented.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pallet_pose_compliance.output.provenance import ProvenanceLabel, ReasonCode
from pallet_pose_compliance.output.schema import (
    Detection,
    Keypoint,
    PositionM,
    PoseResult,
)
from pallet_pose_compliance.sop.checks import (
    SEMANTICS_CALIBRATED_PASS_PROBABILITY,
    SEMANTICS_HEURISTIC_EVIDENCE_QUALITY,
    analyze,
    check_centroid_offset,
    check_column_tilt,
    check_load_height,
    check_overhang,
    check_wrap_presence,
)
from pallet_pose_compliance.sop.config import (
    DEFAULT_SOP_CONFIG_PATH,
    SopConfig,
    load_sop_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / DEFAULT_SOP_CONFIG_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pallet(bbox=(100.0, 200.0, 120.0, 40.0)) -> Detection:
    """A pallet detection whose 120 px width maps to the 1.2 m nominal length,
    giving a convenient scale of exactly 0.01 m/px (1 cm per pixel)."""
    return Detection(cls="pallet", bbox=list(bbox), score=0.9)


def _box(bbox, keypoints=None) -> Detection:
    return Detection(
        cls="box",
        bbox=list(bbox),
        score=0.8,
        keypoints=keypoints or [],
    )


def _available_pose(orientation_deg: float = 0.0) -> PoseResult:
    return PoseResult(
        pose_status="available",
        position_m=PositionM(x=0.0, y=0.0),
        orientation_deg=orientation_deg,
        provenance=ProvenanceLabel.SIMULATED,
    )


def _unavailable_pose(
    reason_code: ReasonCode = ReasonCode.POSE_INSUFFICIENT_KEYPOINTS,
) -> PoseResult:
    return PoseResult(
        pose_status="unavailable",
        reason_code=reason_code,
        provenance=ProvenanceLabel.UNAVAILABLE,
    )


ALLOWED_CONFIDENCE_SEMANTICS = {
    "raw_detector_score",
    "calibrated_pass_probability",
    "heuristic_evidence_quality",
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_loads_four_thresholds_with_units(self):
        cfg = load_sop_config(CONFIG_PATH)
        assert cfg.overhang_max_cm.value == pytest.approx(3.0)
        assert cfg.overhang_max_cm.unit == "cm"
        assert cfg.load_height_max_m.value == pytest.approx(1.8)
        assert cfg.load_height_max_m.unit == "m"
        assert cfg.column_tilt_max_deg.value == pytest.approx(15.0)
        assert cfg.column_tilt_max_deg.unit == "deg"
        assert cfg.centroid_offset_max_cm.value == pytest.approx(10.0)
        assert cfg.centroid_offset_max_cm.unit == "cm"

    def test_thresholds_carry_rule_ids(self):
        cfg = load_sop_config(CONFIG_PATH)
        assert cfg.overhang_max_cm.rule_id == 1
        assert cfg.load_height_max_m.rule_id == 2
        assert cfg.column_tilt_max_deg.rule_id == 3
        assert cfg.centroid_offset_max_cm.rule_id == 7

    def test_source_identifies_file_and_key(self):
        cfg = load_sop_config(CONFIG_PATH)
        assert cfg.overhang_max_cm.source.endswith(
            "#thresholds.overhang.max_overhang_cm"
        )
        assert str(CONFIG_PATH) in cfg.overhang_max_cm.source

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_sop_config(REPO_ROOT / "configs" / "does_not_exist.yaml")

    def test_missing_threshold_key_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            textwrap.dedent(
                """
                thresholds:
                  overhang:
                    rule_id: 1
                    unit: "cm"
                  load_height:
                    rule_id: 2
                    max_height_m: 1.8
                    unit: "m"
                  column_tilt:
                    rule_id: 3
                    max_tilt_deg: 15.0
                    unit: "deg"
                  centroid_offset:
                    rule_id: 7
                    max_offset_cm: 10.0
                    unit: "cm"
                """
            )
        )
        with pytest.raises(ValueError):
            load_sop_config(bad)


# ---------------------------------------------------------------------------
# Individual checks: measurement + pass/fail from config
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg() -> SopConfig:
    return load_sop_config(CONFIG_PATH)


class TestOverhang:
    def test_no_overhang_passes(self, cfg):
        # Box fully inside the pallet (100..220) -> 0 cm overhang.
        detections = [_pallet(), _box((110.0, 180.0, 80.0, 60.0))]
        check = check_overhang(detections, _available_pose(), cfg)
        assert check.rule_id == 1
        assert check.measurement == pytest.approx(0.0)
        assert check.measurement_unit == "cm"
        assert check.threshold_used == pytest.approx(3.0)
        assert check.threshold_source.endswith("#thresholds.overhang.max_overhang_cm")
        assert check.status == "pass"

    def test_overhang_beyond_threshold_fails(self, cfg):
        # Box extends 5 px past the right edge (220) -> 5 cm at 0.01 m/px.
        detections = [_pallet(), _box((150.0, 180.0, 75.0, 60.0))]
        check = check_overhang(detections, _available_pose(), cfg)
        assert check.measurement == pytest.approx(5.0)
        assert check.status == "fail"

    def test_unresolved_without_boxes(self, cfg):
        check = check_overhang([_pallet()], _available_pose(), cfg)
        assert check.status == "unresolved"
        assert check.measurement is None
        assert check.measurement_unit is None
        # threshold provenance is still recorded even when unresolved
        assert check.threshold_used == pytest.approx(3.0)


class TestLoadHeight:
    def test_height_within_limit_passes(self, cfg):
        # pallet bottom = 240; box top = 90 -> 150 px -> 1.5 m at 0.01 m/px.
        detections = [_pallet(), _box((110.0, 90.0, 80.0, 150.0))]
        check = check_load_height(detections, None, cfg)
        assert check.rule_id == 2
        assert check.measurement == pytest.approx(1.5)
        assert check.measurement_unit == "m"
        assert check.threshold_used == pytest.approx(1.8)
        assert check.status == "pass"

    def test_height_over_limit_fails(self, cfg):
        # box top = 40 -> 200 px -> 2.0 m > 1.8 m.
        detections = [_pallet(), _box((110.0, 40.0, 80.0, 200.0))]
        check = check_load_height(detections, None, cfg)
        assert check.measurement == pytest.approx(2.0)
        assert check.status == "fail"


class TestColumnTilt:
    def test_aligned_box_passes(self, cfg):
        # Two keypoints horizontal -> 0 deg tilt vs pallet axis (0 deg).
        box = _box(
            (110.0, 180.0, 80.0, 60.0),
            keypoints=[
                Keypoint(name="tl", u=110.0, v=180.0, visibility="visible"),
                Keypoint(name="tr", u=190.0, v=180.0, visibility="visible"),
            ],
        )
        check = check_column_tilt([_pallet(), box], _available_pose(0.0), cfg)
        assert check.rule_id == 3
        assert check.measurement == pytest.approx(0.0, abs=1e-9)
        assert check.measurement_unit == "deg"
        assert check.status == "pass"

    def test_tilted_box_fails(self, cfg):
        # 45 deg edge -> 45 deg tilt > 15 deg.
        box = _box(
            (110.0, 180.0, 80.0, 60.0),
            keypoints=[
                Keypoint(name="tl", u=100.0, v=100.0, visibility="visible"),
                Keypoint(name="tr", u=200.0, v=200.0, visibility="visible"),
            ],
        )
        check = check_column_tilt([_pallet(), box], _available_pose(0.0), cfg)
        assert check.measurement == pytest.approx(45.0)
        assert check.status == "fail"

    def test_unresolved_without_keypoints(self, cfg):
        # With an available pose (so the pose-dependent invalidation does not
        # apply), a box lacking orientation keypoints leaves the tilt unresolved.
        check = check_column_tilt(
            [_pallet(), _box((110.0, 180.0, 80.0, 60.0))], _available_pose(0.0), cfg
        )
        assert check.status == "unresolved"
        assert check.measurement is None


class TestCentroidOffset:
    def test_centred_load_passes(self, cfg):
        # pallet centre u = 160; single box centred at 160 -> 0 cm offset.
        detections = [_pallet(), _box((120.0, 180.0, 80.0, 60.0))]
        check = check_centroid_offset(detections, _available_pose(), cfg)
        assert check.rule_id == 7
        assert check.measurement == pytest.approx(0.0)
        assert check.measurement_unit == "cm"
        assert check.threshold_used == pytest.approx(10.0)
        assert check.status == "pass"

    def test_offset_load_fails(self, cfg):
        # box centred at 175 -> 15 px -> 15 cm > 10 cm.
        detections = [_pallet(), _box((135.0, 180.0, 80.0, 60.0))]
        check = check_centroid_offset(detections, _available_pose(), cfg)
        assert check.measurement == pytest.approx(15.0)
        assert check.status == "fail"


class TestWrapPresence:
    def test_threshold_free(self, cfg):
        check = check_wrap_presence([_pallet()], None, cfg)
        assert check.rule_id == 5
        assert check.threshold_used is None
        assert check.threshold_source is None
        assert check.status == "pass"


# ---------------------------------------------------------------------------
# Config-driven boundary: changing the YAML moves the boundary, no code change
# ---------------------------------------------------------------------------


def _mk_overhang_config(dir_path: Path, max_overhang_cm: float) -> SopConfig:
    dir_path.mkdir(parents=True, exist_ok=True)
    p = dir_path / "sop.yaml"
    p.write_text(
        textwrap.dedent(
            f"""
            thresholds:
              overhang:
                rule_id: 1
                max_overhang_cm: {max_overhang_cm}
                unit: "cm"
              load_height:
                rule_id: 2
                max_height_m: 1.8
                unit: "m"
              column_tilt:
                rule_id: 3
                max_tilt_deg: 15.0
                unit: "deg"
              centroid_offset:
                rule_id: 7
                max_offset_cm: 10.0
                unit: "cm"
            """
        )
    )
    return load_sop_config(p)


def test_config_change_flips_boundary_no_code_change(tmp_path: Path):
    # A fixed 5 cm overhang measurement (box 5 px past the 220 px edge).
    detections = [_pallet(), _box((150.0, 180.0, 75.0, 60.0))]

    strict = _mk_overhang_config(tmp_path / "strict", 3.0)
    lax = _mk_overhang_config(tmp_path / "lax", 6.0)

    strict_check = check_overhang(detections, _available_pose(), strict)
    lax_check = check_overhang(detections, _available_pose(), lax)

    # Same measurement, opposite verdict driven purely by the config value.
    assert strict_check.measurement == pytest.approx(5.0)
    assert lax_check.measurement == pytest.approx(5.0)
    assert strict_check.status == "fail"
    assert lax_check.status == "pass"
    assert strict_check.threshold_used == pytest.approx(3.0)
    assert lax_check.threshold_used == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# analyze(): full eight-rule set with verifiable subset implemented
# ---------------------------------------------------------------------------


def test_analyze_returns_all_eight_rules(cfg):
    detections = [_pallet(), _box((110.0, 90.0, 80.0, 150.0))]
    checks = analyze(detections, _available_pose(), cfg)
    assert [c.rule_id for c in checks] == list(range(1, 9))


def test_analyze_implemented_subset_measured(cfg):
    detections = [_pallet(), _box((110.0, 90.0, 80.0, 150.0))]
    checks = {c.rule_id: c for c in analyze(detections, _available_pose(), cfg)}
    # Verifiable subset carries measurements + threshold provenance.
    for rid in (1, 2, 7):
        assert checks[rid].measurement is not None
        assert checks[rid].measurement_unit is not None
        assert checks[rid].threshold_used is not None
        assert checks[rid].threshold_source is not None
        assert checks[rid].status in {"pass", "fail"}
    # not_verifiable rules stay unresolved (real discipline finalised in 9.7).
    for rid in (4, 6, 8):
        assert checks[rid].status == "unresolved"
        assert checks[rid].measurement is None


# ---------------------------------------------------------------------------
# Task 9.7: SOP status discipline, pose-dependent invalidation, confidence
# ---------------------------------------------------------------------------


def _full_detections() -> list[Detection]:
    """A pallet + one box that yields measurable geometry for all checks."""
    return [
        _pallet(),
        _box(
            (110.0, 90.0, 80.0, 150.0),
            keypoints=[
                Keypoint(name="tl", u=110.0, v=90.0, visibility="visible"),
                Keypoint(name="tr", u=190.0, v=90.0, visibility="visible"),
            ],
        ),
    ]


class TestNotVerifiableAlwaysUnresolved:
    """R15.3: not_verifiable rules (4, 6, 8) are ALWAYS unresolved."""

    def test_not_verifiable_rules_unresolved_with_available_pose(self, cfg):
        checks = {c.rule_id: c for c in analyze(_full_detections(), _available_pose(), cfg)}
        for rid in (4, 6, 8):
            assert checks[rid].triage == "not_verifiable"
            assert checks[rid].status == "unresolved"
            assert checks[rid].measurement is None

    def test_not_verifiable_rules_unresolved_with_unavailable_pose(self, cfg):
        checks = {c.rule_id: c for c in analyze(_full_detections(), _unavailable_pose(), cfg)}
        for rid in (4, 6, 8):
            assert checks[rid].status == "unresolved"

    def test_not_verifiable_rules_unresolved_with_empty_detections(self, cfg):
        checks = {c.rule_id: c for c in analyze([], None, cfg)}
        for rid in (4, 6, 8):
            assert checks[rid].status == "unresolved"

    def test_not_verifiable_never_pass_or_fail(self, cfg):
        # Try a spread of inputs; rules 4/6/8 must never become pass/fail.
        for detections in ([], [_pallet()], _full_detections()):
            for pose in (None, _available_pose(), _unavailable_pose()):
                checks = {c.rule_id: c for c in analyze(detections, pose, cfg)}
                for rid in (4, 6, 8):
                    assert checks[rid].status == "unresolved"
                    assert checks[rid].status not in {"pass", "fail"}

    def test_not_verifiable_no_processing_failed_reason(self, cfg):
        # The honest not-verifiable case must NOT be labelled a processing
        # failure (that reason is reserved for real runtime failures).
        checks = {c.rule_id: c for c in analyze(_full_detections(), _available_pose(), cfg)}
        for rid in (4, 6, 8):
            assert checks[rid].reason_code != ReasonCode.PROCESSING_FAILED


class TestPoseDependentInvalidation:
    """R15.4: pose-dependent rules (1, 3, 7) invalidated when pose unavailable."""

    @pytest.mark.parametrize("rule_id", (1, 3, 7))
    def test_pose_dependent_invalidated_when_pose_none(self, cfg, rule_id):
        checks = {c.rule_id: c for c in analyze(_full_detections(), None, cfg)}
        assert checks[rule_id].status == "invalidated"
        assert checks[rule_id].reason_code is not None

    @pytest.mark.parametrize("rule_id", (1, 3, 7))
    def test_pose_dependent_invalidated_when_pose_unavailable(self, cfg, rule_id):
        checks = {c.rule_id: c for c in analyze(_full_detections(), _unavailable_pose(), cfg)}
        assert checks[rule_id].status == "invalidated"
        assert checks[rule_id].reason_code is not None

    def test_invalidation_propagates_pose_reason_code(self, cfg):
        pose = _unavailable_pose(ReasonCode.POSE_UNCERTAINTY_EXCEEDED)
        checks = {c.rule_id: c for c in analyze(_full_detections(), pose, cfg)}
        for rid in (1, 3, 7):
            assert checks[rid].reason_code == ReasonCode.POSE_UNCERTAINTY_EXCEEDED

    def test_invalidated_check_has_no_measurement_or_confidence(self, cfg):
        checks = {c.rule_id: c for c in analyze(_full_detections(), None, cfg)}
        for rid in (1, 3, 7):
            assert checks[rid].measurement is None
            assert checks[rid].confidence is None
            assert checks[rid].confidence_semantics is None

    @pytest.mark.parametrize("rule_id", (2, 5))
    def test_non_pose_dependent_not_invalidated_by_pose_loss(self, cfg, rule_id):
        # Rules 2 (load height) and 5 (wrap) do NOT depend on pose, so an
        # unavailable pose must not invalidate them.
        checks = {c.rule_id: c for c in analyze(_full_detections(), _unavailable_pose(), cfg)}
        assert checks[rule_id].status != "invalidated"

    def test_pose_dependent_measured_when_pose_available(self, cfg):
        checks = {c.rule_id: c for c in analyze(_full_detections(), _available_pose(), cfg)}
        for rid in (1, 3, 7):
            assert checks[rid].status in {"pass", "fail"}


class TestConfidenceSemantics:
    """R16.1-R16.4: honest per-check confidence + semantics labelling."""

    def test_every_implemented_check_has_confidence_and_semantics(self, cfg):
        checks = {c.rule_id: c for c in analyze(_full_detections(), _available_pose(), cfg)}
        for rid in (1, 2, 3, 5, 7):
            assert checks[rid].confidence is not None
            assert checks[rid].confidence_semantics in ALLOWED_CONFIDENCE_SEMANTICS

    def test_no_check_labelled_calibrated_pass_probability(self, cfg):
        # No calibration mapping exists -> nothing may claim calibrated semantics.
        for pose in (_available_pose(), _unavailable_pose(), None):
            for c in analyze(_full_detections(), pose, cfg):
                assert c.confidence_semantics != SEMANTICS_CALIBRATED_PASS_PROBABILITY

    def test_geometry_checks_use_raw_detector_score(self, cfg):
        checks = {c.rule_id: c for c in analyze(_full_detections(), _available_pose(), cfg)}
        for rid in (1, 2, 3, 7):
            assert checks[rid].confidence_semantics == "raw_detector_score"

    def test_wrap_uses_heuristic_evidence_quality(self, cfg):
        check = check_wrap_presence(_full_detections(), None, cfg)
        assert check.rule_id == 5
        assert check.confidence is not None
        assert check.confidence_semantics == SEMANTICS_HEURISTIC_EVIDENCE_QUALITY

    def test_detector_confidence_reflects_least_confident_detection(self, cfg):
        # Confidence is the min detection score of the contributing detections.
        pallet = Detection(cls="pallet", bbox=[100.0, 200.0, 120.0, 40.0], score=0.9)
        box = Detection(cls="box", bbox=[110.0, 90.0, 80.0, 150.0], score=0.4)
        check = check_load_height([pallet, box], None, cfg)
        assert check.confidence == pytest.approx(0.4)
        assert check.confidence_semantics == "raw_detector_score"
