"""Unit tests for the versioned Assessment JSON schema and serialisation.

Covers task 1.3: pydantic models (Assessment, PoseResult, SopCheck,
VerdictResult, Keypoint, Detection, PalletModel), the required Assessment
fields, null discipline for unavailable fields, and the serialise/parse
round-trip (R24.1-24.8, R22.1, R22.3).
"""

import json

import pytest
from pydantic import ValidationError

from pallet_pose_compliance.output import (
    SCHEMA_VERSION,
    Assessment,
    Detection,
    DimensionsM,
    FailureSignal,
    Keypoint,
    PalletModel,
    PoseResult,
    PoseUncertainty,
    PositionM,
    ProvenanceLabel,
    QualityFlag,
    ReasonCode,
    SopCheck,
    StageTimings,
    VerdictResult,
)


# ---------------------------------------------------------------------------
# Builders for valid fixtures
# ---------------------------------------------------------------------------


def _available_pose() -> PoseResult:
    return PoseResult(
        pose_status="available",
        position_m=PositionM(x=1.23, y=4.56),
        orientation_deg=42.0,
        face_identity="stringer_front",
        uncertainty=PoseUncertainty(
            position_cov_m2=[[0.0004, 0.0], [0.0, 0.0004]],
            orientation_std_deg=1.5,
            assumptions=["gaussian keypoint noise 1px"],
            provenance=ProvenanceLabel.ESTIMATED,
        ),
        range_m=3.2,
        viewing_angle_deg=20.0,
        provenance=ProvenanceLabel.SIMULATED,
    )


def _unavailable_pose() -> PoseResult:
    return PoseResult(
        pose_status="unavailable",
        reason_code=ReasonCode.ILL_CONDITIONED,
        quality_flags=[QualityFlag.ILL_CONDITIONED],
        provenance=ProvenanceLabel.UNAVAILABLE,
    )


def _eight_sop_checks() -> list[SopCheck]:
    names = {
        1: "overhang",
        2: "load_height",
        3: "column_tilt",
        4: "size_inversion",
        5: "stretch_wrap",
        6: "box_damage",
        7: "load_centroid",
        8: "pallet_damage",
    }
    checks = []
    for rid in range(1, 9):
        checks.append(
            SopCheck(
                rule_id=rid,
                rule_name=names[rid],
                triage="verifiable",
                status="pass",
                confidence=0.9,
                confidence_semantics="raw_detector_score",
                provenance=ProvenanceLabel.SIMULATED,
            )
        )
    return checks


def _verdict() -> VerdictResult:
    return VerdictResult(
        verdict="PASS",
        reason_codes=[],
        reasoning_text="all mandatory rules satisfied with adequate evidence",
        pose_quality_weighting=0.5,
        contributing_checks=[1, 2, 3],
    )


def _assessment(pose: PoseResult | None = None) -> Assessment:
    return Assessment(
        pallet_id="slot-01-p1",
        timestamp="2024-01-01T00:00:00Z",
        source_image_ref="frames/img_0001.png",
        model_id="yolo11-pose@stub",
        calibration_id="calib-synthetic-001",
        pose=pose or _available_pose(),
        sop_checks=_eight_sop_checks(),
        verdict=_verdict(),
        timings=StageTimings(ingest_ms=1.0, detect_ms=12.0, total_ms=30.0),
        failure=FailureSignal(failed=False),
    )


# ---------------------------------------------------------------------------
# Keypoint / Detection
# ---------------------------------------------------------------------------


class TestKeypointDetection:
    def test_keypoint_visible(self):
        kp = Keypoint(name="bl", u=10.0, v=20.0, visibility="visible", score=0.8)
        assert kp.u == 10.0 and kp.visibility == "visible"

    def test_absent_keypoint_forbids_fabricated_coords(self):
        with pytest.raises(ValidationError):
            Keypoint(name="bl", u=0.0, v=0.0, visibility="absent")

    def test_absent_keypoint_null_coords_ok(self):
        kp = Keypoint(name="bl", visibility="absent")
        assert kp.u is None and kp.v is None

    def test_detection_requires_four_element_bbox(self):
        with pytest.raises(ValidationError):
            Detection(cls="pallet", bbox=[1.0, 2.0, 3.0], score=0.9)

    def test_detection_valid(self):
        det = Detection(
            cls="pallet",
            bbox=[0.0, 0.0, 100.0, 50.0],
            score=0.95,
            keypoints=[Keypoint(name="bl", u=1.0, v=2.0, visibility="visible")],
        )
        assert det.cls == "pallet" and len(det.keypoints) == 1


# ---------------------------------------------------------------------------
# PalletModel
# ---------------------------------------------------------------------------


class TestPalletModel:
    def test_valid_pallet_model(self):
        pm = PalletModel(
            bottom_corners_m=[[0, 0, 0], [1.2, 0, 0], [1.2, 0.8, 0], [0, 0.8, 0]],
            top_corners_m=[
                [0, 0, 0.144],
                [1.2, 0, 0.144],
                [1.2, 0.8, 0.144],
                [0, 0.8, 0.144],
            ],
            dimensions_m=DimensionsM(length=1.2, width=0.8, deck_height=0.144),
            dim_uncertainty_m=0.01,
            source="nominal EUR-pallet spec",
            provenance=ProvenanceLabel.ESTIMATED,
        )
        assert pm.provenance is ProvenanceLabel.ESTIMATED
        assert len(pm.bottom_corners_m) == 4

    def test_pallet_model_requires_four_corners(self):
        with pytest.raises(ValidationError):
            PalletModel(
                bottom_corners_m=[[0, 0, 0]],
                top_corners_m=[[0, 0, 1]],
                dimensions_m=DimensionsM(length=1.2, width=0.8, deck_height=0.1),
                source="x",
                provenance=ProvenanceLabel.ESTIMATED,
            )


# ---------------------------------------------------------------------------
# PoseResult null discipline
# ---------------------------------------------------------------------------


class TestPoseResult:
    def test_available_pose_has_metrics(self):
        pose = _available_pose()
        assert pose.position_m.x == 1.23
        assert pose.orientation_deg == 42.0

    def test_unavailable_pose_has_null_metrics(self):
        pose = _unavailable_pose()
        assert pose.position_m is None
        assert pose.orientation_deg is None
        assert pose.face_identity is None
        assert pose.reason_code is ReasonCode.ILL_CONDITIONED

    def test_unavailable_pose_forbids_fabricated_position(self):
        with pytest.raises(ValidationError):
            PoseResult(
                pose_status="unavailable",
                reason_code=ReasonCode.ILL_CONDITIONED,
                position_m=PositionM(x=0.0, y=0.0),
                provenance=ProvenanceLabel.UNAVAILABLE,
            )

    def test_unavailable_pose_requires_reason_code(self):
        with pytest.raises(ValidationError):
            PoseResult(
                pose_status="unavailable",
                provenance=ProvenanceLabel.UNAVAILABLE,
            )

    def test_available_pose_requires_position_and_orientation(self):
        with pytest.raises(ValidationError):
            PoseResult(
                pose_status="available",
                provenance=ProvenanceLabel.SIMULATED,
            )

    def test_orientation_must_be_wrapped(self):
        with pytest.raises(ValidationError):
            PoseResult(
                pose_status="available",
                position_m=PositionM(x=0.0, y=0.0),
                orientation_deg=270.0,
                provenance=ProvenanceLabel.SIMULATED,
            )

    def test_orientation_boundary_180_is_valid(self):
        pose = PoseResult(
            pose_status="available",
            position_m=PositionM(x=0.0, y=0.0),
            orientation_deg=180.0,
            provenance=ProvenanceLabel.SIMULATED,
        )
        assert pose.orientation_deg == 180.0


# ---------------------------------------------------------------------------
# SopCheck / VerdictResult / FailureSignal
# ---------------------------------------------------------------------------


class TestSopAndVerdict:
    def test_sop_rule_id_range(self):
        with pytest.raises(ValidationError):
            SopCheck(
                rule_id=9,
                rule_name="x",
                triage="verifiable",
                status="pass",
                provenance=ProvenanceLabel.SIMULATED,
            )

    def test_sop_measurement_unit_null_when_measurement_missing(self):
        with pytest.raises(ValidationError):
            SopCheck(
                rule_id=1,
                rule_name="overhang",
                triage="verifiable",
                status="unresolved",
                measurement=None,
                measurement_unit="cm",
                provenance=ProvenanceLabel.UNAVAILABLE
                if False
                else ProvenanceLabel.SIMULATED,
            )

    def test_verdict_values(self):
        v = _verdict()
        assert v.verdict == "PASS"

    def test_failure_signal_requires_reason_when_failed(self):
        with pytest.raises(ValidationError):
            FailureSignal(failed=True)

    def test_failure_signal_ok_when_failed_with_reason(self):
        fs = FailureSignal(failed=True, reason_code=ReasonCode.PROCESSING_FAILED)
        assert fs.reason_code is ReasonCode.PROCESSING_FAILED


# ---------------------------------------------------------------------------
# Assessment schema validity + required fields
# ---------------------------------------------------------------------------


class TestAssessmentSchema:
    def test_valid_assessment_has_required_fields(self):
        a = _assessment()
        assert a.schema_version == SCHEMA_VERSION
        assert a.pallet_id and a.timestamp and a.source_image_ref
        assert a.model_id and a.calibration_id
        assert len(a.sop_checks) == 8
        assert a.pose.pose_status == "available"
        assert a.verdict.verdict == "PASS"

    def test_assessment_requires_eight_sop_checks(self):
        with pytest.raises(ValidationError):
            Assessment(
                pallet_id="p",
                timestamp="t",
                source_image_ref="r",
                model_id="m",
                calibration_id="c",
                pose=_available_pose(),
                sop_checks=_eight_sop_checks()[:7],
                verdict=_verdict(),
            )

    def test_assessment_rejects_duplicate_rule_ids(self):
        checks = _eight_sop_checks()
        checks[7].rule_id = 1  # now missing rule 8, rule 1 twice
        with pytest.raises(ValidationError):
            Assessment(
                pallet_id="p",
                timestamp="t",
                source_image_ref="r",
                model_id="m",
                calibration_id="c",
                pose=_available_pose(),
                sop_checks=checks,
                verdict=_verdict(),
            )

    def test_assessment_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            Assessment(
                pallet_id="p",
                timestamp="t",
                source_image_ref="r",
                model_id="m",
                calibration_id="c",
                pose=_available_pose(),
                sop_checks=_eight_sop_checks(),
                verdict=_verdict(),
                bogus_field=123,
            )


# ---------------------------------------------------------------------------
# Serialisation / round-trip and null discipline
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_available_pose(self):
        a = _assessment()
        parsed = Assessment.parse(a.serialise())
        assert parsed == a

    def test_round_trip_unavailable_pose(self):
        a = _assessment(pose=_unavailable_pose())
        parsed = Assessment.parse(a.serialise())
        assert parsed == a

    def test_round_trip_via_dict(self):
        a = _assessment()
        parsed = Assessment.parse(a.to_dict())
        assert parsed == a

    def test_unavailable_fields_serialise_as_explicit_null(self):
        a = _assessment(pose=_unavailable_pose())
        raw = json.loads(a.serialise())
        # Explicit null keys present (never dropped, never zero placeholders).
        assert raw["pose"]["position_m"] is None
        assert raw["pose"]["orientation_deg"] is None
        assert raw["pose"]["face_identity"] is None
        assert raw["pose"]["reason_code"] == "ILL_CONDITIONED"

    def test_enums_serialise_to_string_values(self):
        a = _assessment()
        raw = json.loads(a.serialise())
        assert raw["pose"]["provenance"] == "simulated"
        assert raw["sop_checks"][0]["provenance"] == "simulated"
        assert raw["schema_version"] == SCHEMA_VERSION

    def test_timings_null_stage_serialises_as_null_not_zero(self):
        a = _assessment()
        raw = json.loads(a.serialise())
        # localise_ms was never set -> explicit null, not 0.0
        assert raw["timings"]["localise_ms"] is None
        assert raw["timings"]["ingest_ms"] == 1.0

    def test_failure_signal_round_trips(self):
        a = _assessment()
        a.failure = FailureSignal(
            failed=True, reason_code=ReasonCode.PROCESSING_FAILED
        )
        parsed = Assessment.parse(a.serialise())
        assert parsed.failure.failed is True
        assert parsed.failure.reason_code is ReasonCode.PROCESSING_FAILED
