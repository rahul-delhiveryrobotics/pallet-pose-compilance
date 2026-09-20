"""Unit tests for provenance labelling primitives and honesty mechanism.

Covers task 1.2: Provenance_Label, Reason_Code, Quality_Flag enums, the
labelling helper API, and the output-layer mechanism that refuses to serialise
a result-bearing field without a label (R26.1, R26.3, R22.3, R24.8, R26.4).
"""

import pytest

from pallet_pose_compliance.output import (
    Labelled,
    MissingProvenanceError,
    ProvenanceLabel,
    QualityFlag,
    ReasonCode,
    is_labelled,
    label,
    labelled,
    require_label,
    unavailable,
)


class TestEnums:
    def test_provenance_label_values(self):
        assert {p.value for p in ProvenanceLabel} == {
            "measured",
            "simulated",
            "estimated",
            "unavailable",
        }

    def test_provenance_label_is_str_enum(self):
        # str-Enum members compare equal to their string value.
        assert ProvenanceLabel.MEASURED == "measured"
        assert ProvenanceLabel("estimated") is ProvenanceLabel.ESTIMATED

    def test_reason_codes_from_error_handling_table(self):
        expected = {
            "NO_PALLET_DETECTED",
            "POSE_INSUFFICIENT_KEYPOINTS",
            "DIMENSIONS_UNKNOWN",
            "POSE_AMBIGUOUS",
            "FACE_AMBIGUOUS_SYMMETRY",
            "ILL_CONDITIONED",
            "POSE_UNCERTAINTY_EXCEEDED",
            "CALIBRATION_INVALID",
            "CAMERA_MOVED",
            "LOW_IMAGE_QUALITY",
            "PROCESSING_FAILED",
        }
        assert {r.value for r in ReasonCode} == expected

    def test_quality_flag_is_str_enum(self):
        assert QualityFlag.ILL_CONDITIONED == "ILL_CONDITIONED"
        assert QualityFlag("LOW_IMAGE_QUALITY") is QualityFlag.LOW_IMAGE_QUALITY


class TestLabelHelper:
    def test_label_attaches_provenance(self):
        result = label(3.14, ProvenanceLabel.MEASURED)
        assert result.value == 3.14
        assert result.provenance is ProvenanceLabel.MEASURED
        assert result.is_available is True
        assert result.reason_code is None
        assert result.quality_flags == ()

    def test_labelled_alias_is_label(self):
        assert labelled is label

    def test_label_with_quality_flags(self):
        result = label(
            {"x": 1.0, "y": 2.0},
            ProvenanceLabel.ESTIMATED,
            quality_flags=(QualityFlag.POSE_UNCERTAINTY_EXCEEDED,),
        )
        assert result.quality_flags == (QualityFlag.POSE_UNCERTAINTY_EXCEEDED,)

    def test_label_rejects_non_enum_provenance(self):
        with pytest.raises(MissingProvenanceError):
            label(1.0, "measured")  # type: ignore[arg-type]

    def test_label_rejects_none_value_for_result_bearing_label(self):
        # A measured/simulated/estimated result must carry an actual value;
        # missing values must go through unavailable().
        with pytest.raises(ValueError):
            label(None, ProvenanceLabel.MEASURED)

    def test_label_rejects_bad_quality_flag(self):
        with pytest.raises(ValueError):
            label(1.0, ProvenanceLabel.MEASURED, quality_flags=("nope",))  # type: ignore[arg-type]


class TestUnavailableNullDiscipline:
    def test_unavailable_has_none_value_and_reason(self):
        result = unavailable(ReasonCode.NO_PALLET_DETECTED)
        assert result.value is None
        assert result.provenance is ProvenanceLabel.UNAVAILABLE
        assert result.reason_code is ReasonCode.NO_PALLET_DETECTED
        assert result.is_available is False

    def test_unavailable_requires_reason_code(self):
        with pytest.raises(ValueError):
            # Constructing UNAVAILABLE without a reason code is forbidden.
            Labelled(value=None, provenance=ProvenanceLabel.UNAVAILABLE)

    def test_unavailable_forbids_fabricated_zero_value(self):
        # Null discipline: an unavailable result may not carry a zero placeholder.
        with pytest.raises(ValueError):
            Labelled(
                value=0.0,
                provenance=ProvenanceLabel.UNAVAILABLE,
                reason_code=ReasonCode.PROCESSING_FAILED,
            )

    def test_unavailable_carries_quality_flags(self):
        result = unavailable(
            ReasonCode.ILL_CONDITIONED,
            quality_flags=(QualityFlag.ILL_CONDITIONED,),
        )
        assert result.quality_flags == (QualityFlag.ILL_CONDITIONED,)


class TestToDict:
    def test_available_to_dict(self):
        d = label(1.5, ProvenanceLabel.SIMULATED).to_dict()
        assert d == {
            "value": 1.5,
            "provenance": "simulated",
            "reason_code": None,
            "quality_flags": [],
        }

    def test_unavailable_to_dict_has_explicit_null(self):
        d = unavailable(ReasonCode.CALIBRATION_INVALID).to_dict()
        assert d["value"] is None  # explicit null, key present
        assert d["provenance"] == "unavailable"
        assert d["reason_code"] == "CALIBRATION_INVALID"


class TestRequireLabelMechanism:
    def test_require_label_passes_through_labelled(self):
        wrapped = label(2.0, ProvenanceLabel.MEASURED)
        assert require_label(wrapped, field_name="reprojection_error") is wrapped

    def test_require_label_accepts_object_with_provenance_attr(self):
        class PoseResultLike:
            provenance = ProvenanceLabel.ESTIMATED

        obj = PoseResultLike()
        assert require_label(obj, field_name="pose") is obj

    def test_require_label_refuses_unlabelled_value(self):
        with pytest.raises(MissingProvenanceError):
            require_label(42.0, field_name="orientation_deg")

    def test_require_label_refuses_object_without_provenance(self):
        class NoProv:
            pass

        with pytest.raises(MissingProvenanceError):
            require_label(NoProv(), field_name="position_m")

    def test_is_labelled(self):
        assert is_labelled(label(1.0, ProvenanceLabel.MEASURED)) is True
        assert is_labelled(unavailable(ReasonCode.PROCESSING_FAILED)) is True
        assert is_labelled(1.0) is False
        assert is_labelled(object()) is False
