"""Versioned Assessment JSON schema and serialisation (R24, R22).

This module defines the pydantic v2 models that make up the auditable,
versioned ``Assessment`` produced per pallet per image, plus the nested result
models (``PoseResult``, ``SopCheck``, ``VerdictResult``, ``Keypoint``,
``Detection``, ``PalletModel``) following the Data Models section of the design.

Design anchors
--------------
- **Versioned & auditable** (R24.1, R24.5): every ``Assessment`` carries a
  ``schema_version`` and the identifiers/timestamps needed to trace a result.
- **Null discipline** (R24.8, R26.4, Property 3/20): unavailable measurements
  are serialised as explicit JSON ``null`` — never zero or a placeholder — and
  serialise-then-parse yields an equivalent object.
- **Provenance discipline** (R26.1): result-bearing objects carry a
  :class:`~pallet_pose_compliance.output.provenance.ProvenanceLabel`; unavailable
  ones additionally carry a :class:`ReasonCode`.
- **Failure signalling** (R22.1, R22.3): the ``Assessment`` carries a
  ``failure`` signal object with a reason code when processing could not
  produce a reliable result for a pallet.

The models re-use the enums/helpers from :mod:`.provenance` so there is a single
source of truth for provenance labels, reason codes, and quality flags.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provenance import ProvenanceLabel, QualityFlag, ReasonCode

__all__ = [
    "SCHEMA_VERSION",
    "Keypoint",
    "Detection",
    "DimensionsM",
    "PalletModel",
    "PositionM",
    "PoseUncertainty",
    "PoseResult",
    "SopCheck",
    "VerdictResult",
    "FailureSignal",
    "StageTimings",
    "Assessment",
]

# The current Assessment schema version. Bump on any breaking schema change so
# consumers can branch on it (R24.5).
SCHEMA_VERSION = "1.0.0"


class _StrictModel(BaseModel):
    """Base model with strict, honesty-friendly configuration.

    - ``extra="forbid"`` rejects unknown fields so a typo cannot silently drop
      or fabricate a value.
    - ``use_enum_values=False`` keeps enum members as enums in-memory; they
      serialise to their string values via ``model_dump(mode="json")``.
    - ``validate_assignment=True`` re-validates on attribute assignment.
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Keypoint / Detection
# ---------------------------------------------------------------------------


class Keypoint(_StrictModel):
    """A single structural reference point on the pallet (design: Data Models).

    ``Keypoint = { name, u: px, v: px, visibility, score }``. ``u``/``v`` are
    pixel coordinates; when a keypoint is ``absent`` its coordinates and score
    may be ``None`` (explicit null, never a fabricated zero).
    """

    name: str
    u: Optional[float] = None
    v: Optional[float] = None
    visibility: Literal["visible", "occluded", "absent"]
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _absent_has_no_coords(self) -> "Keypoint":
        # Null discipline: an absent keypoint must not carry fabricated pixels.
        if self.visibility == "absent" and (self.u is not None or self.v is not None):
            raise ValueError(
                "an 'absent' keypoint must have null u/v coordinates, "
                "never a fabricated placeholder"
            )
        return self


class Detection(_StrictModel):
    """A detected object with box, score, and keypoints (design: Data Models).

    ``Detection = { cls: "pallet"|"box", bbox: [x,y,w,h], score, keypoints }``.
    """

    cls: Literal["pallet", "box"]
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)]
    score: float = Field(ge=0.0, le=1.0)
    keypoints: list[Keypoint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pallet 3D model
# ---------------------------------------------------------------------------

_Point3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class DimensionsM(_StrictModel):
    """Pallet nominal dimensions in metres."""

    length: float
    width: float
    deck_height: float


class PalletModel(_StrictModel):
    """Known 3D pallet model used for known-geometry PnP (design: Data Models).

    Bottom corners lie on the floor (z = 0); top corners are elevated. The
    ``source``/``provenance`` fields document where the dimensions came from
    (e.g. ``estimated`` nominal spec unless a pallet was physically measured).
    """

    bottom_corners_m: Annotated[list[_Point3], Field(min_length=4, max_length=4)]
    top_corners_m: Annotated[list[_Point3], Field(min_length=4, max_length=4)]
    dimensions_m: DimensionsM
    dim_uncertainty_m: Optional[float] = None
    source: str
    provenance: ProvenanceLabel


# ---------------------------------------------------------------------------
# PoseResult
# ---------------------------------------------------------------------------


class PositionM(_StrictModel):
    """Floor-projected pallet centre position in metres (Floor_Frame)."""

    x: float
    y: float


class PoseUncertainty(_StrictModel):
    """Quantified pose uncertainty (design: Data Models, R13).

    Carries a covariance/std summary, the method, its documented assumptions,
    and a provenance label (``estimated`` unless empirically calibrated).
    """

    # 2x2 position covariance in m^2, row-major; null when unavailable.
    position_cov_m2: Optional[Annotated[list[list[float]], Field(min_length=2, max_length=2)]] = None
    orientation_std_deg: Optional[float] = Field(default=None, ge=0.0)
    method: str = "monte_carlo"
    assumptions: list[str] = Field(default_factory=list)
    provenance: ProvenanceLabel = ProvenanceLabel.ESTIMATED


class PoseResult(_StrictModel):
    """Metric pose or an explicit unavailable status (design: Data Models, R13.4).

    When ``pose_status == "unavailable"`` the metric fields (``position_m``,
    ``orientation_deg``, ``face_identity``) MUST be null and a ``reason_code``
    MUST be present — never a fabricated or zero pose (Property 8, R24.8).
    """

    pose_status: Literal["available", "unavailable"]
    reason_code: Optional[ReasonCode] = None
    position_m: Optional[PositionM] = None
    orientation_deg: Optional[float] = None
    orientation_symmetry_reduced: bool = False
    face_identity: Optional[str] = None
    admissible_solutions: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: Optional[PoseUncertainty] = None
    range_m: Optional[float] = None
    viewing_angle_deg: Optional[float] = None
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    provenance: ProvenanceLabel

    @model_validator(mode="after")
    def _enforce_status_discipline(self) -> "PoseResult":
        if self.pose_status == "unavailable":
            # Null discipline: unavailable pose carries null metrics + a reason.
            if self.reason_code is None:
                raise ValueError(
                    "an unavailable pose must carry a reason_code (R13.4)"
                )
            fabricated = {
                "position_m": self.position_m,
                "orientation_deg": self.orientation_deg,
                "face_identity": self.face_identity,
            }
            present = [k for k, v in fabricated.items() if v is not None]
            if present:
                raise ValueError(
                    "an unavailable pose must have null metric fields "
                    f"(offending: {present}); never emit a fabricated pose"
                )
            if self.provenance is not ProvenanceLabel.UNAVAILABLE:
                raise ValueError(
                    "an unavailable pose must carry provenance 'unavailable'"
                )
        else:  # available
            if self.position_m is None or self.orientation_deg is None:
                raise ValueError(
                    "an available pose must carry position_m and orientation_deg "
                    "(Property 7)"
                )
            if self.provenance is ProvenanceLabel.UNAVAILABLE:
                raise ValueError(
                    "an available pose must not carry 'unavailable' provenance"
                )
        # Orientation, when present, must be wrapped to (-180, 180].
        if self.orientation_deg is not None and not (
            -180.0 < self.orientation_deg <= 180.0
        ):
            raise ValueError(
                "orientation_deg must be wrapped to (-180, 180] "
                f"(got {self.orientation_deg})"
            )
        return self


# ---------------------------------------------------------------------------
# SopCheck
# ---------------------------------------------------------------------------


class SopCheck(_StrictModel):
    """A single SOP-PAL-03 rule result (design: Data Models, R14-R16)."""

    rule_id: int = Field(ge=1, le=8)
    rule_name: str
    triage: Literal["verifiable", "partially_verifiable", "not_verifiable"]
    status: Literal["pass", "fail", "unresolved", "invalidated"]
    confidence: Optional[float] = None
    confidence_semantics: Optional[
        Literal[
            "raw_detector_score",
            "calibrated_pass_probability",
            "heuristic_evidence_quality",
        ]
    ] = None
    measurement: Optional[float] = None
    measurement_unit: Optional[str] = None
    uncertainty: Optional[float] = None
    threshold_used: Optional[float] = None
    threshold_source: Optional[str] = None
    evidence_refs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    hidden_regions: list[str] = Field(default_factory=list)
    reason_code: Optional[ReasonCode] = None
    provenance: ProvenanceLabel

    @model_validator(mode="after")
    def _measurement_null_discipline(self) -> "SopCheck":
        # If there is no measurement, its unit must also be null (no placeholder).
        if self.measurement is None and self.measurement_unit is not None:
            raise ValueError(
                "measurement_unit must be null when measurement is unavailable"
            )
        return self


# ---------------------------------------------------------------------------
# VerdictResult
# ---------------------------------------------------------------------------


class VerdictResult(_StrictModel):
    """Aggregated per-pallet verdict (design: Data Models, R17)."""

    verdict: Literal["PASS", "FAIL", "MANUAL_INSPECTION"]
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    reasoning_text: str
    pose_quality_weighting: Optional[float] = None
    contributing_checks: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Failure signal + per-stage timings
# ---------------------------------------------------------------------------


class FailureSignal(_StrictModel):
    """Downstream failure signal for a pallet (R22.1, R22.3).

    ``failed`` distinguishes a genuine processing failure from a valid empty
    result; when ``failed`` is true a ``reason_code`` MUST be present.
    """

    failed: bool = False
    reason_code: Optional[ReasonCode] = None

    @model_validator(mode="after")
    def _failed_requires_reason(self) -> "FailureSignal":
        if self.failed and self.reason_code is None:
            raise ValueError(
                "a raised failure signal must carry a reason_code (R22.3)"
            )
        return self


class StageTimings(_StrictModel):
    """Per-stage wall-clock timings in milliseconds (R24.8).

    Each stage is an explicit ``float`` in milliseconds, or ``null`` when the
    stage did not run (never a fabricated zero).
    """

    ingest_ms: Optional[float] = Field(default=None, ge=0.0)
    detect_ms: Optional[float] = Field(default=None, ge=0.0)
    localise_ms: Optional[float] = Field(default=None, ge=0.0)
    pose_ms: Optional[float] = Field(default=None, ge=0.0)
    sop_ms: Optional[float] = Field(default=None, ge=0.0)
    verdict_ms: Optional[float] = Field(default=None, ge=0.0)
    total_ms: Optional[float] = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


class Assessment(_StrictModel):
    """The versioned, auditable per-pallet Assessment (R24).

    Includes schema version, identifiers, timestamp, source-image ref, model
    and calibration ids, the pose, all SOP checks, the verdict, quality flags,
    reason codes, per-stage timings, and a failure signal.

    Round-trip contract (Property 20): ``Assessment.parse(a.serialise())``
    yields an object equal to ``a``, with unavailable fields as explicit
    ``null`` (never zero).
    """

    schema_version: str = SCHEMA_VERSION
    pallet_id: str
    timestamp: str
    source_image_ref: str
    model_id: str
    calibration_id: str

    pose: PoseResult
    sop_checks: Annotated[list[SopCheck], Field(min_length=8, max_length=8)]
    verdict: VerdictResult

    quality_flags: list[QualityFlag] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    timings: StageTimings = Field(default_factory=StageTimings)
    failure: FailureSignal = Field(default_factory=FailureSignal)

    @model_validator(mode="after")
    def _sop_rule_ids_complete(self) -> "Assessment":
        # All eight rules must be present exactly once (design: Property 14
        # shape; enforced here for schema validity).
        ids = sorted(c.rule_id for c in self.sop_checks)
        if ids != list(range(1, 9)):
            raise ValueError(
                "sop_checks must contain rule_id 1..8 exactly once, "
                f"got {ids}"
            )
        return self

    # ---- serialisation / round-trip -------------------------------------

    def serialise(self) -> str:
        """Serialise to a JSON string with explicit nulls for unavailable fields.

        Uses pydantic's JSON mode so enums become their string values and
        ``None`` fields are emitted as explicit JSON ``null`` (null discipline).
        """
        return self.model_dump_json()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (enums as strings, None as null)."""
        return self.model_dump(mode="json")

    @classmethod
    def parse(cls, data: "str | bytes | dict[str, Any]") -> "Assessment":
        """Parse an Assessment from a JSON string/bytes or a dict.

        This is the inverse of :meth:`serialise`/:meth:`to_dict`; the
        round-trip yields an equivalent object.
        """
        if isinstance(data, (str, bytes)):
            return cls.model_validate_json(data)
        return cls.model_validate(data)
