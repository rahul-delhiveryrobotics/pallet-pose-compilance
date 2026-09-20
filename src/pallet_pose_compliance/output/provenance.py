"""Provenance labelling primitives and the honesty-discipline mechanism.

This module is the single source of truth for the system's honesty and
provenance discipline (R26, R27). Every reported result in the pipeline must
carry a :class:`ProvenanceLabel`; where a value is missing or degraded it must
carry a :class:`ReasonCode` rather than a fabricated value.

The output layer owns this module because provenance labelling helpers are used
across *all* stages (detection, geometry, calibration, sop, tracking, verdict)
and the serialisation boundary is where the "refuse to emit an unlabelled
result-bearing field" rule is enforced (design: Provenance & Honesty Mechanism,
Error Handling table).

Public API
----------
- ``ProvenanceLabel`` / ``ReasonCode`` / ``QualityFlag``: the enums.
- ``label`` / ``labelled``: helpers to attach a provenance label to a value,
  producing a :class:`Labelled` wrapper.
- ``unavailable``: helper to construct an ``unavailable`` result carrying a
  reason code and an explicit ``None`` value (null discipline, R24.8/R26.4).
- ``require_label``: the mechanism the output layer calls before serialising a
  result-bearing field; it raises :class:`MissingProvenanceError` when a
  value that carries a result is not labelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

__all__ = [
    "ProvenanceLabel",
    "ReasonCode",
    "QualityFlag",
    "MissingProvenanceError",
    "Labelled",
    "label",
    "labelled",
    "unavailable",
    "is_labelled",
    "require_label",
]


class ProvenanceLabel(str, Enum):
    """Provenance of a reported result (R26.1).

    The four labels are mutually exclusive and exhaustive:

    - ``MEASURED``: computed from actual measurement/evaluation data. Only
      valid when backing sample/measurement data exists (R5.5, R26.1).
    - ``SIMULATED``: produced from synthetic/simulated inputs (e.g. rendered
      pose ground truth). Never presented as a real-world measurement (R26.2).
    - ``ESTIMATED``: a reasoning-based expectation or assumption-driven value
      (e.g. nominal pallet dimensions, target-hardware projections).
    - ``UNAVAILABLE``: the value could not be produced; the field is explicit
      null and accompanied by a :class:`ReasonCode` (R24.8, R26.4).
    """

    MEASURED = "measured"
    SIMULATED = "simulated"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class ReasonCode(str, Enum):
    """Machine-readable reason codes from the Error Handling table.

    Each maps a failure/degradation condition to a stable code surfaced in the
    Assessment so downstream consumers can distinguish absence of load from
    system failure and understand why a result degraded (R22, R27).
    """

    # No pallet found in the image (a valid empty result, distinct from failure).
    NO_PALLET_DETECTED = "NO_PALLET_DETECTED"
    # Occlusion / too few usable keypoints to solve pose.
    POSE_INSUFFICIENT_KEYPOINTS = "POSE_INSUFFICIENT_KEYPOINTS"
    # Pallet dimensions unknown/uncertain beyond a tolerable bound.
    DIMENSIONS_UNKNOWN = "DIMENSIONS_UNKNOWN"
    # Multiple geometrically admissible poses (planar/near-planar ambiguity).
    POSE_AMBIGUOUS = "POSE_AMBIGUOUS"
    # Symmetric geometry: face identity cannot be uniquely resolved.
    FACE_AMBIGUOUS_SYMMETRY = "FACE_AMBIGUOUS_SYMMETRY"
    # Ill-conditioned / degenerate PnP configuration.
    ILL_CONDITIONED = "ILL_CONDITIONED"
    # Pose uncertainty exceeds the documented per-result threshold.
    POSE_UNCERTAINTY_EXCEEDED = "POSE_UNCERTAINTY_EXCEEDED"
    # Calibration invalid or missing.
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    # Suspected camera movement invalidating extrinsics.
    CAMERA_MOVED = "CAMERA_MOVED"
    # Blur / lighting / domain shift lowering detector or keypoint confidence.
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    # Corrupt input or runtime error; failure signal, no fabricated values.
    PROCESSING_FAILED = "PROCESSING_FAILED"


class QualityFlag(str, Enum):
    """Machine-readable indicators of a degraded or unreliable condition.

    Quality flags annotate *why* a result is degraded without necessarily
    forcing it to ``unavailable``; several flags may accompany one result.
    They mirror the conditions in the Error Handling table (R9.5, R13.4, R27).
    """

    ILL_CONDITIONED = "ILL_CONDITIONED"
    POSE_AMBIGUOUS = "POSE_AMBIGUOUS"
    FACE_AMBIGUOUS_SYMMETRY = "FACE_AMBIGUOUS_SYMMETRY"
    POSE_UNCERTAINTY_EXCEEDED = "POSE_UNCERTAINTY_EXCEEDED"
    INSUFFICIENT_KEYPOINTS = "INSUFFICIENT_KEYPOINTS"
    DIMENSIONS_UNKNOWN = "DIMENSIONS_UNKNOWN"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    CAMERA_MOVED = "CAMERA_MOVED"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"


class MissingProvenanceError(ValueError):
    """Raised when a result-bearing field is emitted without a provenance label.

    This is the enforcement mechanism for the honesty discipline: the output
    layer calls :func:`require_label` at the serialisation boundary and this
    error prevents an unlabelled number from ever reaching the JSON schema
    (R26.1, R26.3).
    """


T = TypeVar("T")


@dataclass(frozen=True)
class Labelled(Generic[T]):
    """A value bound to its provenance label (and optional reason/flags).

    Attributes
    ----------
    value:
        The reported value, or ``None`` when ``provenance`` is
        :attr:`ProvenanceLabel.UNAVAILABLE`.
    provenance:
        The :class:`ProvenanceLabel` describing how ``value`` was produced.
    reason_code:
        Required when ``provenance`` is ``UNAVAILABLE``; explains why the value
        could not be produced. May also accompany a degraded-but-present value.
    quality_flags:
        Zero or more :class:`QualityFlag` values annotating degraded conditions.

    Invariants enforced at construction:

    - An ``UNAVAILABLE`` result MUST carry a ``reason_code`` and MUST have a
      ``None`` value (no fabricated/zero placeholder) — Property 3, R24.8/R26.4.
    - A non-``UNAVAILABLE`` result MUST NOT have a ``None`` value; use
      :func:`unavailable` to represent "no value".
    """

    value: Optional[T]
    provenance: ProvenanceLabel
    reason_code: Optional[ReasonCode] = None
    quality_flags: tuple[QualityFlag, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ProvenanceLabel):
            raise MissingProvenanceError(
                f"provenance must be a ProvenanceLabel, got {self.provenance!r}"
            )
        if self.provenance is ProvenanceLabel.UNAVAILABLE:
            if self.value is not None:
                raise ValueError(
                    "an 'unavailable' result must have a None value "
                    "(null discipline); never emit a fabricated/zero placeholder"
                )
            if self.reason_code is None:
                raise ValueError(
                    "an 'unavailable' result must carry a ReasonCode explaining "
                    "why the value is unavailable"
                )
        else:
            if self.value is None:
                raise ValueError(
                    "a result-bearing label "
                    f"({self.provenance.value!r}) must have a non-None value; "
                    "use unavailable(reason_code=...) to represent a missing value"
                )
        if self.reason_code is not None and not isinstance(
            self.reason_code, ReasonCode
        ):
            raise ValueError(
                f"reason_code must be a ReasonCode, got {self.reason_code!r}"
            )
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))
        for flag in self.quality_flags:
            if not isinstance(flag, QualityFlag):
                raise ValueError(
                    f"quality_flags must contain QualityFlag values, got {flag!r}"
                )

    @property
    def is_available(self) -> bool:
        """True when this result carries a value (provenance != unavailable)."""
        return self.provenance is not ProvenanceLabel.UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict with an explicit ``None`` value when unavailable.

        The value key is always present so consumers can distinguish an
        explicit null from a missing field (null discipline, R24.8).
        """
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "reason_code": self.reason_code.value if self.reason_code else None,
            "quality_flags": [f.value for f in self.quality_flags],
        }


def label(
    value: T,
    provenance: ProvenanceLabel,
    *,
    reason_code: Optional[ReasonCode] = None,
    quality_flags: tuple[QualityFlag, ...] = (),
) -> Labelled[T]:
    """Attach a provenance label to ``value``.

    This is the primary helper every stage uses to emit a labelled result.
    For representing a missing value, prefer :func:`unavailable`.

    Raises
    ------
    ValueError
        If the label/value/reason-code combination violates the honesty
        invariants (see :class:`Labelled`).
    """
    return Labelled(
        value=value,
        provenance=provenance,
        reason_code=reason_code,
        quality_flags=tuple(quality_flags),
    )


# Convenient alias so callers can read `labelled(x, MEASURED)` or `label(...)`.
labelled = label


def unavailable(
    reason_code: ReasonCode,
    *,
    quality_flags: tuple[QualityFlag, ...] = (),
) -> Labelled[Any]:
    """Construct an ``unavailable`` result with an explicit ``None`` value.

    Use this wherever a value cannot be produced. The result carries the
    mandatory :class:`ReasonCode` and a ``None`` value, satisfying the null
    discipline (R24.8, R26.4) so no fabricated or zero placeholder is emitted.
    """
    return Labelled(
        value=None,
        provenance=ProvenanceLabel.UNAVAILABLE,
        reason_code=reason_code,
        quality_flags=tuple(quality_flags),
    )


def is_labelled(obj: Any) -> bool:
    """Return True when ``obj`` carries a valid provenance label.

    Accepts either a :class:`Labelled` wrapper or any object exposing a
    ``provenance`` attribute holding a :class:`ProvenanceLabel` (e.g. a
    pydantic model field). This lets the output layer validate both wrapped
    values and richer result objects.
    """
    if isinstance(obj, Labelled):
        return True
    prov = getattr(obj, "provenance", None)
    return isinstance(prov, ProvenanceLabel)


def require_label(obj: Any, *, field_name: str = "result") -> Any:
    """Enforce that a result-bearing field carries a provenance label.

    The output layer calls this at the serialisation boundary for every
    result-bearing field. If the object does not carry a valid
    :class:`ProvenanceLabel`, a :class:`MissingProvenanceError` is raised so an
    unlabelled number can never be serialised (R26.1, R26.3).

    Returns
    -------
    Any
        ``obj`` unchanged, so this can be used inline:
        ``schema_field = require_label(value, field_name="reprojection_error")``.

    Raises
    ------
    MissingProvenanceError
        If ``obj`` does not carry a valid provenance label.
    """
    if not is_labelled(obj):
        raise MissingProvenanceError(
            f"refusing to serialise result-bearing field {field_name!r} without "
            "a Provenance_Label; wrap the value with output.provenance.label(...) "
            "or output.provenance.unavailable(...)"
        )
    return obj
