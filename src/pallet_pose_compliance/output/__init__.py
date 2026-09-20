"""Assessment schema, provenance labelling, and serialisation (R24, R26).

The output subpackage owns the cross-cutting provenance/honesty mechanism used
by every stage. Provenance primitives are re-exported here for convenience.
"""

from .provenance import (
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
from .schema import (
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
    SopCheck,
    StageTimings,
    VerdictResult,
)

__all__ = [
    # provenance primitives
    "ProvenanceLabel",
    "ReasonCode",
    "QualityFlag",
    "Labelled",
    "MissingProvenanceError",
    "label",
    "labelled",
    "unavailable",
    "is_labelled",
    "require_label",
    # assessment schema
    "SCHEMA_VERSION",
    "Assessment",
    "Detection",
    "DimensionsM",
    "FailureSignal",
    "Keypoint",
    "PalletModel",
    "PoseResult",
    "PoseUncertainty",
    "PositionM",
    "SopCheck",
    "StageTimings",
    "VerdictResult",
]
