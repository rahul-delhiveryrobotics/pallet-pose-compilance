"""External verdict-engine configuration loader (R17.6/R17.7, task 10.1).

This module makes the verdict thresholds *external configuration* that the
aggregation engine (:mod:`.engine`) reads at run time, mirroring the SOP
threshold loader (:mod:`pallet_pose_compliance.sop.config`). Changing a value
in ``configs/verdict.yaml`` changes the corresponding verdict boundary with
**no source-code modification**.

Every loaded value is documented in ``configs/verdict.yaml`` with a
justification (R17.6) and the pose-quality weighting is documented as a *gate*
rather than a weighted average (R17.7) so a confirmed critical failure can
never be offset (R17.5).

The loader is intentionally small and pure: it parses YAML into a typed, frozen
:class:`VerdictConfig` and validates every field is present and well-formed,
failing loudly otherwise (no silent defaults that would hide a misconfigured
boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "DEFAULT_VERDICT_CONFIG_PATH",
    "PoseQualityGate",
    "VerdictConfig",
    "load_verdict_config",
]

#: Default repo-relative location of the verdict config (R17.6/R17.7).
DEFAULT_VERDICT_CONFIG_PATH = "configs/verdict.yaml"


@dataclass(frozen=True)
class PoseQualityGate:
    """Pose-quality weighting settings, treated as a GATE (R17.7).

    Attributes
    ----------
    mode:
        Treatment of pose quality. ``"gate"`` invalidates pose-dependent checks
        when the pose is unreliable (degrading the verdict to
        ``MANUAL_INSPECTION``) rather than averaging pose quality against load
        compliance — so a confirmed critical failure can never be offset
        (R17.5/R17.7).
    max_position_std_cm:
        Maximum position uncertainty (1-sigma, cm) tolerated before the pose is
        deemed unreliable for pose-dependent SOP checks.
    max_orientation_std_deg:
        Maximum orientation uncertainty (1-sigma, deg) tolerated before the
        pose is deemed unreliable for pose-dependent SOP checks.
    source:
        A string identifying where these values were read from (config path +
        dotted key) for auditability.
    """

    mode: str
    max_position_std_cm: float
    max_orientation_std_deg: float
    source: str


@dataclass(frozen=True)
class VerdictConfig:
    """Typed, immutable view over ``configs/verdict.yaml`` (R17.6/R17.7).

    Attributes
    ----------
    mandatory_rule_ids:
        The SOP rule ids whose confirmed violation forces ``FAIL`` and whose
        adequate pass is required for ``PASS``.
    confirmed_violation_min_confidence:
        Per-check confidence at or above which a ``fail`` is a CONFIRMED
        violation (rather than inadequate evidence -> ``MANUAL_INSPECTION``).
    adequate_pass_min_confidence:
        Per-check confidence at or above which a ``pass`` is ADEQUATE evidence
        the rule is satisfied.
    pose_quality:
        The pose-quality gate settings (R17.7).
    source:
        The config path the values were read from (auditability).
    """

    mandatory_rule_ids: tuple[int, ...]
    confirmed_violation_min_confidence: float
    adequate_pass_min_confidence: float
    pose_quality: PoseQualityGate
    source: str


def _require_mapping(
    data: Mapping[str, Any], key: str, path: str
) -> Mapping[str, Any]:
    section = data.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(
            f"{path}: expected a mapping under '{key}', got {type(section).__name__}"
        )
    return section


def _require_number(section: Mapping[str, Any], key: str, source: str) -> float:
    if key not in section:
        raise ValueError(f"{source}: missing required key '{key}'")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: '{key}' must be a number, got {value!r}")
    return float(value)


def _require_confidence(section: Mapping[str, Any], key: str, source: str) -> float:
    value = _require_number(section, key, source)
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{source}: '{key}' must be a confidence in [0, 1], got {value!r}"
        )
    return value


def _require_string(section: Mapping[str, Any], key: str, source: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: missing/blank '{key}' string")
    return value


def _require_mandatory_rule_ids(data: Mapping[str, Any], source: str) -> tuple[int, ...]:
    raw = data.get("mandatory_rule_ids")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"{source}: 'mandatory_rule_ids' must be a non-empty list, got {raw!r}"
        )
    ids: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or not (1 <= item <= 8):
            raise ValueError(
                f"{source}: each mandatory rule id must be an int in 1..8, "
                f"got {item!r}"
            )
        ids.append(item)
    # Deduplicate while preserving order.
    seen: set[int] = set()
    unique = tuple(i for i in ids if not (i in seen or seen.add(i)))
    return unique


def _load_pose_quality(data: Mapping[str, Any], file_ref: str) -> PoseQualityGate:
    section = _require_mapping(data, "pose_quality", file_ref)
    source = f"{file_ref}#pose_quality"
    mode = _require_string(section, "mode", source)
    if mode != "gate":
        # The engine implements the documented GATE semantics; a weighted
        # average could offset a confirmed critical failure (R17.5), so we
        # refuse to silently accept another mode.
        raise ValueError(
            f"{source}: pose_quality.mode must be 'gate' (the documented "
            f"non-offsetting treatment, R17.7); got {mode!r}"
        )
    max_position_std_cm = _require_number(section, "max_position_std_cm", source)
    max_orientation_std_deg = _require_number(
        section, "max_orientation_std_deg", source
    )
    if max_position_std_cm < 0 or max_orientation_std_deg < 0:
        raise ValueError(f"{source}: pose-quality gate thresholds must be >= 0")
    return PoseQualityGate(
        mode=mode,
        max_position_std_cm=max_position_std_cm,
        max_orientation_std_deg=max_orientation_std_deg,
        source=source,
    )


def load_verdict_config(
    config_path: str | Path = DEFAULT_VERDICT_CONFIG_PATH,
) -> VerdictConfig:
    """Load and validate the verdict config from ``config_path`` (R17.6/R17.7).

    Parameters
    ----------
    config_path:
        Path to the verdict YAML. Defaults to :data:`DEFAULT_VERDICT_CONFIG_PATH`.

    Returns
    -------
    VerdictConfig
        A frozen config exposing the mandatory rule ids, the two confidence
        thresholds, and the pose-quality gate.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the YAML is malformed or a required field is missing or ill-typed
        (fail loudly; never silently default a boundary).
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"verdict config not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: top-level YAML must be a mapping")

    file_ref = str(config_path)
    return VerdictConfig(
        mandatory_rule_ids=_require_mandatory_rule_ids(data, file_ref),
        confirmed_violation_min_confidence=_require_confidence(
            data, "confirmed_violation_min_confidence", file_ref
        ),
        adequate_pass_min_confidence=_require_confidence(
            data, "adequate_pass_min_confidence", file_ref
        ),
        pose_quality=_load_pose_quality(data, file_ref),
        source=file_ref,
    )
