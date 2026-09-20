"""External SOP-PAL-03 threshold configuration loader (R18, task 9.3).

This module makes the SOP thresholds *external configuration* (R18.1) that the
verifiable-subset checks (:mod:`.checks`) read at run time. Changing a value in
``configs/sop_thresholds.yaml`` changes the corresponding check boundary with
**no source-code modification** (R18.2). Each loaded threshold records:

- its numeric ``value`` (the boundary the check compares against),
- its ``unit`` (recorded into ``SopCheck.measurement_unit`` / documentation),
- a ``source`` string identifying exactly where the value came from — the file
  path plus the dotted config key, e.g.
  ``configs/sop_thresholds.yaml#thresholds.overhang.max_overhang_cm`` — so a
  ``SopCheck`` can record ``threshold_used`` and ``threshold_source`` (R18.3).

The loader is intentionally small and pure: it parses YAML into a typed,
frozen :class:`SopConfig` and validates the four numeric thresholds are present
and well-formed, failing loudly otherwise (no silent defaults that would hide a
misconfigured boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "DEFAULT_SOP_CONFIG_PATH",
    "Threshold",
    "SopConfig",
    "load_sop_config",
]

#: Default repo-relative location of the SOP thresholds config (R18.3).
DEFAULT_SOP_CONFIG_PATH = "configs/sop_thresholds.yaml"


@dataclass(frozen=True)
class Threshold:
    """A single configured threshold with its unit and provenance source.

    Attributes
    ----------
    value:
        The numeric boundary the check compares its measurement against. This
        is the value recorded as ``SopCheck.threshold_used``.
    unit:
        The unit of the threshold/measurement (e.g. ``"cm"``, ``"m"``,
        ``"deg"``), recorded as ``SopCheck.measurement_unit``.
    source:
        A string identifying where this value was read from — the config file
        path plus the dotted key — recorded as ``SopCheck.threshold_source``
        (R18.3).
    rule_id:
        The SOP-PAL-03 rule this threshold belongs to (1..8).
    """

    value: float
    unit: str
    source: str
    rule_id: int


@dataclass(frozen=True)
class SopConfig:
    """Typed, immutable view over ``configs/sop_thresholds.yaml`` (R18).

    Exposes the four configurable numeric thresholds of the verifiable subset,
    each as a :class:`Threshold` carrying its value, unit, and source string.
    A check reads ``config.<threshold>.value`` to derive its pass/fail boundary
    so that editing the YAML changes behaviour without any code change (R18.2).
    """

    overhang_max_cm: Threshold
    load_height_max_m: Threshold
    column_tilt_max_deg: Threshold
    centroid_offset_max_cm: Threshold


def _require_section(data: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    section = data.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(
            f"{path}: expected a mapping under '{key}', got {type(section).__name__}"
        )
    return section


def _require_number(
    section: Mapping[str, Any], key: str, source: str
) -> float:
    if key not in section:
        raise ValueError(f"{source}: missing required threshold key '{key}'")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{source}: threshold '{key}' must be a number, got {value!r}"
        )
    return float(value)


def _require_unit(section: Mapping[str, Any], source: str) -> str:
    unit = section.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"{source}: missing/blank 'unit' string")
    return unit


def _require_rule_id(section: Mapping[str, Any], source: str) -> int:
    rule_id = section.get("rule_id")
    if isinstance(rule_id, bool) or not isinstance(rule_id, int) or not (
        1 <= rule_id <= 8
    ):
        raise ValueError(f"{source}: 'rule_id' must be an int in 1..8, got {rule_id!r}")
    return rule_id


def _build_threshold(
    thresholds: Mapping[str, Any],
    section_key: str,
    value_key: str,
    file_ref: str,
) -> Threshold:
    """Read one threshold from the ``thresholds.<section_key>`` block."""
    section_path = f"thresholds.{section_key}"
    section = _require_section(thresholds, section_key, section_path)
    source = f"{file_ref}#{section_path}.{value_key}"
    value = _require_number(section, value_key, source)
    unit = _require_unit(section, source)
    rule_id = _require_rule_id(section, source)
    return Threshold(value=value, unit=unit, source=source, rule_id=rule_id)


def load_sop_config(
    config_path: str | Path = DEFAULT_SOP_CONFIG_PATH,
) -> SopConfig:
    """Load and validate the SOP thresholds from ``config_path`` (R18.1/18.2).

    Parameters
    ----------
    config_path:
        Path to the SOP thresholds YAML. Defaults to
        :data:`DEFAULT_SOP_CONFIG_PATH`.

    Returns
    -------
    SopConfig
        A frozen config exposing the four verifiable-subset thresholds, each
        with its ``value``, ``unit`` and ``source`` string. The ``source`` uses
        the given ``config_path`` (as provided) plus the dotted key so it stays
        meaningful regardless of where the file lives.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the YAML is malformed or a required threshold/unit/rule_id is
        missing or ill-typed (fail loudly; never silently default a boundary).
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"SOP thresholds config not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: top-level YAML must be a mapping")

    # Use the caller-provided path string in the source label so the recorded
    # threshold_source matches how the config was referenced (R18.3).
    file_ref = str(config_path)
    thresholds = _require_section(data, "thresholds", str(config_path))

    return SopConfig(
        overhang_max_cm=_build_threshold(
            thresholds, "overhang", "max_overhang_cm", file_ref
        ),
        load_height_max_m=_build_threshold(
            thresholds, "load_height", "max_height_m", file_ref
        ),
        column_tilt_max_deg=_build_threshold(
            thresholds, "column_tilt", "max_tilt_deg", file_ref
        ),
        centroid_offset_max_cm=_build_threshold(
            thresholds, "centroid_offset", "max_offset_cm", file_ref
        ),
    )
