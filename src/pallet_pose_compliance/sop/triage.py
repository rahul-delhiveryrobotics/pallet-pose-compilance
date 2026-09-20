"""Authoritative SOP-PAL-03 eight-rule triage table (R14).

This module is the single source of truth for how each of the eight
SOP-PAL-03 rules is triaged against the available **single-side camera view**
(one fixed camera, ~1.2 m high, tilted ~20° down, observing each pallet from
one side only). Each rule is classified as ``verifiable``,
``partially_verifiable``, or ``not_verifiable`` and carries the justification,
assumptions, and the hidden regions that drive that classification (R14.1,
R14.2, R14.3). All eight rules are retained exactly once — none is omitted
(R14.4, design Property 14 shape).

Downstream consumers (task 9.3 verifiable-subset checks, 9.7 status/confidence
discipline, 9.10 pipeline integration) reuse :data:`TRIAGE_TABLE` and the
helpers here so rule names, classifications, and pose-dependence are defined in
one place rather than duplicated.

Classification reasoning (single-side view)
-------------------------------------------
The camera sees only the face(s) of the load turned toward it; the opposite
side, the interior of the stack, and the load's true 3D extent are not
observable. Given that:

- **Verifiable** rules can be decided from the visible face alone (load height,
  presence of stretch-wrap on the observed faces).
- **Partially verifiable** rules can be measured on the visible geometry but
  are *pose-dependent* and cannot be confirmed on hidden faces (overhang,
  column tilt, load-centroid offset). They are implemented but their
  confidence/validity is bounded by pose reliability and the unobserved side.
- **Not verifiable** rules depend on information the single-side view cannot
  recover: relative box sizes through the stack (size inversion), damage on
  hidden faces/interior boxes, and pallet structural integrity (deckboards /
  stringers largely occluded by the load and the floor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

__all__ = [
    "Classification",
    "TriageEntry",
    "TRIAGE_TABLE",
    "RULE_IDS",
    "get_entry",
    "all_entries",
    "rule_names",
    "classifications",
    "pose_dependent_rule_ids",
    "verify_table_complete",
]

#: The allowed triage classifications (matches ``SopCheck.triage`` literals).
Classification = Literal["verifiable", "partially_verifiable", "not_verifiable"]

#: The set of allowed classification string values, for validation/tests.
ALLOWED_CLASSIFICATIONS: frozenset[str] = frozenset(
    {"verifiable", "partially_verifiable", "not_verifiable"}
)

#: The complete, ordered set of SOP-PAL-03 rule ids.
RULE_IDS: tuple[int, ...] = tuple(range(1, 9))


@dataclass(frozen=True)
class TriageEntry:
    """One triaged SOP-PAL-03 rule (R14.1-R14.3).

    Attributes
    ----------
    rule_id:
        The rule number, 1..8.
    rule_name:
        Human-readable rule name (requirements Section 3).
    classification:
        One of ``verifiable`` / ``partially_verifiable`` / ``not_verifiable``
        from the single-side camera view (R14.1).
    justification:
        Why this classification holds given the single-side view (R14.2).
    assumptions:
        The assumptions made when triaging the rule (R14.3).
    hidden_regions:
        The regions the single-side view cannot observe that affect
        verifiability (R14.3). Not-verifiable rules must name at least one
        hidden region justifying non-verifiability.
    pose_dependent:
        Whether deciding this rule requires a reliable metric pose. Pose-
        dependent checks are invalidated when pose is unavailable (R15.4,
        consumed by task 9.7).
    """

    rule_id: int
    rule_name: str
    classification: Classification
    justification: str
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    hidden_regions: tuple[str, ...] = field(default_factory=tuple)
    pose_dependent: bool = False

    def __post_init__(self) -> None:
        if self.rule_id not in RULE_IDS:
            raise ValueError(f"rule_id must be in 1..8, got {self.rule_id!r}")
        if self.classification not in ALLOWED_CLASSIFICATIONS:
            raise ValueError(
                f"classification must be one of {sorted(ALLOWED_CLASSIFICATIONS)}, "
                f"got {self.classification!r}"
            )
        if not self.rule_name.strip():
            raise ValueError(f"rule {self.rule_id}: rule_name must be non-empty")
        if not self.justification.strip():
            raise ValueError(
                f"rule {self.rule_id}: justification must be non-empty (R14.2)"
            )
        # A not-verifiable rule must document at least one hidden region that
        # justifies why the single-side view cannot verify it (R14.3).
        if self.classification == "not_verifiable" and not self.hidden_regions:
            raise ValueError(
                f"rule {self.rule_id}: a not_verifiable rule must name at least "
                "one hidden region justifying non-verifiability (R14.3)"
            )


# ---------------------------------------------------------------------------
# The authoritative eight-rule triage table.
# ---------------------------------------------------------------------------

_ENTRIES: tuple[TriageEntry, ...] = (
    TriageEntry(
        rule_id=1,
        rule_name="No box overhang greater than 3 cm",
        classification="partially_verifiable",
        justification=(
            "Overhang of boxes past the pallet edge can be measured on the "
            "pallet edges and box faces turned toward the camera, but only once "
            "a reliable metric pose relates pixels to the pallet frame. The "
            "far side and rear edges are not imaged, so an overhang there cannot "
            "be confirmed or ruled out from a single view."
        ),
        assumptions=(
            "A reliable metric pose is available to convert pixel offsets to "
            "centimetres in the pallet frame.",
            "The pallet edge is visible and not fully occluded by the load.",
        ),
        hidden_regions=(
            "the rear (far-side) pallet edges not facing the camera",
            "box faces on the occluded side of the load",
        ),
        pose_dependent=True,
    ),
    TriageEntry(
        rule_id=2,
        rule_name="Load height no greater than 1.8 m",
        classification="verifiable",
        justification=(
            "The top of the load and the pallet base are both observable in the "
            "side view, so load height maps to a vertical extent that the "
            "calibrated geometry can convert to metres. Height is the most "
            "directly observable rule from a single side."
        ),
        assumptions=(
            "The top surface of the tallest part of the load is visible (not "
            "cut off by the frame or hidden behind a taller neighbour).",
            "Calibration relates image vertical extent to metric height at the "
            "load's range.",
        ),
        hidden_regions=(
            "any load protrusion taller than the visible front stack but hidden "
            "behind it",
        ),
    ),
    TriageEntry(
        rule_id=3,
        rule_name="Aligned columns: no box rotated more than 15 deg relative to the pallet axes",
        classification="partially_verifiable",
        justification=(
            "Rotation of boxes relative to the pallet axes can be estimated for "
            "boxes whose top/front edges are visible, but this needs the pallet "
            "axes from a reliable pose, and boxes in interior columns or on the "
            "far side cannot be measured."
        ),
        assumptions=(
            "A reliable metric pose defines the pallet axes to measure box "
            "rotation against.",
            "Box edges are sharp enough in the image to estimate orientation.",
        ),
        hidden_regions=(
            "interior columns occluded by the front row of boxes",
            "columns on the far side of the load",
        ),
        pose_dependent=True,
    ),
    TriageEntry(
        rule_id=4,
        rule_name="Larger boxes below smaller boxes (no size inversion)",
        classification="not_verifiable",
        justification=(
            "Verifying size ordering through the stack requires knowing each "
            "box's full dimensions at every layer. From one side the true depth "
            "of a box is unknown and boxes behind the front face are occluded, "
            "so a smaller box hidden beneath a larger one (or vice versa) cannot "
            "be detected."
        ),
        assumptions=(
            "Box depth (dimension away from the camera) is not observable from a "
            "single side without a second view or depth sensor.",
        ),
        hidden_regions=(
            "the depth extent of every box (front face only is imaged)",
            "boxes in interior layers hidden behind the front stack",
            "the far side of the load",
        ),
    ),
    TriageEntry(
        rule_id=5,
        rule_name="Load is stretch-wrapped",
        classification="verifiable",
        justification=(
            "The presence of stretch-wrap on the observed faces is a visual "
            "appearance cue (specular film, wrinkles, edge sheen) that the "
            "detector/analyzer can assess directly on the visible side, so its "
            "presence there is verifiable."
        ),
        assumptions=(
            "Wrap present on the observed faces is representative of the load "
            "being wrapped (a load wrapped only on the hidden side is out of "
            "scope of a single-side check).",
            "Lighting is adequate to distinguish film from bare cardboard.",
        ),
        hidden_regions=(
            "wrap coverage on the far side of the load (presence on the visible "
            "side is what is verified)",
        ),
    ),
    TriageEntry(
        rule_id=6,
        rule_name="No visibly damaged or crushed box",
        classification="not_verifiable",
        justification=(
            "Damage can only be seen where a box face is imaged. Crushing or "
            "tears on the far side, on interior boxes, or on the underside are "
            "not observable from a single view, so the absence of damage cannot "
            "be asserted for the whole load — only for the visible faces."
        ),
        assumptions=(
            "Damage on hidden faces is not inferable from the visible faces.",
        ),
        hidden_regions=(
            "the far side of the load",
            "faces of interior boxes hidden behind the front row",
            "the underside of boxes resting on lower layers",
        ),
    ),
    TriageEntry(
        rule_id=7,
        rule_name="Load centroid within 10 cm of the pallet centre",
        classification="partially_verifiable",
        justification=(
            "The lateral offset of the visible load mass relative to the pallet "
            "centre can be estimated once a reliable pose locates the pallet "
            "centre, but the true 3D centroid depends on the unobserved depth "
            "and far-side mass distribution, so the estimate is bounded and "
            "pose-dependent."
        ),
        assumptions=(
            "A reliable metric pose locates the pallet centre in the floor frame.",
            "The visible load extent is a usable proxy for lateral centroid "
            "offset; depth-direction offset is only weakly observable.",
        ),
        hidden_regions=(
            "the depth (camera-axis) distribution of load mass",
            "mass on the far side of the load",
        ),
        pose_dependent=True,
    ),
    TriageEntry(
        rule_id=8,
        rule_name="Pallet undamaged: no broken boards or split stringers",
        classification="not_verifiable",
        justification=(
            "The pallet structure (deckboards and stringers) is largely occluded "
            "by the load above and the floor below; from a side view most of the "
            "load-bearing structure is hidden, so broken boards or split "
            "stringers cannot be reliably detected."
        ),
        assumptions=(
            "Structural integrity of hidden boards/stringers cannot be inferred "
            "from the small visible portion of the pallet.",
        ),
        hidden_regions=(
            "deckboards and stringers occluded by the load resting on the pallet",
            "the underside and far-side stringers of the pallet",
            "the internal structure of the pallet",
        ),
    ),
)


#: The authoritative triage table: an ordered, read-only mapping
#: ``rule_id -> TriageEntry`` with each rule id 1..8 present exactly once
#: (design Property 14 shape). Built here so the invariant is guaranteed at
#: import time via :func:`verify_table_complete`.
def _build_table(entries: tuple[TriageEntry, ...]) -> Mapping[int, TriageEntry]:
    table: dict[int, TriageEntry] = {}
    for entry in entries:
        if entry.rule_id in table:
            raise ValueError(
                f"duplicate triage entry for rule_id {entry.rule_id}; each rule "
                "must appear exactly once (R14.4)"
            )
        table[entry.rule_id] = entry
    ordered = {rid: table[rid] for rid in sorted(table)}
    return MappingProxyType(ordered)


TRIAGE_TABLE: Mapping[int, TriageEntry] = _build_table(_ENTRIES)


def verify_table_complete(
    table: Mapping[int, TriageEntry] = TRIAGE_TABLE,
) -> bool:
    """Return True iff ``table`` contains rule ids 1..8 exactly once, each valid.

    This is the invariant the design's Property 14 asserts (the property test
    itself is out of scope for this task). It checks that every rule id 1..8 is
    present exactly once, that each entry's ``rule_id`` matches its key, and
    that every classification is in the allowed set. Raises ``ValueError`` on
    violation so misuse fails loudly.
    """
    ids = sorted(table.keys())
    if ids != list(RULE_IDS):
        raise ValueError(
            f"triage table must contain rule_id 1..8 exactly once, got {ids}"
        )
    for rid, entry in table.items():
        if entry.rule_id != rid:
            raise ValueError(
                f"triage entry keyed {rid} has mismatched rule_id {entry.rule_id}"
            )
        if entry.classification not in ALLOWED_CLASSIFICATIONS:
            raise ValueError(
                f"rule {rid}: invalid classification {entry.classification!r}"
            )
    return True


# Fail loudly at import time if the authoritative table is malformed.
verify_table_complete(TRIAGE_TABLE)


# ---------------------------------------------------------------------------
# Accessors reused by tasks 9.3 / 9.7 / 9.10.
# ---------------------------------------------------------------------------


def get_entry(rule_id: int) -> TriageEntry:
    """Return the :class:`TriageEntry` for ``rule_id`` (1..8).

    Raises
    ------
    KeyError
        If ``rule_id`` is not one of the eight SOP-PAL-03 rules.
    """
    try:
        return TRIAGE_TABLE[rule_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no SOP-PAL-03 rule with id {rule_id!r} (valid: 1..8)") from exc


def all_entries() -> tuple[TriageEntry, ...]:
    """Return all eight triage entries in ascending rule-id order."""
    return tuple(TRIAGE_TABLE[rid] for rid in RULE_IDS)


def rule_names() -> Mapping[int, str]:
    """Return a read-only ``rule_id -> rule_name`` mapping for reuse."""
    return MappingProxyType({rid: TRIAGE_TABLE[rid].rule_name for rid in RULE_IDS})


def classifications() -> Mapping[int, Classification]:
    """Return a read-only ``rule_id -> classification`` mapping for reuse."""
    return MappingProxyType(
        {rid: TRIAGE_TABLE[rid].classification for rid in RULE_IDS}
    )


def pose_dependent_rule_ids() -> frozenset[int]:
    """Return the ids of rules whose evaluation depends on a reliable pose."""
    return frozenset(
        rid for rid in RULE_IDS if TRIAGE_TABLE[rid].pose_dependent
    )
