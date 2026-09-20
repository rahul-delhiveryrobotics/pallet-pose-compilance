"""Stub SOP analyzer for the early end-to-end smoke pipeline (task 3.1).

Returns all eight SOP-PAL-03 rule results as *triage-only* placeholders so the
Assessment always carries the complete eight-rule set (Property 14 shape). The
real analyzer with the implemented verifiable subset arrives in task 9.10.

Status discipline observed even in the stub (design: Property 16, R15.3/R15.4):

- Rules triaged ``not_verifiable`` are reported as ``unresolved`` (never
  pass/fail/omitted).
- When pose is unavailable, pose-dependent checks are reported as
  ``invalidated`` with a ``Reason_Code`` (the smoke pipeline runs with the stub
  unavailable pose, so pose-dependent rules are invalidated here).
- No check is reported as ``pass``/``fail`` because no measurement logic exists
  yet — everything is honestly unresolved/invalidated.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..output.provenance import ProvenanceLabel, ReasonCode
from ..output.schema import Detection, PoseResult, SopCheck

__all__ = ["analyze_stub", "RULE_NAMES", "TRIAGE", "POSE_DEPENDENT_RULE_IDS"]

#: Human-readable names for the eight SOP-PAL-03 rules (requirements Section 3).
RULE_NAMES: dict[int, str] = {
    1: "No box overhang greater than 3 cm",
    2: "Load height no greater than 1.8 m",
    3: "Aligned columns: no box rotated more than 15 deg",
    4: "Larger boxes below smaller boxes (no size inversion)",
    5: "Load is stretch-wrapped",
    6: "No visibly damaged or crushed box",
    7: "Load centroid within 10 cm of pallet centre",
    8: "Pallet undamaged (no broken boards/split stringers)",
}

#: Placeholder triage classification per rule. This mirrors the design's
#: single-side-view reasoning; the authoritative triage table is built in
#: task 9.1. Rules 4/6/8 (size inversion, box damage, pallet damage) are not
#: verifiable from a single-side view.
TRIAGE: dict[int, str] = {
    1: "partially_verifiable",   # overhang (needs metric pose)
    2: "verifiable",             # load height
    3: "partially_verifiable",   # column tilt (needs pose axes)
    4: "not_verifiable",         # size inversion (hidden interior)
    5: "verifiable",             # stretch-wrap presence
    6: "not_verifiable",         # box damage (hidden faces)
    7: "partially_verifiable",   # centroid offset (needs metric pose)
    8: "not_verifiable",         # pallet damage (hidden structure)
}

#: Rules whose evaluation depends on a reliable metric pose. When pose is
#: unavailable these are invalidated (R15.4).
POSE_DEPENDENT_RULE_IDS: frozenset[int] = frozenset({1, 3, 7})


def _pose_is_available(pose: Optional[PoseResult]) -> bool:
    return pose is not None and pose.pose_status == "available"


def analyze_stub(
    detections: Optional[Sequence[Detection]] = None,
    pose_result: Optional[PoseResult] = None,
) -> list[SopCheck]:
    """Return all eight SOP checks as triage-only placeholders (stub for 9.10).

    Parameters
    ----------
    detections:
        Detections for the pallet. Unused by the stub (kept for interface
        compatibility with the real analyzer).
    pose_result:
        The pallet's pose. When unavailable, pose-dependent checks are marked
        ``invalidated`` with a reason code.

    Returns
    -------
    list[SopCheck]
        Exactly eight checks (rule_id 1..8), each ``unresolved`` or, for
        pose-dependent rules with an unavailable pose, ``invalidated``.
    """
    pose_available = _pose_is_available(pose_result)

    checks: list[SopCheck] = []
    for rule_id in range(1, 9):
        triage = TRIAGE[rule_id]
        pose_dependent = rule_id in POSE_DEPENDENT_RULE_IDS

        if pose_dependent and not pose_available:
            # Pose-dependent check invalidated because pose is unavailable.
            status = "invalidated"
            reason_code: Optional[ReasonCode] = ReasonCode.POSE_INSUFFICIENT_KEYPOINTS
        else:
            # Everything else is honestly unresolved in the stub (no logic yet).
            status = "unresolved"
            reason_code = None

        checks.append(
            SopCheck(
                rule_id=rule_id,
                rule_name=RULE_NAMES[rule_id],
                triage=triage,  # type: ignore[arg-type]
                status=status,  # type: ignore[arg-type]
                confidence=None,
                confidence_semantics=None,
                measurement=None,
                measurement_unit=None,
                reason_code=reason_code,
                assumptions=["stub analyzer: no measurement logic implemented yet"],
                provenance=ProvenanceLabel.UNAVAILABLE,
            )
        )
    return checks
