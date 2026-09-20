"""Minimal verdict engine for the early end-to-end smoke pipeline (task 3.1).

Produces a schema-valid :class:`VerdictResult`. The real aggregation logic with
critical-failure dominance arrives in task 10.1; this stub implements only the
minimal honest behaviour needed for the smoke pipeline:

- When the pose is unavailable (or any mandatory check is unresolved/invalidated
  with no confirmed violation), the verdict is ``MANUAL_INSPECTION`` — inadequate
  evidence, no confirmed violation (design: Verdict aggregation, R17.4).

Since the stub SOP analyzer never emits a confirmed pass/fail, this stub always
returns ``MANUAL_INSPECTION`` with an honest reasoning string.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..output.provenance import ReasonCode
from ..output.schema import PoseResult, SopCheck, VerdictResult

__all__ = ["aggregate_stub"]


def aggregate_stub(
    pose_result: Optional[PoseResult],
    sop_checks: Sequence[SopCheck],
) -> VerdictResult:
    """Return a minimal, schema-valid verdict (stub for task 10.1).

    With no confirmed pass/fail evidence available from the stub SOP analyzer
    and an unavailable pose, the only honest verdict is
    ``MANUAL_INSPECTION`` (inadequate evidence, no confirmed violation).

    Parameters
    ----------
    pose_result:
        The pallet's pose result (unavailable in the smoke pipeline).
    sop_checks:
        The eight SOP checks for the pallet.

    Returns
    -------
    VerdictResult
        A ``MANUAL_INSPECTION`` verdict with reason code and reasoning text.
    """
    pose_available = pose_result is not None and pose_result.pose_status == "available"

    reason_codes: list[ReasonCode] = []
    if not pose_available:
        reason_codes.append(ReasonCode.POSE_INSUFFICIENT_KEYPOINTS)

    contributing = [c.rule_id for c in sop_checks if c.status in ("unresolved", "invalidated")]

    reasoning = (
        "MANUAL_INSPECTION (stub verdict): no confirmed rule pass/fail evidence "
        "is available and the pose is "
        f"{'available' if pose_available else 'unavailable'}; "
        "insufficient evidence for a PASS/FAIL verdict."
    )

    return VerdictResult(
        verdict="MANUAL_INSPECTION",
        reason_codes=reason_codes,
        reasoning_text=reasoning,
        pose_quality_weighting=None,
        contributing_checks=contributing,
    )
