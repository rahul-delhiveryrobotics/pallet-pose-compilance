"""Per-pallet verdict aggregation with critical-failure dominance (task 10.1).

This module replaces the task-3.1 stub (:mod:`.stub`) with the real
:func:`aggregate` engine that turns a pose result and the eight SOP checks into
a single defensible :class:`~pallet_pose_compliance.output.schema.VerdictResult`
of ``PASS`` / ``FAIL`` / ``MANUAL_INSPECTION`` (R17.1).

Aggregation rules (design: VerdictResult + Error Handling, R17)
---------------------------------------------------------------
Let a *mandatory* rule be one listed in ``verdict_config.mandatory_rule_ids``.

1. **FAIL — confirmed mandatory violation (R17.2, R17.5).** If any mandatory
   rule has ``status == "fail"`` with ``confidence >= confirmed_violation_min_confidence``
   the verdict is ``FAIL``. This is evaluated FIRST and dominates: adding any
   number of passing checks can never change a ``FAIL`` to ``PASS`` or
   ``MANUAL_INSPECTION`` (critical-failure dominance — the engine never averages
   per-check results in a way that offsets a confirmed critical failure, R17.5).

2. **PASS — adequate evidence all mandatory pass (R17.3).** Otherwise, if there
   is no confirmed violation AND every mandatory rule has ``status == "pass"``
   with ``confidence >= adequate_pass_min_confidence`` (adequate positive
   evidence) AND the pose-quality gate is satisfied, the verdict is ``PASS``.

3. **MANUAL_INSPECTION — inadequate evidence, no confirmed violation (R17.4).**
   Otherwise (one or more mandatory rules are ``unresolved`` / ``invalidated`` /
   low-confidence pass / low-confidence fail, or the pose-quality gate is not
   satisfied) the verdict is ``MANUAL_INSPECTION``.

Pose-quality weighting is a GATE, not a weighted average (R17.7)
----------------------------------------------------------------
Pose quality is treated as a *gate* on a ``PASS``: when the pose is unavailable
or its 1-sigma uncertainty exceeds the configured thresholds, the pose is
unreliable for the pose-dependent checks, so a clean ``PASS`` degrades to
``MANUAL_INSPECTION``. Pose quality is never averaged against load compliance,
so it can never offset a confirmed critical failure (R17.5). The applied
pose-quality weight is recorded in ``VerdictResult.pose_quality_weighting``
(``1.0`` = pose adequate / full weight; ``0.0`` = pose gated out) and the
reasoning text documents the gate treatment (R17.7).

Every verdict records machine-readable ``reason_codes``, a human-readable
``reasoning_text``, the ``contributing_checks`` (the mandatory rule ids that
drove the outcome), and the documented ``pose_quality_weighting`` (R17.8).
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..output.provenance import ReasonCode
from ..output.schema import PoseResult, SopCheck, VerdictResult
from .config import VerdictConfig

__all__ = ["aggregate"]

#: Pose-quality weight when the pose is adequate for pose-dependent checks
#: (full weight — gate open).
_POSE_WEIGHT_ADEQUATE = 1.0
#: Pose-quality weight when the pose is gated out (unavailable / too uncertain).
#: Zero, NOT a fractional average — pose quality is a gate, never offsetting a
#: confirmed critical failure (R17.5/R17.7).
_POSE_WEIGHT_GATED = 0.0

#: Centimetres per metre, for converting the pose position std to the config's
#: cm gate threshold.
_CM_PER_M = 100.0


def _position_std_cm(pose_result: PoseResult) -> Optional[float]:
    """Return the larger principal position std (cm) from the pose uncertainty.

    Reads the 2x2 ``position_cov_m2`` from the pose's :class:`PoseUncertainty`
    and returns the square root of its larger eigenvalue converted to
    centimetres. Returns ``None`` when the covariance is unavailable (the pose
    is then treated as failing the gate — we cannot show it is reliable).
    """
    unc = pose_result.uncertainty
    if unc is None or unc.position_cov_m2 is None:
        return None
    cov = unc.position_cov_m2
    # 2x2 symmetric covariance; the larger principal std is sqrt of the larger
    # eigenvalue. Compute eigenvalues in closed form to avoid a numpy import
    # here (small, fixed-size matrix).
    a = cov[0][0]
    b = cov[0][1]
    c = cov[1][0]
    d = cov[1][1]
    trace = a + d
    det = a * d - b * c
    disc = max(0.0, (trace * trace) / 4.0 - det)
    max_eig = trace / 2.0 + disc**0.5
    if max_eig < 0.0:
        return None
    return (max_eig**0.5) * _CM_PER_M


def _pose_gate_satisfied(
    pose_result: Optional[PoseResult], config: VerdictConfig
) -> tuple[bool, Optional[str]]:
    """Evaluate the pose-quality GATE (R17.7).

    Returns ``(satisfied, note)`` where ``satisfied`` is ``True`` when the pose
    is available and its 1-sigma uncertainty is within the configured
    thresholds, and ``note`` is a short human-readable reason when the gate is
    NOT satisfied (``None`` when satisfied).
    """
    gate = config.pose_quality
    if pose_result is None or pose_result.pose_status != "available":
        return False, "pose unavailable"

    pos_std_cm = _position_std_cm(pose_result)
    orient_std = (
        pose_result.uncertainty.orientation_std_deg
        if pose_result.uncertainty is not None
        else None
    )

    if pos_std_cm is None or orient_std is None:
        return False, "pose uncertainty unquantified"

    if pos_std_cm > gate.max_position_std_cm:
        return (
            False,
            f"pose position uncertainty {pos_std_cm:.2f} cm exceeds gate "
            f"{gate.max_position_std_cm:.2f} cm",
        )
    if orient_std > gate.max_orientation_std_deg:
        return (
            False,
            f"pose orientation uncertainty {orient_std:.2f} deg exceeds gate "
            f"{gate.max_orientation_std_deg:.2f} deg",
        )
    return True, None


def _mandatory_checks(
    sop_checks: Sequence[SopCheck], config: VerdictConfig
) -> list[SopCheck]:
    mandatory = set(config.mandatory_rule_ids)
    return [c for c in sop_checks if c.rule_id in mandatory]


def _is_confirmed_violation(check: SopCheck, config: VerdictConfig) -> bool:
    """A mandatory FAIL with adequate confidence is a CONFIRMED violation (R17.2)."""
    if check.status != "fail":
        return False
    conf = check.confidence
    return conf is not None and conf >= config.confirmed_violation_min_confidence


def _is_adequate_pass(check: SopCheck, config: VerdictConfig) -> bool:
    """A mandatory PASS with adequate positive-evidence confidence (R17.3)."""
    if check.status != "pass":
        return False
    conf = check.confidence
    return conf is not None and conf >= config.adequate_pass_min_confidence


def aggregate(
    pose_result: Optional[PoseResult],
    sop_checks: Sequence[SopCheck],
    verdict_config: VerdictConfig,
) -> VerdictResult:
    """Aggregate pose + SOP checks into a per-pallet verdict (R17).

    See the module docstring for the full rule set. In short:

    - a confirmed mandatory violation -> ``FAIL`` (dominant, never averaged
      away, R17.2/R17.5);
    - else adequate evidence all mandatory pass AND pose gate satisfied ->
      ``PASS`` (R17.3);
    - else -> ``MANUAL_INSPECTION`` (inadequate evidence, no confirmed
      violation, R17.4).

    Parameters
    ----------
    pose_result:
        The pallet's pose result (may be ``None`` or unavailable). Used only for
        the pose-quality gate on a ``PASS`` (R17.7) — never to offset a
        confirmed critical failure.
    sop_checks:
        The eight SOP checks for the pallet.
    verdict_config:
        The loaded verdict config (mandatory rule ids + confidence thresholds +
        pose-quality gate). All thresholds come from here; none are hardcoded
        (R17.6).

    Returns
    -------
    VerdictResult
        The verdict with reason codes, human-readable reasoning, the mandatory
        contributing checks, and the documented pose-quality weighting (R17.8).
    """
    mandatory = _mandatory_checks(sop_checks, verdict_config)

    # --- 1. Critical-failure dominance: confirmed mandatory violation -> FAIL.
    # Evaluated FIRST so that adding passing checks can never offset it (R17.5).
    confirmed_violations = [
        c for c in mandatory if _is_confirmed_violation(c, verdict_config)
    ]
    if confirmed_violations:
        # The pose-quality gate is NOT consulted here: a confirmed critical
        # failure dominates regardless of pose quality (R17.5).
        offending_ids = [c.rule_id for c in confirmed_violations]
        details = ", ".join(
            f"rule {c.rule_id} ({c.rule_name}) fail @ conf "
            f"{c.confidence:.2f} >= {verdict_config.confirmed_violation_min_confidence:.2f}"
            for c in confirmed_violations
        )
        reasoning = (
            "FAIL: confirmed mandatory violation(s) — "
            f"{details}. A confirmed critical failure dominates and is never "
            "offset by passing checks or pose quality (R17.2/R17.5). "
            "Pose quality is a gate, not a weighted average, so it cannot "
            "average away this failure (R17.7)."
        )
        # A compliance FAIL is not a processing failure, so no ReasonCode is
        # attached; the reasoning_text documents the offending rule(s).
        return VerdictResult(
            verdict="FAIL",
            reason_codes=[],
            reasoning_text=reasoning,
            pose_quality_weighting=_POSE_WEIGHT_GATED,
            contributing_checks=sorted(offending_ids),
        )

    # No confirmed violation. Assess adequacy of positive evidence for PASS.
    inadequate: list[tuple[int, str]] = []
    for c in mandatory:
        if _is_adequate_pass(c, verdict_config):
            continue
        if c.status == "pass":
            conf = c.confidence if c.confidence is not None else 0.0
            inadequate.append(
                (
                    c.rule_id,
                    f"rule {c.rule_id} pass but confidence {conf:.2f} < "
                    f"{verdict_config.adequate_pass_min_confidence:.2f}",
                )
            )
        elif c.status == "fail":
            conf = c.confidence if c.confidence is not None else 0.0
            inadequate.append(
                (
                    c.rule_id,
                    f"rule {c.rule_id} fail but confidence {conf:.2f} < "
                    f"{verdict_config.confirmed_violation_min_confidence:.2f} "
                    "(unconfirmed)",
                )
            )
        else:  # unresolved / invalidated
            inadequate.append(
                (c.rule_id, f"rule {c.rule_id} {c.status} (no adequate evidence)")
            )

    pose_ok, pose_note = _pose_gate_satisfied(pose_result, verdict_config)
    pose_weight = _POSE_WEIGHT_ADEQUATE if pose_ok else _POSE_WEIGHT_GATED

    # --- 2. PASS: adequate evidence for every mandatory rule AND pose gate open.
    if not inadequate and pose_ok:
        mandatory_ids = sorted(c.rule_id for c in mandatory)
        reasoning = (
            "PASS: adequate evidence that all mandatory rules "
            f"{mandatory_ids} are satisfied (each pass at confidence >= "
            f"{verdict_config.adequate_pass_min_confidence:.2f}) and the "
            "pose-quality gate is satisfied. Pose quality is applied as a gate "
            "(weight 1.0 = adequate), not averaged against load compliance "
            "(R17.3/R17.7)."
        )
        return VerdictResult(
            verdict="PASS",
            reason_codes=[],
            reasoning_text=reasoning,
            pose_quality_weighting=pose_weight,
            contributing_checks=mandatory_ids,
        )

    # --- 3. MANUAL_INSPECTION: inadequate evidence and no confirmed violation.
    reason_codes: list[ReasonCode] = []
    contributing: list[int] = sorted({rid for rid, _ in inadequate})

    reason_bits: list[str] = [note for _, note in inadequate]
    if not pose_ok:
        reason_bits.append(f"pose-quality gate not satisfied ({pose_note})")
        # Surface the pose's own reason code when it explains the gate failure.
        if pose_result is not None and pose_result.reason_code is not None:
            reason_codes.append(pose_result.reason_code)
        # Any pose-dependent mandatory checks that were invalidated by the pose
        # loss are contributing factors too (already captured in ``inadequate``
        # via their invalidated status).

    reasoning = (
        "MANUAL_INSPECTION: no confirmed mandatory violation, but evidence is "
        "inadequate for a PASS — "
        + ("; ".join(reason_bits) if reason_bits else "insufficient evidence")
        + ". Pose quality is applied as a gate (weight "
        + f"{pose_weight:.1f}), never averaged against load compliance, so no "
        "confirmed critical failure could be offset (R17.4/R17.7)."
    )

    return VerdictResult(
        verdict="MANUAL_INSPECTION",
        reason_codes=reason_codes,
        reasoning_text=reasoning,
        pose_quality_weighting=pose_weight,
        contributing_checks=contributing,
    )
