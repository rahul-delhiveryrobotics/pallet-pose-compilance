"""Stub pose estimator for the early end-to-end smoke pipeline (task 3.1).

Returns a defined *unavailable* :class:`PoseResult` for every detection, so the
pipeline exercises the honest unavailable-pose path from the very first
end-to-end run. The real known-geometry PnP estimator arrives in task 6.2/6.10.

Emitting ``pose_status = "unavailable"`` with a ``Reason_Code`` and null metric
fields (never a fabricated/zero pose) is exactly the degrade-loudly behaviour
required by the design (Property 8, R13.4) — it is the correct placeholder while
no calibration or PnP solver is wired.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..output.provenance import ProvenanceLabel, QualityFlag, ReasonCode
from ..output.schema import Detection, PoseResult

__all__ = ["estimate_pose_stub"]


def estimate_pose_stub(
    detection: Optional[Detection] = None,
    *,
    reason_code: ReasonCode = ReasonCode.POSE_INSUFFICIENT_KEYPOINTS,
    quality_flags: Sequence[QualityFlag] = (QualityFlag.INSUFFICIENT_KEYPOINTS,),
) -> PoseResult:
    """Return an *unavailable* :class:`PoseResult` (stub for task 6.10).

    No metric pose is produced because no calibration/PnP solver is wired yet;
    rather than fabricate a pose, the stub returns the defined unavailable
    output with a reason code and null metric fields (R13.4).

    Parameters
    ----------
    detection:
        The detection this pose corresponds to. Currently unused by the stub
        (kept for interface compatibility with the real estimator).
    reason_code:
        The reason the pose is unavailable. Defaults to
        ``POSE_INSUFFICIENT_KEYPOINTS`` (no usable calibrated keypoints yet).
    quality_flags:
        Quality flags annotating the degraded condition.

    Returns
    -------
    PoseResult
        A schema-valid pose result with ``pose_status = "unavailable"``.
    """
    return PoseResult(
        pose_status="unavailable",
        reason_code=reason_code,
        position_m=None,
        orientation_deg=None,
        face_identity=None,
        quality_flags=list(quality_flags),
        provenance=ProvenanceLabel.UNAVAILABLE,
    )
