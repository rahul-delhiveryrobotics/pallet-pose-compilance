"""Stub detector for the early end-to-end smoke pipeline (task 3.1).

This is a placeholder Detector that returns fixed, schema-valid detections so
the whole pipeline (ingest -> detect -> pose -> SOP -> verdict -> Assessment)
can run before the real YOLO-pose detector exists. It is replaced by the
trained/wrapped detector in task 4.8 (design: Detector, R4/R5).

The stub does **not** claim to be a trained detector: its detections carry a
placeholder confidence and the keypoints are synthetic. Nothing here is
presented as a measured result.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..output.schema import Detection, Keypoint

__all__ = ["StubDetector", "detect_stub"]

# Names of the eight pallet structural keypoints (four bottom + four top deck
# corners) matching the annotation schema (design: Keypoint / corner schema).
_KEYPOINT_NAMES = (
    "bottom_corner_0",
    "bottom_corner_1",
    "bottom_corner_2",
    "bottom_corner_3",
    "top_corner_0",
    "top_corner_1",
    "top_corner_2",
    "top_corner_3",
)


def _fixed_keypoints(width: int, height: int) -> list[Keypoint]:
    """Return eight fixed keypoints arranged as a plausible pallet quad.

    The exact pixel positions are placeholders; they are only required to be
    schema-valid for the smoke pipeline. Bottom corners form a wider quad near
    the image lower half; top corners form a narrower quad above them.
    """
    cx, cy = width / 2.0, height * 0.6
    bw, bh = width * 0.30, height * 0.12  # bottom quad half-extents
    tw, th = width * 0.22, height * 0.10  # top quad half-extents
    top_dy = height * 0.18  # top deck raised in image space

    bottom = [
        (cx + bw, cy + bh),
        (cx - bw, cy + bh),
        (cx - bw, cy - bh),
        (cx + bw, cy - bh),
    ]
    top = [
        (cx + tw, cy - top_dy + th),
        (cx - tw, cy - top_dy + th),
        (cx - tw, cy - top_dy - th),
        (cx + tw, cy - top_dy - th),
    ]
    coords = list(bottom) + list(top)

    keypoints: list[Keypoint] = []
    for name, (u, v) in zip(_KEYPOINT_NAMES, coords):
        keypoints.append(
            Keypoint(
                name=name,
                u=float(u),
                v=float(v),
                visibility="visible",
                score=0.5,
            )
        )
    return keypoints


class StubDetector:
    """A fixed-output Detector standing in for the trained detector (task 4.8).

    Returns a fixed number of pallet detections regardless of image content, so
    the end-to-end pipeline is runnable. Replace with the real detector wrapper
    in task 4.8.
    """

    def __init__(self, num_pallets: int = 1, score: float = 0.5) -> None:
        if num_pallets < 0:
            raise ValueError("num_pallets must be non-negative")
        if not (0.0 <= score <= 1.0):
            raise ValueError("score must be in [0, 1]")
        self.num_pallets = num_pallets
        self.score = score

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return fixed pallet detections for ``image`` (content ignored).

        Parameters
        ----------
        image:
            The ingested frame image (``H x W x 3``). Only its shape is used to
            place plausible keypoints.
        """
        if image is None or getattr(image, "ndim", 0) < 2:
            raise ValueError("StubDetector.detect requires an H x W[x3] image")
        height, width = int(image.shape[0]), int(image.shape[1])

        detections: list[Detection] = []
        for i in range(self.num_pallets):
            # Place each pallet box in a horizontal slot across the frame.
            slot_w = width / max(self.num_pallets, 1)
            x = i * slot_w + slot_w * 0.2
            w = slot_w * 0.6
            y = height * 0.45
            h = height * 0.35
            detections.append(
                Detection(
                    cls="pallet",
                    bbox=[float(x), float(y), float(w), float(h)],
                    score=float(self.score),
                    keypoints=_fixed_keypoints(width, height),
                )
            )
        return detections


def detect_stub(
    image: np.ndarray,
    *,
    num_pallets: int = 1,
    detector: Optional[StubDetector] = None,
) -> list[Detection]:
    """Convenience wrapper returning fixed detections (see :class:`StubDetector`)."""
    det = detector if detector is not None else StubDetector(num_pallets=num_pallets)
    return det.detect(image)
