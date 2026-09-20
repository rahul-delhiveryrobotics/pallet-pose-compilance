"""Ingest stage: load/reference a frame, attach metadata, validate integrity.

This is the first pipeline stage (design: Pipeline stages, step 1). It loads or
references an input frame, attaches a source-image reference and an ISO-8601
timestamp, and performs a basic integrity check (file exists, is readable, and
can be decoded as an image). It also supports generating a small synthetic
placeholder frame so the end-to-end smoke pipeline can run without real data.

The ingest stage never fabricates image content silently: if a referenced file
cannot be read or decoded, it raises :class:`IngestError` so the pipeline can
emit a ``PROCESSING_FAILED`` failure signal rather than continuing on a bad
frame (design: Error Handling — corrupt input).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:  # OpenCV is a declared dependency; import defensively for clear errors.
    import cv2
except Exception as exc:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR: Optional[Exception] = exc
else:
    _CV2_IMPORT_ERROR = None

__all__ = [
    "IngestError",
    "Frame",
    "ingest_image",
    "make_synthetic_frame",
]


class IngestError(RuntimeError):
    """Raised when a frame cannot be loaded, read, or decoded (integrity fail)."""


@dataclass(frozen=True)
class Frame:
    """An ingested frame with provenance metadata.

    Attributes
    ----------
    image:
        The decoded image as an ``H x W x 3`` uint8 numpy array (BGR, OpenCV
        convention) or a synthetic placeholder.
    source_image_ref:
        A reference to the source image: the resolved file path for a real
        image, or a synthetic identifier (e.g. ``synthetic://...``) for a
        generated placeholder. Recorded in every Assessment (R24.6).
    timestamp:
        The ingest timestamp as an ISO-8601 string in UTC (R24.6).
    width, height:
        Frame dimensions in pixels.
    """

    image: np.ndarray
    source_image_ref: str
    timestamp: str
    width: int
    height: int


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def make_synthetic_frame(
    width: int = 640,
    height: int = 640,
    *,
    ref: str = "synthetic://placeholder-640x640",
) -> Frame:
    """Generate a small synthetic placeholder frame for the smoke pipeline.

    The frame is a mid-grey image so downstream stub stages have a valid,
    decodable image to reference. It is clearly labelled via a
    ``synthetic://`` source reference so it can never be mistaken for a real
    captured frame (honesty discipline).

    Parameters
    ----------
    width, height:
        Placeholder dimensions in pixels.
    ref:
        The synthetic source-image reference recorded in the Assessment.
    """
    if width <= 0 or height <= 0:
        raise IngestError(
            f"synthetic frame dimensions must be positive (got {width}x{height})"
        )
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    return Frame(
        image=image,
        source_image_ref=ref,
        timestamp=_now_iso(),
        width=width,
        height=height,
    )


def ingest_image(image_path: "str | Path") -> Frame:
    """Load and integrity-check a frame from ``image_path``.

    Performs the basic integrity checks required of the ingest stage:

    - the path exists and is a regular file,
    - the file is readable and can be decoded as an image.

    Parameters
    ----------
    image_path:
        Path to the image file to ingest.

    Returns
    -------
    Frame
        The decoded frame with source reference and ISO-8601 timestamp.

    Raises
    ------
    IngestError
        If the file is missing, unreadable, or cannot be decoded as an image.
    """
    if cv2 is None:  # pragma: no cover - environment dependent
        raise IngestError(
            "OpenCV (cv2) is required to decode images but failed to import: "
            f"{_CV2_IMPORT_ERROR!r}"
        )

    path = Path(image_path)
    if not path.exists():
        raise IngestError(f"input frame does not exist: {path}")
    if not path.is_file():
        raise IngestError(f"input frame is not a regular file: {path}")

    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception as exc:  # pragma: no cover - defensive
        raise IngestError(f"failed to read frame {path}: {exc!r}") from exc

    if image is None or image.size == 0:
        raise IngestError(
            f"frame {path} could not be decoded as an image (corrupt/unsupported)"
        )

    height, width = int(image.shape[0]), int(image.shape[1])
    return Frame(
        image=image,
        source_image_ref=str(path.resolve()),
        timestamp=_now_iso(),
        width=width,
        height=height,
    )
