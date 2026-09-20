"""Annotation tooling: COCO-keypoints-style pallet corner labels (R3).

This module holds the reusable annotation logic so it can be imported and
unit-tested; ``scripts/annotate.py`` is a thin CLI wrapper over it.

Design anchors
--------------
- **Keypoint schema** (design: "Keypoint / corner annotation schema", R3.1):
  the pallet's eight structural reference points — the four **bottom deck
  corners** (floor-contact, used for pose) plus the four **top deck corners**
  (for height/geometry), each with a **visibility label** ∈
  {visible, occluded, absent}. The keypoint *names* match the stub detector
  (:mod:`pallet_pose_compliance.detection.stub`): ``bottom_corner_0..3`` and
  ``top_corner_0..3``.
- **Documented, machine-readable format** (R3.2): a COCO-keypoints-style JSON
  document with ``categories`` (keypoint names + skeleton), ``images``, and
  ``annotations`` whose ``keypoints`` are flat ``[x1, y1, v1, ...]`` triplets.
  Visibility maps to the COCO convention ``v ∈ {0=absent, 1=occluded,
  2=visible}`` (and back).
- **Round-trip** (R3.2 / Property 6): writing annotations then reading them
  back yields equivalent annotations. The read/write API is kept clean so the
  tagged property test (task 4.2) can exercise it.
- **Labelling guideline correspondence** (R3.3): :data:`LABELLING_GUIDELINE`
  documents the exact behaviour of this tooling so DATASET.md can mirror it.

The format is a *subset* of the COCO keypoints spec sufficient for this task;
it is intentionally documented rather than claiming full COCO compliance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union

__all__ = [
    "KEYPOINT_NAMES",
    "SKELETON",
    "CATEGORY_ID",
    "CATEGORY_NAME",
    "Visibility",
    "COCO_VISIBILITY",
    "VISIBILITY_FROM_COCO",
    "LABELLING_GUIDELINE",
    "KeypointAnnotation",
    "ImageAnnotation",
    "PalletAnnotation",
    "AnnotationSet",
    "visibility_to_coco",
    "coco_to_visibility",
    "write_annotations",
    "read_annotations",
]

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

# The eight pallet structural keypoints, in a fixed, documented order. The
# first four are the floor-contact bottom deck corners (used for pose); the
# last four are the top deck corners (used for height/geometry). These names
# match the stub detector so keypoints are consistent across the codebase.
KEYPOINT_NAMES: tuple[str, ...] = (
    "bottom_corner_0",
    "bottom_corner_1",
    "bottom_corner_2",
    "bottom_corner_3",
    "top_corner_0",
    "top_corner_1",
    "top_corner_2",
    "top_corner_3",
)

NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# COCO skeleton links (1-indexed pairs, per COCO convention): the bottom quad,
# the top quad, and the four vertical edges joining them. Documented so the
# labelling guideline and any visualiser agree on connectivity.
SKELETON: tuple[tuple[int, int], ...] = (
    # bottom deck quad (indices 1..4)
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
    # top deck quad (indices 5..8)
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 5),
    # vertical edges bottom->top
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 8),
)

CATEGORY_ID = 1
CATEGORY_NAME = "pallet"

# Visibility labels used throughout the system (design: Keypoint model).
Visibility = Literal["visible", "occluded", "absent"]

# Mapping between the human-readable visibility labels and the COCO integer
# visibility flag. COCO uses v=0 (not labelled/absent), v=1 (labelled but not
# visible / occluded), v=2 (labelled and visible).
COCO_VISIBILITY: dict[str, int] = {
    "absent": 0,
    "occluded": 1,
    "visible": 2,
}
VISIBILITY_FROM_COCO: dict[int, str] = {v: k for k, v in COCO_VISIBILITY.items()}


LABELLING_GUIDELINE = """\
Pallet corner labelling guideline (mirrors annotation tooling behaviour, R3.3)

Each pallet instance is annotated with exactly eight keypoints, in this fixed
order:
  1-4. bottom_corner_0..3 - the four bottom deck-board corners that contact
       the floor. These are the pose reference points.
  5-8. top_corner_0..3     - the four top deck corners, used for load height
       and geometry.

Corner index convention: corner 0 is the near-right corner from the camera;
indices proceed counter-clockwise (0 -> 1 -> 2 -> 3) around the deck when
viewed from above. Bottom and top corners at the same index are the two ends
of the same vertical pallet edge (bottom_corner_i is directly below
top_corner_i).

Visibility label for every keypoint is one of:
  - visible  : the corner is directly observable in the image.
  - occluded : the corner's location is known/inferable but hidden (e.g. behind
               the load or another pallet); its pixel is still annotated.
  - absent   : the corner is outside the frame or cannot be located; its pixel
               coordinates are left null and it is not placed.

Machine-readable format: COCO-keypoints-style JSON. Each annotation stores
keypoints as a flat [x1, y1, v1, ...] triplet list of length 24 (8 corners x 3),
where v in {0=absent, 1=occluded, 2=visible}. Absent corners are written with
x=y=0 and v=0 per the COCO convention and read back as absent with null pixels.
Each annotation also records a bbox and num_keypoints (count of labelled, i.e.
non-absent, corners).
"""


# ---------------------------------------------------------------------------
# Visibility conversion helpers
# ---------------------------------------------------------------------------


def visibility_to_coco(visibility: str) -> int:
    """Map a visibility label to the COCO integer flag.

    Raises
    ------
    ValueError
        If ``visibility`` is not one of {visible, occluded, absent}.
    """
    try:
        return COCO_VISIBILITY[visibility]
    except KeyError as exc:  # pragma: no cover - trivial
        raise ValueError(
            f"unknown visibility {visibility!r}; expected one of "
            f"{sorted(COCO_VISIBILITY)}"
        ) from exc


def coco_to_visibility(flag: int) -> str:
    """Map a COCO integer visibility flag back to a visibility label.

    Raises
    ------
    ValueError
        If ``flag`` is not one of {0, 1, 2}.
    """
    try:
        return VISIBILITY_FROM_COCO[int(flag)]
    except KeyError as exc:
        raise ValueError(
            f"unknown COCO visibility flag {flag!r}; expected one of {{0, 1, 2}}"
        ) from exc


# ---------------------------------------------------------------------------
# Annotation data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeypointAnnotation:
    """A single labelled pallet corner.

    Attributes
    ----------
    name:
        One of :data:`KEYPOINT_NAMES`.
    x, y:
        Pixel coordinates, or ``None`` when the corner is ``absent`` (null
        discipline: an absent corner never carries fabricated pixels).
    visibility:
        One of {visible, occluded, absent}.
    """

    name: str
    x: Optional[float]
    y: Optional[float]
    visibility: str

    def __post_init__(self) -> None:
        if self.name not in KEYPOINT_NAMES:
            raise ValueError(
                f"unknown keypoint name {self.name!r}; expected one of "
                f"{KEYPOINT_NAMES}"
            )
        if self.visibility not in COCO_VISIBILITY:
            raise ValueError(
                f"unknown visibility {self.visibility!r}; expected one of "
                f"{sorted(COCO_VISIBILITY)}"
            )
        if self.visibility == "absent":
            if self.x is not None or self.y is not None:
                raise ValueError(
                    "an 'absent' keypoint must have null x/y coordinates, "
                    "never a fabricated placeholder"
                )
        else:
            if self.x is None or self.y is None:
                raise ValueError(
                    f"a {self.visibility!r} keypoint must carry x and y pixel "
                    "coordinates"
                )


@dataclass(frozen=True)
class PalletAnnotation:
    """One annotated pallet instance within an image.

    ``keypoints`` must contain exactly the eight named corners (in any order on
    input; canonical order is restored on read/write).
    """

    image_id: int
    keypoints: tuple[KeypointAnnotation, ...]
    annotation_id: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        names = [kp.name for kp in self.keypoints]
        if sorted(names) != sorted(KEYPOINT_NAMES):
            raise ValueError(
                "a pallet annotation must contain exactly the eight named "
                f"corners {KEYPOINT_NAMES}; got {names}"
            )
        # Store in canonical order so equality/round-trip is order-independent.
        ordered = tuple(
            sorted(self.keypoints, key=lambda kp: KEYPOINT_NAMES.index(kp.name))
        )
        object.__setattr__(self, "keypoints", ordered)

    @property
    def num_keypoints(self) -> int:
        """Number of labelled (non-absent) corners, per COCO semantics."""
        return sum(1 for kp in self.keypoints if kp.visibility != "absent")

    def computed_bbox_from_corners(self) -> tuple[float, float, float, float]:
        """The tight ``[x, y, w, h]`` box over all labelled (non-absent) corners.

        If no corner is labelled it is ``[0, 0, 0, 0]``. This ignores any
        explicit :attr:`bbox`; use :meth:`computed_bbox` to honour an explicit
        override.
        """
        xs = [kp.x for kp in self.keypoints if kp.x is not None]
        ys = [kp.y for kp in self.keypoints if kp.y is not None]
        if not xs or not ys:
            return (0.0, 0.0, 0.0, 0.0)
        x0, y0 = min(xs), min(ys)
        return (x0, y0, max(xs) - x0, max(ys) - y0)

    def computed_bbox(self) -> tuple[float, float, float, float]:
        """Return the explicit :attr:`bbox` if set, else the box over corners."""
        if self.bbox is not None:
            return self.bbox
        return self.computed_bbox_from_corners()


@dataclass(frozen=True)
class ImageAnnotation:
    """Metadata for one annotated image (COCO ``images`` entry)."""

    image_id: int
    file_name: str
    width: int
    height: int


@dataclass
class AnnotationSet:
    """A full COCO-keypoints-style annotation document for pallet corners.

    Holds the annotated images and their pallet annotations. Serialises to /
    parses from the documented COCO-style JSON via :func:`write_annotations`
    and :func:`read_annotations`, and exposes :meth:`to_coco_dict` /
    :meth:`from_coco_dict` for in-memory round-trips.
    """

    images: list[ImageAnnotation] = field(default_factory=list)
    annotations: list[PalletAnnotation] = field(default_factory=list)

    # -- COCO (de)serialisation ------------------------------------------

    def _category_dict(self) -> dict[str, Any]:
        return {
            "id": CATEGORY_ID,
            "name": CATEGORY_NAME,
            "supercategory": "pallet",
            "keypoints": list(KEYPOINT_NAMES),
            "skeleton": [list(link) for link in SKELETON],
        }

    def to_coco_dict(self) -> dict[str, Any]:
        """Serialise to a COCO-keypoints-style dict.

        Absent corners are written as ``x=y=0, v=0`` per the COCO convention;
        they are restored to ``absent`` with null pixels on read.
        """
        images = [
            {
                "id": img.image_id,
                "file_name": img.file_name,
                "width": img.width,
                "height": img.height,
            }
            for img in self.images
        ]
        annotations = []
        for i, ann in enumerate(self.annotations):
            flat: list[float] = []
            for kp in ann.keypoints:  # already in canonical order
                v = visibility_to_coco(kp.visibility)
                if v == 0:  # absent -> COCO writes 0,0,0
                    flat.extend([0.0, 0.0, 0])
                else:
                    flat.extend([float(kp.x), float(kp.y), v])
            annotations.append(
                {
                    "id": ann.annotation_id if ann.annotation_id is not None else i + 1,
                    "image_id": ann.image_id,
                    "category_id": CATEGORY_ID,
                    "keypoints": flat,
                    "num_keypoints": ann.num_keypoints,
                    "bbox": list(ann.computed_bbox()),
                    "iscrowd": 0,
                }
            )
        return {
            "info": {
                "description": "Pallet deck-corner keypoint annotations",
                "schema": "coco-keypoints-style",
                "keypoint_order": list(KEYPOINT_NAMES),
            },
            "images": images,
            "annotations": annotations,
            "categories": [self._category_dict()],
        }

    @classmethod
    def from_coco_dict(cls, data: dict[str, Any]) -> "AnnotationSet":
        """Parse a COCO-keypoints-style dict into an :class:`AnnotationSet`.

        Raises
        ------
        ValueError
            If the keypoint order/category does not match this tooling's schema
            or a ``keypoints`` list has the wrong length.
        """
        # Validate the category keypoint order matches our schema.
        categories = data.get("categories", [])
        cat = next(
            (c for c in categories if c.get("id", CATEGORY_ID) == CATEGORY_ID),
            categories[0] if categories else None,
        )
        if cat is not None and "keypoints" in cat:
            if tuple(cat["keypoints"]) != KEYPOINT_NAMES:
                raise ValueError(
                    "category keypoint order does not match the pallet corner "
                    f"schema; expected {KEYPOINT_NAMES}, got {tuple(cat['keypoints'])}"
                )

        images = [
            ImageAnnotation(
                image_id=int(img["id"]),
                file_name=str(img["file_name"]),
                width=int(img["width"]),
                height=int(img["height"]),
            )
            for img in data.get("images", [])
        ]

        annotations: list[PalletAnnotation] = []
        for ann in data.get("annotations", []):
            flat = ann["keypoints"]
            if len(flat) != NUM_KEYPOINTS * 3:
                raise ValueError(
                    f"annotation {ann.get('id')!r} has {len(flat)} keypoint "
                    f"values; expected {NUM_KEYPOINTS * 3} (8 corners x 3)"
                )
            kps: list[KeypointAnnotation] = []
            for idx, name in enumerate(KEYPOINT_NAMES):
                x, y, v = flat[idx * 3], flat[idx * 3 + 1], int(flat[idx * 3 + 2])
                visibility = coco_to_visibility(v)
                if visibility == "absent":
                    kps.append(KeypointAnnotation(name=name, x=None, y=None,
                                                  visibility="absent"))
                else:
                    kps.append(
                        KeypointAnnotation(
                            name=name,
                            x=float(x),
                            y=float(y),
                            visibility=visibility,
                        )
                    )
            raw_bbox = ann.get("bbox")
            bbox = tuple(float(b) for b in raw_bbox) if raw_bbox is not None else None
            pallet = PalletAnnotation(
                image_id=int(ann["image_id"]),
                keypoints=tuple(kps),
                annotation_id=int(ann["id"]) if ann.get("id") is not None else None,
                bbox=bbox,
            )
            # Round-trip fidelity: a bbox that merely echoes the auto-computed
            # box over the labelled corners is treated as "not explicitly set"
            # (bbox=None), so write->read preserves the original annotation
            # (Property 6). An explicitly different bbox is retained.
            if bbox is not None and bbox == pallet.computed_bbox_from_corners():
                pallet = PalletAnnotation(
                    image_id=pallet.image_id,
                    keypoints=pallet.keypoints,
                    annotation_id=pallet.annotation_id,
                    bbox=None,
                )
            annotations.append(pallet)
        return cls(images=images, annotations=annotations)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_annotations(
    annotation_set: AnnotationSet,
    path: Union[str, Path],
    *,
    indent: int = 2,
) -> Path:
    """Write ``annotation_set`` to ``path`` as COCO-keypoints-style JSON.

    Returns the :class:`~pathlib.Path` written. Parent directories are created
    if needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(annotation_set.to_coco_dict(), indent=indent),
        encoding="utf-8",
    )
    return out


def read_annotations(path: Union[str, Path]) -> AnnotationSet:
    """Read a COCO-keypoints-style JSON file into an :class:`AnnotationSet`.

    Inverse of :func:`write_annotations`; the round-trip preserves the
    annotations (Property 6).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AnnotationSet.from_coco_dict(data)
