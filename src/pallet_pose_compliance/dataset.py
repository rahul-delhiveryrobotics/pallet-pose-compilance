"""Dataset assembly, manifests, and the train / held-out split protocol (R2, R1.6).

This module assembles a detection/localisation dataset from
:class:`~pallet_pose_compliance.annotation.AnnotationSet` documents and
partitions it into **train** and **held-out** subsets by a *declared grouping
attribute* (scene, capture session, or camera pose). Splitting by a grouping
attribute — rather than by individual image — is what makes the ``Held_Out_Set``
differ from the training data in an explicitly declared way (R2.2, R2.3) and
guarantees that the same scene/session/camera-pose never straddles the two
subsets, so held-out accuracy reflects generalisation rather than memorisation
(R2.1).

Design anchors
--------------
- **Split by a declared attribute** (R2.1, R2.2): every sample carries a
  :attr:`DatasetSample.group_id` recording its grouping-attribute value; the
  split is performed over the *set of groups*, never individual images, so a
  group is wholly in ``train`` or wholly in ``held_out``. The name of the
  attribute is recorded on every manifest as :attr:`SplitManifest.split_by`.
- **Held-out differs in a declared way** (R2.3): the manifest records the
  ``split_by`` attribute and the concrete group ids assigned to each subset, so
  the declared difference is machine-readable.
- **Class + annotation counts** (R1.6): each manifest records per-class object
  counts, the number of pallet annotations, the number of images, and the
  number of groups.
- **Reproducible assignment** (R28.3): group→subset assignment is a
  deterministic function of a ``seed`` (default from ``configs/pipeline.yaml``),
  so the split is reproducible.
- **Disjointness helper** (R2.4, Property 5): :func:`is_disjoint` /
  :func:`overlapping_groups` / :func:`overlapping_images` expose whether two
  manifests share any group or image. The API is kept clean so the tagged
  property test (task 4.4) can assert held-out is disjoint from training.
- **In-memory, no real data required**: assembly and splitting operate over
  ``AnnotationSet`` records / manifest objects, so the whole protocol is
  testable with synthetic fixtures. Manifests (de)serialise to JSON so the
  chosen split can be tracked in the repository.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union

from .annotation import CATEGORY_NAME, AnnotationSet

__all__ = [
    "DEFAULT_SEED",
    "SPLIT_ATTRIBUTES",
    "DatasetSample",
    "Dataset",
    "SplitManifest",
    "SplitResult",
    "assemble_dataset",
    "split_dataset",
    "is_disjoint",
    "overlapping_groups",
    "overlapping_images",
    "write_manifest",
    "read_manifest",
]

# Default seed mirrors ``configs/pipeline.yaml`` (R28.3). Callers that load the
# pipeline config should pass its ``seed`` explicitly; this constant keeps the
# module usable/testable without reading a file.
DEFAULT_SEED = 42

# The declared grouping attributes the split may be performed by (R2.2). The
# split is always *by* one of these; the same value never straddles subsets.
SPLIT_ATTRIBUTES: tuple[str, ...] = ("scene", "session", "camera_pose")


# ---------------------------------------------------------------------------
# Dataset records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSample:
    """One assembled sample: an image plus its grouping-attribute value.

    Attributes
    ----------
    image_id:
        The COCO image id (unique within a source :class:`AnnotationSet`).
    file_name:
        The image file name, carried through for manifest readability.
    group_id:
        The value of the declared grouping attribute for this image (e.g. the
        scene id / capture-session id / camera-pose id). The split is performed
        *by* this value so the same group never straddles train and held-out
        (R2.1, R2.2).
    class_counts:
        Per-class object counts within this image (currently the pallet class;
        the schema supports additional load classes such as boxes).
    annotation_count:
        Number of pallet annotations (instances) in this image.
    """

    image_id: int
    file_name: str
    group_id: str
    class_counts: Mapping[str, int] = field(default_factory=dict)
    annotation_count: int = 0


@dataclass
class Dataset:
    """An assembled dataset over which a split protocol is applied.

    ``split_by`` names the declared grouping attribute (one of
    :data:`SPLIT_ATTRIBUTES`); every sample's :attr:`DatasetSample.group_id`
    is a value of that attribute.
    """

    samples: list[DatasetSample]
    split_by: str

    def __post_init__(self) -> None:
        if self.split_by not in SPLIT_ATTRIBUTES:
            raise ValueError(
                f"unknown split attribute {self.split_by!r}; expected one of "
                f"{SPLIT_ATTRIBUTES}"
            )
        seen: set[int] = set()
        for sample in self.samples:
            if sample.image_id in seen:
                raise ValueError(
                    f"duplicate image_id {sample.image_id!r} in dataset; image "
                    "ids must be unique so the split is well defined"
                )
            seen.add(sample.image_id)

    # -- grouping views ---------------------------------------------------

    def group_ids(self) -> list[str]:
        """The distinct grouping values, in first-seen order (deterministic)."""
        ordered: list[str] = []
        seen: set[str] = set()
        for sample in self.samples:
            if sample.group_id not in seen:
                seen.add(sample.group_id)
                ordered.append(sample.group_id)
        return ordered

    def samples_by_group(self) -> dict[str, list[DatasetSample]]:
        """Map each grouping value to its samples (first-seen group order)."""
        grouped: dict[str, list[DatasetSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.group_id, []).append(sample)
        return grouped


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


@dataclass
class SplitManifest:
    """A machine-readable manifest for one split subset (train or held-out).

    Records the sample/image list, the declared grouping attribute and the
    concrete group values assigned to this subset (R2.2, R2.3), plus per-class
    object counts and pallet annotation counts (R1.6).
    """

    split: str  # "train" | "held_out"
    split_by: str  # declared grouping attribute (R2.2)
    seed: int
    samples: list[DatasetSample] = field(default_factory=list)

    # -- derived counts ---------------------------------------------------

    @property
    def group_ids(self) -> list[str]:
        """Distinct grouping values in this subset, in first-seen order."""
        ordered: list[str] = []
        seen: set[str] = set()
        for sample in self.samples:
            if sample.group_id not in seen:
                seen.add(sample.group_id)
                ordered.append(sample.group_id)
        return ordered

    @property
    def image_ids(self) -> list[int]:
        """The image ids in this subset."""
        return [s.image_id for s in self.samples]

    @property
    def num_images(self) -> int:
        return len(self.samples)

    @property
    def num_groups(self) -> int:
        return len(self.group_ids)

    @property
    def annotation_count(self) -> int:
        """Total pallet annotations across this subset (R1.6)."""
        return sum(s.annotation_count for s in self.samples)

    @property
    def class_counts(self) -> dict[str, int]:
        """Aggregated per-class object counts across this subset (R1.6)."""
        totals: Counter[str] = Counter()
        for sample in self.samples:
            totals.update(sample.class_counts)
        return dict(totals)

    # -- (de)serialisation ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a machine-readable dict (JSON-ready)."""
        return {
            "split": self.split,
            "split_by": self.split_by,
            "seed": self.seed,
            "num_images": self.num_images,
            "num_groups": self.num_groups,
            "annotation_count": self.annotation_count,
            "class_counts": self.class_counts,
            "group_ids": self.group_ids,
            "samples": [
                {
                    "image_id": s.image_id,
                    "file_name": s.file_name,
                    "group_id": s.group_id,
                    "class_counts": dict(s.class_counts),
                    "annotation_count": s.annotation_count,
                }
                for s in self.samples
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitManifest":
        """Parse a manifest dict produced by :meth:`to_dict`."""
        samples = [
            DatasetSample(
                image_id=int(s["image_id"]),
                file_name=str(s["file_name"]),
                group_id=str(s["group_id"]),
                class_counts={str(k): int(v) for k, v in s.get("class_counts", {}).items()},
                annotation_count=int(s.get("annotation_count", 0)),
            )
            for s in data.get("samples", [])
        ]
        return cls(
            split=str(data["split"]),
            split_by=str(data["split_by"]),
            seed=int(data["seed"]),
            samples=samples,
        )


@dataclass
class SplitResult:
    """The outcome of a split: the train and held-out manifests together."""

    train: SplitManifest
    held_out: SplitManifest

    def to_dict(self) -> dict[str, Any]:
        return {"train": self.train.to_dict(), "held_out": self.held_out.to_dict()}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _default_group_of(image: Any) -> str:
    """Fallback grouping value: the image's ``file_name`` stem directory or name.

    Used when no ``group_of`` callable is supplied. This keeps assembly usable
    without a metadata source, but callers SHOULD pass an explicit ``group_of``
    (or a ``groups`` mapping) so the grouping attribute is meaningful.
    """
    name = getattr(image, "file_name", str(getattr(image, "image_id", "")))
    parent = Path(name).parent.name
    return parent if parent else Path(name).stem


def assemble_dataset(
    annotation_set: AnnotationSet,
    *,
    split_by: str,
    group_of: Optional[Callable[[Any], str]] = None,
    groups: Optional[Mapping[int, str]] = None,
) -> Dataset:
    """Assemble a :class:`Dataset` from an :class:`AnnotationSet`.

    Each image becomes a :class:`DatasetSample` carrying its grouping-attribute
    value and its per-class / annotation counts (R1.6). The grouping value is
    resolved, in priority order, from:

    1. ``groups`` — an explicit ``{image_id: group_id}`` mapping, or
    2. ``group_of`` — a callable mapping an :class:`ImageAnnotation` to a group
       id, or
    3. :func:`_default_group_of` — a filename-derived fallback.

    Parameters
    ----------
    annotation_set:
        The parsed annotations to assemble from.
    split_by:
        The declared grouping attribute; one of :data:`SPLIT_ATTRIBUTES`.
    group_of / groups:
        Optional grouping-value sources (see above).

    Raises
    ------
    ValueError
        If ``split_by`` is unknown or an image has no resolvable group id.
    """
    if split_by not in SPLIT_ATTRIBUTES:
        raise ValueError(
            f"unknown split attribute {split_by!r}; expected one of {SPLIT_ATTRIBUTES}"
        )

    # Pre-tally pallet annotations per image so counts are O(images + annots).
    annots_per_image: Counter[int] = Counter()
    for ann in annotation_set.annotations:
        annots_per_image[ann.image_id] += 1

    samples: list[DatasetSample] = []
    for image in annotation_set.images:
        if groups is not None and image.image_id in groups:
            group_id = str(groups[image.image_id])
        elif group_of is not None:
            group_id = str(group_of(image))
        else:
            group_id = _default_group_of(image)
        if not group_id:
            raise ValueError(
                f"image {image.image_id!r} has no resolvable {split_by} group id"
            )
        n_annots = int(annots_per_image.get(image.image_id, 0))
        samples.append(
            DatasetSample(
                image_id=image.image_id,
                file_name=image.file_name,
                group_id=group_id,
                # Currently a single detection class (pallet); the schema
                # supports additional load classes (e.g. boxes) when annotated.
                class_counts={CATEGORY_NAME: n_annots} if n_annots else {},
                annotation_count=n_annots,
            )
        )
    return Dataset(samples=samples, split_by=split_by)


# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------


def split_dataset(
    dataset: Dataset,
    *,
    held_out_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> SplitResult:
    """Partition ``dataset`` into train / held-out **by the grouping attribute**.

    Groups (not images) are shuffled deterministically with ``seed`` and then
    assigned so that approximately ``held_out_fraction`` of the *groups* form
    the held-out subset. Because assignment is at the group level, the same
    scene/session/camera-pose is never split across the two subsets (R2.1), and
    the resulting manifests are disjoint (R2.4, Property 5).

    At least one group is assigned to each subset whenever the dataset has two
    or more groups, so neither subset is empty; with a single group everything
    goes to train and the held-out subset is empty (the caller is expected to
    supply enough groups for a meaningful split).

    Parameters
    ----------
    dataset:
        The assembled dataset to split.
    held_out_fraction:
        Target fraction of *groups* placed in the held-out subset, in [0, 1].
    seed:
        Seed for the deterministic group shuffle (R28.3).

    Raises
    ------
    ValueError
        If ``held_out_fraction`` is outside [0, 1].
    """
    if not 0.0 <= held_out_fraction <= 1.0:
        raise ValueError(
            f"held_out_fraction must be in [0, 1]; got {held_out_fraction!r}"
        )

    groups = dataset.group_ids()
    n_groups = len(groups)

    # Deterministic shuffle of the group ids (reproducible with the seed).
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)

    if n_groups <= 1:
        n_held = 0
    else:
        n_held = int(round(n_groups * held_out_fraction))
        # Guarantee both subsets are non-empty when a split is requested.
        if held_out_fraction > 0.0:
            n_held = max(1, n_held)
        n_held = min(n_held, n_groups - 1)

    held_groups = set(shuffled[:n_held])

    samples_by_group = dataset.samples_by_group()
    train_samples: list[DatasetSample] = []
    held_samples: list[DatasetSample] = []
    # Preserve original sample order within each subset for stable manifests.
    for sample in dataset.samples:
        if sample.group_id in held_groups:
            held_samples.append(sample)
        else:
            train_samples.append(sample)

    train = SplitManifest(
        split="train",
        split_by=dataset.split_by,
        seed=seed,
        samples=train_samples,
    )
    held_out = SplitManifest(
        split="held_out",
        split_by=dataset.split_by,
        seed=seed,
        samples=held_samples,
    )
    return SplitResult(train=train, held_out=held_out)


# ---------------------------------------------------------------------------
# Disjointness helpers (R2.4, Property 5)
# ---------------------------------------------------------------------------


def overlapping_groups(
    train: SplitManifest, held_out: SplitManifest
) -> set[str]:
    """Return the set of grouping values present in *both* manifests.

    Empty when the split is group-disjoint (the expected case, R2.1).
    """
    return set(train.group_ids) & set(held_out.group_ids)


def overlapping_images(
    train: SplitManifest, held_out: SplitManifest
) -> set[int]:
    """Return the set of image ids present in *both* manifests.

    Empty when the split is image-disjoint (R2.4). Because the split is by
    group, group-disjointness implies image-disjointness.
    """
    return set(train.image_ids) & set(held_out.image_ids)


def is_disjoint(train: SplitManifest, held_out: SplitManifest) -> bool:
    """Whether the held-out manifest is disjoint from the training manifest.

    Returns ``True`` when the two subsets share **no grouping value and no
    image** — i.e. held-out data never appeared in training (R2.4, Property 5).
    This is the clean predicate the tagged property test (task 4.4) asserts.
    """
    return not overlapping_groups(train, held_out) and not overlapping_images(
        train, held_out
    )


# ---------------------------------------------------------------------------
# Manifest file I/O
# ---------------------------------------------------------------------------


def write_manifest(
    manifest: SplitManifest,
    path: Union[str, Path],
    *,
    indent: int = 2,
) -> Path:
    """Write ``manifest`` to ``path`` as machine-readable JSON (tracked in-repo).

    Returns the :class:`~pathlib.Path` written; parent directories are created.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=indent), encoding="utf-8")
    return out


def read_manifest(path: Union[str, Path]) -> SplitManifest:
    """Read a manifest JSON file written by :func:`write_manifest`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitManifest.from_dict(data)
