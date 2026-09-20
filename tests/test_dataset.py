"""Unit tests for dataset assembly, manifests, and the split protocol (task 4.3).

Covers :mod:`pallet_pose_compliance.dataset`:

- **Split-by-attribute correctness** (R2.1, R2.2): the split is performed by the
  declared grouping attribute so the same scene/session/camera-pose never
  straddles train and held-out.
- **Count recording** (R1.6): manifests record per-class object counts, pallet
  annotation counts, image counts, and group counts.
- **Disjointness** (R2.4): the disjointness helpers report held-out is disjoint
  from training, and the split produces disjoint subsets.
- **Reproducibility** (R28.3): the same seed yields the same split.
- **Manifest round-trip**: write -> read preserves the manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pallet_pose_compliance.annotation import (
    AnnotationSet,
    ImageAnnotation,
    KeypointAnnotation,
    PalletAnnotation,
)
from pallet_pose_compliance.dataset import (
    DEFAULT_SEED,
    SPLIT_ATTRIBUTES,
    Dataset,
    DatasetSample,
    SplitManifest,
    assemble_dataset,
    is_disjoint,
    overlapping_groups,
    overlapping_images,
    read_manifest,
    split_dataset,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _keypoints() -> tuple[KeypointAnnotation, ...]:
    """A minimal valid set of eight corners (all visible)."""
    kps = []
    for i in range(4):
        kps.append(KeypointAnnotation(f"bottom_corner_{i}", float(i), float(i), "visible"))
    for i in range(4):
        kps.append(KeypointAnnotation(f"top_corner_{i}", float(i), float(i) + 10.0, "visible"))
    return tuple(kps)


def _annotation_set(
    image_groups: dict[int, str],
    *,
    annots_per_image: dict[int, int] | None = None,
) -> tuple[AnnotationSet, dict[int, str]]:
    """Build an AnnotationSet from ``{image_id: group_id}`` plus a groups map.

    ``annots_per_image`` overrides the number of pallet annotations per image
    (default 1 each).
    """
    annots_per_image = annots_per_image or {}
    images = [
        ImageAnnotation(image_id=iid, file_name=f"{grp}/img_{iid}.png", width=640, height=480)
        for iid, grp in image_groups.items()
    ]
    annotations: list[PalletAnnotation] = []
    next_ann_id = 1
    for iid in image_groups:
        n = annots_per_image.get(iid, 1)
        for _ in range(n):
            annotations.append(
                PalletAnnotation(image_id=iid, keypoints=_keypoints(), annotation_id=next_ann_id)
            )
            next_ann_id += 1
    groups = {iid: grp for iid, grp in image_groups.items()}
    return AnnotationSet(images=images, annotations=annotations), groups


# ---------------------------------------------------------------------------
# Assembly + count recording (R1.6)
# ---------------------------------------------------------------------------


def test_assemble_records_group_and_counts():
    ann_set, groups = _annotation_set(
        {1: "scene_a", 2: "scene_a", 3: "scene_b"},
        annots_per_image={1: 2, 2: 1, 3: 3},
    )
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)

    assert len(dataset.samples) == 3
    by_id = {s.image_id: s for s in dataset.samples}
    assert by_id[1].group_id == "scene_a"
    assert by_id[1].annotation_count == 2
    assert by_id[1].class_counts == {"pallet": 2}
    assert by_id[3].group_id == "scene_b"
    assert by_id[3].annotation_count == 3


def test_manifest_counts_aggregate_correctly():
    ann_set, groups = _annotation_set(
        {1: "s1", 2: "s1", 3: "s2", 4: "s2"},
        annots_per_image={1: 2, 2: 1, 3: 3, 4: 1},
    )
    dataset = assemble_dataset(ann_set, split_by="session", groups=groups)
    result = split_dataset(dataset, held_out_fraction=0.5, seed=DEFAULT_SEED)

    # Combined counts across both subsets must equal the whole dataset.
    total_annots = result.train.annotation_count + result.held_out.annotation_count
    total_images = result.train.num_images + result.held_out.num_images
    assert total_annots == 7
    assert total_images == 4

    total_pallets = (
        result.train.class_counts.get("pallet", 0)
        + result.held_out.class_counts.get("pallet", 0)
    )
    assert total_pallets == 7


def test_image_with_no_annotations_has_empty_class_counts():
    ann_set, groups = _annotation_set({1: "scene_a"}, annots_per_image={1: 0})
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)
    sample = dataset.samples[0]
    assert sample.annotation_count == 0
    assert sample.class_counts == {}


def test_assemble_rejects_unknown_split_attribute():
    ann_set, groups = _annotation_set({1: "a"})
    with pytest.raises(ValueError):
        assemble_dataset(ann_set, split_by="not_an_attribute", groups=groups)


def test_group_of_callable_is_used():
    ann_set, _ = _annotation_set({1: "ignored", 2: "ignored"})
    dataset = assemble_dataset(
        ann_set,
        split_by="camera_pose",
        group_of=lambda img: "pose_high" if img.image_id == 1 else "pose_low",
    )
    by_id = {s.image_id: s for s in dataset.samples}
    assert by_id[1].group_id == "pose_high"
    assert by_id[2].group_id == "pose_low"


# ---------------------------------------------------------------------------
# Split-by-attribute correctness (R2.1, R2.2)
# ---------------------------------------------------------------------------


def test_split_never_straddles_a_group():
    # Several images per group; splitting must keep each group whole.
    image_groups = {}
    iid = 1
    for scene in ("A", "B", "C", "D", "E"):
        for _ in range(3):
            image_groups[iid] = f"scene_{scene}"
            iid += 1
    ann_set, groups = _annotation_set(image_groups)
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)
    result = split_dataset(dataset, held_out_fraction=0.4, seed=DEFAULT_SEED)

    train_groups = set(result.train.group_ids)
    held_groups = set(result.held_out.group_ids)
    # No group appears in both subsets.
    assert train_groups.isdisjoint(held_groups)
    # Every original group landed somewhere.
    assert train_groups | held_groups == set(f"scene_{s}" for s in "ABCDE")


def test_split_records_declared_attribute_on_manifests():
    ann_set, groups = _annotation_set({1: "a", 2: "b", 3: "c"})
    dataset = assemble_dataset(ann_set, split_by="camera_pose", groups=groups)
    result = split_dataset(dataset, seed=DEFAULT_SEED)
    assert result.train.split_by == "camera_pose"
    assert result.held_out.split_by == "camera_pose"
    assert result.train.split == "train"
    assert result.held_out.split == "held_out"


def test_split_both_subsets_nonempty_with_multiple_groups():
    ann_set, groups = _annotation_set({1: "a", 2: "b", 3: "c", 4: "d"})
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)
    result = split_dataset(dataset, held_out_fraction=0.25, seed=DEFAULT_SEED)
    assert result.train.num_groups >= 1
    assert result.held_out.num_groups >= 1


def test_single_group_goes_entirely_to_train():
    ann_set, groups = _annotation_set({1: "only", 2: "only"})
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)
    result = split_dataset(dataset, held_out_fraction=0.5, seed=DEFAULT_SEED)
    assert result.train.num_images == 2
    assert result.held_out.num_images == 0


def test_split_rejects_bad_fraction():
    ann_set, groups = _annotation_set({1: "a", 2: "b"})
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)
    with pytest.raises(ValueError):
        split_dataset(dataset, held_out_fraction=1.5)


# ---------------------------------------------------------------------------
# Reproducibility (R28.3)
# ---------------------------------------------------------------------------


def test_same_seed_yields_same_split():
    image_groups = {i: f"g{i % 6}" for i in range(1, 25)}
    ann_set, groups = _annotation_set(image_groups)
    dataset = assemble_dataset(ann_set, split_by="session", groups=groups)
    r1 = split_dataset(dataset, held_out_fraction=0.3, seed=7)
    r2 = split_dataset(dataset, held_out_fraction=0.3, seed=7)
    assert r1.train.image_ids == r2.train.image_ids
    assert r1.held_out.image_ids == r2.held_out.image_ids


def test_different_seed_can_change_group_assignment():
    image_groups = {i: f"g{i}" for i in range(1, 11)}  # 10 singleton groups
    ann_set, groups = _annotation_set(image_groups)
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)
    held_a = set(split_dataset(dataset, held_out_fraction=0.5, seed=1).held_out.group_ids)
    held_b = set(split_dataset(dataset, held_out_fraction=0.5, seed=999).held_out.group_ids)
    # With 10 groups and different seeds the held-out sets should differ.
    assert held_a != held_b


# ---------------------------------------------------------------------------
# Disjointness (R2.4, Property 5 helper)
# ---------------------------------------------------------------------------


def test_split_produces_disjoint_manifests():
    ann_set, groups = _annotation_set({i: f"g{i % 4}" for i in range(1, 17)})
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)
    result = split_dataset(dataset, held_out_fraction=0.5, seed=DEFAULT_SEED)

    assert is_disjoint(result.train, result.held_out)
    assert overlapping_groups(result.train, result.held_out) == set()
    assert overlapping_images(result.train, result.held_out) == set()


def test_is_disjoint_detects_shared_group():
    shared_sample = DatasetSample(image_id=99, file_name="x", group_id="shared", annotation_count=1)
    train = SplitManifest(split="train", split_by="scene", seed=1, samples=[shared_sample])
    held = SplitManifest(split="held_out", split_by="scene", seed=1, samples=[shared_sample])
    assert not is_disjoint(train, held)
    assert overlapping_groups(train, held) == {"shared"}
    assert overlapping_images(train, held) == {99}


# ---------------------------------------------------------------------------
# Manifest file round-trip
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path):
    ann_set, groups = _annotation_set(
        {1: "a", 2: "a", 3: "b"}, annots_per_image={1: 2, 2: 1, 3: 1}
    )
    dataset = assemble_dataset(ann_set, split_by="scene", groups=groups)
    result = split_dataset(dataset, held_out_fraction=0.5, seed=DEFAULT_SEED)

    path = tmp_path / "train_manifest.json"
    write_manifest(result.train, path)
    loaded = read_manifest(path)

    assert loaded.split == result.train.split
    assert loaded.split_by == result.train.split_by
    assert loaded.seed == result.train.seed
    assert loaded.image_ids == result.train.image_ids
    assert loaded.group_ids == result.train.group_ids
    assert loaded.annotation_count == result.train.annotation_count
    assert loaded.class_counts == result.train.class_counts


def test_all_split_attributes_are_accepted():
    ann_set, groups = _annotation_set({1: "a", 2: "b"})
    for attr in SPLIT_ATTRIBUTES:
        dataset = assemble_dataset(ann_set, split_by=attr, groups=groups)
        result = split_dataset(dataset, seed=DEFAULT_SEED)
        assert result.train.split_by == attr
