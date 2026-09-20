"""Unit tests for the COCO-keypoints-style annotation tooling (task 4.1, R3).

Covers the reusable annotation module
(:mod:`pallet_pose_compliance.annotation`) and the thin CLI wrapper
(``scripts/annotate.py``): keypoint schema, visibility mapping, absent-corner
null discipline, bbox/num_keypoints, and the write -> read round-trip that the
tagged property test (task 4.2) builds on (R3.1, R3.2, R3.3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pallet_pose_compliance.annotation import (
    CATEGORY_ID,
    COCO_VISIBILITY,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    AnnotationSet,
    ImageAnnotation,
    KeypointAnnotation,
    LABELLING_GUIDELINE,
    PalletAnnotation,
    coco_to_visibility,
    read_annotations,
    visibility_to_coco,
    write_annotations,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _full_keypoints(offset: float = 0.0) -> tuple[KeypointAnnotation, ...]:
    """Eight corners: bottom visible, top mixed occluded/absent."""
    kps = []
    for i in range(4):
        kps.append(
            KeypointAnnotation(
                name=f"bottom_corner_{i}",
                x=10.0 * i + offset,
                y=20.0 * i + offset,
                visibility="visible",
            )
        )
    # top corners: two occluded, one visible, one absent
    kps.append(KeypointAnnotation(name="top_corner_0", x=5.0, y=6.0,
                                  visibility="occluded"))
    kps.append(KeypointAnnotation(name="top_corner_1", x=7.0, y=8.0,
                                  visibility="occluded"))
    kps.append(KeypointAnnotation(name="top_corner_2", x=9.0, y=10.0,
                                  visibility="visible"))
    kps.append(KeypointAnnotation(name="top_corner_3", x=None, y=None,
                                  visibility="absent"))
    return tuple(kps)


def _sample_set() -> AnnotationSet:
    return AnnotationSet(
        images=[ImageAnnotation(image_id=1, file_name="img_0001.png",
                                width=640, height=480)],
        annotations=[
            PalletAnnotation(image_id=1, keypoints=_full_keypoints(),
                             annotation_id=1)
        ],
    )


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_eight_keypoints_bottom_then_top(self):
        assert NUM_KEYPOINTS == 8
        assert KEYPOINT_NAMES[:4] == (
            "bottom_corner_0",
            "bottom_corner_1",
            "bottom_corner_2",
            "bottom_corner_3",
        )
        assert KEYPOINT_NAMES[4:] == (
            "top_corner_0",
            "top_corner_1",
            "top_corner_2",
            "top_corner_3",
        )

    def test_matches_stub_detector_names(self):
        from pallet_pose_compliance.detection.stub import _KEYPOINT_NAMES

        assert tuple(_KEYPOINT_NAMES) == KEYPOINT_NAMES

    def test_guideline_mentions_all_three_visibilities(self):
        for label in ("visible", "occluded", "absent"):
            assert label in LABELLING_GUIDELINE


# ---------------------------------------------------------------------------
# Visibility mapping
# ---------------------------------------------------------------------------


class TestVisibilityMapping:
    def test_visibility_to_coco(self):
        assert visibility_to_coco("absent") == 0
        assert visibility_to_coco("occluded") == 1
        assert visibility_to_coco("visible") == 2

    def test_coco_to_visibility(self):
        assert coco_to_visibility(0) == "absent"
        assert coco_to_visibility(1) == "occluded"
        assert coco_to_visibility(2) == "visible"

    def test_visibility_round_trip(self):
        for label in COCO_VISIBILITY:
            assert coco_to_visibility(visibility_to_coco(label)) == label

    def test_unknown_visibility_rejected(self):
        with pytest.raises(ValueError):
            visibility_to_coco("blurry")

    def test_unknown_coco_flag_rejected(self):
        with pytest.raises(ValueError):
            coco_to_visibility(3)


# ---------------------------------------------------------------------------
# KeypointAnnotation invariants
# ---------------------------------------------------------------------------


class TestKeypointAnnotation:
    def test_absent_keypoint_forbids_coords(self):
        with pytest.raises(ValueError):
            KeypointAnnotation(name="top_corner_3", x=0.0, y=0.0,
                               visibility="absent")

    def test_visible_keypoint_requires_coords(self):
        with pytest.raises(ValueError):
            KeypointAnnotation(name="bottom_corner_0", x=None, y=None,
                               visibility="visible")

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError):
            KeypointAnnotation(name="left_corner", x=1.0, y=2.0,
                               visibility="visible")


# ---------------------------------------------------------------------------
# PalletAnnotation invariants
# ---------------------------------------------------------------------------


class TestPalletAnnotation:
    def test_requires_all_eight_corners(self):
        with pytest.raises(ValueError):
            PalletAnnotation(image_id=1, keypoints=_full_keypoints()[:7])

    def test_keypoints_stored_in_canonical_order(self):
        shuffled = tuple(reversed(_full_keypoints()))
        ann = PalletAnnotation(image_id=1, keypoints=shuffled)
        assert tuple(kp.name for kp in ann.keypoints) == KEYPOINT_NAMES

    def test_num_keypoints_counts_non_absent(self):
        ann = PalletAnnotation(image_id=1, keypoints=_full_keypoints())
        # 4 bottom visible + 2 top occluded + 1 top visible = 7; 1 absent
        assert ann.num_keypoints == 7

    def test_computed_bbox_covers_labelled_corners(self):
        ann = PalletAnnotation(image_id=1, keypoints=_full_keypoints())
        x, y, w, h = ann.computed_bbox()
        # labelled xs: 0,10,20,30 (bottom) + 5,7,9 (top) -> min 0, max 30
        assert x == 0.0 and w == 30.0
        assert y == 0.0 and h == 60.0

    def test_explicit_bbox_is_preserved(self):
        ann = PalletAnnotation(image_id=1, keypoints=_full_keypoints(),
                               bbox=(1.0, 2.0, 3.0, 4.0))
        assert ann.computed_bbox() == (1.0, 2.0, 3.0, 4.0)


# ---------------------------------------------------------------------------
# COCO dict shape
# ---------------------------------------------------------------------------


class TestCocoDict:
    def test_category_has_keypoint_names_and_skeleton(self):
        d = _sample_set().to_coco_dict()
        cat = next(c for c in d["categories"] if c["id"] == CATEGORY_ID)
        assert cat["keypoints"] == list(KEYPOINT_NAMES)
        assert len(cat["skeleton"]) > 0

    def test_flat_keypoint_triplets_length(self):
        d = _sample_set().to_coco_dict()
        flat = d["annotations"][0]["keypoints"]
        assert len(flat) == NUM_KEYPOINTS * 3

    def test_absent_corner_written_as_zero_triplet(self):
        d = _sample_set().to_coco_dict()
        flat = d["annotations"][0]["keypoints"]
        # top_corner_3 is index 7 and absent -> [0,0,0]
        assert flat[21:24] == [0.0, 0.0, 0]

    def test_num_keypoints_in_dict(self):
        d = _sample_set().to_coco_dict()
        assert d["annotations"][0]["num_keypoints"] == 7


# ---------------------------------------------------------------------------
# Round-trip (in-memory and via file)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_in_memory_round_trip(self):
        aset = _sample_set()
        reparsed = AnnotationSet.from_coco_dict(aset.to_coco_dict())
        assert reparsed == aset

    def test_file_round_trip(self, tmp_path: Path):
        aset = _sample_set()
        out = write_annotations(aset, tmp_path / "ann.json")
        assert out.exists()
        loaded = read_annotations(out)
        assert loaded == aset

    def test_absent_corner_restored_with_null_pixels(self, tmp_path: Path):
        aset = _sample_set()
        loaded = read_annotations(write_annotations(aset, tmp_path / "ann.json"))
        top3 = next(
            kp for kp in loaded.annotations[0].keypoints
            if kp.name == "top_corner_3"
        )
        assert top3.visibility == "absent"
        assert top3.x is None and top3.y is None

    def test_empty_set_round_trips(self, tmp_path: Path):
        aset = AnnotationSet()
        loaded = read_annotations(write_annotations(aset, tmp_path / "empty.json"))
        assert loaded == aset

    def test_written_file_is_valid_json_with_categories(self, tmp_path: Path):
        out = write_annotations(_sample_set(), tmp_path / "ann.json")
        raw = json.loads(out.read_text())
        assert raw["categories"][0]["keypoints"] == list(KEYPOINT_NAMES)


# ---------------------------------------------------------------------------
# Schema-mismatch handling
# ---------------------------------------------------------------------------


class TestSchemaMismatch:
    def test_wrong_keypoint_order_rejected(self):
        d = _sample_set().to_coco_dict()
        d["categories"][0]["keypoints"] = list(reversed(KEYPOINT_NAMES))
        with pytest.raises(ValueError):
            AnnotationSet.from_coco_dict(d)

    def test_wrong_triplet_length_rejected(self):
        d = _sample_set().to_coco_dict()
        d["annotations"][0]["keypoints"] = [0.0, 0.0, 2]  # too short
        with pytest.raises(ValueError):
            AnnotationSet.from_coco_dict(d)


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------


class TestCli:
    def _load_cli(self):
        import importlib.util

        script = Path(__file__).resolve().parent.parent / "scripts" / "annotate.py"
        spec = importlib.util.spec_from_file_location("annotate_cli", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_guideline_command(self, capsys):
        cli = self._load_cli()
        assert cli.main(["guideline"]) == 0
        assert "labelling guideline" in capsys.readouterr().out.lower()

    def test_template_then_validate(self, tmp_path: Path):
        cli = self._load_cli()
        target = tmp_path / "template.json"
        assert cli.main(["template", str(target)]) == 0
        assert target.exists()
        assert cli.main(["validate", str(target)]) == 0
