#!/usr/bin/env python3
"""Merge three Roboflow detection datasets into one unified 2-class YOLO detection dataset.

Unified classes:
    0: pallet
    1: pallet_pocket

Class mapping (by source dataset, using each dataset's own name list):
    'pallet', 'Pallet'          -> pallet (0)
    'hole', 'pallet_pocket'     -> pallet_pocket (1)
    everything else             -> DROPPED (fork, front, wood, hole_left,
                                            hole_right, pallet_front)

Label handling:
    - Plain 5-col YOLO box labels: kept, class id remapped.
    - Polygon/segmentation labels (odd col count > 5: class + N (x,y) pairs):
      converted to an axis-aligned bounding box (xywh) from min/max of the
      polygon points. Points are already normalized 0..1.

Split assignment is preserved (train->train, valid->valid, test->test).
Filenames are prefixed with a short dataset tag to avoid collisions.

Images that end up with >=1 kept box are copied. Images whose label file is
empty after dropping classes are SKIPPED (not kept as backgrounds) so the
detector trains only on frames with relevant annotations. This is documented
in reports/detection_training_notes.md.

Re-runnable: rebuilds data/merged_detection/ from scratch each run.
"""
import json
import shutil
from collections import Counter
from pathlib import Path

# --- config -----------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "merged_detection"
REPORTS = REPO / "reports"

UNIFIED_NAMES = ["pallet", "pallet_pocket"]
PALLET, POCKET = 0, 1

# Per-dataset: (tag, dir, {source_class_id: unified_id or None to drop})
# Mappings derived from each dataset's data.yaml `names` list.
DATASETS = [
    (
        "b9",
        "pallet-detect-b9dwy",
        # ['Pallet','fork','front','hole','pallet','pallet_front','pallet_pocket']
        {0: PALLET, 1: None, 2: None, 3: POCKET, 4: PALLET, 5: None, 6: POCKET},
    ),
    (
        "plh",
        "plh-c-1-veibf",
        # ['front','hole','hole_left','hole_right','pallet','pallet_front','pallet_pocket','wood']
        {0: None, 1: POCKET, 2: None, 3: None, 4: PALLET, 5: None, 6: POCKET, 7: None},
    ),
    (
        "ff",
        "pallet-ff5lh-fp7tw",
        # ['hole','pallet']
        {0: POCKET, 1: PALLET},
    ),
]

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def poly_to_box(coords):
    """coords: flat list of normalized x,y pairs -> (cx, cy, w, h) normalized."""
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return cx, cy, w, h


def clamp01(v):
    return max(0.0, min(1.0, v))


def convert_label_file(path, class_map):
    """Return list of remapped box lines (strings) and per-class counts."""
    out_lines = []
    counts = Counter()
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            try:
                src_cls = int(float(parts[0]))
            except ValueError:
                continue
            dst = class_map.get(src_cls, None)
            if dst is None:
                continue  # dropped class

            nvals = len(parts) - 1
            if nvals == 4:
                # plain box: cx cy w h
                cx, cy, w, h = (float(x) for x in parts[1:5])
            elif nvals >= 6 and nvals % 2 == 0:
                # polygon: N (x,y) pairs -> axis-aligned box
                coords = [float(x) for x in parts[1:]]
                cx, cy, w, h = poly_to_box(coords)
            else:
                # unexpected format; skip defensively
                continue

            cx, cy, w, h = clamp01(cx), clamp01(cy), clamp01(w), clamp01(h)
            if w <= 0 or h <= 0:
                continue
            out_lines.append(f"{dst} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            counts[dst] += 1
    return out_lines, counts


def main():
    # fresh output
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in SPLITS:
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    summary = {
        "unified_names": UNIFIED_NAMES,
        "splits": {},
        "per_dataset": {},
        "skipped_empty_after_drop": {},
        "missing_image_for_label": 0,
    }
    split_img_counts = {s: 0 for s in SPLITS}
    split_box_counts = {s: Counter() for s in SPLITS}
    skipped_empty = {s: 0 for s in SPLITS}
    missing_img = 0

    for tag, dname, cmap in DATASETS:
        ds_imgs = {s: 0 for s in SPLITS}
        ds_boxes = {s: Counter() for s in SPLITS}
        for s in SPLITS:
            lbl_dir = DATA / dname / s / "labels"
            img_dir = DATA / dname / s / "images"
            if not lbl_dir.is_dir():
                continue
            for lf in sorted(lbl_dir.glob("*.txt")):
                lines, counts = convert_label_file(lf, cmap)
                if not lines:
                    skipped_empty[s] += 1
                    continue
                # find matching image
                stem = lf.stem
                img_path = None
                for ext in IMG_EXTS:
                    cand = img_dir / (stem + ext)
                    if cand.exists():
                        img_path = cand
                        break
                if img_path is None:
                    # try any file starting with stem
                    matches = list(img_dir.glob(stem + ".*"))
                    matches = [m for m in matches if m.suffix.lower() in IMG_EXTS]
                    if matches:
                        img_path = matches[0]
                if img_path is None:
                    missing_img += 1
                    continue

                new_stem = f"{tag}_{stem}"
                shutil.copy2(img_path, OUT / s / "images" / (new_stem + img_path.suffix))
                with open(OUT / s / "labels" / (new_stem + ".txt"), "w") as out:
                    out.write("\n".join(lines) + "\n")

                split_img_counts[s] += 1
                split_box_counts[s].update(counts)
                ds_imgs[s] += 1
                ds_boxes[s].update(counts)

        summary["per_dataset"][dname] = {
            "tag": tag,
            "images": ds_imgs,
            "boxes": {s: {UNIFIED_NAMES[k]: v for k, v in ds_boxes[s].items()} for s in SPLITS},
        }

    for s in SPLITS:
        summary["splits"][s] = {
            "images": split_img_counts[s],
            "boxes": {UNIFIED_NAMES[k]: v for k, v in split_box_counts[s].items()},
        }
        summary["skipped_empty_after_drop"][s] = skipped_empty[s]
    summary["missing_image_for_label"] = missing_img

    # write data.yaml with absolute paths for ultralytics
    yaml_text = (
        f"path: {OUT.resolve()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n\n"
        f"nc: {len(UNIFIED_NAMES)}\n"
        f"names: {UNIFIED_NAMES}\n"
    )
    (OUT / "data.yaml").write_text(yaml_text)

    (REPORTS / "merged_detection_summary.json").write_text(json.dumps(summary, indent=2))

    # console summary
    print("=== Merged detection dataset summary ===")
    print(f"Output: {OUT}")
    for s in SPLITS:
        b = summary["splits"][s]["boxes"]
        print(f"  {s}: images={split_img_counts[s]} boxes={b} (skipped_empty={skipped_empty[s]})")
    print(f"  missing_image_for_label={missing_img}")
    print(f"data.yaml written to {OUT / 'data.yaml'}")
    print(f"summary written to {REPORTS / 'merged_detection_summary.json'}")


if __name__ == "__main__":
    main()
