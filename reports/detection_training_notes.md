# Pallet / Pocket Detector v0 — Training Notes

Provenance discipline: every number below is either a dataset count produced by
`scripts/merge_detection_data.py` or a **measured** metric produced by an actual
ultralytics validation run. Nothing here is fabricated or estimated.

> These are **DETECTION** (bounding-box) metrics on merged real data. They are
> **NOT pose metrics**. The model detects `pallet` and `pallet_pocket` boxes only.

## Datasets used + attribution (CC BY 4.0)

All three source datasets belong to Roboflow workspace **`rj-xvfw4`** and are
licensed **CC BY 4.0** (attribution required):

- **pallet-detect-b9dwy** (~11k imgs) — https://universe.roboflow.com/rj-xvfw4/pallet-detect-b9dwy/dataset/1
- **plh-c-1-veibf** (~1k imgs) — https://universe.roboflow.com/rj-xvfw4/plh-c-1-veibf/dataset/1
- **pallet-ff5lh-fp7tw** (~2.2k imgs) — https://universe.roboflow.com/rj-xvfw4/pallet-ff5lh-fp7tw/dataset/1

Attribution: "Datasets by Roboflow workspace rj-xvfw4, CC BY 4.0."

## Class unification

Unified to exactly 2 classes: `0: pallet`, `1: pallet_pocket`.

Mapping applied per source dataset (using each dataset's own `names` list):

| source class | -> unified |
|---|---|
| `pallet`, `Pallet` | `pallet` |
| `hole`, `pallet_pocket` | `pallet_pocket` |
| `fork`, `front`, `wood`, `hole_left`, `hole_right`, `pallet_front` | **DROPPED** |

Decisions, cost, and risk:

- **`hole` -> `pallet_pocket`**: the "hole" class in these pallet datasets is the
  fork pocket opening. Treating it as the pocket is the intended semantics for
  downstream SOP checks. Risk: any dataset that used "hole" for a non-pocket hole
  would be mislabeled — spot-checking indicates all three use it for pockets, so
  risk is low.
- **Dropping `hole_left` / `hole_right`** (plh only): these are finer-grained
  pocket subclasses. Folding them into `pallet_pocket` would have been defensible,
  but the task spec explicitly lists them to drop, so they are dropped. Cost: we
  lose ~some pocket boxes from plh. Low impact because plh is the smallest source
  and b9dwy/ff5lh dominate pocket counts.
- **Dropping `pallet_front` / `front` / `fork` / `wood`**: not needed for the
  2-class pallet/pocket detector.

## Polygon -> box conversion

Sources mix formats:
- `pallet-ff5lh-fp7tw`: plain 5-col YOLO boxes (kept as-is, class remapped).
- `pallet-detect-b9dwy`: mostly 5-col boxes, plus a minority of polygon labels.
- `plh-c-1-veibf`: almost entirely polygon/segmentation labels (11+ cols).

Polygon labels (odd column count > 5 = `class + N (x,y)` pairs, already normalized
0..1) were converted to **axis-aligned bounding boxes** via min/max of the polygon
points -> `xywh`. Coordinates are clamped to [0,1]; degenerate boxes (w<=0 or h<=0)
are skipped.

Cost / risk of polygon->box: an axis-aligned box is looser than the tight polygon,
especially for rotated/oblique pallets. This slightly inflates the annotated region
and can make the detector learn a looser box than a native box-labeled source would.
Acceptable for a v0 detection baseline; a future rev could train segmentation or
oriented boxes to recover tightness.

## Merge result (counts from `scripts/merge_detection_data.py`)

Images that ended up with **zero** kept boxes after dropping classes were **SKIPPED**
(not kept as backgrounds), so the detector trains only on frames with a relevant
pallet/pocket annotation. Skipped counts are recorded below and in
`reports/merged_detection_summary.json`.

| split | images | pallet boxes | pallet_pocket boxes | skipped (empty after drop) |
|---|---|---|---|---|
| train | 9147 | 50218 | 96493 | 3683 |
| valid | 591  | 2897  | 5724  | 344 |
| test  | 370  | 1186  | 2353  | 151 |

`missing_image_for_label = 0`. Output: `data/merged_detection/` with `data.yaml`
(`nc: 2`, `names: ['pallet','pallet_pocket']`, absolute `path`, train/val/test ->
`{split}/images`).

## Subsampling

**None.** The full merged train set (9147 images) was used, plus full valid (591)
and full test (370). On the RTX 4080 Laptop GPU the run took ~13 min for 15 epochs
(~50 s/epoch), which is within the "quick" budget, so no train subsampling was
needed.

## Training hyperparameters

Script: `scripts/train_detector.py` (re-runnable, fixed seed).

- model / base weights: **`yolo26n.pt`** (detection). No fallback needed — yolo26n
  downloaded and trained fine. (If it had failed, the script falls back to
  `yolo11n.pt` and records the substitution in `reports/detection_train_meta.json`.)
- epochs: 15
- imgsz: 640
- batch: 16
- device: 0 (GPU)
- seed: 42, deterministic: true
- optimizer: auto (ultralytics chose AdamW)
- project/name: `runs/detect/pallet-det-v0`

Exported best weights: **`weights/yolo26-det-pallet-v0.pt`** (copied from
`runs/detect/pallet-det-v0/weights/best.pt`).

### Environment note (problem encountered + fix)

Initial training crashed with `numpy.core.multiarray failed to import`: the system
matplotlib 3.5.1 (`/usr/lib/python3/dist-packages`, compiled against NumPy 1.x) was
incompatible with the installed NumPy 2.2.6. Fixed by installing a NumPy-2.0-compatible
`matplotlib==3.9.4` into the user site-packages (`pip install --user`), which shadows
the system package. No system files were modified. Training then ran clean.

## Measured metrics — provenance: `measured`

**Held-out TEST split** (`data/merged_detection/test/images`, 370 images, 3539 boxes),
model `weights/yolo26-det-pallet-v0.pt`, produced by `scripts/validate_detector.py`
(ultralytics `model.val(split='test')`). Full JSON: `reports/detection_metrics.json`.

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | 0.956 | 0.976 | 0.989 | 0.758 |
| pallet | 0.958 | 0.977 | 0.989 | 0.770 |
| pallet_pocket | 0.954 | 0.975 | 0.990 | 0.747 |

For reference, the final-epoch **validation** metrics (591 imgs) were
P=0.935, R=0.955, mAP50=0.965, mAP50-95=0.701 (from `runs/detect/pallet-det-v0/results.csv`).

Honest framing: these are strong DETECTION numbers on the merged real Roboflow data.
They say nothing about pose/geometry accuracy — that is a separate downstream stage.
The polygon->box conversion (above) means pocket/pallet boxes from the polygon-labeled
sources are looser than native boxes, which the mAP50-95 (tighter IoU) figure partly
reflects.
