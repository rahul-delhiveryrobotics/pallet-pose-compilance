# Dataset card: pallet detection and pose compliance

**Status:** assembled dataset and provenance record for the pallet-pose-compliance project.

This document distinguishes **real detection data** from **simulated pose data**. A number labelled `measured` below is derived from repository artifacts produced from the real downloaded detection data. Synthetic pose images and their ground truth are labelled `simulated`; they are not real photographs and are not evidence of real-world pose accuracy.

## 1. Purpose and dataset boundaries

The assembled real dataset is used for **object detection/localisation at bounding-box level** for two classes: `pallet` and `pallet_pocket`. It does not contain the eight-corner pallet pose annotations required by the pose estimator.

The pose training/evaluation corpus is separate: `data/synthetic_pose/` contains 3,000 rendered/projected samples and a self-constructed `ground_truth.json`. It is used only for pose training/evaluation and is explicitly simulated. Its split is 2,100 train / 450 validation / 450 test, generated with fixed seed 42 by `scripts/generate_synthetic_pose.py`.

`SOP_CV_Lab_Photos/` contains 115 JPG and 100 HEIC files. These are demonstration photos, not annotated training or evaluation data. They are **not included in model training**, and no real pose ground truth is claimed for them. `SOP_CV_Lab_Photos_JPG/` is likewise not treated as annotated model data.

## 2. Real detection sources and provenance

All three real sources were downloaded publicly from Roboflow Universe, workspace `rj-xvfw4`. The source dataset pages and citations are:

| Source | Access method | Licence and attribution |
|---|---|---|
| `pallet-detect-b9dwy`, dataset version 1 | Public Roboflow Universe download | CC BY 4.0; attribution required. [Source dataset](https://universe.roboflow.com/rj-xvfw4/pallet-detect-b9dwy/dataset/1) |
| `plh-c-1-veibf`, dataset version 1 | Public Roboflow Universe download | CC BY 4.0; attribution required. [Source dataset](https://universe.roboflow.com/rj-xvfw4/plh-c-1-veibf/dataset/1) |
| `pallet-ff5lh-fp7tw`, dataset version 1 | Public Roboflow Universe download | CC BY 4.0; attribution required. [Source dataset](https://universe.roboflow.com/rj-xvfw4/pallet-ff5lh-fp7tw/dataset/1) |

The repository records the licence as CC BY 4.0 for each source. That licence permits this assignment's use provided the required attribution is retained. Attribution for the assembled real detection data: **“Datasets by Roboflow workspace rj-xvfw4, CC BY 4.0.”** The licence reference is [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The downloaded source data was normalized locally by `scripts/merge_detection_data.py` into `data/merged_detection/`. The merge output and count record are `data/merged_detection/` and `reports/merged_detection_summary.json`.

## 3. Normalization and class decisions

The merged detection taxonomy is exactly:

- `pallet` (class 0)
- `pallet_pocket` (class 1)

The source-name mapping was:

| Source label(s) | Assembled label | Decision |
|---|---|---|
| `pallet`, `Pallet` | `pallet` | Kept and remapped. |
| `hole`, `pallet_pocket` | `pallet_pocket` | Kept and remapped; `hole` is treated as the fork-pocket opening in these sources. |
| `fork`, `front`, `wood`, `hole_left`, `hole_right`, `pallet_front` | none | Dropped; these are outside the two-class target, or are finer/inconsistent variants not retained by the merge specification. |

The dropped classes are not silently counted as negatives. Images that became empty after dropping all source classes were skipped by the merge script.

### Polygon and image handling

Source labels were handled as follows by `scripts/merge_detection_data.py`:

- Plain five-column YOLO boxes were retained after class remapping.
- Polygon/segmentation labels were converted to axis-aligned `xywh` boxes from the polygon min/max coordinates. This is a normalization step, not a claim that the original polygon geometry was preserved.
- Coordinates were clamped to `[0, 1]`; degenerate boxes were skipped.
- Images with no kept annotation after class dropping were skipped rather than retained as backgrounds.
- Matching source images were copied with dataset prefixes to avoid filename collisions. No missing image for a retained label was recorded.

## 4. Assembled real detection counts

The following counts are **measured from the merge report** `reports/merged_detection_summary.json`. “Boxes” means bounding-box annotations, not pose keypoints.

| Split | Images | `pallet` boxes | `pallet_pocket` boxes | Images skipped after class dropping |
|---|---:|---:|---:|---:|
| train | 9,147 | 50,218 | 96,493 | 3,683 |
| valid | 591 | 2,897 | 5,724 | 344 |
| test (held out) | 370 | 1,186 | 2,353 | 151 |
| **Total** | **10,108** | **54,301** | **104,570** | **4,178** |

There were **0 missing images for labels** in the merge report. The assembled real detection output is `data/merged_detection/`, with `data.yaml` declaring the two classes and train/valid/test image directories.

For traceability, the retained-image and box contributions by source were:

| Source | Train images / boxes (`pallet`, `pallet_pocket`) | Valid images / boxes | Test images / boxes |
|---|---|---|---|
| `pallet-detect-b9dwy` | 6,765 / 28,743, 57,399 | 396 / 1,554, 3,222 | 262 / 768, 1,580 |
| `plh-c-1-veibf` | 834 / 153, 1,668 | 40 / 14, 88 | 29 / 0, 58 |
| `pallet-ff5lh-fp7tw` | 1,548 / 21,322, 37,426 | 155 / 1,329, 2,414 | 79 / 418, 715 |

These are detection annotations only. No count in this section is a count of the 8-corner pose labels.

## 5. Sourcing effort and cost

- **Access effort:** public Roboflow downloads from the three cited dataset pages, followed by local merging and validation.
- **Preprocessing effort:** engineering time was spent on download, filename collision avoidance, class normalization, polygon-to-axis-aligned-box conversion, skipped-empty handling, and validation/count reporting.
- **Time/cost record:** no engineering-hour total was recorded. No monetary cost was recorded; the public downloads and local preprocessing had no recorded monetary charge.
- **Attribution obligation:** CC BY 4.0 attribution must remain with any redistribution or reuse of the real detection data. The attribution is recorded in Section 2.

## 6. Pose data: separate, simulated, and limited

`data/synthetic_pose/` is a locally generated **simulated** pose dataset, not a real detection dataset and not a substitute for real pose ground truth. It contains 3,000 simulated samples generated by `scripts/generate_synthetic_pose.py` with seed 42:

| Split | Simulated samples |
|---|---:|
| train | 2,100 |
| validation | 450 |
| test | 450 |
| **Total** | **3,000** |

The generator renders/projectively creates eight pallet-corner keypoints and writes `ground_truth.json`. The ground truth is self-constructed from known simulated poses and is labelled `simulated`. It is used only for pose training/evaluation. The generated imagery is deliberately simple and not photorealistic. The repository does not contain measured real pose labels for the demonstration photos or for the Roboflow detection corpus.

No real pose accuracy claim can be inferred from the simulated pose split. In particular, the real Roboflow test split supports held-out **detection** evaluation, while the synthetic test split supports simulated **pose** evaluation; they must not be combined or described as the same evaluation set.

## 7. Candidates and classes not used

- The Roboflow sources were used for their detection annotations only. They were not treated as pose datasets because the available assembled labels are boxes (including converted polygons), not the required eight named pallet corners with visibility.
- `fork`, `front`, `wood`, `hole_left`, `hole_right`, and `pallet_front` were not used in the assembled two-class detector. They were dropped during merge because they are outside the target taxonomy or are finer/inconsistent variants not retained by the specified mapping. Dropping them costs potentially useful subtype information and can reduce context/background diversity.
- `SOP_CV_Lab_Photos/` (115 JPG + 100 HEIC) and the derived `SOP_CV_Lab_Photos_JPG/` (100 JPG) were not used for training or evaluation. They are demonstration photos without annotation manifests; a dataset licence/provenance record for them was not available in this repository, so licence permission was not claimed. They remain useful for qualitative inspection only.
- No other candidate third-party dataset is claimed as considered or used. No API keys or secrets are included in this document or required for the recorded public-download provenance.

## 8. Split protocol and held-out data

### Real detection split

The real detection split preserves each source Roboflow dataset's original `train`, `valid`, and `test` assignment. `scripts/merge_detection_data.py` maps source `train → train`, `valid → valid`, and `test → test`; it does not reshuffle retained images. The held-out set for real detection is therefore the original Roboflow `test` split: 370 retained images and 3,539 retained boxes (1,186 `pallet` plus 2,353 `pallet_pocket`).

The split attribute is **the original source-provided split assignment**, not a newly constructed scene, pallet-instance, session, or camera-pose grouping. Source metadata needed to form a grouped scene/session split was unavailable. This is an explicit limitation: duplicates or near-duplicates may cross source split boundaries, and generalisation to new scenes/sessions is not established. Source split provenance is retained rather than replaced with an invented grouping claim.

When real detection accuracy is reported, it must be computed on this held-out original test set, never on the training images. The repository's measured detection report (`reports/detection_metrics.json`) identifies this test split and labels its metrics `measured`; those metrics are bounding-box detection metrics only.

### Synthetic pose split

The synthetic pose generator deterministically assigns the 3,000 simulated samples by index using the documented 70/15/15 fractions and seed 42: 2,100 train, 450 validation, and 450 test. The synthetic test set is held out from synthetic pose training by this deterministic assignment. It differs from training in membership and is explicitly simulated; it is not a held-out real-world scene/session evaluation.

## 9. One-page practical pallet-corner labeling guideline

This guide matches the schema and behavior in `scripts/annotate.py` and `src/pallet_pose_compliance/annotation.py`. It applies to future **pose keypoint annotation** and does not retroactively convert the real detection boxes into pose labels.

### What to label

Annotate each pallet instance with exactly eight named structural corners, in this fixed order:

1. `bottom_corner_0`
2. `bottom_corner_1`
3. `bottom_corner_2`
4. `bottom_corner_3`
5. `top_corner_0`
6. `top_corner_1`
7. `top_corner_2`
8. `top_corner_3`

Corners 0–3 are the four bottom deck-board corners that contact the floor and are the pose reference points. Corners 4–7 are the corresponding top deck corners used for height and geometry. For each index, `bottom_corner_i` and `top_corner_i` are the ends of the same vertical pallet edge.

The corner index convention is: corner 0 is the near-right corner from the camera; indices proceed counter-clockwise (`0 → 1 → 2 → 3`) around the deck when viewed from above. Use the same index on the bottom and top corner at an edge; do not reorder corners to follow apparent image left-to-right order.

### Visibility and placement

Give every keypoint exactly one visibility value:

- **`visible`** — the corner is directly observable in the image; place its pixel coordinate on the corner.
- **`occluded`** — the corner is hidden by the load or another pallet, but its location is known/inferable from visible structure; place the inferred pixel coordinate and mark it occluded.
- **`absent`** — the corner is outside the frame or cannot be located reliably; do not invent a coordinate. The machine-readable writer stores absent coordinates as `x=0, y=0, v=0` for COCO compatibility and reads them back as null pixels with visibility `absent`.

Never invent a keypoint merely to complete the eight-point pattern. If the corner cannot be located with defensible geometric evidence, use `absent`; if it is hidden but inferable, use `occluded`. Do not label a pallet edge, box corner, fork opening, or shadow as a pallet deck corner.

### Image integrity and review

Before labeling, confirm the image opens, has valid dimensions, is not corrupt or partially decoded, and shows enough of the pallet to make a decision. Preserve the source image reference; do not silently rotate, crop, mirror, or otherwise alter the image while placing labels. Record the image dimensions in the annotation document.

After labeling, review all eight names and their order, check bottom/top correspondence, verify that visible points lie on the actual structural corners, and inspect occluded points against the visible deck geometry. Check that absent points have no fabricated pixel coordinates. Reject or send for review any annotation with swapped indices, impossible edge connections, a point on a non-pallet object, or a low-quality/corrupt image. A second review is required for ambiguous perspective, heavy occlusion, truncation, or uncertain corner identity; resolve only from image evidence and mark unresolved points `absent` when evidence is insufficient.

### Machine-readable output

Use `python scripts/annotate.py template <output.json>` to create a template and `python scripts/annotate.py validate <input.json>` to validate and round-trip it. The format is a documented COCO-keypoints-style JSON subset with `images`, `annotations`, and `categories`. Each annotation stores 8 × 3 = 24 keypoint values `[x, y, v]`, a bounding box, and `num_keypoints`; visibility maps to COCO `v=2` visible, `v=1` occluded, and `v=0` absent. The category keypoint order and skeleton are fixed by the tooling. The tool accepts exactly the visibility values `visible`, `occluded`, and `absent` and rejects malformed or incomplete eight-corner annotations.

## 10. Known biases, limitations, and gaps

- **Source-domain bias:** the real detector inherits the camera viewpoints, environments, pallet appearances, lighting, and annotation conventions represented by the three Roboflow sources. Performance in the target warehouse may differ.
- **Duplicates and near-duplicates:** possible duplicates or near-duplicates are not ruled out because source scene/session metadata was unavailable. The original source split is preserved, but it is not a grouped scene/session holdout.
- **Inconsistent taxonomy:** source names and granularity differ (`Pallet`/`pallet`, `hole`/`pallet_pocket`, and dropped subtype labels). The merge decision improves a two-class baseline but can introduce semantic mismatch.
- **Polygon-to-box looseness:** polygons were reduced to axis-aligned boxes. Boxes can be looser for rotated or oblique objects, and the result is not equivalent to a native segmentation or oriented-box annotation.
- **Background selection:** images empty after dropping all target classes were skipped, so the merged detector corpus is not a random sample of source backgrounds and contains no intentionally retained empty negatives from that filtering step.
- **No measured camera calibration in this dataset card:** camera calibration and physical camera geometry are separate pipeline artefacts. The pose corpus uses a simulated calibration; no measured calibration or real pose labels are claimed here.
- **No measured real pose labels:** the real detection boxes do not establish pallet corner correspondences, metric pose, face identity, or measured pose error.
- **Single-side visibility:** the intended camera sees each pallet from one side, leaving hidden corners and rear load structure uncertain. This limits keypoint annotation and downstream SOP verification.
- **Synthetic-to-real gap:** the 3,000 pose images are simple rendered/projected samples, not photorealistic warehouse captures. Simulated pose metrics cannot be presented as real-world pose accuracy.
- **Demonstration-photo gap:** the 115 JPG and 100 HEIC files in `SOP_CV_Lab_Photos/` are not annotated training/evaluation data, so they cannot close the real-pose ground-truth gap.
- **Label uncertainty:** occluded keypoints may be inferred from visible geometry, while absent points are intentionally unavailable. This affects any future localisation metric and must be reported by visibility/provenance rather than collapsed into a single claim.

## 11. Reproducibility and provenance summary

- Rebuild the real detection merge with `python scripts/merge_detection_data.py`; it writes `data/merged_detection/` and `reports/merged_detection_summary.json`.
- Inspect the resulting real detection taxonomy in `data/merged_detection/data.yaml`.
- Use `python scripts/annotate.py guideline` to print the implemented labeling text, or `template`/`validate` for the documented COCO-keypoints-style format.
- Recreate the simulated pose corpus with `python scripts/generate_synthetic_pose.py --n-samples 3000 --seed 42`; outputs include `data/synthetic_pose/ground_truth.json` labelled `simulated`.

All claims in this document are limited to the repository artifacts and facts recorded above. Real detection counts are measured from the merge summary; synthetic pose counts and ground truth are simulated; unmeasured real pose performance, camera calibration, and demonstration-photo annotations remain unavailable rather than inferred.
