# Accuracy ceiling estimate

**Provenance: estimated.** This is an honest ceiling assessment for the current *real detection task*, not a measured accuracy number and not a claim about the downstream pose system.

## Evidence available

The merged real Roboflow corpus contains 9,147 training images, 591 validation images, and 370 held-out test images for two classes: `pallet` and `pallet_pocket`. The merge report counts 50,218/96,493 training boxes for those classes and 1,186/2,353 test boxes. The measured detector report on the held-out original source test split is context only: overall precision 0.956, recall 0.976, mAP50 0.989, and mAP50-95 0.758 (`reports/detection_metrics.json`). Those measured metrics are not the ceiling.

## Estimated ceiling posture

A defensible numerical ceiling cannot be computed from the repository evidence. The data does not include a human relabelling study, repeated-label agreement, irreducible-noise estimate, or a scene/session-grouped test set. Therefore this report deliberately does **not** invent a percentage. The estimated ceiling is **unquantified from the available evidence**: it is bounded by annotation consistency and by the coverage of the merged data, but its numeric value is unavailable.

The current measured detector result should be read as performance on this particular held-out source split, not as the maximum achievable performance and not as expected warehouse performance. The original source split was preserved rather than grouped by scene, pallet instance, capture session, or camera pose; duplicates or near-duplicates may cross splits. Consequently, the measured test result may overstate generalisation to new warehouse scenes.

## Why the ceiling is limited

- Polygon/segmentation labels were converted to axis-aligned boxes. This loses boundary/orientation detail and can make the target box definition looser for oblique pallets.
- Images that became empty after dropping non-target classes were skipped. The merged detector therefore lacks intentionally retained empty negatives from that filtering step and does not sample backgrounds randomly.
- Source taxonomies differed (`pallet`/`Pallet`, `hole`/`pallet_pocket`, and dropped labels such as `fork`, `front`, `wood`, `hole_left`, `hole_right`, `pallet_front`). Semantic mismatch and discarded context can impose an annotation ceiling.
- The sources carry different viewpoints, environments, pallet appearances, lighting, and annotation conventions. The held-out split is source-provided, not a new-scene evaluation.
- The real detection corpus has boxes, not eight-corner pose labels. It cannot establish a real pose-accuracy ceiling or real-world compliance ceiling.

## What would raise the ceiling

1. Add a licence-confirmed, target-warehouse dataset covering the actual camera, ranges, lighting, pallet variants, occlusions, empty/background frames, and failure conditions.
2. Create a scene/session/pallet-instance grouped holdout and remove near-duplicates across splits so the generalisation estimate is defensible.
3. Re-annotate a stratified audit set with independent reviewers, retain native polygons or oriented boxes where appropriate, measure agreement, and resolve taxonomy disagreements.
4. Retain representative empty negatives and hard negatives instead of skipping all images that become empty after class filtering.
5. Add real eight-corner pallet annotations and measured camera calibration/floor pose references for the pose and compliance stages; the current simulated pose results cannot substitute for this.
6. Re-evaluate after each data change on a locked, target-domain test set and report distributions with sample counts.

These are data/evidence improvements, not guarantees that a future model reaches a particular percentage. No real-world compliance claim is made here.
