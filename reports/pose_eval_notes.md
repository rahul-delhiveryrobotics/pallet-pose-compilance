# Pose evaluation notes (SIMULATED ground truth)

## Method

- Ground truth is **self-constructed and SIMULATED**: the nominal pallet model is placed at a known Floor_Frame pose `(x, y, theta)` and projected through the SIMULATED `sim-cal-v0` calibration with `cv2.projectPoints` (see `scripts/generate_synthetic_pose.py`, sidecar `data/synthetic_pose/ground_truth.json`).
- GT is **never** read from the estimator under test. The forward model (place + project) and the inverse model (`estimate_pose_with_uncertainty`: keypoints -> pose via known-geometry PnP) are independent code paths.
- The trained model `weights/yolo26-pose-pallet-v0.pt` predicts the 8 pallet corner keypoints on each held-out TEST image. Predicted keypoints (named `bottom_corner_0..3` / `top_corner_0..3` in the generator's order) feed the PnP estimator with `sim-cal-v0` + the nominal pallet model.
- Estimated Floor_Frame pose is compared to the simulated GT. Translation error (per-axis + radial, cm) and rotation error (deg, folded modulo the long-axis ambiguity) are reported as distributions with sample counts via the existing `pose_error.compute_pose_error_report` helper.

## Results (SIMULATED)

- TEST samples: **450** (YOLO detected: 450, no detection: 0).
- Pose available: **374**, unavailable: **76** (unavailable rate 0.1689).
- Radial translation error (cm): median 3.435, p95 8.516, max 11.24, n=374.
- Rotation error (deg): median 0.554, p95 1.968, max 3.567, n=374.

### Tolerance pass-fractions (target ±2 cm / ±3°, NOT a guarantee)

- Radial translation within ±2 cm: **0.2861**
- Per-axis dx within ±2 cm: 0.885; dy within ±2 cm: 0.3128
- Orientation within ±3°: **0.9947**
- Combined (radial AND orientation): **0.2834**
- Overall usable-envelope label (whole TEST set as one condition): **fails_tolerance**

## Honest caveats

- All ground truth is **simulated**, never measured. The tolerance is an evaluation **target**, not a guarantee, and these numbers do **not** establish real-world compliance.
- Training and evaluation use **simple synthetic renders** (filled/edged quadrilaterals), not photorealistic imagery. There is a substantial **synthetic-to-real gap**; real-camera pose accuracy will differ and must be measured separately on real, measured ground truth.
- Unavailable predictions (no detection, or the PnP estimator degrading loudly) are **counted**, not dropped, so the pass-fractions are computed only over available poses while coverage/unavailable rate is reported alongside.

_Provenance: simulated. Seed: 42. Weights: /home/delhivery/Desktop/CV_SOP/weights/yolo26-pose-pallet-v0.pt. Calibration: sim-cal-v0 (simulated)._
