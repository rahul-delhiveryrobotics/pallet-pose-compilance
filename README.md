# Pallet pose compliance — assignment deliverables

This repository implements a single-image pallet detection → keypoint/geometry pose → SOP-PAL-03 analysis → verdict → versioned Assessment pipeline. The project follows a strict provenance discipline: **measured** means computed from an actual evaluation run on real data; **simulated** means synthetic renders/calibration/ground truth; **estimated** means assumption-based reasoning; **unavailable** means the evidence was not produced. No full real-world compliance claim is made.

## 1. Approach and significant decisions

The real detection data was merged from three public Roboflow Universe datasets in workspace `rj-xvfw4`, normalized to two classes (`pallet`, `pallet_pocket`), and retained with the source train/valid/test assignments. The merged corpus has 9,147 train images, 591 valid images, and 370 held-out test images. Polygon annotations were converted to axis-aligned boxes; images empty after target-class filtering were skipped. These choices make a reproducible detector baseline, but cost polygon detail, background-negative coverage, taxonomy consistency, and scene/session split strength. Full provenance, CC BY 4.0 attribution, counts, and limitations are in `DATASET.md`.

The detector uses Ultralytics YOLO26n weights (`weights/yolo26-det-pallet-v0.pt`) at 640 px. The recorded detector run used 15 epochs, batch 16, seed 42, and full merged training data; the short-run choice costs potentially higher accuracy from longer tuning. Ultralytics/YOLO source is documented at [Ultralytics](https://github.com/ultralytics/ultralytics); the YOLO26 family is used as recorded in the repository metadata. Roboflow source citations are [pallet-detect-b9dwy](https://universe.roboflow.com/rj-xvfw4/pallet-detect-b9dwy/dataset/1), [plh-c-1-veibf](https://universe.roboflow.com/rj-xvfw4/plh-c-1-veibf/dataset/1), and [pallet-ff5lh-fp7tw](https://universe.roboflow.com/rj-xvfw4/pallet-ff5lh-fp7tw/dataset/1); attribution is “Datasets by Roboflow workspace rj-xvfw4, CC BY 4.0.”

Pose uses known-geometry PnP, not floor homography on elevated corners and not bounding-box-plus-tilt. The pallet model contains four floor corners and four elevated top corners; calibrated distortion, visibility, ray-plane cross-checks, ambiguity, and loud unavailable outputs are handled in `src/pallet_pose_compliance/geometry/`. Because there was no physical camera access, `sim-cal-v0` is a **simulated** calibration and the pose model was trained/evaluated on simple **simulated** renders. Monte Carlo uncertainty is **estimated** from documented noise assumptions. This is the right runnable fallback, but it cannot establish real pose accuracy.

SOP triage covers all eight rules. Rules 1/3/7 are partially verifiable and pose-dependent; rules 2/5 are verifiable from visible evidence; rules 4/6/8 remain unresolved from a single-side view. The verdict engine uses critical-failure dominance and a pose-quality gate rather than averaging away unreliable evidence. Thresholds are externalized in `configs/sop_thresholds.yaml`; verdict thresholds and pose gate are in `configs/verdict.yaml`.

## 2. Results as distributions

### Real detection — measured

`reports/detection_metrics.json` is a **measured** Ultralytics validation report on the original held-out real source test split: **370 images** and **3,539 boxes** (1,186 `pallet`, 2,353 `pallet_pocket`). Overall precision is 0.956, recall 0.976, mAP50 0.989, and mAP50-95 0.758. Per-class mAP50-95 is 0.770 for `pallet` and 0.747 for `pallet_pocket`. These are box-detection metrics, not pose/localisation metrics, and the source split is not a grouped scene/session holdout.

A numeric accuracy ceiling is not supported by the available evidence. `reports/accuracy_ceiling.md` labels the ceiling **estimated** and deliberately leaves its numeric value unquantified because no relabelling agreement study, irreducible-noise estimate, or grouped target-domain holdout exists. The measured mAP is context, not the ceiling.

### Pose evaluation — simulated, separate from detection

`reports/pose_eval_metrics.json` evaluates the trained pose model against self-constructed simulated ground truth on **450 simulated test renders**. A pose was available for **374** samples and unavailable for **76**; the available radial translation distribution has median **3.435 cm**, p95 **8.516 cm**, max **11.24 cm** (n=374). The orientation distribution has median **0.554°**, p95 **1.968°**, max **3.567°** (n=374). The combined ±2 cm/±3° target pass fraction is 0.2834 over available poses, while unavailable coverage is reported separately. These are **simulated** results and do not establish real-world compliance.

### Sensitivity — simulated

Run `python3 scripts/pose_sensitivity.py` to regenerate `reports/pose_sensitivity.json` and `reports/pose_sensitivity.md`. The fixed small grid has 12 evaluated conditions: height errors 0/0.10/0.20 m and tilt errors 0/3/6°, each at short depth 2.5–3.5 m and long depth 6.0–8.0 m, with 16 simulated samples per condition and seed 42. Each row includes separate radial/per-axis position and rotation distributions with counts, plus one of `meets_tolerance`, `fails_tolerance`, or `insufficient_evidence`. These results are all **simulated**; no real camera measurement is used and no unevaluated condition is extrapolated.

## 3. Failure analysis

The repository includes exactly three **representative** failure-case images, not statistically ranked worst cases:

1. [`outputs/failure_cases/representative_01_simulated_keypoint_localisation_gap.png`](outputs/failure_cases/representative_01_simulated_keypoint_localisation_gap.png) — keypoint localisation error can amplify into pose error, especially when the real corpus lacks eight-corner labels.
2. [`outputs/failure_cases/representative_02_simulated_pose_uncertainty.png`](outputs/failure_cases/representative_02_simulated_pose_uncertainty.png) — insufficient/ill-conditioned evidence or propagated uncertainty should produce explicit unavailable pose and manual inspection rather than a fabricated number.
3. [`outputs/failure_cases/representative_03_simulated_domain_or_visibility_gap.png`](outputs/failure_cases/representative_03_simulated_domain_or_visibility_gap.png) — simple synthetic appearance, hidden regions, and the real/synthetic domain gap limit evidence.

Detailed symptom, root cause, provenance, and mitigation are in [`reports/failure_analysis.md`](reports/failure_analysis.md). The images originate from simulated test renders and carry banners only; they are not claims that the selected files are the three worst samples.

## 4. What could not be finished and why

- **Measured real camera calibration:** unavailable because no physical camera/fiducial setup was available; `sim-cal-v0` is explicitly simulated.
- **Measured real pose ground truth:** unavailable. The real Roboflow corpus contains detection boxes, while demonstration JPG/HEIC photos have no eight-corner metric labels. The 3,000-sample pose corpus is synthetic and its ground truth is self-constructed.
- **Real-world compliance validation:** unavailable; pose and SOP outputs on real scenes are not evidence of compliance.
- **Quantisation accuracy cost:** not completed; no INT8 task-metric delta is reported.
- **Jetson Orin Nano benchmark:** unavailable. `scripts/benchmark.py` distinguishes host measured latency from an estimated Jetson expectation; this README does not claim a target-hardware measurement or guaranteed 15 FPS.
- **Five-minute screen recording:** not included because it must be recorded by the user.
- Full test suite verification after the deliverables: `406 passed`. Targeted syntax/JSON/file checks also passed.

## 5. AI tool usage and reproducibility

The assistant helped with implementation and documentation, including the sensitivity wrapper, reports, examples, and traceability documents. One concrete AI mistake was caught: the initial live demo failed because `roboflow` installed `opencv-python-headless`, which broke `cv2.imshow`; this was diagnosed and fixed by removing the headless package and reinstalling GUI-enabled OpenCV. The incident is also in `reports/decision_log.md`.

### Setup and run commands

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

# reliable slideshow over images
./run.sh images data/synthetic_pose/images/test
# live camera demo
./run.sh demo
# one-image pipeline and JSON Assessment output
./run.sh image path/to/image.jpg
# deterministic sensitivity reports
python3 scripts/pose_sensitivity.py
# targeted tests (do not use this assignment note as a full pytest claim)
python3 -m py_compile scripts/pose_sensitivity.py
```

`./run.sh test` invokes the full pytest suite, but it was intentionally not run for this deliverable. Existing reports and configs are the reproducibility record: seed 42 in `configs/pipeline.yaml`, model IDs in report JSON, and calibration ID `sim-cal-v0` in simulated pose artifacts.

### Output JSON and provenance

A normal pipeline run writes one versioned Assessment per detected pallet under `outputs/`. The schema contains pallet ID, timestamp, source image, model/calibration IDs, pose and uncertainty, all eight SOP checks, verdict reasoning, quality/reason codes, timings, and a failure signal. Unavailable values are explicit JSON `null`. The three files under `outputs/examples/` are schema-valid **illustrative** PASS, FAIL, and MANUAL_INSPECTION examples; see their sidecar README. Reports distinguish measured real detection (`detection_metrics.json`) from simulated pose/sensitivity (`pose_eval_metrics.json`, `pose_sensitivity.json`) and estimated ceiling/uncertainty/target-hardware expectations. No simulated result has been upgraded to measured, and no full real-world compliance claim is made.

The assignment traceability map is [`reports/requirements_checklist.md`](reports/requirements_checklist.md), and significant tradeoffs are [`reports/decision_log.md`](reports/decision_log.md).
