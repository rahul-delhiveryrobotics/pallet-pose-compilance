# Requirements checklist (R1–R29)

Status vocabulary: **complete** means an artifact/code path exists in this repository; **partial** means some acceptance criteria are implemented but a required evidence gap remains; **unavailable/not completed** means the assignment evidence was not produced and is not inferred. Provenance labels are stated where relevant.

| Requirement | Status | Artifact / evidence | Honest limitation |
|---|---|---|---|
| R1 Dataset sourcing and provenance | complete | `DATASET.md` | Demonstration-photo licence/provenance is unavailable; this is explicitly not claimed. |
| R2 Split protocol | partial | `DATASET.md`, `reports/merged_detection_summary.json` | Original source split is preserved, but no scene/session/pallet grouped holdout was available. |
| R3 Annotation tooling | complete | `scripts/annotate.py`, `src/pallet_pose_compliance/annotation.py` | Tooling exists; real pose annotations for the detection corpus do not. |
| R4 Trained detection model and weights | complete | `weights/yolo26-det-pallet-v0.pt`, training metadata/notes | Architecture citation is included in README; cost is recorded as run notes, not a full engineering-hour ledger. |
| R5 Separate detection/localisation accuracy | partial | `reports/detection_metrics.json`, `reports/pose_eval_metrics.json`, metric helpers | Real detection is measured on 370 test images; real eight-corner localisation data is unavailable. Pose metrics are simulated and separate. |
| R6 Accuracy ceiling estimate | complete | `reports/accuracy_ceiling.md` | Estimated but deliberately unquantified; no unsupported percentage is invented. |
| R7 Metric pose output | complete | `src/.../geometry/pose.py`, output schema, pipeline | Real-world pose validity is unverified because calibration/ground truth are simulated. |
| R8 Camera calibration artefacts | partial | `configs/calibration/sim-cal-v0.yaml`, calibration module | Intrinsics/extrinsics and stated reprojection error are **simulated**, not real measured calibration. |
| R9 Pose method/geometric soundness | complete | geometry pose/PnP, ray-plane cross-check, ambiguity and rejection paths | Physical scene validation is unavailable. |
| R10 Self-constructed pose evaluation/GT | complete | `src/.../geometry/pose_eval.py`, `reports/pose_eval_metrics.json`, notes | Ground truth is self-constructed **simulated**, never assignment-provided or measured. |
| R11 Pose error distributions/tolerance | complete | `reports/pose_eval_metrics.json`, `reports/pose_eval_notes.md` | 450 simulated test samples; ±2 cm/±3° is an evaluation target, not a guarantee. |
| R12 Sensitivity/usable envelope | complete | `scripts/pose_sensitivity.py`, `reports/pose_sensitivity.json`, `.md` | 12 listed grid conditions only; all **simulated**, no extrapolation or real camera measurement. |
| R13 Pose uncertainty/unreliable behavior | complete | Monte Carlo propagation in geometry pose module, schema/config | Uncertainty is **estimated** from assumptions, not measured against real GT. |
| R14 SOP rule triage | complete | `src/.../sop/triage.py`, `DATASET.md`/README summary | Single-side camera limitations remain inherent. |
| R15 Verifiable SOP subset | complete | `src/.../sop/checks.py`, pipeline | Rules 4/6/8 remain unresolved; pose-dependent checks invalidate on unavailable pose. |
| R16 Per-check confidence semantics | complete | SOP checks/schema | No calibrated pass-probability mapping is claimed. |
| R17 Per-pallet verdict aggregation | complete | `configs/verdict.yaml`, verdict engine | Results depend on available evidence and may correctly become `MANUAL_INSPECTION`. |
| R18 Configurable SOP thresholds | complete | `configs/sop_thresholds.yaml`, SOP config/checks | Thresholds are externalized; no unrelated code changes needed. |
| R19 Latency analysis/honest hardware provenance | partial | `scripts/benchmark.py` | Host/stub path is available; a benchmark run is not used here as a trained-target claim. Jetson measurement is unavailable. |
| R20 Target hardware expectations | partial | `scripts/benchmark.py`, README | Jetson Orin Nano 15 W / 15 FPS discussion is **estimated** only; no target hardware benchmark. |
| R21 Quantisation accuracy cost | unavailable/not completed | No measured INT8 task-metric delta report exists | Quantisation measurement on keypoint/pose/SOP metrics was not completed. |
| R22 Failure behavior/downstream signaling | complete | schema `FailureSignal`, CLI, pipeline, reason codes | Real-world operating failures still require deployment testing. |
| R23 Temporal persistence | complete | `src/.../tracking/tracker.py`, design notes | No measured multi-frame stability benchmark is claimed. |
| R24 Auditable versioned output schema | complete | `src/.../output/schema.py`, generated outputs | Examples are illustrative; actual per-image outputs depend on the run. |
| R25 Illustrative verdict examples | complete | `outputs/examples/*.json`, `outputs/examples/README.txt` | Explicitly illustrative, not measured outcomes. |
| R26 Honesty/provenance discipline | complete | provenance enums/helpers, all reports and configs | Missing evidence is labelled unavailable/simulated/estimated rather than filled in. |
| R27 Uncertainty discipline | complete | Monte Carlo pose uncertainty, verdict gate, SOP statuses | Assumption-based uncertainty is estimated, not empirically calibrated. |
| R28 Reproducibility | complete | `configs/pipeline.yaml` seed 42, scripts, reports, README commands | Full pytest was intentionally not run for this deliverable; targeted checks are listed in README. |
| R29 README/deliverables | complete | root `README.md`, reports, images, examples | No five-minute screen recording is included because it must be recorded by the user. |

## Explicit unavailable/not-completed evidence

- **Real pose calibration:** unavailable; `sim-cal-v0` is simulated.
- **Real pose ground truth:** unavailable; demonstration JPG/HEIC images and real detection boxes have no eight-corner metric pose labels.
- **Quantisation measurement:** unavailable/not completed; no task-metric INT8 delta is reported.
- **Jetson Orin Nano hardware benchmark:** unavailable; any Jetson performance discussion is estimated only.
- **Five-minute screen recording:** not included; it must be recorded by the user.
