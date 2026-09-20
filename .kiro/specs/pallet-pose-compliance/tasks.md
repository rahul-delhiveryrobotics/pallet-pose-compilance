# Implementation Plan: Pallet Pose Estimation & Load Compliance

## Overview

This plan converts the design into incremental, test-backed coding tasks for a 5-day build. The ordering front-loads a minimal end-to-end pipeline (ingest → detect stub → pose stub → SOP stub → verdict → Assessment JSON) so there is always a runnable system, then deepens each stage in rubric-weight order (dataset+detection 30%, pose 35%, SOP 25%, deployment 10%). Nothing references a not-yet-built component.

Provenance discipline is built first (types + schema + null discipline) because every later stage depends on it. Property-based tests use Hypothesis, run ≥100 iterations each, one test per property, tagged `Feature: pallet-pose-compliance, Property {n}: {text}`. Implementation language: **Python** (matches design: Hypothesis, Ultralytics YOLO-pose, OpenCV PnP).

Conventions:
- Tasks marked with `*` are optional/stretch or hardware-/documentation-heavy and may be skipped in a time-boxed run-all-tasks flow.
- Core implementation tasks (unmarked) must be implemented.
- Each task references the requirements it implements and, where relevant, the property it validates.

## Tasks

- [x] 1. Project scaffolding, config, provenance types, and Assessment schema
  - [x] 1.1 Create project skeleton and configs
    - Create the directory layout under `src/`, `scripts/`, `configs/`, `tests/`, `reports/`, `outputs/`, `weights/`, `data/`
    - Add `pyproject.toml`/`requirements.txt` (numpy, opencv-python, pydantic, PyYAML, hypothesis, pytest, ultralytics), `pytest.ini`, `.gitignore` (large data/weights)
    - Add `configs/pipeline.yaml` (seed, model_id, input resolution 640, calibration_id), `configs/sop_thresholds.yaml` (overhang 3cm, height 1.8m, column tilt 15°, centroid 10cm), `configs/verdict.yaml` (thresholds + pose-quality weighting)
    - _Requirements: 18.1, 18.3, 28.3_
  - [x] 1.2 Implement Provenance_Label type and reason-code enums
    - Define `Provenance_Label ∈ {measured, simulated, estimated, unavailable}` and the `Reason_Code`/`Quality_Flag` enums from the Error Handling table
    - Implement a labelling helper API used by all stages; the output layer must refuse to serialise a result-bearing field without a label
    - _Requirements: 26.1, 26.3, 22.3_
  - [x] 1.3 Implement versioned Assessment JSON schema and serialisation with null discipline
    - Define pydantic models: `Assessment`, `PoseResult`, `SopCheck`, `VerdictResult`, `Keypoint`, `Detection` per Data Models; include schema_version, pallet_id, timestamp, source_image_ref, model_id, calibration_id, per-stage timings, quality_flags, reason_codes, failure signal
    - Serialise unavailable fields as explicit `null` (never zero/placeholder); provide load/parse round-trip
    - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 22.1, 22.3_
  - [ ]* 1.4 Write property test for provenance labelling
    - **Property 1: Every reported result carries a valid provenance label**
    - **Validates: Requirements 5.5, 6.2, 8.3, 10.3, 19.2, 19.4, 21.3, 25.2, 26.1**
  - [ ]* 1.5 Write property test for no downstream provenance upgrade
    - **Property 2: Provenance labels are never upgraded downstream**
    - **Validates: Requirements 26.2**
  - [ ]* 1.6 Write property test for unavailable-null discipline
    - **Property 3: Unavailable results are stated, never fabricated**
    - **Validates: Requirements 24.8, 26.4**

- [x] 2. Coordinate/geometry module with pinned conventions
  - [x] 2.1 Implement Floor_Frame, angle wrapping, symmetry reduction, and pallet 3D model
    - Implement `wrap_to_180`, circular helpers, Floor_Frame origin/axis definitions, floor-projected pallet-centre computation, symmetry-modulo orientation reduction, `Face_Identity` derivation
    - Implement `PalletModel` (bottom/top corners, dimensions, dim_uncertainty, source, provenance=`estimated` for nominal spec)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [ ]* 2.2 Write unit tests pinning coordinate conventions
    - Floor_Frame origin/axis sign checks; `wrap_to_180` boundaries (180°, −180°, 0°); metre↔cm and rad↔deg exact conversions; fixed symmetry-reduction cases
    - _Requirements: 7.2, 7.4_
  - [ ]* 2.3 Write property test for orientation symmetry invariance
    - **Property 10: Orientation is invariant under geometric symmetry**
    - **Validates: Requirements 9.7**
  - [ ]* 2.4 Write property test for circular-statistics aggregation
    - **Property 22: Orientation temporal aggregation uses correct circular statistics**
    - **Validates: Requirements 23.2**

- [x] 3. Minimal end-to-end pipeline wiring with stubs (early smoke test)
  - [x] 3.1 Wire ingest → stubbed detect/pose/SOP → verdict → Assessment output
    - Implement ingest (frame ref + timestamp + integrity check), a stub detector returning fixed detections, a stub pose returning `unavailable`, a stub SOP returning triage-only results, and the verdict engine call, producing a schema-valid Assessment written to `outputs/`
    - Add a minimal CLI entrypoint that processes one image end to end
    - _Requirements: 24.1_
  - [x] 3.2 Add end-to-end smoke test
    - Run the full pipeline on one synthetic frame and assert exactly N Assessments produced and schema-valid
    - _Requirements: 24.1_

- [x] 4. Detection & localisation (dataset, annotation, training, metrics) — 30%
  - [x] 4.1 Implement annotation tooling
    - `scripts/annotate.py` producing COCO-keypoints-style JSON (four bottom + four top deck corners, visibility ∈ {visible, occluded, absent}); matches DATASET.md labelling guideline
    - _Requirements: 3.1, 3.2, 3.3_
  - [ ]* 4.2 Write property test for annotation round-trip
    - **Property 6: Annotation round-trip preserves labels**
    - **Validates: Requirements 3.2**
  - [x] 4.3 Implement dataset assembly, manifests, and split protocol
    - Build train/held-out manifests; split by a declared attribute (scene/session/camera pose); record class + annotation counts; expose disjointness helper
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 1.6_
  - [ ]* 4.4 Write property test for held-out disjointness
    - **Property 5: Held-out evaluation is disjoint from training data**
    - **Validates: Requirements 2.4**
  - [x] 4.5 Implement detector wrapper and executable training workflow with blocked-training fallback
    - `src/detection` YOLO-pose wrapper `detect(image) -> List[Detection]`; `scripts/train.py` fully executable with fixed seed, documented hyperparameters, checkpoint selection by held-out localisation metric
    - Blocked-training fallback: on blocker, report the blocker explicitly; any stock checkpoint run is labelled `estimated` and flagged not-assignment-trained — never mislabelled as assignment-trained
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.5, 26.3_
  - [x] 4.6 Implement separate detection and localisation metric reporting
    - `scripts/eval.py`: detection metrics (precision/recall/AP) as distributions with sample_count; localisation metrics (per-keypoint pixel + normalised error, percentiles, visibility-conditioned, failure rates) as distributions with sample_count; `measured` label only from actual eval runs
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 4.7 Write property test for distribution sample counts
    - **Property 4: Reported distributions state a correct sample count**
    - **Validates: Requirements 5.4, 11.3**
  - [x] 4.8 Integrate real detector into the pipeline (replace stub)
    - Swap the stub detector for the trained/wrapped detector; keep the fallback path wired; re-run smoke test
    - _Requirements: 4.1_
  - [ ]* 4.9 Accuracy ceiling estimate
    - Compute and report an `estimated` accuracy ceiling given current data volume with narrative on how to raise it
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Calibration and PnP pose estimation — 35%
  - [x] 6.1 Implement calibration loading and reprojection error
    - `src/calibration`: `load_calibration(id)` returning intrinsics/distortion/extrinsics (camera→Floor_Frame) + reprojection_error + provenance + id; checkerboard intrinsics + floor extrinsics path; synthetic/simulated calibration clearly labelled when no physical access
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 6.2 Implement known-geometry PnP pose estimator core
    - `estimate_pose(keypoints, calibration, pallet_model)`: undistort keypoints, solve PnP with true corner heights, compose extrinsics to Floor_Frame; ray-plane (Z=0) intersection for floor-contact points and cross-check; explicitly NOT floor homography on elevated corners and NOT bbox+tilt
    - Output `PoseResult` with position, wrapped orientation, face identity, range/viewing-angle
    - _Requirements: 7.1, 7.2, 7.3, 9.1, 9.2, 9.3, 9.4_
  - [ ]* 6.3 Write property test for projection round-trip
    - **Property 9: Known-geometry projection round-trip recovers pose**
    - **Validates: Requirements 9.2, 9.4**
  - [ ]* 6.4 Write property test for pose output completeness/ranges
    - **Property 7: Pose output completeness and valid ranges**
    - **Validates: Requirements 7.1, 7.2, 7.3, 13.1, 27.1**
  - [x] 6.5 Implement ill-conditioned rejection, ambiguity set, symmetry, and unavailable output
    - Reject on too-few keypoints, near-collinear/coplanar degeneracy, high PnP residual, or reprojection cross-check disagreement → `ILL_CONDITIONED` + unavailable with null metrics; record admissible-solution set for planar ambiguity (never silent pick); report orientation modulo symmetry; defined unavailable `PoseResult`
    - _Requirements: 9.5, 9.6, 9.7, 13.4_
  - [ ]* 6.6 Write property test for unavailable-pose degradation
    - **Property 8: Unreliable pose degrades to unavailable, never fabricated**
    - **Validates: Requirements 9.5, 13.4, 27.2**
  - [ ]* 6.7 Write property test for ambiguity set
    - **Property 11: Ambiguous geometry yields a solution set, not a silent pick**
    - **Validates: Requirements 9.6**
  - [x] 6.8 Implement Monte Carlo uncertainty propagation
    - Sample keypoint/calibration/dimension noise from documented distributions, re-solve per sample, summarise position covariance + orientation std; attach `estimated`/assumption-based provenance and documented assumptions; degrade to unavailable when uncertainty exceeds documented threshold
    - _Requirements: 13.1, 13.2, 13.3, 27.2, 27.3_
  - [ ]* 6.9 Write unit tests for uncertainty-propagation sanity
    - Zero input noise → zero spread; monotonic spread growth with input noise
    - _Requirements: 13.1_
  - [x] 6.10 Integrate calibration + real pose into pipeline (replace stub)
    - Wire keypoints → pose estimator; ensure `model_id`/`calibration_id` populate the Assessment; re-run smoke test
    - _Requirements: 8.4, 24.7, 28.2_
  - [ ]* 6.11 Write property test for identifier resolvability
    - **Property 21: Model and calibration identifiers are present and resolvable**
    - **Validates: Requirements 8.4, 24.7, 28.2**

- [x] 7. Self-constructed pose evaluation, sensitivity, and usable envelope
  - [x] 7.1 Implement simulated pose ground truth and evaluation harness
    - Self-construct pose GT via synthetic render with known projection labels (never GT from the estimator under test); attach `simulated` provenance; optional measured floor-point GT labelled `measured`; never present GT as assignment-provided
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  - [x] 7.2 Report translation/rotation error distributions and tolerance evaluation
    - Report translation and rotation error separately as distributions with sample counts; evaluate against ±2 cm / ±3° as an evaluation result (per-axis + radial pass-fraction); treat tolerance as target, not guarantee
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  - [ ]* 7.3 Write property test for tolerance-evaluation consistency
    - **Property 13: Tolerance evaluation is computed consistently**
    - **Validates: Requirements 11.4**
  - [x] 7.4 Implement sensitivity analysis and usable-envelope classification
    - Report pose-error sensitivity to camera height error and tilt error at short and long range; classify each evaluated condition as exactly one of {meets_tolerance, fails_tolerance, insufficient_evidence}; declare usable and non-usable envelope; no extrapolation beyond evaluated conditions
    - _Requirements: 12.1, 12.2, 12.3, 12.4_
  - [ ]* 7.5 Write property test for envelope partitioning
    - **Property 12: Usable-envelope classification partitions evaluated conditions**
    - **Validates: Requirements 12.4**
  - [ ]* 7.6 Measured pose ground truth (hardware-dependent, stretch)
    - Collect a small measured floor-point/pose set on real hardware, labelled `measured`, reported separately; if unavailable, state real-world compliance unverified with reasoning
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. SOP-PAL-03 analysis — 25%
  - [x] 9.1 Implement the 8-rule triage table
    - Encode all eight rules exactly once with classification ∈ {verifiable, partially_verifiable, not_verifiable}, justification, assumptions, hidden regions
    - _Requirements: 14.1, 14.2, 14.3, 14.4_
  - [ ]* 9.2 Write property test for triage coverage
    - **Property 14: SOP triage covers all eight rules**
    - **Validates: Requirements 14.1, 14.4**
  - [x] 9.3 Implement the verifiable-subset checks with config-driven thresholds
    - `analyze(detections, pose_result, sop_config)`: implement overhang, load height, column tilt, wrap presence, centroid checks reading thresholds from `configs/sop_thresholds.yaml`; produce `SopCheck` with measurement, unit, threshold_used/source
    - _Requirements: 15.1, 15.2, 18.1, 18.2_
  - [ ]* 9.4 Write property test for triage→implementation mapping
    - **Property 15: Triage classification determines implementation**
    - **Validates: Requirements 15.1**
  - [ ]* 9.5 Write property test for config-driven thresholds
    - **Property 19: SOP thresholds are driven by configuration**
    - **Validates: Requirements 18.2**
  - [ ]* 9.6 Write unit tests for SOP threshold boundaries
    - Checks exactly at 3 cm overhang, 1.8 m height, 15° tilt, 10 cm centroid (from config)
    - _Requirements: 15.1, 18.2_
  - [x] 9.7 Implement SOP status discipline, pose-dependent invalidation, and confidence semantics
    - not_verifiable rules → `unresolved` (never pass/fail/omitted); when pose unavailable, pose-dependent checks → `invalidated` + reason; attach confidence with `confidence_semantics` ∈ {raw_detector_score, calibrated_pass_probability, heuristic_evidence_quality}; never label uncalibrated confidence as calibrated
    - _Requirements: 15.3, 15.4, 16.1, 16.2, 16.3, 16.4_
  - [ ]* 9.8 Write property test for SOP status discipline
    - **Property 16: SOP status discipline**
    - **Validates: Requirements 15.3, 15.4**
  - [ ]* 9.9 Write property test for per-check confidence semantics
    - **Property 17: Per-check confidence has valid, honest semantics**
    - **Validates: Requirements 16.1, 16.2, 16.3, 16.4**
  - [x] 9.10 Integrate real SOP analyzer into pipeline (replace stub)
    - Wire detections + pose into the analyzer; ensure all eight rule results appear in the Assessment; re-run smoke test
    - _Requirements: 15.1, 24.3_

- [x] 10. Verdict engine
  - [x] 10.1 Implement verdict aggregation with critical-failure dominance
    - `aggregate(pose_result, sop_checks, verdict_config)` → PASS/FAIL/MANUAL_INSPECTION; confirmed mandatory violation → FAIL; adequate evidence all mandatory pass → PASS; inadequate evidence + no confirmed violation → MANUAL_INSPECTION; never average away a confirmed critical failure; record reason codes + human-readable reasoning + documented pose-quality weighting
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8_
  - [ ]* 10.2 Write property test for verdict truth table
    - **Property 18: Verdict aggregation truth table with critical-failure dominance**
    - **Validates: Requirements 17.1, 17.2, 17.3, 17.4, 17.5**
  - [ ]* 10.3 Write unit tests for verdict logic
    - Hand-built PASS / FAIL / MANUAL / critical-failure-dominance cases
    - _Requirements: 17.2, 17.3, 17.4, 17.5_

- [x] 11. Temporal persistence (tracker)
  - [x] 11.1 Implement per-slot tracker with circular-statistics orientation aggregation
    - `update(track_state, detection, pose_result)`: per-slot identity, circular-mean orientation aggregation, movement/stale detection with state reset, effective-sample-size treatment of correlated frames; document effect on stability/throughput and that averaging does not remove systematic calibration bias
    - _Requirements: 23.1, 23.2_
  - [ ]* 11.2 Write property test for circular-statistics aggregation (tracker path)
    - Reuse/confirm **Property 22** across the tracker aggregation path (wrap-boundary circular mean, e.g. 179° and −179° ≈ 180°)
    - **Property 22: Orientation temporal aggregation uses correct circular statistics**
    - **Validates: Requirements 23.2**

- [x] 12. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. CLI, failure handling, and reproducibility
  - [x] 13.1 Implement CLI with exit codes and failure handling
    - Finalise CLI: distinguish "no pallet detected" (exit 0, valid empty result) from "processing failed" (non-zero exit); map every Error Handling condition to status + Reason_Code; corrupt input/runtime error emits failure signal without fabricating values
    - Wire fixed seed + fixed config for deterministic runs
    - _Requirements: 22.1, 22.2, 22.3, 28.3_
  - [ ]* 13.2 Write property test for Assessment schema validity/round-trip
    - **Property 20: Every Assessment is schema-valid, versioned, and round-trips**
    - **Validates: Requirements 17.8, 22.1, 22.3, 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.8, 25.1**
  - [ ]* 13.3 Write property test for deterministic reproducibility
    - **Property 23: Deterministic runs are reproducible**
    - **Validates: Requirements 28.3**
  - [ ]* 13.4 Write unit tests for missing/invalid evidence and JSON schema
    - Null keypoints, invalid calibration, corrupt image → correct status + reason; valid and deliberately invalid Assessments against the versioned schema
    - _Requirements: 22.1, 22.3, 24.8_

- [x] 14. Deployment & robustness — 10%
  - [x] 14.1 Implement benchmark script (measured hardware only) with Jetson estimated reasoning
    - `scripts/benchmark.py`: measure latency only on actual hardware, labelled `measured` with hardware named; report Jetson Orin Nano 15 W / 15 FPS strictly as `estimated`/unmeasured reasoning, never as measured on target
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 20.1, 20.2, 20.3_
  - [ ]* 14.2 Implement export/quantisation cost measurement (stretch)
    - `scripts/export.py`: INT8 export; measure task-relevant deltas (keypoint pixel precision, pose error, tolerance success rate, changed SOP verdicts) with INT8 calibration data kept separate from the test set; attach provenance
    - _Requirements: 21.1, 21.2, 21.3_

- [ ] 15. Deliverables and documentation
  - [ ]* 15.1 Write DATASET.md
    - Provenance of every source (name, access, citation), sourcing cost, licences + rejection reasons, class/annotation counts, one-page labelling guideline, known biases/gaps
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.1, 2.2, 2.3_
  - [ ]* 15.2 Write README with the five required sections
    - Approach + significant decisions with cost; results as distributions with sample counts; three worst failure cases with images, root-caused; what could not be finished and why; AI tools used and one thing an AI tool got wrong that was caught
    - _Requirements: 29.1, 29.2, 29.3, 29.4, 29.5, 28.1_
  - [ ]* 15.3 Write decision log and requirements checklist
    - `reports/decision_log.md` (each significant decision + cost + AI-tool note); `reports/requirements_checklist.md` (R1–R29 → where satisfied, honestly-marked unfinished items)
    - _Requirements: 4.3, 4.4, 28.1, 29.1, 29.4_
  - [ ]* 15.4 Generate illustrative PASS/FAIL/MANUAL_INSPECTION Assessment examples
    - At least one Assessment example per verdict type in `outputs/`, each labelled illustrative (never presented as measured)
    - _Requirements: 25.1, 25.2, 25.3_

- [x] 16. Final checkpoint — verification and failure analysis
  - Run the full test suite (unit + all 23 property tests, ≥100 iterations each) and the end-to-end pipeline; confirm all Assessments schema-valid; root-cause the three worst failure cases for the README; ensure all tests pass, ask the user if questions arise.
  - _Requirements: 28.1, 29.3_

## Notes

- Tasks marked with `*` are optional/stretch or documentation-/hardware-heavy and can be skipped in a time-boxed run-all-tasks flow; unmarked top-level and sub-tasks are core and must be implemented.
- The plan front-loads a runnable end-to-end pipeline (Task 3) then replaces stubs stage by stage (Tasks 4.8, 6.10, 9.10) so nothing references a not-yet-built component.
- All 23 correctness properties are covered by exactly one property-based test each (Hypothesis, ≥100 iterations, tagged `Feature: pallet-pose-compliance, Property {n}: {text}`); Property 22 is exercised both in the geometry module (2.4) and the tracker path (11.2).
- Checkpoints (Tasks 5, 8, 12, 16) ensure incremental validation at rubric boundaries.
- Rubric alignment: dataset+detection (Task 4, 30%), pose (Tasks 6–7, 35%), SOP+verdict (Tasks 9–10, 25%), deployment (Task 14, 10%).
