# Requirements Document

## Introduction

This document specifies requirements for a **Pallet Pose Estimation & Load Compliance** system built for the Delhivery take-home assignment. A single fixed camera, mounted approximately 1.2 m above a warehouse floor and tilted down approximately 20°, observes loaded pallets in numbered floor slots. From each image the system must (a) estimate the metric pose of each pallet (position in metres in a defined floor frame, orientation, and face identity) and (b) determine whether the load complies with warehouse stacking standard **SOP-PAL-03**.

The system MUST operate without fiducial markers; pose is derived from the pallet's own geometry. The evaluation tolerance bar for pose is **±2 cm position** and **±3° orientation** — these are targets the system MUST **evaluate itself against and report against**, NOT accuracy levels it is required to guarantee.

A foundational, non-negotiable principle governs the entire system and its deliverables: **honesty and provenance discipline**. Every reported number MUST be labelled as *measured*, *simulated*, *estimated*, or *unavailable*. The system and its documentation MUST NOT invent training runs, annotations, calibration artefacts, ground truth, metrics, or hardware benchmarks. Confidently wrong answers are treated as the worst possible outcome; the system MUST quantify and surface uncertainty instead of masking it.

This is a 5-day, scoped engineering effort. A completed, honestly-reported 70% solution is preferred over an abandoned or overclaimed 95% solution. The assignment is deliberately underspecified; the system's authors MUST make and document decisions.

## Glossary

- **System**: The complete pallet pose estimation and load compliance software, including detection, pose estimation, SOP analysis, and output generation components.
- **Detector**: The component that locates pallets (and load-relevant objects such as boxes) within an image.
- **Localiser**: The component that assigns image-space or metric position to detected pallets, evaluated separately from detection.
- **Pose_Estimator**: The component that produces metric pose (x, y in metres, orientation θ, face identity) for a detected pallet.
- **Calibrator**: The component and associated artefacts that establish camera intrinsics and extrinsics relative to the floor frame.
- **SOP_Analyzer**: The component that evaluates SOP-PAL-03 rules and produces per-rule verdicts and confidences.
- **Verdict_Engine**: The component that aggregates pose quality and per-rule SOP results into an overall per-pallet verdict.
- **Assessment**: The versioned JSON output object produced per pallet per image.
- **Floor_Frame**: The defined 2D metric coordinate frame on the warehouse floor in which pallet position (x, y) and orientation θ are expressed.
- **Pose_Tolerance**: The evaluation bar of ±2 cm position error and ±3° orientation error.
- **Face_Identity**: The identified physical face/orientation label of a pallet, since pallets are not rotationally symmetric in use.
- **Usable_Envelope**: The declared region of operating conditions (e.g. range, angle) within which the System meets Pose_Tolerance, and the region where it does not.
- **Held_Out_Set**: An evaluation dataset that differs from the training data in a declared, documented way.
- **Ground_Truth**: Reference values against which accuracy is measured; for pose these MUST be self-constructed with a justified method.
- **Provenance_Label**: One of {measured, simulated, estimated, unavailable} attached to every reported result.
- **Verdict**: One of {PASS, FAIL, MANUAL_INSPECTION} produced per pallet.
- **Reason_Code**: A machine-readable code explaining why a particular verdict, flag, or degraded output was produced.
- **Quality_Flag**: A machine-readable indicator of a degraded or unreliable condition (e.g. ill-conditioned pose, occlusion, low confidence).
- **Target_Hardware**: NVIDIA Jetson Orin Nano operating at 15 W, with a throughput goal of at least 15 FPS.
- **SOP-PAL-03**: The warehouse stacking standard containing eight rules governing load compliance.
- **Reprojection_Error**: The pixel-space error between observed calibration/reference points and their reprojection through the calibrated camera model.

## Requirements

---

## Section 1 — Dataset & Detection (30%)

### Requirement 1: Dataset Sourcing and Provenance

**User Story:** As an evaluator, I want a fully documented dataset with declared provenance, so that I can trust and reproduce the detection results.

#### Acceptance Criteria

1. THE System SHALL be accompanied by a DATASET.md document describing the dataset used for detection and localisation.
2. THE DATASET.md SHALL record the provenance of every data source, including source name, access method, and citation.
3. THE DATASET.md SHALL record the sourcing cost of the dataset in effort and, where applicable, monetary terms.
4. WHERE a candidate third-party dataset (for example a Roboflow dataset) is used, THE DATASET.md SHALL record the licence and confirm the licence permits the assignment's use.
5. WHERE a candidate third-party dataset is considered but not used, THE DATASET.md SHALL record the reason for rejection, including any failure to verify licence or pose annotations.
6. THE DATASET.md SHALL record class counts and annotation counts for the assembled dataset.
7. THE DATASET.md SHALL include a one-page labelling guideline describing how annotations were produced.
8. THE DATASET.md SHALL document known biases and gaps in the dataset.

### Requirement 2: Split Protocol

**User Story:** As an evaluator, I want a declared and defensible train/evaluation split, so that reported accuracy reflects generalisation rather than memorisation.

#### Acceptance Criteria

1. THE DATASET.md SHALL describe the split protocol used to partition data into training and evaluation subsets.
2. THE DATASET.md SHALL declare the attribute that the split is performed by (for example by scene, by pallet instance, by capture session, or by camera pose).
3. THE Held_Out_Set SHALL differ from the training data in a way that is explicitly declared in the DATASET.md.
4. WHEN accuracy is reported, THE System SHALL report it on the Held_Out_Set rather than on training data.

### Requirement 3: Annotation Tooling

**User Story:** As a developer, I want annotation tooling, so that I can produce and maintain labels consistently.

#### Acceptance Criteria

1. THE System SHALL include annotation tooling used to create or adapt the dataset labels.
2. THE annotation tooling SHALL produce labels in a documented, machine-readable format.
3. THE labelling guideline in DATASET.md SHALL correspond to the behaviour of the annotation tooling.

### Requirement 4: Trained Detection Model and Weights

**User Story:** As an evaluator, I want the trained model and its weights, so that I can reproduce and inspect the detection component.

#### Acceptance Criteria

1. THE System SHALL include a trained Detector model together with its trained weights.
2. WHERE an open-source model architecture is used, THE System SHALL cite the architecture and its source.
3. THE System SHALL document the training decisions made, including hyperparameters and data choices, with reasoning for each significant decision.
4. THE System SHALL document the cost of each significant training decision.

### Requirement 5: Separate Detection and Localisation Accuracy Reporting

**User Story:** As an evaluator, I want detection accuracy and localisation accuracy reported separately as distributions, so that I can distinguish "found the pallet" from "placed it correctly".

#### Acceptance Criteria

1. THE System SHALL report Detector accuracy separately from Localiser accuracy.
2. THE System SHALL report detection accuracy as a distribution over the Held_Out_Set, not as a single point estimate.
3. THE System SHALL report localisation accuracy as a distribution over the Held_Out_Set, not as a single point estimate.
4. WHEN accuracy distributions are reported, THE System SHALL state the sample count used to compute each distribution.
5. WHEN accuracy is reported, THE System SHALL attach a Provenance_Label of "measured" only where the numbers are computed from actual evaluation runs.

### Requirement 6: Accuracy Ceiling Estimate

**User Story:** As an evaluator, I want an estimate of the accuracy ceiling given the available data, so that I understand the headroom and how to raise it.

#### Acceptance Criteria

1. THE System SHALL provide an estimate of the accuracy ceiling given the current data volume.
2. WHEN the accuracy ceiling is reported, THE System SHALL attach a Provenance_Label of "estimated".
3. THE System SHALL describe what changes would raise the accuracy ceiling.

---

## Section 2 — Pose Estimation (35%)

### Requirement 7: Metric Pose Output

**User Story:** As a warehouse operations consumer, I want each pallet's metric pose, so that I know where and how it is positioned in the floor frame.

#### Acceptance Criteria

1. THE Pose_Estimator SHALL produce, for each detected pallet, a position (x, y) expressed in metres in the Floor_Frame.
2. THE Pose_Estimator SHALL produce, for each detected pallet, an orientation θ expressed in a documented angular unit and reference direction.
3. THE Pose_Estimator SHALL produce, for each detected pallet, a Face_Identity.
4. THE System SHALL document the definition of the Floor_Frame, including its origin and axis directions.

### Requirement 8: Camera Calibration Artefacts

**User Story:** As an evaluator, I want camera calibration artefacts with reprojection error, so that I can trust the metric geometry.

#### Acceptance Criteria

1. THE System SHALL include camera calibration artefacts covering intrinsics and extrinsics relative to the Floor_Frame.
2. THE System SHALL report the Reprojection_Error of the calibration.
3. WHEN the Reprojection_Error is reported, THE System SHALL attach the appropriate Provenance_Label.
4. THE System SHALL assign an identifier to each calibration artefact set so that Assessments can reference the calibration used.

### Requirement 9: Pose Method and Geometric Soundness

**User Story:** As an evaluator, I want the pose method to respect the scene geometry, so that metric estimates are not silently invalid.

#### Acceptance Criteria

1. THE System SHALL document the pose estimation method and the reasoning behind its selection.
2. THE Pose_Estimator SHALL NOT apply a floor-plane homography to corners that are elevated above the floor plane.
3. THE Pose_Estimator SHALL NOT infer metric position from a bounding box combined with an approximate tilt angle.
4. THE Pose_Estimator SHALL account for keypoint height above the floor, lens distortion, and keypoint visibility when computing pose.
5. IF a pose solution is ill-conditioned, THEN THE Pose_Estimator SHALL reject the solution and emit a Quality_Flag with a Reason_Code.
6. WHERE more than one pose solution is geometrically admissible (ambiguity), THE Pose_Estimator SHALL represent the ambiguous solution set rather than silently selecting one.
7. WHERE pallet geometry is symmetric such that orientation is only determinable modulo a symmetry, THE Pose_Estimator SHALL report orientation modulo that symmetry.

### Requirement 10: Self-Constructed Pose Evaluation and Ground Truth

**User Story:** As an evaluator, I want a self-constructed pose evaluation with justified ground truth, so that error claims are credible in the absence of provided pose ground truth.

#### Acceptance Criteria

1. THE System SHALL include a self-constructed evaluation for pose estimation.
2. THE System SHALL construct pose Ground_Truth and justify the method used to obtain it.
3. WHEN pose Ground_Truth is used, THE System SHALL attach a Provenance_Label distinguishing measured, simulated, or estimated ground truth.
4. THE System SHALL NOT present pose Ground_Truth as provided by the assignment.

### Requirement 11: Pose Error Distributions and Tolerance Evaluation

**User Story:** As an evaluator, I want translation and rotation error distributions evaluated against the tolerance bar, so that I understand where the system meets ±2 cm / ±3°.

#### Acceptance Criteria

1. THE System SHALL report translation error as a distribution, separately from rotation error.
2. THE System SHALL report rotation error as a distribution, separately from translation error.
3. WHEN error distributions are reported, THE System SHALL state the sample count for each distribution.
4. THE System SHALL evaluate its measured error distributions against the Pose_Tolerance of ±2 cm position and ±3° orientation and report the outcome as an evaluation result.
5. THE System SHALL treat the Pose_Tolerance as a target to evaluate against and SHALL NOT present the tolerance as a guaranteed accuracy level.

### Requirement 12: Sensitivity and Usable Envelope

**User Story:** As an evaluator, I want sensitivity analysis and a declared usable envelope, so that I know when to trust the pose and when not to.

#### Acceptance Criteria

1. THE System SHALL report the sensitivity of pose error to camera height error, at both short range and long range.
2. THE System SHALL report the sensitivity of pose error to camera tilt error, at both short range and long range.
3. THE System SHALL declare the Usable_Envelope in which measured error meets the Pose_Tolerance.
4. THE System SHALL declare the region of operating conditions in which measured error does not meet the Pose_Tolerance.

### Requirement 13: Pose Uncertainty and Unreliable-Pose Behaviour

**User Story:** As a downstream consumer, I want quantified pose uncertainty and a clear signal when pose is unreliable, so that I do not act on a confidently wrong pose.

#### Acceptance Criteria

1. THE Pose_Estimator SHALL attach a quantified uncertainty to each metric pose output.
2. THE System SHALL document the uncertainty quantification method and its justification.
3. WHERE Monte Carlo propagation or an equivalent method is used, THE System SHALL document the assumptions of that method.
4. IF the Pose_Estimator cannot produce a reliable pose, THEN THE System SHALL emit a defined output that identifies the pose as unavailable, with a Reason_Code, rather than emitting a fabricated pose.

---

## Section 3 — Load Analysis & SOP-PAL-03 (25%)

SOP-PAL-03 comprises eight rules:
1. No box overhang greater than 3 cm.
2. Load height no greater than 1.8 m.
3. Aligned columns: no box rotated more than 15° relative to the pallet axes.
4. Larger boxes below smaller boxes: no size inversion.
5. Load is stretch-wrapped.
6. No visibly damaged or crushed box.
7. Load centroid within 10 cm of the pallet centre.
8. Pallet undamaged: no broken boards or split stringers.

The camera observes each pallet from one side only.

### Requirement 14: SOP Rule Triage

**User Story:** As an evaluator, I want a triage of all eight SOP rules, so that I understand which are verifiable from a single-side view and which are not.

#### Acceptance Criteria

1. THE System SHALL classify each of the eight SOP-PAL-03 rules as verifiable, partially verifiable, or not verifiable from the available single-side camera view.
2. WHEN a rule is triaged, THE System SHALL record the justification for the classification.
3. WHEN a rule is triaged, THE System SHALL record the assumptions made and the hidden regions that affect verifiability.
4. THE System SHALL retain all eight rules in the triage, and SHALL NOT omit any rule from the triage.

### Requirement 15: Implementation of the Verifiable Subset

**User Story:** As a warehouse operator, I want the verifiable SOP checks implemented, so that I get automated compliance results where they are trustworthy.

#### Acceptance Criteria

1. THE SOP_Analyzer SHALL implement the subset of SOP-PAL-03 rules classified as verifiable or partially verifiable.
2. THE System SHALL document the reasoning for each implemented check.
3. WHERE a rule is classified as not verifiable, THE System SHALL keep the rule as unresolved in the output, and SHALL NOT report it as passed or omitted.
4. IF pose is unreliable for a given pallet, THEN THE SOP_Analyzer SHALL mark pose-dependent checks for that pallet as invalidated with a Reason_Code.

### Requirement 16: Per-Check Confidence Semantics

**User Story:** As an evaluator, I want per-check confidence with clear derivation semantics, so that a detector score is not mistaken for a calibrated pass probability.

#### Acceptance Criteria

1. THE SOP_Analyzer SHALL attach a confidence to each implemented per-rule check.
2. THE System SHALL document the derivation semantics of each per-check confidence.
3. THE System SHALL distinguish a raw detector score from a calibrated pass probability in the confidence semantics documentation.
4. WHERE a confidence is derived from a raw detector score without calibration, THE System SHALL label the confidence semantics accordingly.

### Requirement 17: Per-Pallet Verdict Aggregation

**User Story:** As a warehouse operator, I want a single defensible verdict per pallet, so that I know whether to pass it, fail it, or send it for manual inspection.

#### Acceptance Criteria

1. THE Verdict_Engine SHALL produce a Verdict of PASS, FAIL, or MANUAL_INSPECTION for each pallet.
2. WHEN there is a confirmed violation of a mandatory rule, THE Verdict_Engine SHALL produce a Verdict of FAIL.
3. WHEN there is adequate evidence that all mandatory rules are satisfied, THE Verdict_Engine SHALL produce a Verdict of PASS.
4. IF evidence is inadequate for one or more mandatory rules and no confirmed violation exists, THEN THE Verdict_Engine SHALL produce a Verdict of MANUAL_INSPECTION.
5. THE Verdict_Engine SHALL NOT average per-check results in a way that offsets a confirmed critical failure.
6. THE System SHALL document the thresholds used in the Verdict, with justification for each threshold.
7. THE System SHALL document the reasoning for weighting pose quality against load compliance in the Verdict.
8. WHEN a Verdict is produced, THE Verdict_Engine SHALL record a Reason_Code and human-readable reasoning for the Verdict.

### Requirement 18: Configurable SOP Thresholds

**User Story:** As an evaluator conducting a live follow-up, I want SOP thresholds to be configurable, so that a rule can be changed without editing code.

#### Acceptance Criteria

1. THE System SHALL expose SOP-PAL-03 thresholds as external configuration.
2. WHEN a threshold value is changed in configuration, THE System SHALL apply the changed value without requiring source code modification.
3. THE System SHALL document the configuration format and the location of each configurable threshold.

---

## Section 4 — Deployment & Robustness (10%)

### Requirement 19: Latency Analysis with Honest Hardware Provenance

**User Story:** As an evaluator, I want latency analysis that is honest about the hardware it was measured on, so that no benchmark is fabricated for hardware not owned.

#### Acceptance Criteria

1. THE System SHALL report latency analysis and SHALL clearly state the hardware on which each latency figure was actually measured.
2. WHEN a latency figure is reported, THE System SHALL attach a Provenance_Label of "measured".
3. IF the Target_Hardware is not available for measurement, THEN THE System SHALL state that the Target_Hardware is unavailable and provide reasoning-based expectations labelled "estimated".
4. THE System SHALL NOT report a benchmark figure as measured on the Target_Hardware unless it was actually measured on the Target_Hardware.

### Requirement 20: Target Hardware Expectations

**User Story:** As an evaluator, I want expected performance on the Jetson Orin Nano with reasoning, so that I understand feasibility against the 15 W / 15 FPS goal.

#### Acceptance Criteria

1. THE System SHALL describe the expected changes in performance when running on the Target_Hardware at 15 W.
2. THE System SHALL provide reasoning for each expected performance change.
3. THE System SHALL evaluate the expected performance against the throughput goal of at least 15 FPS and report the outcome as an estimate.

### Requirement 21: Quantisation Accuracy Cost

**User Story:** As an evaluator, I want the accuracy cost of quantisation measured on what matters most, so that export tradeoffs are quantified.

#### Acceptance Criteria

1. WHERE the model is exported or quantised, THE System SHALL report the accuracy cost of quantisation.
2. WHEN a quantisation accuracy cost is reported, THE System SHALL measure it on the metric that matters most to the task and state which metric was used.
3. WHEN a quantisation accuracy cost is reported, THE System SHALL attach the appropriate Provenance_Label.

### Requirement 22: Failure Behaviour and Downstream Signalling

**User Story:** As a downstream consumer, I want to know when the system has failed, so that I do not consume unreliable results.

#### Acceptance Criteria

1. WHEN the System cannot produce a reliable result for a pallet, THE System SHALL emit a defined failure signal in the Assessment.
2. THE System SHALL document how a downstream consumer detects that a result has failed or is degraded.
3. WHEN a failure signal is emitted, THE System SHALL include a Reason_Code.

### Requirement 23: Temporal Persistence

**User Story:** As an evaluator, I want the system to exploit pallets remaining in the same slot across frames, so that stability and efficiency can be improved.

#### Acceptance Criteria

1. THE System SHALL describe how a pallet remaining in the same location across many frames can be exploited.
2. WHERE temporal persistence is implemented, THE System SHALL document its behaviour and its effect on pose stability or throughput.

---

## Cross-Cutting Requirements

### Requirement 24: Auditable Versioned Output Schema

**User Story:** As an evaluator, I want one auditable, versioned Assessment per pallet, so that every result is traceable and machine-consumable.

#### Acceptance Criteria

1. THE System SHALL produce one Assessment per pallet per image in a versioned JSON schema.
2. THE Assessment SHALL include the pose with its uncertainty, the orientation, and the Face_Identity.
3. THE Assessment SHALL include per-rule SOP verdicts with per-check confidence.
4. THE Assessment SHALL include the overall Verdict together with its reasoning.
5. THE Assessment SHALL include a schema version identifier.
6. THE Assessment SHALL include a pallet identifier, a timestamp, and a reference to the source image.
7. THE Assessment SHALL include the model identifier and the calibration identifier used to produce the result.
8. THE Assessment SHALL include Quality_Flags, Reason_Codes, and per-stage timings.

### Requirement 25: Illustrative Verdict Examples

**User Story:** As an evaluator, I want clearly labelled examples of each verdict type, so that I can see the schema exercised across outcomes.

#### Acceptance Criteria

1. THE System SHALL provide at least one Assessment example for each of PASS, FAIL, and MANUAL_INSPECTION.
2. WHEN an example Assessment is provided, THE System SHALL label it as illustrative or as measured.
3. THE System SHALL NOT present an illustrative example as a measured result.

### Requirement 26: Honesty and Provenance Discipline

**User Story:** As an evaluator, I want every reported result labelled by provenance, so that measured, simulated, estimated, and unavailable results are never confused.

#### Acceptance Criteria

1. WHEN the System reports any result, THE System SHALL attach a Provenance_Label of measured, simulated, estimated, or unavailable.
2. THE System SHALL NOT present a simulated or synthetic result as a real-world accuracy measurement.
3. THE System SHALL NOT invent training runs, annotations, calibration artefacts, ground truth, metrics, or hardware benchmarks.
4. WHERE data or hardware is unavailable, THE System SHALL state the unavailability and provide reasoning rather than a fabricated result.
5. WHERE an open-source model, library, or dataset is used, THE System SHALL include a citation.

### Requirement 27: Uncertainty Quantification Discipline

**User Story:** As a downstream consumer, I want uncertainty surfaced wherever results are used, so that confidently wrong outputs are avoided.

#### Acceptance Criteria

1. THE System SHALL attach quantified uncertainty or confidence to pose outputs and to per-rule SOP checks.
2. WHEN uncertainty exceeds a documented threshold for a result, THE System SHALL degrade the affected output to unavailable or MANUAL_INSPECTION with a Reason_Code.
3. THE System SHALL document the method used to quantify uncertainty for each result type.

### Requirement 28: Reproducibility

**User Story:** As an evaluator, I want to reproduce the reported results, so that the claims are verifiable.

#### Acceptance Criteria

1. THE System SHALL document the steps required to reproduce training, calibration, and evaluation.
2. THE System SHALL reference the model identifier and calibration identifier associated with each reported result.
3. THE System SHALL declare fixed configuration and, where applicable, random seeds used to produce reported results.

### Requirement 29: README and Deliverables

**User Story:** As an evaluator, I want a README with the required sections, so that I can assess approach, results, failures, scope, and AI tool usage.

#### Acceptance Criteria

1. THE README SHALL describe the approach and its significant decisions, and SHALL state what each significant decision cost.
2. THE README SHALL report results as distributions with sample counts, and SHALL NOT report results as point estimates alone.
3. THE README SHALL present the three worst failure cases with images, each root-caused.
4. THE README SHALL state what could not be finished and why.
5. THE README SHALL state which AI tools were used, for what, and one thing an AI tool got wrong that was caught.
