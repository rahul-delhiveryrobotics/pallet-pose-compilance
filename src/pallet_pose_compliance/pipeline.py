"""End-to-end pipeline wiring (task 3.1; detector integrated in task 4.8).

Runs the full single-image pipeline (design: Architecture, Pipeline stages):

    ingest -> detect -> pose (known-geometry PnP + MC uncertainty)
           -> SOP (stub, triage-only) -> verdict (stub) -> Assessment output

As of task 4.8 the **detect** stage uses the real
:class:`~pallet_pose_compliance.detection.detector.YoloPoseDetector` by default,
constructed from ``pipeline.yaml``: it prefers an *assignment-trained* weights
artefact at ``weights/<model_id>.pt`` and otherwise falls back to a configurable
*stock* checkpoint (``stock_model_id``), which is honestly flagged
not-assignment-trained (R4.1, R5.5, R26.3).

**Fallback / honesty discipline** (design principle 5, R26.3/R26.4): if the real
detector is unavailable — ultralytics not installed or weights missing, raising
:class:`DetectorUnavailableError` — the pipeline **surfaces the blocker** by
default rather than silently substituting fake detections. A development-only
:class:`StubDetector` fallback is available **only** when explicitly opted in
(``allow_stub_fallback=True``); when used, its detections are clearly labelled
not-real via the returned :class:`DetectorSelection` so stub output is never
mistaken for a measured/real detection. Injecting ``detector=`` always overrides
construction (used by the smoke test).

One :class:`~pallet_pose_compliance.output.schema.Assessment` is produced per
detected pallet (R24.1) and written to the configured ``outputs/`` directory as
JSON. Per-stage wall-clock timings are recorded (R24.8).

As of task 6.10 the **pose** stage uses the real known-geometry PnP estimator
(:func:`~pallet_pose_compliance.geometry.pose.estimate_pose_with_uncertainty`):
the configured calibration is loaded once per run and the nominal pallet model
built once, then each detection's keypoints are solved to a metric Floor_Frame
pose carrying Monte Carlo uncertainty. The pose degrades loudly to a defined
``unavailable`` result when the solve is untrustworthy, and to
``CALIBRATION_INVALID`` when the calibration artefact cannot be loaded — never a
fabricated pose. Every Assessment records ``model_id`` and ``calibration_id``
(R8.4, R24.7, R28.2).

As of task 9.10 the **SOP** stage uses the real verifiable-subset analyzer
(:func:`~pallet_pose_compliance.sop.checks.analyze`) instead of the triage-only
stub: the SOP thresholds config (:func:`~pallet_pose_compliance.sop.config.load_sop_config`)
is loaded once per run from the configured ``sop_thresholds_path`` and the
detections + metric pose are wired into the analyzer per pallet. Every
Assessment carries the complete eight-rule SOP set (rule_id 1..8, R24.3) with
honest status/confidence discipline: not_verifiable rules (4/6/8) stay
``unresolved`` (R15.3), pose-dependent rules (1/3/7) are ``invalidated`` with a
Reason_Code when the pose is unavailable (R15.4), and every implemented check
carries an honest confidence/semantics (R16).

As of task 10.1 the **verdict** stage uses the real aggregation engine
(:func:`~pallet_pose_compliance.verdict.engine.aggregate`) instead of the stub:
the verdict config (:func:`~pallet_pose_compliance.verdict.config.load_verdict_config`)
is loaded once per run from the configured ``verdict_config_path`` and the pose
+ SOP checks are aggregated per pallet into a ``PASS`` / ``FAIL`` /
``MANUAL_INSPECTION`` verdict with critical-failure dominance (a confirmed
mandatory violation dominates and is never averaged away, R17.5) and a
pose-quality gate (R17.7). With the pallet-only stub detector the mandatory
checks resolve to ``unresolved``/``invalidated``, so the honest verdict is
``MANUAL_INSPECTION`` (inadequate evidence, no confirmed violation, R17.4).

Per-pallet box association is a documented limitation: the current pipeline
passes the frame's full detection set to the analyzer for each pallet (the stub
detector emits pallet-only detections, so verifiable box-driven checks resolve
honestly to ``unresolved``/``invalidated`` rather than being fabricated). A real
per-pallet box grouping arrives with the trained detector/tracker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import yaml

from .calibration.calibration import (
    Calibration,
    CalibrationError,
    load_calibration,
)
from .detection.detector import DetectorUnavailableError, YoloPoseDetector
from .detection.stub import StubDetector
from .geometry.pallet_model import build_nominal_pallet_model
from .geometry.pose import estimate_pose_with_uncertainty
from .ingest import Frame, IngestError, ingest_image, make_synthetic_frame
from .output.provenance import ProvenanceLabel, QualityFlag, ReasonCode
from .output.schema import Assessment, FailureSignal, PoseResult, StageTimings
from .sop.checks import analyze
from .sop.config import DEFAULT_SOP_CONFIG_PATH, SopConfig, load_sop_config
from .verdict.config import (
    DEFAULT_VERDICT_CONFIG_PATH,
    VerdictConfig,
    load_verdict_config,
)
from .verdict.engine import aggregate

__all__ = [
    "PipelineConfig",
    "Detector",
    "DetectorSelection",
    "load_pipeline_config",
    "build_detector",
    "run_pipeline",
    "process_image",
]


@runtime_checkable
class Detector(Protocol):
    """Structural type for any detector usable by the pipeline.

    Both :class:`~pallet_pose_compliance.detection.detector.YoloPoseDetector`
    and the development :class:`~pallet_pose_compliance.detection.stub.StubDetector`
    satisfy this protocol.
    """

    def detect(self, image: Any) -> list:  # pragma: no cover - structural
        ...

# Default location of the pipeline config relative to the repository root.
_DEFAULT_CONFIG_PATH = Path("configs/pipeline.yaml")


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved pipeline configuration (subset needed by the pipeline)."""

    seed: int
    model_id: str
    calibration_id: str
    input_resolution: int
    schema_version: str
    outputs_dir: Path
    weights_dir: Path
    calibration_dir: Path
    sop_thresholds_path: Path
    verdict_config_path: Path
    stock_model_id: str
    detector_conf: float
    allow_stub_fallback: bool

    @property
    def weights_path(self) -> Path:
        """Path to the assignment-trained weights artefact (``weights/<model_id>.pt``)."""
        return self.weights_dir / f"{self.model_id}.pt"

    @classmethod
    def from_dict(cls, data: dict, *, base_dir: Path) -> "PipelineConfig":
        paths = data.get("paths", {}) or {}
        outputs_dir = base_dir / paths.get("outputs_dir", "outputs")
        weights_dir = base_dir / paths.get("weights_dir", "weights")
        calibration_dir = base_dir / paths.get(
            "calibration_dir", "configs/calibration"
        )
        # SOP thresholds config read by the real analyzer (task 9.10). Resolved
        # relative to the repo root, defaulting to the standard location so a
        # config without an explicit entry still works (R18.3).
        sop_thresholds_path = base_dir / paths.get(
            "sop_thresholds_path", DEFAULT_SOP_CONFIG_PATH
        )
        # Verdict engine config read by the real aggregator (task 10.1).
        # Resolved relative to the repo root, defaulting to the standard
        # location so a config without an explicit entry still works (R17.6).
        verdict_config_path = base_dir / paths.get(
            "verdict_config_path", DEFAULT_VERDICT_CONFIG_PATH
        )
        detector_cfg = data.get("detector", {}) or {}
        return cls(
            seed=int(data.get("seed", 0)),
            model_id=str(data["model_id"]),
            calibration_id=str(data["calibration_id"]),
            input_resolution=int(data.get("input_resolution", 640)),
            schema_version=str(data.get("schema_version", "1.0.0")),
            outputs_dir=outputs_dir,
            weights_dir=weights_dir,
            calibration_dir=calibration_dir,
            sop_thresholds_path=sop_thresholds_path,
            verdict_config_path=verdict_config_path,
            # Stock/pretrained checkpoint used only when no assignment-trained
            # weights exist. Any inference from it is 'estimated' and flagged
            # not-assignment-trained (R5.5, R26.3).
            stock_model_id=str(detector_cfg.get("stock_model_id", "yolo11n-pose.pt")),
            detector_conf=float(detector_cfg.get("conf", 0.25)),
            # Honesty discipline: the dev-only stub fallback is opt-in and
            # defaults OFF, so a missing detector surfaces as an explicit
            # blocker rather than fabricated detections (R26.3/R26.4).
            allow_stub_fallback=bool(detector_cfg.get("allow_stub_fallback", False)),
        )


@dataclass(frozen=True)
class DetectorSelection:
    """The detector chosen for a run plus its honesty/provenance record.

    Lets callers (and the Assessment provenance discipline) know whether
    detections came from an assignment-trained model, a stock checkpoint, or the
    development stub — so stub/stock output is never mistaken for a measured
    real detection (R5.5, R26.3).

    Attributes
    ----------
    detector:
        The constructed detector exposing ``detect(image)``.
    kind:
        One of ``"assignment_trained"``, ``"stock"``, ``"stub"`` or
        ``"injected"`` (when a caller supplied its own detector).
    provenance:
        The :class:`ProvenanceLabel` any detector-derived result inherits:
        ``MEASURED``-eligible only for an assignment-trained model; ``ESTIMATED``
        for a stock checkpoint; ``UNAVAILABLE`` for the not-real stub fallback.
    is_real:
        ``True`` when detections come from an actual detection model (trained or
        stock); ``False`` for the stub, whose detections are placeholders.
    note:
        Short human-readable honesty note describing the detector origin.
    """

    detector: Detector
    kind: str
    provenance: ProvenanceLabel
    is_real: bool
    note: str


def load_pipeline_config(
    config_path: "str | Path" = _DEFAULT_CONFIG_PATH,
) -> PipelineConfig:
    """Load the pipeline configuration from ``config_path`` (YAML).

    The ``outputs_dir`` is resolved relative to the config file's parent's
    parent (the repository root), matching the repo layout where configs live
    under ``configs/``.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # configs/pipeline.yaml -> repo root is the config file's parent's parent.
    base_dir = path.resolve().parent.parent
    return PipelineConfig.from_dict(data, base_dir=base_dir)


def build_detector(
    config: PipelineConfig,
    *,
    detector: Optional[Detector] = None,
) -> DetectorSelection:
    """Select and construct the detector for a pipeline run (task 4.8).

    Selection order:

    1. **Injected** — if ``detector`` is provided it is used verbatim
       (overrides construction; this is what the smoke test uses).
    2. **Real detector** — otherwise a
       :class:`~pallet_pose_compliance.detection.detector.YoloPoseDetector` is
       constructed from ``config``: it prefers an assignment-trained weights
       artefact at ``config.weights_path`` (``weights/<model_id>.pt``) and
       otherwise uses the configured stock checkpoint. The model is loaded
       eagerly here so an unavailable detector surfaces *now* as a blocker
       rather than mid-run.
    3. **Fallback** — if the real detector is unavailable
       (:class:`DetectorUnavailableError`: ultralytics/weights missing), the
       behaviour depends on honesty discipline:

       - ``config.allow_stub_fallback`` **False** (default): the blocker is
         re-raised so the CLI can report it explicitly. Fake detections are
         never fabricated (R26.3/R26.4).
       - ``config.allow_stub_fallback`` **True** (dev opt-in): a
         :class:`StubDetector` is used and clearly labelled not-real
         (``provenance=UNAVAILABLE``, ``is_real=False``) so its placeholder
         detections are never mistaken for measured output.

    Raises
    ------
    DetectorUnavailableError
        When the real detector is unavailable and stub fallback is not allowed.
    """
    if detector is not None:
        prov = getattr(detector, "provenance", None)
        if not isinstance(prov, ProvenanceLabel):
            # An injected detector without a provenance record is treated
            # conservatively as not-real/unavailable-provenance so it is never
            # mistaken for a measured detection.
            prov = ProvenanceLabel.UNAVAILABLE
        is_real = isinstance(detector, YoloPoseDetector)
        return DetectorSelection(
            detector=detector,
            kind="injected",
            provenance=prov,
            is_real=is_real,
            note="Caller-injected detector overriding construction.",
        )

    weights_path = config.weights_path
    use_trained = weights_path.exists()
    real = YoloPoseDetector(
        weights_path=weights_path if use_trained else None,
        stock_model_id=None if use_trained else config.stock_model_id,
        imgsz=config.input_resolution,
        conf=config.detector_conf,
    )
    try:
        real.load()
    except DetectorUnavailableError as exc:
        if not config.allow_stub_fallback:
            # Surface the blocker honestly (design principle 5, R26.3/R26.4).
            raise
        # Dev-only opt-in fallback: clearly labelled not-real so stub
        # detections are never presented as measured/real output.
        return DetectorSelection(
            detector=StubDetector(),
            kind="stub",
            provenance=ProvenanceLabel.UNAVAILABLE,
            is_real=False,
            note=(
                "DEV STUB FALLBACK: the real detector is unavailable "
                f"({exc}); using StubDetector. Detections are PLACEHOLDERS, "
                "NOT real/measured — enabled only because "
                "detector.allow_stub_fallback is set."
            ),
        )

    return DetectorSelection(
        detector=real,
        kind="assignment_trained" if use_trained else "stock",
        provenance=real.provenance,
        is_real=True,
        note=real.weights_provenance.note,
    )


def _elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds since ``start`` (a ``time.perf_counter`` value)."""
    return (time.perf_counter() - start) * 1000.0


#: Monte Carlo sample count used for the *pipeline* pose-uncertainty pass. A
#: modest value keeps the per-pose solve fast enough for the end-to-end pipeline
#: while still stabilising the propagated 2x2 covariance + orientation std; the
#: pose module's own default (``MC_DEFAULT_SAMPLES`` = 128) is used elsewhere.
_PIPELINE_MC_SAMPLES = 64


def _calibration_invalid_pose() -> PoseResult:
    """Return a *defined* unavailable pose for an unloadable calibration (R26.4).

    When the configured calibration artefact cannot be loaded (missing or
    malformed), the pose stage degrades loudly rather than fabricating a pose or
    crashing the run: it emits ``pose_status = "unavailable"`` with
    ``CALIBRATION_INVALID`` and the matching quality flag, with null metric
    fields and ``UNAVAILABLE`` provenance (design principle 5, R26.3/R26.4).
    """
    return PoseResult(
        pose_status="unavailable",
        reason_code=ReasonCode.CALIBRATION_INVALID,
        position_m=None,
        orientation_deg=None,
        face_identity=None,
        quality_flags=[QualityFlag.CALIBRATION_INVALID],
        provenance=ProvenanceLabel.UNAVAILABLE,
    )


def process_image(
    frame: Frame,
    config: PipelineConfig,
    *,
    detector: Optional[Detector] = None,
) -> list[Assessment]:
    """Process a single ingested ``frame`` end to end, returning Assessments.

    Runs detect -> pose (real PnP + MC uncertainty) -> SOP (real verifiable
    subset) -> verdict (real aggregation) and builds one schema-valid Assessment
    per detected pallet. Per-stage timings are recorded. The calibration is
    loaded once per run and the nominal pallet model built once; the SOP
    thresholds and verdict configs are also loaded once per run and shared
    across every pallet's checks/verdict.
    An unloadable calibration degrades the pose to a defined
    ``CALIBRATION_INVALID`` unavailable result (and raises the Assessment
    failure signal) rather than crashing or fabricating a pose.

    The detect stage uses the real detector by default (constructed from
    ``config`` via :func:`build_detector`); passing ``detector=`` overrides
    construction. If the real detector is unavailable and stub fallback is not
    permitted, :class:`DetectorUnavailableError` propagates.

    Parameters
    ----------
    frame:
        The ingested frame (from :func:`~pallet_pose_compliance.ingest`).
    config:
        Resolved pipeline configuration (supplies model/calibration ids and
        detector construction settings).
    detector:
        Optional detector override (e.g. an injected stub in tests, or a
        pre-constructed real detector).

    Returns
    -------
    list[Assessment]
        One Assessment per detected pallet (possibly empty if no detections).

    Raises
    ------
    DetectorUnavailableError
        When the real detector is unavailable and stub fallback is disabled.
    """
    selection = build_detector(config, detector=detector)

    # Load the calibration and build the pallet model ONCE per run (not per
    # detection) for efficiency. If the calibration artefact cannot be loaded we
    # do NOT crash the run and do NOT fabricate a calibration: instead the pose
    # stage degrades loudly to a defined unavailable pose (CALIBRATION_INVALID)
    # for every pallet (design principle 5, R26.3/R26.4).
    calibration: Optional[Calibration] = None
    calibration_error: Optional[str] = None
    try:
        calibration = load_calibration(
            config.calibration_id, calibration_dir=config.calibration_dir
        )
    except CalibrationError as exc:
        calibration_error = str(exc)

    pallet_model = build_nominal_pallet_model()

    # Load the SOP thresholds config ONCE per run (not per pallet). This is the
    # single source of every ``threshold_used``/``threshold_source`` in the SOP
    # checks; editing configs/sop_thresholds.yaml changes the boundaries with no
    # code change (R18.2). A malformed/missing config fails loudly at load time.
    sop_config: SopConfig = load_sop_config(config.sop_thresholds_path)

    # Load the verdict config ONCE per run (not per pallet). This is the single
    # source of every verdict threshold (mandatory rule ids, confirmed-violation
    # / adequate-pass confidences, pose-quality gate); editing configs/verdict.yaml
    # changes the boundaries with no code change (R17.6/R17.7). A malformed or
    # missing config fails loudly at load time.
    verdict_config: VerdictConfig = load_verdict_config(config.verdict_config_path)

    # --- detect -----------------------------------------------------------
    t_detect = time.perf_counter()
    detections = selection.detector.detect(frame.image)
    detect_ms = _elapsed_ms(t_detect)

    assessments: list[Assessment] = []
    for index, detection in enumerate(detections):
        # --- pose (real known-geometry PnP + MC uncertainty) -------------
        # Wire the detection's keypoints -> pose estimator (task 6.10). The
        # uncertainty-aware entry point attaches Monte Carlo uncertainty to an
        # available pose and degrades loudly (defined unavailable) when the
        # solve is untrustworthy or too uncertain. When the calibration could
        # not be loaded we emit a defined CALIBRATION_INVALID unavailable pose
        # rather than fabricating one.
        t_pose = time.perf_counter()
        if calibration is None:
            pose = _calibration_invalid_pose()
        else:
            pose = estimate_pose_with_uncertainty(
                detection.keypoints,
                calibration,
                pallet_model,
                n_samples=_PIPELINE_MC_SAMPLES,
                seed=config.seed,
            )
        pose_ms = _elapsed_ms(t_pose)

        # --- SOP (real verifiable-subset analyzer, task 9.10) ------------
        # Wire the detections + metric pose into the real analyzer. It returns
        # all eight SOP-PAL-03 rules (R24.3): the implemented verifiable subset
        # (1/2/3/5/7) measured against the config thresholds with honest
        # confidence semantics (R16); not_verifiable rules (4/6/8) always
        # unresolved (R15.3); pose-dependent rules (1/3/7) invalidated with a
        # Reason_Code when the pose is unavailable (R15.4).
        #
        # Per-pallet box association is not yet available (documented
        # limitation): the full frame detection set is passed so the analyzer
        # can find the pallet detection. With the pallet-only stub detector no
        # box detections are present, so box-driven verifiable checks resolve
        # honestly to unresolved/invalidated rather than being fabricated. A
        # real per-pallet box grouping arrives with the trained detector/tracker.
        t_sop = time.perf_counter()
        sop_checks = analyze(detections, pose, sop_config)
        sop_ms = _elapsed_ms(t_sop)

        # --- verdict (real aggregation, task 10.1) -----------------------
        # Aggregate the pose + SOP checks into a single per-pallet verdict with
        # critical-failure dominance (R17). Thresholds come from verdict_config;
        # a confirmed mandatory violation dominates and is never averaged away,
        # and pose quality is applied as a gate on a PASS (never offsetting a
        # failure, R17.5/R17.7).
        t_verdict = time.perf_counter()
        verdict = aggregate(pose, sop_checks, verdict_config)
        verdict_ms = _elapsed_ms(t_verdict)

        timings = StageTimings(
            detect_ms=detect_ms,
            pose_ms=pose_ms,
            sop_ms=sop_ms,
            verdict_ms=verdict_ms,
            total_ms=detect_ms + pose_ms + sop_ms + verdict_ms,
        )

        # Surface the pose reason code at the Assessment level too.
        reason_codes = []
        if pose.reason_code is not None:
            reason_codes.append(pose.reason_code)

        # A calibration that could not be loaded is a genuine processing blocker
        # for this pallet (the pose cannot be solved), so raise the failure
        # signal honestly rather than reporting a clean empty result (R22.1/22.3,
        # R26.4). A merely-unavailable pose from a well-conditioned degrade
        # (e.g. insufficient keypoints) is NOT a failure signal — that is the
        # expected honest-unavailable path.
        if calibration is None:
            failure = FailureSignal(
                failed=True, reason_code=ReasonCode.CALIBRATION_INVALID
            )
        else:
            failure = FailureSignal(failed=False)

        assessment = Assessment(
            schema_version=config.schema_version,
            pallet_id=f"pallet-{index}",
            timestamp=frame.timestamp,
            source_image_ref=frame.source_image_ref,
            model_id=config.model_id,
            calibration_id=config.calibration_id,
            pose=pose,
            sop_checks=sop_checks,
            verdict=verdict,
            quality_flags=list(pose.quality_flags),
            reason_codes=reason_codes,
            timings=timings,
            failure=failure,
        )
        assessments.append(assessment)

    return assessments


def _write_assessments(
    assessments: list[Assessment], outputs_dir: Path, *, stem: str
) -> list[Path]:
    """Write each Assessment to ``outputs_dir`` as JSON; return written paths."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for assessment in assessments:
        out_path = outputs_dir / f"{stem}__{assessment.pallet_id}.json"
        out_path.write_text(assessment.serialise(), encoding="utf-8")
        written.append(out_path)
    return written


def run_pipeline(
    image_path: Optional["str | Path"] = None,
    *,
    config_path: "str | Path" = _DEFAULT_CONFIG_PATH,
    detector: Optional[Detector] = None,
    write: bool = True,
) -> list[Assessment]:
    """Run the end-to-end pipeline on one image and write Assessments.

    Parameters
    ----------
    image_path:
        Path to the input image. When ``None`` a synthetic placeholder frame is
        generated (clearly labelled ``synthetic://``) so the pipeline is
        runnable without real data.
    config_path:
        Path to ``pipeline.yaml``.
    detector:
        Optional detector override. When omitted the real detector is
        constructed from config (see :func:`build_detector`); passing a detector
        (e.g. a :class:`StubDetector` in tests) overrides construction.
    write:
        When True (default), Assessments are written to the configured
        ``outputs/`` directory as JSON.

    Returns
    -------
    list[Assessment]
        The Assessments produced (one per detected pallet).

    Raises
    ------
    IngestError
        If a provided image cannot be ingested (missing/corrupt).
    DetectorUnavailableError
        If the real detector is unavailable and stub fallback is disabled.
    """
    config = load_pipeline_config(config_path)

    t_ingest = time.perf_counter()
    if image_path is None:
        frame = make_synthetic_frame(
            width=config.input_resolution, height=config.input_resolution
        )
        stem = "synthetic-frame"
    else:
        frame = ingest_image(image_path)
        stem = Path(image_path).stem
    ingest_ms = _elapsed_ms(t_ingest)

    assessments = process_image(frame, config, detector=detector)

    # Fold ingest timing into each Assessment's timings.
    for assessment in assessments:
        base = assessment.timings
        assessment.timings = StageTimings(
            ingest_ms=ingest_ms,
            detect_ms=base.detect_ms,
            pose_ms=base.pose_ms,
            sop_ms=base.sop_ms,
            verdict_ms=base.verdict_ms,
            total_ms=(base.total_ms or 0.0) + ingest_ms,
        )

    if write:
        _write_assessments(assessments, config.outputs_dir, stem=stem)

    return assessments
