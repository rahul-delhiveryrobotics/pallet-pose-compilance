"""Tests for real-detector integration into the pipeline (task 4.8).

These cover the detector *selection/construction* behaviour added when the stub
detector was swapped for the real YOLO-pose detector, without requiring
ultralytics/torch to be installed:

- Injecting a detector overrides construction (the smoke-test pattern).
- By default (no injection) the pipeline constructs the real
  ``YoloPoseDetector``, preferring assignment-trained weights at
  ``weights/<model_id>.pt`` and otherwise a configured stock checkpoint.
- Honesty discipline: when the real detector is unavailable
  (``DetectorUnavailableError``), the pipeline surfaces the blocker by default
  and only uses the clearly-labelled dev stub when explicitly opted in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pallet_pose_compliance.detection.detector import (
    DetectorUnavailableError,
    YoloPoseDetector,
)
from pallet_pose_compliance.detection.stub import StubDetector
from pallet_pose_compliance.output.provenance import ProvenanceLabel
from pallet_pose_compliance.pipeline import (
    PipelineConfig,
    build_detector,
    load_pipeline_config,
    process_image,
    run_pipeline,
)
from pallet_pose_compliance.ingest import make_synthetic_frame

_CONFIG_PATH = "configs/pipeline.yaml"


def _config(tmp_path: Path, **overrides) -> PipelineConfig:
    """Build a PipelineConfig rooted at tmp_path with sane defaults."""
    base = load_pipeline_config(_CONFIG_PATH)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(exist_ok=True)
    defaults = dict(
        seed=base.seed,
        model_id=base.model_id,
        calibration_id=base.calibration_id,
        input_resolution=base.input_resolution,
        schema_version=base.schema_version,
        outputs_dir=tmp_path / "outputs",
        weights_dir=weights_dir,
        calibration_dir=base.calibration_dir,
        # Point at the repo's real SOP thresholds config (resolved to an
        # absolute path by load_pipeline_config) so the real analyzer loads its
        # boundaries; the config is required, so we supply it rather than making
        # it optional (task 9.10 honesty discipline).
        sop_thresholds_path=base.sop_thresholds_path,
        # Point at the repo's real verdict config (resolved to an absolute path
        # by load_pipeline_config) so the real aggregator loads its thresholds;
        # the config is required, so we supply it rather than making it optional
        # (task 10.1 honesty discipline).
        verdict_config_path=base.verdict_config_path,
        stock_model_id=base.stock_model_id,
        detector_conf=base.detector_conf,
        allow_stub_fallback=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_repo_config_exposes_detector_settings(self):
        cfg = load_pipeline_config(_CONFIG_PATH)
        assert cfg.stock_model_id  # non-empty stock fallback id
        assert 0.0 <= cfg.detector_conf <= 1.0
        # Honesty default: stub fallback is OFF unless explicitly enabled.
        assert cfg.allow_stub_fallback is False

    def test_weights_path_derived_from_model_id(self, tmp_path):
        cfg = _config(tmp_path, model_id="my-model")
        assert cfg.weights_path == cfg.weights_dir / "my-model.pt"


# ---------------------------------------------------------------------------
# Injection overrides construction (the smoke-test pattern)
# ---------------------------------------------------------------------------


class TestInjectionOverrides:
    def test_injected_detector_is_used_verbatim(self, tmp_path):
        cfg = _config(tmp_path)
        stub = StubDetector(num_pallets=2)
        selection = build_detector(cfg, detector=stub)
        assert selection.detector is stub
        assert selection.kind == "injected"

    def test_injected_stub_yields_assessments_without_touching_weights(
        self, tmp_path
    ):
        # No weights, no ultralytics — injecting a stub must still work.
        cfg = _config(tmp_path)
        frame = make_synthetic_frame(width=64, height=64)
        assessments = process_image(frame, cfg, detector=StubDetector(num_pallets=2))
        assert len(assessments) == 2

    def test_injected_yolo_detector_marked_real(self, tmp_path):
        cfg = _config(tmp_path)
        det = YoloPoseDetector(stock_model_id="yolo11n-pose.pt")
        selection = build_detector(cfg, detector=det)
        assert selection.is_real is True
        assert selection.provenance is det.provenance


# ---------------------------------------------------------------------------
# Default construction from config
# ---------------------------------------------------------------------------


class TestDefaultConstruction:
    def test_prefers_assignment_trained_weights_when_present(self, tmp_path):
        cfg = _config(tmp_path)
        # Create the assignment-trained weights artefact so it is preferred.
        cfg.weights_path.write_bytes(b"not-a-real-model")

        captured = {}

        def _fake_load(self):
            captured["is_trained"] = self.is_assignment_trained
            captured["weights_path"] = self.weights_path
            # Pretend the model loaded successfully.

        # Patch load so we don't need ultralytics; verify selection wiring.
        original = YoloPoseDetector.load
        YoloPoseDetector.load = _fake_load  # type: ignore[method-assign]
        try:
            selection = build_detector(cfg)
        finally:
            YoloPoseDetector.load = original  # type: ignore[method-assign]

        assert selection.kind == "assignment_trained"
        assert selection.is_real is True
        assert selection.provenance is ProvenanceLabel.MEASURED
        assert captured["is_trained"] is True
        assert captured["weights_path"] == cfg.weights_path

    def test_falls_back_to_stock_when_no_trained_weights(self, tmp_path):
        cfg = _config(tmp_path)  # weights dir empty

        def _fake_load(self):
            return None

        original = YoloPoseDetector.load
        YoloPoseDetector.load = _fake_load  # type: ignore[method-assign]
        try:
            selection = build_detector(cfg)
        finally:
            YoloPoseDetector.load = original  # type: ignore[method-assign]

        assert selection.kind == "stock"
        assert selection.is_real is True
        # A stock checkpoint is 'estimated', never 'measured'.
        assert selection.provenance is ProvenanceLabel.ESTIMATED


# ---------------------------------------------------------------------------
# Fallback / honesty discipline when detector unavailable
# ---------------------------------------------------------------------------


class TestUnavailableFallback:
    def test_surfaces_blocker_by_default(self, tmp_path):
        cfg = _config(tmp_path, allow_stub_fallback=False)

        def _raise(self):
            raise DetectorUnavailableError("ultralytics not installed")

        original = YoloPoseDetector.load
        YoloPoseDetector.load = _raise  # type: ignore[method-assign]
        try:
            with pytest.raises(DetectorUnavailableError):
                build_detector(cfg)
        finally:
            YoloPoseDetector.load = original  # type: ignore[method-assign]

    def test_run_pipeline_propagates_blocker_by_default(self, tmp_path):
        cfg = _config(tmp_path, allow_stub_fallback=False)

        def _raise(self):
            raise DetectorUnavailableError("weights missing")

        original = YoloPoseDetector.load
        YoloPoseDetector.load = _raise  # type: ignore[method-assign]
        try:
            frame = make_synthetic_frame(width=64, height=64)
            with pytest.raises(DetectorUnavailableError):
                process_image(frame, cfg)
        finally:
            YoloPoseDetector.load = original  # type: ignore[method-assign]

    def test_opt_in_stub_fallback_is_labelled_not_real(self, tmp_path):
        cfg = _config(tmp_path, allow_stub_fallback=True)

        def _raise(self):
            raise DetectorUnavailableError("ultralytics not installed")

        original = YoloPoseDetector.load
        YoloPoseDetector.load = _raise  # type: ignore[method-assign]
        try:
            selection = build_detector(cfg)
        finally:
            YoloPoseDetector.load = original  # type: ignore[method-assign]

        assert selection.kind == "stub"
        assert selection.is_real is False
        # Stub detections are never presented as measured/real output.
        assert selection.provenance is ProvenanceLabel.UNAVAILABLE
        assert "STUB" in selection.note.upper()
        assert isinstance(selection.detector, StubDetector)

    def test_opt_in_stub_fallback_still_produces_honest_assessments(self, tmp_path):
        cfg = _config(tmp_path, allow_stub_fallback=True)

        def _raise(self):
            raise DetectorUnavailableError("ultralytics not installed")

        original = YoloPoseDetector.load
        YoloPoseDetector.load = _raise  # type: ignore[method-assign]
        try:
            frame = make_synthetic_frame(width=64, height=64)
            assessments = process_image(frame, cfg)
        finally:
            YoloPoseDetector.load = original  # type: ignore[method-assign]

        # The dev stub yields a pallet, but pose remains honestly unavailable
        # and the verdict is MANUAL_INSPECTION (never a fabricated measurement).
        assert len(assessments) == 1
        assert assessments[0].pose.pose_status == "unavailable"
        assert assessments[0].verdict.verdict == "MANUAL_INSPECTION"
