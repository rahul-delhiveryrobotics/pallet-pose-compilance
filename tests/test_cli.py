"""Unit tests for the finalised CLI exit-code contract + determinism (task 13.1).

These tests pin the CLI's exit-code contract from the design "Error Handling"
table and requirements R22.1/R22.2/R22.3/R28.3:

- **No pallet detected** -> exit 0 with a valid empty result (NOT a failure).
- **Corrupt / missing input** (IngestError) -> non-zero exit (PROCESSING_FAILED)
  with a failure message and NO fabricated Assessment.
- **Unexpected runtime error** during processing -> non-zero exit
  (PROCESSING_FAILED), no fabricated values.
- **Detector unavailable** (blocker) -> distinct non-zero exit.
- **Determinism** -> two runs on the same synthetic input produce identical
  Assessment JSON (fixed seed + fixed config, no CLI randomisation).

The pipeline-invoking cases monkeypatch ``cli.run_pipeline`` so the CLI's
exit-code/failure-handling logic is exercised without requiring ultralytics or
real weights. Determinism is validated through the real synthetic-frame pipeline
path using an injected :class:`StubDetector` (the CLI's underlying deterministic
run), avoiding any dependency on the real detector.
"""

import json

import pytest

from pallet_pose_compliance import cli
from pallet_pose_compliance.detection.detector import DetectorUnavailableError
from pallet_pose_compliance.detection.stub import StubDetector
from pallet_pose_compliance.ingest import IngestError
from pallet_pose_compliance.output.schema import Assessment
from pallet_pose_compliance.pipeline import run_pipeline

_CONFIG_PATH = "configs/pipeline.yaml"


# ---------------------------------------------------------------------------
# No pallet detected -> exit 0 (valid empty result, NOT a failure)
# ---------------------------------------------------------------------------


def test_no_pallet_detected_exits_zero_with_empty_result(monkeypatch, capsys) -> None:
    """An empty result (no pallet) is a valid result: exit 0, no failure signal."""
    monkeypatch.setattr(cli, "run_pipeline", lambda **kwargs: [])

    code = cli.main(["--no-write"])

    assert code == cli.EXIT_OK
    out = capsys.readouterr()
    # The human-readable summary calls out the valid empty result.
    assert "No pallet detected" in out.out
    # Not reported as a failure on stderr.
    assert "processing failed" not in out.err.lower()


def test_no_pallet_detected_json_status_is_not_failed(monkeypatch, capsys) -> None:
    """--json emits status=no_pallet_detected, failed=False, empty assessments."""
    monkeypatch.setattr(cli, "run_pipeline", lambda **kwargs: [])

    code = cli.main(["--no-write", "--json"])

    assert code == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no_pallet_detected"
    assert payload["failed"] is False
    assert payload["assessments"] == []


# ---------------------------------------------------------------------------
# Corrupt / missing input (IngestError) -> non-zero, no fabricated Assessment
# ---------------------------------------------------------------------------


def test_corrupt_input_exits_nonzero_with_failure_message(monkeypatch, capsys) -> None:
    """A corrupt/missing input raises IngestError -> PROCESSING_FAILED exit."""

    def _raise(**kwargs):
        raise IngestError("frame could not be decoded (corrupt)")

    monkeypatch.setattr(cli, "run_pipeline", _raise)

    code = cli.main(["missing-or-corrupt.png"])

    assert code == cli.EXIT_PROCESSING_FAILED
    assert code != cli.EXIT_OK
    out = capsys.readouterr()
    assert "processing failed" in out.err.lower()
    # No fabricated Assessment values are printed to stdout.
    assert out.out.strip() == ""


def test_corrupt_input_json_failure_signal_has_no_fabricated_assessment(
    monkeypatch, capsys
) -> None:
    """--json failure signal for corrupt input carries no fabricated Assessment."""

    def _raise(**kwargs):
        raise IngestError("frame could not be decoded (corrupt)")

    monkeypatch.setattr(cli, "run_pipeline", _raise)

    code = cli.main(["corrupt.png", "--json"])

    assert code == cli.EXIT_PROCESSING_FAILED
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "processing_failed"
    assert payload["failed"] is True
    assert payload["reason_code"] == "PROCESSING_FAILED"
    # Explicit empty list -> nothing fabricated.
    assert payload["assessments"] == []


def test_missing_input_via_real_ingest_exits_nonzero(capsys, tmp_path) -> None:
    """End-to-end: a path that does not exist yields a non-zero PROCESSING_FAILED exit.

    Exercises the real ingest integrity check (no monkeypatch) so the CLI's
    IngestError handling is validated against the actual failure path.
    """
    missing = tmp_path / "does-not-exist.png"

    code = cli.main([str(missing), "--config", _CONFIG_PATH])

    assert code == cli.EXIT_PROCESSING_FAILED
    err = capsys.readouterr().err.lower()
    assert "processing failed" in err


# ---------------------------------------------------------------------------
# Unexpected runtime error -> non-zero PROCESSING_FAILED, no fabricated values
# ---------------------------------------------------------------------------


def test_unexpected_runtime_error_exits_processing_failed(monkeypatch, capsys) -> None:
    """An unexpected runtime error during processing maps to PROCESSING_FAILED."""

    def _raise(**kwargs):
        raise RuntimeError("boom in the middle of processing")

    monkeypatch.setattr(cli, "run_pipeline", _raise)

    code = cli.main(["--no-write", "--json"])

    assert code == cli.EXIT_PROCESSING_FAILED
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "processing_failed"
    assert payload["failed"] is True
    assert payload["reason_code"] == "PROCESSING_FAILED"
    assert payload["assessments"] == []


# ---------------------------------------------------------------------------
# Detector unavailable (blocker) -> distinct non-zero exit
# ---------------------------------------------------------------------------


def test_detector_unavailable_exits_with_distinct_code(monkeypatch, capsys) -> None:
    """A detector-unavailable blocker exits with the distinct non-zero code (2)."""

    def _raise(**kwargs):
        raise DetectorUnavailableError("ultralytics not installed")

    monkeypatch.setattr(cli, "run_pipeline", _raise)

    code = cli.main(["--no-write"])

    assert code == cli.EXIT_DETECTOR_UNAVAILABLE
    assert code != cli.EXIT_OK
    assert code != cli.EXIT_PROCESSING_FAILED
    err = capsys.readouterr().err.lower()
    assert "detector unavailable" in err
    # Honesty: it explicitly states no detections were fabricated.
    assert "fabricated" in err


# ---------------------------------------------------------------------------
# Success with detections -> exit 0
# ---------------------------------------------------------------------------


def test_success_with_detections_exits_zero(capsys) -> None:
    """A run producing Assessments (injected stub) exits 0 with a summary."""
    detector = StubDetector(num_pallets=2)

    def _run(**kwargs):
        # Delegate to the real pipeline with the injected stub detector, honouring
        # the CLI's write flag, so the success path is realistically exercised.
        return run_pipeline(
            image_path=kwargs.get("image_path"),
            config_path=kwargs.get("config_path", _CONFIG_PATH),
            detector=detector,
            write=kwargs.get("write", True),
        )

    import pallet_pose_compliance.cli as cli_mod

    orig = cli_mod.run_pipeline
    cli_mod.run_pipeline = _run
    try:
        code = cli_mod.main(["--no-write"])
    finally:
        cli_mod.run_pipeline = orig

    assert code == cli.EXIT_OK
    assert "Produced 2 Assessment(s)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Determinism (R28.3): identical input -> identical Assessment JSON
# ---------------------------------------------------------------------------


def test_deterministic_runs_produce_identical_assessment_json() -> None:
    """Two runs on the same synthetic input yield byte-identical Assessment JSON.

    This validates the CLI's underlying deterministic run (fixed seed from
    config, fixed config, no CLI randomisation, R28.3): running the pipeline the
    CLI drives twice on the same synthetic frame must produce identical
    serialised Assessments. An injected StubDetector avoids requiring the real
    detector while keeping the pose/uncertainty stages (seeded from config)
    exercised.
    """
    first = run_pipeline(
        image_path=None,
        config_path=_CONFIG_PATH,
        detector=StubDetector(num_pallets=2),
        write=False,
    )
    second = run_pipeline(
        image_path=None,
        config_path=_CONFIG_PATH,
        detector=StubDetector(num_pallets=2),
        write=False,
    )

    assert len(first) == len(second) == 2

    # Timestamps are ingest-time wall clock and timings are wall-clock ms; both
    # are legitimately non-deterministic. Determinism applies to the *analysed*
    # result (pose/SOP/verdict/reason codes/failure). Normalise the two
    # non-deterministic fields, then require byte-identical JSON.
    def _normalise(a: Assessment) -> dict:
        d = a.to_dict()
        d["timestamp"] = "<normalised>"
        d["timings"] = "<normalised>"
        return d

    for a, b in zip(first, second):
        assert json.dumps(_normalise(a), sort_keys=True) == json.dumps(
            _normalise(b), sort_keys=True
        )


def test_deterministic_pose_and_verdict_fields_match() -> None:
    """The analysed pose + verdict are identical across repeated runs (R28.3)."""
    runs = [
        run_pipeline(
            image_path=None,
            config_path=_CONFIG_PATH,
            detector=StubDetector(num_pallets=1),
            write=False,
        )
        for _ in range(2)
    ]

    a0 = runs[0][0]
    a1 = runs[1][0]

    assert a0.pose.pose_status == a1.pose.pose_status
    assert a0.pose.reason_code == a1.pose.reason_code
    assert a0.verdict.verdict == a1.verdict.verdict
    assert a0.reason_codes == a1.reason_codes
    assert a0.failure == a1.failure
    # If a pose is available, its metric fields are identical to full precision.
    if a0.pose.pose_status == "available":
        assert a0.pose.position_m == a1.pose.position_m
        assert a0.pose.orientation_deg == a1.pose.orientation_deg
