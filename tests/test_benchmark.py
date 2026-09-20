"""Unit tests for the benchmark.py honesty + latency logic (task 14.1).

Covers ``scripts/benchmark.py`` without ultralytics/real weights (the measured
host figure is produced via the injected ``StubDetector`` pipeline path):

- Measured host figures are labelled ``measured`` and NAME a host (R19.1/R19.2).
- The Jetson section is labelled ``estimated`` and is NEVER labelled measured
  and NEVER measured-on-target (R19.3/R19.4).
- The latency summary reports the correct ``sample_count`` and a valid
  distribution (R19.1).
- The >=15 FPS evaluation outcome is present as an estimate (R20.3).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load scripts/benchmark.py as a module (it is a script, not an installed package).
_spec = importlib.util.spec_from_file_location(
    "pallet_benchmark_script", _REPO_ROOT / "scripts" / "benchmark.py"
)
assert _spec and _spec.loader
bench = importlib.util.module_from_spec(_spec)
sys.modules["pallet_benchmark_script"] = bench
_spec.loader.exec_module(bench)

from pallet_pose_compliance.detection.stub import StubDetector  # noqa: E402
from pallet_pose_compliance.output.provenance import ProvenanceLabel  # noqa: E402

_CONFIG = str(_REPO_ROOT / "configs" / "pipeline.yaml")


# ---------------------------------------------------------------------------
# Pure helpers: latency summary math
# ---------------------------------------------------------------------------


def test_summarise_latencies_reports_correct_sample_count_and_distribution():
    summary = bench.summarise_latencies([10.0, 20.0, 30.0, 40.0])
    assert summary.sample_count == 4
    assert summary.min_ms == 10.0
    assert summary.max_ms == 40.0
    assert summary.median_ms == 25.0
    assert summary.mean_ms == pytest.approx(25.0)
    # p90 lies between the top two samples (never below the median here).
    assert summary.median_ms <= summary.p90_ms <= summary.max_ms


def test_summarise_latencies_single_sample():
    summary = bench.summarise_latencies([7.5])
    assert summary.sample_count == 1
    assert summary.mean_ms == summary.median_ms == summary.p90_ms == 7.5


def test_summarise_latencies_rejects_empty():
    with pytest.raises(ValueError):
        bench.summarise_latencies([])


# ---------------------------------------------------------------------------
# Host info: names a host, never fabricates a GPU
# ---------------------------------------------------------------------------


def test_gather_host_info_names_a_host_and_does_not_fabricate_gpu():
    host = bench.gather_host_info()
    assert host.label  # non-empty: a host is named (R19.1)
    assert host.platform
    assert host.python_version
    # No GPU was measured on -> must be honest about the CPU path, not invented.
    assert "GPU" in host.gpu or "gpu" in host.gpu
    assert "no GPU" in host.gpu


def test_gather_host_info_records_gpu_only_when_explicitly_used():
    host = bench.gather_host_info(gpu_used=True, gpu_note="Test GPU X")
    assert host.gpu == "Test GPU X"


# ---------------------------------------------------------------------------
# Measured host report: labelled measured + names a host (R19.1, R19.2)
# ---------------------------------------------------------------------------


def test_measured_report_is_labelled_measured_and_names_host():
    # Default (no injected detector) -> uses the StubDetector path so it runs
    # without ultralytics; the report must be honest that a stub was used.
    report = bench.measure_host_latency(iterations=3, warmup=1, config_path=_CONFIG)
    assert report.provenance is ProvenanceLabel.MEASURED
    assert report.provenance.value == "measured"
    # Names the actual host the figure was measured on.
    assert report.host.label
    # Distribution carries the requested sample count.
    assert report.total.sample_count == 3
    # Per-stage summaries are also distributions with matching sample counts.
    assert report.per_stage, "expected at least one per-stage latency summary"
    for summary in report.per_stage.values():
        assert summary.sample_count == 3
    # Honest about the stub detector / host-only scope.
    assert "stub" in report.detector.lower()
    assert "not" in report.note.lower() and "jetson" in report.note.lower()


def test_measure_host_latency_rejects_bad_iterations():
    with pytest.raises(ValueError):
        bench.measure_host_latency(
            iterations=0, warmup=0, config_path=_CONFIG, detector=StubDetector()
        )


# ---------------------------------------------------------------------------
# Jetson estimate: estimated, never measured, never measured-on-target
# ---------------------------------------------------------------------------


def _host_total(median_ms: float) -> "bench.LatencySummary":
    return bench.LatencySummary(
        sample_count=10,
        mean_ms=median_ms,
        median_ms=median_ms,
        p90_ms=median_ms,
        min_ms=median_ms,
        max_ms=median_ms,
    )


def test_jetson_estimate_is_estimated_and_never_measured_on_target():
    est = bench.build_jetson_estimate(_host_total(20.0))
    assert est.provenance is ProvenanceLabel.ESTIMATED
    assert est.provenance.value == "estimated"
    # NEVER measured, NEVER on target.
    assert est.measured_on_target is False
    assert est.provenance is not ProvenanceLabel.MEASURED
    assert "not available" in est.availability_note.lower()
    assert "jetson" in est.target_hardware.lower()
    # Reasoning for each expected change is present (R20.1/R20.2).
    assert est.expected_changes
    for change in est.expected_changes:
        assert change["reasoning"].strip()
        assert change["expected_change"].strip()
    assert est.assumptions


def test_jetson_estimate_evaluates_fps_goal_as_estimate():
    est = bench.build_jetson_estimate(_host_total(20.0), fps_goal=15.0)
    assert est.fps_goal == 15.0
    # The >=15 FPS evaluation outcome is present as one of the estimate labels.
    assert est.meets_goal_estimate in {"likely_meets", "uncertain", "likely_misses"}
    assert est.goal_evaluation_note.strip()
    lo, hi = est.estimated_fps_range
    assert 0.0 < lo <= hi


def test_jetson_estimate_goal_outcome_tracks_latency_band():
    # Very fast host -> optimistic band should clear the goal.
    fast = bench.build_jetson_estimate(_host_total(2.0), fps_goal=15.0)
    assert fast.meets_goal_estimate == "likely_meets"
    # Very slow host -> even optimistic estimate misses the goal.
    slow = bench.build_jetson_estimate(_host_total(500.0), fps_goal=15.0)
    assert slow.meets_goal_estimate == "likely_misses"


# ---------------------------------------------------------------------------
# Full report + serialisation honesty
# ---------------------------------------------------------------------------


def test_run_benchmark_full_report_keeps_sections_labelled_correctly():
    report = bench.run_benchmark(
        iterations=3, warmup=1, config_path=_CONFIG, detector=StubDetector()
    )
    data = report.to_dict()
    # Measured section stays 'measured' and names a host.
    assert data["measured"]["provenance"] == "measured"
    assert data["measured"]["host"]["label"]
    assert data["measured"]["total"]["sample_count"] == 3
    # Jetson section stays 'estimated' and is never measured-on-target.
    assert data["jetson_estimate"]["provenance"] == "estimated"
    assert data["jetson_estimate"]["measured_on_target"] is False
    assert data["jetson_estimate"]["meets_goal_estimate"] in {
        "likely_meets",
        "uncertain",
        "likely_misses",
    }
    # Honesty invariant: the Jetson section is never labelled 'measured'.
    assert data["jetson_estimate"]["provenance"] != "measured"


def test_main_json_smoke(capsys):
    rc = bench.main(["--iterations", "2", "--warmup", "0", "--config", _CONFIG, "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"provenance": "measured"' in out
    assert '"provenance": "estimated"' in out
    assert '"measured_on_target": false' in out
