"""Latency benchmark with honest hardware provenance (R19, R20).

Implemented in task 14.1 (stub at scaffolding time, task 1.1).

What this script does
---------------------
1. **Measured host latency** — times the end-to-end pipeline on the *actual*
   host this script runs on, over a synthetic frame, for a configurable number
   of warmup + timed iterations. It reports the per-stage and total latency as a
   **distribution** (mean / median / p90 with an explicit sample_count) and
   labels every measured figure ``measured`` while **naming the host hardware it
   was actually measured on** (platform / processor / CPU / python) (R19.1,
   R19.2). It does NOT require ultralytics: by default a ``StubDetector`` is
   injected so the *pipeline* latency (ingest->detect->pose->SOP->verdict) is
   measured on the host without a trained model. The measured figure is honest
   about this: it is host pipeline latency with a stub detector, on the named
   host — never a figure attributed to the trained model on target hardware.

2. **Estimated Jetson section** — the Target_Hardware (NVIDIA Jetson Orin Nano
   @ 15 W) is NOT available for measurement. The script **states it is
   unavailable** and produces a reasoning-based expectation section labelled
   ``estimated``: documented reasoning about how host->Jetson latency/throughput
   would change (edge GPU vs host, INT8/TensorRT, the 15 W power cap, thermal)
   and an **estimated** throughput evaluation against the >=15 FPS goal (R19.3,
   R19.4, R20.1, R20.2, R20.3). Nothing in this section is ever labelled
   ``measured`` and it is never presented as measured-on-target.

Honesty discipline (Property 1, R26): the measured section carries provenance
``measured`` and names the host; the Jetson section carries provenance
``estimated`` and is explicitly ``measured_on_target = False``. The pure helpers
(host-info gathering, latency-summary math, Jetson-estimate assembly) are
importable and unit-tested without ultralytics via the stub path.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# Ensure the src-layout package is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pallet_pose_compliance.detection.stub import StubDetector  # noqa: E402
from pallet_pose_compliance.ingest import make_synthetic_frame  # noqa: E402
from pallet_pose_compliance.output.provenance import ProvenanceLabel  # noqa: E402
from pallet_pose_compliance.pipeline import (  # noqa: E402
    load_pipeline_config,
    process_image,
)

__all__ = [
    "HostInfo",
    "LatencySummary",
    "MeasuredLatencyReport",
    "JetsonEstimate",
    "BenchmarkReport",
    "gather_host_info",
    "summarise_latencies",
    "build_jetson_estimate",
    "measure_host_latency",
    "run_benchmark",
    "main",
]

#: The Target_Hardware description (never measured here — estimate only).
TARGET_HARDWARE = "NVIDIA Jetson Orin Nano @ 15 W"
#: The throughput goal the Jetson estimate is evaluated against (R20.3).
TARGET_FPS_GOAL = 15.0


@dataclass(frozen=True)
class HostInfo:
    """Identifies the host hardware a measured figure was ACTUALLY taken on.

    This is the honesty anchor for R19.1: every measured latency figure names
    the machine it was measured on. Only facts the ``platform`` module can
    report are recorded — no GPU details are fabricated.
    """

    label: str
    platform: str
    machine: str
    processor: str
    python_version: str
    gpu: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "python_version": self.python_version,
            "gpu": self.gpu,
        }


def gather_host_info(
    *, gpu_used: bool = False, gpu_note: Optional[str] = None
) -> HostInfo:
    """Detect and name the host hardware from the ``platform`` module.

    Never fabricates GPU information: unless the caller explicitly measured on a
    GPU (``gpu_used=True`` with a real ``gpu_note``), the GPU field records that
    no GPU was detected/used for this measurement, so the measured figure is a
    CPU-path figure honestly (R19.1, R26.3).
    """
    proc = platform.processor() or "unknown-processor"
    machine = platform.machine() or "unknown-machine"
    plat = platform.platform()
    if gpu_used and gpu_note:
        gpu = gpu_note
    else:
        gpu = "no GPU detected/used for this measurement (CPU path)"
    return HostInfo(
        label=f"{plat} / {proc}",
        platform=plat,
        machine=machine,
        processor=proc,
        python_version=platform.python_version(),
        gpu=gpu,
    )


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Nearest-rank/linear-interp percentile of a sorted, non-empty sequence."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence is undefined")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = pct / 100.0 * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


@dataclass(frozen=True)
class LatencySummary:
    """A latency distribution summary in milliseconds with a sample count.

    Reported as a distribution (mean/median/p90/min/max) with an explicit
    ``sample_count`` — never a single point estimate (R19.1).
    """

    sample_count: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


def summarise_latencies(samples: Sequence[float]) -> LatencySummary:
    """Summarise per-iteration latencies (ms) as a distribution.

    Raises
    ------
    ValueError
        If ``samples`` is empty (a distribution needs at least one sample; we do
        not fabricate a zero-sample summary).
    """
    values = [float(s) for s in samples]
    if not values:
        raise ValueError("cannot summarise an empty latency sample set")
    ordered = sorted(values)
    return LatencySummary(
        sample_count=len(values),
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        p90_ms=_percentile(ordered, 90.0),
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


@dataclass(frozen=True)
class MeasuredLatencyReport:
    """Measured host pipeline latency, labelled ``measured`` on the named host.

    ``provenance`` is always :attr:`ProvenanceLabel.MEASURED` and ``host`` names
    the hardware the figures were actually taken on (R19.1, R19.2). ``detector``
    records that a stub detector was used, so the measured figure is never
    mistaken for a trained-model-on-target figure.
    """

    provenance: ProvenanceLabel
    host: HostInfo
    detector: str
    warmup_iterations: int
    total: LatencySummary
    per_stage: dict[str, LatencySummary]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.value,
            "host": self.host.to_dict(),
            "detector": self.detector,
            "warmup_iterations": self.warmup_iterations,
            "total": self.total.to_dict(),
            "per_stage": {k: v.to_dict() for k, v in self.per_stage.items()},
            "note": self.note,
        }


@dataclass(frozen=True)
class JetsonEstimate:
    """Reasoning-based Jetson expectation labelled ``estimated`` (R19.3, R20).

    This section is NEVER measured on the target. ``measured_on_target`` is
    always ``False`` and ``provenance`` is always
    :attr:`ProvenanceLabel.ESTIMATED`. It documents the expected performance
    changes on the Jetson @ 15 W *with reasoning for each* (R20.1/R20.2) and
    evaluates an estimated throughput against the >=15 FPS goal, reporting the
    outcome as an estimate (R20.3).
    """

    provenance: ProvenanceLabel
    target_hardware: str
    measured_on_target: bool
    availability_note: str
    expected_changes: list[dict[str, str]]
    fps_goal: float
    estimated_latency_range_ms: tuple[float, float]
    estimated_fps_range: tuple[float, float]
    meets_goal_estimate: str
    goal_evaluation_note: str
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.value,
            "target_hardware": self.target_hardware,
            "measured_on_target": self.measured_on_target,
            "availability_note": self.availability_note,
            "expected_changes": self.expected_changes,
            "fps_goal": self.fps_goal,
            "estimated_latency_range_ms": list(self.estimated_latency_range_ms),
            "estimated_fps_range": list(self.estimated_fps_range),
            "meets_goal_estimate": self.meets_goal_estimate,
            "goal_evaluation_note": self.goal_evaluation_note,
            "assumptions": self.assumptions,
        }


def build_jetson_estimate(
    host_total: Optional[LatencySummary],
    *,
    fps_goal: float = TARGET_FPS_GOAL,
) -> JetsonEstimate:
    """Assemble the ``estimated`` Jetson section (never measured-on-target).

    The estimate is anchored on the *measured host* total latency where
    available (as a reasoning starting point only), then adjusted by documented
    reasoning factors for the edge GPU, INT8/TensorRT, the 15 W power cap and
    thermal throttling. All figures are ``estimated``; none are attributed to a
    measurement on the Jetson (R19.3, R19.4, R20). The estimated latency band is
    intentionally wide to reflect that this is a projection, not a measurement.
    Throughput is then evaluated against ``fps_goal`` and the outcome reported
    as an estimate (R20.3).
    """
    availability_note = (
        f"{TARGET_HARDWARE} is NOT available for measurement. No latency figure "
        "here was measured on the target hardware; the entire Jetson section is "
        "reasoning-based and labelled 'estimated' (R19.3, R19.4)."
    )

    expected_changes = [
        {
            "factor": "Edge GPU vs host CPU/GPU",
            "expected_change": (
                "Detector inference moves to the Orin Nano integrated Ampere "
                "GPU. Versus a host CPU path this is typically faster; versus a "
                "discrete host GPU it is slower."
            ),
            "reasoning": (
                "The Orin Nano GPU has far fewer CUDA cores and lower memory "
                "bandwidth than a desktop discrete GPU, but is purpose-built for "
                "efficient edge inference, so per-frame detector latency lands "
                "between a host CPU and a host discrete GPU."
            ),
        },
        {
            "factor": "INT8 / TensorRT export",
            "expected_change": (
                "Exporting the detector to TensorRT with INT8 quantisation is "
                "expected to reduce detector latency substantially vs FP32."
            ),
            "reasoning": (
                "TensorRT fuses layers and INT8 uses the Orin tensor cores at "
                "higher throughput and lower memory traffic; the accuracy cost of "
                "INT8 is reported separately by scripts/export.py (R21)."
            ),
        },
        {
            "factor": "15 W power cap",
            "expected_change": (
                "At the 15 W power mode clocks are capped below the peak power "
                "mode, so latency is higher than the Orin best case."
            ),
            "reasoning": (
                "The 15 W cap limits GPU/CPU clock frequencies; throughput scales "
                "roughly with the available power budget, so the 15 W figure is a "
                "conservative fraction of the unconstrained figure."
            ),
        },
        {
            "factor": "Thermal throttling (sustained)",
            "expected_change": (
                "Sustained streaming can trigger thermal throttling, raising "
                "steady-state latency above a short-burst measurement."
            ),
            "reasoning": (
                "Passive/limited cooling at 15 W means sustained load heats the "
                "SoC; the scheduler lowers clocks to stay in the thermal budget, "
                "so steady-state FPS is below burst FPS."
            ),
        },
        {
            "factor": "Non-detector pipeline stages (pose/SOP/verdict)",
            "expected_change": (
                "These CPU-bound stages are expected to be slower on the Orin "
                "Arm CPU than on a typical x86 host CPU."
            ),
            "reasoning": (
                "The pose PnP + Monte-Carlo, SOP checks and verdict run on the "
                "CPU; the Orin Arm cores are slower per-core than desktop x86, "
                "so these stages add proportionally more on the target."
            ),
        },
    ]

    assumptions = [
        "Detector exported to TensorRT INT8 and run on the Orin integrated GPU.",
        "Orin Nano configured in the 15 W power mode (not the peak power mode).",
        "Input resolution unchanged from the host configuration (pipeline.yaml).",
        "Single-stream inference; no batching across pallets/frames.",
        (
            "Anchored on the host pipeline latency as a reasoning starting point "
            "only; the host figure is measured with a stub detector, not the "
            "trained model, so the anchor itself is approximate."
        ),
    ]

    # Reasoning-based projection. Deliberately a WIDE band to signal this is an
    # estimate, not a measurement. If a host anchor exists, scale around it;
    # otherwise fall back to a documented nominal band.
    if host_total is not None and host_total.median_ms > 0:
        anchor = host_total.median_ms
        low_ms = max(1.0, anchor * 0.5)
        high_ms = max(low_ms + 1.0, anchor * 4.0)
    else:  # pragma: no cover - host anchor is always present via the pipeline
        low_ms, high_ms = 15.0, 120.0

    # FPS is the inverse of latency (single stream): high latency -> low FPS.
    fps_high = 1000.0 / low_ms
    fps_low = 1000.0 / high_ms

    if fps_low >= fps_goal:
        meets = "likely_meets"
        eval_note = (
            f"Even the conservative estimated throughput ({fps_low:.1f} FPS) is "
            f"at or above the {fps_goal:.0f} FPS goal, so the goal is likely met "
            "(estimated, not measured on target)."
        )
    elif fps_high >= fps_goal:
        meets = "uncertain"
        eval_note = (
            f"The estimated throughput band ({fps_low:.1f}-{fps_high:.1f} FPS) "
            f"straddles the {fps_goal:.0f} FPS goal; meeting it depends on INT8/"
            "TensorRT gains and the 15 W/thermal envelope. Outcome uncertain "
            "(estimated, not measured on target)."
        )
    else:
        meets = "likely_misses"
        eval_note = (
            f"Even the optimistic estimated throughput ({fps_high:.1f} FPS) is "
            f"below the {fps_goal:.0f} FPS goal, so the goal is likely NOT met "
            "without further optimisation (estimated, not measured on target)."
        )

    return JetsonEstimate(
        provenance=ProvenanceLabel.ESTIMATED,
        target_hardware=TARGET_HARDWARE,
        measured_on_target=False,
        availability_note=availability_note,
        expected_changes=expected_changes,
        fps_goal=fps_goal,
        estimated_latency_range_ms=(low_ms, high_ms),
        estimated_fps_range=(fps_low, fps_high),
        meets_goal_estimate=meets,
        goal_evaluation_note=eval_note,
        assumptions=assumptions,
    )


@dataclass(frozen=True)
class BenchmarkReport:
    """The full report: a ``measured`` host section + an ``estimated`` Jetson section."""

    measured: MeasuredLatencyReport
    jetson_estimate: JetsonEstimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured": self.measured.to_dict(),
            "jetson_estimate": self.jetson_estimate.to_dict(),
        }


def measure_host_latency(
    *,
    iterations: int,
    warmup: int,
    config_path: "str | Path",
    detector: Optional[Any] = None,
    host_info: Optional[HostInfo] = None,
) -> MeasuredLatencyReport:
    """Measure host pipeline latency over a synthetic frame (labelled measured).

    Runs ``warmup`` untimed iterations then ``iterations`` timed iterations of
    :func:`~pallet_pose_compliance.pipeline.process_image` on a synthetic frame.
    Per-stage timings come from each Assessment's ``StageTimings``; the total is
    the wall-clock time of the whole ``process_image`` call.

    A ``StubDetector`` is injected by default so the *pipeline* latency is
    measured on the host without requiring ultralytics. The report is honest
    about this via ``detector`` and ``note``.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1 (a distribution needs a sample)")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    config = load_pipeline_config(config_path)
    det = detector if detector is not None else StubDetector()
    detector_desc = (
        "injected detector (caller-supplied)"
        if detector is not None
        else "StubDetector (placeholder; pipeline latency only, NOT a trained model)"
    )

    frame = make_synthetic_frame(
        width=config.input_resolution, height=config.input_resolution
    )

    # Warmup (untimed) — lets imports/caches settle so timed samples are
    # representative and not dominated by first-call costs.
    for _ in range(warmup):
        process_image(frame, config, detector=det)

    total_samples: list[float] = []
    stage_samples: dict[str, list[float]] = {
        "ingest_ms": [],
        "detect_ms": [],
        "pose_ms": [],
        "sop_ms": [],
        "verdict_ms": [],
    }
    for _ in range(iterations):
        start = time.perf_counter()
        assessments = process_image(frame, config, detector=det)
        total_ms = (time.perf_counter() - start) * 1000.0
        total_samples.append(total_ms)
        if assessments:
            timings = assessments[0].timings
            for key in stage_samples:
                value = getattr(timings, key, None)
                if value is not None:
                    stage_samples[key].append(float(value))

    per_stage: dict[str, LatencySummary] = {}
    for key, samples in stage_samples.items():
        if samples:
            per_stage[key] = summarise_latencies(samples)

    host = host_info if host_info is not None else gather_host_info()
    note = (
        "Measured end-to-end pipeline latency (ingest->detect->pose->SOP->"
        "verdict) on the named host over a synthetic frame. The detect stage "
        "uses a stub detector, so this is HOST PIPELINE latency, not trained-"
        "model-on-target latency. Figure is 'measured' on this host only "
        "(R19.1, R19.2); it is NOT a measurement on the Jetson target."
    )
    return MeasuredLatencyReport(
        provenance=ProvenanceLabel.MEASURED,
        host=host,
        detector=detector_desc,
        warmup_iterations=warmup,
        total=summarise_latencies(total_samples),
        per_stage=per_stage,
        note=note,
    )


def run_benchmark(
    *,
    iterations: int,
    warmup: int,
    config_path: "str | Path",
    detector: Optional[Any] = None,
    fps_goal: float = TARGET_FPS_GOAL,
) -> BenchmarkReport:
    """Run the measured host benchmark and assemble the estimated Jetson section."""
    measured = measure_host_latency(
        iterations=iterations,
        warmup=warmup,
        config_path=config_path,
        detector=detector,
    )
    jetson = build_jetson_estimate(measured.total, fps_goal=fps_goal)
    return BenchmarkReport(measured=measured, jetson_estimate=jetson)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure host pipeline latency (labelled 'measured' on the named "
            "host) and report Jetson Orin Nano @ 15 W / 15 FPS expectations "
            "strictly as 'estimated'/unmeasured reasoning — never as measured on "
            "the target hardware (R19, R20)."
        )
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of timed iterations (the measured distribution's sample count).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of untimed warmup iterations before timing.",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "pipeline.yaml"),
        help="Path to pipeline.yaml.",
    )
    parser.add_argument(
        "--fps-goal",
        type=float,
        default=TARGET_FPS_GOAL,
        help="Throughput goal the Jetson estimate is evaluated against (default 15).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON to stdout.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON report; also prints a summary.",
    )
    return parser


def _format_human(report: BenchmarkReport) -> str:
    m = report.measured
    j = report.jetson_estimate
    lines: list[str] = []
    lines.append("=== MEASURED host pipeline latency ===")
    lines.append(f"  provenance : {m.provenance.value}")
    lines.append(f"  host       : {m.host.label}")
    lines.append(f"  gpu        : {m.host.gpu}")
    lines.append(f"  detector   : {m.detector}")
    lines.append(
        f"  total ms   : mean={m.total.mean_ms:.2f} median={m.total.median_ms:.2f} "
        f"p90={m.total.p90_ms:.2f} (n={m.total.sample_count}, warmup={m.warmup_iterations})"
    )
    for stage, summ in m.per_stage.items():
        lines.append(
            f"    {stage:<11}: mean={summ.mean_ms:.3f} median={summ.median_ms:.3f} "
            f"p90={summ.p90_ms:.3f} (n={summ.sample_count})"
        )
    lines.append("")
    lines.append(f"=== ESTIMATED {j.target_hardware} (NOT measured on target) ===")
    lines.append(f"  provenance          : {j.provenance.value}")
    lines.append(f"  measured_on_target  : {j.measured_on_target}")
    lines.append(f"  {j.availability_note}")
    lines.append(
        f"  estimated latency ms: {j.estimated_latency_range_ms[0]:.1f}"
        f"-{j.estimated_latency_range_ms[1]:.1f}"
    )
    lines.append(
        f"  estimated FPS       : {j.estimated_fps_range[0]:.1f}"
        f"-{j.estimated_fps_range[1]:.1f} (goal >= {j.fps_goal:.0f})"
    )
    lines.append(f"  meets goal estimate : {j.meets_goal_estimate}")
    lines.append(f"  {j.goal_evaluation_note}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns 0 on success."""
    args = _build_arg_parser().parse_args(argv)

    report = run_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        config_path=args.config,
        fps_goal=args.fps_goal,
    )

    payload = json.dumps(report.to_dict(), indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Wrote benchmark report to {out_path}", file=sys.stderr)

    if args.json:
        print(payload)
    else:
        print(_format_human(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
