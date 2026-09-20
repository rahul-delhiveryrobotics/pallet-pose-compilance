"""CLI entrypoint: process one image end to end with an explicit exit-code contract.

This is the finalised command-line interface (task 13.1). It ingests one image
(or a synthetic placeholder), runs the full pipeline
(:func:`~pallet_pose_compliance.pipeline.run_pipeline`), writes one Assessment
per detected pallet to ``outputs/``, prints a short summary, and returns a
process exit code that lets a downstream consumer distinguish *absence of load*
from *system failure* (R22.2).

Determinism (R28.3)
-------------------
Runs are deterministic: the CLI performs no randomisation of its own and always
uses the fixed pipeline config (``configs/pipeline.yaml`` by default), whose
``seed`` is threaded into every stochastic stage (e.g. Monte Carlo pose
uncertainty). Repeated runs on identical input therefore produce identical
Assessments.

Exit-code contract (downstream consumers, R22.1/R22.2/R22.3)
------------------------------------------------------------
The exit code separates a *valid result* (including a valid **empty** result)
from a *processing failure*:

- **0 — success (includes "no pallet detected").** The run produced a valid
  result. This covers both one-or-more Assessments AND a valid empty result
  (no pallet detected). It also covers Assessments that carry an honest
  per-pallet failure signal / degraded ``unavailable`` status (e.g.
  ``POSE_INSUFFICIENT_KEYPOINTS``, ``POSE_AMBIGUOUS``, ``ILL_CONDITIONED``,
  ``POSE_UNCERTAINTY_EXCEEDED``, ``CALIBRATION_INVALID``, ``CAMERA_MOVED``,
  ``LOW_IMAGE_QUALITY``, ``DIMENSIONS_UNKNOWN``): a per-pallet failure signal
  with a ``Reason_Code`` is itself a *valid emitted result*, so the process
  still exits 0 — the reason is surfaced inside the Assessment, never as a lost
  run. This matches the design "Error Handling" table where every
  honest-unavailable/degrade condition maps to CLI exit 0.

- **1 — processing failed (``PROCESSING_FAILED``).** A genuine processing
  failure prevented producing any result: corrupt/missing input
  (:class:`~pallet_pose_compliance.ingest.IngestError`) or an unexpected
  runtime error during processing. No Assessment values are fabricated; a
  failure signal is emitted to stderr and (with ``--json``) to stdout.

- **2 — detector unavailable (blocker).** The real detector could not be
  loaded (ultralytics/weights missing) and the dev stub fallback is not
  enabled (:class:`~pallet_pose_compliance.detection.detector.DetectorUnavailableError`).
  This is a distinct honest blocker: no detections are fabricated. It is a
  non-zero (failure) exit because no result could be produced.

Usage
-----
    python -m pallet_pose_compliance.cli [IMAGE_PATH] [--config PATH]
                                         [--no-write] [--json]

When ``IMAGE_PATH`` is omitted a synthetic placeholder frame is generated.
``--json`` emits a machine-readable summary to stdout for downstream consumers.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .detection.detector import DetectorUnavailableError
from .ingest import IngestError
from .output.provenance import ReasonCode
from .pipeline import run_pipeline

# Exit codes (documented contract, R22.2). Kept as named constants so tests and
# downstream tooling can reference them rather than magic numbers.
EXIT_OK = 0  # valid result, including a valid empty "no pallet detected" result
EXIT_PROCESSING_FAILED = 1  # corrupt input / runtime error; no values fabricated
EXIT_DETECTOR_UNAVAILABLE = 2  # honest blocker: real detector could not be loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pallet-pose-compliance",
        description=(
            "Run the pallet pose-compliance pipeline on one image "
            "(ingest -> detect -> pose -> SOP -> verdict -> Assessment JSON). "
            "Exit 0 = valid result (including 'no pallet detected'); "
            "exit 1 = processing failed (corrupt input / runtime error, no "
            "values fabricated); exit 2 = detector unavailable (blocker)."
        ),
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to input image. If omitted, a synthetic frame is used.",
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help=(
            "Path to pipeline.yaml (default: configs/pipeline.yaml). The "
            "config's fixed seed makes runs deterministic (R28.3)."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write Assessment JSON files to outputs/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit a machine-readable JSON summary to stdout so downstream "
            "consumers can distinguish 'no pallet' from 'processing failed'."
        ),
    )
    return parser


def _emit_failure(
    message: str,
    reason_code: ReasonCode,
    exit_code: int,
    *,
    as_json: bool,
) -> int:
    """Emit an honest failure signal (no fabricated Assessment values) and return the exit code.

    The human-readable message always goes to stderr. When ``as_json`` is set a
    structured failure signal is also written to stdout so a downstream consumer
    can parse it: ``{"status": "processing_failed"|"detector_unavailable",
    "reason_code": ..., "assessments": [], ...}``. No Assessment values are
    fabricated in either channel (R22.1/R22.3).
    """
    print(message, file=sys.stderr)
    if as_json:
        status = (
            "detector_unavailable"
            if exit_code == EXIT_DETECTOR_UNAVAILABLE
            else "processing_failed"
        )
        payload = {
            "status": status,
            "failed": True,
            "reason_code": reason_code.value,
            "message": message,
            # No fabricated results: an explicit empty list, never a
            # placeholder Assessment.
            "assessments": [],
        }
        print(json.dumps(payload, sort_keys=True))
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI on one image. Returns a process exit code (see module docstring)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        assessments = run_pipeline(
            image_path=args.image_path,
            config_path=args.config,
            write=not args.no_write,
        )
    except IngestError as exc:
        # Corrupt/missing input is a genuine processing failure: no result can
        # be produced and no Assessment values are fabricated (design: Error
        # Handling — corrupt input -> PROCESSING_FAILED, non-zero exit).
        return _emit_failure(
            f"processing failed: {exc}",
            ReasonCode.PROCESSING_FAILED,
            EXIT_PROCESSING_FAILED,
            as_json=args.json,
        )
    except DetectorUnavailableError as exc:
        # Honest blocker: the real detector is unavailable (ultralytics/weights
        # missing) and the dev stub fallback is not enabled. Surface it
        # explicitly rather than fabricating detections (R26.3/R26.4). This is a
        # distinct non-zero exit (2) so downstream tooling can tell a missing
        # detector apart from corrupt input.
        message = (
            f"detector unavailable (blocker): {exc}\n"
            "No detections were fabricated. Provide assignment-trained weights "
            "at weights/<model_id>.pt, install ultralytics, or enable "
            "detector.allow_stub_fallback in the config for a clearly-labelled "
            "development stub."
        )
        return _emit_failure(
            message,
            ReasonCode.PROCESSING_FAILED,
            EXIT_DETECTOR_UNAVAILABLE,
            as_json=args.json,
        )
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all
        # Any other unexpected runtime error during processing maps to
        # PROCESSING_FAILED with a non-zero exit. We degrade loudly and do NOT
        # fabricate any Assessment values (design: Error Handling — runtime
        # error; R22.1). The exception type/message is surfaced for debugging.
        return _emit_failure(
            f"processing failed: unexpected runtime error: {exc!r}",
            ReasonCode.PROCESSING_FAILED,
            EXIT_PROCESSING_FAILED,
            as_json=args.json,
        )

    # Success path (exit 0), INCLUDING the valid empty "no pallet detected"
    # result. A per-pallet honest failure signal / degraded unavailable pose is
    # a valid emitted result and stays exit 0 — the reason is carried inside the
    # Assessment, never surfaced as a lost run.
    if args.json:
        payload = {
            "status": "no_pallet_detected" if not assessments else "ok",
            "failed": False,
            "written": not args.no_write,
            "assessments": [a.to_dict() for a in assessments],
        }
        print(json.dumps(payload, sort_keys=True))
    else:
        if not assessments:
            print(
                "No pallet detected (valid empty result, "
                f"{ReasonCode.NO_PALLET_DETECTED.value})."
            )
        else:
            print(
                f"Produced {len(assessments)} Assessment(s) "
                f"({'not written' if args.no_write else 'written to outputs/'})."
            )
            for assessment in assessments:
                failed = assessment.failure.failed
                reason = (
                    assessment.failure.reason_code.value
                    if assessment.failure.reason_code is not None
                    else "-"
                )
                print(
                    f"  {assessment.pallet_id}: verdict={assessment.verdict.verdict} "
                    f"pose={assessment.pose.pose_status} "
                    f"failure_signal={failed} reason={reason}"
                )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
