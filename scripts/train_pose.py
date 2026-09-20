"""Train a YOLO26-pose pallet keypoint model on the SIMULATED pose dataset.

Trains an Ultralytics YOLO-pose model to localise the eight pallet corners on
the simulated dataset produced by ``scripts/generate_synthetic_pose.py``. The
trained keypoints feed the known-geometry PnP pose estimator downstream.

Base weights (documented, verified)
-----------------------------------
Prefers ``yolo26n-pose.pt`` (verified present in the Ultralytics asset registry
and downloadable). Falls back to ``yolo11n-pose.pt`` if the primary base cannot
be loaded, and records the substitution in ``reports/pose_train_meta.json`` so
the report is honest about which base was actually used.

Quick settings (fixed seed 42, deterministic)
----------------------------------------------
``imgsz=640, epochs=20, batch=16, device=0, seed=42, deterministic=True``. These
are "quick" defaults appropriate for a learnable synthetic dataset; ``--epochs``
can lower them further if training is slow.

Outputs
-------
- ``runs/pose/pallet-pose-v0/`` — Ultralytics run directory (best.pt inside).
- ``weights/yolo26-pose-pallet-v0.pt`` — copy of the selected best weights.
- ``reports/pose_train_meta.json`` — hyperparameters + which base weights were
  used (and any fallback), plus dataset/run references.

Honesty: the model is trained on SIMULATED data. Any accuracy claim from it is
against simulated ground truth; there is a synthetic-to-real gap.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure the src-layout package is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DEFAULT_DATA = _REPO_ROOT / "data" / "synthetic_pose" / "data.yaml"
_DEFAULT_PROJECT = _REPO_ROOT / "runs" / "pose"
_DEFAULT_RUN_NAME = "pallet-pose-v0"
_OUTPUT_WEIGHTS = _REPO_ROOT / "weights" / "yolo26-pose-pallet-v0.pt"
_META_PATH = _REPO_ROOT / "reports" / "pose_train_meta.json"

_PRIMARY_BASE = "yolo26n-pose.pt"
_FALLBACK_BASE = "yolo11n-pose.pt"

__all__ = ["resolve_base_model", "train_pose", "main"]


def _load_seed(config_path: Path) -> int:
    """Read the fixed seed from configs/pipeline.yaml (default 42)."""
    import yaml

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return 42
    return int(data.get("seed", 42))


def resolve_base_model(primary: str = _PRIMARY_BASE, fallback: str = _FALLBACK_BASE) -> tuple[str, bool, str]:
    """Return ``(base_model, used_fallback, note)`` for the pose base weights.

    Tries to construct the primary base (which triggers a download if needed);
    on any failure falls back to the documented fallback base. Never fabricates
    a trained model — this only selects a *base* checkpoint to fine-tune from.
    """
    from ultralytics import YOLO  # type: ignore

    try:
        YOLO(primary)
        return primary, False, f"primary base {primary!r} loaded"
    except Exception as exc:  # pragma: no cover - depends on network/registry
        YOLO(fallback)
        return (
            fallback,
            True,
            f"primary base {primary!r} unavailable ({exc}); "
            f"substituted fallback {fallback!r}",
        )


def train_pose(
    *,
    data_yaml: Path = _DEFAULT_DATA,
    epochs: int = 20,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    seed: int = 42,
    project: Path = _DEFAULT_PROJECT,
    run_name: str = _DEFAULT_RUN_NAME,
    output_weights: Path = _OUTPUT_WEIGHTS,
    meta_path: Path = _META_PATH,
) -> dict[str, Any]:
    """Run YOLO-pose training and copy best weights; write training meta.

    Raises
    ------
    RuntimeError
        If the dataset YAML is missing (explicit blocker; no model claimed).
    """
    if not Path(data_yaml).exists():
        raise RuntimeError(
            f"[TRAINING BLOCKED: NO_DATASET] dataset YAML {str(data_yaml)!r} not "
            "found; run scripts/generate_synthetic_pose.py first. No model is "
            "produced."
        )

    from ultralytics import YOLO  # type: ignore

    base_model, used_fallback, base_note = resolve_base_model()

    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "batch": int(batch),
        "imgsz": int(imgsz),
        "device": device,
        "seed": int(seed),
        "deterministic": True,
        "project": str(project),
        "name": run_name,
        "exist_ok": True,
        # Corner ordering is orientation-specific: disable horizontal flip so
        # augmentation never swaps the keypoint identities.
        "fliplr": 0.0,
    }

    model = YOLO(base_model)
    results = model.train(**train_args)

    save_dir = Path(getattr(results, "save_dir", project / run_name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(
            f"training finished but best weights not found at {best!s}"
        )

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, output_weights)

    # Pull the final validation metrics from the results (best-effort).
    final_metrics: dict[str, Any] = {}
    try:
        rd = getattr(results, "results_dict", None)
        if isinstance(rd, dict):
            final_metrics = {str(k): float(v) for k, v in rd.items()}
    except Exception:
        final_metrics = {}

    meta = {
        "provenance": "simulated",
        "trainer": "scripts/train_pose.py",
        "base_model_requested": _PRIMARY_BASE,
        "base_model_used": base_model,
        "used_fallback_base": used_fallback,
        "base_model_note": base_note,
        "hyperparameters": {
            "epochs": int(epochs),
            "batch": int(batch),
            "imgsz": int(imgsz),
            "device": device,
            "seed": int(seed),
            "deterministic": True,
            "fliplr": 0.0,
        },
        "checkpoint_selection": (
            "Ultralytics best.pt (selected by held-out pose fitness/mAP, not "
            "training loss)"
        ),
        "dataset": {
            "data_yaml": str(Path(data_yaml).resolve()),
            "note": "SIMULATED synthetic pose dataset (see synthetic_pose_summary.json)",
        },
        "outputs": {
            "run_dir": str(save_dir.resolve()),
            "best_weights": str(best.resolve()),
            "copied_weights": str(output_weights.resolve()),
        },
        "final_validation_metrics": final_metrics,
        "honesty_note": (
            "Trained on SIMULATED data; accuracy is against simulated GT only. "
            "There is a synthetic-to-real gap (simple rendered pallets)."
        ),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLO26-pose (fallback yolo11n-pose) on the SIMULATED pallet "
            "pose dataset. Quick settings; fixed seed 42; deterministic."
        )
    )
    parser.add_argument("--data", default=str(_DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default: pipeline.yaml seed / 42).")
    parser.add_argument("--project", default=str(_DEFAULT_PROJECT))
    parser.add_argument("--name", default=_DEFAULT_RUN_NAME)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    seed = args.seed if args.seed is not None else _load_seed(_REPO_ROOT / "configs" / "pipeline.yaml")

    print(
        f"Training YOLO-pose: base={_PRIMARY_BASE} (fallback {_FALLBACK_BASE}) "
        f"epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
        f"device={args.device} seed={seed} deterministic=True"
    )
    try:
        meta = train_pose(
            data_yaml=Path(args.data),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            seed=seed,
            project=Path(args.project),
            run_name=args.name,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        f"Training complete. base_used={meta['base_model_used']} "
        f"(fallback={meta['used_fallback_base']}); "
        f"weights -> {meta['outputs']['copied_weights']}"
    )
    print(f"Training meta written to {_META_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
