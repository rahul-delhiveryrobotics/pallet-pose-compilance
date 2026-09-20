"""Executable YOLO-pose training workflow with blocked-training fallback.

Implements R4, R5.5, R26.3. Trains the pallet detection/localisation YOLO-pose
model with a fixed seed (from configs/pipeline.yaml), documented
hyperparameters (epochs / learning rate / batch / image size), and selects the
final checkpoint by the best held-out localisation metric (not training loss),
per the design's Detector "Training schedule / checkpoint selection" note
(R4.3, R4.4, R28.3).

Honesty / blocked-training fallback (R5.5, R26.3, R26.4): training may be
blocked in-window by a missing dependency (ultralytics/torch), missing dataset,
or no compute. When blocked, the script reports the blocker explicitly (clear
message + non-zero exit) and does NOT produce or claim a trained model. It never
presents a generic pretrained checkpoint as assignment-trained; if a stock
checkpoint is later used for inference, the detector wrapper labels that run
`estimated` and flags it not-assignment-trained.

The logic is factored into small, importable, side-effect-free helpers
(hyperparameter resolution, blocker detection, checkpoint selection) so the
workflow is unit-testable without ultralytics/torch installed. Actual training
only runs when `run_training` is invoked and the dependencies/data are present.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Repository root (scripts/train.py -> repo root is the parent of scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class TrainingHyperparameters:
    """Documented, fixed training hyperparameters (R4.3, R28.3).

    Each field is a deliberate, documented choice; `reasoning` records why so
    the decision log / README can cite it (R4.3, R4.4). The values are modest
    defaults suited to a small assembled dataset within a 5-day window.
    """

    seed: int = 42
    epochs: int = 100
    learning_rate: float = 0.01  # SGD initial LR (Ultralytics `lr0`).
    batch_size: int = 16
    imgsz: int = 640
    base_model: str = "yolo11n-pose.pt"  # small, Jetson-exportable base.
    # Metric used to *select* the best checkpoint (held-out localisation, not
    # training loss). Ultralytics exposes pose mAP as `metrics/mAP50-95(P)`.
    checkpoint_selection_metric: str = "metrics/mAP50-95(P)"
    reasoning: dict[str, str] = field(
        default_factory=lambda: {
            "epochs": "100 epochs balances convergence against a 5-day window on a small dataset.",
            "learning_rate": "0.01 SGD lr0 is the Ultralytics default, stable for fine-tuning a pose base.",
            "batch_size": "16 fits typical single-GPU memory at 640px; adjust to hardware.",
            "imgsz": "640 matches pipeline.yaml input_resolution (box-visibility vs latency trade-off).",
            "base_model": "yolo11n-pose is small and TensorRT-exportable for the Jetson estimate path.",
            "checkpoint_selection_metric": "Select by held-out pose localisation mAP, never training loss, so the checkpoint reflects generalisation (R4.3).",
        }
    )

    def as_ultralytics_train_args(self, *, data_yaml: str, project: str) -> dict[str, Any]:
        """Build the Ultralytics `model.train(**kwargs)` argument mapping.

        The returned mapping fixes the seed and pins the documented
        hyperparameters. `deterministic=True` aids reproducibility (R28.3).
        """
        return {
            "data": data_yaml,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.imgsz,
            "lr0": self.learning_rate,
            "seed": self.seed,
            "deterministic": True,
            "project": project,
            "name": "pallet-yolo-pose",
            "exist_ok": True,
        }


@dataclass(frozen=True)
class TrainingBlocker:
    """An explicit training blocker (R26.3/R26.4 honest degrade).

    `kind` is a stable machine-readable code; `message` is human-readable.
    """

    kind: str
    message: str


def load_seed(config_path: "str | Path" = _DEFAULT_CONFIG_PATH) -> int:
    """Read the fixed random seed from configs/pipeline.yaml (R28.3).

    Falls back to 42 (the documented default) when the key/file is absent.
    """
    path = Path(config_path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return 42
    return int(data.get("seed", 42))


def resolve_hyperparameters(
    config_path: "str | Path" = _DEFAULT_CONFIG_PATH,
    *,
    overrides: Optional[dict[str, Any]] = None,
) -> TrainingHyperparameters:
    """Resolve fixed hyperparameters, seeding from pipeline.yaml (R28.3).

    `overrides` may pin any field (used by tests / CLI flags). The seed and
    `imgsz` default to the pipeline config so training matches inference.
    """
    path = Path(config_path)
    seed = 42
    imgsz = 640
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        seed = int(data.get("seed", 42))
        imgsz = int(data.get("input_resolution", 640))
    except FileNotFoundError:
        pass
    base: dict[str, Any] = {"seed": seed, "imgsz": imgsz}
    if overrides:
        base.update(overrides)
    return TrainingHyperparameters(**base)


def detect_blocker(
    *,
    data_yaml: Optional["str | Path"],
    require_ultralytics: bool = True,
) -> Optional[TrainingBlocker]:
    """Return an explicit TrainingBlocker if training cannot run, else None.

    Checks, in order:

    1. Missing dataset - `data_yaml` is None or the file is absent.
    2. Missing dependency - ultralytics/torch not importable.

    The dependency check is skipped when `require_ultralytics` is False (used by
    tests to isolate the data-missing path). This function has no side effects
    and never fabricates a model.
    """
    if data_yaml is None:
        return TrainingBlocker(
            kind="NO_DATASET",
            message=(
                "no dataset YAML provided; assemble a dataset (see dataset.py / "
                "DATASET.md) and pass --data. Training is blocked; no model is "
                "produced and no metrics are claimed."
            ),
        )
    if not Path(data_yaml).exists():
        return TrainingBlocker(
            kind="NO_DATASET",
            message=(
                f"dataset YAML {str(data_yaml)!r} does not exist. Training is "
                "blocked; no model is produced and no metrics are claimed."
            ),
        )
    if require_ultralytics:
        try:
            import ultralytics  # noqa: F401
        except Exception:  # pragma: no cover - depends on env
            return TrainingBlocker(
                kind="NO_ULTRALYTICS",
                message=(
                    "ultralytics is not installed; cannot train the YOLO-pose "
                    "model. Install the declared dependency. Training is "
                    "blocked; no model is produced and a stock checkpoint (if "
                    "used for inference) is labelled 'estimated' and flagged "
                    "not-assignment-trained (R5.5, R26.3)."
                ),
            )
    return None


def select_best_checkpoint(
    metrics_by_checkpoint: dict[str, float],
    *,
    higher_is_better: bool = True,
) -> str:
    """Select the checkpoint with the best held-out localisation metric.

    `metrics_by_checkpoint` maps a checkpoint reference to its held-out
    localisation metric value (never training loss, R4.3). Ties resolve to the
    first-seen checkpoint for determinism.

    Raises
    ------
    ValueError
        If `metrics_by_checkpoint` is empty.
    """
    if not metrics_by_checkpoint:
        raise ValueError("no checkpoints to select from")
    items = list(metrics_by_checkpoint.items())
    best_ref, best_val = items[0]
    for ref, val in items[1:]:
        if (higher_is_better and val > best_val) or (
            not higher_is_better and val < best_val
        ):
            best_ref, best_val = ref, val
    return best_ref


def run_training(
    *,
    data_yaml: Optional["str | Path"],
    hyperparameters: Optional[TrainingHyperparameters] = None,
    weights_dir: "str | Path" = _REPO_ROOT / "weights",
    config_path: "str | Path" = _DEFAULT_CONFIG_PATH,
) -> Path:
    """Run the full training workflow, returning the selected checkpoint path.

    Reports any blocker explicitly by raising RuntimeError with the blocker
    message (the CLI turns this into a non-zero exit). On success, trains with
    the fixed seed / documented hyperparameters and selects the best checkpoint
    by the held-out localisation metric.

    Raises
    ------
    RuntimeError
        If training is blocked (missing deps/data) - an explicit, honest blocker
        rather than a fabricated model.
    """
    hp = hyperparameters or resolve_hyperparameters(config_path)

    blocker = detect_blocker(data_yaml=data_yaml)
    if blocker is not None:
        raise RuntimeError(f"[TRAINING BLOCKED: {blocker.kind}] {blocker.message}")

    # Dependencies + data confirmed present: import lazily and train.
    from ultralytics import YOLO  # type: ignore

    project = str(Path(weights_dir))
    model = YOLO(hp.base_model)
    results = model.train(
        **hp.as_ultralytics_train_args(data_yaml=str(data_yaml), project=project)
    )

    # Ultralytics writes best.pt (selected by its own fitness, dominated by mAP
    # - a held-out localisation metric, not training loss).
    save_dir = Path(getattr(results, "save_dir", Path(project) / "pallet-yolo-pose"))
    best = save_dir / "weights" / "best.pt"
    return best


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the pallet YOLO-pose detector with a fixed seed and "
            "documented hyperparameters. Reports an explicit blocker (non-zero "
            "exit) when training cannot run; never fabricates a trained model."
        )
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the YOLO dataset YAML (train/held-out). Required to train.",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="Path to configs/pipeline.yaml (supplies the fixed seed / imgsz).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the documented epoch count."
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, non-zero blocked)."""
    args = _build_arg_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    hp = resolve_hyperparameters(args.config, overrides=overrides)

    print(
        "Training config (fixed, documented): "
        f"seed={hp.seed} epochs={hp.epochs} lr0={hp.learning_rate} "
        f"batch={hp.batch_size} imgsz={hp.imgsz} base={hp.base_model}"
    )
    print(
        "Checkpoint selection: best held-out localisation metric "
        f"({hp.checkpoint_selection_metric}), NOT training loss."
    )

    try:
        best = run_training(
            data_yaml=args.data, hyperparameters=hp, config_path=args.config
        )
    except RuntimeError as exc:
        # Explicit blocker: report and exit non-zero. No model is claimed.
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Training complete. Selected checkpoint (assignment-trained): {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
