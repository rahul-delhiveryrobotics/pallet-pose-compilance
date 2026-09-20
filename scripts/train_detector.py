#!/usr/bin/env python3
"""Train a QUICK YOLO detector on the merged pallet/pocket detection dataset.

Base weights: yolo26n.pt (detection). Falls back to yolo11n.pt if yolo26n
cannot be loaded/downloaded (substitution is recorded in the run + notes).

Quick settings: imgsz=640, epochs=15, batch=16, device=0, seed=42, deterministic.
Full train/valid/test are used (no subsampling); the merged train set (~9.1k
images) trains in a reasonable time on the RTX 4080.

After training, copies best.pt to weights/yolo26-det-pallet-v0.pt.
"""
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DATA_YAML = REPO / "data" / "merged_detection" / "data.yaml"
WEIGHTS_DIR = REPO / "weights"
REPORTS = REPO / "reports"
PROJECT = REPO / "runs" / "detect"
NAME = "pallet-det-v0"

SEED = 42
EPOCHS = 15
IMGSZ = 640
BATCH = 16


def load_base_model():
    """Try yolo26n.pt, fall back to yolo11n.pt. Return (model, used_weights)."""
    for candidate in ["yolo26n.pt", "yolo11n.pt"]:
        try:
            m = YOLO(candidate)
            print(f"[train] loaded base weights: {candidate}")
            return m, candidate
        except Exception as e:  # noqa: BLE001
            print(f"[train] failed to load {candidate}: {e}")
    raise RuntimeError("Could not load any base detection weights")


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    model, base_weights = load_base_model()

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        seed=SEED,
        deterministic=True,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
    )

    run_dir = PROJECT / NAME
    best = run_dir / "weights" / "best.pt"
    dest = WEIGHTS_DIR / "yolo26-det-pallet-v0.pt"
    shutil.copy2(best, dest)
    print(f"[train] copied {best} -> {dest}")

    # record which base weights were actually used
    (REPORTS / "detection_train_meta.json").write_text(
        json.dumps(
            {
                "base_weights_used": base_weights,
                "requested_base_weights": "yolo26n.pt",
                "substituted": base_weights != "yolo26n.pt",
                "epochs": EPOCHS,
                "imgsz": IMGSZ,
                "batch": BATCH,
                "seed": SEED,
                "deterministic": True,
                "device": 0,
                "best_weights": str(best),
                "exported_weights": str(dest),
            },
            indent=2,
        )
    )
    print(f"[train] base_weights_used={base_weights}")


if __name__ == "__main__":
    main()
