#!/usr/bin/env python3
"""Validate the trained detector on the held-out TEST split and record real metrics.

Loads weights/yolo26-det-pallet-v0.pt, runs ultralytics val on split='test'
of the merged detection dataset, and writes measured metrics (overall +
per-class) to reports/detection_metrics.json with provenance 'measured'.
"""
import json
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DATA_YAML = REPO / "data" / "merged_detection" / "data.yaml"
WEIGHTS = REPO / "weights" / "yolo26-det-pallet-v0.pt"
REPORTS = REPO / "reports"

SEED = 42


def main():
    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        project=str(REPO / "runs" / "detect"),
        name="pallet-det-v0-test",
        exist_ok=True,
        plots=False,
    )

    names = model.names  # {0: 'pallet', 1: 'pallet_pocket'}
    box = metrics.box

    per_class = {}
    # metrics.box.ap_class_index maps rows to class ids
    for i, cls_id in enumerate(box.ap_class_index):
        per_class[names[int(cls_id)]] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]),
            "mAP50-95": float(box.ap[i]),
        }

    out = {
        "provenance": "measured",
        "task": "detection",
        "note": (
            "REAL detection metrics on the merged real Roboflow data (2 classes: "
            "pallet, pallet_pocket). These are DETECTION (bounding-box) metrics, "
            "NOT pose metrics."
        ),
        "model": {
            "weights": str(WEIGHTS),
            "base": "yolo26n.pt",
            "arch": "YOLO26n (detection)",
        },
        "dataset": str(DATA_YAML),
        "split": "test",
        "imgsz": 640,
        "seed": SEED,
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        },
        "per_class": per_class,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "detection_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwritten to {REPORTS / 'detection_metrics.json'}")


if __name__ == "__main__":
    main()
