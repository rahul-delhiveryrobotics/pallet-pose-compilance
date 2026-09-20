#!/usr/bin/env bash
#
# run.sh — one entry point for the Pallet Pose + SOP Compliance project.
#
# Usage:
#   ./run.sh <command> [extra args...]
#
# Commands:
#   demo         Live webcam demo (detect pallets + pose + verdict on video)
#   images       Slideshow demo over the real pallet test images (RELIABLE:
#                guaranteed to show detection + pose + verdict)
#   image FILE   Run the pipeline on ONE image file and print the Assessment
#   detect       Live webcam demo, detection boxes only (cleanest live view)
#   record       Live webcam demo AND record it to outputs/demo.mp4
#   test         Run the full test suite
#   reports      Show where the measured/simulated result reports live
#   help         Show this help
#
# Everything runs from the project root; PYTHONPATH is set to src/ for you.

set -euo pipefail

# Resolve the project root (directory of this script) so it works from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="src"

CMD="${1:-help}"
shift || true

case "$CMD" in
  demo)
    echo "[run] Live webcam demo — point the camera at a pallet (or a pallet photo on your phone)."
    echo "[run] Keys: q/ESC quit, s snapshot. If the wrong camera opens, try: ./run.sh demo --source 1"
    python3 scripts/live_demo.py "$@"
    ;;

  images)
    DIR="${1:-data/synthetic_pose/images/test}"
    echo "[run] Image slideshow demo over: $DIR"
    echo "[run] Keys: any key = next image, s = save snapshot, q/ESC = quit."
    python3 scripts/live_demo.py --images "$DIR"
    ;;

  image)
    if [ "$#" -lt 1 ]; then
      echo "usage: ./run.sh image <path-to-image>"; exit 1
    fi
    echo "[run] Running the full pipeline on: $1"
    python3 -m pallet_pose_compliance.cli "$1" --json
    ;;

  detect)
    echo "[run] Live webcam demo (detection only — cleanest view)."
    python3 scripts/live_demo.py --detect-only "$@"
    ;;

  record)
    OUT="${1:-outputs/demo.mp4}"
    echo "[run] Live webcam demo, recording to: $OUT"
    mkdir -p "$(dirname "$OUT")"
    python3 scripts/live_demo.py --save "$OUT"
    echo "[run] Saved recording to $OUT"
    ;;

  test)
    echo "[run] Running the full test suite..."
    python3 -m pytest -q
    ;;

  reports)
    echo "[run] Result reports (measured = real data, simulated = synthetic):"
    ls -1 reports/*.json reports/*.md 2>/dev/null || echo "  (none yet)"
    echo
    echo "  detection (REAL data):   reports/detection_metrics.json"
    echo "  pose (SIMULATED):        reports/pose_eval_metrics.json + reports/pose_eval_notes.md"
    echo "  generated Assessments:   outputs/"
    ;;

  help|*)
    sed -n '3,20p' "$ROOT/run.sh" | sed 's/^#\s\{0,1\}//'
    ;;
esac
