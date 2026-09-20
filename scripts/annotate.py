"""Annotation tooling CLI (R3).

Thin command-line wrapper over
:mod:`pallet_pose_compliance.annotation`, which holds the reusable annotation
logic. Produces / validates COCO-keypoints-style JSON labels for pallet deck
corners (four bottom + four top corners, visibility ∈ {visible, occluded,
absent}). The behaviour here mirrors the labelling guideline documented in
:data:`pallet_pose_compliance.annotation.LABELLING_GUIDELINE` (R3.2, R3.3).

Subcommands
-----------
- ``guideline``  Print the labelling guideline (the one-page guideline that
  DATASET.md mirrors, R3.3).
- ``template``   Write an empty COCO-keypoints-style annotation document
  (categories + keypoint order populated) to a file, ready to be filled in.
- ``validate``   Read an annotation file and round-trip it through the schema,
  reporting image/annotation counts. Fails non-zero on a malformed file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly (``python3 scripts/annotate.py``) from a source tree
# without installation, mirroring the src-layout used by the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pallet_pose_compliance.annotation import (  # noqa: E402
    LABELLING_GUIDELINE,
    AnnotationSet,
    read_annotations,
    write_annotations,
)


def _cmd_guideline(_args: argparse.Namespace) -> int:
    print(LABELLING_GUIDELINE)
    return 0


def _cmd_template(args: argparse.Namespace) -> int:
    # An empty set still writes the categories/keypoint-order metadata so a
    # labeller (or downstream tool) sees the exact expected schema.
    out = write_annotations(AnnotationSet(), args.output)
    print(f"wrote empty annotation template -> {out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    aset = read_annotations(args.input)
    # Round-trip to confirm the file conforms to the documented schema.
    reparsed = AnnotationSet.from_coco_dict(aset.to_coco_dict())
    ok = reparsed == aset
    print(
        f"validated {args.input}: {len(aset.images)} image(s), "
        f"{len(aset.annotations)} annotation(s); round_trip={'ok' if ok else 'FAILED'}"
    )
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotate",
        description=(
            "Pallet deck-corner annotation tooling (COCO-keypoints-style JSON)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_guideline = sub.add_parser(
        "guideline", help="print the labelling guideline"
    )
    p_guideline.set_defaults(func=_cmd_guideline)

    p_template = sub.add_parser(
        "template", help="write an empty annotation document to a file"
    )
    p_template.add_argument("output", help="output JSON path")
    p_template.set_defaults(func=_cmd_template)

    p_validate = sub.add_parser(
        "validate", help="validate/round-trip an annotation file"
    )
    p_validate.add_argument("input", help="input JSON path")
    p_validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
