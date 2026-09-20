"""Real YOLO-pose detector wrapper with weights provenance discipline (R4, R5.5, R26.3).

This module wraps an Ultralytics YOLO-pose model behind the same
``detect(image) -> list[Detection]`` interface as the stub detector
(:mod:`pallet_pose_compliance.detection.stub`), converting YOLO-pose outputs
into the system's :class:`~pallet_pose_compliance.output.schema.Detection`
schema (box + score + eight named keypoints with visibility + score).

Design anchors
--------------
- **Model choice** (design: Detector, R4.2): Ultralytics YOLO-family pose model
  (YOLO11-pose / YOLOv8-pose) — open-source, jointly predicts boxes +
  keypoints, exportable to ONNX/TensorRT. The architecture/source is cited in
  README/DATASET.
- **Keypoint schema**: eight structural pallet corners — four bottom deck
  corners (floor-contact, pose reference) + four top deck corners — in the
  fixed order :data:`KEYPOINT_NAMES`, matching the annotation tooling and stub.
- **Weights provenance / honesty** (R5.5, R26.3): the wrapper tracks whether it
  is running **assignment-trained** weights (eligible to back ``measured``
  downstream metrics) versus a **stock/pretrained** checkpoint (must be labelled
  ``estimated`` and flagged *not-assignment-trained*). A stock checkpoint is
  **never** mislabelled as assignment-trained. Downstream metric reporting
  (task 4.6) consults :attr:`YoloPoseDetector.weights_provenance` /
  :attr:`YoloPoseDetector.is_assignment_trained` to honour this.
- **Defensive import** (design: scope posture): Ultralytics/torch may not be
  installed in every environment, so ``ultralytics`` is imported lazily inside
  :meth:`YoloPoseDetector.load` — this module (and its output-conversion /
  provenance logic) imports and unit-tests without ultralytics present. When
  the library or the weights are unavailable, the wrapper degrades **honestly**
  by raising :class:`DetectorUnavailableError` rather than fabricating a model.

The output-conversion and provenance logic is deliberately decoupled from the
Ultralytics runtime (see :func:`detections_from_yolo_result`) so it can be
exercised with a lightweight fake result object in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..output.provenance import ProvenanceLabel
from ..output.schema import Detection, Keypoint

__all__ = [
    "KEYPOINT_NAMES",
    "WeightsProvenance",
    "DetectorUnavailableError",
    "detections_from_yolo_result",
    "YoloPoseDetector",
]

# The eight pallet structural keypoints in the fixed, documented order shared
# with the annotation tooling and the stub detector: four bottom deck corners
# (floor-contact, pose reference) then four top deck corners.
KEYPOINT_NAMES: tuple[str, ...] = (
    "bottom_corner_0",
    "bottom_corner_1",
    "bottom_corner_2",
    "bottom_corner_3",
    "top_corner_0",
    "top_corner_1",
    "top_corner_2",
    "top_corner_3",
)

# Keypoint score at/above which a corner is considered directly ``visible``;
# below it (but present) the corner is treated as ``occluded`` (its pixel is
# still known/inferable). This mirrors the annotation visibility convention.
_VISIBLE_SCORE_THRESHOLD = 0.5

# Class-id -> class-name map for the pallet detection task. YOLO emits integer
# class ids; the pallet model is trained with pallet=0, box=1.
_CLASS_NAMES: dict[int, str] = {0: "pallet", 1: "box"}


@dataclass(frozen=True)
class WeightsProvenance:
    """Provenance of the detector weights (R5.5, R26.3).

    This is the honesty record that lets downstream metric reporting decide
    whether detector-derived numbers may carry a ``measured`` provenance label.

    Attributes
    ----------
    is_assignment_trained:
        ``True`` only when the loaded weights were trained for *this assignment*
        (via ``scripts/train.py``). A stock/pretrained checkpoint is always
        ``False`` and must never be reported as assignment-trained.
    provenance:
        The :class:`ProvenanceLabel` any detector-derived result inherits:
        ``MEASURED``-eligible only for assignment-trained weights; a stock
        checkpoint yields ``ESTIMATED`` (reasoning/placeholder, not a measured
        result on the assignment model).
    weights_ref:
        A human-readable reference to the weights artefact (path or model id).
    note:
        A short human-readable honesty note describing the weights origin.
    """

    is_assignment_trained: bool
    provenance: ProvenanceLabel
    weights_ref: str
    note: str

    def __post_init__(self) -> None:
        # Honesty invariant: a stock checkpoint may NEVER be labelled
        # assignment-trained, and assignment-trained weights may never be
        # downgraded below ``measured`` eligibility silently. Enforce the
        # two consistent combinations only.
        if self.is_assignment_trained:
            if self.provenance is ProvenanceLabel.UNAVAILABLE:
                raise ValueError(
                    "assignment-trained weights cannot carry 'unavailable' "
                    "provenance"
                )
        else:
            # A checkpoint that is not assignment-trained must be surfaced as
            # 'estimated' (or 'unavailable') — never 'measured' (R5.5, R26.3).
            if self.provenance is ProvenanceLabel.MEASURED:
                raise ValueError(
                    "a stock/pretrained checkpoint must not be labelled "
                    "'measured'; label it 'estimated' and flag it as "
                    "not-assignment-trained (R5.5, R26.3)"
                )

    @classmethod
    def assignment_trained(cls, weights_ref: str) -> "WeightsProvenance":
        """Provenance for weights trained for this assignment (measured-eligible)."""
        return cls(
            is_assignment_trained=True,
            provenance=ProvenanceLabel.MEASURED,
            weights_ref=weights_ref,
            note=(
                "Weights trained for this assignment via scripts/train.py; "
                "detector-derived metrics may be reported as 'measured'."
            ),
        )

    @classmethod
    def stock(cls, weights_ref: str) -> "WeightsProvenance":
        """Provenance for a stock/pretrained checkpoint (estimated, not trained)."""
        return cls(
            is_assignment_trained=False,
            provenance=ProvenanceLabel.ESTIMATED,
            weights_ref=weights_ref,
            note=(
                "Stock/pretrained checkpoint NOT trained for this assignment; "
                "flagged not-assignment-trained. Any inference is 'estimated' "
                "and must never be reported as assignment-trained (R5.5, R26.3)."
            ),
        )


class DetectorUnavailableError(RuntimeError):
    """Raised when the detector cannot be constructed/loaded (honest degrade).

    Signals that Ultralytics is not installed or the weights artefact could not
    be resolved. Callers (the pipeline / CLI) surface this as an explicit
    blocker rather than fabricating detections (design: training-blocked
    fallback, R26.3/R26.4).
    """


def _visibility_from_score(score: Optional[float]) -> str:
    """Map a keypoint confidence to a visibility label.

    ``score >= threshold`` -> ``visible``; a present but low score ->
    ``occluded`` (location known/inferable). A missing score (``None``) means
    the model does not emit per-keypoint confidence, so the keypoint's pixel is
    present and it is treated as ``visible`` (unscored) rather than fabricating
    an ``absent`` label.
    """
    if score is None:
        return "visible"
    return "visible" if score >= _VISIBLE_SCORE_THRESHOLD else "occluded"


def _keypoints_from_arrays(
    xy: Sequence[Sequence[float]],
    scores: Optional[Sequence[float]],
) -> list[Keypoint]:
    """Build the eight named :class:`Keypoint`\\ s from YOLO keypoint arrays.

    Parameters
    ----------
    xy:
        Sequence of ``(u, v)`` pixel coordinates, one per keypoint, in the
        canonical :data:`KEYPOINT_NAMES` order.
    scores:
        Optional per-keypoint confidence in ``[0, 1]``; ``None`` means the
        model does not emit per-keypoint scores (all treated as visible).

    Raises
    ------
    ValueError
        If the number of keypoints does not match the schema (eight corners).
    """
    if len(xy) != len(KEYPOINT_NAMES):
        raise ValueError(
            f"expected {len(KEYPOINT_NAMES)} keypoints (one per pallet corner), "
            f"got {len(xy)}"
        )
    keypoints: list[Keypoint] = []
    for i, name in enumerate(KEYPOINT_NAMES):
        u, v = float(xy[i][0]), float(xy[i][1])
        score = None if scores is None else float(scores[i])
        visibility = _visibility_from_score(score)
        if visibility == "absent":
            # Null discipline: an absent corner carries no fabricated pixels.
            keypoints.append(
                Keypoint(name=name, u=None, v=None, visibility="absent", score=None)
            )
        else:
            keypoints.append(
                Keypoint(
                    name=name,
                    u=u,
                    v=v,
                    visibility=visibility,
                    score=None if score is None else max(0.0, min(1.0, score)),
                )
            )
    return keypoints


def detections_from_yolo_result(result: Any) -> list[Detection]:
    """Convert one Ultralytics YOLO-pose ``Results`` object into ``Detection``\\ s.

    This is the pure output-conversion boundary and is deliberately independent
    of the Ultralytics runtime: it only reads the numeric arrays exposed by a
    ``Results`` object (``boxes.xywh``, ``boxes.conf``, ``boxes.cls``,
    ``keypoints.xy``, ``keypoints.conf``). Tests exercise it with a lightweight
    fake result object, so no ultralytics/torch install is required.

    Parameters
    ----------
    result:
        An object exposing ``boxes`` and ``keypoints`` attributes shaped like
        Ultralytics results. Missing keypoints (``keypoints is None``) yield
        detections with no keypoints.

    Returns
    -------
    list[Detection]
        One :class:`Detection` per detected object, keypoints converted to the
        eight named pallet corners with visibility + score.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xywh = np.asarray(_to_numpy(boxes.xywh), dtype=float).reshape(-1, 4)
    conf = np.asarray(_to_numpy(boxes.conf), dtype=float).reshape(-1)
    cls_ids = np.asarray(_to_numpy(boxes.cls), dtype=float).reshape(-1).astype(int)

    keypoints_obj = getattr(result, "keypoints", None)
    kpts_xy = None
    kpts_conf = None
    if keypoints_obj is not None:
        kpts_xy = np.asarray(_to_numpy(keypoints_obj.xy), dtype=float)
        # Shape: (num_objects, num_keypoints, 2)
        kpts_xy = kpts_xy.reshape(kpts_xy.shape[0], -1, 2) if kpts_xy.size else kpts_xy
        raw_conf = getattr(keypoints_obj, "conf", None)
        if raw_conf is not None:
            kpts_conf = np.asarray(_to_numpy(raw_conf), dtype=float)
            kpts_conf = (
                kpts_conf.reshape(kpts_conf.shape[0], -1)
                if kpts_conf.size
                else kpts_conf
            )

    detections: list[Detection] = []
    for i in range(xywh.shape[0]):
        cx, cy, w, h = (float(v) for v in xywh[i])
        # Convert YOLO centre-format xywh to the schema's top-left [x, y, w, h].
        bbox = [cx - w / 2.0, cy - h / 2.0, w, h]
        cls_name = _CLASS_NAMES.get(int(cls_ids[i]), "pallet")
        score = max(0.0, min(1.0, float(conf[i])))

        keypoints: list[Keypoint] = []
        if kpts_xy is not None and i < kpts_xy.shape[0] and kpts_xy.size:
            per_obj_scores = (
                kpts_conf[i]
                if kpts_conf is not None and i < kpts_conf.shape[0]
                else None
            )
            keypoints = _keypoints_from_arrays(kpts_xy[i], per_obj_scores)

        detections.append(
            Detection(cls=cls_name, bbox=bbox, score=score, keypoints=keypoints)
        )
    return detections


def _to_numpy(obj: Any) -> Any:
    """Best-effort conversion of a tensor/array-like to a numpy array.

    Supports torch tensors (``.cpu().numpy()``), objects exposing ``.numpy()``,
    and anything :func:`numpy.asarray` accepts. Kept tiny so the conversion
    path works without torch installed (tests pass numpy arrays / lists).
    """
    if obj is None:
        return np.empty((0,))
    cpu = getattr(obj, "cpu", None)
    if callable(cpu):
        obj = cpu()
    to_numpy = getattr(obj, "numpy", None)
    if callable(to_numpy):
        return to_numpy()
    return np.asarray(obj)


class YoloPoseDetector:
    """YOLO-pose detector wrapper exposing ``detect(image) -> list[Detection]``.

    The Ultralytics model is loaded lazily (see :meth:`load`) so importing this
    module never requires ultralytics/torch. Construct with either an explicit
    ``weights_path`` (an assignment-trained checkpoint) or a ``stock_model_id``
    (a stock/pretrained checkpoint), which sets :attr:`weights_provenance`
    accordingly and honestly.

    Parameters
    ----------
    weights_path:
        Path to an assignment-trained weights artefact (``.pt``). When present
        and existing, the detector is flagged assignment-trained
        (``measured``-eligible downstream).
    stock_model_id:
        Identifier of a stock/pretrained checkpoint (e.g. ``"yolo11n-pose.pt"``)
        used only when no assignment-trained weights exist. Any inference from
        it is ``estimated`` and flagged not-assignment-trained.
    imgsz:
        Inference input resolution (square), from ``pipeline.yaml``.
    conf:
        Detection confidence threshold passed to the model.
    """

    def __init__(
        self,
        *,
        weights_path: Optional["str | Path"] = None,
        stock_model_id: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.25,
    ) -> None:
        if weights_path is None and stock_model_id is None:
            raise ValueError(
                "provide either weights_path (assignment-trained) or "
                "stock_model_id (stock/pretrained fallback)"
            )
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.stock_model_id = stock_model_id
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self._model: Any = None

        # Resolve weights provenance honestly at construction time. An
        # assignment-trained weights path takes precedence; otherwise the
        # detector is running a stock checkpoint and is flagged as such.
        if self.weights_path is not None:
            self.weights_provenance = WeightsProvenance.assignment_trained(
                str(self.weights_path)
            )
        else:
            self.weights_provenance = WeightsProvenance.stock(str(self.stock_model_id))

    # -- honesty accessors -------------------------------------------------

    @property
    def is_assignment_trained(self) -> bool:
        """Whether the loaded weights were trained for this assignment (R5.5)."""
        return self.weights_provenance.is_assignment_trained

    @property
    def provenance(self) -> ProvenanceLabel:
        """The provenance label detector-derived results inherit (R5.5, R26.3)."""
        return self.weights_provenance.provenance

    # -- model loading -----------------------------------------------------

    def load(self) -> None:
        """Load the Ultralytics YOLO-pose model (imported lazily/defensively).

        Raises
        ------
        DetectorUnavailableError
            If ultralytics is not installed or the weights artefact cannot be
            resolved. This is an *honest* degrade: no model is fabricated.
        """
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on env
            raise DetectorUnavailableError(
                "ultralytics is not installed; cannot load the YOLO-pose "
                "detector. Install the declared dependency or run the "
                "training-blocked fallback. No detections are fabricated."
            ) from exc

        if self.weights_path is not None:
            if not self.weights_path.exists():
                raise DetectorUnavailableError(
                    f"assignment-trained weights not found at "
                    f"{self.weights_path!s}; run scripts/train.py or provide a "
                    "valid weights artefact. No detections are fabricated."
                )
            weights_ref: str = str(self.weights_path)
        else:
            weights_ref = str(self.stock_model_id)

        try:
            self._model = YOLO(weights_ref)
        except Exception as exc:  # pragma: no cover - depends on env/weights
            raise DetectorUnavailableError(
                f"failed to load YOLO-pose weights {weights_ref!r}: {exc}"
            ) from exc

    # -- inference ---------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on ``image`` and return schema ``Detection``\\ s.

        Loads the model on first use. Converts the first YOLO ``Results`` object
        (single image) into the system's :class:`Detection` schema.

        Raises
        ------
        DetectorUnavailableError
            If the model/weights are unavailable (honest degrade).
        ValueError
            If ``image`` is not an ``H x W[x3]`` array.
        """
        if image is None or getattr(image, "ndim", 0) < 2:
            raise ValueError("detect requires an H x W[x3] image")
        self.load()
        results = self._model.predict(
            image, imgsz=self.imgsz, conf=self.conf, verbose=False
        )
        if not results:
            return []
        return detections_from_yolo_result(results[0])
