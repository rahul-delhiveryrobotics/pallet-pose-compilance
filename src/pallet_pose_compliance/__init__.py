"""Pallet Pose Estimation & Load Compliance package.

Top-level package for the SOP-PAL-03 pallet pose and load compliance system.
Subpackages map to the design's component boundaries:

- detection:   Detector wrapper + detection/localisation metrics
- geometry:    keypoint schema, coordinate frames, PnP, ray-plane pose
- calibration: intrinsics/extrinsics loading + reprojection error
- sop:         eight-rule triage table + implemented SOP checks
- tracking:    temporal persistence + circular statistics
- output:      Assessment schema, provenance, serialisation
- verdict:     per-pallet verdict aggregation
"""

__version__ = "0.1.0"
