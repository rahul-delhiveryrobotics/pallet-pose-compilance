"""Unit tests for calibration loading and reprojection error (task 6.1, R8).

Covers:
- Loading the synthetic ``sim-cal-v0`` artefact yields a ``SIMULATED``-labelled
  Calibration with a resolvable id and well-formed intrinsics/extrinsics
  (R8.1-R8.4, supports Property 21).
- Reprojection error is ~0 for perfectly self-consistent synthetic
  correspondences and grows monotonically with perturbation (R8.2).
- A missing calibration id degrades honestly (raises), never fabricates (R26.4).
- Synthetic vs real path provenance labelling is honest (R8.3, R26.2).
"""

from pathlib import Path

import numpy as np
import pytest

from pallet_pose_compliance.calibration import (
    Calibration,
    CalibrationError,
    CalibrationNotFoundError,
    build_calibration_from_checkerboard_and_floor,
    build_synthetic_calibration,
    compute_reprojection_error,
    load_calibration,
)
from pallet_pose_compliance.output.provenance import ProvenanceLabel

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = REPO_ROOT / "configs" / "calibration"
SIM_CAL_ID = "sim-cal-v0"


class TestLoadSyntheticArtefact:
    def test_sim_cal_resolves_and_is_simulated(self):
        cal = load_calibration(SIM_CAL_ID, calibration_dir=CALIBRATION_DIR)
        assert isinstance(cal, Calibration)
        # Resolvable id referenced by every Assessment (R8.4, Property 21).
        assert cal.id == SIM_CAL_ID
        # No physical access -> clearly labelled simulated (R26.4), never measured.
        assert cal.provenance is ProvenanceLabel.SIMULATED
        assert cal.provenance is not ProvenanceLabel.MEASURED

    def test_sim_cal_shapes_and_reprojection_error(self):
        cal = load_calibration(SIM_CAL_ID, calibration_dir=CALIBRATION_DIR)
        assert cal.intrinsics.shape == (3, 3)
        assert cal.rotation.shape == (3, 3)
        assert cal.translation.shape == (3,)
        # A stated, non-negative reprojection error is reported (R8.2).
        assert cal.reprojection_error >= 0.0
        assert np.isfinite(cal.reprojection_error)
        # Rotation is a valid rotation matrix (orthonormal, det +1).
        assert np.allclose(cal.rotation @ cal.rotation.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(cal.rotation), 1.0, atol=1e-9)
        # Extrinsic translation encodes the ~1.2 m mounting height.
        assert np.isclose(cal.translation[2], 1.2, atol=1e-6)

    def test_pipeline_calibration_id_is_resolvable(self):
        # The active calibration_id in configs/pipeline.yaml must resolve so the
        # pipeline can populate Assessment.calibration_id (Property 21 support).
        import yaml

        with (REPO_ROOT / "configs" / "pipeline.yaml").open() as fh:
            pipeline = yaml.safe_load(fh)
        cal = load_calibration(
            pipeline["calibration_id"], calibration_dir=CALIBRATION_DIR
        )
        assert cal.id == pipeline["calibration_id"]


class TestMissingIdDegradesHonestly:
    def test_missing_id_raises_not_found(self):
        with pytest.raises(CalibrationNotFoundError):
            load_calibration("no-such-cal", calibration_dir=CALIBRATION_DIR)

    def test_empty_id_raises(self):
        with pytest.raises(CalibrationError):
            load_calibration("", calibration_dir=CALIBRATION_DIR)

    def test_id_mismatch_raises(self, tmp_path):
        # An artefact whose internal id disagrees with the filename is rejected
        # rather than silently trusted.
        bad = tmp_path / "mismatch.yaml"
        bad.write_text(
            "id: something-else\n"
            "provenance: simulated\n"
            "intrinsics: [[900,0,640],[0,900,360],[0,0,1]]\n"
            "distortion: [0,0,0,0,0]\n"
            "extrinsics:\n"
            "  rotation: [[1,0,0],[0,1,0],[0,0,1]]\n"
            "  translation: [0,0,1.2]\n"
            "reprojection_error: {value: 0.3, provenance: simulated}\n"
        )
        with pytest.raises(CalibrationError):
            load_calibration("mismatch", calibration_dir=tmp_path)


class TestSyntheticBuilder:
    def test_build_synthetic_is_simulated(self):
        cal = build_synthetic_calibration("tmp-sim")
        assert cal.provenance is ProvenanceLabel.SIMULATED
        assert cal.id == "tmp-sim"
        assert cal.intrinsics[0, 0] > 0  # positive focal length

    def test_roundtrip_to_dict_reloads(self, tmp_path):
        import yaml

        cal = build_synthetic_calibration(SIM_CAL_ID)
        path = tmp_path / f"{SIM_CAL_ID}.yaml"
        path.write_text(yaml.safe_dump(cal.to_dict()))
        reloaded = load_calibration(SIM_CAL_ID, calibration_dir=tmp_path)
        assert reloaded.provenance is ProvenanceLabel.SIMULATED
        assert np.allclose(reloaded.intrinsics, cal.intrinsics)
        assert np.allclose(reloaded.rotation, cal.rotation)


class TestReprojectionError:
    def _consistent_correspondences(self):
        """Build a self-consistent (object, image) set via cv2.projectPoints."""
        import cv2

        cal = build_synthetic_calibration(SIM_CAL_ID)
        # A grid of floor points in front of the camera (Floor_Frame, Z=0).
        obj = np.array(
            [[x, y, 0.0] for x in (-0.5, 0.0, 0.5) for y in (1.0, 2.0, 3.0)],
            dtype=float,
        )
        # World->camera pose (inverse of stored camera->floor extrinsics).
        R_cw = cal.rotation.T
        t_cw = -cal.rotation.T @ cal.translation
        rvec, _ = cv2.Rodrigues(R_cw)
        tvec = t_cw.reshape(3, 1)
        img, _ = cv2.projectPoints(obj, rvec, tvec, cal.intrinsics, cal.distortion)
        return obj, img.reshape(-1, 2), cal.intrinsics, cal.distortion, rvec, tvec

    def test_zero_error_for_consistent_correspondences(self):
        obj, img, K, dist, rvec, tvec = self._consistent_correspondences()
        rms = compute_reprojection_error(obj, img, K, dist, rvec, tvec)
        assert rms == pytest.approx(0.0, abs=1e-6)

    def test_error_grows_with_perturbation(self):
        obj, img, K, dist, rvec, tvec = self._consistent_correspondences()
        rng = np.random.default_rng(0)
        small = img + rng.normal(0.0, 0.5, size=img.shape)
        large = img + rng.normal(0.0, 3.0, size=img.shape)
        rms0 = compute_reprojection_error(obj, img, K, dist, rvec, tvec)
        rms_small = compute_reprojection_error(obj, small, K, dist, rvec, tvec)
        rms_large = compute_reprojection_error(obj, large, K, dist, rvec, tvec)
        assert rms0 < rms_small < rms_large

    def test_mismatched_lengths_raise(self):
        with pytest.raises(CalibrationError):
            compute_reprojection_error(
                [[0, 0, 0], [1, 0, 0]],
                [[0, 0]],
                np.eye(3),
                np.zeros(5),
                np.zeros(3),
                np.zeros(3),
            )


class TestRealPathProvenance:
    def test_checkerboard_floor_path_is_measured(self):
        # Exercise the real calibration path with synthetic-but-consistent
        # correspondences: the number is computed from actual correspondences,
        # so it is honestly labelled MEASURED (distinct from the SIMULATED path).
        import cv2

        # Ground-truth camera used only to synthesise consistent observations.
        gt = build_synthetic_calibration("gt")
        K_gt, dist_gt = gt.intrinsics, gt.distortion
        R_cw = gt.rotation.T
        t_cw = (-gt.rotation.T @ gt.translation).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R_cw)

        # A planar checkerboard (Z=0 in a board frame) observed in a few poses.
        board = np.array(
            [[c * 0.1, r * 0.1, 0.0] for r in range(4) for c in range(4)],
            dtype=np.float32,
        )
        obj_views, img_views = [], []
        for dz in (2.0, 2.5, 3.0):
            world = board + np.array([0.0, dz, 0.0], dtype=np.float32)
            img, _ = cv2.projectPoints(world, rvec, t_cw, K_gt, dist_gt)
            obj_views.append(board.copy())
            img_views.append(img.reshape(-1, 2).astype(np.float32))

        floor_obj = np.array(
            [[x, y, 0.0] for x in (-0.5, 0.5) for y in (1.5, 2.5)], dtype=np.float32
        )
        floor_img, _ = cv2.projectPoints(floor_obj, rvec, t_cw, K_gt, dist_gt)

        cal = build_calibration_from_checkerboard_and_floor(
            "real-cal-test",
            checkerboard_object_points=obj_views,
            checkerboard_image_points=img_views,
            image_size=gt.image_size,
            floor_object_points=floor_obj,
            floor_image_points=floor_img.reshape(-1, 2),
        )
        # Computed from actual correspondences -> honestly MEASURED (not
        # SIMULATED). The magnitude depends on cv2's estimate quality; the
        # honest guarantee is a finite, non-negative RMS.
        assert cal.provenance is ProvenanceLabel.MEASURED
        assert np.isfinite(cal.reprojection_error)
        assert cal.reprojection_error >= 0.0
