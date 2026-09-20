"""End-to-end smoke test for the minimal stub pipeline (task 3.2).

Runs the full pipeline (ingest -> detect stub -> pose stub -> SOP stub ->
verdict -> Assessment output) on one synthetic frame and asserts that exactly N
schema-valid Assessments are produced, one per detected pallet (R24.1).

The pipeline is exercised via :func:`run_pipeline` with ``write=False`` (and,
in one case, via a tmp_path so the write path is covered without polluting the
repo ``outputs/`` directory). Each Assessment is checked to be schema-valid by
round-tripping through ``serialise()``/``parse()`` and by carrying the complete
eight-rule SOP set plus the expected stub verdict/pose status.
"""

import dataclasses
import json

import pytest

from pallet_pose_compliance.detection.stub import StubDetector
from pallet_pose_compliance.ingest import make_synthetic_frame
from pallet_pose_compliance.output.provenance import (
    ProvenanceLabel,
    QualityFlag,
    ReasonCode,
)
from pallet_pose_compliance.output.schema import Assessment
from pallet_pose_compliance.pipeline import (
    load_pipeline_config,
    process_image,
    run_pipeline,
)

pytestmark = pytest.mark.e2e

# Config lives at the repository root under configs/.
_CONFIG_PATH = "configs/pipeline.yaml"


def _assert_schema_valid(assessment: Assessment) -> None:
    """Assert an Assessment is schema-valid via a serialise/parse round-trip."""
    serialised = assessment.serialise()
    # It must be valid JSON.
    payload = json.loads(serialised)
    assert isinstance(payload, dict)
    # And it must round-trip back to an equivalent object (R24.1, Property 20).
    reparsed = Assessment.parse(serialised)
    assert reparsed == assessment


def _assert_pose_honest_and_schema_valid(assessment: Assessment) -> None:
    """Assert the pose is either available OR a defined-unavailable result.

    As of task 6.10 the pose stage runs the *real* known-geometry PnP estimator
    (with Monte Carlo uncertainty) on each detection's keypoints, so the outcome
    depends on the detection geometry rather than always being a stub-unavailable
    pose. This assertion is robust to either outcome while staying strict on
    schema validity and *honesty* (no fabricated pose):

    - **available**: must carry a real position + orientation and a non-null,
      non-``UNAVAILABLE`` provenance (a measured/simulated/estimated pose).
    - **unavailable**: must be a *defined* unavailable result — a ``Reason_Code``
      present, null metric fields (never zero/placeholder), and ``UNAVAILABLE``
      provenance.
    """
    pose = assessment.pose
    assert pose.pose_status in ("available", "unavailable")

    if pose.pose_status == "available":
        # A real pose: metrics present, provenance is not the unavailable label
        # and never a silent upgrade to a fabricated value.
        assert pose.position_m is not None
        assert pose.orientation_deg is not None
        assert pose.provenance is not ProvenanceLabel.UNAVAILABLE
    else:
        # Degrade-loudly: defined unavailable pose, never a fabricated one.
        assert pose.reason_code is not None
        assert pose.position_m is None
        assert pose.orientation_deg is None
        assert pose.provenance is ProvenanceLabel.UNAVAILABLE


def _assert_stub_assessment(assessment: Assessment) -> None:
    """Assert the pipeline invariants for a single Assessment."""
    # Exactly the eight SOP-PAL-03 rules, each present once (schema enforces
    # this, but assert explicitly to document the smoke expectation).
    assert len(assessment.sop_checks) == 8
    assert sorted(c.rule_id for c in assessment.sop_checks) == list(range(1, 9))

    # The pose is honest: available with real metrics, or a defined-unavailable
    # result — never a fabricated pose.
    _assert_pose_honest_and_schema_valid(assessment)

    # The (still-stub) verdict is one of the schema-valid verdict labels. With
    # the real pose wired the exact label depends on pose availability, so we
    # assert membership rather than hard-coding MANUAL_INSPECTION.
    assert assessment.verdict.verdict in ("PASS", "FAIL", "MANUAL_INSPECTION")

    # Provenance / identifiers are populated from config (R8.4, R24.7, R28.2).
    assert assessment.schema_version
    assert assessment.model_id
    assert assessment.calibration_id
    assert assessment.source_image_ref.startswith("synthetic://")


@pytest.mark.parametrize("num_pallets", [1, 3])
def test_run_pipeline_produces_n_schema_valid_assessments(num_pallets: int) -> None:
    """The pipeline yields exactly N schema-valid Assessments for N pallets."""
    detector = StubDetector(num_pallets=num_pallets)

    assessments = run_pipeline(
        image_path=None,  # synthetic frame
        config_path=_CONFIG_PATH,
        detector=detector,
        write=False,
    )

    # Exactly one Assessment per detected pallet (R24.1).
    assert len(assessments) == num_pallets

    # Pallet ids are the distinct per-index ids.
    assert [a.pallet_id for a in assessments] == [
        f"pallet-{i}" for i in range(num_pallets)
    ]

    for assessment in assessments:
        assert isinstance(assessment, Assessment)
        _assert_stub_assessment(assessment)
        _assert_schema_valid(assessment)


def test_run_pipeline_zero_pallets_produces_no_assessments() -> None:
    """A detector finding no pallets yields zero Assessments (valid empty)."""
    detector = StubDetector(num_pallets=0)

    assessments = run_pipeline(
        image_path=None,
        config_path=_CONFIG_PATH,
        detector=detector,
        write=False,
    )

    assert assessments == []


def test_run_pipeline_write_path_writes_one_file_per_pallet(tmp_path) -> None:
    """The write path emits one schema-valid JSON file per detected pallet.

    Uses a tmp config pointing outputs at ``tmp_path`` so the repo ``outputs/``
    directory is not polluted.
    """
    num_pallets = 2

    # Build a temp config that redirects outputs_dir to tmp_path. outputs_dir is
    # resolved relative to the config file's parent's parent, so nest the config
    # two directories deep and set paths.outputs_dir accordingly.
    base_cfg = load_pipeline_config(_CONFIG_PATH)
    outputs_dir = tmp_path / "outputs"
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "pipeline.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"seed: {base_cfg.seed}",
                f'model_id: "{base_cfg.model_id}"',
                f'calibration_id: "{base_cfg.calibration_id}"',
                f"input_resolution: {base_cfg.input_resolution}",
                f'schema_version: "{base_cfg.schema_version}"',
                "paths:",
                '  outputs_dir: "outputs"',
                # The SOP thresholds config is required by the real analyzer
                # (task 9.10). This temp config lives under tmp_path, so point
                # at the repo's real config via its absolute path (an absolute
                # right-hand path wins in ``base_dir / path``) so the boundaries
                # load rather than 404ing against tmp_path/configs.
                f'  sop_thresholds_path: "{base_cfg.sop_thresholds_path}"',
                # Likewise the verdict config is required by the real
                # aggregator (task 10.1); point at the repo's real config via
                # its absolute path so it loads rather than 404ing against
                # tmp_path/configs.
                f'  verdict_config_path: "{base_cfg.verdict_config_path}"',
            ]
        ),
        encoding="utf-8",
    )

    detector = StubDetector(num_pallets=num_pallets)
    assessments = run_pipeline(
        image_path=None,
        config_path=str(cfg_path),
        detector=detector,
        write=True,
    )

    assert len(assessments) == num_pallets

    written = sorted(outputs_dir.glob("*.json"))
    assert len(written) == num_pallets

    # Each written file is a schema-valid Assessment that matches what was
    # returned in memory.
    returned_by_id = {a.pallet_id: a for a in assessments}
    for path in written:
        parsed = Assessment.parse(path.read_text(encoding="utf-8"))
        assert parsed == returned_by_id[parsed.pallet_id]


def test_assessments_carry_model_and_calibration_ids_from_config() -> None:
    """Every Assessment records the config's model_id and calibration_id.

    Every reported result MUST reference the detector ``model_id`` and the
    active ``calibration_id`` for traceability/reproducibility (R8.4, R24.7,
    R28.2). Assert the ids on each Assessment match the loaded config exactly.
    """
    config = load_pipeline_config(_CONFIG_PATH)
    num_pallets = 2
    detector = StubDetector(num_pallets=num_pallets)

    assessments = run_pipeline(
        image_path=None,
        config_path=_CONFIG_PATH,
        detector=detector,
        write=False,
    )

    assert len(assessments) == num_pallets
    for assessment in assessments:
        assert assessment.model_id == config.model_id
        assert assessment.calibration_id == config.calibration_id
        # These must be the real configured ids, not empty placeholders.
        assert assessment.model_id
        assert assessment.calibration_id


def test_bad_calibration_id_degrades_pose_to_unavailable_without_crashing() -> None:
    """An unloadable calibration degrades pose to CALIBRATION_INVALID (honestly).

    Rather than crashing the run or fabricating a calibration/pose, a
    ``calibration_id`` with no artefact must yield a *defined* unavailable pose
    (``CALIBRATION_INVALID`` + matching quality flag, null metrics, UNAVAILABLE
    provenance) and raise the Assessment failure signal, while still recording
    the configured ids (R26.3/R26.4, R22.1/R22.3).
    """
    base = load_pipeline_config(_CONFIG_PATH)
    bad_config = dataclasses.replace(
        base, calibration_id="does-not-exist-calibration"
    )

    frame = make_synthetic_frame(
        width=base.input_resolution, height=base.input_resolution
    )
    detector = StubDetector(num_pallets=1)

    # Must NOT raise even though the calibration cannot be loaded.
    assessments = process_image(frame, bad_config, detector=detector)
    assert len(assessments) == 1
    assessment = assessments[0]

    pose = assessment.pose
    assert pose.pose_status == "unavailable"
    assert pose.reason_code is ReasonCode.CALIBRATION_INVALID
    assert QualityFlag.CALIBRATION_INVALID in pose.quality_flags
    assert pose.position_m is None
    assert pose.orientation_deg is None
    assert pose.provenance is ProvenanceLabel.UNAVAILABLE

    # The processing blocker is surfaced honestly as a failure signal.
    assert assessment.failure.failed is True
    assert assessment.failure.reason_code is ReasonCode.CALIBRATION_INVALID

    # Ids are still populated (bad calibration id is still recorded honestly).
    assert assessment.model_id == bad_config.model_id
    assert assessment.calibration_id == "does-not-exist-calibration"

    # And the Assessment remains schema-valid.
    _assert_schema_valid(assessment)
