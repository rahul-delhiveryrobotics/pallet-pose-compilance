# Pose sensitivity analysis (SIMULATED)

Every row below is **simulated**: a known simulated pose is rendered with a simulated true calibration, then estimated with a simulated calibration containing the listed camera-height or down-tilt error. There is no real camera measurement or real pose ground truth. The ±2 cm / ±3° values are evaluation targets, not guarantees.

- Seed: `42`; samples per condition: `16`; conditions: `12`.
- Short range: `[2.5, 3.5] m` depth; long range: `[6.0, 8.0] m` depth.
- Classification: `meets_tolerance` requires combined pass fraction over all evaluated samples ≥ 0.95; fewer than 8 samples is `insufficient_evidence`; otherwise `fails_tolerance`.
- No unevaluated height/tilt/range combinations are inferred.

## Results by evaluated condition

| error | range | total / available | radial error median / p95 (cm) | rotation median / p95 (deg) | combined pass fraction | classification |
|---|---|---:|---:|---:|---:|---|
| camera_height = 0.0 m | short | 16 / 16 | 8.882e-14 / 1.786e-13 | 7.105e-15 / 2.842e-14 | 1 | `meets_tolerance` |
| camera_height = 0.0 m | long | 16 / 16 | 2.68e-13 / 1.378e-12 | 1.421e-14 / 3.197e-14 | 1 | `meets_tolerance` |
| camera_height = 0.1 m | short | 16 / 0 | n/a / n/a | n/a / n/a | n/a | `fails_tolerance` |
| camera_height = 0.1 m | long | 16 / 0 | n/a / n/a | n/a / n/a | n/a | `fails_tolerance` |
| camera_height = 0.2 m | short | 16 / 0 | n/a / n/a | n/a / n/a | n/a | `fails_tolerance` |
| camera_height = 0.2 m | long | 16 / 0 | n/a / n/a | n/a / n/a | n/a | `fails_tolerance` |
| camera_tilt = 0.0 deg | short | 16 / 16 | 4.578e-14 / 1.459e-13 | 1.421e-14 / 2.842e-14 | 1 | `meets_tolerance` |
| camera_tilt = 0.0 deg | long | 16 / 16 | 2.231e-13 / 1.399e-12 | 1.421e-14 / 5.862e-14 | 1 | `meets_tolerance` |
| camera_tilt = 3.0 deg | short | 16 / 0 | n/a / n/a | n/a / n/a | n/a | `fails_tolerance` |
| camera_tilt = 3.0 deg | long | 16 / 0 | n/a / n/a | n/a / n/a | n/a | `fails_tolerance` |
| camera_tilt = 6.0 deg | short | 16 / 0 | n/a / n/a | n/a / n/a | n/a | `fails_tolerance` |
| camera_tilt = 6.0 deg | long | 16 / 0 | n/a / n/a | n/a / n/a | n/a | `fails_tolerance` |

## Distribution fields and interpretation

The JSON report includes separate `dx_m`, `dy_m`, `radial_m`/`radial_cm`, and `orientation_deg` distributions. Each has a `sample_count`, summary statistics, and the underlying simulated values. `available_samples` and `total_samples` are shown separately; rejected/unavailable poses count against the all-sample classification as required by the existing `pose_envelope` helper.

Envelope partition over the evaluated grid: `{'meets_tolerance': 4, 'fails_tolerance': 8, 'insufficient_evidence': 0}`. This is a classification of these simulated grid points only, not a physical operating envelope and not an extrapolation.

_Provenance: simulated. No real camera measurement was used._
