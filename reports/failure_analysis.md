# Representative failure analysis

These are exactly three **representative failure cases**, not the three statistically worst cases. The repository does not contain a ranked real-world failure ledger, and the selected source images are simulated test images. The annotations are visual provenance markers only; no detector output was fabricated or asserted from the pixels.

## Case 1 — representative keypoint/localisation gap

- **Image path:** `outputs/failure_cases/representative_01_simulated_keypoint_localisation_gap.png`
- **Observed symptom:** A visually plausible pallet render is presented with a failure-case marker; downstream pose/localisation can be sensitive to small corner/keypoint placement errors.
- **Root cause:** The pose inverse problem uses eight named corners and known geometry. Pixel localisation error is amplified by perspective and range, and the real detection corpus does not provide matching eight-corner labels. This is a representative failure mode, not a measured diagnosis for this individual image.
- **Provenance:** Source `data/synthetic_pose/images/test/pallet_002550.png`; simulated render and simulated labels. Output image adds only a clearly labelled banner.
- **Mitigation:** Add real eight-corner annotations with visibility, measure keypoint pixel error by visibility/range, use the existing Monte Carlo uncertainty gate, and send unavailable/high-uncertainty poses to `MANUAL_INSPECTION`.

## Case 2 — representative pose uncertainty/unavailability

- **Image path:** `outputs/failure_cases/representative_02_simulated_pose_uncertainty.png`
- **Observed symptom:** The case is marked as representative of an end-to-end pose-unavailable or pose-uncertain outcome; no numeric pose is claimed for this image in this report.
- **Root cause:** The existing estimator deliberately degrades when keypoints are insufficient, geometry is ill-conditioned, calibration is invalid, or propagated uncertainty exceeds its documented gate. The measured pose evaluation reports 76 unavailable poses out of 450 simulated test samples, but that aggregate does not identify this selected image as one of them.
- **Provenance:** Source `data/synthetic_pose/images/test/pallet_002551.png`; simulated render. The banner is generated locally; no real camera measurement is used.
- **Mitigation:** Preserve explicit null pose fields and reason codes, improve keypoint coverage/visibility labels, collect measured calibration and floor-pose ground truth, and retain the verdict pose-quality gate.

## Case 3 — representative visibility/domain gap

- **Image path:** `outputs/failure_cases/representative_03_simulated_domain_or_visibility_gap.png`
- **Observed symptom:** The image is a representative example of a condition where synthetic appearance, visibility, occlusion, or real-scene domain differences can make localisation and SOP evidence incomplete.
- **Root cause:** Pose training uses deliberately simple simulated renders, while the real detector data contains bounding boxes (including polygon-to-axis-aligned-box conversions), not eight-corner pose labels. A single-side camera also leaves rear/interior regions hidden. These are documented system limitations, not a claim that this selected image was statistically worst.
- **Provenance:** Source `data/synthetic_pose/images/test/pallet_002552.png`; simulated render. No real demonstration-photo label or ground truth is attached.
- **Mitigation:** Expand licence-confirmed target-domain imagery, annotate occluded/absent corners rather than inventing them, add grouped scene/session holdouts, and keep hidden SOP rules unresolved or route uncertain cases to manual inspection.

No case above is evidence of real-world compliance. Real failure ranking requires a labelled real evaluation set, measured calibration, and measured pose ground truth, none of which is present here.
