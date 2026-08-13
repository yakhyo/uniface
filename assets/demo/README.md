# Demo set

Photographs and rendered figures covering every component that a still image can show.
`assets/source/` holds only the photographs the figures read; this folder holds the figures. Rebuild with:

```bash
python3 tools/demo/build_demos.py assets
```

46 source photographs (16 MB), 20 figures (5.9 MB). Every number below is measured by that script,
not quoted from a paper. Rerun it after changing a source and update this file from its output.

## Naming

One pattern for every source: **`<task>_<variant>.jpg`**, task first. A trailing `2` marks a second
set of subjects for the same task (`state_mask` and `state_b_mask` are different people). Verification
uses `verify_<name>_<year>`, because a figure needs several photographs of one person and the
name is what tells you which.

## Source photographs, by task

| Source                                       | Feeds                 | Notes                                                                                                  |
| -------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------ |
| `detect_group.jpg`                         | detection, quality    | 7 faces                                                                                                |
| `detect_crowd.jpg`                         | detection_alt         | 29 faces, 37–46px each                                                                                |
| `anon_group.jpg`                           | anonymization         | 5 faces                                                                                                |
| `landmarks_face.jpg`                       | landmarks             |                                                                                                        |
| `mesh_face.jpg`                            | face_mesh             |                                                                                                        |
| `parse_face.jpg`                           | parsing               | 13 of 19 classes present                                                                               |
| `seg_face.jpg`                             | segmentation          |                                                                                                        |
| `seg_occluded.jpg`                         | segmentation_occluded | headscarf, so only the exposed region masks                                                            |
| `matte_face.jpg`                           | matting               |                                                                                                        |
| `matte_hair.jpg`                           | matting_alt           | flyaway hair against a plain background                                                                |
| `pose_left/center/right.jpg`               | headpose              | yaw −78° / +9° / +40°                                                                              |
| `gaze_away/averted/right.jpg`              | gaze                  | yaw −30° / −19° / +23°, so not left/centre/right                                                  |
| `age_child/adult/middle/senior.jpg`        | demography            | 3-9, 20-29, 40-49, 60-69; sorted by prediction, not filename                                           |
| `emotion_*.jpg` (8)                        | emotion               | one per AffectNet-8 class                                                                              |
| `state_closed/glasses/sunglasses/mask.jpg` | face_states           | `emotion_happy` fills the eyes-open slot                                                             |
| `state_b_glasses/sunglasses/mask.jpg`      | face_states_alt       | second set;`age_adult` and `mesh_face` fill the two accessory-free slots                           |
| `spoof_live/print/screen.jpg`              | spoofing              | a live capture and two replays of it                                                                   |
| `verify_now_2010/2014/2024.jpg`            | verification          | one living subject, three dates, left unnamed                                                          |
| `verify_einstein_1921/1947.jpg`            | verification_alt      | `_1947` is also the `tests/test_blazeface.py` fixture and notebook 04's query, so do not remove it |
| `verify_curie.jpg`                         | verification_alt      | the unpaired negative; year not recorded                                                               |
| `verify_bohr_1910/1935.jpg`                | verification_alt      | second identity, 25 years apart                                                                        |

Missing names are skipped with a warning rather than failing the run, so a partial set still builds.

## Figures

| File                      | Model                | Measured                                                                  |
| ------------------------- | -------------------- | ------------------------------------------------------------------------- |
| detection.jpg             | SCRFD-10G            | 7 faces                                                                   |
| detection_alt.jpg         | SCRFD-10G            | 29 faces, 37–46px wide, weakest score 0.73                               |
| landmarks.jpg             | 2d106det, PIPNet     | 106 / 98 / 68 points                                                      |
| face_mesh.jpg             | MediaPipe            | 468 and 478 points; landmarks above, 2556-edge tessellation below         |
| parsing.jpg               | BiSeNet ResNet-34    | 13 of 19 classes present                                                  |
| segmentation.jpg          | XSeg                 | input / mask / cut out                                                    |
| segmentation_occluded.jpg | XSeg                 | 8.6% of frame masked                                                      |
| matting.jpg               | MODNet               | input / matte / composite                                                 |
| matting_alt.jpg           | MODNet               | fine hair, plain background                                               |
| headpose.jpg              | ResNet-34            | yaw −78° / +9° / +40°                                                 |
| gaze.jpg                  | MobileGaze ResNet-18 | yaw −30° / −19° / +23°                                               |
| demography.jpg            | FairFace             | 3-9, 20-29, 40-49, 60-69                                                  |
| emotion.jpg               | AffectNet-8          | all 8 classes, p 0.75–0.99                                               |
| face_states.jpg           | FaceAttribNet        | glasses 0.74, shades 1.00, mask 1.00                                      |
| face_states_alt.jpg       | FaceAttribNet        | glasses 1.00, shades 1.00, mask 1.00                                      |
| quality.jpg               | eDifFIQA(L)          | 0.398 … 0.749 across 7 faces                                             |
| spoofing.jpg              | MiniFASNet           | live Real 1.00; print Fake 0.66, screen Fake 0.99                         |
| anonymization.jpg         | BlurFace             | 4 of 5 methods, 5 faces                                                   |
| verification.jpg          | AdaFace IR-101       | +0.746 at 4 yr, +0.721 at 10 yr; −0.049 and −0.040 reject               |
| verification_alt.jpg      | AdaFace IR-101       | Einstein +0.583 at 26 yr, Bohr +0.689 at 25 yr; +0.001 and −0.031 reject |

Not covered: **tracking** needs video, and the **FAISS store** needs a database rather than an
image. Anti-spoofing is covered now, but only because the three frames come from one capture setup:
MiniFASNet judges presentation, so a found photograph is a replay by definition.

## Choices worth keeping

- **Gaze uses ResNet-18.** Against ResNet-34/50 and MobileNetV2 on the same three subjects it was
  the only backbone returning a positive yaw on the third face, so the row reads leftward to
  rightward instead of all-leftward.
- **Gaze subjects are not left/centre/right.** Measured at −30°, −19° and +23°, the middle face is
  still looking left, which is why the filenames say `away` and `averted`.
- **Demography uses FairFace, not AgeGender.** AgeGender put a child at 30 and called an elderly
  woman Male; FairFace's buckets order correctly. The figure sorts by predicted bucket, so filename
  order does not matter.
- **Head pose follows `tools/headpose.py`**: angles estimated on the unpadded bbox crop, drawn with
  `draw_head_pose(draw_type='cube')`.
- **Head pose prints pitch and roll only below 60° of yaw.** Past that this model returns 35–82° of
  tilt on a level head, so `pose_left` at −78° shows yaw alone.
- **Parsing crops to the face first.** BiSeNet trains on CelebAMask-HQ, which is face-centred crops,
  so a full-body portrait leaves the face too small for eyes, brows and lips to resolve.
- **Quality runs on one photograph and shows it.** Pooling faces from several sources made the count
  unverifiable, since the reader never saw where they came from.
- **Verification avoids twins.** An identical-twin pair scored above a genuine same-person match,
  which reads as a bug rather than a demonstration. Negatives are man-vs-man so the reject is not
  trivially separable by sex.
- **One type scale** across the set: footer 20, label 22, value 38, sub 19, legend 26, with a 34px
  clear band above every footer.

## Credits

Source photographs come from Pexels, Unsplash, Pixabay and Wikimedia Commons. The three
`spoof_*.jpg` frames come from [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing).

| File                     | Source                                                                                            | Author                                     | Licence                        |
| ------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------ |
| `verify_bohr_1910.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Niels_Bohr_-_LOC_-_ggbain_-_35303.jpg) | Bain News Service, via Library of Congress | PD-Bain, no known restrictions |
| `verify_bohr_1935.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Niels_Bohr_1935.jpg)                   | Unknown                                    | PD-anon-70-EU                  |
