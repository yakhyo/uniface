# Changelog

All notable changes to this project will be documented in this file.

## [3.1.1] - 2026-03-19

- Drop Python 3.10 support; minimum is now Python 3.11 ([#95](https://github.com/yakhyo/uniface/pull/95))
- Bump `scikit-image` to `>=0.26.0` and use `SimilarityTransform.from_estimate()` ([#95](https://github.com/yakhyo/uniface/pull/95))
- Add Python 3.14 support ([#95](https://github.com/yakhyo/uniface/pull/95))

## [3.1.0] - 2026-03-11

- Add FAISS vector database for fast face search ([#86](https://github.com/yakhyo/uniface/pull/86))
- Add `ModelInfo` dataclass for centralized model registry ([#90](https://github.com/yakhyo/uniface/pull/90))
- Add download retry with exponential backoff ([#90](https://github.com/yakhyo/uniface/pull/90))
- Add dataset documentation ([#85](https://github.com/yakhyo/uniface/pull/85))

## [3.0.0] - 2026-02-14

- Add ByteTrack multi-object face tracking ([#81](https://github.com/yakhyo/uniface/pull/81))
- Add 5 gaze estimation backbones ([#82](https://github.com/yakhyo/uniface/pull/82))
- Add configurable cache directory ([#80](https://github.com/yakhyo/uniface/pull/80))
- Redesign unified API with standardized return types ([#82](https://github.com/yakhyo/uniface/pull/82))

## [2.3.0] - 2026-02-05

- Add XSeg face segmentation ([#72](https://github.com/yakhyo/uniface/pull/72))
- Update documentation and README ([#78](https://github.com/yakhyo/uniface/pull/78))

## [2.2.1] - 2026-01-18

- Add ONNX Runtime provider selection (CUDA, CoreML, CPU) ([#68](https://github.com/yakhyo/uniface/pull/68))
- Fix cache directory check on startup ([#67](https://github.com/yakhyo/uniface/pull/67))

## [2.2.0] - 2026-01-07

- Add YOLOv8-Face detection ([#62](https://github.com/yakhyo/uniface/pull/62))
- Add AdaFace recognition ([#61](https://github.com/yakhyo/uniface/pull/61))
- Add MkDocs documentation site ([#51](https://github.com/yakhyo/uniface/pull/51))
- Add Google Colab support ([#52](https://github.com/yakhyo/uniface/pull/52))

## [2.0.0] - 2025-12-30

- Initial v2 release
- Detection: RetinaFace, SCRFD, YOLOv5-Face
- Recognition: ArcFace, MobileFace, SphereFace
- 106-point facial landmarks
- Face parsing (BiSeNet, 19 classes)
- Gaze estimation (MobileGaze)
- Age/gender/race prediction (AgeGender, FairFace)
- Emotion recognition (DDAMFN)
- Anti-spoofing (MiniFASNet)
- Face anonymization (5 blur methods)
- SHA-256 weight verification
- Pure ONNX Runtime inference

[3.1.1]: https://github.com/yakhyo/uniface/compare/v3.1.0...v3.1.1
[3.1.0]: https://github.com/yakhyo/uniface/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/yakhyo/uniface/compare/v2.3.0...v3.0.0
[2.3.0]: https://github.com/yakhyo/uniface/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/yakhyo/uniface/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/yakhyo/uniface/compare/v2.0.0...v2.2.0
[2.0.0]: https://github.com/yakhyo/uniface/releases/tag/v2.0.0
