# Licenses & Attribution

## UniFace License

UniFace is released under the [MIT License](https://opensource.org/licenses/MIT).

That licence covers UniFace's own source code. It does not cover the pretrained
weights, which are third-party artifacts downloaded on first use and governed by
their upstream terms. Several of them prohibit commercial use — check the
**Weights** column below before shipping.

---

## Model Credits

| Model | Source | Code | Weights |
|-------|--------|------|---------|
| RetinaFace | [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) | MIT | Follows source |
| SCRFD | [InsightFace](https://github.com/deepinsight/insightface) | MIT | **Non-commercial**[^insightface] |
| CenterFace | [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace) | MIT | Follows source |
| BlazeFace | [yakhyo/mediapipe-face-mesh-onnx](https://github.com/yakhyo/mediapipe-face-mesh-onnx) — architecture & weights from Google [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | Apache-2.0 |
| YOLOv5-Face | [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference) | GPL-3.0 | Follows source |
| YOLOv8-Face | [yakhyo/yolov8-face-onnx-inference](https://github.com/yakhyo/yolov8-face-onnx-inference) | GPL-3.0 | Follows source |
| AdaFace | [yakhyo/adaface-onnx](https://github.com/yakhyo/adaface-onnx) — architecture & weights from [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) | MIT | MIT |
| ArcFace | [InsightFace](https://github.com/deepinsight/insightface) | MIT | **Non-commercial**[^insightface] |
| EdgeFace | [yakhyo/edgeface-onnx](https://github.com/yakhyo/edgeface-onnx) — architecture & weights from [otroshi/edgeface](https://github.com/otroshi/edgeface), Idiap Research Institute | BSD-3-Clause | BSD-3-Clause |
| MobileFace | [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) | MIT | Follows source |
| SphereFace | [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) | MIT | Follows source |
| BiSeNet | [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) | MIT | Follows source |
| MobileGaze | [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) | MIT | Follows source |
| MODNet | [yakhyo/modnet](https://github.com/yakhyo/modnet) | Apache-2.0 | Apache-2.0 |
| MiniFASNet | [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) | Apache-2.0 | Apache-2.0 |
| FairFace | [yakhyo/fairface-onnx](https://github.com/yakhyo/fairface-onnx) | CC BY 4.0 | CC BY 4.0 |
| AgeGender | [InsightFace](https://github.com/deepinsight/insightface) — `genderage` from the buffalo packs | MIT | **Non-commercial**[^insightface] |
| FaceAttribNet | [yakhyo/face-attribute](https://github.com/yakhyo/face-attribute) — architecture & weights © Qualcomm Technologies, Inc. ([qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)) | BSD-3-Clause | BSD-3-Clause |
| PIPNet | [yakhyo/pipnet-onnx](https://github.com/yakhyo/pipnet-onnx) — meanface tables vendored from [jhb86253817/PIPNet](https://github.com/jhb86253817/PIPNet) | MIT | Follows source |
| Landmark106 | [InsightFace](https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg) — `2d106det` | MIT | **Non-commercial**[^insightface] |
| Face Mesh | [yakhyo/mediapipe-face-mesh-onnx](https://github.com/yakhyo/mediapipe-face-mesh-onnx) — topology & weights from Google [MediaPipe](https://github.com/google-ai-edge/mediapipe), sourced via [PINTO0309's ONNX conversion](https://github.com/PINTO0309/facemesh_onnx_tensorrt); tessellation tables vendored from the same source | Apache-2.0 | Apache-2.0 |

!!! note "Rows marked *Follows source*"
    We have not identified weights terms separate from the source project's
    licence, so the source project's terms apply.

!!! warning "Training-data provenance"
    Separately from the licences above, many face models are trained on datasets
    whose own terms restrict use to non-commercial research — WebFace260M
    (AdaFace, EdgeFace), WIDER FACE (most detectors), MS-Celeb-1M. The upstream
    authors released these weights under the permissive licences listed here and
    state no further restriction, and this page does not attempt to judge
    whether a dataset's terms reach you as a downstream recipient of the
    weights. If you are deploying commercially, do your own diligence.

---

## Non-commercial weights

These weights cannot be used commercially under their upstream terms, even
though UniFace itself is MIT and the upstream **code** is permissive.

### InsightFace — SCRFD, ArcFace, AgeGender, Landmark106 { #insightface }

> The code of InsightFace is released under the MIT License. There is no
> limitation for both academic and commercial usage.
>
> The training data containing the annotation (and the models trained with these
> data) are available for non-commercial research purposes only.

— [deepinsight/insightface](https://github.com/deepinsight/insightface)

ArcFace's `w600k_r50` is the `buffalo_l` recognition model. InsightFace directs
commercial licensing enquiries for it to `recognition-oss-pack@insightface.ai`.

[^insightface]: InsightFace's code is MIT, but models trained on their data are
    for non-commercial research only. See [Non-commercial weights](#insightface).
