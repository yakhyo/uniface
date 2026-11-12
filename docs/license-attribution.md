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
| RetinaFace | [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch); `RESNET50` weights from [HivisionIDPhotos](https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting) | MIT | Follows source; `RESNET50` Apache-2.0 |
| SCRFD | [InsightFace](https://github.com/deepinsight/insightface) | MIT | **Non-commercial**[^insightface] |
| CenterFace | [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace) | MIT | Follows source |
| BlazeFace | [yakhyo/mediapipe-face-mesh-onnx](https://github.com/yakhyo/mediapipe-face-mesh-onnx) — architecture & weights from Google [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | Apache-2.0 |
| YOLOv5-Face | [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference), original implementation [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face) | GPL-3.0 | Follows source |
| YOLOv8-Face | [yakhyo/yolov8-face-onnx-inference](https://github.com/yakhyo/yolov8-face-onnx-inference), original implementation [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face) | GPL-3.0 | Follows source |
| AdaFace | [yakhyo/adaface-onnx](https://github.com/yakhyo/adaface-onnx) — architecture & weights from [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) | MIT | MIT |
| ArcFace | [InsightFace](https://github.com/deepinsight/insightface) | MIT | **Non-commercial**[^insightface] |
| AuraFace | [yakhyo/auraface-onnx](https://github.com/yakhyo/auraface-onnx), architecture & weights from [fal/AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) | Apache-2.0 | Apache-2.0 |
| EdgeFace | [yakhyo/edgeface-onnx](https://github.com/yakhyo/edgeface-onnx) — architecture & weights from [otroshi/edgeface](https://github.com/otroshi/edgeface), Idiap Research Institute | BSD-3-Clause | **Conflicting**: BSD-3-Clause in the GitHub repo, CC BY-NC-SA 4.0 on Hugging Face[^edgeface] |
| HyperFace | [yakhyo/hyperface-onnx](https://github.com/yakhyo/hyperface-onnx), architecture & weights from [Idiap/HyperFace-50k-StyleGAN](https://huggingface.co/Idiap/HyperFace-50k-StyleGAN) | MIT | MIT |
| MobileFace | [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) | MIT | Follows source |
| SphereFace | [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) | MIT | Follows source |
| BiSeNet | [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) | MIT | Follows source |
| MobileGaze | [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) | MIT | Follows source |
| HeadPose | [yakhyo/head-pose-estimation](https://github.com/yakhyo/head-pose-estimation) | MIT | Follows source |
| XSeg | [yakhyo/face-segmentation](https://github.com/yakhyo/face-segmentation), model from [iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab) | GPL-3.0 | GPL-3.0 |
| MODNet | [yakhyo/modnet](https://github.com/yakhyo/modnet) | Apache-2.0 | Apache-2.0 |
| PP-Matting | [yakhyo/ppmatting](https://github.com/yakhyo/ppmatting) | Apache-2.0 | Apache-2.0, training data undocumented upstream |
| MiniFASNet | [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) | Apache-2.0 | Apache-2.0 |
| FairFace | [yakhyo/fairface-onnx](https://github.com/yakhyo/fairface-onnx) | MIT | CC BY 4.0 |
| Emotion | [SainingZhang/DDAMFN](https://github.com/SainingZhang/DDAMFN) | None stated | **No licence stated upstream**[^emotion] |
| eDifFIQA | [yakhyo/face-image-quality-assessment](https://github.com/yakhyo/face-image-quality-assessment), model from [LSIbabnikz/eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA) | MIT | MIT |
| AgeGender | [InsightFace](https://github.com/deepinsight/insightface) — `genderage` from the buffalo packs | MIT | **Non-commercial**[^insightface] |
| FaceAttribNet | [yakhyo/face-attribute](https://github.com/yakhyo/face-attribute) — architecture & weights © Qualcomm Technologies, Inc. ([qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)) | BSD-3-Clause | BSD-3-Clause |
| PIPNet | [yakhyo/pipnet-onnx](https://github.com/yakhyo/pipnet-onnx) — meanface tables vendored from [jhb86253817/PIPNet](https://github.com/jhb86253817/PIPNet) | MIT | Follows source |
| Landmark106 | [InsightFace](https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg) — `2d106det` | MIT | **Non-commercial**[^insightface] |
| Face Mesh | [yakhyo/mediapipe-face-mesh-onnx](https://github.com/yakhyo/mediapipe-face-mesh-onnx) — topology & weights from Google [MediaPipe](https://github.com/google-ai-edge/mediapipe), `V1_468` weights via [PINTO0309's ONNX conversion](https://github.com/PINTO0309/facemesh_onnx_tensorrt), `V2_478` weights read from Google's `face_landmarker.task`; tessellation tables vendored from MediaPipe | Apache-2.0 | Apache-2.0 |

!!! note "Rows marked *Follows source*"
    We have not identified weights terms separate from the source project's
    licence, so the source project's terms apply.

!!! warning "Training-data provenance"
    Separately from the licences above, many face models are trained on datasets
    whose own terms restrict use to non-commercial research — WebFace260M
    (AdaFace, EdgeFace), WIDER FACE (most detectors), MS-Celeb-1M. Except where
    noted above, the upstream authors released these weights under the permissive
    licences listed here and state no further restriction, and this page does not attempt to judge
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

---

## Unclear terms: EdgeFace and Emotion { #unclear-terms }

Two models do not have clear terms for commercial use. Treat them as non-commercial
unless you have confirmed otherwise with the authors.

- **EdgeFace**: the [otroshi/edgeface](https://github.com/otroshi/edgeface) GitHub
  repository is BSD-3-Clause, but Idiap's Hugging Face releases of the same models
  (for example [Idiap/EdgeFace-Base](https://huggingface.co/Idiap/EdgeFace-Base)) are
  CC BY-NC-SA 4.0, which prohibits commercial use.
- **Emotion**: the [SainingZhang/DDAMFN](https://github.com/SainingZhang/DDAMFN)
  repository has no licence file and its README states no licence, so no permission
  is granted by default. The model is trained on AffectNet, which is distributed
  for research use.

[^edgeface]: See [Unclear terms](#unclear-terms).

[^emotion]: See [Unclear terms](#unclear-terms).

[^insightface]: InsightFace's code is MIT, but models trained on their data are
    for non-commercial research only. See [Non-commercial weights](#insightface).
