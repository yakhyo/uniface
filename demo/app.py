# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# License: MIT
# UniFace Gradio Demo for Hugging Face Spaces

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import uniface
from uniface.constants import (
    AdaFaceWeights,
    ArcFaceWeights,
    GazeWeights,
    MiniFASNetWeights,
    MobileFaceWeights,
    ParsingWeights,
    RetinaFaceWeights,
    SCRFDWeights,
    SphereFaceWeights,
    YOLOv5FaceWeights,
    YOLOv8FaceWeights,
)
from uniface.visualization import (
    FACE_PARSING_COLORS,
    FACE_PARSING_LABELS,
    draw_detections,
    draw_gaze,
    vis_parsing_maps,
)

# ---------------------------------------------------------------------------
# Resolve asset paths relative to this script so examples work from any cwd
# ---------------------------------------------------------------------------
_DEMO_ASSETS_DIR = Path(__file__).resolve().parent / 'assets'
EXAMPLE_DEFAULT = str(_DEMO_ASSETS_DIR / 'image.png')
EXAMPLE_VER_IMG1 = str(_DEMO_ASSETS_DIR / 'verification' / 'image1.png')
EXAMPLE_VER_IMG2 = str(_DEMO_ASSETS_DIR / 'verification' / 'image2.png')
EXAMPLE_VER_IMG3 = str(_DEMO_ASSETS_DIR / 'verification' / 'image3.png')
EXAMPLE_ATTR_1 = str(_DEMO_ASSETS_DIR / 'attribute' / 'image1.png')
EXAMPLE_ATTR_2 = str(_DEMO_ASSETS_DIR / 'attribute' / 'image2.png')
EXAMPLE_GAZE_1 = str(_DEMO_ASSETS_DIR / 'gaze' / 'image1.png')
EXAMPLE_GAZE_2 = str(_DEMO_ASSETS_DIR / 'gaze' / 'image2.png')
EXAMPLE_SPOOF_1 = str(_DEMO_ASSETS_DIR / 'spoofing' / 'image1.jpg')
EXAMPLE_SPOOF_2 = str(_DEMO_ASSETS_DIR / 'spoofing' / 'image2.jpg')
EXAMPLE_SPOOF_3 = str(_DEMO_ASSETS_DIR / 'spoofing' / 'image3.jpg')
EXAMPLE_ANONYMIZE = str(_DEMO_ASSETS_DIR / 'anonymize' / 'image.png')

# ---------------------------------------------------------------------------
# Model cache: lazily create and reuse model instances
# ---------------------------------------------------------------------------
_model_cache: dict[str, object] = {}


def _get_model(key: str, factory, *args, **kwargs):
    """Get a cached model or create a new one."""
    if key not in _model_cache:
        _model_cache[key] = factory(*args, **kwargs)
    return _model_cache[key]


# ---------------------------------------------------------------------------
# Detector family -> variant mappings
# ---------------------------------------------------------------------------
DETECTOR_VARIANTS: dict[str, list[str]] = {
    'RetinaFace': [w.value for w in RetinaFaceWeights],
    'SCRFD': [w.value for w in SCRFDWeights],
    'YOLOv5-Face': [w.value for w in YOLOv5FaceWeights],
    'YOLOv8-Face': [w.value for w in YOLOv8FaceWeights],
}

DETECTOR_METHOD_MAP: dict[str, str] = {
    'RetinaFace': 'retinaface',
    'SCRFD': 'scrfd',
    'YOLOv5-Face': 'yolov5face',
    'YOLOv8-Face': 'yolov8face',
}

RECOGNIZER_VARIANTS: dict[str, list[str]] = {
    'ArcFace': [w.value for w in ArcFaceWeights],
    'AdaFace': [w.value for w in AdaFaceWeights],
    'MobileFace': [w.value for w in MobileFaceWeights],
    'SphereFace': [w.value for w in SphereFaceWeights],
}

RECOGNIZER_METHOD_MAP: dict[str, str] = {
    'ArcFace': 'arcface',
    'AdaFace': 'adaface',
    'MobileFace': 'mobileface',
    'SphereFace': 'sphereface',
}

GAZE_VARIANT_MAP: dict[str, GazeWeights] = {
    'ResNet18': GazeWeights.RESNET18,
    'ResNet34': GazeWeights.RESNET34,
    'ResNet50': GazeWeights.RESNET50,
    'MobileNetV2': GazeWeights.MOBILENET_V2,
    'MobileOne-S0': GazeWeights.MOBILEONE_S0,
}

PARSING_VARIANT_MAP: dict[str, ParsingWeights] = {
    'ResNet18': ParsingWeights.RESNET18,
    'ResNet34': ParsingWeights.RESNET34,
}

SPOOFING_VARIANT_MAP: dict[str, MiniFASNetWeights] = {
    'V1SE': MiniFASNetWeights.V1SE,
    'V2': MiniFASNetWeights.V2,
}

# Reverse lookup: enum .value string -> enum member (for all weight enums)
_STR_TO_ENUM: dict[str, object] = {}
for _enum_cls in (
    RetinaFaceWeights,
    SCRFDWeights,
    YOLOv5FaceWeights,
    YOLOv8FaceWeights,
    ArcFaceWeights,
    AdaFaceWeights,
    MobileFaceWeights,
    SphereFaceWeights,
):
    for _member in _enum_cls:
        _STR_TO_ENUM[_member.value] = _member


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _expand_bbox(
    bbox: np.ndarray,
    image_shape: tuple[int, ...],
    expand_ratio: float = 0.2,
    expand_top_ratio: float = 0.4,
) -> tuple[int, int, int, int]:
    """Expand bbox to capture full head for parsing."""
    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = image_shape[:2]
    fw, fh = x2 - x1, y2 - y1
    ex = int(fw * expand_ratio)
    ey_bottom = int(fh * expand_ratio)
    ey_top = int(fh * expand_top_ratio)
    return max(0, x1 - ex), max(0, y1 - ey_top), min(w, x2 + ex), min(h, y2 + ey_bottom)


# ===================================================================
# Tab 1: Face Detection
# ===================================================================
def _update_detector_variants(detector_family: str):
    variants = DETECTOR_VARIANTS.get(detector_family, [])
    return gr.update(choices=variants, value=variants[0] if variants else None)


def detect_faces_fn(
    image: np.ndarray,
    detector_family: str,
    detector_variant: str,
    confidence: float,
    nms_threshold: float,
) -> tuple[np.ndarray, str]:
    if image is None:
        return None, ''

    bgr = _rgb_to_bgr(image)
    method = DETECTOR_METHOD_MAP[detector_family]
    model_enum = _STR_TO_ENUM[detector_variant]
    cache_key = f'det_{method}_{detector_variant}_{confidence}_{nms_threshold}'
    detector = _get_model(
        cache_key,
        uniface.create_detector,
        method,
        model_name=model_enum,
        confidence_threshold=confidence,
        nms_threshold=nms_threshold,
    )

    faces = detector.detect(bgr)
    result = bgr.copy()

    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]
    draw_detections(image=result, bboxes=bboxes, scores=scores, landmarks=landmarks, draw_score=True, fancy_bbox=True)

    faces_json = {}
    for i, f in enumerate(faces, 1):
        x1, y1, x2, y2 = map(int, f.bbox)
        lmk = f.landmarks.astype(int)
        faces_json[f'face_{i}'] = {
            'confidence': round(float(f.confidence), 4),
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'left_eye': {'x': int(lmk[0][0]), 'y': int(lmk[0][1])},
            'right_eye': {'x': int(lmk[1][0]), 'y': int(lmk[1][1])},
            'nose': {'x': int(lmk[2][0]), 'y': int(lmk[2][1])},
            'mouth_left': {'x': int(lmk[3][0]), 'y': int(lmk[3][1])},
            'mouth_right': {'x': int(lmk[4][0]), 'y': int(lmk[4][1])},
        }

    return _bgr_to_rgb(result), json.dumps({'num_faces': len(faces), **faces_json}, indent=2)


# ===================================================================
# Tab 2: Face Verification
# ===================================================================
def _update_recognizer_variants(recognizer_family: str):
    variants = RECOGNIZER_VARIANTS.get(recognizer_family, [])
    return gr.update(choices=variants, value=variants[0] if variants else None)


def verify_faces_fn(
    image_a: np.ndarray,
    image_b: np.ndarray,
    recognizer_family: str,
    recognizer_variant: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if image_a is None or image_b is None:
        return None, None, json.dumps({'error': 'Please upload both images.'}, indent=2)

    bgr_a = _rgb_to_bgr(image_a)
    bgr_b = _rgb_to_bgr(image_b)

    # Detector (shared lightweight)
    det = _get_model('det_retina_default', uniface.create_detector, 'retinaface')

    method = RECOGNIZER_METHOD_MAP[recognizer_family]
    rec_enum = _STR_TO_ENUM[recognizer_variant]
    rec_key = f'rec_{method}_{recognizer_variant}'
    rec = _get_model(rec_key, uniface.create_recognizer, method, model_name=rec_enum)

    faces_a = det.detect(bgr_a)
    faces_b = det.detect(bgr_b)

    if not faces_a:
        return None, None, json.dumps({'error': 'No face detected in Image A.'}, indent=2)
    if not faces_b:
        return None, None, json.dumps({'error': 'No face detected in Image B.'}, indent=2)

    face_a = faces_a[0]
    face_b = faces_b[0]

    emb_a = rec.get_normalized_embedding(bgr_a, face_a.landmarks)
    emb_b = rec.get_normalized_embedding(bgr_b, face_b.landmarks)
    similarity = float(uniface.compute_similarity(emb_a, emb_b))

    # Aligned face crops for display
    crop_a, _ = uniface.face_alignment(bgr_a, face_a.landmarks, image_size=112)
    crop_b, _ = uniface.face_alignment(bgr_b, face_b.landmarks, image_size=112)

    verdict = 'Same Person' if similarity > 0.4 else 'Different Person'

    return (
        _bgr_to_rgb(crop_a),
        _bgr_to_rgb(crop_b),
        json.dumps(
            {
                'cosine_similarity': round(similarity, 4),
                'threshold': 0.4,
                'verdict': verdict,
            },
            indent=2,
        ),
    )


# ===================================================================
# Tab 3: Face Analysis (Attributes)
# ===================================================================
def analyze_faces_fn(
    image: np.ndarray,
    attr_model: str,
) -> tuple[np.ndarray, str]:
    if image is None:
        return None, ''

    bgr = _rgb_to_bgr(image)

    det = _get_model('det_retina_default', uniface.create_detector, 'retinaface')

    if attr_model == 'AgeGender':
        ag = _get_model('agegender', uniface.AgeGender)
        analyzer = uniface.FaceAnalyzer(det, age_gender=ag)
    else:
        ff = _get_model('fairface', uniface.FairFace)
        analyzer = uniface.FaceAnalyzer(det, fairface=ff)

    faces = analyzer.analyze(bgr)
    result = bgr.copy()

    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]
    draw_detections(image=result, bboxes=bboxes, scores=scores, landmarks=landmarks, fancy_bbox=True)

    # Draw attribute labels
    for face in faces:
        x1, y1 = int(face.bbox[0]), int(face.bbox[1])
        label_parts = []
        if face.sex is not None:
            label_parts.append(face.sex)
        if face.age is not None:
            label_parts.append(f'{face.age}y')
        if face.age_group is not None:
            label_parts.append(face.age_group)
        if face.race is not None:
            label_parts.append(face.race)
        label = ', '.join(label_parts)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(result, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    faces_json = {}
    for i, face in enumerate(faces, 1):
        x1, y1, x2, y2 = map(int, face.bbox)
        entry: dict = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        if face.sex is not None:
            entry['gender'] = face.sex
        if face.age is not None:
            entry['age'] = face.age
        if face.age_group is not None:
            entry['age_group'] = face.age_group
        if face.race is not None:
            entry['race'] = face.race
        faces_json[f'face_{i}'] = entry

    return _bgr_to_rgb(result), json.dumps({'model': attr_model, 'num_faces': len(faces), **faces_json}, indent=2)


# ===================================================================
# Tab 4: Landmarks (106-point)
# ===================================================================
def landmarks_fn(image: np.ndarray) -> tuple[np.ndarray, str]:
    if image is None:
        return None, ''

    bgr = _rgb_to_bgr(image)
    det = _get_model('det_retina_default', uniface.create_detector, 'retinaface')
    landmarker = _get_model('landmarker_106', uniface.create_landmarker, '2d106det')

    faces = det.detect(bgr)
    result = bgr.copy()

    # Color palette for 106 landmarks
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]

    faces_json = {}
    for i, face in enumerate(faces):
        lmk106 = landmarker.get_landmarks(bgr, face.bbox)
        x1, y1, x2, y2 = map(int, face.bbox)
        lmk_dict = {f'pt_{j}': {'x': int(pt[0]), 'y': int(pt[1])} for j, pt in enumerate(lmk106)}
        faces_json[f'face_{i + 1}'] = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'num_landmarks': len(lmk106),
            'landmarks': lmk_dict,
        }

        for j, pt in enumerate(lmk106):
            color = colors[j % len(colors)]
            cv2.circle(result, (int(pt[0]), int(pt[1])), 2, color, -1)

    return _bgr_to_rgb(result), json.dumps({'num_faces': len(faces), **faces_json}, indent=2)


# ===================================================================
# Tab 5: Face Parsing
# ===================================================================
def parsing_fn(
    image: np.ndarray,
    model_variant: str,
) -> tuple[np.ndarray, str]:
    if image is None:
        return None, ''

    bgr = _rgb_to_bgr(image)
    det = _get_model('det_retina_default', uniface.create_detector, 'retinaface')

    weights = PARSING_VARIANT_MAP[model_variant]
    parser = _get_model(f'parser_{weights.value}', uniface.create_face_parser, weights)

    faces = det.detect(bgr)
    result = bgr.copy()

    faces_json = {}
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = _expand_bbox(face.bbox, bgr.shape)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        mask = parser.parse(crop)
        unique_classes = sorted(set(mask.flatten()))
        class_names = [FACE_PARSING_LABELS[c] for c in unique_classes if c < len(FACE_PARSING_LABELS)]
        faces_json[f'face_{i + 1}'] = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'num_classes': len(unique_classes),
            'classes': ', '.join(class_names),
        }

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        vis = vis_parsing_maps(crop_rgb, mask, save_image=False)
        result[y1:y2, x1:x2] = vis

    # Build legend map
    legend = {}
    for idx, name in enumerate(FACE_PARSING_LABELS):
        if idx == 0:
            continue
        r, g, b = FACE_PARSING_COLORS[idx]
        legend[name] = f'rgb({r}, {g}, {b})'

    return _bgr_to_rgb(result), json.dumps(
        {
            'model': model_variant,
            'num_faces': len(faces),
            **faces_json,
            'legend': legend,
        },
        indent=2,
    )


# ===================================================================
# Tab 6: Gaze Estimation
# ===================================================================
def gaze_fn(
    image: np.ndarray,
    backbone: str,
) -> tuple[np.ndarray, str]:
    if image is None:
        return None, ''

    bgr = _rgb_to_bgr(image)
    det = _get_model('det_retina_default', uniface.create_detector, 'retinaface')

    weights = GAZE_VARIANT_MAP[backbone]
    gaze = _get_model(f'gaze_{weights.value}', uniface.create_gaze_estimator, model_name=weights)

    faces = det.detect(bgr)
    result = bgr.copy()

    faces_json = {}
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        gaze_result = gaze.estimate(crop)
        pitch_deg = float(np.degrees(gaze_result.pitch))
        yaw_deg = float(np.degrees(gaze_result.yaw))
        faces_json[f'face_{i + 1}'] = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'pitch_deg': round(pitch_deg, 2),
            'yaw_deg': round(yaw_deg, 2),
        }

        draw_gaze(result, face.bbox, gaze_result.pitch, gaze_result.yaw, draw_angles=True)

    return _bgr_to_rgb(result), json.dumps({'backbone': backbone, 'num_faces': len(faces), **faces_json}, indent=2)


# ===================================================================
# Tab 7: Anti-Spoofing
# ===================================================================
def spoofing_fn(
    image: np.ndarray,
    model_variant: str,
) -> tuple[np.ndarray, str]:
    if image is None:
        return None, ''

    bgr = _rgb_to_bgr(image)
    det = _get_model('det_retina_default', uniface.create_detector, 'retinaface')

    weights = SPOOFING_VARIANT_MAP[model_variant]
    spoofer = _get_model(f'spoof_{weights.value}', uniface.create_spoofer, weights)

    faces = det.detect(bgr)
    result = bgr.copy()

    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    lmks = [f.landmarks for f in faces]
    draw_detections(image=result, bboxes=bboxes, scores=scores, landmarks=lmks, fancy_bbox=True)

    faces_json = {}
    for i, face in enumerate(faces):
        spoof_result = spoofer.predict(bgr, face.bbox)
        label = 'REAL' if spoof_result.is_real else 'FAKE'
        color = (0, 255, 0) if spoof_result.is_real else (0, 0, 255)
        x1, y1, x2, y2 = map(int, face.bbox)
        faces_json[f'face_{i + 1}'] = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'verdict': label,
            'is_real': bool(spoof_result.is_real),
            'confidence': round(float(spoof_result.confidence), 4),
        }

        x1, y1 = int(face.bbox[0]), int(face.bbox[1])
        text = f'{label} {spoof_result.confidence:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(result, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return _bgr_to_rgb(result), json.dumps({'num_faces': len(faces), **faces_json}, indent=2)


# ===================================================================
# Tab 8: Face Anonymization
# ===================================================================
def anonymize_fn(
    image: np.ndarray,
    blur_method: str,
) -> tuple[np.ndarray, str]:
    if image is None:
        return None, ''

    bgr = _rgb_to_bgr(image)
    result = uniface.anonymize_faces(bgr, method=blur_method)

    return _bgr_to_rgb(result), json.dumps({'method': blur_method, 'status': 'done'}, indent=2)


# ===================================================================
# Gradio UI
# ===================================================================
def build_app() -> gr.Blocks:
    with gr.Blocks(title='UniFace Demo') as app:
        gr.Markdown(
            '<div style="text-align: center;">'
            '<h1 style="margin: 0;">UniFace</h1>'
            '<h3 style="margin: 4px 0 0;">All-in-One Face Analysis Library</h3>'
            f'<p style="margin: 4px 0 8px;">v{uniface.__version__} &nbsp;·&nbsp; '
            'Built on ONNX Runtime &nbsp;·&nbsp; Fast, lightweight, production-ready</p>'
            '<p style="margin: 0;">'
            'Face Detection · Recognition · Landmarks · Parsing · '
            'Gaze Estimation · Attribute Analysis · Anti-Spoofing · Anonymization'
            '</p>'
            '<p style="margin: 8px 0 0;">'
            '<a href="https://github.com/yakhyo/uniface" target="_blank">⭐ Star on GitHub</a>'
            ' &nbsp;·&nbsp; '
            '<a href="https://pypi.org/project/uniface/" target="_blank">PyPI</a>'
            ' &nbsp;·&nbsp; '
            '<a href="https://yakhyo.github.io/uniface/" target="_blank">Docs</a>'
            ' &nbsp;·&nbsp; '
            '<a href="https://www.kaggle.com/yakhyokhuja/code" target="_blank">Kaggle Notebooks</a>'
            '</p>'
            '</div>',
        )

        # ------ Tab 1: Detection ------
        with gr.Tab('Face Detection'):
            gr.Markdown('Detect faces using multiple detector architectures with adjustable thresholds.')
            with gr.Row():
                with gr.Column():
                    det_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Accordion('Settings', open=False):
                        with gr.Row():
                            det_family = gr.Dropdown(
                                choices=list(DETECTOR_VARIANTS.keys()),
                                value='RetinaFace',
                                label='Detector',
                            )
                            det_variant = gr.Dropdown(
                                choices=DETECTOR_VARIANTS['RetinaFace'],
                                value=RetinaFaceWeights.MNET_V2.value,
                                label='Model Variant',
                            )
                        det_conf = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label='Confidence Threshold')
                        det_nms = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label='NMS Threshold')
                    det_btn = gr.Button('Detect Faces', variant='primary')
                with gr.Column():
                    det_output = gr.Image(label='Result')
                    det_text = gr.Textbox(label='Results', lines=8, show_copy_button=True)

            det_family.change(_update_detector_variants, inputs=det_family, outputs=det_variant)
            det_btn.click(
                detect_faces_fn,
                inputs=[det_image, det_family, det_variant, det_conf, det_nms],
                outputs=[det_output, det_text],
            )

            gr.Examples(
                examples=[[EXAMPLE_DEFAULT, 'RetinaFace', RetinaFaceWeights.MNET_V2.value, 0.5, 0.4]],
                inputs=[det_image, det_family, det_variant, det_conf, det_nms],
                label='Try an example',
            )

        # ------ Tab 2: Verification ------
        with gr.Tab('Face Verification'):
            gr.Markdown('Compare two faces to determine if they belong to the same person.')
            with gr.Row():
                with gr.Column():
                    ver_image_a = gr.Image(label='Image A', type='numpy')
                    ver_image_b = gr.Image(label='Image B', type='numpy')
                    with gr.Accordion('Settings', open=False), gr.Row():
                        ver_family = gr.Dropdown(
                            choices=list(RECOGNIZER_VARIANTS.keys()),
                            value='ArcFace',
                            label='Recognizer',
                        )
                        ver_variant = gr.Dropdown(
                            choices=RECOGNIZER_VARIANTS['ArcFace'],
                            value=ArcFaceWeights.RESNET.value,
                            label='Model Variant',
                        )
                    ver_btn = gr.Button('Verify', variant='primary')
                with gr.Column():
                    with gr.Row():
                        ver_crop_a = gr.Image(label='Aligned Face A')
                        ver_crop_b = gr.Image(label='Aligned Face B')
                    ver_text = gr.Textbox(label='Results', lines=5, show_copy_button=True)

            ver_family.change(_update_recognizer_variants, inputs=ver_family, outputs=ver_variant)
            ver_btn.click(
                verify_faces_fn,
                inputs=[ver_image_a, ver_image_b, ver_family, ver_variant],
                outputs=[ver_crop_a, ver_crop_b, ver_text],
            )

            gr.Examples(
                examples=[
                    [EXAMPLE_VER_IMG1, EXAMPLE_VER_IMG2, 'ArcFace', ArcFaceWeights.RESNET.value],
                    [EXAMPLE_VER_IMG1, EXAMPLE_VER_IMG3, 'ArcFace', ArcFaceWeights.RESNET.value],
                    [EXAMPLE_VER_IMG2, EXAMPLE_VER_IMG3, 'ArcFace', ArcFaceWeights.RESNET.value],
                ],
                inputs=[ver_image_a, ver_image_b, ver_family, ver_variant],
                label='Try an example',
            )

        # ------ Tab 3: Analysis ------
        with gr.Tab('Face Analysis'):
            gr.Markdown('Predict age, gender, and race attributes for each detected face.')
            with gr.Row():
                with gr.Column():
                    ana_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Accordion('Settings', open=False):
                        ana_model = gr.Radio(
                            choices=['AgeGender', 'FairFace'],
                            value='AgeGender',
                            label='Attribute Model',
                        )
                    ana_btn = gr.Button('Analyze', variant='primary')
                with gr.Column():
                    ana_output = gr.Image(label='Result')
                    ana_text = gr.Textbox(label='Attributes', lines=8, show_copy_button=True)

            ana_btn.click(
                analyze_faces_fn,
                inputs=[ana_image, ana_model],
                outputs=[ana_output, ana_text],
            )

            gr.Examples(
                examples=[
                    [EXAMPLE_ATTR_1, 'AgeGender'],
                    [EXAMPLE_ATTR_2, 'FairFace'],
                ],
                inputs=[ana_image, ana_model],
                label='Try an example',
            )

        # ------ Tab 4: Landmarks ------
        with gr.Tab('Landmarks (106-pt)'):
            gr.Markdown('Detect 106 facial keypoints for detailed face geometry analysis.')
            with gr.Row():
                with gr.Column():
                    lmk_image = gr.Image(label='Input Image', type='numpy')
                    lmk_btn = gr.Button('Detect Landmarks', variant='primary')
                with gr.Column():
                    lmk_output = gr.Image(label='Result')
                    lmk_text = gr.Textbox(label='Landmarks', lines=8, show_copy_button=True)

            lmk_btn.click(
                landmarks_fn,
                inputs=[lmk_image],
                outputs=[lmk_output, lmk_text],
            )

            gr.Examples(
                examples=[[EXAMPLE_DEFAULT]],
                inputs=[lmk_image],
                label='Try an example',
            )

        # ------ Tab 5: Parsing ------
        with gr.Tab('Face Parsing'):
            gr.Markdown('Segment facial components (skin, eyes, nose, hair, etc.) into 19 semantic classes.')
            with gr.Row():
                with gr.Column():
                    par_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Accordion('Settings', open=False):
                        par_variant = gr.Radio(
                            choices=list(PARSING_VARIANT_MAP.keys()),
                            value='ResNet18',
                            label='Parsing Model',
                        )
                    par_btn = gr.Button('Parse Faces', variant='primary')
                with gr.Column():
                    par_output = gr.Image(label='Segmentation Result')
                    par_text = gr.Textbox(label='Parsing Results', lines=10, show_copy_button=True)

            par_btn.click(
                parsing_fn,
                inputs=[par_image, par_variant],
                outputs=[par_output, par_text],
            )

            gr.Examples(
                examples=[[EXAMPLE_DEFAULT, 'ResNet18']],
                inputs=[par_image, par_variant],
                label='Try an example',
            )

        # ------ Tab 6: Gaze ------
        with gr.Tab('Gaze Estimation'):
            gr.Markdown('Estimate where each person is looking by predicting pitch and yaw gaze angles.')
            with gr.Row():
                with gr.Column():
                    gaze_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Accordion('Settings', open=False):
                        gaze_backbone = gr.Dropdown(
                            choices=list(GAZE_VARIANT_MAP.keys()),
                            value='ResNet34',
                            label='Backbone',
                        )
                    gaze_btn = gr.Button('Estimate Gaze', variant='primary')
                with gr.Column():
                    gaze_output = gr.Image(label='Result')
                    gaze_text = gr.Textbox(label='Gaze Angles', lines=6, show_copy_button=True)

            gaze_btn.click(
                gaze_fn,
                inputs=[gaze_image, gaze_backbone],
                outputs=[gaze_output, gaze_text],
            )

            gr.Examples(
                examples=[
                    [EXAMPLE_GAZE_1, 'ResNet34'],
                    [EXAMPLE_GAZE_2, 'ResNet34'],
                ],
                inputs=[gaze_image, gaze_backbone],
                label='Try an example',
            )

        # ------ Tab 7: Anti-Spoofing ------
        with gr.Tab('Anti-Spoofing'):
            gr.Markdown('Determine whether a face is real (live) or a spoof (printed photo, screen replay).')
            with gr.Row():
                with gr.Column():
                    spf_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Accordion('Settings', open=False):
                        spf_variant = gr.Radio(
                            choices=list(SPOOFING_VARIANT_MAP.keys()),
                            value='V2',
                            label='Model Variant',
                        )
                    spf_btn = gr.Button('Check Liveness', variant='primary')
                with gr.Column():
                    spf_output = gr.Image(label='Result')
                    spf_text = gr.Textbox(label='Verdict', lines=6, show_copy_button=True)

            spf_btn.click(
                spoofing_fn,
                inputs=[spf_image, spf_variant],
                outputs=[spf_output, spf_text],
            )

            gr.Examples(
                examples=[
                    [EXAMPLE_SPOOF_1, 'V2'],
                    [EXAMPLE_SPOOF_2, 'V2'],
                    [EXAMPLE_SPOOF_3, 'V2'],
                ],
                inputs=[spf_image, spf_variant],
                label='Try an example',
            )

        # ------ Tab 8: Anonymization ------
        with gr.Tab('Face Anonymization'):
            gr.Markdown('Blur or mask detected faces to protect privacy in images.')
            with gr.Row():
                with gr.Column():
                    anon_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Accordion('Settings', open=False):
                        anon_method = gr.Dropdown(
                            choices=['gaussian', 'pixelate', 'median', 'blackout', 'elliptical'],
                            value='elliptical',
                            label='Blur Method',
                        )
                    anon_btn = gr.Button('Anonymize', variant='primary')
                with gr.Column():
                    anon_output = gr.Image(label='Anonymized')
                    anon_text = gr.Textbox(label='Details', lines=4, show_copy_button=True)

            anon_btn.click(
                anonymize_fn,
                inputs=[anon_image, anon_method],
                outputs=[anon_output, anon_text],
            )

            gr.Examples(
                examples=[[EXAMPLE_ANONYMIZE, 'elliptical']],
                inputs=[anon_image, anon_method],
                label='Try an example',
            )

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(allowed_paths=[str(_DEMO_ASSETS_DIR)])
