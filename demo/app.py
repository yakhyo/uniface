# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# License: MIT
# UniFace Gradio Demo for Hugging Face Spaces

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gradio as gr
import gradio_client.utils as _gc_utils
import numpy as np

import uniface
from uniface.constants import (
    AdaFaceWeights,
    ArcFaceWeights,
    EdgeFaceWeights,
    GazeWeights,
    HeadPoseWeights,
    MiniFASNetWeights,
    MobileFaceWeights,
    MODNetWeights,
    ParsingWeights,
    RetinaFaceWeights,
    SCRFDWeights,
    SphereFaceWeights,
    XSegWeights,
    YOLOv5FaceWeights,
    YOLOv8FaceWeights,
)
from uniface.draw import (
    FACE_PARSING_COLORS,
    FACE_PARSING_LABELS,
    draw_detections,
    draw_gaze,
    draw_head_pose,
    draw_tracks,
    vis_parsing_maps,
)

# Workaround for gradio_client bug: ``_json_schema_to_python_type`` recurses into
# ``additionalProperties`` even when its value is a bool (a valid JSON Schema form),
# producing ``TypeError: argument of type 'bool' is not iterable`` when Gradio's
# SSR layer calls ``/info`` at boot. Patch defensively so any boolean schema
# resolves to ``Any`` instead of crashing.
_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return 'Any'
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

# ---------------------------------------------------------------------------
# Resolve asset paths relative to this script so examples work from any cwd
# ---------------------------------------------------------------------------
_DEMO_ASSETS_DIR = Path(__file__).resolve().parent / 'assets'
EXAMPLE_DEFAULT = str(_DEMO_ASSETS_DIR / 'image.jpg')
EXAMPLE_VER_IMG1 = str(_DEMO_ASSETS_DIR / 'verification' / 'image1.jpg')
EXAMPLE_VER_IMG2 = str(_DEMO_ASSETS_DIR / 'verification' / 'image2.jpg')
EXAMPLE_VER_IMG3 = str(_DEMO_ASSETS_DIR / 'verification' / 'image3.jpg')
EXAMPLE_ATTR_1 = str(_DEMO_ASSETS_DIR / 'attribute' / 'image1.jpg')
EXAMPLE_ATTR_2 = str(_DEMO_ASSETS_DIR / 'attribute' / 'image2.jpg')
EXAMPLE_GAZE_1 = str(_DEMO_ASSETS_DIR / 'gaze' / 'image1.jpg')
EXAMPLE_GAZE_2 = str(_DEMO_ASSETS_DIR / 'gaze' / 'image2.jpg')
EXAMPLE_HEADPOSE_1 = str(_DEMO_ASSETS_DIR / 'headpose' / 'image1.jpg')
EXAMPLE_HEADPOSE_2 = str(_DEMO_ASSETS_DIR / 'headpose' / 'image2.jpg')
EXAMPLE_HEADPOSE_3 = str(_DEMO_ASSETS_DIR / 'headpose' / 'image3.jpg')
EXAMPLE_MATTING_1 = str(_DEMO_ASSETS_DIR / 'matting' / 'image1.jpg')
EXAMPLE_MATTING_2 = str(_DEMO_ASSETS_DIR / 'matting' / 'image2.jpg')
EXAMPLE_PARSING_1 = str(_DEMO_ASSETS_DIR / 'parsing' / 'image1.jpg')
EXAMPLE_PARSING_2 = str(_DEMO_ASSETS_DIR / 'parsing' / 'image2.jpg')
EXAMPLE_SPOOF_1 = str(_DEMO_ASSETS_DIR / 'spoofing' / 'image1.jpg')
EXAMPLE_SPOOF_2 = str(_DEMO_ASSETS_DIR / 'spoofing' / 'image2.jpg')
EXAMPLE_SPOOF_3 = str(_DEMO_ASSETS_DIR / 'spoofing' / 'image3.jpg')
EXAMPLE_ANONYMIZE = str(_DEMO_ASSETS_DIR / 'anonymize' / 'image.jpg')

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
    'EdgeFace': [w.value for w in EdgeFaceWeights],
    'MobileFace': [w.value for w in MobileFaceWeights],
    'SphereFace': [w.value for w in SphereFaceWeights],
}

RECOGNIZER_METHOD_MAP: dict[str, str] = {
    'ArcFace': 'arcface',
    'AdaFace': 'adaface',
    'EdgeFace': 'edgeface',
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

HEADPOSE_VARIANT_MAP: dict[str, HeadPoseWeights] = {
    'ResNet18': HeadPoseWeights.RESNET18,
    'ResNet34': HeadPoseWeights.RESNET34,
    'ResNet50': HeadPoseWeights.RESNET50,
    'MobileNetV2': HeadPoseWeights.MOBILENET_V2,
    'MobileNetV3-Small': HeadPoseWeights.MOBILENET_V3_SMALL,
    'MobileNetV3-Large': HeadPoseWeights.MOBILENET_V3_LARGE,
}

PARSING_VARIANT_MAP: dict[str, ParsingWeights | XSegWeights] = {
    'BiSeNet (ResNet18) — 19 classes': ParsingWeights.RESNET18,
    'BiSeNet (ResNet34) — 19 classes': ParsingWeights.RESNET34,
    'XSeg — face mask': XSegWeights.DEFAULT,
}

SPOOFING_VARIANT_MAP: dict[str, MiniFASNetWeights] = {
    'V1SE': MiniFASNetWeights.V1SE,
    'V2': MiniFASNetWeights.V2,
}

MATTING_VARIANT_MAP: dict[str, MODNetWeights] = {
    'Photographic': MODNetWeights.PHOTOGRAPHIC,
    'Webcam': MODNetWeights.WEBCAM,
}

MATTING_BACKGROUNDS: dict[str, tuple[int, int, int] | str] = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'green': (0, 255, 0),
    'blur': 'blur',
    'alpha': 'alpha',
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
    EdgeFaceWeights,
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
    draw_detections(image=result, bboxes=bboxes, scores=scores, landmarks=landmarks, draw_score=True, corner_bbox=True)

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
        analyzer = uniface.FaceAnalyzer(detector=det, recognizer=None, attributes=[ag])
    else:
        ff = _get_model('fairface', uniface.FairFace)
        analyzer = uniface.FaceAnalyzer(detector=det, recognizer=None, attributes=[ff])

    faces = analyzer.analyze(bgr)
    result = bgr.copy()

    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]
    draw_detections(image=result, bboxes=bboxes, scores=scores, landmarks=landmarks, corner_bbox=True)

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
    is_xseg = isinstance(weights, XSegWeights)

    faces_json: dict = {}
    for i, face in enumerate(faces):
        if is_xseg:
            mask = parser.parse(bgr, landmarks=face.landmarks)  # (H, W) float in [0, 1]
            x1, y1, x2, y2 = map(int, face.bbox)
            mask_3 = np.clip(mask, 0.0, 1.0)[..., None]
            overlay_color = np.array([0, 255, 0], dtype=np.float32)
            overlay = mask_3 * overlay_color + (1.0 - mask_3) * bgr.astype(np.float32)
            blended = (0.55 * bgr.astype(np.float32) + 0.45 * overlay).astype(np.uint8)
            result = np.where(mask_3 > 0.05, blended, result).astype(np.uint8)
            faces_json[f'face_{i + 1}'] = {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'mask_mean': round(float(mask.mean()), 4),
                'face_pixels': int((mask > 0.5).sum()),
            }
        else:
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

    info: dict = {'model': model_variant, 'num_faces': len(faces), **faces_json}
    if not is_xseg:
        legend = {}
        for idx, name in enumerate(FACE_PARSING_LABELS):
            if idx == 0:
                continue
            r, g, b = FACE_PARSING_COLORS[idx]
            legend[name] = f'rgb({r}, {g}, {b})'
        info['legend'] = legend

    return _bgr_to_rgb(result), json.dumps(info, indent=2)


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
    draw_detections(image=result, bboxes=bboxes, scores=scores, landmarks=lmks, corner_bbox=True)

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
# Tab: Head Pose
# ===================================================================
def headpose_fn(
    image: np.ndarray,
    backbone: str,
    draw_type: str,
) -> tuple[np.ndarray, str]:
    if image is None:
        return None, ''

    bgr = _rgb_to_bgr(image)
    det = _get_model('det_retina_default', uniface.create_detector, 'retinaface')

    weights = HEADPOSE_VARIANT_MAP[backbone]
    estimator = _get_model(
        f'headpose_{weights.value}',
        uniface.create_head_pose_estimator,
        model_name=weights,
    )

    faces = det.detect(bgr)
    result = bgr.copy()

    faces_json = {}
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pose = estimator.estimate(crop)
        faces_json[f'face_{i + 1}'] = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'pitch_deg': round(float(pose.pitch), 2),
            'yaw_deg': round(float(pose.yaw), 2),
            'roll_deg': round(float(pose.roll), 2),
        }

        draw_head_pose(
            result,
            face.bbox,
            pitch=pose.pitch,
            yaw=pose.yaw,
            roll=pose.roll,
            draw_type=draw_type,
            draw_bbox=True,
            draw_angles=True,
        )

    return _bgr_to_rgb(result), json.dumps(
        {'backbone': backbone, 'draw_type': draw_type, 'num_faces': len(faces), **faces_json},
        indent=2,
    )


# ===================================================================
# Tab: Portrait Matting (MODNet)
# ===================================================================
def matting_fn(
    image: np.ndarray,
    model_variant: str,
    background: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if image is None:
        return None, None, ''

    bgr = _rgb_to_bgr(image)
    weights = MATTING_VARIANT_MAP[model_variant]
    matting = _get_model(f'modnet_{weights.value}', uniface.create_matting_model, weights)

    matte = matting.predict(bgr)  # float32 (H, W) in [0, 1]
    alpha = (matte * 255.0).clip(0, 255).astype(np.uint8)
    alpha_3ch = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)

    bg_choice = MATTING_BACKGROUNDS[background]
    if bg_choice == 'alpha':
        composited = alpha_3ch
    elif bg_choice == 'blur':
        bg_img = cv2.GaussianBlur(bgr, (0, 0), sigmaX=21, sigmaY=21)
        matte_3 = matte[..., None]
        composited = (bgr.astype(np.float32) * matte_3 + bg_img.astype(np.float32) * (1.0 - matte_3)).astype(np.uint8)
    else:
        bg_img = np.full_like(bgr, bg_choice, dtype=np.uint8)
        matte_3 = matte[..., None]
        composited = (bgr.astype(np.float32) * matte_3 + bg_img.astype(np.float32) * (1.0 - matte_3)).astype(np.uint8)

    info = {
        'model': model_variant,
        'background': background,
        'image_size': {'height': int(bgr.shape[0]), 'width': int(bgr.shape[1])},
        'alpha_mean': round(float(matte.mean()), 4),
        'alpha_max': round(float(matte.max()), 4),
        'foreground_pixels': int((matte > 0.5).sum()),
    }

    return _bgr_to_rgb(composited), alpha, json.dumps(info, indent=2)


# ===================================================================
# Tab: Face Tracking (Video)
# ===================================================================
def _assign_track_ids(faces, tracks: np.ndarray) -> list:
    """Match BYTETracker outputs back to ``Face`` objects by center distance."""
    if len(tracks) == 0 or len(faces) == 0:
        return []

    track_ids = tracks[:, 4].astype(int)
    track_centers = np.stack(
        [(tracks[:, 0] + tracks[:, 2]) * 0.5, (tracks[:, 1] + tracks[:, 3]) * 0.5],
        axis=1,
    )
    face_centers = np.array(
        [[(f.bbox[0] + f.bbox[2]) * 0.5, (f.bbox[1] + f.bbox[3]) * 0.5] for f in faces],
        dtype=np.float32,
    )

    for ti in range(len(tracks)):
        dists = np.sum((track_centers[ti] - face_centers) ** 2, axis=1)
        faces[int(np.argmin(dists))].track_id = int(track_ids[ti])

    return [f for f in faces if f.track_id is not None]


def tracking_fn(
    video_path: str | None,
    confidence: float,
    track_buffer: int,
    max_frames: int,
) -> tuple[str | None, str]:
    if not video_path:
        return None, json.dumps({'error': 'Please upload a video.'}, indent=2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, json.dumps({'error': f'Could not open video: {video_path}'}, indent=2)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = min(total_frames, max_frames) if max_frames > 0 else total_frames

    out_path = str(Path(video_path).with_name(Path(video_path).stem + '_tracked.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None, json.dumps({'error': 'Could not initialise video writer.'}, indent=2)

    detector = _get_model('det_retina_default', uniface.create_detector, 'retinaface')
    # Always use a fresh tracker so IDs restart each run
    tracker = uniface.BYTETracker(track_thresh=confidence, track_buffer=track_buffer)

    seen_ids: set[int] = set()
    processed = 0
    while processed < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)
        if faces:
            dets = np.array(
                [[*f.bbox, f.confidence] for f in faces if f.confidence >= confidence],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 5), dtype=np.float32)

        tracks = tracker.update(dets) if len(dets) else np.empty((0, 5))
        tracked_faces = _assign_track_ids(faces, tracks)
        for tf in tracked_faces:
            if tf.track_id is not None:
                seen_ids.add(int(tf.track_id))

        draw_tracks(image=frame, faces=tracked_faces)
        cv2.putText(
            frame,
            f'frame {processed + 1}  tracks: {len(tracked_faces)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        writer.write(frame)
        processed += 1

    cap.release()
    writer.release()

    info = {
        'frames_processed': processed,
        'video_total_frames': total_frames,
        'fps': round(float(fps), 2),
        'resolution': {'width': width, 'height': height},
        'unique_track_ids': sorted(seen_ids),
        'output_path': out_path,
    }
    return out_path, json.dumps(info, indent=2)


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
    det = _get_model('det_retina_default', uniface.create_detector, 'retinaface')
    blurrer = _get_model(f'blur_{blur_method}', uniface.BlurFace, method=blur_method)
    faces = det.detect(bgr)
    result = blurrer.anonymize(bgr, faces)

    return _bgr_to_rgb(result), json.dumps({'method': blur_method, 'num_faces': len(faces), 'status': 'done'}, indent=2)


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
            'Gaze · Head Pose · Portrait Matting · Tracking · '
            'Attributes · Anti-Spoofing · Anonymization'
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
                            value=next(iter(PARSING_VARIANT_MAP)),
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

            _parsing_default_variant = next(iter(PARSING_VARIANT_MAP))
            _parsing_xseg_variant = next(
                (k for k, v in PARSING_VARIANT_MAP.items() if isinstance(v, XSegWeights)),
                _parsing_default_variant,
            )
            gr.Examples(
                examples=[
                    [EXAMPLE_PARSING_1, _parsing_default_variant],
                    [EXAMPLE_PARSING_2, _parsing_xseg_variant],
                ],
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

        # ------ Tab: Head Pose ------
        with gr.Tab('Head Pose'):
            gr.Markdown('Estimate yaw, pitch, and roll of the head and visualize as a 3D cube or axis.')
            with gr.Row():
                with gr.Column():
                    hp_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Accordion('Settings', open=False):
                        hp_backbone = gr.Dropdown(
                            choices=list(HEADPOSE_VARIANT_MAP.keys()),
                            value='ResNet18',
                            label='Backbone',
                        )
                        hp_draw_type = gr.Radio(
                            choices=['cube', 'axis'],
                            value='cube',
                            label='Visualization',
                        )
                    hp_btn = gr.Button('Estimate Head Pose', variant='primary')
                with gr.Column():
                    hp_output = gr.Image(label='Result')
                    hp_text = gr.Textbox(label='Pose Angles', lines=8, show_copy_button=True)

            hp_btn.click(
                headpose_fn,
                inputs=[hp_image, hp_backbone, hp_draw_type],
                outputs=[hp_output, hp_text],
            )

            gr.Examples(
                examples=[
                    [EXAMPLE_HEADPOSE_1, 'ResNet18', 'cube'],
                    [EXAMPLE_HEADPOSE_2, 'ResNet18', 'axis'],
                    [EXAMPLE_HEADPOSE_3, 'MobileNetV2', 'cube'],
                ],
                inputs=[hp_image, hp_backbone, hp_draw_type],
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

        # ------ Tab: Portrait Matting ------
        with gr.Tab('Portrait Matting'):
            gr.Markdown(
                'Trimap-free portrait matting with MODNet. Returns a soft alpha matte and '
                'composites the foreground over a chosen background.',
            )
            with gr.Row():
                with gr.Column():
                    mat_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Accordion('Settings', open=False):
                        mat_variant = gr.Radio(
                            choices=list(MATTING_VARIANT_MAP.keys()),
                            value='Photographic',
                            label='Model Variant',
                        )
                        mat_background = gr.Dropdown(
                            choices=list(MATTING_BACKGROUNDS.keys()),
                            value='white',
                            label='Background',
                        )
                    mat_btn = gr.Button('Run Matting', variant='primary')
                with gr.Column():
                    mat_output = gr.Image(label='Composited Result')
                    mat_alpha = gr.Image(label='Alpha Matte')
                    mat_text = gr.Textbox(label='Details', lines=8, show_copy_button=True)

            mat_btn.click(
                matting_fn,
                inputs=[mat_image, mat_variant, mat_background],
                outputs=[mat_output, mat_alpha, mat_text],
            )

            gr.Examples(
                examples=[
                    [EXAMPLE_MATTING_1, 'Photographic', 'white'],
                    [EXAMPLE_MATTING_2, 'Photographic', 'blur'],
                ],
                inputs=[mat_image, mat_variant, mat_background],
                label='Try an example',
            )

        # ------ Tab: Face Tracking (Video) ------
        with gr.Tab('Face Tracking'):
            gr.Markdown(
                'Multi-face tracking on a short video using ByteTrack. '
                'Each track gets a persistent ID and a unique colour across frames.',
            )
            with gr.Row():
                with gr.Column():
                    trk_video = gr.Video(label='Input Video', sources=['upload'])
                    with gr.Accordion('Settings', open=False):
                        trk_conf = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label='Detection Confidence')
                        trk_buffer = gr.Slider(5, 90, value=30, step=1, label='Track Buffer (frames)')
                        trk_max_frames = gr.Slider(0, 600, value=300, step=10, label='Max Frames (0 = all)')
                    trk_btn = gr.Button('Run Tracking', variant='primary')
                with gr.Column():
                    trk_output = gr.Video(label='Tracked Result')
                    trk_text = gr.Textbox(label='Tracking Stats', lines=8, show_copy_button=True)

            trk_btn.click(
                tracking_fn,
                inputs=[trk_video, trk_conf, trk_buffer, trk_max_frames],
                outputs=[trk_output, trk_text],
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
    app.launch(allowed_paths=[str(_DEMO_ASSETS_DIR)], ssr_mode=False, show_api=False)
