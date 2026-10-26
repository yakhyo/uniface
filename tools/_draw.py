# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Shared drawing helpers for the demo tools (attribute cards and face parsing overlays).

OpenCV-only, so the tools need nothing beyond UniFace's own dependencies. These live in
``tools/`` on purpose: they style the demos and are not part of the ``uniface`` API.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ACCENT = (225, 235, 120)  # BGR teal used for boxes, confidence bars and percentages
CARD_BG = (17, 17, 17)
CARD_ALPHA = 0.78
MUTED = (180, 180, 180)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def ui_scale(image: np.ndarray) -> float:
    """Scale factor for overlay sizes, relative to a 1080 px short edge."""
    return float(np.clip(min(image.shape[:2]) / 1080.0, 0.45, 2.0))


# --------------------------------------------------------------------------- boxes
def draw_corner_box(
    image: np.ndarray,
    bbox: np.ndarray | list[float],
    color: tuple[int, int, int] = ACCENT,
    thickness: int | None = None,
    corner_frac: float = 0.22,
) -> None:
    """Draw a thin full rectangle with bold corners, in place.

    Args:
        image: BGR image to draw on.
        bbox: Box as [x1, y1, x2, y2].
        color: Line color in BGR.
        thickness: Corner thickness in pixels. Defaults to a value scaled to the image.
        corner_frac: Corner length as a fraction of the box side.
    """
    x1, y1, x2, y2 = (round(v) for v in bbox[:4])
    t = thickness or max(2, round(4 * ui_scale(image)))
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, dst=image)
    lx, ly = int((x2 - x1) * corner_frac), int((y2 - y1) * corner_frac)
    for x, y, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(image, (x, y), (x + sx * lx, y), color, t, cv2.LINE_AA)
        cv2.line(image, (x, y), (x, y + sy * ly), color, t, cv2.LINE_AA)


# --------------------------------------------------------------------------- attribute cards
@dataclass
class Card:
    """Content of one attribute card.

    Attributes:
        title: Small grey caption, e.g. ``'Emotion'``.
        value: Main value in large type, e.g. ``'Happy'``.
        confidence: Optional score in [0, 1], shown as a percentage and a bar.
        subtitle: Optional second line, e.g. ``'Male · Age 37'``.
    """

    title: str
    value: str
    confidence: float | None = None
    subtitle: str | None = None


def _card_size(card: Card, s: float) -> tuple[int, int, dict]:
    pad = int(20 * s)
    sizes = {
        'title': (0.55 * s, max(1, round(1.2 * s))),
        'value': (1.05 * s, max(1, round(1.8 * s))),
        'pct': (0.6 * s, max(1, round(1.4 * s))),
        'sub': (0.6 * s, max(1, round(1.4 * s))),
    }

    def tw(text, key, font=FONT):
        sc, th = sizes[key]
        return cv2.getTextSize(text, font, sc, th)[0]

    w_title, h_title = tw(card.title, 'title')
    w_value, h_value = tw(card.value, 'value', FONT_BOLD)
    w_pct = tw(f'{card.confidence * 100:.0f}%', 'pct')[0] + int(14 * s) if card.confidence is not None else 0
    w_sub, h_sub = tw(card.subtitle, 'sub') if card.subtitle else (0, 0)
    width = max(w_title, w_value + w_pct, w_sub, int(220 * s)) + 2 * pad
    height = pad + h_title + int(14 * s) + h_value + pad // 2
    if card.confidence is not None:
        height += int(16 * s)
    if card.subtitle:
        height += h_sub + int(14 * s)
    height += pad // 2
    return width, height, {'pad': pad, 'sizes': sizes, 'h_title': h_title, 'h_value': h_value, 'h_sub': h_sub}


def _render_card(image: np.ndarray, card: Card, x: int, y: int, s: float) -> None:
    w, h, m = _card_size(card, s)
    H, W = image.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    # translucent rounded background, blended only inside the card's region
    roi = image[y0:y1, x0:x1]
    bg = roi.copy()
    r = int(18 * s)
    rx0, ry0, rx1, ry1 = x - x0, y - y0, x - x0 + w - 1, y - y0 + h - 1
    cv2.rectangle(bg, (rx0 + r, ry0), (rx1 - r, ry1), CARD_BG, -1)
    cv2.rectangle(bg, (rx0, ry0 + r), (rx1, ry1 - r), CARD_BG, -1)
    for cx, cy in ((rx0 + r, ry0 + r), (rx1 - r, ry0 + r), (rx0 + r, ry1 - r), (rx1 - r, ry1 - r)):
        cv2.circle(bg, (cx, cy), r, CARD_BG, -1, cv2.LINE_AA)
    cv2.addWeighted(bg, CARD_ALPHA, roi, 1 - CARD_ALPHA, 0, dst=roi)

    pad, sz = m['pad'], m['sizes']
    tx = x + pad
    ty = y + pad + m['h_title']
    cv2.putText(image, card.title, (tx, ty), FONT, sz['title'][0], MUTED, sz['title'][1], cv2.LINE_AA)
    ty += int(14 * s) + m['h_value']
    cv2.putText(image, card.value, (tx, ty), FONT_BOLD, sz['value'][0], WHITE, sz['value'][1], cv2.LINE_AA)
    if card.confidence is not None:
        pct = f'{card.confidence * 100:.0f}%'
        pw = cv2.getTextSize(pct, FONT, sz['pct'][0], sz['pct'][1])[0][0]
        cv2.putText(image, pct, (x + w - pad - pw, ty), FONT, sz['pct'][0], ACCENT, sz['pct'][1], cv2.LINE_AA)
        by = ty + int(12 * s)
        bh = max(2, int(5 * s))
        cv2.rectangle(image, (tx, by), (x + w - pad, by + bh), (70, 70, 70), -1)
        fill = tx + int((w - 2 * pad) * float(np.clip(card.confidence, 0, 1)))
        cv2.rectangle(image, (tx, by), (fill, by + bh), ACCENT, -1)
        ty = by + bh
    if card.subtitle:
        ty += int(14 * s) + m['h_sub']
        cv2.putText(image, card.subtitle, (tx, ty), FONT, sz['sub'][0], WHITE, sz['sub'][1], cv2.LINE_AA)


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_attribute_cards(
    image: np.ndarray,
    items: list[tuple[np.ndarray, Card]],
    *,
    draw_boxes: bool = True,
) -> None:
    """Draw a corner box and an attribute card for every face, in place.

    Cards are placed beside their face (right, then left), or below / above it when the
    sides are taken, avoiding the other faces and cards so several people stay readable.

    Args:
        image: BGR image to draw on.
        items: One ``(bbox, Card)`` per face, bbox as [x1, y1, x2, y2].
        draw_boxes: Also draw the corner box around each face.
    """
    s = ui_scale(image)
    H, W = image.shape[:2]
    gap, margin = int(16 * s), int(10 * s)
    boxes = [tuple(float(v) for v in b[:4]) for b, _ in items]
    if draw_boxes:
        for b in boxes:
            draw_corner_box(image, b)

    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(items)), key=lambda k: boxes[k][0])  # left -> right
    for rank, k in enumerate(order):
        x1, y1, x2, y2 = boxes[k]
        card = items[k][1]
        w, h, _ = _card_size(card, s)
        cx = (x1 + x2) / 2 - w / 2
        # outer side first when several faces sit next to each other
        sides = ['left', 'right'] if (len(order) > 1 and rank == 0) else ['right', 'left']
        candidates = {
            'right': (x2 + gap, y1),
            'left': (x1 - gap - w, y1),
            'below': (x1 if rank == 0 and len(order) > 1 else cx, y2 + gap),
            'above': (cx, y1 - gap - h),
        }
        if len(order) > 1 and rank == len(order) - 1:
            candidates['below'] = (x2 - w, y2 + gap)
        chosen = None
        for name in [*sides, 'below', 'above']:
            px, py = candidates[name]
            px = int(np.clip(px, margin, max(margin, W - w - margin)))
            py = int(np.clip(py, margin, max(margin, H - h - margin)))
            rect = (px, py, px + w, py + h)
            others = [boxes[j] for j in range(len(boxes)) if j != k]
            if any(_overlaps(rect, o) for o in placed + others) or _overlaps(rect, boxes[k]):
                continue
            chosen = (px, py)
            break
        if chosen is None:  # crowded frame: fall back to just below the face
            chosen = (int(np.clip(cx, margin, max(margin, W - w - margin))), int(min(y2 + gap, H - h - margin)))
        placed.append((chosen[0], chosen[1], chosen[0] + w, chosen[1] + h))
        _render_card(image, card, chosen[0], chosen[1], s)


# --------------------------------------------------------------------------- face parsing
# CelebAMask-HQ classes predicted by BiSeNet. Only face parts are coloured; background,
# neck, clothing and accessories stay untouched so there is no visible crop rectangle.
PARSING_COLORS_BGR = {
    1: (150, 196, 255),  # skin
    2: (255, 90, 120),  # left brow
    3: (255, 90, 120),  # right brow
    4: (255, 200, 90),  # left eye
    5: (255, 200, 90),  # right eye
    6: (90, 230, 255),  # eyeglasses
    7: (90, 140, 255),  # left ear
    8: (90, 140, 255),  # right ear
    10: (225, 235, 120),  # nose
    11: (135, 105, 255),  # mouth
    12: (110, 70, 255),  # upper lip
    13: (110, 70, 255),  # lower lip
    17: (255, 120, 170),  # hair
}


def _parsing_lut(alpha: float, skin_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lut = np.zeros((256, 3), np.float32)
    lut_a = np.zeros(256, np.float32)
    for cls, color in PARSING_COLORS_BGR.items():
        lut[cls] = color
        lut_a[cls] = skin_alpha if cls == 1 else alpha
    return lut, lut_a


def draw_face_parsing(
    image: np.ndarray,
    mask: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    *,
    alpha: float = 0.62,
    skin_alpha: float = 0.38,
    contours: bool = True,
) -> None:
    """Overlay a face-parsing mask on the image, in place.

    Args:
        image: BGR image to draw on (the full frame).
        mask: Class-id mask from ``BiSeNet.parse``. Either full-frame sized, or the size of
            ``region`` when the face was parsed from a crop.
        region: ``(x1, y1, x2, y2)`` of the crop the mask belongs to; None if full-frame.
        alpha: Opacity for face parts.
        skin_alpha: Opacity for skin, kept lighter so the face stays readable.
        contours: Draw thin white lines between parts.
    """
    x1, y1, x2, y2 = region if region is not None else (0, 0, image.shape[1], image.shape[0])
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    m = mask.astype(np.uint8)
    if m.shape[:2] != roi.shape[:2]:
        m = cv2.resize(m, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    lut, lut_a = _parsing_lut(alpha, skin_alpha)
    color = lut[m]
    a = lut_a[m]
    sigma = max(0.6, min(roi.shape[:2]) / 400)  # soften the class edges a touch
    color = cv2.GaussianBlur(color, (0, 0), sigma)
    a = cv2.GaussianBlur(a, (0, 0), sigma)[..., None]
    out = roi.astype(np.float32) * (1 - a) + color * a
    if contours:
        shown = np.where(lut_a[m] > 0, m, 0).astype(np.uint8)
        edges = cv2.Canny(shown, 0, 0) > 0
        out[edges] = out[edges] * 0.4 + 255 * 0.6
    roi[:] = np.clip(out, 0, 255).astype(np.uint8)
