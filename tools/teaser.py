# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Teaser videos for UniFace features: before/after wipe, social-media formats, showcases.

One run of a feature does the whole job: UniFace inference on every frame, then a render with a
minimal white title card, a before/after wipe panel and a ``pip install`` outro, in three formats:

- ``9x16`` (1080x1920) for Reels, Shorts and TikTok
- ``1x1`` (1080x1080) for feed posts, LinkedIn, X and READMEs
- ``16x9`` (1920x1080) for YouTube: title column on the left, panel on the right

In 9x16 and 16x9 the models (or options) behind each feature appear as chips once the output is revealed;
the one running in the demo is filled teal (``Feature.chips``). 9x16 keeps the bottom 320 px clear for the
Reels / Shorts caption and buttons.

``--output-only`` instead writes the whole source frame with just the overlay (no crop, wipe or
cards), at the source frame rate and frame count, so it lines up with the original clip in an editor.

Usage:
    python tools/teaser.py detection --source clip.mp4
    python tools/teaser.py emotion --source clip.mp4 --crop 1900 --y-bias 0.1 --start 0.8 --end 13.8
    python tools/teaser.py recognition --source clip.mp4 --enroll-frame 277 --enroll-face 1
    python tools/teaser.py recognition --source clip.mp4 --enroll-image person.jpg
    python tools/teaser.py mesh --source clip.mp4 --output-only
    python tools/teaser.py concat a.mp4@0-4.2 b.mp4@0.4-6.2 c.mp4 --output states_source.mp4
    python tools/teaser.py showcase --teasers teasers/ --kind all   # + chapter timestamps for the full one

Features: detection, landmarks, mesh, tracking, recognition, attributes, emotion, headpose, gaze,
states, parsing, segmentation, matting, anonymization.

Framing: wide sources are cropped to a square that follows the faces. ``--crop`` sets the crop size
in source pixels (default: the short edge) and ``--y-bias`` moves the crop below the faces by that
fraction of the crop (keeps hands, phones or a mask strap in shot).

Settings used for the published teasers (sources listed in teaser_videos_final/SOURCES.md):

    detection     --crop 2160 --y-bias 0.15         landmarks     --crop 900 --y-bias 0.12
    emotion       --crop 1900 --y-bias 0.1 --start 0.8 --end 13.8
    recognition   --crop 1800 --y-bias 0.08 --enroll-frame 277 --enroll-face 1
    segmentation  --crop 1800 --y-bias 0.1          parsing       --crop 1800 --y-bias 0.05
    attributes    --crop 2160 --y-bias 0.12         gaze          --crop 1360 --y-bias 0.1  (4K source)
    states        source built with ``concat`` from four clips (sunglasses, eyes, mask, glasses)
    mesh, tracking, anonymization, headpose, matting: defaults

Fonts: the cards use Inter (Regular, Medium, SemiBold) and a monospace font. Point ``--font-dir`` at
a folder with ``Inter-*.otf``/``.ttf`` if Inter is not installed; DejaVu or Pillow's default is used
otherwise. Requires ffmpeg on PATH and Pillow.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import cache
from itertools import pairwise
import json
from pathlib import Path
import subprocess  # nosec B404
import sys
from typing import ClassVar

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from uniface.detection import SCRFD
from uniface.types import Face

# ---------------------------------------------------------------- style
P = 1080  # panel size (square)
FPS = 25  # teaser frame rate; output-only videos keep the source rate
INK, MUTED = (17, 17, 17), (110, 110, 110)
ACCENT = (120, 235, 225)  # teal used for overlays
PALETTE = [
    (120, 235, 225), (255, 179, 71), (255, 105, 135), (140, 160, 255), (170, 230, 110),
    (255, 230, 90), (215, 140, 255), (90, 200, 255), (255, 140, 90), (120, 255, 180),
]  # fmt: skip
URL = 'github.com/yakhyo/uniface'
PIP = 'pip install uniface'
S = 16  # sub-pixel precision for cv2 drawing (shift=4)

FONT_DIRS = [
    '/usr/share/fonts/opentype/inter',
    '/usr/share/fonts/truetype/inter',
    '~/Library/Fonts',
    '/Library/Fonts',
    '~/.local/share/fonts',
    '/usr/share/fonts/truetype/dejavu',
    '/System/Library/Fonts',
    'C:/Windows/Fonts',
]


@cache
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """Inter at the given weight ('Regular', 'Medium', 'SemiBold') or a monospace font ('Mono')."""
    if weight == 'Mono':
        names = ['DejaVuSansMono.ttf', 'Menlo.ttc', 'consola.ttf']
    else:
        bold = weight == 'SemiBold'
        names = [f'Inter-{weight}.otf', f'Inter-{weight}.ttf', f'Inter_18pt-{weight}.ttf']
        names += ['DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', 'arialbd.ttf' if bold else 'arial.ttf']
    for name in names:
        for d in FONT_DIRS:
            path = Path(d).expanduser() / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


class Layer:
    """RGBA float layer covering rows [y0, y0 + h) of a ``P`` px wide block; ``dx`` shifts it on wider canvases."""

    def __init__(self, rgba: np.ndarray, y0: int):
        self.rgb, self.a, self.y0 = rgba[..., :3], rgba[..., 3:4], y0

    @staticmethod
    def draw(width: int, height: int, fn) -> Layer:
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        fn(ImageDraw.Draw(img))
        arr = np.asarray(img).astype(np.float32) / 255.0
        rows = np.where(arr[..., 3].max(axis=1) > 0)[0]
        if len(rows) == 0:
            return Layer(arr[:1], 0)
        return Layer(arr[rows[0] : rows[-1] + 1].copy(), int(rows[0]))

    def scaled(self, s: float) -> Layer:
        """The layer resized by ``s`` (top-left anchored, ``y0`` kept); blend it with ``dx`` to re-centre."""
        h, w = self.rgb.shape[:2]
        size = (max(1, round(w * s)), max(1, round(h * s)))
        rgba = cv2.resize(np.concatenate([self.rgb, self.a], axis=2), size, interpolation=cv2.INTER_AREA)
        return Layer(rgba, self.y0)

    def blend(self, base: np.ndarray, alpha: float = 1.0, dy: int = 0, dx: int = 0) -> np.ndarray:
        if alpha <= 0:
            return base
        y, (h, w) = self.y0 + dy, self.rgb.shape[:2]
        s0, s1 = max(0, -y), min(h, base.shape[0] - y)
        t0, t1 = max(0, -dx), min(w, base.shape[1] - dx)
        if s1 > s0 and t1 > t0:
            a = self.a[s0:s1, t0:t1] * alpha
            region = base[y + s0 : y + s1, dx + t0 : dx + t1]
            base[y + s0 : y + s1, dx + t0 : dx + t1] = region * (1 - a) + self.rgb[s0:s1, t0:t1] * a
        return base


def fit_font(d: ImageDraw.ImageDraw, text: str, weight: str, size: int, max_w: int) -> ImageFont.FreeTypeFont:
    f = font(weight, size)
    while d.textlength(text, font=f) > max_w and size > 20:
        size -= 2
        f = font(weight, size)
    return f


def centered(d: ImageDraw.ImageDraw, y: float, text: str, f, fill, width: int = P) -> None:
    d.text(((width - d.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)


def text_layer(y: float, text: str, weight: str, size: int, color, height: int = P, max_w: int = 980) -> Layer:
    return Layer.draw(P, height, lambda d: centered(d, y, text, fit_font(d, text, weight, size, max_w), color))


def pill_label(d: ImageDraw.ImageDraw, text: str, x: float, y: float, anchor: str = 'left', size: int = 30):
    f = font('Medium', size)
    w, h = d.textlength(text, font=f) + 36, size + 26
    if anchor == 'right':
        x = P - x - w
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=(20, 20, 20, 150))
    d.text((x + 18, y + 8), text, font=f, fill=(255, 255, 255, 255))


_SPRITES: dict = {}


def tag_sprite(text: str, color=(20, 20, 20), fg=(255, 255, 255), size: int = 26, alpha: float = 0.85) -> np.ndarray:
    """Small rounded tag (cached), float RGBA."""
    key = (text, tuple(color), tuple(fg), size, alpha)
    if key not in _SPRITES:
        f = font('SemiBold', size)
        w, h = int(ImageDraw.Draw(Image.new('RGBA', (1, 1))).textlength(text, font=f)) + 24, size + 16
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(*color, int(255 * alpha)))
        d.text((12, 5), text, font=f, fill=(*fg, 255))
        _SPRITES[key] = np.asarray(img).astype(np.float32) / 255.0
    return _SPRITES[key]


def blit(img: np.ndarray, sprite: np.ndarray, x: float, y: float, alpha: float = 1.0) -> np.ndarray:
    h, w = sprite.shape[:2]
    x, y = round(float(x)), round(float(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 > x0 and y1 > y0:
        s = sprite[y0 - y : y1 - y, x0 - x : x1 - x]
        a = s[..., 3:4] * alpha
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a
    return img


def ip(p) -> tuple[int, int]:
    return tuple(round(float(v) * S) for v in p)


def corner_box(img: np.ndarray, box, color, thick: int = 4, frac: float = 0.22, thin_alpha: float = 0.45):
    """Thin full rectangle plus bold corners. Colour in 0-255 RGB."""
    x1, y1, x2, y2 = (float(v) for v in box)
    c = tuple(v / 255.0 for v in color)
    ov = img.copy()
    cv2.rectangle(ov, ip((x1, y1)), ip((x2, y2)), c, 1, cv2.LINE_AA, shift=4)
    img[:] = img * (1 - thin_alpha) + ov * thin_alpha
    lx, ly = (x2 - x1) * frac, (y2 - y1) * frac
    for x, y, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, ip((x, y)), ip((x + sx * lx, y)), c, thick, cv2.LINE_AA, shift=4)
        cv2.line(img, ip((x, y)), ip((x, y + sy * ly)), c, thick, cv2.LINE_AA, shift=4)
    return img


def glow_draw(base: np.ndarray, draw_fn, sigma: float = 4, k: float = 0.9) -> np.ndarray:
    """Draw with cv2 into an empty colour layer + alpha mask, then composite with a soft glow."""
    layer = np.zeros_like(base)
    alpha = np.zeros(base.shape[:2], np.float32)
    draw_fn(layer, alpha)
    g = cv2.GaussianBlur(alpha, (0, 0), sigma) * k
    a = np.clip(alpha + g, 0, 1)[..., None]
    gc = cv2.GaussianBlur(layer, (0, 0), sigma)
    gw = cv2.GaussianBlur(alpha, (0, 0), sigma)[..., None] + 1e-6
    color = np.where(alpha[..., None] > 0.01, layer / np.maximum(alpha[..., None], 1e-6), gc / gw)
    return base * (1 - a) + np.clip(color, 0, 1) * a


def card(title: str, rows: list, width: int = 300) -> np.ndarray:
    """Dark rounded card: grey title, then rows of (label, value, bar in [0, 1] or None, colour)."""
    f_t, f_l, f_v = font('Regular', 24), font('SemiBold', 34), font('Medium', 24)
    h = 52 + 62 * len(rows) + 6
    img = Image.new('RGBA', (width, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, width - 1, h - 1), radius=22, fill=(17, 17, 17, 205))
    d.text((22, 14), title, font=f_t, fill=(180, 180, 180, 255))
    y = 46
    for label, value, bar, col in rows:
        d.text((22, y), label, font=f_l if len(rows) == 1 else f_v, fill=(255, 255, 255, 255))
        if value is not None:
            d.text((width - 22 - d.textlength(value, font=f_v), y + 2), value, font=f_v, fill=(*col, 255))
        if bar is not None:
            by = y + (44 if len(rows) == 1 else 34)
            d.rounded_rectangle((22, by, width - 22, by + 6), radius=3, fill=(70, 70, 70, 255))
            bw = (width - 44) * float(np.clip(bar, 0, 1))
            d.rounded_rectangle((22, by, 22 + bw, by + 6), radius=3, fill=(*col, 255))
        y += 62
    return np.asarray(img).astype(np.float32) / 255.0


def backdrop(w: int, h: int) -> np.ndarray:
    """Teal vertical gradient used behind mattes and masks."""
    yy = np.linspace(0, 1, h)[:, None, None]
    grad = (np.array([62, 160, 166]) * (1 - yy) + np.array([24, 78, 84]) * yy) / 255.0
    return np.broadcast_to(grad, (h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------- wipe panel + video writers
SPLIT_KEYS = [(0.0, 1.0), (1.2, 1.0), (2.4, 0.5), (7.0, 0.5), (8.2, 0.0), (999, 0.0)]  # sweep, hold half, sweep
SPLIT_KEYS_SHORT = [(0.0, 1.0), (0.8, 1.0), (1.8, 0.5), (4.4, 0.5), (5.4, 0.0), (999, 0.0)]  # clips < 260 frames
SINGLE_KEYS = [(0.0, 1.0), (1.2, 1.0), (2.4, 0.0), (999, 0.0)]  # one sweep: the divider never parks on a face


def divider_at(t: float, keys) -> float:
    for (t0, v0), (t1, v1) in pairwise(keys):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0))
    return keys[-1][1]


def reveal_time(keys) -> float:
    """Time when the wipe has fully revealed the output."""
    return keys[4][0] if len(keys) >= 6 else keys[2][0]


class Panel:
    """Before/after wipe with an 'Input' / output label pair, then the full output with its own label."""

    def __init__(self, label_in: str, label_out: str, label_full: str, keys):
        self.keys = keys
        self.lbl_in = Layer.draw(P, P, lambda d: pill_label(d, label_in, 40, P - 100))
        self.lbl_out = Layer.draw(P, P, lambda d: pill_label(d, label_out, 40, P - 100, 'right'))
        self.lbl_full = Layer.draw(P, P, lambda d: pill_label(d, label_full, 40, P - 100))
        f = font('SemiBold', 36)
        self.corner = Layer.draw(
            P, P, lambda d: d.text((P - 40 - d.textlength('uniface', font=f), 30), 'uniface', font=f, fill=(255,) * 4)
        )
        self._yy, self._xx = np.ogrid[:P, :P]
        if len(keys) >= 6:
            self.t_split, self.t_full = (keys[2][0] - 0.2, keys[3][0] - 0.2), keys[4][0] + 0.2
        else:
            self.t_split, self.t_in_end, self.t_full = None, keys[1][0], keys[2][0] + 0.1

    def compose(self, inp: np.ndarray, out: np.ndarray, t: float, corner: bool) -> np.ndarray:
        x = round(divider_at(t, self.keys) * P)
        if x >= P:
            img = inp.copy()
        elif x <= 0:
            img = out.copy()
        else:
            img = inp.copy()
            img[:, x:] = out[:, x:]
            img[:, max(x - 2, 0) : x + 2] = 1.0
            d2 = (self._yy - P // 2) ** 2 + (self._xx - x) ** 2
            img[d2 <= 26**2] = 1.0
            img[d2 <= 20**2] = np.array(INK, np.float32) / 255.0
        if self.t_split:
            a = ease((t - self.t_split[0]) / 0.4) * (1 - ease((t - self.t_split[1]) / 0.4))
            self.lbl_in.blend(img, a)
            self.lbl_out.blend(img, a)
        else:
            self.lbl_in.blend(img, ease(t / 0.4) * (1 - ease((t - self.t_in_end + 0.2) / 0.3)))
        self.lbl_full.blend(img, ease((t - self.t_full) / 0.5))
        if corner:
            self.corner.blend(img)
        return img


class Encoder:
    """Raw RGB frames -> H.264 mp4 through ffmpeg."""

    def __init__(self, path: str, width: int, height: int, fps: str = str(FPS), crf: int = 18):
        self.ff = subprocess.Popen(  # nosec B603 B607
            ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
             '-r', fps, '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', path],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    def emit(self, img: np.ndarray) -> None:
        self.ff.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())

    def close(self) -> None:
        self.ff.stdin.close()
        self.ff.wait()


class Chips:
    """A small caps label over a group of pills naming the models (or options) behind a feature.

    The pills appear one after another (fade + short slide up) from ``t0``; pills marked as shown in the
    demo then fill teal. ``one_row`` keeps a single row and folds what does not fit into '+N more';
    otherwise the pills wrap into rows of balanced width. Drawn ``P`` px wide, centred; shift with ``dx``.
    """

    LINE, WHITE = (205, 205, 205), (255, 255, 255)

    def __init__(self, label: str, items, label_y: int, chip_y: int, size: int, max_w: int, one_row: bool,
                 height: int, divider_y: int | None = None, gap: int = 12, row_gap: int = 12):  # fmt: skip
        f = font('Medium', size)
        measure = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        h = size + 22
        width = {text: measure.textlength(text, font=f) + 34 for text, _ in items}
        if one_row:
            n = len(items)

            def row_w(k):
                w = sum(width[t] for t, _ in items[:k]) + gap * (k - 1)
                return w + (gap + measure.textlength(f'+{len(items) - k} more', font=f) + 34 if k < len(items) else 0)

            while n > 1 and row_w(n) > max_w:
                n -= 1
            more = f'+{len(items) - n} more'
            width[more] = measure.textlength(more, font=f) + 34
            rows = [list(items[:n]) + ([(more, False)] if n < len(items) else [])]
        else:
            rows = self._wrap(items, width, gap, max_w)
            while max_w > 200 and len(self._wrap(items, width, gap, max_w - 10)) == len(rows):
                max_w -= 10  # narrowest width with the same row count: no pill left alone on a row
                rows = self._wrap(items, width, gap, max_w)

        def head(d):
            if divider_y is not None:
                d.line((P / 2 - 40, divider_y, P / 2 + 40, divider_y), fill=(*self.LINE, 255), width=2)
            centered(d, label_y, label, font('SemiBold', 22), MUTED)

        self.label = Layer.draw(P, height, head)
        self.pills, self.fills = [], []
        y = chip_y
        for row in rows:
            x = (P - (sum(width[t] for t, _ in row) + gap * (len(row) - 1))) / 2
            for text, shown in row:
                box = (x, y, x + width[text], y + h)

                def pill(d, box=box, text=text, fill=None):
                    if fill:
                        d.rounded_rectangle(box, radius=h // 2, fill=(*fill, 255))
                    else:
                        d.rounded_rectangle(box, radius=h // 2, outline=(*self.LINE, 255), width=2,
                                            fill=(*self.WHITE, 255))  # fmt: skip
                    d.text((box[0] + 17, box[1] + 9), text, font=f, fill=(*INK, 255))

                self.pills.append(Layer.draw(P, height, pill))
                if shown:
                    self.fills.append(Layer.draw(P, height, lambda d, pill=pill: pill(d, fill=ACCENT)))
                x += width[text] + gap
            y += h + row_gap
        self.bottom = y - row_gap

    @staticmethod
    def _wrap(items, width, gap, max_w):
        rows, row, w = [], [], 0.0
        for text, shown in items:
            if row and w + gap + width[text] > max_w:
                rows.append(row)
                row, w = [], 0.0
            w = width[text] if not row else w + gap + width[text]
            row.append((text, shown))
        return [*rows, row]

    def blend(self, img: np.ndarray, t: float, t0: float, dy: int = 0, dx: int = 0) -> np.ndarray:
        self.label.blend(img, ease_out((t - t0) / 0.4), dy, dx)
        for k, layer in enumerate(self.pills):
            a = ease_out((t - t0 - 0.15 - 0.07 * k) / 0.4)
            layer.blend(img, a, dy + round(12 * (1 - a)), dx)
        t_fill = t0 + 0.15 + 0.07 * len(self.pills) + 0.35
        for layer in self.fills:
            layer.blend(img, ease_out((t - t_fill) / 0.35), dy, dx)
        return img


EXTRAS = ('ALSO IN UNIFACE', (('Anti-spoofing', False), ('Face quality', False), ('Face search (FAISS)', False),
                              ('FaceAnalyzer', False)))  # fmt: skip

SIZES = {'9x16': (P, 1920), '1x1': (P, P), '16x9': (1920, P)}  # (width, height)
WIDE_CARD_DX, WIDE_COL_DX, WIDE_X0 = 420, -120, 1920 - P  # 16x9: centred card, title column centre 420, panel x
WIDE_HEAD_DY = -118  # 16x9: the title block sits higher in its column, leaving room for the model chips
REEL_Y0 = 330  # 9x16: panel top; header above it, chips + url below, all clear of the bottom 320 px (Reels UI)


class TeaserWriter:
    """Title card -> wipe panel -> outro, in 1x1, 9x16 and/or 16x9. Call ``frame(inp, out)`` per frame.

    9x16: the title glides up into a header above the panel. 16x9: it glides left (and up) into a title
    column beside the panel (right). 1x1: the title card cross-fades into the panel. In 9x16 and 16x9 the
    feature's model chips appear under the panel / title once the wipe has revealed the output.
    """

    INTRO, OUTRO, XFADE = int(2.2 * FPS), int(2.6 * FPS), int(0.6 * FPS)
    REEL_CARD_DY, REEL_HEAD_TOP, REEL_HEAD_S = 420, 100, 0.92  # 9x16 header: top of 'uniface', scale

    def __init__(self, prefix: str, title: str, subtitle: str, labels, n_frames: int, keys, formats, chips=('', ())):
        self.n, self.i = n_frames, 0
        self.panel = Panel(*labels, keys=keys)
        self.t_chips = reveal_time(keys) - 0.6  # with the last sweep, so they have time to be read
        self.enc = {fmt: Encoder(f'{prefix}_{fmt}.mp4', *SIZES[fmt]) for fmt in formats}
        self.t_word = text_layer(418, 'uniface', 'Medium', 34, MUTED)
        self.t_title = text_layer(468, title, 'SemiBold', 96, INK)
        self.t_sub = text_layer(602, subtitle, 'Regular', 36, MUTED)
        self.o_pip = Layer.draw(P, P, lambda d: centered(d, 476, PIP, font('Mono', 50), INK))
        self.o_url = Layer.draw(P, P, lambda d: centered(d, 566, URL, font('Regular', 34), MUTED))
        label, items = chips
        # 9x16: the title card glides up and shrinks into a compact header; chips and url under the panel
        self.r_card = (self.t_word, self.t_title, self.t_sub)
        self.r_head = [layer.scaled(self.REEL_HEAD_S) for layer in self.r_card]
        self.r_chips = Chips(label, items, 1436, 1472, 23, 1040, one_row=True, height=1920) if items else None
        url_y = self.r_chips.bottom + 14 if items else 1450
        self.r_url = Layer.draw(P, 1920, lambda d: centered(d, url_y, URL, font('Regular', 30), MUTED))
        # 16x9: the same title block, fitted to the 840 px column it glides into; chips below it
        self.w_text = [
            text_layer(418, 'uniface', 'Medium', 34, MUTED, max_w=720),
            text_layer(468, title, 'SemiBold', 96, INK, max_w=720),
            text_layer(602, subtitle, 'Regular', 36, MUTED, max_w=720),
        ]
        self.w_chips = Chips(label, items, 598, 642, 26, 700, one_row=False, height=P, divider_y=574) if items else None
        self.w_url = Layer.draw(P, 120, lambda d: centered(d, 40, URL, font('Regular', 30), MUTED))
        for k in range(self.INTRO):
            for fmt, enc in self.enc.items():
                enc.emit(self.intro(fmt, k / FPS))

    def _canvas(self, fmt: str) -> np.ndarray:
        w, h = SIZES[fmt]
        return np.ones((h, w, 3), np.float32)

    def intro(self, fmt: str, t: float) -> np.ndarray:
        img = self._canvas(fmt)
        dy, dx = self.REEL_CARD_DY if fmt == '9x16' else 0, WIDE_CARD_DX if fmt == '16x9' else 0
        layers = self.w_text if fmt == '16x9' else (self.t_word, self.t_title, self.t_sub)
        for layer, t0 in zip(layers, (0.1, 0.25, 0.45), strict=True):
            layer.blend(img, ease_out((t - t0) / 0.6), dy, dx)
        return img

    def outro(self, fmt: str, t: float) -> np.ndarray:
        img = self._canvas(fmt)
        dy, dx = self.REEL_CARD_DY if fmt == '9x16' else 0, WIDE_CARD_DX if fmt == '16x9' else 0
        self.o_pip.blend(img, ease_out(t / 0.5), dy, dx)
        self.o_url.blend(img, ease_out((t - 0.25) / 0.5), dy, dx)
        return img

    def frame(self, inp: np.ndarray, out: np.ndarray) -> None:
        i, t = self.i, self.i / FPS
        a_in = ease(i / self.XFADE)
        a_out = ease((i - (self.n - self.XFADE)) / self.XFADE) if i >= self.n - self.XFADE else 0.0
        a_move, a_pan = ease(i / (0.5 * FPS)), ease((i - 0.4 * FPS) / (0.5 * FPS))
        for fmt, enc in self.enc.items():
            if fmt == '1x1':
                img = self.panel.compose(inp, out, t, corner=True)
                if a_in < 1:
                    img = self.intro(fmt, 9) * (1 - a_in) + img * a_in
            elif fmt == '16x9':  # the title glides left (and up) into its column, then the panel fades in beside it
                img = self._canvas(fmt)
                pan = self.panel.compose(inp, out, t, corner=False)
                img[:, WIDE_X0:] = 1.0 * (1 - a_pan) + pan * a_pan
                dx = round(WIDE_CARD_DX + (WIDE_COL_DX - WIDE_CARD_DX) * a_move)
                for layer in self.w_text:
                    layer.blend(img, 1.0, round(WIDE_HEAD_DY * a_move), dx)
                if self.w_chips:
                    self.w_chips.blend(img, t, self.t_chips, 0, WIDE_COL_DX)
                self.w_url.blend(img, a_pan, 950, WIDE_COL_DX)
            else:  # the title glides up and settles as a smaller header, then the panel fades in underneath it
                img = self._canvas(fmt)
                pan = self.panel.compose(inp, out, t, corner=False)
                img[REEL_Y0 : REEL_Y0 + P] = 1.0 * (1 - a_pan) + pan * a_pan
                self._reel_title(img, a_move)
                if self.r_chips:
                    self.r_chips.blend(img, t, self.t_chips)
                self.r_url.blend(img, a_pan)
            if a_out > 0:
                img = img * (1 - a_out) + self.outro(fmt, 0) * a_out
            enc.emit(img)
        self.i += 1

    def _reel_title(self, img: np.ndarray, a: float) -> None:
        """Title block at progress ``a`` from the centred card (0) to the header above the panel (1)."""
        s = 1 + (self.REEL_HEAD_S - 1) * a
        top = self.r_card[0].y0
        for card, head in zip(self.r_card, self.r_head, strict=True):
            layer = head if a >= 1 else card.scaled(s) if a > 0 else card
            y_card = card.y0 + self.REEL_CARD_DY
            y_head = self.REEL_HEAD_TOP + (card.y0 - top) * self.REEL_HEAD_S
            y = y_card + (y_head - y_card) * a
            layer.blend(img, 1.0, round(y) - layer.y0, round((P - layer.rgb.shape[1]) / 2))

    def close(self) -> None:
        for k in range(self.OUTRO):
            for fmt, enc in self.enc.items():
                enc.emit(self.outro(fmt, k / FPS))
        for enc in self.enc.values():
            enc.close()


class OutputWriter:
    """Output-only video: the drawn frame as is, at the source frame rate."""

    def __init__(self, path: str, fps: str):
        self.path, self.fps, self.enc = path, fps, None

    def frame(self, inp: np.ndarray, out: np.ndarray) -> None:
        if self.enc is None:
            self.enc = Encoder(self.path, out.shape[1], out.shape[0], fps=self.fps, crf=16)
        self.enc.emit(out)

    def close(self) -> None:
        if self.enc is not None:
            self.enc.close()


# ---------------------------------------------------------------- framing
class Framer:
    """Map source pixels into the drawing canvas.

    Teaser mode: a square crop of ``side`` source px, resized to ``P``, whose centre follows the
    faces (Gaussian-smoothed over ``sigma_s`` seconds). Output-only mode: the whole frame with its
    short edge resized to ``P``.
    """

    def __init__(self, src_w: int, src_h: int, rows: list, side: float | None = None, y_bias: float = 0.0,
                 sigma_s: float = 1.2, full: bool = False):  # fmt: skip
        self.full = full
        if full:
            self.s = P / min(src_w, src_h)
            self.W, self.H = round(src_w * self.s / 2) * 2, round(src_h * self.s / 2) * 2
            self.ox = self.oy = np.zeros(1)
            return
        self.W = self.H = P
        side = side or min(src_w, src_h)
        self.s = P / side
        n = max(len(rows), 1)
        cx, cy = np.full(n, np.nan), np.full(n, np.nan)
        for k, row in enumerate(rows):
            if row['faces']:
                b = np.array([f['bbox'] for f in row['faces']])
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                cx[k] = np.average((b[:, 0] + b[:, 2]) / 2, weights=area)
                cy[k] = np.average((b[:, 1] + b[:, 3]) / 2, weights=area)
        sig = sigma_s * FPS
        kk = np.arange(-int(3 * sig), int(3 * sig) + 1)
        g = np.exp(-0.5 * (kk / sig) ** 2)
        g /= g.sum()

        def smooth(v, default):
            idx = np.arange(n)
            ok = ~np.isnan(v)
            v = np.interp(idx, idx[ok], v[ok]) if ok.any() else np.full(n, default)
            return np.convolve(np.pad(v, len(kk) // 2, mode='edge'), g, mode='valid')

        self.ox = np.full(n, (src_w - side) / 2.0)
        self.oy = np.full(n, (src_h - side) / 2.0)
        if src_w > side:
            self.ox = np.clip(smooth(cx, src_w / 2) - side / 2, 0, src_w - side)
        if src_h > side:
            self.oy = np.clip(smooth(cy, src_h / 2) + y_bias * side - side / 2, 0, src_h - side)

    def offset(self, i: int) -> tuple[float, float]:
        j = min(i, len(self.ox) - 1)
        return float(self.ox[j]), float(self.oy[j])

    def warp(self, img: np.ndarray, i: int, interp=cv2.INTER_LANCZOS4) -> np.ndarray:
        """Any source-aligned array (image, mask, matte) -> canvas space."""
        if self.full:
            return cv2.resize(img, (self.W, self.H), interpolation=interp)
        ox, oy = self.offset(i)
        M = np.float32([[self.s, 0, -ox * self.s], [0, self.s, -oy * self.s]])
        return cv2.warpAffine(img, M, (P, P), flags=interp)

    def crop(self, img_bgr: np.ndarray, i: int) -> np.ndarray:
        """Source BGR frame -> canvas float RGB."""
        return cv2.cvtColor(self.warp(img_bgr, i), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def pt(self, xy, i: int) -> np.ndarray:
        """Source pixel coords -> canvas coords."""
        ox, oy = self.offset(i)
        return (np.asarray(xy, np.float32) - np.array([ox, oy], np.float32)) * self.s

    def box(self, f: dict, i: int) -> tuple[float, float, float, float]:
        (x1, y1), (x2, y2) = self.pt([f['bbox'][:2], f['bbox'][2:]], i)
        return x1, y1, x2, y2


# ---------------------------------------------------------------- video io + shared helpers
def probe(path: str) -> dict:
    out = subprocess.run(  # nosec B603 B607
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate',
         '-of', 'csv=p=0', path], capture_output=True, text=True, check=True,
    ).stdout.strip().split(',')  # fmt: skip
    num, den = [*out[2].split('/'), '1'][:2]
    return {'w': int(out[0]), 'h': int(out[1]), 'rate': out[2], 'fps': float(num) / float(den)}


def frame_range(path: str, start: float | None, end: float | None) -> tuple[int, int | None]:
    fps = probe(path)['fps']
    return (round(start * fps) if start else 0), (round(end * fps) if end else None)


def read_frames(path: str, a: int = 0, b: int | None = None):
    """Yield (index from ``a``, BGR frame) for frames [a, b)."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    k = 0
    while b is None or a + k < b:
        ok, frame = cap.read()
        if not ok:
            break
        yield k, frame
        k += 1
    cap.release()


def face_dict(f: Face, pad: int = 0) -> dict:
    return {
        'bbox': np.asarray(f.bbox[:4], np.float32) - pad,
        'conf': float(f.confidence),
        'kps': np.asarray(f.landmarks, np.float32) - pad,
    }


def as_face(f: dict) -> Face:
    return Face(bbox=f['bbox'], confidence=f['conf'], landmarks=f['kps'])


def area(f: dict) -> float:
    return float((f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))


def largest(faces: list) -> dict | None:
    return max(faces, key=area) if faces else None


def crop_box(img: np.ndarray, box, pad: float = 0.0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    x2, y2 = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def downscale(mask: np.ndarray, max_w: int = 1280) -> np.ndarray:
    """Keep per-frame masks small in memory; they are upsampled again when drawn."""
    if mask.shape[1] > max_w:
        mask = cv2.resize(mask, (max_w, round(max_w * mask.shape[0] / mask.shape[1])), interpolation=cv2.INTER_AREA)
    return mask


class Detector:
    """SCRFD; frames where a face fills the image are retried with a padded border."""

    def __init__(self):
        self.model = SCRFD()

    def __call__(self, frame: np.ndarray) -> list[dict]:
        faces = self.model.detect(frame)
        if faces:
            return [face_dict(f) for f in faces]
        pad = min(frame.shape[:2]) // 2
        big = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return [face_dict(f, pad) for f in self.model.detect(big)]


class Clip:
    """Everything a feature needs while drawing: per-frame results, the framer and the source."""

    def __init__(self, args, rows: list, info: dict, fr: Framer, a: int):
        self.args, self.rows, self.info, self.fr, self.a = args, rows, info, fr, a
        self.n = len(rows)

    def window(self, i: int, r: int) -> list:
        return self.rows[max(0, i - r) : i + r + 1]


# ---------------------------------------------------------------- features
class Feature:
    """Base: ``infer`` fills one row per frame, ``prepare`` sees the whole clip, ``draw`` renders a frame."""

    slug = title = subtitle = ''
    labels: tuple[str, str, str] = ('Input', '', '')
    chips: tuple = ('', ())  # (label, ((model, shown in this demo), ...)) listed under the panel
    sweep = 'split'  # 'split' (sweep, hold half, sweep) or 'single'
    keys = None  # explicit wipe keys

    def load(self) -> None: ...

    def infer(self, frame: np.ndarray, row: dict) -> None: ...

    def prepare(self, clip: Clip) -> None: ...

    def draw(self, clip: Clip, inp: np.ndarray, i: int) -> np.ndarray:
        raise NotImplementedError

    def wipe_keys(self, n: int):
        if self.keys:
            return self.keys
        if self.sweep == 'single':
            return SINGLE_KEYS
        return SPLIT_KEYS if n >= 260 else SPLIT_KEYS_SHORT


class Detection(Feature):
    slug, title, subtitle = 'face_detection', 'Face Detection', 'Every face in the frame, big or small'
    labels = ('Input', 'UniFace · SCRFD', 'Boxes · 5 landmarks · scores')
    chips = (
        '6 DETECTORS IN UNIFACE',
        (
            ('SCRFD', True), ('RetinaFace', False), ('YOLOv5-Face', False), ('YOLOv8-Face', False),
            ('CenterFace', False), ('BlazeFace', False),
        ),
    )  # fmt: skip

    def draw(self, clip, inp, i):
        out, fr = inp.copy(), clip.fr
        for f in clip.rows[i]['faces']:
            box = fr.box(f, i)
            corner_box(out, box, ACCENT, thick=4)
            for p in fr.pt(f['kps'], i):
                cv2.circle(out, ip(p), 6 * S, (1, 1, 1), -1, cv2.LINE_AA, shift=4)
                cv2.circle(out, ip(p), 4 * S, tuple(v / 255 for v in ACCENT), -1, cv2.LINE_AA, shift=4)
            s = tag_sprite(f'{f["conf"]:.2f}', color=ACCENT, fg=INK, size=20, alpha=0.95)
            ty = box[1] - s.shape[0] - 6
            blit(out, s, box[0], ty if ty > 90 else box[3] + 6)
        return out


class Landmarks(Feature):
    slug, title, subtitle = 'facial_landmarks', 'Facial Landmarks', '106 points tracked on every frame'
    labels = ('Input', 'UniFace · Landmark106', '106 facial landmarks')
    chips = (
        'LANDMARK MODELS IN UNIFACE',
        (
            ('Landmark106 · 106 pts', True), ('PIPNet · 98 pts', False), ('PIPNet · 68 pts', False),
            ('FaceMesh · 478 pts', False),
        ),
    )  # fmt: skip
    sweep = 'single'

    def load(self):
        from uniface.landmark import Landmark106

        self.model = Landmark106()

    def infer(self, frame, row):
        for f in row['faces']:
            f['lm106'] = self.model.get_landmarks(frame, f['bbox']).astype(np.float32)

    def draw(self, clip, inp, i):
        pts_all = [clip.fr.pt(f['lm106'], i) for f in clip.rows[i]['faces'] if 'lm106' in f]

        def draw(layer, alpha):
            c = tuple(v / 255 for v in ACCENT)
            for pts in pts_all:
                for p in pts:
                    for lay, cc in ((layer, c), (alpha, 1.0)):
                        cv2.circle(lay, ip(p), int(4.2 * S), cc, -1, cv2.LINE_AA, shift=4)

        return glow_draw(inp, draw, sigma=3, k=0.9)


class Mesh(Feature):
    slug, title, subtitle = 'face_mesh', 'Face Mesh', '478 3D landmarks, irises included'
    labels = ('Input', 'UniFace · FaceMesh', '468 mesh + 10 iris points')
    chips = ('FACEMESH VERSIONS', (('FaceMesh v2 · 478 pts', True), ('FaceMesh v1 · 468 pts', False)))
    IRIS = (255, 92, 92)

    def load(self):
        from uniface.constants import FaceMeshWeights
        from uniface.landmark import FaceMesh
        from uniface.landmark._tessellation import FACEMESH_TESSELATION_FULL, IRIS_LEFT, IRIS_RIGHT

        self.model = FaceMesh(model_name=FaceMeshWeights.V2_478)
        self.edges, self.irises = np.array(FACEMESH_TESSELATION_FULL), (IRIS_LEFT, IRIS_RIGHT)

    def infer(self, frame, row):
        f = largest(row['faces'])
        row['mesh'] = None if f is None else self.model.predict(frame, [as_face(f)])[0].landmarks.astype(np.float32)

    def draw(self, clip, inp, i):
        lm = clip.rows[i]['mesh']
        if lm is None:
            return inp
        pts, z = clip.fr.pt(lm[:, :2], i), lm[:468, 2]
        zn = (z - z.min()) / max(np.ptp(z), 1e-6)
        layer = np.zeros((*inp.shape[:2], 3), np.uint8)
        p = np.rint(pts * S).astype(np.int32)
        for a, b in self.edges:  # depth-shaded edges: nearer is brighter
            k = 1.0 - 0.55 * (zn[a] + zn[b]) / 2
            cv2.line(layer, tuple(p[a]), tuple(p[b]), tuple(int(v * k) for v in ACCENT), 1, cv2.LINE_AA, shift=4)
        for iris in self.irises:
            q = pts[iris]
            r = float(np.linalg.norm(q[1:] - q[0], axis=1).mean())
            c = tuple(np.rint(q[0] * S).astype(int))
            cv2.circle(layer, c, int(r * S), self.IRIS, 2, cv2.LINE_AA, shift=4)
            cv2.circle(layer, c, 3 * S, self.IRIS, -1, cv2.LINE_AA, shift=4)
        f = layer.astype(np.float32) / 255.0
        glow = cv2.GaussianBlur(f, (0, 0), 5) * 1.2
        alpha = np.clip(f.max(axis=2, keepdims=True) * 1.1 + glow.max(axis=2, keepdims=True) * 0.45, 0, 1)
        return inp * (1 - alpha) + np.clip(f + glow, 0, 1) * alpha


class Tracking(Feature):
    slug, title, subtitle = 'face_tracking', 'Face Tracking', 'Persistent IDs across frames with ByteTrack'
    labels = ('Input', 'UniFace · ByteTrack', 'Each face keeps its ID')
    chips = (
        'BYTETRACK + ANY UNIFACE DETECTOR',
        (
            ('ByteTrack', True), ('SCRFD', True), ('RetinaFace', False), ('YOLOv5-Face', False), ('YOLOv8-Face', False),
            ('CenterFace', False), ('BlazeFace', False),
        ),
    )  # fmt: skip

    def prepare(self, clip):
        from uniface.tracking import BYTETracker

        tracker = BYTETracker(track_thresh=0.5, track_buffer=30)
        self.tracks = []
        for row in clip.rows:
            dets = np.array([[*f['bbox'], f['conf']] for f in row['faces']]) if row['faces'] else np.empty((0, 5))
            self.tracks.append(tracker.update(dets))
        ids = sorted({int(t[4]) for tr in self.tracks for t in tr})
        self.color = {tid: PALETTE[k % len(PALETTE)] for k, tid in enumerate(ids)}

    def draw(self, clip, inp, i):
        out = inp.copy()
        for x1, y1, x2, y2, tid in self.tracks[i]:
            (px1, py1), (px2, py2) = clip.fr.pt([[x1, y1], [x2, y2]], i)
            col = self.color[int(tid)]
            corner_box(out, (px1, py1, px2, py2), col, thick=4)
            s = tag_sprite(f'ID {int(tid)}', color=col, fg=INK, size=24, alpha=0.95)
            ty = py1 - s.shape[0] - 8
            blit(out, s, px1, ty if ty >= 90 else py2 + 8)  # keep clear of the top edge / corner mark
        return out


class Recognition(Feature):
    slug, title, subtitle = 'face_recognition', 'Face Recognition', 'Find one person from a single photo'
    labels = ('Input', 'UniFace · ArcFace', 'Enrolled once, matched every frame')
    chips = (
        '5 RECOGNITION MODELS',
        (
            ('ArcFace', True), ('AdaFace', False), ('EdgeFace', False), ('MobileFace', False), ('SphereFace', False),
        ),
    )  # fmt: skip
    THRESH = 0.35
    GREY = (170, 170, 170)

    def load(self):
        from uniface.recognition import ArcFace

        self.model = ArcFace()

    def infer(self, frame, row):
        for f in row['faces']:
            f['emb'] = self.model.get_normalized_embedding(frame, f['kps']).astype(np.float32).ravel()

    def prepare(self, clip):
        args = clip.args
        if args.enroll_image:  # enrol from a separate photo
            img = cv2.imread(args.enroll_image)
            ref = largest(Detector()(img))
            if ref is None:
                sys.exit(f'no face found in {args.enroll_image}')
            ref['emb'] = self.model.get_normalized_embedding(img, ref['kps']).astype(np.float32).ravel()
        else:  # enrol one face (left -> right index) from one frame of the clip
            k = args.enroll_frame if args.enroll_frame is not None else max(clip.n - 20, 0)
            ref = sorted(clip.rows[k]['faces'], key=lambda f: f['bbox'][0])[args.enroll_face]
            img = next(fr for _, fr in read_frames(args.source, clip.a + k, clip.a + k + 1))
        self.ref = ref['emb'] / np.linalg.norm(ref['emb'])
        x1, y1, x2, y2 = ref['bbox']
        pad = 0.25 * (x2 - x1)
        crop = img[int(max(0, y1 - pad)) : int(y2 + pad), int(max(0, x1 - pad)) : int(x2 + pad)]
        th = cv2.resize(crop, (140, int(140 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
        th = cv2.cvtColor(th, cv2.COLOR_BGR2RGB)
        canvas = Image.new('RGBA', (156, th.shape[0] + 56), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((0, 0, canvas.width - 1, canvas.height - 1), radius=18, fill=(17, 17, 17, 215))
        canvas.paste(Image.fromarray(th), (8, 8))
        d.rectangle((8, 8, 147, 8 + th.shape[0] - 1), outline=(*ACCENT, 255), width=3)
        d.text((14, th.shape[0] + 14), 'Enrolled', font=font('SemiBold', 22), fill=(255, 255, 255, 255))
        self.ref_card = np.asarray(canvas).astype(np.float32) / 255.0

    def draw(self, clip, inp, i):
        out, fr = inp.copy(), clip.fr
        for f in clip.rows[i]['faces']:
            sim = float(np.dot(f['emb'], self.ref) / (np.linalg.norm(f['emb']) + 1e-9))
            box = fr.box(f, i)
            if sim > self.THRESH:
                corner_box(out, box, ACCENT, thick=5)
                s = tag_sprite(f'Match  {sim:.2f}', color=ACCENT, fg=INK, size=24, alpha=0.95)
            else:
                corner_box(out, box, self.GREY, thick=3, thin_alpha=0.3)
                s = tag_sprite(f'Unknown  {sim:.2f}', color=(40, 40, 40), fg=(220, 220, 220), size=22, alpha=0.85)
            ty = box[1] - s.shape[0] - 8
            blit(out, s, box[0], ty if ty > 30 else box[3] + 8)
        c = self.ref_card
        blit(out, c, fr.W - c.shape[1] - 36, fr.H - c.shape[0] - 130)
        return out


class Attributes(Feature):
    """Emotion per frame plus AgeGender; age is the per-person median over the clip, shown as a 5-year band."""

    slug, title, subtitle = 'face_attributes', 'Face Attributes', 'Emotion, age and gender in real time'
    labels = ('Input', 'UniFace · Emotion + AgeGender', 'Emotion · age range · gender')
    chips = (
        'ATTRIBUTE MODELS',
        (
            ('Emotion', True), ('AgeGender', True), ('FairFace', False), ('FaceAttribNet', False),
        ),
    )  # fmt: skip

    def load(self):
        from uniface.attribute import AgeGender, Emotion

        self.emo, self.ag = Emotion(), AgeGender()

    def infer(self, frame, row):
        row['faces'].sort(key=lambda f: f['bbox'][0] + f['bbox'][2])  # slot = left -> right position
        for f in row['faces']:
            e, a = self.emo.predict(frame, as_face(f)), self.ag.predict(frame, as_face(f))
            f.update(emotion=(e.emotion, float(e.confidence)), age=a.age, sex=a.sex)

    def attr_card(self, clip, i, slot):
        def hist(r):
            return [row['faces'][slot] for row in clip.window(i, r) if len(row['faces']) > slot]

        emos = [h['emotion'] for h in hist(4)]
        label = Counter(e for e, _ in emos).most_common(1)[0][0]
        conf = float(np.mean([c for e, c in emos if e == label]))
        med = float(np.median([row['faces'][slot]['age'] for row in clip.rows if len(row['faces']) > slot]))
        lo = int(med // 5 * 5)
        sex = Counter(h['sex'] for h in hist(12)).most_common(1)[0][0]
        f1, f2, f3 = font('SemiBold', 40), font('Regular', 26), font('Medium', 26)
        w, h = 300, 168
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=22, fill=(17, 17, 17, 200))
        d.text((22, 14), 'Emotion', font=f2, fill=(180, 180, 180, 255))
        d.text((22, 44), label, font=f1, fill=(255, 255, 255, 255))
        pct = f'{conf * 100:.0f}%'
        d.text((w - 22 - d.textlength(pct, font=f3), 56), pct, font=f3, fill=(*ACCENT, 255))
        d.rounded_rectangle((22, 100, w - 22, 106), radius=3, fill=(70, 70, 70, 255))
        d.rounded_rectangle((22, 100, 22 + (w - 44) * conf, 106), radius=3, fill=(*ACCENT, 255))
        d.text((22, 122), f'{sex} · Age {lo}\u2013{lo + 5}', font=f3, fill=(255, 255, 255, 255))
        return np.asarray(img).astype(np.float32) / 255.0

    def draw(self, clip, inp, i):
        out, fr, row = inp.copy(), clip.fr, clip.rows[i]['faces']
        for slot, f in enumerate(row):
            x1, y1, x2, y2 = fr.box(f, i)
            corner_box(out, (x1, y1, x2, y2), ACCENT, thick=5)
            c = self.attr_card(clip, i, slot)
            cw, ch = c.shape[1], c.shape[0]
            if slot == 0 and len(row) > 1:  # outer faces hang their card from the outer box edge
                cx = x1
            elif slot == len(row) - 1 and len(row) > 1:
                cx = x2 - cw
            else:
                cx = (x1 + x2) / 2 - cw / 2
            cx = min(max(cx, 20), fr.W - cw - 20)
            cy = y2 + 18 if y2 + 18 + ch < fr.H - 120 else max(y1 - ch - 18, 20)
            blit(out, c, cx, cy)
        return out


class EmotionFeature(Feature):
    slug, title, subtitle = 'emotion_recognition', 'Emotion Recognition', 'Seven expressions, frame by frame'
    labels = ('Input', 'UniFace · Emotion', 'Happy · Surprise · Neutral · …')
    chips = ('EMOTION MODELS', (('AffectNet · 7 classes', True), ('AffectNet · 8 classes', False)))
    sweep = 'single'

    def load(self):
        from uniface.attribute import Emotion

        self.model = Emotion()

    def infer(self, frame, row):
        f = largest(row['faces'])
        if f is not None:
            e = self.model.predict(frame, as_face(f))
            f['emotion'] = (e.emotion, float(e.confidence))

    def prepare(self, clip):
        self.side = side_for_card(clip, 300)

    def draw(self, clip, inp, i):
        out = inp.copy()
        f = largest(clip.rows[i]['faces'])
        if f is None:
            return out
        hist = [largest(r['faces'])['emotion'] for r in clip.window(i, 6) if r['faces']]  # ±6 frames: no flicker
        label = Counter(e for e, _ in hist).most_common(1)[0][0]
        conf = float(np.mean([c for e, c in hist if e == label]))
        box = clip.fr.box(f, i)
        corner_box(out, box, ACCENT, thick=5)
        place_card(clip, out, card('Emotion', [(label, f'{conf * 100:.0f}%', conf, ACCENT)]), box, self.side)
        return out


def side_for_card(clip: Clip, cw: int) -> str:
    """Pick the side of the face where a card fits in more frames, so it never jumps sides."""
    fits_r = fits_l = 0
    for j, row in enumerate(clip.rows):
        f = largest(row['faces'])
        if f is not None:
            x1, _, x2, _ = clip.fr.box(f, j)
            fits_r += x2 + 24 + cw < clip.fr.W - 20
            fits_l += x1 - 24 - cw > 20
    return 'r' if fits_r >= fits_l else 'l'


def place_card(clip: Clip, out: np.ndarray, c: np.ndarray, box, side: str) -> None:
    x1, y1, x2, _ = box
    cw, ch = c.shape[1], c.shape[0]
    x = x2 + 24 if side == 'r' else x1 - 24 - cw
    blit(out, c, float(np.clip(x, 20, clip.fr.W - cw - 20)), max(min(y1, clip.fr.H - ch - 130), 30))


class HeadPoseFeature(Feature):
    """Axes drawn like ``uniface.draw.draw_head_pose_axis``: origin at the box centre, size_ratio 0.5."""

    slug, title, subtitle = 'head_pose', 'Head Pose', 'Pitch, yaw and roll for every face'
    labels = ('Input', 'UniFace · HeadPose', '3D head orientation per face')
    chips = (
        '6 HEAD POSE BACKBONES',
        (
            ('ResNet-18', True), ('ResNet-34', False), ('ResNet-50', False), ('MobileNetV2', False),
            ('MobileNetV3-S', False), ('MobileNetV3-L', False),
        ),
    )  # fmt: skip
    keys = ((0.0, 1.0), (1.4, 1.0), (2.6, 0.0), (999, 0.0))
    AXES = ((255, 55, 55), (40, 220, 80), (50, 110, 255))  # X red, Y green, Z blue

    def load(self):
        from uniface.headpose import HeadPose

        self.model = HeadPose()

    def infer(self, frame, row):
        for f in row['faces']:
            c, _ = crop_box(frame, f['bbox'])
            if c.size:
                p = self.model.estimate(c)
                f['pose'] = (float(p.pitch), float(p.yaw), float(p.roll))

    @staticmethod
    def axis_pts(o, size, pitch, yaw, roll):
        yaw_r, pitch_r, roll_r = np.radians([-yaw, pitch, roll])
        cy, sy = np.cos(yaw_r), np.sin(yaw_r)
        cp, sp = np.cos(pitch_r), np.sin(pitch_r)
        cr, sr = np.cos(roll_r), np.sin(roll_r)
        ex = np.array([cy * cr, cp * sr + cr * sp * sy])
        ey = np.array([-cy * sr, cp * cr - sp * sy * sr])
        ez = np.array([sy, -cy * sp])
        return [o + size * e for e in (ex, ey, ez)]

    def pose_card(self, pitch, yaw, roll):
        f1, f2 = font('Medium', 26), font('SemiBold', 30)
        w, h = 250, 158
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=22, fill=(17, 17, 17, 200))
        for k, (name, v, col) in enumerate(zip(('Pitch', 'Yaw', 'Roll'), (pitch, yaw, roll), self.AXES, strict=True)):
            y = 16 + k * 44
            d.ellipse((22, y + 11, 36, y + 25), fill=(*col, 255))
            d.text((50, y + 2), name, font=f1, fill=(200, 200, 200, 255))
            txt = f'{v:+.0f}°'
            d.text((w - 22 - d.textlength(txt, font=f2), y - 2), txt, font=f2, fill=(255, 255, 255, 255))
        return np.asarray(img).astype(np.float32) / 255.0

    def draw(self, clip, inp, i):
        items = []
        for f in clip.rows[i]['faces']:
            if 'pose' in f:
                box = clip.fr.box(f, i)
                items.append((np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]), box, f['pose']))

        def draw(layer, alpha):
            for o, box, pose in items:
                ends = self.axis_pts(o, 0.5 * min(box[2] - box[0], box[3] - box[1]), *pose)
                for end, col in zip(ends, self.AXES, strict=True):
                    c = tuple(v / 255 for v in col)
                    for lay, cc in ((layer, c), (alpha, 1.0)):
                        cv2.line(lay, ip(o), ip(end), cc, 6, cv2.LINE_AA, shift=4)
                        cv2.circle(lay, ip(end), 3 * S, cc, -1, cv2.LINE_AA, shift=4)  # round line cap

        out = glow_draw(inp, draw, sigma=4, k=0.7)
        if len(items) == 1:  # readout averaged over ±2 frames so it is legible; the axes use per-frame angles
            win = [r['faces'][0]['pose'] for r in clip.window(i, 2) if r['faces'] and 'pose' in r['faces'][0]]
            c = self.pose_card(*np.mean(win, axis=0))
            box = items[0][1]
            blit(out, c, min(box[2] + 30, clip.fr.W - c.shape[1] - 30), box[1] + 20)
        return out


class Gaze(Feature):
    slug, title, subtitle = 'gaze_estimation', 'Gaze Estimation', 'Where every person is looking'
    labels = ('Input', 'UniFace · MobileGaze', 'Pitch + yaw per face')
    chips = (
        '5 GAZE BACKBONES',
        (
            ('ResNet-34', True), ('ResNet-18', False), ('ResNet-50', False), ('MobileNetV2', False),
            ('MobileOne-S0', False),
        ),
    )  # fmt: skip
    WARM = (255, 196, 64)

    def load(self):
        from uniface.gaze import MobileGaze

        self.model = MobileGaze()

    def infer(self, frame, row):
        for f in row['faces']:
            c, _ = crop_box(frame, f['bbox'])
            if c.size:
                g = self.model.estimate(c)
                f['gaze'] = (float(g.pitch), float(g.yaw))

    def draw(self, clip, inp, i):
        items = []
        for f in clip.rows[i]['faces']:
            if 'gaze' in f:
                x1, _, x2, _ = clip.fr.box(f, i)
                items.append((clip.fr.pt(f['kps'][:2], i), x2 - x1, *f['gaze']))

        def draw(layer, alpha):
            col = tuple(v / 255 for v in self.WARM)
            for eyes, w, pitch, yaw in items:
                d = 1.1 * w * np.array([-np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
                o = eyes.mean(0)
                for lay, c in ((layer, col), (alpha, 1.0)):
                    cv2.arrowedLine(lay, ip(o), ip(o + d), c, 4, cv2.LINE_AA, shift=4, tipLength=0.22)
                    for e in eyes:
                        cv2.circle(lay, ip(e), 5 * S, c, -1, cv2.LINE_AA, shift=4)

        return glow_draw(inp, draw)


class States(Feature):
    slug, title, subtitle = 'face_states', 'Face States', 'Eyes, glasses, sunglasses and mask'
    labels = ('Input', 'UniFace · FaceAttribNet', 'Live face-state flags')
    chips = (
        'FACEATTRIBNET · 5 OUTPUTS PER FACE',
        (
            ('Left eye', False), ('Right eye', False), ('Glasses', False), ('Sunglasses', False), ('Mask', False),
        ),
    )  # fmt: skip
    sweep = 'single'
    ROWS = (('Eyes open', None), ('Glasses', 'eyeglasses'), ('Sunglasses', 'sunglasses'), ('Mask', 'mask'))
    YES, NO, NA = (120, 235, 225), (255, 120, 120), (150, 150, 150)

    def load(self):
        from uniface.attribute import FaceAttribNet

        self.model = FaceAttribNet()

    def infer(self, frame, row):
        f = largest(row['faces'])
        if f is not None:
            f['state'] = self.model.predict(frame, as_face(f)).as_dict()

    def draw(self, clip, inp, i):
        out = inp.copy()
        f = largest(clip.rows[i]['faces'])
        if f is None:
            return out

        def med(key, r):  # median over a window: a hand passing over the face doesn't flash a false flag
            return float(np.median([largest(row['faces'])['state'][key] for row in clip.window(i, r) if row['faces']]))

        p = {key: med(key, 8) for key in ('eyeglasses', 'sunglasses', 'mask')}
        p['eyes'] = (med('left_eye_open', 2) + med('right_eye_open', 2)) / 2
        rows = []
        for label, key in self.ROWS:
            if key is None and p['sunglasses'] > 0.5:  # eyes hidden behind sunglasses
                rows.append((label, 'n/a', None, self.NA))
                continue
            v = p['eyes' if key is None else key]
            rows.append((label, 'Yes' if v > 0.5 else 'No', v, self.YES if v > 0.5 else self.NO))
        corner_box(out, clip.fr.box(f, i), ACCENT, thick=5)
        c = card('Face states', rows, width=320)  # close-ups fill the frame: card in a fixed corner
        blit(out, c, clip.fr.W - c.shape[1] - 36, clip.fr.H - c.shape[0] - 130)
        return out


class Parsing(Feature):
    slug, title, subtitle = 'face_parsing', 'Face Parsing', 'Pixel-level segmentation of facial parts'
    labels = ('Input', 'UniFace · BiSeNet (ResNet34)', 'Hair · brows · eyes · nose · lips · skin')
    chips = ('BISENET · 19 FACE CLASSES', (('ResNet-34', True), ('ResNet-18', False)))
    COLORS: ClassVar[dict[int, tuple[int, int, int]]] = {  # CelebAMask-HQ classes; only face-region classes are coloured
        1: (255, 196, 150), 2: (120, 90, 255), 3: (120, 90, 255), 4: (90, 200, 255), 5: (90, 200, 255),
        6: (255, 230, 90), 7: (255, 140, 90), 8: (255, 140, 90), 10: (120, 235, 225), 11: (255, 105, 135),
        12: (255, 70, 110), 13: (255, 70, 110), 17: (170, 120, 255),
    }  # fmt: skip

    def load(self):
        from uniface.constants import ParsingWeights
        from uniface.parsing import BiSeNet

        self.model = BiSeNet(model_name=ParsingWeights.RESNET34)
        self.lut, self.lut_a = np.zeros((256, 3), np.float32), np.zeros(256, np.float32)
        for k, c in self.COLORS.items():
            self.lut[k] = np.array(c) / 255.0
            self.lut_a[k] = 0.38 if k == 1 else 0.62  # skin lighter so the face still reads

    def infer(self, frame, row):
        f = largest(row['faces'])
        row['parse'] = None
        if f is not None:
            c, box = crop_box(frame, f['bbox'], pad=0.45)
            row['parse'] = (self.model.parse(c).astype(np.uint8), box)

    def draw(self, clip, inp, i):
        if clip.rows[i]['parse'] is None:
            return inp
        mask, (x1, y1, x2, y2) = clip.rows[i]['parse']
        full = np.zeros((clip.info['h'], clip.info['w']), np.uint8)
        full[y1:y2, x1:x2] = mask
        m = clip.fr.warp(full, i, cv2.INTER_NEAREST)
        col = cv2.GaussianBlur(self.lut[m], (0, 0), 1.2)  # soften the upscaled class edges
        a = cv2.GaussianBlur(self.lut_a[m], (0, 0), 1.2)[..., None]
        out = inp * (1 - a) + col * a
        edges = cv2.Canny(np.where(self.lut_a[m] > 0, m, 0).astype(np.uint8), 0, 0) > 0  # only between parts
        out[edges] = out[edges] * 0.4 + 0.6
        return out


class Segmentation(Feature):
    slug, title, subtitle = 'face_segmentation', 'Face Segmentation', 'Pixel-accurate face mask with XSeg'
    labels = ('Input', 'UniFace · XSeg', 'Face mask, background tinted')
    chips = (
        'MASK MODELS IN UNIFACE',
        (
            ('XSeg · face mask', True), ('BiSeNet · 19 parts', False), ('MODNet · portrait alpha', False),
        ),
    )  # fmt: skip
    sweep = 'single'

    def load(self):
        from uniface.parsing import XSeg

        self.model = XSeg()

    def infer(self, frame, row):
        f = largest(row['faces'])
        row['faces'] = [] if f is None else [f]
        row['mask'] = None
        if f is not None:
            row['mask'] = downscale((self.model.parse(frame, landmarks=f['kps']) * 255).astype(np.uint8))

    def prepare(self, clip):
        self.bg = backdrop(clip.fr.W, clip.fr.H)

    def draw(self, clip, inp, i):
        mt = clip.rows[i]['mask']
        if mt is None:
            return inp
        mt = cv2.resize(mt, (clip.info['w'], clip.info['h']), interpolation=cv2.INTER_LINEAR)
        m = clip.fr.warp(mt, i, cv2.INTER_LINEAR).astype(np.float32) / 255.0
        bg = inp * 0.30 + self.bg * 0.70  # outside the face: dimmed and teal-tinted
        out = inp * m[..., None] + bg * (1 - m[..., None])
        edge = cv2.Canny((m > 0.5).astype(np.uint8) * 255, 0, 0)
        edge = np.clip(cv2.GaussianBlur(edge.astype(np.float32) / 255.0, (0, 0), 1.0) * 2.5, 0, 1)[..., None]
        return out * (1 - edge) + edge


class Matting(Feature):
    slug, title, subtitle = 'portrait_matting', 'Portrait Matting', 'Swap the background, keep every strand of hair'
    labels = ('Input', 'UniFace · MODNet', 'Alpha matte → new background')
    chips = ('MODNET VARIANTS', (('Photographic', True), ('Webcam', False)))

    def load(self):
        from uniface.matting import MODNet

        self.model = MODNet()

    def infer(self, frame, row):
        row['matte'] = downscale((self.model.predict(frame) * 255).astype(np.uint8))

    def prepare(self, clip):
        self.bg = backdrop(clip.fr.W, clip.fr.H)

    def draw(self, clip, inp, i):
        mt = cv2.resize(clip.rows[i]['matte'], (clip.info['w'], clip.info['h']), interpolation=cv2.INTER_LINEAR)
        m = clip.fr.warp(mt, i, cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
        return inp * m + self.bg * (1 - m)


class Anonymization(Feature):
    slug, title, subtitle = 'face_anonymization', 'Face Anonymization', 'Blur every face automatically for privacy'
    labels = ('Input', 'UniFace · BlurFace', 'Pixelate · every face, every frame')
    chips = (
        '5 ANONYMIZATION METHODS',
        (
            ('Pixelate', True), ('Gaussian', False), ('Blackout', False), ('Elliptical', False), ('Median', False),
        ),
    )  # fmt: skip

    def draw(self, clip, inp, i):
        from uniface.privacy import BlurFace

        u8 = cv2.cvtColor((inp * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for f in clip.rows[i]['faces']:
            x1, y1, x2, y2 = clip.fr.box(f, i)
            mx, my = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            # block size scales with the face, so close-ups stay unrecognisable (~7 blocks across)
            blur = BlurFace(method='pixelate', pixel_blocks=max(10, int((x2 - x1 + 2 * mx) / 7)))
            blur.blur_regions(u8, [[x1 - mx, y1 - my, x2 + mx, y2 + my]], inplace=True)
        return cv2.cvtColor(u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


FEATURES = {
    'detection': Detection, 'landmarks': Landmarks, 'mesh': Mesh, 'tracking': Tracking,
    'recognition': Recognition, 'attributes': Attributes, 'emotion': EmotionFeature,
    'headpose': HeadPoseFeature, 'gaze': Gaze, 'states': States, 'parsing': Parsing,
    'segmentation': Segmentation, 'matting': Matting, 'anonymization': Anonymization,
}  # fmt: skip


def run_feature(args) -> None:
    feat = FEATURES[args.command]()
    info = probe(args.source)
    a, b = frame_range(args.source, args.start, args.end)
    detector = Detector()
    feat.load()
    rows = []
    total = (b - a) if b is not None else int(cv2.VideoCapture(args.source).get(cv2.CAP_PROP_FRAME_COUNT)) - a
    for _, frame in tqdm(read_frames(args.source, a, b), total=total, desc=f'{args.command}: inference'):
        row = {'faces': detector(frame)}
        feat.infer(frame, row)
        rows.append(row)
    if not rows:
        sys.exit(f'no frames read from {args.source}')

    fr = Framer(info['w'], info['h'], rows, side=args.crop, y_bias=args.y_bias, full=args.output_only)
    clip = Clip(args, rows, info, fr, a)
    feat.prepare(clip)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / f'uniface_{feat.slug}')
    if args.output_only:
        writer = OutputWriter(f'{prefix}_output.mp4', info['rate'])
    else:
        keys = feat.wipe_keys(len(rows))
        writer = TeaserWriter(prefix, feat.title, feat.subtitle, feat.labels, len(rows), keys, args.formats, feat.chips)
    for i, frame in tqdm(read_frames(args.source, a, b), total=len(rows), desc=f'{args.command}: render'):
        if i >= len(rows):
            break
        inp = fr.crop(frame, i)
        writer.frame(inp, feat.draw(clip, inp, i))
    writer.close()
    outs = [f'{prefix}_output.mp4'] if args.output_only else [f'{prefix}_{fmt}.mp4' for fmt in args.formats]
    print('\n'.join(f'wrote {p}' for p in outs))


# ---------------------------------------------------------------- concat (multi-clip source)
def run_concat(args) -> None:
    """Square-crop several clips around their faces and join them with short crossfades."""
    detector = Detector()
    parts, filters = [], []
    for k, spec in enumerate(args.clips):
        path, _, rng = spec.partition('@')
        t0, _, t1 = rng.partition('-')
        t0 = float(t0) if t0 else 0.0
        info = probe(path)
        dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',  # nosec B603 B607
                                    path], capture_output=True, text=True).stdout)  # fmt: skip
        t1 = float(t1) if t1 else dur
        side = min(info['w'], info['h'])
        mid = round((t0 + t1) / 2 * info['fps'])
        frame = next(fr for _, fr in read_frames(path, mid, mid + 1))
        f = largest(detector(frame))
        cx = (f['bbox'][0] + f['bbox'][2]) / 2 if f else info['w'] / 2
        x0 = int(np.clip(cx - side / 2, 0, info['w'] - side))
        parts.append(t1 - t0)
        filters.append(
            f'[{k}:v]trim={t0}:{t1},setpts=PTS-STARTPTS,crop={side}:{side}:{x0}:0,'
            f'scale={P}:{P}:flags=lanczos,fps={FPS},format=yuv420p,settb=1/{FPS}[s{k}]'
        )
    xf, chain, offset = args.xfade, '[s0]', 0.0
    for k in range(1, len(parts)):
        offset += parts[k - 1] - xf
        filters.append(f'{chain}[s{k}]xfade=transition=fade:duration={xf}:offset={offset:.3f}[x{k}]')
        chain = f'[x{k}]'
    inputs = [x for spec in args.clips for x in ('-i', spec.partition('@')[0])]
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', ';'.join(filters), '-map', chain,  # nosec B603 B607
                    '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', args.output], check=True)  # fmt: skip
    print(f'wrote {args.output}')


# ---------------------------------------------------------------- showcases
INTRO, OUTRO = TeaserWriter.INTRO, TeaserWriter.OUTRO
ORDER = ['detection', 'landmarks', 'mesh', 'tracking', 'recognition', 'attributes', 'emotion', 'headpose', 'gaze',
         'states', 'parsing', 'segmentation', 'matting', 'anonymization']  # fmt: skip
SECTIONS = [
    ('detect_track', 'Detect & Track', ['detection', 'landmarks', 'mesh', 'tracking']),
    ('identity_expression', 'Identity & Expression', ['recognition', 'attributes', 'emotion']),
    ('pose_attention', 'Pose & Attention', ['headpose', 'gaze', 'states']),
    ('segment_protect', 'Segment & Protect', ['parsing', 'segmentation', 'matting', 'anonymization']),
]
SHORT_NAMES = {
    'detection': 'Detection', 'landmarks': 'Landmarks', 'mesh': 'Face Mesh', 'tracking': 'Tracking',
    'recognition': 'Recognition', 'attributes': 'Attributes', 'emotion': 'Emotion', 'headpose': 'Head Pose',
    'gaze': 'Gaze', 'states': 'Face States', 'parsing': 'Parsing', 'segmentation': 'Segmentation',
    'matting': 'Matting', 'anonymization': 'Anonymization',
}  # fmt: skip
CHAPTER_HOLD = 2.6  # seconds kept after the reveal: the model chips fill in, then a beat to read them
HOOK_AT = {'emotion': 3.2, 'states': 13.2}  # hook moments picked for the published clips (seconds into the demo)
UNCUT = {'states'}
FINALE_S = 4.0  # several short scenes: keep them all
FORMATS = SIZES
CARD_DY = {'9x16': 420, '1x1': 0, '16x9': 0}
CARD_DX = {'9x16': 0, '1x1': 0, '16x9': WIDE_CARD_DX}


def count_frames(path: str) -> int:
    out = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',  # nosec B603 B607
                          'stream=nb_read_frames', '-of', 'csv=p=0', path], capture_output=True, text=True)  # fmt: skip
    return int(out.stdout.strip())


def decode(path: str, a: int, b: int, w: int, h: int):
    """Frames [a, b) of a video as float RGB."""
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'trim=start_frame={a}:end_frame={b}',  # nosec B603 B607
                           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)  # fmt: skip
    size = w * h * 3
    while len(buf := ff.stdout.read(size)) == size:
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
    ff.wait()


class Showcase:
    """Join rendered teasers: tighter title cards, demos cut ~2.6 s after the reveal, a hook, section cards."""

    def __init__(self, teasers: str, fmt: str, path: str):
        self.dir, self.fmt = Path(teasers), fmt
        self.w, self.h = FORMATS[fmt]
        self.enc = Encoder(path, self.w, self.h)
        self.n = 0

    def emit(self, img: np.ndarray) -> None:
        self.enc.emit(img)
        self.n += 1

    def path(self, name: str) -> str:
        p = self.dir / f'uniface_{FEATURES[name].slug}_{self.fmt}.mp4'
        if not p.is_file():
            sys.exit(f'missing teaser: {p}')
        return str(p)

    def reveal(self, name: str, n_total: int) -> float:
        return reveal_time(FEATURES[name]().wipe_keys(n_total - INTRO - OUTRO))

    def blank(self) -> np.ndarray:
        return np.ones((self.h, self.w, 3), np.float32)

    def chapter(self, name: str) -> None:
        p = self.path(name)
        n = count_frames(p)
        for f in decode(p, 0, 27, self.w, self.h):  # title card fade-in ...
            self.emit(f)
        for f in decode(p, INTRO - 8, INTRO, self.w, self.h):  # ... and only the end of its static hold
            self.emit(f)
        full = n - INTRO - OUTRO
        keep = full if name in UNCUT else round(max(self.reveal(name, n) + CHAPTER_HOLD, 6.0) * FPS)
        fade = keep < full - 8
        keep = keep if fade else full
        for k, f in enumerate(decode(p, INTRO, INTRO + keep, self.w, self.h)):
            if fade and k >= keep - 12:  # the uncut demo already ends on white; a cut one fades there
                a = ease((k - (keep - 12) + 1) / 12)
                f = f * (1 - a) + a
            self.emit(f)

    def text_card(self, lines: list, seconds: float) -> None:
        layers = [text_layer(*line) for line in lines]
        n = int(seconds * FPS)
        for k in range(n):
            t = k / FPS
            out = 1 - ease_out((t - (seconds - 0.35)) / 0.35)
            img = self.blank()
            for j, layer in enumerate(layers):
                layer.blend(img, ease_out((t - 0.1 - 0.15 * j) / 0.6) * out, CARD_DY[self.fmt], CARD_DX[self.fmt])
            self.emit(img)

    def opener(self) -> None:
        self.text_card([(400, 'UniFace', 'SemiBold', 140, INK),
                        (590, 'A Unified Face Analysis Library for Python', 'Regular', 44, MUTED)], 2.4)  # fmt: skip

    def title_card(self, top: str, title: str, names: list, seconds: float) -> None:
        feats = ' · '.join(SHORT_NAMES[n] for n in names)
        y_top = 418 if top == 'uniface' else 410
        size_top = 34 if top == 'uniface' else 40
        self.text_card([(y_top, top, 'Medium', size_top, MUTED), (468, title, 'SemiBold', 96, INK),
                        (602, feats, 'Regular', 36, MUTED)], seconds)  # fmt: skip

    def finale(self) -> None:
        """pip install card, then a row of what the library also does beyond the demos."""
        line = f'{len(ORDER)} face analysis features in one library'
        layers = [
            text_layer(300, line, 'Regular', 36, MUTED),
            Layer.draw(P, P, lambda d: centered(d, 384, PIP, font('Mono', 50), INK)),
            Layer.draw(P, P, lambda d: centered(d, 474, URL, font('Regular', 34), MUTED)),
        ]
        extras = Chips(*EXTRAS, 592, 636, 26, 1000, one_row=False, height=P, divider_y=562)
        dy, dx = CARD_DY[self.fmt], CARD_DX[self.fmt]
        for k in range(int(FINALE_S * FPS)):
            img = self.blank()
            for j, layer in enumerate(layers):
                layer.blend(img, ease_out((k / FPS - 0.25 * j) / 0.5), dy, dx)
            extras.blend(img, k / FPS, 0.9, dy, dx)
            self.emit(img)

    def hook(self, names: list, cut: int, header: str) -> None:
        """Quick cuts of each feature's finished output (reel / wide: panels beside a fixed header)."""
        head = [text_layer(96, 'uniface', 'Medium', 32, MUTED, 300), text_layer(140, header, 'SemiBold', 72, INK, 300)]
        foot = Layer.draw(P, 120, lambda d: centered(d, 40, URL, font('Regular', 30), MUTED))
        side = [text_layer(440, 'uniface', 'Medium', 34, MUTED, max_w=720),
                text_layer(490, header, 'SemiBold', 72, INK, max_w=720)]  # fmt: skip
        side_url = Layer.draw(P, 120, lambda d: centered(d, 40, URL, font('Regular', 30), MUTED))
        total, k = cut * len(names), 0
        for name in names:
            p = self.path(name)
            t = HOOK_AT.get(name, self.reveal(name, count_frames(p)) + 0.6)
            a = INTRO + round(t * FPS)
            for f in decode(p, a, a + cut, self.w, self.h):
                if self.fmt == '9x16':
                    img = self.blank()
                    img[REEL_Y0 : REEL_Y0 + P] = f[REEL_Y0 : REEL_Y0 + P]
                    for layer in head:
                        layer.blend(img)
                    foot.blend(img, 1.0, REEL_Y0 + P + 40)
                elif self.fmt == '16x9':
                    img = self.blank()
                    img[:, WIDE_X0:] = f[:, WIDE_X0:]
                    for layer in side:
                        layer.blend(img, 1.0, 0, WIDE_COL_DX)
                    side_url.blend(img, 1.0, 950, WIDE_COL_DX)
                else:
                    img = f.copy()
                v = ease(k / 6) * (1 - ease((k - (total - 6)) / 6))  # in from white, out to white
                self.emit(img * v + (1 - v))
                k += 1

    def close(self) -> float:
        self.enc.close()
        return self.n / FPS


def run_showcase(args) -> None:
    kinds = ['full'] + [s[0] for s in SECTIONS] if args.kind == 'all' else [args.kind]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        for fmt in args.formats:
            path = out_dir / f'uniface_showcase_{kind}_{fmt}.mp4'
            sc = Showcase(args.teasers, fmt, str(path))
            segments = []  # (start s, duration s, label): the timeline a voice-over is written against

            def seg(label, fn, *a, sc=sc, segments=segments):
                n0 = sc.n
                fn(*a)
                segments.append((n0 / FPS, (sc.n - n0) / FPS, label))

            if kind == 'full':  # opener -> hook of all features -> numbered sections -> finale
                sections, features = [], []
                seg('opener', sc.opener)
                seg('hook', sc.hook, ORDER, 9, f'{len(ORDER)} features, one library')
                for k, (_, title, names) in enumerate(SECTIONS, start=1):
                    sections.append((0.0 if k == 1 else sc.n / FPS, title))  # first chapter includes the intro
                    seg(f'section {k}: {title}', sc.title_card, f'{k:02d} / {len(SECTIONS):02d}', title, names, 1.5)
                    for name in names:
                        features.append((sc.n / FPS, FEATURES[name].title))
                        seg(name, sc.chapter, name)
            else:  # theme card -> short hook of its own features -> chapters -> finale
                _, title, names = next(s for s in SECTIONS if s[0] == kind)
                sc.title_card('uniface', title, names, 2.0)
                sc.hook(names, 14, title)
                for name in names:
                    sc.chapter(name)
            seg('finale', sc.finale)
            print(f'wrote {path} ({sc.close():.1f} s)')
            if kind == 'full':  # timestamps are the same for every format
                stamp = lambda t: f'{int(t) // 60}:{int(t) % 60:02d}'  # noqa: E731
                lines = ['YouTube chapters (paste into the description):']
                lines += [f'{stamp(t)} {title}' for t, title in sections]
                lines += ['', 'Features:'] + [f'{stamp(t)} {title}' for t, title in features]
                (out_dir / 'uniface_showcase_full_chapters.txt').write_text('\n'.join(lines) + '\n')
                (out_dir / 'uniface_showcase_full_segments.json').write_text(json.dumps(segments) + '\n')


# ---------------------------------------------------------------- cli
def main() -> None:
    parser = argparse.ArgumentParser(description='UniFace teaser videos', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--font-dir', type=str, default=None, help='Folder with Inter-*.otf/.ttf fonts')
    sub = parser.add_subparsers(dest='command', required=True)

    for name, cls in FEATURES.items():
        p = sub.add_parser(name, help=cls.title)
        p.add_argument('--source', type=str, required=True, help='Input video')
        p.add_argument('--out-dir', type=str, default='teasers', help='Output directory')
        p.add_argument('--formats', nargs='+', default=list(SIZES), choices=list(SIZES))
        p.add_argument('--output-only', action='store_true', help='Whole frame + overlay, source fps, no cards')
        p.add_argument('--crop', type=float, default=None, help='Square crop size in source px (default: short edge)')
        p.add_argument('--y-bias', type=float, default=0.0, help='Move the crop below the faces (fraction of crop)')
        p.add_argument('--start', type=float, default=None, help='Start time in seconds')
        p.add_argument('--end', type=float, default=None, help='End time in seconds')
        if name == 'recognition':
            p.add_argument('--enroll-image', type=str, default=None, help='Photo of the person to find')
            p.add_argument('--enroll-frame', type=int, default=None, help='Clip frame to enrol from (default: n-20)')
            p.add_argument('--enroll-face', type=int, default=0, help='Face index, left to right, in that frame')

    p = sub.add_parser('concat', help='Join clips into one square source (e.g. for face states)')
    p.add_argument('clips', nargs='+', help='path[@start-end] in seconds, e.g. clip.mp4@0.4-6.2')
    p.add_argument('--output', type=str, required=True, help='Output video')
    p.add_argument('--xfade', type=float, default=0.4, help='Crossfade length in seconds')

    p = sub.add_parser('showcase', help='Join rendered teasers into showcase videos')
    p.add_argument('--teasers', type=str, default='teasers', help='Folder with uniface_<feature>_<fmt>.mp4 files')
    p.add_argument('--out-dir', type=str, default='showcases', help='Output directory')
    p.add_argument('--kind', default='all', choices=['all', 'full'] + [s[0] for s in SECTIONS])
    p.add_argument('--formats', nargs='+', default=list(SIZES), choices=list(SIZES))

    args = parser.parse_args()
    if args.font_dir:
        FONT_DIRS.insert(0, args.font_dir)
    if args.command == 'concat':
        run_concat(args)
    elif args.command == 'showcase':
        run_showcase(args)
    else:
        run_feature(args)


if __name__ == '__main__':
    main()
