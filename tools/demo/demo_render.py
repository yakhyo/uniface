"""Annotation renderer for UniFace demo images.

Draws with PIL rather than OpenCV so text is a real anti-aliased typeface instead
of a Hershey stroke font, and so brackets and chips can be alpha-blended. Every
stroke is laid over a soft dark shadow, which is what lets a single accent colour
stay legible on both a bright wall and a night crowd.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACCENT = (34, 211, 238)
MATCH = (16, 205, 140)
REJECT = (251, 89, 110)
AMBER = (251, 176, 36)
VIOLET = (167, 139, 250)
INK = (11, 15, 25)
PAPER = (248, 250, 252)

_FONT_CANDIDATES = (
    '/System/Library/Fonts/SFNSRounded.ttf',
    '/System/Library/Fonts/HelveticaNeue.ttc',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
)
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _font_path() -> str:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path

    import matplotlib

    hits = glob.glob(os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data/fonts/ttf/DejaVuSans-Bold.ttf'))
    if not hits:
        raise RuntimeError('no usable TrueType font found')
    return hits[0]


def font(size: int) -> ImageFont.FreeTypeFont:
    size = max(9, int(size))
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(_font_path(), size)
    return _FONT_CACHE[size]


def to_pil(image: np.ndarray) -> Image.Image:
    """BGR uint8 array (OpenCV convention) -> RGB PIL image."""
    return Image.fromarray(image[:, :, ::-1].copy())


def to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image.convert('RGB'))[:, :, ::-1].copy()


class Canvas:
    """A PIL drawing surface with an alpha overlay for translucent fills."""

    def __init__(self, image: Image.Image | np.ndarray):
        self.base = to_pil(image) if isinstance(image, np.ndarray) else image.convert('RGB')
        self.overlay = Image.new('RGBA', self.base.size, (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.overlay)

    @property
    def size(self) -> tuple[int, int]:
        return self.base.size

    def scale(self, factor: float = 1.0) -> float:
        """A resolution-independent unit: 1.0 at 1600px on the long edge."""
        return max(self.base.size) / 1600 * factor

    def result(self) -> Image.Image:
        out = self.base.convert('RGBA')
        out.alpha_composite(self.overlay)
        return out.convert('RGB')

    def save(self, path: str, quality: int = 85) -> str:
        img = self.result()
        if path.lower().endswith(('.jpg', '.jpeg')):
            img.save(path, quality=quality, subsampling=0)
        else:
            img.save(path)
        return path

    def _line(self, pts, color, width, alpha=255):
        self.draw.line(tuple(pts), fill=(*color, alpha), width=max(1, int(width)), joint='curve')

    def corner_box(self, bbox, color=ACCENT, width=None, proportion=0.24, edge_alpha=42):
        """Corner-bracket box: 8 short strokes, each over a dark shadow stroke.

        The faint full rectangle stays at low alpha so the box still reads as a
        rectangle without competing with the face inside it.
        """
        x1, y1, x2, y2 = (float(v) for v in bbox)
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            return

        width = width or max(1.8, min(w, h) * 0.055, max(self.base.size) * 0.0016)
        length = max(4.0, proportion * min(w, h))

        self.draw.rectangle([x1, y1, x2, y2], outline=(*color, edge_alpha), width=max(1, int(width * 0.4)))

        arms = (
            ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
            ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
            ((x1, y2 - length), (x1, y2), (x1 + length, y2)),
            ((x2 - length, y2), (x2, y2), (x2, y2 - length)),
        )
        for arm in arms:
            self._line(arm, (0, 0, 0), width * 1.9, alpha=70)
        for arm in arms:
            self._line(arm, color, width, alpha=255)

    def chip(self, text, xy, color=ACCENT, text_color=INK, size=None, anchor='bottom-left', radius=None):
        """Rounded label chip. Clamped so it never leaves the frame."""
        size = int(size or 13 * self.scale())
        f = font(size)
        pad_x, pad_y = size * 0.5, size * 0.3
        left, top, right, bottom = self.draw.textbbox((0, 0), text, font=f)
        tw, th = right - left, bottom - top
        bw, bh = tw + pad_x * 2, th + pad_y * 2
        x, y = xy

        if anchor == 'bottom-left':
            box = [x, y - bh, x + bw, y]
        elif anchor == 'top-left':
            box = [x, y, x + bw, y + bh]
        else:
            raise ValueError(anchor)

        W, H = self.base.size
        dx = min(0, W - 2 - box[2]) - min(0, box[0] - 2)
        dy = min(0, H - 2 - box[3]) - min(0, box[1] - 2)
        box = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]

        rad = radius if radius is not None else bh * 0.28
        self.draw.rounded_rectangle([box[0] + 1, box[1] + 2, box[2] + 1, box[3] + 2], rad, fill=(0, 0, 0, 60))
        self.draw.rounded_rectangle(box, rad, fill=(*color, 255))
        self.draw.text((box[0] + pad_x - left, box[1] + pad_y - top), text, font=f, fill=(*text_color, 255))
        return box

    def points(self, pts, color=ACCENT, radius=None, ring=True):
        """Landmark dots in one accent colour, each with a dark ring for contrast."""
        radius = radius or max(1.5, 2.2 * self.scale())
        for x, y in np.asarray(pts, dtype=float)[:, :2]:
            if ring:
                self.draw.ellipse(
                    [x - radius * 1.75, y - radius * 1.75, x + radius * 1.75, y + radius * 1.75],
                    fill=(0, 0, 0, 90),
                )
            self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(*color, 255))

    def arrow(self, start, end, color=ACCENT, width=None, head=None):
        width = width or max(2.0, 3.0 * self.scale())
        head = head or width * 3.2
        (x1, y1), (x2, y2) = start, end
        ang = np.arctan2(y2 - y1, x2 - x1)

        self._line([(x1, y1), (x2, y2)], (0, 0, 0), width * 1.6, alpha=48)
        self._line([(x1, y1), (x2, y2)], color, width, alpha=255)

        for sign in (1, -1):
            a = ang + sign * np.deg2rad(150)
            self._line(
                [(x2, y2), (x2 + head * np.cos(a), y2 + head * np.sin(a))],
                (0, 0, 0),
                width * 1.6,
                alpha=48,
            )
        for sign in (1, -1):
            a = ang + sign * np.deg2rad(150)
            self._line([(x2, y2), (x2 + head * np.cos(a), y2 + head * np.sin(a))], color, width, alpha=255)

    def banner(self, text, side='bottom-left', size=None, fg=PAPER, bg=(0, 0, 0), bg_alpha=150):
        """Caption bar. Sized to survive the downscale a README applies."""
        size = int(size or 30 * self.scale())
        f = font(size)
        pad = size * 0.55
        left, top, right, bottom = self.draw.textbbox((0, 0), text, font=f)
        tw, th = right - left, bottom - top
        W, H = self.base.size
        margin = size * 0.7
        bw, bh = tw + pad * 2, th + pad * 1.5
        x = margin if 'left' in side else W - margin - bw
        y = margin if 'top' in side else H - margin - bh
        self.draw.rounded_rectangle([x, y, x + bw, y + bh], bh * 0.24, fill=(*bg, bg_alpha))
        self.draw.text((x + pad - left, y + pad * 0.75 - top), text, font=f, fill=(*fg, 255))


def hstack(images, gap=16, bg=(138, 138, 138)):
    """Side-by-side strip on a mid-grey ground that reads in light and dark themes."""
    ims = [im if isinstance(im, Image.Image) else to_pil(im) for im in images]
    h = max(i.height for i in ims)
    ims = [i.resize((round(i.width * h / i.height), h), Image.LANCZOS) for i in ims]
    W = sum(i.width for i in ims) + gap * (len(ims) - 1)
    out = Image.new('RGB', (W, h), bg)
    x = 0
    for i in ims:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out
