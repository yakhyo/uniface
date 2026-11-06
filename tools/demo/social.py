# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Social cards: the GitHub repo preview (1280x640, still and animated) and the docs link preview (1200x630).

Every picture on the cards is a frame of a ``tools/teaser.py --output-only`` clip, and the logo is
``docs/assets/brand/logo/uniface-logo-dark-2400.png``. Needs ``ffmpeg`` on PATH, Pillow and NumPy.

Inputs:
    --grid      Folder of overlay clips for the six-tile card; the same folder ``hero.py`` reads
                (``face_detection_original_output.mp4`` and so on).
    --features  Folder of overlay clips of one person, one clip per feature. The published cards use
                one second of ``face_parsing_original.mp4``:

        for f in mesh detection landmarks headpose parsing matting anonymization; do
            python tools/teaser.py $f --source face_parsing_original.mp4 --output-only \\
                --start 3.6 --end 4.6 --out-dir social_src
        done

Usage:
    python tools/demo/social.py --grid demo_videos/original_output --features demo_videos/social_src
    python tools/demo/social.py --features demo_videos/social_src --only og

Outputs:
    <out>/uniface-social-preview.png  1280x640 six-tile card, for the repo social preview
    <out>/uniface-social-preview.jpg  the same card as a JPEG
    <out>/uniface-social-preview.gif  1280x640 one face, wiping through seven features, 10 fps
    <out>/uniface-preview.mp4         the same animation at 25 fps, H.264
    <og>                              1200x630 Face Mesh card for Open Graph and X link previews
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from functools import cache
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
LOGO = ROOT / 'docs/assets/brand/logo/uniface-logo-dark-2400.png'
SS = 2  # layouts are drawn at twice the final size, then downscaled, for smooth text and corners

INK = (11, 15, 23)  # brand ink: card background
TEXT = (240, 243, 246)  # brand off-white
SUBTLE = (201, 209, 217)
MUTED = (139, 148, 158)
ACCENT = (238, 90, 47)  # brand orange
LINK = (88, 166, 255)
BOX = (22, 27, 34)
BOX_EDGE = (48, 54, 61)

TAGLINE = ('A Unified Face Analysis', 'Library for Python')
BADGE = 'OPEN SOURCE · MIT'
PIP = 'pip install uniface'
FACTS = ('15 tasks', 'ONNX Runtime', 'CPU & GPU')
URL = 'github.com/yakhyo/uniface'

# Six-tile card: (clip stem in --grid, label, crop as (w, h, x, y) in source px or None for the full frame)
TILES = [
    ('face_detection_original', 'Detection', None),
    ('face_mesh_original', 'Face Mesh', (1080, 608, 0, 150)),
    ('face_tracking_original', 'Tracking', None),
    ('face_attributes_original', 'Attributes', None),
    ('face_parsing_original', 'Face Parsing', None),
    ('face_anonymization_original', 'Anonymization', None),
]

# Wipe animation and link preview: (teaser slug in --features, label), in play order
FEATURES = [
    ('face_mesh', 'Face Mesh'),
    ('face_detection', 'Detection'),
    ('facial_landmarks', 'Landmarks'),
    ('head_pose', 'Head Pose'),
    ('face_parsing', 'Face Parsing'),
    ('portrait_matting', 'Portrait Matting'),
    ('face_anonymization', 'Anonymization'),
]
HOLD, WIPE = 1.1, 0.5  # seconds each feature stays, seconds the wipe to the next one takes

FONT_DIRS = [
    '~/Library/Fonts',
    '/Library/Fonts',
    '/usr/share/fonts/opentype/inter',
    '/usr/share/fonts/truetype/inter',
    '~/.local/share/fonts',
    '/usr/share/fonts/truetype/jetbrains-mono',
    '/usr/share/fonts/truetype/dejavu',
    '/System/Library/Fonts',
    'C:/Windows/Fonts',
]
FONT_FILES = {
    'Regular': ['Inter-Regular.otf', 'Inter-Regular.ttf', 'DejaVuSans.ttf', 'arial.ttf'],
    'Medium': ['Inter-Medium.otf', 'Inter-Medium.ttf', 'DejaVuSans.ttf', 'arial.ttf'],
    'SemiBold': ['Inter-SemiBold.otf', 'Inter-SemiBold.ttf', 'DejaVuSans-Bold.ttf', 'arialbd.ttf'],
    'Mono': [
        'JetBrainsMono-Regular.ttf',
        'JetBrainsMonoNerdFontMono-Regular.ttf',
        'SFNSMono.ttf',
        'Menlo.ttc',
        'DejaVuSansMono.ttf',
        'consola.ttf',
    ],
}


@cache
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """Inter at ``weight`` ('Regular', 'Medium', 'SemiBold') or a monospace font ('Mono'), with fallbacks."""
    for name in FONT_FILES[weight]:
        for folder in FONT_DIRS:
            path = Path(folder).expanduser() / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    print(f'warning: no {weight} font found, using the Pillow default (see --font-dir)', file=sys.stderr)
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------- clips
def _duration(path: Path) -> float:
    out = subprocess.check_output(  # nosec B603 B607
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)]
    )
    return float(json.loads(out)['format']['duration'])


def grab(path: Path, at: float | None = None) -> Image.Image:
    """One frame of a clip as RGB, ``at`` seconds in (the middle of the clip by default)."""
    if not path.exists():
        sys.exit(f'Missing clip: {path}')
    t = _duration(path) / 2 if at is None else at
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', str(path), '-frames:v', '1']
    png = subprocess.check_output([*cmd, '-f', 'image2pipe', '-c:v', 'png', '-'])  # nosec B603 B607
    return Image.open(BytesIO(png)).convert('RGB')


def cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Centre-crop ``img`` to the aspect ratio of ``w`` x ``h`` and resize it to that size."""
    scale = max(w / img.width, h / img.height)
    cw, ch = w / scale, h / scale
    x, y = (img.width - cw) / 2, (img.height - ch) / 2
    return img.resize((w, h), Image.Resampling.LANCZOS, box=(x, y, x + cw, y + ch))


# ---------------------------------------------------------------- drawing
class Card:
    """A card drawn at ``SS`` times its size; every coordinate passed in is in final pixels.

    Args:
        width: Final width.
        height: Final height.
        opaque: Fill with the background colour; otherwise start transparent (for overlays).
    """

    def __init__(self, width: int, height: int, opaque: bool = True):
        self.w, self.h = width, height
        self.img = Image.new('RGBA', (width * SS, height * SS), (*INK, 255) if opaque else (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.img)

    def text(self, x: float, y: float, text: str, weight: str, size: int, fill, spacing: float = 0) -> float:
        """Draw ``text`` with its baseline at ``y``; returns the x where it ends."""
        f = font(weight, size * SS)
        if not spacing:
            self.draw.text((x * SS, y * SS), text, font=f, fill=fill, anchor='ls')
            return x + self.draw.textlength(text, font=f) / SS
        for ch in text:  # Pillow has no letter spacing, so place the characters one by one
            self.draw.text((x * SS, y * SS), ch, font=f, fill=fill, anchor='ls')
            x += self.draw.textlength(ch, font=f) / SS + spacing
        return x - spacing

    def box(self, x0: float, y0: float, x1: float, y1: float, radius: float, fill, outline=None) -> None:
        self.draw.rounded_rectangle(
            (x0 * SS, y0 * SS, x1 * SS, y1 * SS), radius=radius * SS, fill=fill, outline=outline, width=SS
        )

    def logo(self, x: float, y: float, width: float, path: Path) -> float:
        """Paste the logo with its top-left corner at (x, y); returns its height."""
        logo = Image.open(path).convert('RGBA')
        w = round(width * SS)
        h = round(logo.height * w / logo.width)
        self.img.alpha_composite(logo.resize((w, h), Image.Resampling.LANCZOS), (round(x * SS), round(y * SS)))
        return h / SS

    def photo(self, img: Image.Image, x: float, y: float, w: float, h: float, radius: float) -> None:
        """Paste ``img`` cropped to fill the box, with rounded corners."""
        size = (round(w * SS), round(h * SS))
        mask = Image.new('L', size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius * SS, fill=255)
        self.img.paste(cover(img, *size), (round(x * SS), round(y * SS)), mask)

    def tag(self, x: float, y: float, text: str, size: int, anchor: str = 'bl') -> float:
        """Dark rounded label; ``anchor`` 'bl' puts its bottom-left corner at (x, y), 'br' its bottom-right.

        Returns:
            The x of the label's centre.
        """
        f = font('Medium', size * SS)
        pad_x, pad_y = size * 0.6, size * 0.38
        cap = f.getbbox('H')[3] - f.getbbox('H')[1]
        w = self.draw.textlength(text, font=f) / SS + 2 * pad_x
        h = cap / SS + 2 * pad_y
        x0 = x - w if anchor == 'br' else x
        layer = Image.new('RGBA', self.img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle((x0 * SS, (y - h) * SS, (x0 + w) * SS, y * SS), radius=size * 0.35 * SS, fill=(*INK, 190))
        d.text(((x0 + pad_x) * SS, (y - pad_y) * SS), text, font=f, fill=(*TEXT, 255), anchor='ls')
        self.img.alpha_composite(layer)
        return x0 + w / 2

    def final(self) -> Image.Image:
        """The card at its final size: RGB when opaque, RGBA for overlays."""
        img = self.img.resize((self.w, self.h), Image.Resampling.LANCZOS)
        return img.convert('RGB') if img.getextrema()[3][0] == 255 else img


def pip_box(card: Card, x: float, y: float, size: int) -> None:
    """``$ pip install uniface`` in a dark rounded box with its top-left corner at (x, y)."""
    f = font('Mono', size * SS)
    pad_x, height = size * 0.85, size * 2.35
    width = card.draw.textlength(f'$ {PIP}', font=f) / SS + 2 * pad_x
    card.box(x, y, x + width, y + height, radius=size * 0.4, fill=BOX, outline=BOX_EDGE)
    base = y + height / 2 + size * 0.36
    end = card.text(x + pad_x, base, '$ ', 'Mono', size, ACCENT)
    card.text(end, base, PIP, 'Mono', size, TEXT)


# ---------------------------------------------------------------- six-tile card
def tile_card(grid: Path, logo: Path) -> Image.Image:
    """The 1280x640 still: name, tagline and install line on the left, six features on the right."""
    card = Card(1280, 640)
    x = 60
    card.text(x, 147, BADGE, 'SemiBold', 15, ACCENT, spacing=1.6)
    card.logo(x, 172, 330, logo)
    for i, line in enumerate(TAGLINE):
        card.text(x, 302 + i * 38, line, 'Regular', 29, SUBTLE)
    pip_box(card, x, 378, 20)
    end = card.text(x, 486, FACTS[0], 'SemiBold', 18, TEXT)
    card.text(end, 486, ''.join(f' · {fact}' for fact in FACTS[1:]), 'Regular', 18, MUTED)
    card.text(x, 515, URL, 'Medium', 18, LINK)

    tw, th, gap = 320, 180, 8
    x0, y0 = 1280 - 60 - 2 * tw - gap, (640 - 3 * th - 2 * gap) / 2
    for i, (stem, label, crop) in enumerate(TILES):
        frame = grab(grid / f'{stem}_output.mp4')
        if crop:
            w, h, cx, cy = crop
            frame = frame.crop((cx, cy, cx + w, cy + h))
        tx, ty = x0 + (i % 2) * (tw + gap), y0 + (i // 2) * (th + gap)
        card.photo(frame, tx, ty, tw, th, radius=10)
        card.tag(tx + 8, ty + th - 8, label, 14)
    return card.final()


# ---------------------------------------------------------------- feature card (wipe animation + link preview)
class FeatureCard:
    """Name on the left, one face on the right that fades into the background, label in the corner.

    Args:
        width: Card width.
        height: Card height.
        frames: One RGB frame per feature, all of the same person at the same moment.
        labels: Feature label for each frame.
        logo: Logo PNG for dark backgrounds.
        focus: Face centre as a fraction of frame width and height.
        view: Share of the frame height that fills the card height.
    """

    def __init__(self, width: int, height: int, frames, labels, logo: Path, focus=(0.474, 0.44), view=0.82):
        self.w, self.h = width, height
        self.px = round(width * 0.42)  # where the photo starts
        fade = round(width * 0.2)

        card = Card(width, height)
        top = height / 2 - 158
        logo_h = card.logo(64, top, 400, logo)
        for i, line in enumerate(TAGLINE):
            card.text(64, top + logo_h + 74 + i * 45, line, 'Regular', 35, SUBTLE)
        pip_box(card, 64, top + logo_h + 156, 26)
        self.base = np.asarray(card.final(), np.float32)

        pw = width - self.px
        win_h = frames[0].height * view
        scale = height / win_h
        cx, cy = focus[0] * frames[0].width, focus[1] * frames[0].height
        left = cx - 0.55 * pw / scale  # face at 55% across the photo
        top_src = cy - 0.47 * win_h
        box = (left, top_src, left + pw / scale, top_src + win_h)
        self.photos = [
            np.asarray(f.resize((pw, height), Image.Resampling.LANCZOS, box=box), np.float32) for f in frames
        ]

        ramp = np.clip(np.arange(pw, dtype=np.float32) / fade, 0, 1)
        self.alpha = (ramp * ramp * (3 - 2 * ramp))[None, :, None]  # smoothstep from background to photo

        self.tags = []  # (premultiplied colour, alpha, centre x) per label
        for label in labels:
            tag = Card(width, height, opaque=False)
            centre = tag.tag(width - 22, height - 22, label, 20, anchor='br')
            rgba = np.asarray(tag.final(), np.float32) / 255.0
            self.tags.append((rgba[..., :3] * rgba[..., 3:] * 255.0, rgba[..., 3:], centre))

    def frame(self, k: int, nxt: int | None = None, p: float = 0.0) -> np.ndarray:
        """Feature ``k``, or the wipe from ``k`` to ``nxt`` at progress ``p`` in [0, 1] (``nxt`` on the left)."""
        photo = self.photos[k]
        d = 0
        if nxt is not None and p > 0:
            d = round(p * (self.w - self.px))
            photo = photo.copy()
            photo[:, :d] = self.photos[nxt][:, :d]
        out = self.base.copy()
        out[:, self.px :] = out[:, self.px :] * (1 - self.alpha) + photo * self.alpha
        if 0 < d < self.w - self.px:
            a = self.alpha[0, min(d, self.alpha.shape[1] - 1), 0] * 0.9
            out[:, self.px + d - 1 : self.px + d + 1] = out[:, self.px + d - 1 : self.px + d + 1] * (1 - a) + 255 * a
        label = nxt if nxt is not None and self.px + d > self.tags[k][2] else k  # switch once the divider passes it
        color, a, _ = self.tags[label]
        out = out * (1 - a) + color
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def wipe(self, fps: float) -> Iterator[np.ndarray]:
        """Every frame of one loop: hold each feature, then wipe to the next; the last wipe returns to the first."""
        n, seg = len(self.photos), HOLD + WIPE
        for i in range(round(n * seg * fps)):
            t = i / fps
            k, u = int(t // seg) % n, t % seg
            if u < HOLD:
                yield self.frame(k)
            else:
                s = (u - HOLD) / WIPE
                yield self.frame(k, (k + 1) % n, s * s * (3 - 2 * s))


# ---------------------------------------------------------------- encoding
def encode(frames: Iterator[np.ndarray], size: tuple[int, int], fps: float, out: Path, args: list[str]) -> None:
    """Pipe raw RGB frames into ffmpeg."""
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}']
    cmd += ['-r', str(fps), '-i', '-', *args, str(out)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:  # nosec B603
        stdin = proc.stdin
        if stdin is None:
            sys.exit('could not open a pipe to ffmpeg')
        for frame in frames:
            stdin.write(frame.tobytes())
        stdin.close()
    if proc.returncode:
        sys.exit(f'ffmpeg failed writing {out}')


def report(path: Path) -> None:
    size = path.stat().st_size
    note = '  (over 1 MB: GitHub rejects it as a social preview)' if size > 1_000_000 else ''
    print(f'{path}  {size / 1e3:.0f} KB{note}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Render the UniFace social preview and link preview cards.')
    parser.add_argument('--grid', type=Path, help='Folder of hero overlay clips (<stem>_output.mp4)')
    parser.add_argument('--features', type=Path, help='Folder of uniface_<slug>_output.mp4 clips of one person')
    parser.add_argument('--out', type=Path, default=ROOT / 'assets/social', help='Folder for the repo preview files')
    parser.add_argument('--og', type=Path, default=ROOT / 'docs/assets/og-image.jpg', help='Link preview path')
    parser.add_argument('--only', choices=['card', 'wipe', 'og'])
    parser.add_argument('--at', type=float, help='Seconds into each feature clip (default: the middle)')
    parser.add_argument('--logo', type=Path, default=LOGO, help='Logo PNG for dark backgrounds')
    parser.add_argument('--font-dir', help='Folder with Inter-*.otf/.ttf, if Inter is not installed')
    args = parser.parse_args()

    if shutil.which('ffmpeg') is None:
        sys.exit('ffmpeg is required')
    if args.font_dir:
        FONT_DIRS.insert(0, args.font_dir)
    if args.only in (None, 'card') and args.grid is None:
        parser.error('--grid is required for the six-tile card')
    if args.only in (None, 'wipe', 'og') and args.features is None:
        parser.error('--features is required for the wipe and the link preview')
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only in (None, 'card'):
        card = tile_card(args.grid, args.logo)
        card.save(args.out / 'uniface-social-preview.png', optimize=True)
        card.save(args.out / 'uniface-social-preview.jpg', quality=90, optimize=True, progressive=True)
        report(args.out / 'uniface-social-preview.png')
        report(args.out / 'uniface-social-preview.jpg')

    if args.only in (None, 'wipe', 'og'):
        frames = [grab(args.features / f'uniface_{slug}_output.mp4', args.at) for slug, _ in FEATURES]
        labels = [label for _, label in FEATURES]

    if args.only in (None, 'og'):
        og = FeatureCard(1200, 630, frames, labels, args.logo)
        args.og.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(og.frame(0)).save(args.og, quality=90, optimize=True, progressive=True)
        report(args.og)

    if args.only in (None, 'wipe'):
        wipe = FeatureCard(1280, 640, frames, labels, args.logo)
        mp4 = args.out / 'uniface-preview.mp4'
        x264 = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
        encode(wipe.wipe(25), (1280, 640), 25, mp4, x264)
        report(mp4)
        gif = args.out / 'uniface-social-preview.gif'
        # One palette and ordered dithering: pixels the wipe has not reached stay identical, so the GIF stores
        # only the changed strip of each frame; mpdecimate turns each hold into a single long frame. 128 colours
        # keep the file under GitHub's 1 MB social preview limit.
        palette = 'mpdecimate,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4'
        encode(wipe.wipe(10), (1280, 640), 10, gif, ['-vf', palette, '-loop', '0'])
        report(gif)


if __name__ == '__main__':
    main()
