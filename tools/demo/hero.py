# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""README hero: a 3x3 grid of UniFace overlay videos playing at once, each with a feature label.

Input is the folder of overlay-only clips that ``tools/teaser.py`` writes (``*_original_output.mp4``).
Needs ``ffmpeg`` 6.1+ built with libx264, libwebp and libsvtav1.

Usage:
    python tools/demo/hero.py --src teaser_final_new/original_output --out assets/demo
    python tools/demo/hero.py --src teaser_final_new/original_output --out assets/demo --only avif
    python tools/demo/hero.py --src teaser_final_new/original_output --out assets/demo --name uniface_hero_alt

Outputs:
    <name>.avif  1920x1080, 25 fps, ~4 MB animated AVIF for the README; AV1 keeps only what changes between frames
    <name>.webp  1088x614, 10 fps, ~4 MB animated WebP, the README fallback for browsers without animated AVIF
    <name>.mp4   1920x1080, 25 fps H.264 for social posts, the docs site and GitHub video attachments

In the README, let the browser pick:
    <picture>
      <source srcset="assets/demo/<name>.avif" type="image/avif">
      <img src="assets/demo/<name>.webp" width="100%" alt="...">
    </picture>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

DURATION = 8.0
BACKGROUND = '0x0d1117'  # GitHub dark canvas, so the gutters disappear in dark mode

# (clip stem, label, crop as w:h:x:y in source pixels, or None for a full 16:9 frame)
TILES = [
    ('face_detection_original', 'Detection', None),
    ('facial_landmarks_original', 'Landmarks', None),
    ('face_mesh_original', 'Face Mesh', '1080:608:0:150'),
    ('face_tracking_original', 'Tracking', None),
    ('face_recognition_original', 'Recognition', None),
    ('face_attributes_original', 'Attributes', None),
    ('head_pose_original', 'Head Pose', None),
    ('face_parsing_original', 'Face Parsing', None),
    ('face_anonymization_original', 'Anonymization', None),
]

FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/inter/Inter-Medium.otf',
    '/Library/Fonts/Inter-Medium.otf',
    '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',
    '/Library/Fonts/Inter-SemiBold.otf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
]


def _duration(path: Path) -> float:
    out = subprocess.check_output(  # nosec B603 B607
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)]
    )
    return float(json.loads(out)['format']['duration'])


def _font(explicit: str | None) -> str:
    for candidate in [explicit, *FONT_CANDIDATES]:
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit('No usable font found; pass --font /path/to/font.ttf')


def render_label(text: str, font_path: str, size: int, out: Path) -> tuple[int, int]:
    """Draw one label as a rounded, semi-transparent tag and save it as an RGBA PNG.

    Args:
        text: Label text.
        font_path: Font file.
        size: Font size in pixels.
        out: PNG path to write.

    Returns:
        The tag's width and height in pixels.
    """
    font = ImageFont.truetype(font_path, size)
    left, top, _, bottom = font.getbbox('H')
    cap = bottom - top  # cap height keeps every tag the same height, with or without descenders
    pad_x, pad_y = round(size * 0.6), round(size * 0.42)
    width = round(font.getlength(text)) + 2 * pad_x
    height = cap + 2 * pad_y
    tag = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tag)
    bg = tuple(int(BACKGROUND[i : i + 2], 16) for i in (2, 4, 6))
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=round(size * 0.42), fill=(*bg, 175))
    draw.text((pad_x - left, pad_y + cap), text, font=font, fill=(255, 255, 255, 240), anchor='ls')
    tag.save(out)
    return width, height


def build(src: Path, out: Path, font: str, tile_w: int, tile_h: int, gap: int, fps: int, codec: list[str]) -> None:
    """Render one grid video.

    Args:
        src: Folder with ``<stem>_output.mp4`` overlay clips.
        out: Output file path; the extension picks the container.
        font: Font file for the labels.
        tile_w: Width of one tile in pixels.
        tile_h: Height of one tile in pixels.
        gap: Gutter between tiles in pixels.
        fps: Output frame rate.
        codec: Encoder arguments passed to ffmpeg.
    """
    font_size = max(11, round(tile_h * 0.048))
    margin = round(font_size * 0.75)
    inputs: list[str] = []
    chains: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (stem, _, crop) in enumerate(TILES):
            path = src / f'{stem}_output.mp4'
            if not path.exists():
                sys.exit(f'Missing clip: {path}')
            start = max(0.0, (_duration(path) - DURATION) / 2)
            inputs += ['-ss', f'{start:.2f}', '-t', f'{DURATION}', '-i', str(path)]
            crop_filter = f'crop={crop},' if crop else ''
            chains.append(
                f'[{i}:v]{crop_filter}fps={fps},scale={tile_w}:{tile_h}:flags=lanczos,setsar=1,'
                f'trim=duration={DURATION},setpts=PTS-STARTPTS[v{i}]'
            )
        for i, (_, label, _) in enumerate(TILES):
            png = Path(tmp) / f'label_{i}.png'
            _, tag_h = render_label(label, font, font_size, png)
            inputs += ['-i', str(png)]
            chains.append(f'[v{i}][{len(TILES) + i}:v]overlay=x={margin}:y={tile_h - tag_h - margin}[t{i}]')
        layout = '|'.join(f'{(i % 3) * (tile_w + gap)}_{(i // 3) * (tile_h + gap)}' for i in range(9))
        graph = ';'.join(chains) + ';' + ''.join(f'[t{i}]' for i in range(9))
        graph += f'xstack=inputs=9:layout={layout}:fill={BACKGROUND},format=yuv420p[v]'
        cmd = ['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', graph, '-map', '[v]', '-an', *codec, str(out)]
        subprocess.run(cmd, check=True)  # nosec B603
    print(f'{out}  {out.stat().st_size / 1e6:.1f} MB')


def main() -> None:
    parser = argparse.ArgumentParser(description='Render the README hero grid from teaser overlay clips.')
    parser.add_argument('--src', type=Path, required=True, help='Folder of *_original_output.mp4 clips from teaser.py')
    parser.add_argument('--out', type=Path, default=Path('assets/demo'))
    parser.add_argument('--only', choices=['avif', 'webp', 'mp4'])
    parser.add_argument('--font', help='Font file for the labels (Inter Medium by default when installed)')
    parser.add_argument('--name', default='uniface_hero', help='Output file name without extension')
    args = parser.parse_args()

    if shutil.which('ffmpeg') is None:
        sys.exit('ffmpeg is required')
    font = _font(args.font)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only in (None, 'avif'):
        build(
            args.src,
            args.out / f'{args.name}.avif',
            font,
            636,
            356,
            6,
            25,
            ['-c:v', 'libsvtav1', '-preset', '8', '-crf', '32', '-g', '250', '-loop', '0'],
        )
    if args.only in (None, 'webp'):
        build(
            args.src,
            args.out / f'{args.name}.webp',
            font,
            360,
            202,
            4,
            10,
            ['-c:v', 'libwebp', '-lossless', '0', '-q:v', '55', '-compression_level', '6', '-loop', '0'],
        )
    if args.only in (None, 'mp4'):
        build(
            args.src,
            args.out / f'{args.name}.mp4',
            font,
            636,
            356,
            6,
            25,
            ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-movflags', '+faststart'],
        )


if __name__ == '__main__':
    main()
