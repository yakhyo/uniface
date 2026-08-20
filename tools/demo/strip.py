"""Face-tile strip: N square face crops in a row or grid.

Each tile carries a headline value and a sub-label under it. Used for age/gender, emotion,
face states and FairFace so those components share one visual language.
"""

import cv2
from demo_render import Canvas, font, to_pil
from PIL import Image

from uniface.constants import SCRFDWeights
from uniface.detection import SCRFD

_det = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=0.4, providers=['CPUExecutionProvider'])


def load(path, long=1600):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = long / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def biggest(img):
    fs = _det.detect(img)
    if not fs:
        return None
    return max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def tile(img, face, size=330, margin=0.5):
    x1, y1, x2, y2 = face.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
    H, W = img.shape[:2]
    half = min(half, cx, cy, W - cx, H - cy)
    crop = img[int(cy - half) : int(cy + half), int(cx - half) : int(cx + half)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def render(items, out, footer, cols=None, size=330, headline_pt=38, sub_pt=19, foot_gap=34, foot_pt=20):
    """items: list of (tile_bgr, headline, sub, colour)."""
    cols = cols or len(items)
    nrows = (len(items) + cols - 1) // cols
    PAD, G, CAP = 26, 14, headline_pt + sub_pt + 26
    W = PAD * 2 + size * cols + G * (cols - 1)
    H = PAD + (size + CAP) * nrows + G * (nrows - 1) + foot_gap + foot_pt
    c = Canvas(Image.new('RGB', (W, H), (26, 30, 38)))
    fh, fs_ = font(headline_pt), font(sub_pt)
    for i, (t, head, sub, col) in enumerate(items):
        r, k = divmod(i, cols)
        x = PAD + k * (size + G)
        y = PAD + r * (size + CAP + G)
        c.base.paste(to_pil(t), (x, y))
        c.draw.text((x + size / 2, y + size + 26), head, font=fh, fill=(*col, 255), anchor='mm')
        c.draw.text((x + size / 2, y + size + 56), sub, font=fs_, fill=(150, 158, 170, 255), anchor='mm')
    c.draw.text((PAD, H - foot_pt - 8), footer, font=font(foot_pt), fill=(150, 158, 170, 255))
    c.save(out)
    print(out, c.size, f'ratio={c.size[0] / c.size[1]:.2f}')
    for _, head, sub, _ in items:
        print(f'   {head:<22} {sub}')
