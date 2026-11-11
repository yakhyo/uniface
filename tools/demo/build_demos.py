"""Build every component figure from the demo photographs.

    python3 tools/demo/build_demos.py assets

Companion modules `demo_render.py` (drawing primitives) and `strip.py` (face crops and tile
strips) sit alongside this file and are imported by name.

Reads `<root>/source/` by filename convention and writes `<root>/demo/`. Missing sources are
skipped with a warning rather than failing the run, so a partial set still produces figures.
"""

import os
import sys

import cv2
from demo_render import ACCENT, AMBER, MATCH, PAPER, REJECT, VIOLET, Canvas, font, to_bgr, to_pil
import numpy as np
from PIL import Image
import strip

CPU = ['CPUExecutionProvider']

# One type scale for every figure so captions match across the set.
PT_FOOTER = 20  # the single grey line at the bottom
PT_LABEL = 22  # panel and tile captions
PT_VALUE = 38  # the big number or class name
PT_SUB = 19  # secondary line under a value
PT_LEGEND = 26  # parsing class legend
FOOT_GAP = 34  # clear space above the footer line
ROOT = sys.argv[1].rstrip('/')
SRC, OUT = f'{ROOT}/source', f'{ROOT}/demo'
os.makedirs(OUT, exist_ok=True)


def has(*names):
    missing = [n for n in names if not os.path.exists(f'{SRC}/{n}')]
    if missing:
        print(f'  skip: missing {", ".join(missing)}')
    return not missing


def sq(path, size=600, margin=0.85):
    """Square crop around the largest face.

    Shifts the centre rather than shrinking the radius, so a face near a frame edge still yields a
    full-size crop.
    """
    img = strip.load(path, long=2200)
    f = strip.biggest(img)
    if f is None:
        return None, None, None
    x1, y1, x2, y2 = f.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = img.shape[:2]
    half = min(max(x2 - x1, y2 - y1) * (1 + margin) / 2, W / 2, H / 2)
    cx, cy = min(max(cx, half), W - half), min(max(cy, half), H - half)
    x0, y0 = int(cx - half), int(cy - half)
    crop = img[y0 : int(cy + half), x0 : int(cx + half)]
    s = size / crop.shape[1]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    b = np.array([(x1 - x0) * s, (y1 - y0) * s, (x2 - x0) * s, (y2 - y0) * s], dtype=np.float32)
    lm = np.asarray(f.landmarks, dtype=np.float32).copy()
    lm[:, 0] = (lm[:, 0] - x0) * s
    lm[:, 1] = (lm[:, 1] - y0) * s
    return crop, b, lm


def facesub(crop, b, pad_frac=0.2):
    x1, y1, x2, y2 = [int(v) for v in b]
    p = int((x2 - x1) * pad_frac)
    H, W = crop.shape[:2]
    return crop[max(0, y1 - p) : min(H, y2 + p), max(0, x1 - p) : min(W, x2 + p)]


def sheet(panels, name, footer, cols=None):
    """Panels in a row, or wrapped into rows of `cols` when given."""
    cols = cols or len(panels)
    rows = (len(panels) + cols - 1) // cols
    h = panels[0][0].shape[0]
    PAD, G = 20, 12
    ROWCAP = 12 + PT_LABEL + 10
    CAP = 12 + PT_LABEL + FOOT_GAP + PT_FOOTER + 14
    widest = max(sum(p[0].shape[1] for p in panels[r * cols : (r + 1) * cols]) + G * (cols - 1) for r in range(rows))
    W = PAD * 2 + widest
    H = PAD + rows * h + (rows - 1) * ROWCAP + CAP
    c = Canvas(Image.new('RGB', (W, H), (26, 30, 38)))
    for i, (img, label) in enumerate(panels):
        r, k = divmod(i, cols)
        x = PAD + sum(p[0].shape[1] + G for p in panels[r * cols : r * cols + k])
        y = PAD + r * (h + ROWCAP)
        c.base.paste(to_pil(img), (x, y))
        c.draw.text((x + 6, y + h + 12), label, font=font(PT_LABEL), fill=(*PAPER, 255))
    c.draw.text(
        (PAD + 6, PAD + rows * h + (rows - 1) * ROWCAP + 12 + PT_LABEL + FOOT_GAP),
        footer,
        font=font(PT_FOOTER),
        fill=(150, 158, 170, 255),
    )
    p = f'{OUT}/{name}.jpg'
    c.save(p)
    print(f'  {name:<16} {c.size}  ratio={c.size[0] / c.size[1]:.2f}')


# ----------------------------------------------------------------- detection
def detection(src='detect_group.jpg', out='detection'):
    if not has(src):
        return
    from uniface.constants import SCRFDWeights
    from uniface.detection import SCRFD

    img = strip.load(f'{SRC}/{src}', long=1500)
    det = SCRFD(
        model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=0.35, input_size=(1280, 1280), providers=CPU
    )
    faces = det.detect(img)
    c = Canvas(img.copy())
    for f in faces:
        col = MATCH if f.confidence >= 0.7 else (ACCENT if f.confidence >= 0.5 else AMBER)
        c.corner_box(f.bbox, color=col)
        c.points(f.landmarks, color=col, radius=max(1.4, (f.bbox[2] - f.bbox[0]) * 0.03))
    c.banner(f'Face Detection - SCRFD-10G  ·  box + 5 keypoints  ·  {len(faces)} faces', size=40 * c.scale())
    c.save(f'{OUT}/{out}.jpg')
    print(f'  {out:<16} {c.size}  {len(faces)} faces')


# ----------------------------------------------------------------- landmarks + mesh
def landmarks(src='landmarks_face.jpg', out='landmarks'):
    if not has(src):
        return
    from uniface.constants import PIPNetWeights
    from uniface.landmark import Landmark106, PIPNet

    crop, _, _ = sq(f'{SRC}/{src}', size=560, margin=0.55)
    box = np.array([0, 0, crop.shape[1], crop.shape[0]], dtype=np.float32)
    sets = [
        ('106-point · 2d106det', Landmark106(providers=CPU).get_landmarks(crop, box), ACCENT),
        (
            '98-point · PIPNet WFLW',
            PIPNet(model_name=PIPNetWeights.WFLW_98, providers=CPU).get_landmarks(crop, box),
            VIOLET,
        ),
        (
            '68-point · PIPNet 300W',
            PIPNet(model_name=PIPNetWeights.DW300_CELEBA_68, providers=CPU).get_landmarks(crop, box),
            MATCH,
        ),
    ]
    panels = []
    for nm, pts, col in sets:
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        cv = Canvas(crop.copy())
        cv.points(p, color=col, radius=2.8)
        panels.append((to_bgr(cv.result()), f'{nm}  ·  {len(p)} pts'))
    sheet(panels, out, 'Facial Landmarks - 2d106det, PIPNet  ·  106 / 98 / 68 points')


def mesh(src='mesh_face.jpg', out='face_mesh', wire=True):
    """468 beside 478 points.

    `wire` adds a third panel in `full` mode, which draws the dense tessellation as edges only,
    with no landmark dots.
    """
    if not has(src):
        return
    from uniface.constants import FaceMeshWeights
    from uniface.draw import draw_mesh
    from uniface.landmark import FaceMesh

    crop, b, lm = sq(f'{SRC}/{src}', size=900, margin=0.6)
    variants = [('468-point', FaceMeshWeights.V1_468), ('478-point, with irises', FaceMeshWeights.V2_478)]
    pts = {}
    for nm, w in variants:
        pts[nm] = FaceMesh(model_name=w, providers=CPU).predict(crop, bboxes=[b], keypoints=[lm])[0].landmarks

    # BGR.  cyan throughout: the rows differ by what is drawn, dots against edges, not by hue.
    # The library draws iris points in a fixed red, which stays legible against it.
    DOT_EDGE, DOT_PT = (120, 90, 30), (238, 211, 34)
    WIRE = DOT_PT

    top = []
    for nm, _ in variants:
        panel = crop.copy()
        draw_mesh(panel, pts[nm], mode='partial', color=DOT_EDGE, point_color=DOT_PT, point_radius=3)
        top.append((panel, f'{nm}  ·  {len(pts[nm])} pts'))
    if not wire:
        sheet(top, out, 'Face Mesh - MediaPipe  ·  468 / 478 points, irises in the 478')
        return

    bottom = []
    for nm, _ in variants:
        panel = crop.copy()
        draw_mesh(panel, pts[nm], mode='full', color=WIRE)
        bottom.append((panel, nm))
    sheet(top + bottom, out, 'Face Mesh - MediaPipe  ·  468 / 478 points, irises in the 478', cols=2)


# ----------------------------------------------------------------- parsing / xseg / matting
def parsing():
    if not has('parse_face.jpg'):
        return
    from uniface.constants import ParsingWeights
    from uniface.draw import FACE_PARSING_COLORS, FACE_PARSING_LABELS, vis_parsing_maps
    from uniface.parsing import BiSeNet

    crop, _, _ = sq(f'{SRC}/parse_face.jpg', size=900, margin=0.6)
    mask = BiSeNet(model_name=ParsingWeights.RESNET34, providers=CPU).parse(crop)
    over = vis_parsing_maps(crop, mask)
    present = [i for i in np.unique(mask) if i != 0]
    PAD = 20
    W = PAD * 2 + crop.shape[1] * 2 + 12
    # dry pass: how many rows will the legend wrap to?
    _m = Canvas(Image.new('RGB', (W, 10), (0, 0, 0)))
    _fl = font(PT_LEGEND)
    _x, _rows = PAD, 1
    for cid in present:
        _tw = _m.draw.textbbox((0, 0), FACE_PARSING_LABELS[cid], font=_fl)[2]
        if _x + 34 + _tw > W - PAD:
            _x, _rows = PAD, _rows + 1
        _x += 34 + _tw + 22
    ROWH, GAP_TOP, FOOT = PT_LEGEND + 14, 22, FOOT_GAP + PT_FOOTER + 14
    LEG = GAP_TOP + _rows * ROWH + FOOT
    c = Canvas(Image.new('RGB', (W, PAD + crop.shape[0] + LEG), (26, 30, 38)))
    c.base.paste(to_pil(crop), (PAD, PAD))
    c.base.paste(to_pil(over), (PAD + crop.shape[1] + 12, PAD))
    fl = _fl
    x, y = PAD, PAD + crop.shape[0] + GAP_TOP
    for cid in present:
        lab = FACE_PARSING_LABELS[cid]
        bgr = FACE_PARSING_COLORS[cid]
        tw = c.draw.textbbox((0, 0), lab, font=fl)[2]
        if x + 34 + tw > W - PAD:
            x, y = PAD, y + ROWH
        c.draw.rounded_rectangle([x, y + 5, x + 24, y + 29], 6, fill=(bgr[2], bgr[1], bgr[0], 255))
        c.draw.text((x + 32, y), lab, font=fl, fill=(*PAPER, 255))
        x += 34 + tw + 22
    c.draw.text(
        (PAD, PAD + crop.shape[0] + GAP_TOP + _rows * ROWH + FOOT_GAP - 12),
        'Face Parsing - BiSeNet ResNet-34  ·  19 classes',
        font=font(PT_FOOTER),
        fill=(150, 158, 170, 255),
    )
    c.save(f'{OUT}/parsing.jpg')
    print(f'  parsing          {c.size}  {len(present)} classes')


def _board(shape, cell_div=20):
    H, W = shape[:2]
    sq_ = max(14, W // cell_div)
    yy, xx = np.mgrid[0:H, 0:W]
    b = np.zeros((H, W, 3), np.uint8)
    b[:] = (150, 150, 150)
    b[((yy // sq_ + xx // sq_) % 2) == 0] = (198, 198, 198)
    return b


def segmentation(name='seg_face.jpg', out='segmentation'):
    if not has(name):
        return
    from uniface.parsing import XSeg

    crop, _, lm = sq(f'{SRC}/{name}', size=560, margin=0.6)
    m = XSeg(providers=CPU).parse(crop, landmarks=lm).astype(np.float32)
    if m.max() > 1.5:
        m /= 255.0
    a = m[..., None]
    tint = np.zeros_like(crop)
    tint[:] = (200, 220, 60)
    over = (crop * (1 - 0.45 * a) + tint * (0.45 * a)).astype(np.uint8)
    cut = (crop * a + _board(crop.shape) * (1 - a)).astype(np.uint8)
    sheet(
        [(crop, 'input'), (over, 'XSeg mask'), (cut, 'cut out')], out, 'Face Segmentation - XSeg  ·  binary face mask'
    )


def matting(src='matte_face.jpg', out='matting'):
    if not has(src):
        return
    from uniface.matting import MODNet

    crop, _, _ = sq(f'{SRC}/{src}', size=600, margin=1.7)
    m = MODNet(providers=CPU).predict(crop)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() > 1.5:
        m = m / 255.0
    a = m[..., None].astype(np.float32)
    cut = (crop * a + _board(crop.shape, 22) * (1 - a)).astype(np.uint8)
    comp = (crop * a + np.full_like(crop, (64, 177, 0)) * (1 - a)).astype(np.uint8)
    sheet(
        [(crop, 'input'), (cut, 'alpha matte'), (comp, 'composite')],
        out,
        'Portrait Matting - MODNet  ·  trimap-free alpha matte',
    )


# ----------------------------------------------------------------- pose / gaze
def pose_or_gaze(kind, srcs=None, out=None):
    trio = (
        srcs
        or {
            'headpose': ['pose_left.jpg', 'pose_center.jpg', 'pose_right.jpg'],
            'gaze': ['gaze_away.jpg', 'gaze_averted.jpg', 'gaze_right.jpg'],
        }[kind]
    )
    out = out or kind
    if not has(*trio):
        return
    from uniface.constants import GazeWeights, HeadPoseWeights
    from uniface.draw import draw_gaze, draw_head_pose
    from uniface.gaze import MobileGaze
    from uniface.headpose import HeadPose

    hp = HeadPose(model_name=HeadPoseWeights.RESNET34, providers=CPU) if kind == 'headpose' else None
    gz = MobileGaze(model_name=GazeWeights.RESNET18, providers=CPU) if kind == 'gaze' else None
    panels = []
    for n in trio:
        crop, b, _ = sq(f'{SRC}/{n}', size=600, margin=0.85 if kind == 'headpose' else 0.42)
        if kind == 'headpose':
            # exactly as tools/headpose.py: estimate on the unpadded bbox crop, draw the cube
            x1, y1, x2, y2 = [int(v) for v in b]
            r = hp.estimate(crop[y1:y2, x1:x2])
            draw_head_pose(crop, b, r.pitch, r.yaw, r.roll, draw_type='cube', draw_angles=False)
            # Past roughly 60 deg of yaw the pitch and roll heads become unreliable -- a level
            # head reads back 60-80 deg of tilt -- so only quote them where they are trustworthy.
            lab = f'yaw {r.yaw:+.0f}°'
            if abs(r.yaw) < 60:
                lab += f'   pitch {r.pitch:+.0f}°   roll {r.roll:+.0f}°'
            panels.append((crop, lab))
        else:
            r = gz.estimate(facesub(crop, b))
            draw_gaze(crop, b, r.pitch, r.yaw, draw_bbox=False, draw_angles=False)
            panels.append((crop, f'gaze yaw {np.degrees(r.yaw):+.0f}°   pitch {np.degrees(r.pitch):+.0f}°'))
    foot = (
        'Head Pose - ResNet-34  ·  pitch, yaw, roll'
        if kind == 'headpose'
        else 'Gaze Estimation - MobileGaze ResNet-18  ·  pitch, yaw'
    )
    sheet(panels, out, foot)


# ----------------------------------------------------------------- attribute strips
def demography():
    import glob

    names = [os.path.basename(x) for x in sorted(glob.glob(f'{SRC}/age_*.jpg'))]
    if len(names) < 3:
        print('  skip: need at least three age_*.jpg')
        return
    from uniface.attribute import FairFace

    ff = FairFace(providers=CPU)
    items = []
    for n in names:
        img = strip.load(f'{SRC}/{n}')
        f = strip.biggest(img)
        r = ff.predict(img, f)
        items.append((strip.tile(img, f, size=440), r.age_group, r.sex, VIOLET if r.sex == 'Female' else ACCENT))
    items.sort(key=lambda it: int(str(it[1]).split('-')[0].rstrip('+')))
    strip.render(items, f'{OUT}/demography.jpg', 'Age and Sex - FairFace  ·  9 age buckets', size=440)


def emotion():
    order = [
        'emotion_happy.jpg',
        'emotion_surprise.jpg',
        'emotion_angry.jpg',
        'emotion_disgust.jpg',
        'emotion_fear.jpg',
        'emotion_sad.jpg',
        'emotion_contempt.jpg',
        'emotion_neutral.jpg',
    ]
    avail = [n for n in order if os.path.exists(f'{SRC}/{n}')]
    if not avail:
        print('  skip: no emotion_*.jpg')
        return
    from uniface.attribute import Emotion
    from uniface.constants import EmotionWeights

    em = Emotion(model_name=EmotionWeights.AFFECNET8)
    COL = {
        'Happy': MATCH,
        'Surprise': ACCENT,
        'Angry': REJECT,
        'Disgust': AMBER,
        'Fear': VIOLET,
        'Sad': (96, 165, 250),
        'Neutral': (165, 175, 190),
        'Contempt': (244, 143, 177),
    }
    items = []
    for n in avail:
        img = strip.load(f'{SRC}/{n}')
        f = strip.biggest(img)
        r = em.predict(img, f)
        lab = str(r).split("'")[1]
        items.append((strip.tile(img, f, size=440, margin=0.45), lab, f'p = {r.confidence:.2f}', COL.get(lab, ACCENT)))
    strip.render(items, f'{OUT}/emotion.jpg', 'Emotion - AffectNet-8  ·  8 classes', cols=4, size=440)


def face_states(setname='state', out='face_states', open_src='emotion_happy.jpg', second_src=None):
    """All five heads per face.

    They are independent binary outputs, so one face can trigger several at once -- a masked face
    also reads eyes-closed. Positives are highlighted. Neither set has a dedicated eyes-open
    photograph, so `open_src` names one; pass a different file per set so the two figures do not
    share a subject. `second_src` does the same for the second tile, which is just another face:
    every tile prints all five probabilities, so none of them has to show a particular state.
    """
    from uniface.attribute import FaceAttribNet

    order = [
        (open_src, 0),
        (second_src or f'{setname}_closed.jpg', 0),
        (f'{setname}_glasses.jpg', 0),
        (f'{setname}_sunglasses.jpg', 0),
        (f'{setname}_mask.jpg', 0),
    ]
    spec = [(n, i) for n, i in order if os.path.exists(f'{SRC}/{n}')]
    if len(spec) < 2:
        print('  skip: not enough state_*.jpg sources')
        return
    fa = FaceAttribNet(providers=CPU)
    TILE, PAD, G, ROWH = 330, 24, 16, PT_SUB + 11
    LINES = 5
    W = PAD * 2 + TILE * len(spec) + G * (len(spec) - 1)
    H = PAD + TILE + 16 + ROWH * LINES + FOOT_GAP + PT_FOOTER
    c = Canvas(Image.new('RGB', (W, H), (26, 30, 38)))
    for k, (n, idx) in enumerate(spec):
        img = strip.load(f'{SRC}/{n}')
        fs = sorted(strip._det.detect(img), key=lambda f: -(f.bbox[2] - f.bbox[0]))
        f = fs[min(idx, len(fs) - 1)]
        r = fa.predict(img, f)
        x = PAD + k * (TILE + G)
        c.base.paste(to_pil(strip.tile(img, f, size=TILE, margin=0.7)), (x, PAD))
        rows = [
            ('eye L open', r.left_eye_open, 'open'),
            ('eye R open', r.right_eye_open, 'open'),
            ('eyeglasses', r.eyeglasses, 'pos'),
            ('sunglasses', r.sunglasses, 'pos'),
            ('mask', r.mask, 'pos'),
        ]
        y = PAD + TILE + 16
        for name, val, kind in rows:
            hot = val >= 0.5  # each head is an independent binary decision at 0.5
            col = (MATCH if kind == 'open' else ACCENT) if hot else (122, 130, 145)
            c.draw.text((x + 4, y), name, font=font(PT_SUB), fill=(180, 188, 200, 255))
            c.draw.text(
                (x + TILE - 62, y), 'True' if hot else 'False', font=font(PT_SUB), fill=(*col, 255), anchor='ra'
            )
            c.draw.text((x + TILE - 4, y), f'{val:.2f}', font=font(PT_SUB - 2), fill=(112, 120, 134, 255), anchor='ra')
            y += ROWH
        print(
            f'    {n:<24} eyes {r.left_eye_open:.2f}/{r.right_eye_open:.2f}  glasses {r.eyeglasses:.2f}  shades {r.sunglasses:.2f}  mask {r.mask:.2f}'
        )
    c.draw.text(
        (PAD, H - PT_FOOTER - 8),
        'Face States - FaceAttribNet  ·  5 independent binary heads',
        font=font(PT_FOOTER),
        fill=(150, 158, 170, 255),
    )
    c.save(f'{OUT}/{out}.jpg')
    print(f'  {out:<16} {c.size}')


def quality():
    """One photograph, its own faces ranked.

    Pooling faces from every source made the count unverifiable, since the reader never sees where
    they came from.
    """
    if not has('detect_group.jpg'):
        return
    from uniface.constants import EDifFIQAWeights
    from uniface.quality import EDifFIQA

    q = EDifFIQA(model_name=EDifFIQAWeights.L, providers=CPU)
    img = strip.load(f'{SRC}/detect_group.jpg', long=2000)
    faces = strip._det.detect(img)
    scored = []
    for f in faces:
        sc = float(getattr(q.predict(img, np.asarray(f.landmarks, dtype=np.float32)), 'score', 0))
        scored.append((sc, f))
    scored.sort(key=lambda r: r[0])
    if len(scored) < 4:
        print('  skip: fewer than four faces to rank')
        return

    # source panel, cropped to the band the faces occupy so no space goes to legs and floor
    b = np.array([f.bbox for _, f in scored])
    fh = float(np.median(b[:, 3] - b[:, 1]))
    x0 = max(0, int(b[:, 0].min() - fh * 0.8))
    x1_ = min(img.shape[1], int(b[:, 2].max() + fh * 0.8))
    y0 = max(0, int(b[:, 1].min() - fh * 1.1))
    y1_ = min(img.shape[0], int(b[:, 3].max() + fh * 1.6))
    src = Canvas(img[y0:y1_, x0:x1_].copy())
    for sc, f in scored:
        col = MATCH if sc >= 0.6 else (AMBER if sc >= 0.35 else REJECT)
        bb = np.array(f.bbox, dtype=np.float32) - [x0, y0, x0, y0]
        src.corner_box(bb, color=col)
    src_bgr = to_bgr(src.result())

    TILE, PAD, G, CAP = 430, 22, 14, PT_VALUE + FOOT_GAP + PT_FOOTER + 8
    picks = [scored[0], scored[len(scored) // 3], scored[2 * len(scored) // 3], scored[-1]]
    W = PAD * 2 + TILE * 4 + G * 3
    sw = W - PAD * 2
    sh = round(src_bgr.shape[0] * sw / src_bgr.shape[1])
    src_bgr = cv2.resize(src_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    H = PAD + sh + G + TILE + CAP
    c = Canvas(Image.new('RGB', (W, H), (26, 30, 38)))
    c.base.paste(to_pil(src_bgr), (PAD, PAD))
    for i, (sc, f) in enumerate(picks):
        x, y = PAD + i * (TILE + G), PAD + sh + G
        c.base.paste(to_pil(strip.tile(img, f, size=TILE, margin=0.45)), (x, y))
        col = MATCH if sc >= 0.6 else (AMBER if sc >= 0.35 else REJECT)
        c.draw.text((x + TILE / 2, y + TILE + 22), f'{sc:.3f}', font=font(PT_VALUE), fill=(*col, 255), anchor='mm')
    c.draw.text(
        (PAD, H - PT_FOOTER - 8),
        f'Face Quality - eDifFIQA(L)  ·  one score per face  ·  {len(faces)} faces',
        font=font(PT_FOOTER),
        fill=(150, 158, 170, 255),
    )
    c.save(f'{OUT}/quality.jpg')
    print(f'  quality          {c.size}  {len(faces)} faces  {picks[0][0]:.3f} .. {picks[-1][0]:.3f}')


def anonymization():
    if not has('anon_group.jpg'):
        return
    from uniface.privacy import BlurFace

    img = strip.load(f'{SRC}/anon_group.jpg', long=1500)
    faces = strip._det.detect(img)
    outs = [
        (BlurFace(method=m).anonymize(img.copy(), faces), m) for m in ['pixelate', 'gaussian', 'elliptical', 'blackout']
    ]
    # one row of four: matches the other strips' proportions and puts the methods side by side
    PW = 460
    s = PW / img.shape[1]
    ph = round(img.shape[0] * s)
    small = [(cv2.resize(p_, (PW, ph), interpolation=cv2.INTER_AREA), m) for p_, m in outs]
    PAD, G, CAP = 20, 12, 12 + PT_LABEL + FOOT_GAP + PT_FOOTER + 14
    W = PAD * 2 + PW * 4 + G * 3
    c = Canvas(Image.new('RGB', (W, PAD + ph + CAP), (26, 30, 38)))
    for i, (p_, m) in enumerate(small):
        x = PAD + i * (PW + G)
        c.base.paste(to_pil(p_), (x, PAD))
        c.draw.text((x + 6, PAD + ph + 12), m, font=font(PT_LABEL), fill=(*PAPER, 255))
    c.draw.text(
        (PAD + 6, PAD + ph + 12 + PT_LABEL + FOOT_GAP),
        'Face Anonymization - BlurFace  ·  4 of 5 methods',
        font=font(PT_FOOTER),
        fill=(150, 158, 170, 255),
    )
    c.save(f'{OUT}/anonymization.jpg')
    print(f'  anonymization    {c.size}  {len(faces)} faces')


def verification(
    pat_a='verify_now_*.jpg',
    pat_b='verify_nowb_*.jpg',
    out='verification',
    pat_c='verify_nowc_*.jpg',
    name_a='person A',
    name_b='person B',
    name_c='person C',
):
    """Same person across years, plus unrelated negatives.

    No twins -- an identical-twin pair scores above a genuine match, which reads as a bug rather
    than a demonstration.
    """
    import glob

    same = sorted(glob.glob(f'{SRC}/{pat_a}'))
    # a set can name its own negatives via pat_b; otherwise fall back to spare portraits
    others = sorted(glob.glob(f'{SRC}/{pat_b}'))
    if not others:
        others = [
            f'{SRC}/{n}'
            for n in ('parse_face.jpg', 'state_glasses.jpg', 'age_senior.jpg')
            if os.path.exists(f'{SRC}/{n}')
        ]
    if len(same) < 2 or not others:
        print(f'  skip: need two {pat_a} and one other portrait')
        return
    from uniface.constants import AdaFaceWeights
    from uniface.face_utils import compute_similarity, face_alignment
    from uniface.recognition import AdaFace

    rec = AdaFace(model_name=AdaFaceWeights.IR_101, providers=CPU)

    def one(path, tile=340):
        img = strip.load(path)
        f = max(strip._det.detect(img), key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        a, _ = face_alignment(img, np.asarray(f.landmarks, dtype=np.float32))
        return rec.get_embedding(a), strip.tile(img, f, size=tile)

    def yr(path):
        d = ''.join(ch for ch in os.path.basename(path) if ch.isdigit())[:4]
        return d or '?'

    S = [one(p) for p in same]
    OTH = [one(p) for p in others]
    # a second identity, when supplied, earns its own same-person row and becomes the most
    # interesting negative -- two different people rather than a stranger off the street
    second = sorted(glob.glob(f'{SRC}/{pat_c}'))
    rows = []
    rows.append(
        (
            S[0],
            f'{name_a}, {yr(same[0])}',
            S[1],
            f'{name_a}, {yr(same[1])}',
            f'{int(yr(same[1])) - int(yr(same[0]))} years apart',
            'same',
        )
    )
    if len(second) >= 2:
        C = [one(x) for x in second]
        rows.append(
            (
                C[0],
                f'{name_c}, {yr(second[0])}',
                C[1],
                f'{name_c}, {yr(second[1])}',
                f'{int(yr(second[1])) - int(yr(second[0]))} years apart',
                'same',
            )
        )
        rows.append((S[-1], name_a, C[0], name_c, 'different people', 'diff'))
        rows.append((S[-1], name_a, OTH[0], name_b, 'different people', 'diff'))
    else:
        if len(S) > 2:
            rows.append(
                (
                    S[1],
                    f'{name_a}, {yr(same[1])}',
                    S[2],
                    f'{name_a}, {yr(same[2])}',
                    f'{int(yr(same[2])) - int(yr(same[1]))} years apart',
                    'same',
                )
            )
        rows.append((S[-1], f'{name_a}, {yr(same[-1])}', OTH[0], name_b, 'different people', 'diff'))
        if len(OTH) > 1:
            rows.append((OTH[0], name_b, OTH[1], name_c, 'different people', 'diff'))

    TH, TILE, MID, PADX, PADY, CAP, CG, RG = 0.40, 340, 210, 30, 28, 44, 58, 26
    cols = 2
    nrows = (len(rows) + 1) // cols
    pw = TILE * 2 + MID
    W = PADX * 2 + pw * cols + CG
    c = Canvas(Image.new('RGB', (W, PADY * 2 + (TILE + CAP) * nrows + RG * (nrows - 1) + 52), (26, 30, 38)))
    for i, ((ea, ta), na, (eb, tb), nb, note, exp) in enumerate(rows):
        r, k = divmod(i, cols)
        sc = float(compute_similarity(ea, eb))
        ok = sc >= TH
        wrong = (exp == 'same' and not ok) or (exp == 'diff' and ok)
        col = AMBER if wrong else (MATCH if ok else REJECT)
        x0, y = PADX + k * (pw + CG), PADY + r * (TILE + CAP + RG)
        for t, nm, x in ((ta, na, x0), (tb, nb, x0 + TILE + MID)):
            c.base.paste(to_pil(t), (x, y))
            c.draw.text((x, y + TILE + 10), nm, font=font(PT_LABEL), fill=(*PAPER, 255))
        cx = x0 + TILE + MID / 2
        c.draw.text((cx, y + TILE * 0.32), f'{sc:+.3f}', font=font(PT_VALUE + 6), fill=(*col, 255), anchor='mm')
        c.draw.text(
            (cx, y + TILE * 0.50),
            ('MATCH' if ok else 'NO MATCH') + ('  x' if wrong else ''),
            font=font(PT_LABEL),
            fill=(*col, 255),
            anchor='mm',
        )
        c.draw.text((cx, y + TILE * 0.63), note, font=font(PT_SUB), fill=(150, 158, 170, 255), anchor='mm')
        print(f'    {sc:+.3f} {"MATCH" if ok else "NO MATCH":<9} {note}')
    c.draw.text(
        (PADX, c.size[1] - PT_FOOTER - 12),
        'Face Recognition - AdaFace IR-101  ·  cosine similarity, threshold 0.40',
        font=font(PT_FOOTER),
        fill=(150, 158, 170, 255),
    )
    c.save(f'{OUT}/{out}.jpg')
    print(f'  {out:<16} {c.size}')


# ----------------------------------------------------------------- anti-spoofing
def spoofing():
    """A live capture beside replays of it.

    MiniFASNet judges presentation, not identity, so the frames have to come from the same setup: a
    found photograph is out of domain by definition.
    """
    order = [
        ('spoof_live.jpg', 'live capture'),
        ('spoof_print.jpg', 'printed photo'),
        ('spoof_screen.jpg', 'screen replay'),
    ]
    avail = [(n, lab) for n, lab in order if has(n)]
    if len(avail) < 2:
        print('  skip: need at least two spoof_*.jpg')
        return
    from uniface.spoofing import MiniFASNet

    sp = MiniFASNet(providers=CPU)
    panels = []
    for n, lab in avail:
        img = strip.load(f'{SRC}/{n}', long=900)
        f = strip.biggest(img)
        r = sp.predict(img, f.bbox)
        col = MATCH if r.is_real else REJECT
        c = Canvas(img.copy())
        c.corner_box(f.bbox, color=col)
        panel = cv2.resize(to_bgr(c.result()), (600, 800), interpolation=cv2.INTER_AREA)
        verdict = 'Real' if r.is_real else 'Fake'
        panels.append((panel, f'{lab}  ·  {verdict} {r.confidence:.2f}'))
        print(f'    {n:<20} {verdict:<5} {r.confidence:.4f}')
    sheet(panels, 'spoofing', 'Anti-Spoofing - MiniFASNet  ·  real or replayed')


def extras():
    """Additional variants of components that benefit from a second example."""
    if os.path.exists(f'{SRC}/seg_occluded.jpg'):
        segmentation('seg_occluded.jpg', 'segmentation_occluded')
    if os.path.exists(f'{SRC}/matte_hair.jpg'):
        matting('matte_hair.jpg', 'matting_alt')
    if os.path.exists(f'{SRC}/detect_crowd.jpg'):
        detection('detect_crowd.jpg', 'detection_alt')
    import glob as _g0

    if _g0.glob(f'{SRC}/state_b_*.jpg'):
        face_states('state_b', 'face_states_alt', open_src='age_adult.jpg', second_src='mesh_face.jpg')
    import glob as _g

    if _g.glob(f'{SRC}/verify_einstein_*.jpg'):
        verification(
            'verify_einstein_*.jpg',
            'verify_curie*.jpg',
            'verification_alt',
            'verify_bohr_*.jpg',
            name_a='Einstein',
            name_b='Curie',
            name_c='Bohr',
        )


if __name__ == '__main__':
    print(f'building {OUT} from {SRC}')
    for fn in (
        detection,
        landmarks,
        mesh,
        parsing,
        segmentation,
        matting,
        lambda: pose_or_gaze('headpose'),
        lambda: pose_or_gaze('gaze'),
        demography,
        emotion,
        face_states,
        quality,
        spoofing,
        anonymization,
        verification,
        extras,
    ):
        try:
            fn()
        except Exception as e:
            print(f'  FAILED {getattr(fn, "__name__", "lambda")}: {type(e).__name__}: {str(e)[:90]}')
    print(f'\n{len(os.listdir(OUT))} figures in {OUT}')
