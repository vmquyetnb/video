"""Turn an ordinary photo into a graphite pencil drawing (classical image processing)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from . import analysis as A
from .imaging import luminance

F32 = np.float32
PAPER = np.array([0.955, 0.948, 0.928], F32)
GRAPHITE = np.array([0.21, 0.215, 0.235], F32)  # densest soft graphite: a cool grey, not black


def crop_to(rgb: np.ndarray, aspect: float) -> np.ndarray:
    """Crop to ``aspect`` (width / height), keeping the subject in view.

    A detected face is kept horizontally centred and in the upper part of the
    frame; otherwise the crop follows the most detailed area.
    """
    h, w = rgb.shape[:2]
    if abs(w / h - aspect) < 1e-3:
        return rgb
    focus = A.detect_focus(luminance(rgb))
    fx, fy = focus.centre
    if w / h > aspect:  # too wide: choose a horizontal window
        cw = int(round(h * aspect))
        x0 = int(np.clip(fx - cw / 2, 0, w - cw))
        return rgb[:, x0:x0 + cw]
    ch = int(round(w / aspect))
    target = fy - (0.38 if focus.face is not None else 0.5) * ch
    y0 = int(np.clip(target, 0, h - ch))
    return rgb[y0:y0 + ch]


def photo_to_pencil(rgb: np.ndarray, size: Tuple[int, int], seed: int = 0,
                    paper: np.ndarray = PAPER) -> np.ndarray:
    """Graphite drawing of ``rgb`` (float RGB in [0, 1]) at ``size`` = (width, height).

    Fine contours come from a colour-dodge sketch (the photo divided by a
    blurred negative); tones come from the locally contrast-enhanced photo,
    compressed to the range of graphite on paper. The edges fade into the
    paper like a hand-drawn illustration.
    """
    width, height = size
    img = crop_to(rgb, width / height)
    interp = cv2.INTER_AREA if img.shape[1] > width else cv2.INTER_CUBIC
    img = np.clip(cv2.resize(img, (width, height), interpolation=interp), 0.0, 1.0)
    s = min(width, height) / 1080.0
    rng = np.random.default_rng(seed)

    gray = luminance(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(cv2.convertScaleAbs(gray, alpha=255.0)).astype(F32) / 255.0
    gray = A.blur(gray, 0.6 * s)

    # Contours: colour dodge of the photo with its blurred negative.
    negative = A.blur(1.0 - gray, 5.0 * s + 1.0)
    lines = np.clip(gray / np.maximum(1.0 - negative, 1e-3), 0.0, 1.0) ** 1.7
    # Tones: smoothed and compressed, lighter overall like graphite on white paper.
    tone = np.clip((A.blur(gray, 2.0 * s) - 0.04) / 0.92, 0.0, 1.0) ** 0.85
    value = lines * (0.40 + 0.60 * tone)
    value = 0.10 + 0.90 * value

    # Graphite grain in the darker areas, faint paper grain everywhere.
    grain = A.blur(rng.standard_normal((height, width)).astype(F32), 0.8 * s)
    grain /= grain.std() + 1e-6
    darkness = (1.0 - value) * (1.0 + 0.10 * grain)
    darkness *= _soft_border(height, width, rng, s)
    value = np.clip(1.0 - darkness, 0.0, 1.0)
    paper_grain = A.blur(rng.standard_normal((height, width)).astype(F32), 0.6 * s) * 0.006
    return np.clip(value[..., None] * paper[None, None, :] * (1.0 + paper_grain[..., None]),
                   0.0, 1.0).astype(F32)


def graphite_drawing(rgb: np.ndarray, size: Tuple[int, int], seed: int = 0,
                     paper: np.ndarray = PAPER, border: float = 0.03) -> np.ndarray:
    """Pencil drawing of ``rgb`` (float RGB in [0, 1]) at ``size`` = (width, height)
    in which the graphite itself is visible.

    Tones are laid in hatching strokes over a lightly blended base: the strokes
    follow the form where the picture has a direction (hair, folds) and run
    diagonally in flat areas, and a crossing set is added in the darks. Light
    strokes skip the pits of the paper, outlines vary in pressure, and the
    darkest graphite is a cool grey rather than black. ``border`` is the share
    of the shorter side over which the drawing fades out at its edges.
    """
    width, height = size
    img = crop_to(rgb, width / height)
    interp = cv2.INTER_AREA if img.shape[1] > width else cv2.INTER_CUBIC
    img = np.clip(cv2.resize(img, (width, height), interpolation=interp), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    px = min(width, height) / 400.0  # stroke scale, relative to the size of the drawing

    gray = luminance(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    gray = A.blur(clahe.apply(cv2.convertScaleAbs(gray, alpha=255.0)).astype(F32) / 255.0,
                  0.4 * px)

    # Outlines: colour dodge with the blurred negative; the pressure changes along them.
    negative = A.blur(1.0 - gray, 2.5 * px + 1.0)
    dodge = np.clip(gray / np.maximum(1.0 - negative, 1e-3), 0.0, 1.0)
    outline = np.clip((1.0 - dodge) * 2.4, 0.0, 1.0) ** 1.3
    outline *= 0.7 + 0.3 * A.uniformize(A.blur(rng.standard_normal((height, width)).astype(F32),
                                               5.0 * px))

    # How much graphite each area needs (0 = bare paper).
    need = np.clip((0.96 - A.blur(gray, 1.0 * px)) / 0.86, 0.0, 1.0) ** 1.1

    # Stroke textures: streaks along the form, and a crossing set.
    vx, vy, _ = A.orientation_field(gray, max(px, 0.6))
    length, step = max(6, round(11 * px)), max(0.8, 1.0 * px)
    dots = [A.blur((rng.random((height, width)) < 0.045).astype(F32), 0.45 * max(px, 0.8))
            for _ in range(2)]
    (along,) = A.line_integral_convolution(vx, vy, [dots[0]], length, step)
    cvx, cvy = A.rotate_field(vx, vy, 70.0)
    (across,) = A.line_integral_convolution(cvx, cvy, [dots[1]], length, step)
    hatch_a, hatch_b = A.uniformize(along), A.uniformize(across)
    first = A.smoothstep((np.clip(1.5 * need, 0.0, 1.0) - hatch_a) / 0.14)
    second = A.smoothstep((np.clip(1.8 * (need - 0.42), 0.0, 1.0) - hatch_b) / 0.14)
    strokes = 1.0 - (1.0 - first) * (1.0 - 0.85 * second)
    darkness = need * (0.38 + 0.62 * strokes)

    # Light pressure leaves the pits of the paper white.
    tooth = A.uniformize(A.blur(rng.standard_normal((height, width)).astype(F32), 0.5 * max(px, 0.8)))
    darkness *= 1.0 - 0.65 * A.ramp(0.34 - tooth, 0.0, 0.14) * (1.0 - need) ** 0.6
    darkness = 1.0 - (1.0 - darkness) * (1.0 - 0.85 * outline)
    grain = A.blur(rng.standard_normal((height, width)).astype(F32), 0.6 * max(px, 0.8))
    darkness *= 1.0 + 0.06 * grain / (grain.std() + 1e-6)
    darkness = np.clip(darkness * _soft_border(height, width, rng, px, border), 0.0, 1.0)[..., None]
    paper_grain = A.blur(rng.standard_normal((height, width)).astype(F32), 0.6) * 0.006
    sheet = paper[None, None, :] * (1.0 + paper_grain[..., None])
    return np.clip(sheet * (1.0 - darkness) + GRAPHITE[None, None, :] * darkness,
                   0.0, 1.0).astype(F32)


def _soft_border(height: int, width: int, rng: np.random.Generator, s: float,
                 share: float = 0.08) -> np.ndarray:
    """1 inside, fading irregularly to 0 at the edges (a vignetted sketch)."""
    margin = max(1.0, share * min(width, height))
    yy, xx = np.mgrid[0:height, 0:width].astype(F32)
    edge = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy])
    wobble = cv2.resize(rng.standard_normal((height // 40 + 2, width // 40 + 2)).astype(F32),
                        (width, height), interpolation=cv2.INTER_CUBIC) * 0.15 * margin
    return A.smoothstep((edge + wobble) / margin)
