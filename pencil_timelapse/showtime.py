"""Showmanship for the football-players video, to keep viewers watching.

* an intro hook: the finished group picture first, a punchy headline, then an
  eraser sweep back to blank paper and "Let's draw!";
* before each player a "Who's next?" teaser with a mystery silhouette;
* after each player a highlight: the camera pushes in, a light glint runs over
  the graphite and a numbered badge pops on;
* an ending call to action: the next-team question, a comment prompt and
  animated Like / Subscribe / bell buttons that a cursor clicks;
* short synthesised sound effects for all of these.

Everything is drawn with OpenCV and Pillow; nothing is downloaded.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

F32 = np.float32
RATE = 44100

DISPLAY_FONTS = ("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Impact.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
                 "C:/Windows/Fonts/impact.ttf")
BOLD_FONTS = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf")
HAND_FONTS = ("/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
              "/System/Library/Fonts/Noteworthy.ttc", "C:/Windows/Fonts/segoepr.ttf")

INK = (0.227, 0.22, 0.204)
WHITE = (1.0, 1.0, 1.0)
RED = (0.86, 0.11, 0.09)  # subscribe-button red
GREY = (0.55, 0.54, 0.52)


@dataclass
class ShowOptions:
    """What the players video adds around the drawing."""

    intro: bool = True
    highlights: bool = True  # "who's next?" teaser and a highlight after each player
    outro: bool = True  # like / subscribe / comment ending
    sound_effects: bool = True
    hook: str = ""  # intro headline; the title when empty
    question: str = "Which team should I draw next?"


@dataclass
class Clip:
    """A stretch of video made by an effect: ``seconds`` long, frames from
    ``frames()``, and sound effects at times relative to its start."""

    seconds: float
    frames: Callable[[], Iterator[np.ndarray]]
    sounds: List[Tuple[float, str]] = field(default_factory=list)
    last: Optional[Callable[[], np.ndarray]] = None  # its final frame, when cheap to get
    speech: List[Tuple[float, np.ndarray]] = field(default_factory=list)  # (time, samples) said in it


# --------------------------------------------------------------------------- basics

def _font(paths: Sequence[str], size: int) -> ImageFont.FreeTypeFont:
    """The first font found; "file.ttc#2" picks a face inside a collection."""
    for path in paths:
        file, _, index = path.partition("#")
        if os.path.isfile(file):
            return ImageFont.truetype(file, max(8, int(size)), index=int(index or 0))
    return ImageFont.load_default()


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def back(t: float, overshoot: float = 1.7) -> float:
    """Ease out with a small overshoot (a pop)."""
    t = min(max(t, 0.0), 1.0) - 1.0
    return 1.0 + t * t * ((overshoot + 1.0) * t + overshoot)


def _phase(t: float, start: float, length: float) -> float:
    return min(max((t - start) / length, 0.0), 1.0)


class Sprite:
    """An RGBA picture (float, straight alpha) that can be pasted scaled and faded."""

    def __init__(self, rgba: np.ndarray):
        self.rgb = rgba[..., :3].astype(F32)
        self.alpha = rgba[..., 3:].astype(F32)

    @property
    def size(self) -> Tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]

    def paste_left(self, frame: np.ndarray, x: float, y: float, reveal: float = 1.0,
                   opacity: float = 1.0) -> None:
        """Blend with the left edge at ``x`` and the top at ``y``, showing only
        the first ``reveal`` share of its width (text being written)."""
        w = int(round(self.rgb.shape[1] * min(max(reveal, 0.0), 1.0)))
        if w <= 0 or opacity <= 0.003:
            return
        edge = np.ones((1, w, 1), F32)
        soft = min(w, max(2, self.rgb.shape[0] // 3))
        if reveal < 1.0:  # a soft leading edge, like ink still arriving
            edge[0, w - soft:, 0] = np.linspace(1.0, 0.0, soft)
        part = Sprite(np.concatenate([self.rgb[:, :w], self.alpha[:, :w] * edge], axis=2))
        part.paste(frame, x + w / 2, y + self.rgb.shape[0] / 2, opacity=opacity)

    def paste(self, frame: np.ndarray, cx: float, cy: float, scale: float = 1.0,
              opacity: float = 1.0, angle: float = 0.0) -> None:
        """Blend onto ``frame`` (float RGB) centred at (cx, cy)."""
        if opacity <= 0.003 or scale <= 0.01:
            return
        rgb, alpha = self.rgb, self.alpha
        if scale != 1.0 or angle:
            h, w = alpha.shape[:2]
            side = int(math.ceil(math.hypot(w, h) * scale)) + 2 if angle else 0
            out_w, out_h = (side, side) if angle else (max(1, int(w * scale)), max(1, int(h * scale)))
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            m[0, 2] += out_w / 2 - w / 2
            m[1, 2] += out_h / 2 - h / 2
            rgb = cv2.warpAffine(rgb, m, (out_w, out_h), flags=cv2.INTER_LINEAR)
            alpha = cv2.warpAffine(alpha, m, (out_w, out_h), flags=cv2.INTER_LINEAR)[..., None]
        h, w = alpha.shape[:2]
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        a = alpha[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] * opacity
        region = frame[fy0:fy1, fx0:fx1]
        region *= 1.0 - a
        region += rgb[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] * a


def _render(draw_fn: Callable[[ImageDraw.ImageDraw], None], size: Tuple[int, int],
            supersample: int = 2) -> Sprite:
    """Draw with Pillow at a higher resolution, then scale down (smooth edges)."""
    w, h = size
    img = Image.new("RGBA", (w * supersample, h * supersample), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img), supersample)
    img = img.resize((w, h), Image.LANCZOS)
    return Sprite(np.asarray(img).astype(F32) / 255.0)


def _rgb255(colour: Tuple[float, float, float], alpha: float = 1.0) -> Tuple[int, int, int, int]:
    return tuple(int(round(c * 255)) for c in colour) + (int(round(alpha * 255)),)


def text_sprite(text: str, fonts: Sequence[str], size: int, fill=WHITE, stroke=0,
                stroke_fill=INK, tracking: float = 0.0, max_width: Optional[int] = None) -> Sprite:
    """A line of text; ``tracking`` adds letter spacing (share of the size);
    shrinks to ``max_width`` if given."""
    font = _font(fonts, size)

    def width(f: ImageFont.FreeTypeFont) -> float:
        return f.getlength(text) + tracking * f.size * max(0, len(text) - 1) + 2 * stroke

    while max_width and width(font) > max_width and font.size > 10:
        font = _font(fonts, int(font.size * 0.92))
    pad = stroke + int(font.size * 0.15) + 2
    asc, desc = font.getmetrics()
    w, h = int(width(font)) + 2 * pad, asc + desc + 2 * pad
    scale = 2
    big = _font(fonts, font.size * scale)

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        x = pad * s
        for ch in text if tracking else [text]:
            d.text((x, pad * s), ch, font=big, fill=_rgb255(fill), stroke_width=stroke * s,
                   stroke_fill=_rgb255(stroke_fill))
            x += big.getlength(ch) + tracking * big.size

    return _render(draw, (w, h), scale)


def zoom_view(img: np.ndarray, cx: float, cy: float, z: float) -> np.ndarray:
    """The frame seen by a camera zoomed ``z`` times towards (cx, cy)."""
    if z <= 1.0001:
        return img
    h, w = img.shape[:2]
    cx = min(max(cx, w / (2 * z)), w - w / (2 * z))
    cy = min(max(cy, h / (2 * z)), h - h / (2 * z))
    m = np.array([[z, 0, w / 2 - z * cx], [0, z, h / 2 - z * cy]], np.float32)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _f(img: np.ndarray) -> np.ndarray:
    return img.astype(F32) / 255.0 if img.dtype == np.uint8 else img.astype(F32)


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- icons

def _thumb(d: ImageDraw.ImageDraw, x: float, y: float, s: float, colour) -> None:
    """A thumbs-up in a box of side ``s`` at (x, y)."""
    def r(a, b, c, e, rad):
        d.rounded_rectangle([x + a * s, y + b * s, x + c * s, y + e * s], radius=rad * s, fill=colour)
    r(0.04, 0.44, 0.24, 0.96, 0.04)  # cuff
    r(0.30, 0.42, 0.92, 0.96, 0.10)  # fist
    d.polygon([(x + 0.30 * s, y + 0.50 * s), (x + 0.44 * s, y + 0.05 * s),
               (x + 0.60 * s, y + 0.08 * s), (x + 0.58 * s, y + 0.46 * s)], fill=colour)


def _bell(d: ImageDraw.ImageDraw, x: float, y: float, s: float, colour) -> None:
    d.ellipse([x + 0.43 * s, y + 0.02 * s, x + 0.57 * s, y + 0.16 * s], fill=colour)
    d.chord([x + 0.18 * s, y + 0.08 * s, x + 0.82 * s, y + 0.9 * s], 180, 360, fill=colour)
    d.rectangle([x + 0.18 * s, y + 0.48 * s, x + 0.82 * s, y + 0.74 * s], fill=colour)
    d.polygon([(x + 0.18 * s, y + 0.72 * s), (x + 0.82 * s, y + 0.72 * s),
               (x + 0.94 * s, y + 0.82 * s), (x + 0.06 * s, y + 0.82 * s)], fill=colour)
    d.ellipse([x + 0.40 * s, y + 0.82 * s, x + 0.60 * s, y + 1.0 * s], fill=colour)


def _bubble(d: ImageDraw.ImageDraw, x: float, y: float, s: float, colour, dots) -> None:
    d.rounded_rectangle([x, y, x + s, y + 0.72 * s], radius=0.2 * s, fill=colour)
    d.polygon([(x + 0.2 * s, y + 0.66 * s), (x + 0.16 * s, y + 0.98 * s),
               (x + 0.45 * s, y + 0.68 * s)], fill=colour)
    for i in range(3):
        cx = x + (0.3 + 0.2 * i) * s
        d.ellipse([cx - 0.06 * s, y + 0.3 * s, cx + 0.06 * s, y + 0.42 * s], fill=dots)


def _cursor(size: int) -> Sprite:
    pts = [(0, 0), (0, 0.78), (0.2, 0.6), (0.34, 0.92), (0.46, 0.87), (0.32, 0.56), (0.58, 0.56)]

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        k, o = size * s * 0.85, size * s * 0.08
        d.polygon([(o + px * k, o + py * k) for px, py in pts], fill=(255, 255, 255, 255),
                  outline=(20, 20, 20, 255), width=max(2, int(0.05 * k)))

    return _render(draw, (size, size))


def _pill(text: str, height: int, fill, text_fill, icon=None, outline=None) -> Sprite:
    font = _font(BOLD_FONTS, int(height * 0.42))
    icon_w = height * 0.62 if icon else 0
    w = int(font.getlength(text) + icon_w + height * 0.9)

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        big = _font(BOLD_FONTS, int(height * 0.42) * s)
        d.rounded_rectangle([s, s, w * s - s, height * s - s], radius=height * s / 2,
                            fill=_rgb255(fill), outline=_rgb255(outline) if outline else None,
                            width=max(2, int(height * 0.05)) * s)
        x = height * 0.45 * s
        if icon:
            icon(d, x, height * 0.2 * s, height * 0.55 * s, _rgb255(text_fill))
            x += icon_w * s
        asc, desc = big.getmetrics()
        d.text((x, (height * s - asc - desc * 0.4) / 2), text, font=big, fill=_rgb255(text_fill))

    return _render(draw, (w, height))


# --------------------------------------------------------------------------- clips

def intro_clip(final: np.ndarray, blank: np.ndarray, headline: str, kicker: str, tagline: str,
               fps: int) -> Clip:
    """The finished picture first, with a headline slammed on, then an eraser
    sweep back to blank paper and "Let's draw!"."""
    H, W = final.shape[:2]
    art, paper = _f(final), _f(blank)
    dim = art * 0.42 + np.array([0.06, 0.06, 0.07], F32) * 0.58
    k_s = text_sprite(kicker, BOLD_FONTS, int(0.045 * H), WHITE, tracking=0.18, max_width=int(0.9 * W))
    h_s = text_sprite(headline, DISPLAY_FONTS, int(0.2 * H), WHITE, stroke=max(2, H // 160),
                      max_width=int(0.92 * W))
    t_s = text_sprite(tagline, HAND_FONTS, int(0.058 * H), (0.96, 0.93, 0.85), max_width=int(0.9 * W))
    go = text_sprite("Let's draw!", HAND_FONTS, int(0.15 * H), INK, max_width=int(0.8 * W))
    show, sweep, total = 2.7, 0.8, 5.0

    def titled(t: float) -> np.ndarray:
        z = 1.0 + 0.16 * (1.0 - smooth(t / show))
        frame = zoom_view(dim, W / 2, H / 2, z).copy()
        a = _phase(t, 0.15, 0.3)
        k_s.paste(frame, W / 2, 0.33 * H - 0.04 * H * (1 - smooth(a)), opacity=a)
        b = _phase(t, 0.35, 0.28)
        h_s.paste(frame, W / 2, 0.5 * H, scale=1.0 + 0.6 * (1.0 - back(b)), opacity=min(1.0, b * 2))
        t_s.paste(frame, W / 2, 0.68 * H, opacity=_phase(t, 1.0, 0.4))
        return frame

    def frames() -> Iterator[np.ndarray]:
        n = int(round(total * fps))
        last = titled(show)
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        diag = (xx + 0.45 * yy) / (W + 0.45 * H)  # 0..1 along the sweep
        for i in range(n):
            t = i / fps
            if t < show:
                yield _u8(titled(t))
            elif t < show + sweep:
                p = smooth((t - show) / sweep) * 1.08 - 0.04
                gone = np.clip((p - diag) / 0.04 + 0.5, 0.0, 1.0)[..., None]
                streak = np.exp(-((diag - p) / 0.015) ** 2)[..., None] * 0.35
                yield _u8(np.clip(last * (1 - gone) + paper * gone + streak, 0.0, 1.0))
            else:
                frame = paper.copy()
                u = t - show - sweep
                fade = 1.0 - _phase(t, total - 0.35, 0.35)
                go.paste(frame, W / 2, 0.5 * H, scale=0.55 + 0.45 * back(_phase(u, 0.05, 0.35)),
                         opacity=min(1.0, _phase(u, 0.05, 0.12)) * fade)
                yield _u8(frame)

    return Clip(total, frames, [(0.42, "thud"), (show, "whoosh"), (show + sweep + 0.08, "pop")])


def sweep_clip(start: np.ndarray, end: np.ndarray, fps: int, seconds: float = 0.8) -> Clip:
    """An eraser sweep across the page from ``start`` to ``end``."""
    H, W = start.shape[:2]
    a, b = _f(start), _f(end)

    def frames() -> Iterator[np.ndarray]:
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        diag = (xx + 0.45 * yy) / (W + 0.45 * H)
        for i in range(int(round(seconds * fps))):
            p = smooth(i / max(seconds * fps - 1, 1)) * 1.08 - 0.04
            gone = np.clip((p - diag) / 0.04 + 0.5, 0.0, 1.0)[..., None]
            streak = np.exp(-((diag - p) / 0.015) ** 2)[..., None] * 0.35
            yield _u8(np.clip(a * (1 - gone) + b * gone + streak, 0.0, 1.0))

    return Clip(seconds, frames, [(0.0, "whoosh")])


def spotlight_clip(page: np.ndarray, cell: Tuple[int, int, int, int],
                   name_box: Tuple[int, int, int, int], number: int, fps: int) -> Clip:
    """Push in on the portrait just finished: a glint over the graphite and a
    numbered badge, then back out."""
    H, W = page.shape[:2]
    base = _f(page)
    x, y, w, h = cell
    nx, ny, nw, nh = name_box
    bx0, by0 = min(x, nx), y
    bx1, by1 = max(x + w, nx + nw), ny + nh
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    z = max(1.0, min(0.86 * H / (by1 - by0), 0.86 * W / (bx1 - bx0), 2.6))
    r = max(14, int(0.085 * h))

    def badge_draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse([s, s, (2 * r - 1) * s, (2 * r - 1) * s], fill=_rgb255(INK),
                  outline=_rgb255((0.98, 0.97, 0.94)), width=max(2, r // 7) * s)
        font = _font(DISPLAY_FONTS, int(r * 1.2) * s)
        label = str(number)
        asc, desc = font.getmetrics()
        d.text((r * s - font.getlength(label) / 2, r * s - (asc - desc * 0.3) / 2 - desc * 0.2),
               label, font=font, fill=(255, 255, 255, 255))

    badge = _render(badge_draw, (2 * r, 2 * r))
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    slant = (xx + 0.5 * yy) / (w + 0.5 * h)
    total, zin, zout = 2.6, 0.6, 0.6

    def frames() -> Iterator[np.ndarray]:
        for i in range(int(round(total * fps))):
            t = i / fps
            k = smooth(t / zin) if t < total - zout else smooth((total - t) / zout)
            frame = base.copy()
            g = _phase(t, 0.55, 0.7)
            if 0.0 < g < 1.0:  # light running over the graphite, like a sheen
                band = np.exp(-((slant - (g * 1.4 - 0.2)) / 0.07) ** 2)[..., None]
                region = frame[y:y + h, x:x + w]
                region += band * 0.55 * (1.0 - region) * (region < 0.85)
            b = _phase(t, 0.62, 0.3)
            fade = 1.0 - _phase(t, total - zout, zout * 0.8)
            badge.paste(frame, x + 0.6 * r, y + 0.6 * r, scale=back(b), opacity=min(1.0, 3 * b) * fade)
            yield _u8(np.clip(zoom_view(frame, cx, cy, 1.0 + (z - 1.0) * k), 0.0, 1.0))

    return Clip(total, frames, [(0.05, "swish"), (0.64, "ding")])


def teaser_clip(page: np.ndarray, cell: Tuple[int, int, int, int], silhouette: np.ndarray,
                label: str, count: str, fps: int) -> Clip:
    """A mystery silhouette with a big question mark where the next player goes."""
    H, W = page.shape[:2]
    base = _f(page)
    x, y, w, h = cell
    shape = cv2.resize(silhouette.astype(F32), (w, h))[..., None]
    q = text_sprite("?", DISPLAY_FONTS, int(0.5 * h), WHITE, stroke=max(2, h // 90))
    tag = text_sprite(label, DISPLAY_FONTS, int(0.12 * h), WHITE, stroke=max(2, h // 140),
                      max_width=int(1.15 * w))
    num = text_sprite(count, BOLD_FONTS, int(0.06 * h), WHITE, stroke=max(1, h // 200))
    dark = np.array([0.17, 0.17, 0.19], F32)
    total = 1.4

    def frames() -> Iterator[np.ndarray]:
        for i in range(int(round(total * fps))):
            t = i / fps
            a = smooth(_phase(t, 0.0, 0.25)) * (1.0 - smooth(_phase(t, total - 0.3, 0.3)))
            frame = base.copy()
            region = frame[y:y + h, x:x + w]
            region[:] = region * (1 - 0.85 * a * shape) + dark * 0.85 * a * shape
            pulse = 1.0 + 0.07 * math.sin(math.pi * _phase(t, 0.3, 0.5))
            q.paste(frame, x + w / 2, y + 0.42 * h, scale=pulse * (0.6 + 0.4 * back(_phase(t, 0.05, 0.3))),
                    opacity=a)
            tag.paste(frame, x + w / 2, y + 0.84 * h, opacity=a)
            num.paste(frame, x + w / 2, y + 0.95 * h, opacity=a)
            yield _u8(frame)

    return Clip(total, frames, [(0.08, "tick")])


def outro_clip(final: np.ndarray, question: str, credits: Sequence[str], fps: int,
               credit_label: str = "Photos: Wikimedia Commons") -> Clip:
    """Call to action: the next-team question, a comment prompt and Like /
    Subscribe / bell buttons that a cursor clicks one after another; the photo
    ``credits`` in small print under ``credit_label``."""
    H, W = final.shape[:2]
    art = _f(final)
    soft = cv2.GaussianBlur(art, (0, 0), 0.01 * W)
    bg = soft * 0.35 + np.array([0.97, 0.96, 0.93], F32) * 0.65
    size = int(0.105 * H)
    lines = _wrap(question.upper(), DISPLAY_FONTS, size, int(0.86 * W))
    while len(lines) > 3 and size > 20:  # a long question in a tall frame: smaller, at most 3 lines
        size = int(size * 0.88)
        lines = _wrap(question.upper(), DISPLAY_FONTS, size, int(0.86 * W))
    step = size * 1.1
    q_s = [text_sprite(line, DISPLAY_FONTS, size, INK, max_width=int(0.9 * W)) for line in lines]
    bh = int(min(0.085 * H, 0.082 * W))  # the three buttons fit across a tall frame too
    c_s = text_sprite("Tell me in the comments!", HAND_FONTS, int(0.06 * H), INK,
                      max_width=int(min(0.8 * W, 0.92 * W - 0.14 * H)))  # room for the bubble icon
    bubble = _render(lambda d, s: _bubble(d, s, s, (int(0.07 * H) - 2) * s, _rgb255(INK),
                                          (255, 255, 255, 255)), (int(0.07 * H), int(0.07 * H)))
    like = _pill("LIKE", bh, WHITE, INK, _thumb, outline=INK)
    liked = _pill("LIKED", bh, INK, WHITE, _thumb)
    sub = _pill("SUBSCRIBE", bh, RED, WHITE)
    subbed = _pill("SUBSCRIBED", bh, GREY, WHITE)
    bell = _render(lambda d, s: (d.ellipse([s, s, (bh - 1) * s, (bh - 1) * s], fill=_rgb255(WHITE),
                                           outline=_rgb255(INK), width=max(2, bh // 20) * s),
                                 _bell(d, bh * 0.24 * s, bh * 0.22 * s, bh * 0.52 * s, _rgb255(INK))),
                   (bh, bh))
    cursor = _cursor(int(0.07 * H))
    small = [text_sprite(line, BOLD_FONTS, int(0.019 * H), (0.35, 0.34, 0.32), max_width=int(0.94 * W))
             for line in _wrap(f"{credit_label} - " + " | ".join(credits), BOLD_FONTS,
                               int(0.019 * H), int(0.92 * W))[:3]] if credits else []
    gap = int(0.03 * W)
    widths = [max(like.size[0], liked.size[0]), max(sub.size[0], subbed.size[0]), bh]
    row = sum(widths) + 2 * gap
    xs = [W / 2 - row / 2 + sum(widths[:i]) + i * gap + widths[i] / 2 for i in range(3)]
    by = 0.66 * H
    q_top = 0.2 * H if len(lines) > 1 else 0.27 * H
    clicks = [(2.3, 0), (3.2, 1), (4.0, 2)]  # time, button
    total = 7.0

    def cursor_at(t: float) -> Tuple[float, float]:
        path = [(1.5, W * 0.92, H * 1.05)] + [(c - 0.15, xs[b] + 0.15 * widths[b], by + 0.2 * bh)
                                             for c, b in clicks] + [(5.2, W * 0.95, H * 1.1)]
        for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
            if t <= t1:
                k = smooth((t - t0) / max(t1 - t0, 1e-6)) if t >= t0 else 0.0
                return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
        return path[-1][1], path[-1][2]

    def frames() -> Iterator[np.ndarray]:
        for i in range(int(round(total * fps))):
            t = i / fps
            frame = zoom_view(bg, W / 2, H / 2, 1.0 + 0.05 * t / total).copy()
            for j, s in enumerate(q_s):
                a = _phase(t, 0.15 + 0.12 * j, 0.3)
                s.paste(frame, W / 2, q_top + j * step, scale=1.0 + 0.4 * (1 - back(a)),
                        opacity=min(1.0, 2 * a))
            a = _phase(t, 0.7, 0.35)
            cy = q_top + len(lines) * step + 0.03 * H
            bubble.paste(frame, W / 2 - c_s.size[0] / 2 - 0.05 * H, cy, scale=back(a), opacity=a)
            c_s.paste(frame, W / 2 + 0.01 * H, cy, opacity=a)
            for b, (normal, done) in enumerate([(like, liked), (sub, subbed), (bell, bell)]):
                a = _phase(t, 1.0 + 0.15 * b, 0.3)
                clicked = t >= clicks[b][0]
                press = 1.0 - 0.1 * math.sin(math.pi * _phase(t, clicks[b][0] - 0.05, 0.2))
                angle = 0.0
                if b == 2 and clicked:  # the bell rings
                    angle = 22.0 * math.sin(10 * math.pi * _phase(t, clicks[b][0], 0.8)) * \
                        (1.0 - _phase(t, clicks[b][0], 0.8))
                (done if clicked else normal).paste(frame, xs[b], by, scale=back(a) * press,
                                                    opacity=min(1.0, 2 * a), angle=angle)
            if 1.5 <= t <= 5.2:
                px, py = cursor_at(t)
                cursor.paste(frame, px + cursor.size[0] * 0.42, py + cursor.size[1] * 0.42)
            for j, s in enumerate(small):
                s.paste(frame, W / 2, 0.88 * H + j * 0.028 * H, opacity=_phase(t, 1.2, 0.5))
            if t > total - 0.35:  # end on a clean sheet
                frame += (np.array([0.97, 0.96, 0.93], F32) - frame) * _phase(t, total - 0.35, 0.35)
            yield _u8(frame)

    sounds = [(0.2, "thud")] + [(c, "click") for c, _ in clicks] + [(clicks[2][0] + 0.05, "ding")]
    return Clip(total, frames, sounds)


def _wrap(text: str, fonts: Sequence[str], size: int, width: int) -> List[str]:
    font = _font(fonts, size)
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.getlength(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


# --------------------------------------------------------------------------- sound

def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def _noise(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    spec[(f < lo) | (f > hi)] = 0
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-9)


def sound(name: str, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """A short synthesised sound effect (mono, ``RATE`` Hz)."""
    rng = rng or np.random.default_rng(7)
    if name == "whoosh":
        n = int(0.7 * RATE)
        t = np.arange(n) / n
        return 0.35 * _noise(n, 250, 3500, rng) * np.sin(np.pi * t) ** 2
    if name == "swish":
        n = int(0.4 * RATE)
        t = np.arange(n) / n
        return 0.16 * _noise(n, 600, 5000, rng) * np.sin(np.pi * t) ** 3
    if name == "pop":
        n = int(0.18 * RATE)
        t = np.arange(n) / RATE
        f = 620 - 300 * np.minimum(t / 0.08, 1.0)
        return 0.5 * np.sin(2 * np.pi * np.cumsum(f) / RATE) * _env(n, 0.002, 0.05)
    if name == "ding":
        n = int(1.2 * RATE)
        t = np.arange(n) / RATE
        tone = np.sin(2 * np.pi * 1318.5 * t) + 0.45 * np.sin(2 * np.pi * 2637 * t)
        return 0.18 * tone * _env(n, 0.003, 0.35)
    if name == "click":
        n = int(0.06 * RATE)
        t = np.arange(n) / RATE
        return 0.4 * (np.sin(2 * np.pi * 2100 * t) * _env(n, 0.0005, 0.008)
                      + 0.5 * _noise(n, 1000, 8000, rng) * _env(n, 0.0002, 0.004))
    if name == "tick":
        n = int(0.3 * RATE)
        out = np.zeros(n)
        for start in (0, int(0.14 * RATE)):
            m = int(0.04 * RATE)
            t = np.arange(m) / RATE
            out[start:start + m] += np.sin(2 * np.pi * 2900 * t) * _env(m, 0.0005, 0.01)
        return 0.3 * out
    if name == "sparkle":  # a rising shimmer: the drawing comes to life
        n = int(1.1 * RATE)
        out = np.zeros(n)
        for k, f in enumerate((1046.5, 1318.5, 1568.0, 2093.0, 2637.0)):
            start = int(k * 0.07 * RATE)
            t = np.arange(n - start) / RATE
            out[start:] += np.sin(2 * np.pi * f * t) * _env(n - start, 0.004, 0.25) * (0.9 - 0.1 * k)
        t = np.arange(n) / n
        out += 0.25 * _noise(n, 5000, 12000, rng) * np.sin(np.pi * t) ** 2
        return 0.12 * out
    if name == "boing":  # a spring: rising pitch with a wobble
        n = int(0.45 * RATE)
        t = np.arange(n) / RATE
        f = 180 + 420 * np.minimum(t / 0.25, 1.0) + 40 * np.sin(2 * np.pi * 18 * t)
        return 0.32 * np.sin(2 * np.pi * np.cumsum(f) / RATE) * _env(n, 0.003, 0.16)
    if name == "step":  # a soft footstep
        n = int(0.09 * RATE)
        t = np.arange(n) / RATE
        return 0.22 * (np.sin(2 * np.pi * 95 * t) * _env(n, 0.002, 0.03)
                       + 0.4 * _noise(n, 200, 1800, rng) * _env(n, 0.001, 0.012))
    if name == "clap":  # two hands meeting
        n = int(0.12 * RATE)
        return 0.5 * (_noise(n, 600, 6000, rng) * _env(n, 0.0005, 0.018)
                      + 0.3 * _noise(n, 150, 900, rng) * _env(n, 0.001, 0.03))
    if name == "crunch":  # a bite of something crisp
        n = int(0.22 * RATE)
        out = np.zeros(n)
        for k in range(5):
            start = int(k * 0.035 * RATE)
            m = n - start
            out[start:] += _noise(m, 900, 7000, rng) * _env(m, 0.0005, 0.012) * (1 - 0.15 * k)
        return 0.32 * out
    if name == "scrub":  # a brush going back and forth
        n = int(0.55 * RATE)
        t = np.arange(n) / RATE
        return 0.13 * _noise(n, 2500, 9000, rng) * (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 3.6 * t))) \
            * np.sin(np.pi * t / t[-1]) ** 0.5
    if name == "splash":  # running water
        n = int(0.9 * RATE)
        t = np.arange(n) / n
        return 0.2 * _noise(n, 400, 6000, rng) * np.sin(np.pi * t) ** 0.6 \
            * (0.7 + 0.3 * np.sin(2 * np.pi * 23 * t * 0.9))
    if name == "tada":  # a happy little fanfare
        n = int(1.0 * RATE)
        out = np.zeros(n)
        for k, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
            start = int(k * 0.08 * RATE)
            t = np.arange(n - start) / RATE
            tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
            out[start:] += tone * _env(n - start, 0.004, 0.45 if k == 3 else 0.12)
        return 0.12 * out
    if name == "scratch":  # a record scratch: the picture freezes
        n = int(0.45 * RATE)
        t = np.arange(n) / RATE
        bend = np.interp(t, (0, 0.12, 0.2, 0.32, 0.45), (0.4, 1.6, 0.6, 1.4, 0.2))
        f = 900 * bend
        tone = np.sin(2 * np.pi * np.cumsum(f) / RATE)
        return 0.32 * (0.6 * _noise(n, 700, 6000, rng) + 0.4 * tone) * _env(n, 0.003, 0.2)
    if name == "rewind":  # a tape winding back, rising
        n = int(0.9 * RATE)
        t = np.arange(n) / RATE
        f = 300 + 1500 * (t / t[-1]) ** 1.5
        wobble = 1 + 0.08 * np.sin(2 * np.pi * 23 * t)
        tone = np.sin(2 * np.pi * np.cumsum(f * wobble) / RATE)
        return 0.16 * (tone + 0.5 * _noise(n, 1500, 7000, rng)) * np.sin(np.pi * t / t[-1]) ** 0.5
    if name == "thud":
        n = int(0.5 * RATE)
        t = np.arange(n) / RATE
        f = 120 - 60 * np.minimum(t / 0.15, 1.0)
        return 0.7 * (np.sin(2 * np.pi * np.cumsum(f) / RATE) * _env(n, 0.002, 0.12)
                      + 0.3 * _noise(n, 80, 1500, rng) * _env(n, 0.001, 0.03))
    raise ValueError(f"unknown sound {name}")


def sound_track(events: Sequence[Tuple[float, str]], seconds: float) -> np.ndarray:
    """All sound effects at their times, as one mono track of ``seconds``."""
    track = np.zeros(int(math.ceil(seconds * RATE)) + 1, np.float32)
    rng = np.random.default_rng(11)
    for at, name in events:
        clip = sound(name, rng)
        start = int(at * RATE)
        end = min(len(track), start + len(clip))
        if 0 <= start < end:
            track[start:end] += clip[:end - start]
    return np.clip(track, -1.0, 1.0)
