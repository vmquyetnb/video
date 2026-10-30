"""Cartoon backdrops drawn by the program: ten settings, three times of day
and a little weather, in flat colours with dark outlines.

A backdrop is a still sky (with the sun, the moon or stars) and a few layers
that scroll at different speeds when the character walks (parallax). Small
things move on their own: clouds drift, birds flap, stars twinkle, fish swim,
bubbles rise, rain and snow fall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

F32 = np.float32
SETTINGS = ("garden", "meadow", "forest", "playground", "beach", "city", "snow", "desert", "mountain", "room",
            "kitchen", "space", "underwater")
INDOORS = ("room", "kitchen", "space", "underwater")
TIMES = ("day", "sunset", "night")
WEATHER = ("clear", "rain", "snow", "leaves")
FLOATING = ("space", "underwater")  # places where floating looks natural

SKY = {"day": ((0.36, 0.62, 0.95), (0.74, 0.89, 1.0)),
       "sunset": ((0.38, 0.33, 0.64), (1.0, 0.68, 0.44)),
       "night": ((0.03, 0.05, 0.17), (0.13, 0.17, 0.36))}
TINT = {"day": (1.0, 1.0, 1.0), "sunset": (1.0, 0.84, 0.74), "night": (0.4, 0.46, 0.7)}
HORIZON = {"day": (0.86, 0.94, 1.0), "sunset": (1.0, 0.8, 0.62), "night": (0.2, 0.25, 0.45)}

Draw = ImageDraw.ImageDraw
Painter = Callable[[np.ndarray, float, float], None]  # (frame, t, scroll)


@dataclass
class Backdrop:
    """Everything behind (and some things in front of) the character."""

    width: int
    height: int
    ground: float  # y of the ground line, where the character's feet stand
    sky: np.ndarray  # float RGB, still
    layers: List[Tuple[np.ndarray, float]] = field(default_factory=list)  # (uint8 RGBA, speed)
    behind: List[Painter] = field(default_factory=list)  # drawn over the sky, under the layers
    front: List[Painter] = field(default_factory=list)  # drawn over the character
    floating: bool = False
    _cache: Optional[Tuple[int, np.ndarray]] = None

    def scenery(self, scroll: float) -> np.ndarray:
        """The sky and the layers for a camera ``scroll`` pixels along (a copy)."""
        key = int(round(scroll))
        if self._cache is None or self._cache[0] != key:
            frame = self.sky.copy()
            for layer, speed in self.layers:
                _blend(frame, _window(layer, key * speed, self.width))
            self._cache = (key, frame)
        return self._cache[1].copy()

    def background(self, t: float, scroll: float) -> np.ndarray:
        """The sky with what moves in it, and the layers."""
        if not self.behind:
            return self.scenery(scroll)
        frame = self.sky.copy()
        for paint in self.behind:
            paint(frame, t, scroll)
        for layer, speed in self.layers:
            _blend(frame, _window(layer, scroll * speed, self.width))
        return frame

    def foreground(self, frame: np.ndarray, t: float, scroll: float) -> None:
        for paint in self.front:
            paint(frame, t, scroll)


def _window(layer: np.ndarray, offset: float, width: int) -> np.ndarray:
    """``width`` columns of a horizontally repeating layer, from ``offset``."""
    lw = layer.shape[1]
    x0 = int(round(offset)) % lw
    if x0 + width <= lw:
        return layer[:, x0:x0 + width]
    return np.concatenate([layer[:, x0:], layer[:, :width - (lw - x0)]], axis=1)


def _blend(frame: np.ndarray, rgba: np.ndarray) -> None:
    a = rgba[..., 3:].astype(F32) / 255.0
    frame *= 1.0 - a
    frame += rgba[..., :3].astype(F32) / 255.0 * a


# --------------------------------------------------------------------------- drawing helpers

S = 2  # layers are drawn at twice the size, then scaled down (smooth edges)


def _c(colour, tint=(1.0, 1.0, 1.0), alpha: float = 1.0) -> Tuple[int, int, int, int]:
    return tuple(int(np.clip(c * k, 0, 1) * 255) for c, k in zip(colour, tint)) + (int(alpha * 255),)


def _dark(colour, k: float = 0.62):
    return tuple(c * k for c in colour)


class Canvas:
    """A layer being drawn: coordinates in final pixels, drawn at ``S`` times."""

    def __init__(self, width: int, height: int, tint):
        self.w, self.h, self.tint = width, height, tint
        self.img = Image.new("RGBA", (width * S, height * S), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.line = max(2, int(round(min(width, height) / 400)))  # outline width

    def poly(self, pts, fill, outline=True, lit=False):
        tint = (1, 1, 1) if lit else self.tint
        pts = [(x * S, y * S) for x, y in pts]
        self.d.polygon(pts, fill=_c(fill, tint))
        if outline:
            self.d.line(pts + [pts[0]], fill=_c(_dark(fill), tint), width=self.line * S, joint="curve")

    def ellipse(self, x0, y0, x1, y1, fill, outline=True, lit=False):
        tint = (1, 1, 1) if lit else self.tint
        self.d.ellipse([x0 * S, y0 * S, x1 * S, y1 * S], fill=_c(fill, tint),
                       outline=_c(_dark(fill), tint) if outline else None, width=self.line * S)

    def rect(self, x0, y0, x1, y1, fill, outline=True, radius=0, lit=False):
        tint = (1, 1, 1) if lit else self.tint
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=radius * S, fill=_c(fill, tint),
                                 outline=_c(_dark(fill), tint) if outline else None, width=self.line * S)

    def lines(self, pts, colour, width):
        self.d.line([(x * S, y * S) for x, y in pts], fill=_c(colour, self.tint), width=int(width * S),
                    joint="curve")

    def done(self) -> np.ndarray:
        return np.asarray(self.img.resize((self.w, self.h), Image.LANCZOS))


def _wave(length: int, base: float, amp: float, seed: int, terms=(1, 2, 3, 5)) -> np.ndarray:
    """A hill line that repeats every ``length`` pixels: y for each x (step 6 px)."""
    rng = np.random.default_rng(seed)
    xs = np.arange(0, length + 6, 6, dtype=F32)
    ys = np.full_like(xs, base)
    for k in terms:
        ys += amp / k ** 0.8 * rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * k * xs / length + rng.uniform(0, 6.3))
    return np.stack([xs, ys], axis=1)


def _hills(cv: Canvas, base: float, amp: float, colour, seed: int, terms=(1, 2, 3, 5)) -> np.ndarray:
    line = _wave(cv.w, base, amp, seed, terms)
    cv.poly([tuple(p) for p in line] + [(cv.w, cv.h + 5), (0, cv.h + 5)], colour)
    return line


def _top(line: np.ndarray, x: float) -> float:
    return float(np.interp(x % line[-1, 0], line[:, 0], line[:, 1]))


def _both(cv: Canvas, x: float, margin: float, fn) -> None:
    """Draw ``fn(x)`` and again one layer-width away when it crosses an edge (seamless repeat)."""
    fn(x)
    if x - margin < 0:
        fn(x + cv.w)
    if x + margin > cv.w:
        fn(x - cv.w)


def _round_tree(cv: Canvas, x, ground, size, leaf=(0.32, 0.68, 0.3), seed=0):
    rng = np.random.default_rng(abs(int(seed)))
    cv.rect(x - 0.07 * size, ground - 0.55 * size, x + 0.07 * size, ground + 0.02 * size, (0.5, 0.33, 0.2))
    for dx, dy, r in ((0, -0.95, 0.42), (-0.3, -0.7, 0.32), (0.3, -0.72, 0.33)):
        r *= size * rng.uniform(0.9, 1.1)
        cx, cy = x + dx * size, ground + dy * size
        cv.ellipse(cx - r, cy - r, cx + r, cy + r, leaf)


def _pine(cv: Canvas, x, ground, size, colour=(0.18, 0.5, 0.33), snow=False):
    cv.rect(x - 0.06 * size, ground - 0.2 * size, x + 0.06 * size, ground + 0.02 * size, (0.45, 0.3, 0.2))
    for k, (w, y0, y1) in enumerate(((0.42, -0.15, -0.6), (0.34, -0.45, -0.85), (0.25, -0.72, -1.08))):
        pts = [(x - w * size, ground + y0 * size), (x + w * size, ground + y0 * size), (x, ground + y1 * size)]
        cv.poly(pts, colour)
        if snow:
            cv.poly([(x - 0.35 * w * size, ground + (y1 + 0.55 * (y0 - y1)) * size),
                     (x + 0.35 * w * size, ground + (y1 + 0.55 * (y0 - y1)) * size), (x, ground + y1 * size)],
                    (0.97, 0.98, 1.0), outline=False)


def _house(cv: Canvas, x, ground, w, h, wall, roof, lit):
    cv.rect(x, ground - h, x + w, ground, wall)
    cv.poly([(x - 0.08 * w, ground - h), (x + w / 2, ground - h - 0.55 * w), (x + 1.08 * w, ground - h)], roof)
    glow = (1.0, 0.86, 0.4) if lit else (0.62, 0.82, 0.95)
    for i in range(2):
        wx = x + (0.18 + 0.44 * i) * w
        cv.rect(wx, ground - 0.78 * h, wx + 0.22 * w, ground - 0.5 * h, glow, lit=lit)
    cv.rect(x + 0.4 * w, ground - 0.38 * h, x + 0.6 * w, ground, (0.55, 0.35, 0.22), radius=0.05 * w)


def _flowers(cv: Canvas, ground, depth, count, seed):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        x, y = rng.uniform(0, cv.w), ground + rng.uniform(0.15, 1.0) * depth
        r = rng.uniform(0.004, 0.007) * cv.h
        colour = [(1.0, 0.45, 0.55), (1.0, 0.85, 0.3), (0.95, 0.95, 1.0), (0.7, 0.5, 1.0)][rng.integers(4)]
        _both(cv, x, r, lambda xx: cv.ellipse(xx - r, y - r, xx + r, y + r, colour, outline=False))


def _sprite_layer(w: int, h: int, draw: Callable[[Canvas], None], tint=(1.0, 1.0, 1.0)) -> np.ndarray:
    cv = Canvas(w, h, tint)
    draw(cv)
    return cv.done().astype(F32) / 255.0


def _paste(frame: np.ndarray, rgba: np.ndarray, x: float, y: float, opacity: float = 1.0) -> None:
    """Paste a float RGBA sprite with its top-left at (x, y), clipped to the frame."""
    h, w = rgba.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    part = rgba[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = part[..., 3:] * opacity
    region = frame[fy0:fy1, fx0:fx1]
    region *= 1.0 - a
    region += part[..., :3] * a


# --------------------------------------------------------------------------- sky

def _sky(w: int, h: int, time: str, inside: bool = False, overcast: bool = False) -> np.ndarray:
    top, bottom = (np.asarray(c, F32) for c in SKY[time])
    k = np.linspace(0.0, 1.0, h, dtype=F32)[:, None, None] ** 0.9
    sky = np.broadcast_to(top + (bottom - top) * k, (h, w, 3)).copy()
    if inside:
        return sky
    u = min(w, h)
    band = np.exp(-((np.linspace(0, 1, h, dtype=F32) - 0.8) / 0.18) ** 2)[:, None, None]  # glow above the horizon
    sky += (np.asarray(HORIZON[time], F32) - sky) * band * 0.55
    if overcast:  # a rainy sky: grey all over, no sun or moon
        grey = sky.mean(axis=2, keepdims=True) * np.asarray((0.94, 0.97, 1.04), F32)
        return sky * 0.3 + grey * 0.7
    if time == "day":
        cx, cy, r = 0.82 * w, 0.17 * h, 0.07 * u
        glow = np.exp(-((np.arange(w)[None, :] - cx) ** 2 + (np.arange(h)[:, None] - cy) ** 2) / (2 * (2.6 * r) ** 2))
        sky += (1.0 - sky) * glow[..., None].astype(F32) * 0.45
        _rays(sky, cx, cy, 0.9 * u, (1.0, 1.0, 0.92), 0.16)
        sun = _sprite_layer(int(2 * r) + 8, int(2 * r) + 8,
                            lambda cv: cv.ellipse(4, 4, 4 + 2 * r, 4 + 2 * r, (1.0, 0.86, 0.3)))
        _paste(sky, sun, cx - r - 4, cy - r - 4)
    elif time == "sunset":
        cx, cy, r = 0.7 * w, 0.62 * h, 0.12 * u
        glow = np.exp(-((np.arange(w)[None, :] - cx) ** 2 + (np.arange(h)[:, None] - cy) ** 2) / (2 * (3 * r) ** 2))
        sky += (np.asarray((1.0, 0.75, 0.45), F32) - sky) * glow[..., None].astype(F32) * 0.55
        _rays(sky, cx, cy, 1.1 * u, (1.0, 0.85, 0.6), 0.2, 18)
        sun = _sprite_layer(int(2 * r) + 8, int(2 * r) + 8,
                            lambda cv: cv.ellipse(4, 4, 4 + 2 * r, 4 + 2 * r, (1.0, 0.55, 0.25), outline=False))
        _paste(sky, sun, cx - r - 4, cy - r - 4)
    else:
        rng = np.random.default_rng(4)
        for _ in range(int(0.00012 * w * h)):
            x, y = rng.uniform(0, w), rng.uniform(0, 0.75 * h)
            cv2.circle(sky, (int(x), int(y)), int(rng.choice([1, 1, 1, 2])), (0.95, 0.95, 0.85), -1, cv2.LINE_AA)
        cx, cy, r = 0.8 * w, 0.18 * h, 0.06 * u
        glow = np.exp(-((np.arange(w)[None, :] - cx) ** 2 + (np.arange(h)[:, None] - cy) ** 2) / (2 * (3.5 * r) ** 2))
        sky += (np.asarray((0.5, 0.58, 0.85), F32) - sky) * glow[..., None].astype(F32) * 0.4
        moon = _sprite_layer(int(2 * r) + 8, int(2 * r) + 8,
                             lambda cv: cv.ellipse(4, 4, 4 + 2 * r, 4 + 2 * r, (0.98, 0.95, 0.8), outline=False))
        dent = cv2.circle(np.zeros(moon.shape[:2], F32), (int(4 + 1.45 * r), int(4 + 0.7 * r)), int(0.95 * r), 1.0, -1)
        moon[..., 3] *= 1.0 - dent
        _paste(sky, moon, cx - r - 4, cy - r - 4)
    return sky


# --------------------------------------------------------------------------- moving things

def _clouds(w: int, h: int, time: str, count: int = 4, seed: int = 1, rain: bool = False) -> Painter:
    rng = np.random.default_rng(seed)
    u = min(w, h)
    colour = {"day": (1.0, 1.0, 1.0), "sunset": (1.0, 0.86, 0.8), "night": (0.45, 0.5, 0.66)}[time]
    if rain:
        colour = tuple(c * 0.72 for c in colour)
    clouds = []
    for i in range(count):
        cw = rng.uniform(0.16, 0.26) * u * 1.6
        ch = cw * 0.5

        def draw(cv: Canvas, cw=cw, ch=ch):
            lobes = ((0.3, 0.62, 0.26), (0.52, 0.45, 0.3), (0.72, 0.62, 0.22), (0.5, 0.7, 0.24))
            under = tuple(c * 0.86 + 0.1 for c in colour)
            for dx, dy, r in lobes:  # a shaded underside, then the lit tops
                cv.ellipse((dx - r) * cw, (dy - r * 1.4) * ch, (dx + r) * cw, (dy + r * 1.4) * ch, under,
                           outline=False)
            for dx, dy, r in lobes:
                r2 = r * 0.9
                cv.ellipse((dx - r2 - 0.02) * cw, (dy - 0.08 - r2 * 1.4) * ch, (dx + r2 - 0.02) * cw,
                           (dy - 0.08 + r2 * 1.4) * ch, colour, outline=False)

        clouds.append((_sprite_layer(int(cw), int(ch), draw), rng.uniform(0, w), rng.uniform(0.04, 0.3) * h,
                       rng.uniform(4, 12) * u / 1000))

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        for sprite, x0, y, speed in clouds:
            cw = sprite.shape[1]
            x = (x0 + speed * t * 30 - scroll * 0.05) % (w + cw) - cw
            _paste(frame, sprite, x, y, 0.95)

    return paint


def _birds(w: int, h: int, count: int = 3, seed: int = 2) -> Painter:
    rng = np.random.default_rng(seed)
    u = min(w, h)
    birds = [(rng.uniform(0, w), rng.uniform(0.08, 0.32) * h, rng.uniform(0.012, 0.02) * u,
              rng.uniform(0.04, 0.07) * w, rng.uniform(0, 6)) for _ in range(count)]

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        for x0, y0, size, speed, phase in birds:
            x = (x0 + speed * t - scroll * 0.1) % (w + 4 * size) - 2 * size
            y = y0 + 0.2 * size * math.sin(t * 1.3 + phase)
            flap = math.sin(t * 9 + phase) * 0.6
            pts = np.array([(x - size, y - size * flap), (x, y + 0.2 * size), (x + size, y - size * flap)],
                           np.int32)
            cv2.polylines(frame, [pts], False, (0.15, 0.15, 0.2), max(2, int(size / 5)), cv2.LINE_AA)

    return paint


def _twinkle(w: int, h: int, count: int, top: float = 0.7, seed: int = 3) -> Painter:
    rng = np.random.default_rng(seed)
    stars = [(rng.uniform(0, w), rng.uniform(0, top * h), rng.uniform(0, 6), rng.uniform(1.5, 3.5))
             for _ in range(count)]
    size = max(2, int(min(w, h) / 300))

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        for x, y, phase, speed in stars:
            k = 0.5 + 0.5 * math.sin(t * speed + phase)
            if k > 0.55:
                r = int(size * (0.6 + k))
                x2 = int((x - scroll * 0.02) % w)
                cv2.line(frame, (x2 - r, int(y)), (x2 + r, int(y)), (1.0, 1.0, 0.85), 1, cv2.LINE_AA)
                cv2.line(frame, (x2, int(y) - r), (x2, int(y) + r), (1.0, 1.0, 0.85), 1, cv2.LINE_AA)
                cv2.circle(frame, (x2, int(y)), max(1, r // 2), (1.0, 1.0, 0.9), -1, cv2.LINE_AA)

    return paint


def _fireflies(w: int, h: int, ground: float, count: int = 14, seed: int = 5) -> Painter:
    rng = np.random.default_rng(seed)
    flies = [(rng.uniform(0, w), rng.uniform(0.5 * ground, ground), rng.uniform(0, 6)) for _ in range(count)]
    r = max(2, int(min(w, h) / 220))

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        for x0, y0, p in flies:
            x = int((x0 + 30 * math.sin(t * 0.7 + p) - scroll * 0.9) % w)
            y = int(min(max(y0 + 20 * math.sin(t * 1.1 + 2 * p), 0), frame.shape[0] - 1))
            k = 0.5 + 0.5 * math.sin(t * 3 + p)  # blinking
            if k < 0.3:
                continue
            glow = frame[y, x] * (1 - 0.45 * k) + np.asarray((0.85, 1.0, 0.45), F32) * 0.45 * k
            cv2.circle(frame, (x, y), 3 * r, tuple(float(c) for c in glow), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), r, (1.0, 1.0, 0.75), -1, cv2.LINE_AA)

    return paint


def _bubbles(w: int, h: int, count: int = 18, seed: int = 6) -> Painter:
    rng = np.random.default_rng(seed)
    items = [(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.004, 0.012) * min(w, h),
              rng.uniform(0.05, 0.12) * h, rng.uniform(0, 6)) for _ in range(count)]

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        for x0, y0, r, speed, p in items:
            y = (y0 - speed * t) % (h + 4 * r) - 2 * r
            x = (x0 + 8 * math.sin(t * 2 + p) - scroll * 0.6) % w
            cv2.circle(frame, (int(x), int(y)), int(r), (0.85, 0.95, 1.0), max(1, int(r / 4)), cv2.LINE_AA)
            cv2.circle(frame, (int(x - r / 3), int(y - r / 3)), max(1, int(r / 4)), (1, 1, 1), -1, cv2.LINE_AA)

    return paint


def _fish(w: int, h: int, ground: float, count: int = 4, seed: int = 7) -> Painter:
    rng = np.random.default_rng(seed)
    u = min(w, h)
    school = []
    for _ in range(count):
        size = rng.uniform(0.03, 0.05) * u
        colour = [(1.0, 0.55, 0.2), (1.0, 0.85, 0.3), (0.95, 0.4, 0.5), (0.4, 0.8, 0.9)][rng.integers(4)]

        def draw(cv: Canvas, size=size, colour=colour):
            cv.ellipse(0.25 * size, 0.1 * size, 1.45 * size, 0.9 * size, colour)
            cv.poly([(0.32 * size, 0.5 * size), (0.02 * size, 0.15 * size), (0.02 * size, 0.85 * size)], colour)
            cv.ellipse(1.1 * size, 0.35 * size, 1.24 * size, 0.49 * size, (0.1, 0.1, 0.1), outline=False)

        sprite = _sprite_layer(int(1.5 * size), int(size), draw)
        school.append((sprite, rng.uniform(0, w), rng.uniform(0.2 * ground, 0.8 * ground),
                       rng.uniform(0.03, 0.07) * w, rng.uniform(0, 6)))

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        for sprite, x0, y0, speed, p in school:
            sw = sprite.shape[1]
            x = (x0 + speed * t - scroll * 0.7) % (w + 2 * sw) - sw
            _paste(frame, sprite, x, y0 + 0.02 * h * math.sin(t + p))

    return paint


def _falling(w: int, h: int, kind: str, seed: int = 8) -> Painter:
    """Rain, snow or leaves in front of everything."""
    rng = np.random.default_rng(seed)
    u = min(w, h)
    count = {"rain": 140, "snow": 110, "leaves": 26}[kind]
    items = [(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.6, 1.0), rng.uniform(0, 6)) for _ in range(count)]
    leaf = None
    if kind == "leaves":
        size = 0.022 * u
        leaf = [_sprite_layer(int(size), int(size * 0.6),
                              lambda cv, c=c: cv.ellipse(1, 1, size - 1, size * 0.6 - 1, c))
                for c in ((0.92, 0.55, 0.15), (0.85, 0.3, 0.15), (0.95, 0.75, 0.2))]

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        for i, (x0, y0, depth, p) in enumerate(items):
            if kind == "rain":
                speed = 1.6 * h * depth
                y = (y0 + speed * t) % (h + 40) - 20
                x = (x0 - 0.1 * speed * t - scroll * depth) % w
                cv2.line(frame, (int(x), int(y)), (int(x - 0.006 * u), int(y + 0.035 * u * depth)),
                         (0.75, 0.82, 0.95), max(1, int(u / 900)), cv2.LINE_AA)
            elif kind == "snow":
                y = (y0 + 0.12 * h * depth * t) % (h + 20) - 10
                x = (x0 + 0.015 * w * math.sin(t * 0.8 + p) - scroll * depth) % w
                cv2.circle(frame, (int(x), int(y)), max(2, int(0.005 * u * depth)), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
            else:
                y = (y0 + 0.08 * h * depth * t) % (h + 40) - 20
                x = (x0 + 0.04 * w * math.sin(t * 0.9 + p) - scroll * depth) % w
                sprite = leaf[i % len(leaf)]
                _paste(frame, sprite, x, y)

    return paint


# --------------------------------------------------------------------------- the settings

def _outdoor_ground(cv: Canvas, ground: float, colour, seed: int, flowers: int = 0) -> None:
    line = _wave(cv.w, ground - 0.01 * cv.h, 0.006 * cv.h, seed, (2, 5))
    cv.poly([tuple(p) for p in line] + [(cv.w, cv.h + 5), (0, cv.h + 5)], colour)
    rng = np.random.default_rng(seed)
    for _ in range(int(cv.w / 60)):  # tufts / texture
        x, y = rng.uniform(0, cv.w), ground + rng.uniform(0.03, 1.0) * (cv.h - ground)
        r = 0.006 * cv.h
        _both(cv, x, 3 * r, lambda xx: cv.lines([(xx - 2 * r, y), (xx, y - 1.6 * r), (xx + 2 * r, y)],
                                                 _dark(colour, 0.82), max(1.5, r / 2)))
    if flowers:
        _flowers(cv, ground, cv.h - ground, flowers, seed + 1)


def build(setting: str, time: str, weather: str, width: int, height: int, seed: int = 0) -> Backdrop:
    """The backdrop of ``setting`` at ``time`` with ``weather``."""
    if setting not in SETTINGS:
        setting = "meadow"
    time = time if time in TIMES else "day"
    W, H = width, height
    u = min(W, H)
    L = 2 * W  # every layer repeats every two screen widths
    ground = 0.84 * H
    outdoor = setting not in INDOORS
    rain = weather == "rain" and outdoor
    tint = TINT[time] if not rain else tuple(c * k for c, k in zip(TINT[time], (0.84, 0.87, 0.93)))
    night = time == "night"
    sky = _sky(W, H, time, inside=setting in INDOORS, overcast=rain)
    bd = Backdrop(W, H, ground, sky, floating=setting in FLOATING)
    if outdoor and not night:
        bd.behind.append(_clouds(W, H, time, count=7 if rain else 4, seed=seed + 1, rain=rain))
    if outdoor and night:
        bd.behind.append(_twinkle(W, H, 30, seed=seed + 2))

    def birds(count: int) -> None:
        if not rain:
            bd.behind.append(_birds(W, H, count, seed=seed + 3))

    def layer(draw: Callable[[Canvas], None], speed: float, lit_tint=None) -> None:
        cv = Canvas(L, H, lit_tint or tint)
        draw(cv)
        depth = 0.0 if speed >= 0.7 else (1.0 if speed <= 0.2 else 0.4)
        rgba = _finish(cv.done(), u, time, depth, HORIZON[time], outdoor)
        if speed >= 1.0:
            _shade_low(rgba, ground, H)
        bd.layers.append((rgba, speed))

    if setting == "garden":
        _garden(layer, W, u, ground, seed, night)
        if night:
            bd.front.append(_fireflies(W, H, ground, seed=seed + 4))
        else:
            birds(2)
    elif setting == "playground":
        _playground(layer, u, ground, seed)
        if not night:
            birds(2)
    elif setting == "kitchen":
        _kitchen(layer, u, ground, time)

    elif setting == "meadow":
        layer(lambda cv: _hills(cv, ground - 0.24 * u, 0.06 * u, (0.55, 0.78, 0.5), seed), 0.15)

        def near(cv):
            line = _hills(cv, ground - 0.1 * u, 0.04 * u, (0.42, 0.72, 0.36), seed + 1, (1, 2, 4))
            for x in np.linspace(0, cv.w, 5, endpoint=False) + 0.1 * cv.w:
                _both(cv, x, 0.3 * u, lambda xx: _round_tree(cv, xx, _top(line, xx) + 0.02 * u, 0.32 * u,
                                                             seed=int(x)))
        layer(near, 0.45)
        layer(lambda cv: _outdoor_ground(cv, ground, (0.38, 0.7, 0.3), seed, flowers=60), 1.0)
        if not night:
            birds(3)
        else:
            bd.front.append(_fireflies(W, H, ground, seed=seed + 4))
    elif setting == "forest":
        def far(cv):  # a distant tree line: a bumpy dark green edge
            _hills(cv, ground - 0.2 * u, 0.03 * u, (0.25, 0.48, 0.33), seed, (1, 2))
            rng = np.random.default_rng(seed)
            for x in np.arange(0, cv.w, 0.07 * u):
                r = rng.uniform(0.06, 0.1) * u
                y = ground - 0.2 * u - 0.6 * r
                _both(cv, x, r, lambda xx: cv.ellipse(xx - r, y - r, xx + r, y + r, (0.25, 0.48, 0.33),
                                                       outline=False))
        layer(far, 0.15)

        def mid(cv):
            rng = np.random.default_rng(seed)
            for k, x in enumerate(np.arange(0.05 * cv.w, cv.w, 0.3 * u)):
                x += rng.uniform(-0.05, 0.05) * u
                if k % 2:
                    _both(cv, x, 0.25 * u, lambda xx: _pine(cv, xx, ground - 0.02 * u, 0.55 * u, (0.2, 0.5, 0.3)))
                else:
                    _both(cv, x, 0.3 * u, lambda xx: _round_tree(cv, xx, ground - 0.02 * u, 0.5 * u,
                                                                 (0.3, 0.6, 0.28), seed=int(x)))
        layer(mid, 0.5)

        def floor(cv):
            _outdoor_ground(cv, ground, (0.33, 0.55, 0.25), seed)
            cv.poly([(0, ground + 0.05 * u), (cv.w, ground + 0.05 * u), (cv.w, ground + 0.12 * u),
                     (0, ground + 0.12 * u)], (0.66, 0.5, 0.33), outline=False)
            rng = np.random.default_rng(seed + 9)
            for _ in range(7):
                x, s = rng.uniform(0, cv.w), rng.uniform(0.03, 0.05) * u
                y = ground + rng.uniform(0.0, 0.03) * u

                def mush(xx, y=y, s=s):
                    cv.rect(xx - 0.25 * s, y - 0.8 * s, xx + 0.25 * s, y, (0.96, 0.93, 0.85))
                    cv.d.chord([(xx - s) * S, (y - 1.6 * s) * S, (xx + s) * S, (y - 0.3 * s) * S], 180, 360,
                               fill=_c((0.88, 0.2, 0.18), cv.tint), outline=_c((0.5, 0.1, 0.1), cv.tint),
                               width=cv.line * S)
                    for dx in (-0.45, 0.1, 0.5):
                        r = 0.13 * s
                        cv.ellipse(xx + dx * s - r, y - 1.2 * s - r, xx + dx * s + r, y - 1.2 * s + r,
                                   (1, 1, 1), outline=False)
                _both(cv, x, s, mush)
        layer(floor, 1.0)
        if night:
            bd.front.append(_fireflies(W, H, ground, seed=seed + 4))
    elif setting == "beach":
        def sea(cv):
            horizon = ground - 0.2 * u
            cv.poly([(0, horizon), (cv.w, horizon), (cv.w, ground + 5), (0, ground + 5)], (0.2, 0.55, 0.85))
            rng = np.random.default_rng(seed)
            for _ in range(int(cv.w / 50)):
                x, y = rng.uniform(0, cv.w), horizon + rng.uniform(0.05, 0.95) * (ground - horizon)
                r = 0.012 * u * (y - horizon) / (ground - horizon) + 3
                _both(cv, x, r, lambda xx: cv.lines([(xx - r, y), (xx, y - r * 0.4), (xx + r, y)],
                                                     (0.75, 0.9, 1.0), max(1.5, r / 6)))
        layer(sea, 0.1)

        def sand(cv):
            line = _wave(cv.w, ground - 0.015 * u, 0.008 * u, seed, (2, 3))
            cv.poly([tuple(p) for p in line] + [(cv.w, cv.h + 5), (0, cv.h + 5)], (0.98, 0.86, 0.6))
            cv.lines([tuple(p) for p in line], (1, 1, 1), 0.006 * u)
            for x in (0.15 * cv.w, 0.62 * cv.w):
                def palm(xx):
                    top = (xx + 0.08 * u, ground - 0.55 * u)
                    cv.lines([(xx, ground), (xx + 0.03 * u, ground - 0.3 * u), top], (0.55, 0.38, 0.22), 0.035 * u)
                    for ang in (-160, -120, -60, -20, 30, 200):
                        a = math.radians(ang)
                        tip = (top[0] + math.cos(a) * 0.2 * u, top[1] + math.sin(a) * 0.12 * u + 0.05 * u)
                        mid = ((top[0] + tip[0]) / 2, min(top[1], tip[1]) - 0.04 * u)
                        cv.poly([top, mid, tip, ((top[0] + tip[0]) / 2, (top[1] + tip[1]) / 2 + 0.01 * u)],
                                (0.2, 0.62, 0.28))
                _both(cv, x, 0.3 * u, palm)
            rng = np.random.default_rng(seed + 5)
            for _ in range(10):
                x, y, r = rng.uniform(0, cv.w), ground + rng.uniform(0.05, 0.9) * (cv.h - ground), 0.01 * u
                _both(cv, x, r, lambda xx: cv.ellipse(xx - r, y - 0.6 * r, xx + r, y + 0.6 * r, (1.0, 0.7, 0.65)))
        layer(sand, 1.0)
        if not night:
            birds(2)
    elif setting == "city":
        def skyline(cv):
            rng = np.random.default_rng(seed)
            x = 0.0
            while x < cv.w:
                bw, bh = rng.uniform(0.08, 0.16) * u, rng.uniform(0.25, 0.55) * u
                cv.rect(x, ground - 0.05 * u - bh, x + bw, ground, (0.62, 0.68, 0.8))
                if night:
                    for wy in np.arange(ground - bh, ground - 0.1 * u, 0.05 * u):
                        for wx in np.arange(x + 0.02 * u, x + bw - 0.03 * u, 0.04 * u):
                            if rng.random() < 0.4:
                                cv.rect(wx, wy, wx + 0.015 * u, wy + 0.022 * u, (1.0, 0.85, 0.4), False, lit=True)
                x += bw + 0.01 * u
        layer(skyline, 0.2)

        def street(cv):
            rng = np.random.default_rng(seed + 1)
            x = 0.0
            colours = [(0.95, 0.6, 0.5), (0.98, 0.85, 0.55), (0.6, 0.8, 0.9), (0.75, 0.9, 0.65), (0.9, 0.7, 0.9)]
            while x < cv.w - 0.25 * u:
                hw, hh = rng.uniform(0.2, 0.27) * u, rng.uniform(0.2, 0.3) * u
                _house(cv, x + 0.03 * u, ground - 0.02 * u, hw, hh, colours[rng.integers(5)],
                       (0.75, 0.28, 0.22), night)
                x += hw + 0.08 * u
            cv.rect(-5, ground - 0.02 * u, cv.w + 5, cv.h + 5, (0.7, 0.7, 0.72), outline=False)
            cv.rect(-5, ground + 0.07 * u, cv.w + 5, cv.h + 5, (0.38, 0.38, 0.42), outline=False)
            for lx in np.arange(0.1 * cv.w, cv.w, 0.5 * u):
                def lamp(xx):
                    cv.rect(xx - 0.006 * u, ground - 0.32 * u, xx + 0.006 * u, ground + 0.02 * u, (0.25, 0.25, 0.3))
                    cv.ellipse(xx - 0.025 * u, ground - 0.36 * u, xx + 0.025 * u, ground - 0.31 * u,
                               (1.0, 0.9, 0.5) if night else (0.9, 0.9, 0.85), lit=night)
                _both(cv, lx, 0.05 * u, lamp)
        layer(street, 1.0)
        if not night:
            birds(2)
    elif setting == "snow":
        layer(lambda cv: _hills(cv, ground - 0.22 * u, 0.06 * u, (0.85, 0.9, 0.97), seed), 0.15)

        def pines(cv):
            line = _hills(cv, ground - 0.08 * u, 0.03 * u, (0.93, 0.96, 1.0), seed + 1)
            for x in np.arange(0.03 * cv.w, cv.w, 0.3 * u):
                _both(cv, x, 0.25 * u, lambda xx: _pine(cv, xx, _top(line, xx) + 0.02 * u, 0.4 * u, snow=True))
        layer(pines, 0.45)
        layer(lambda cv: _outdoor_ground(cv, ground, (0.97, 0.98, 1.0), seed), 1.0)
        if weather == "clear":
            weather = "snow"
    elif setting == "desert":
        layer(lambda cv: _hills(cv, ground - 0.2 * u, 0.07 * u, (0.98, 0.78, 0.5), seed, (1, 2)), 0.15)

        def dunes(cv):
            _hills(cv, ground - 0.06 * u, 0.035 * u, (0.97, 0.7, 0.4), seed + 1, (1, 2, 3))
            for x in (0.2 * cv.w, 0.7 * cv.w):
                def cactus(xx):
                    g = ground + 0.01 * u
                    cv.rect(xx - 0.035 * u, g - 0.36 * u, xx + 0.035 * u, g, (0.35, 0.65, 0.35), radius=0.035 * u)
                    cv.rect(xx - 0.12 * u, g - 0.26 * u, xx - 0.06 * u, g - 0.13 * u, (0.35, 0.65, 0.35),
                            radius=0.03 * u)
                    cv.rect(xx - 0.12 * u, g - 0.16 * u, xx - 0.02 * u, g - 0.12 * u, (0.35, 0.65, 0.35), False)
                    cv.rect(xx + 0.06 * u, g - 0.3 * u, xx + 0.12 * u, g - 0.18 * u, (0.35, 0.65, 0.35),
                            radius=0.03 * u)
                    cv.rect(xx + 0.02 * u, g - 0.22 * u, xx + 0.12 * u, g - 0.18 * u, (0.35, 0.65, 0.35), False)
                _both(cv, x, 0.15 * u, cactus)
        layer(dunes, 0.5)
        layer(lambda cv: _outdoor_ground(cv, ground, (0.96, 0.82, 0.55), seed), 1.0)
    elif setting == "mountain":
        def peaks(cv):
            rng = np.random.default_rng(seed)
            for x in np.arange(0, cv.w, 0.32 * u):
                hx, hh = x + rng.uniform(-0.05, 0.05) * u, rng.uniform(0.35, 0.5) * u

                def peak(xx, hh=hh):
                    base = ground - 0.1 * u
                    cv.poly([(xx - 0.3 * u, base), (xx, base - hh), (xx + 0.3 * u, base)], (0.55, 0.6, 0.75))
                    cv.poly([(xx - 0.09 * u, base - 0.7 * hh), (xx, base - hh), (xx + 0.09 * u, base - 0.7 * hh),
                             (xx + 0.03 * u, base - 0.66 * hh), (xx - 0.03 * u, base - 0.72 * hh)],
                            (0.97, 0.98, 1.0), outline=False)
                _both(cv, hx, 0.35 * u, peak)
        layer(peaks, 0.1)

        def hills(cv):
            line = _hills(cv, ground - 0.1 * u, 0.04 * u, (0.45, 0.7, 0.42), seed + 2)
            for x in np.arange(0.05 * cv.w, cv.w, 0.4 * u):
                _both(cv, x, 0.2 * u, lambda xx: _pine(cv, xx, _top(line, xx) + 0.02 * u, 0.3 * u))
        layer(hills, 0.4)
        layer(lambda cv: _outdoor_ground(cv, ground, (0.5, 0.68, 0.38), seed, flowers=20), 1.0)
        if not night:
            birds(3)
    elif setting == "room":
        def room(cv):
            wall = (0.98, 0.9, 0.76)
            cv.rect(-5, -5, cv.w + 5, ground, wall, outline=False)
            for x in np.arange(0, cv.w, 0.08 * u):
                cv.rect(x, 0, x + 0.035 * u, ground, (0.96, 0.85, 0.7), outline=False)
            cv.rect(-5, ground - 0.02 * u, cv.w + 5, cv.h + 5, (0.75, 0.53, 0.33), outline=False)
            for y in np.arange(ground, cv.h, 0.05 * u):
                cv.lines([(0, y), (cv.w, y)], (0.62, 0.42, 0.25), 0.003 * u)
            sky_top, sky_bottom = SKY[time]
            for wx in (0.18 * cv.w, 0.68 * cv.w):
                def window(xx):
                    x0, y0, ww, wh = xx, 0.16 * cv.h, 0.3 * u, 0.3 * u
                    cv.rect(x0 - 0.015 * u, y0 - 0.015 * u, x0 + ww + 0.015 * u, y0 + wh + 0.015 * u,
                            (0.98, 0.98, 0.98))
                    cv.rect(x0, y0, x0 + ww, y0 + wh, sky_bottom if time != "night" else sky_top, lit=True)
                    if time == "night":
                        cv.ellipse(x0 + 0.62 * ww, y0 + 0.15 * wh, x0 + 0.82 * ww, y0 + 0.35 * wh, (0.98, 0.95, 0.8),
                                   outline=False, lit=True)
                    cv.lines([(x0 + ww / 2, y0), (x0 + ww / 2, y0 + wh)], (0.98, 0.98, 0.98), 0.012 * u)
                    cv.lines([(x0, y0 + wh / 2), (x0 + ww, y0 + wh / 2)], (0.98, 0.98, 0.98), 0.012 * u)
                    cv.rect(x0 - 0.04 * u, y0 - 0.03 * u, x0 + 0.05 * u, y0 + wh + 0.05 * u, (0.85, 0.35, 0.35))
                    cv.rect(x0 + ww - 0.05 * u, y0 - 0.03 * u, x0 + ww + 0.04 * u, y0 + wh + 0.05 * u,
                            (0.85, 0.35, 0.35))
                _both(cv, wx, 0.4 * u, window)
            for bx in (0.45 * cv.w,):
                def bed(xx):
                    g = ground + 0.02 * u
                    cv.rect(xx, g - 0.16 * u, xx + 0.5 * u, g - 0.04 * u, (0.4, 0.55, 0.85), radius=0.02 * u)
                    cv.rect(xx + 0.02 * u, g - 0.2 * u, xx + 0.13 * u, g - 0.13 * u, (1, 1, 1), radius=0.02 * u)
                    cv.rect(xx - 0.02 * u, g - 0.28 * u, xx + 0.02 * u, g, (0.6, 0.4, 0.25))
                    cv.rect(xx + 0.48 * u, g - 0.2 * u, xx + 0.52 * u, g, (0.6, 0.4, 0.25))
                _both(cv, bx, 0.6 * u, bed)
            for px in (0.05 * cv.w, 0.95 * cv.w):
                def lamp(xx):
                    g = ground + 0.02 * u
                    cv.rect(xx - 0.05 * u, g - 0.16 * u, xx + 0.05 * u, g, (0.7, 0.48, 0.3))
                    cv.rect(xx - 0.008 * u, g - 0.3 * u, xx + 0.008 * u, g - 0.16 * u, (0.4, 0.4, 0.4))
                    cv.poly([(xx - 0.05 * u, g - 0.3 * u), (xx + 0.05 * u, g - 0.3 * u), (xx + 0.035 * u, g - 0.38 * u),
                             (xx - 0.035 * u, g - 0.38 * u)], (1.0, 0.85, 0.45), lit=night)
                _both(cv, px, 0.1 * u, lamp)
            cv.ellipse(0.3 * cv.w, ground + 0.05 * u, 0.3 * cv.w + 0.6 * u, ground + 0.13 * u, (0.85, 0.4, 0.45))
        tint_in = {"day": (1, 1, 1), "sunset": (1.0, 0.9, 0.82), "night": (0.62, 0.62, 0.8)}[time]
        layer(room, 0.25, tint_in)
    elif setting == "space":
        bd.sky = sky = _sky(W, H, "night", inside=True) * 0.6
        rng = np.random.default_rng(seed)
        for _ in range(int(0.0002 * W * H)):
            cv2.circle(sky, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.choice([1, 1, 2])),
                       (0.95, 0.95, 0.9), -1, cv2.LINE_AA)
        bd.behind.append(_twinkle(W, H, 40, top=0.9, seed=seed + 2))

        def planets(cv):
            for x, y, r, colour in ((0.25 * cv.w, 0.25 * cv.h, 0.09 * u, (0.95, 0.6, 0.35)),
                                    (0.75 * cv.w, 0.35 * cv.h, 0.06 * u, (0.5, 0.75, 0.95))):
                def planet(xx, y=y, r=r, colour=colour):
                    cv.d.ellipse([(xx - 1.9 * r) * S, (y - 0.45 * r) * S, (xx + 1.9 * r) * S, (y + 0.45 * r) * S],
                                 outline=_c((0.95, 0.85, 0.6)), width=int(0.12 * r) * S)
                    cv.ellipse(xx - r, y - r, xx + r, y + r, colour)
                    cv.d.arc([(xx - 1.9 * r) * S, (y - 0.45 * r) * S, (xx + 1.9 * r) * S, (y + 0.45 * r) * S], 0, 180,
                             fill=_c((0.95, 0.85, 0.6)), width=int(0.12 * r) * S)
                _both(cv, x, 2 * r, planet)
        layer(planets, 0.1, (1, 1, 1))

        def moon_ground(cv):
            line = _wave(cv.w, ground - 0.02 * u, 0.012 * u, seed, (2, 3, 7))
            cv.poly([tuple(p) for p in line] + [(cv.w, cv.h + 5), (0, cv.h + 5)], (0.72, 0.72, 0.76))
            rng2 = np.random.default_rng(seed + 3)
            for _ in range(12):
                x, y = rng2.uniform(0, cv.w), ground + rng2.uniform(0.05, 0.8) * (cv.h - ground)
                r = rng2.uniform(0.02, 0.05) * u
                _both(cv, x, r, lambda xx: cv.ellipse(xx - r, y - 0.35 * r, xx + r, y + 0.35 * r, (0.6, 0.6, 0.65)))
        layer(moon_ground, 1.0, (1, 1, 1))
    elif setting == "underwater":
        top, bottom = np.asarray((0.15, 0.6, 0.85), F32), np.asarray((0.03, 0.2, 0.42), F32)
        k = np.linspace(0, 1, H, dtype=F32)[:, None, None]
        bd.sky = sky = np.broadcast_to(top + (bottom - top) * k, (H, W, 3)).copy()
        if night:
            sky *= 0.55
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        for x0 in (0.2, 0.45, 0.75):  # light rays from the surface
            ray = np.clip(1.0 - np.abs(xx - x0 * W - 0.35 * yy) / (0.05 * W), 0, 1) * (1 - yy / H)
            sky += ray[..., None] * 0.12
        bd.behind += [_fish(W, H, ground, seed=seed + 1)]
        bd.front.append(_bubbles(W, H, seed=seed + 2))

        def seabed(cv):
            line = _wave(cv.w, ground - 0.02 * u, 0.012 * u, seed, (2, 3))
            cv.poly([tuple(p) for p in line] + [(cv.w, cv.h + 5), (0, cv.h + 5)], (0.93, 0.83, 0.6))
            rng2 = np.random.default_rng(seed + 4)
            for _ in range(9):
                x, hgt = rng2.uniform(0, cv.w), rng2.uniform(0.15, 0.32) * u
                colour = [(0.25, 0.65, 0.35), (0.95, 0.45, 0.5), (0.95, 0.6, 0.3)][rng2.integers(3)]

                def weed(xx, hgt=hgt, colour=colour):
                    pts = [(xx + 0.02 * u * math.sin(i * 1.3), ground - hgt * i / 8) for i in range(9)]
                    cv.lines(pts, colour, 0.018 * u)
                _both(cv, x, 0.05 * u, weed)
        layer(seabed, 1.0, (1, 1, 1) if not night else (0.6, 0.65, 0.8))
    if setting in ("garden", "meadow", "forest", "playground", "mountain"):
        bd.front.append(_front_plants(W, H, ground, time, seed, (0.3, 0.55, 0.24) if setting == "forest"
                                      else (0.34, 0.64, 0.27)))
    if outdoor and weather in ("clear", "leaves") and time != "night":
        bd.front.append(_motes(W, H, time, seed=seed + 12))
    if weather in ("rain", "snow", "leaves") and outdoor:
        bd.front.append(_falling(W, H, weather, seed=seed + 9))
    return bd


# --------------------------------------------------------------------------- light, depth and air

LIGHT = {"day": ((1.0, 0.97, 0.86), 0.4), "sunset": ((1.0, 0.8, 0.6), 0.45), "night": ((0.72, 0.8, 1.0), 0.2)}


def _finish(rgba: np.ndarray, u: float, time: str, depth: float, horizon, outdoor: bool) -> np.ndarray:
    """Give a flat layer light and depth: sunlit rims on the edges facing up and
    left, soft shade on the edges facing down, a little shade low on the
    ground, and, for far layers, haze towards the sky's colour and a slight blur."""
    img = rgba.astype(F32) / 255.0
    a = img[..., 3]
    rgb = img[..., :3]
    sigma = max(1.0, 0.011 * u)
    soft = cv2.GaussianBlur(a, (0, 0), sigma)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    facing = (0.85 * gy + 0.45 * gx) * sigma * 2.6
    colour, strength = LIGHT[time]
    if not outdoor:
        strength *= 0.6
    lit = (np.clip(facing, 0.0, 1.0) * a)[..., None]
    shade = (np.clip(-facing, 0.0, 1.0) * a)[..., None]
    rgb = rgb + (np.asarray(colour, F32) - rgb) * lit * strength
    rgb = rgb * (1.0 - 0.24 * shade)
    if outdoor and depth > 0.0:
        rgb = rgb + (np.asarray(horizon, F32) - rgb) * (0.32 * depth)
        blur = 0.0016 * u * depth
        if blur >= 0.5:
            pre = cv2.GaussianBlur(rgb * a[..., None], (0, 0), blur)
            a = cv2.GaussianBlur(a, (0, 0), blur)
            rgb = pre / np.maximum(a[..., None], 1e-4)
    return (np.dstack([np.clip(rgb, 0, 1), np.clip(a, 0, 1)]) * 255 + 0.5).astype(np.uint8)


def _shade_low(rgba: np.ndarray, ground: float, height: int) -> None:
    """Darken the ground a little towards the bottom of the frame (in place)."""
    y = np.arange(rgba.shape[0], dtype=F32)
    k = 1.0 - 0.16 * np.clip((y - ground) / max(height - ground, 1.0), 0.0, 1.0)
    rgba[..., :3] = (rgba[..., :3].astype(F32) * k[:, None, None]).astype(np.uint8)


def _rays(sky: np.ndarray, cx: float, cy: float, reach: float, colour, strength: float, count: int = 14) -> None:
    h, w = sky.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    ang = np.arctan2(yy - cy, xx - cx)
    dist = np.hypot(xx - cx, yy - cy)
    beams = (0.5 + 0.5 * np.cos(count * ang)) ** 6 * np.exp(-dist / reach)
    sky += (np.asarray(colour, F32) - sky) * (beams * strength)[..., None]


def _motes(w: int, h: int, time: str, count: int = 24, seed: int = 11) -> Painter:
    """Specks of pollen and dust drifting in the light."""
    rng = np.random.default_rng(seed)
    u = min(w, h)
    r = max(3, int(0.006 * u))
    g = cv2.getGaussianKernel(4 * r + 1, r)
    blob = (g @ g.T).astype(F32)
    blob /= blob.max()
    colour = {"day": (1.0, 1.0, 0.85), "sunset": (1.0, 0.85, 0.6), "night": (0.85, 1.0, 0.6)}[time]
    sprite = np.dstack([np.ones(blob.shape + (3,), F32) * np.asarray(colour, F32), blob * 0.8])
    items = [(rng.uniform(0, w), rng.uniform(0.15, 0.95) * h, rng.uniform(0.3, 1.0), rng.uniform(0, 6))
             for _ in range(count)]

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        for x0, y0, depth, p in items:
            x = (x0 + 0.02 * w * t * depth + 0.03 * w * math.sin(t * 0.4 + p) - scroll * depth) % w
            y = y0 - 0.015 * h * t * depth % h + 0.02 * h * math.sin(t * 0.6 + 2 * p)
            k = 0.5 + 0.5 * math.sin(t * 1.3 + p)
            _paste(frame, sprite, x - 2 * r, y % h - 2 * r, opacity=0.35 + 0.5 * k * depth)

    return paint


def _front_plants(w: int, h: int, ground: float, time: str, seed: int, colour=(0.32, 0.62, 0.26)) -> Painter:
    """Grass and flowers right at the bottom of the frame, passing in front."""
    u = min(w, h)
    L = 2 * w
    cv = Canvas(L, h, TINT[time])
    rng = np.random.default_rng(seed + 31)
    x = rng.uniform(0, 0.1) * L
    while x < L:
        size = rng.uniform(0.07, 0.13) * u
        base = h + 0.02 * u

        def tuft(xx, size=size, base=base):
            for k in range(7):
                a = math.radians(rng.uniform(-35, 35))
                tip = (xx + math.sin(a) * size + rng.uniform(-0.02, 0.02) * u, base - math.cos(a) * size)
                cv.poly([(xx - 0.012 * u, base), tip, (xx + 0.012 * u, base)], _dark(colour, rng.uniform(0.8, 1.05)))
            if rng.random() < 0.55:
                fx, fy, r = xx + rng.uniform(-0.3, 0.3) * size, base - 0.9 * size, 0.018 * u
                cv.lines([(fx, base), (fx, fy)], _dark(colour, 0.75), 0.006 * u)
                petal = [(1.0, 0.55, 0.65), (1.0, 0.88, 0.35), (0.98, 0.98, 1.0), (0.75, 0.6, 1.0)][rng.integers(4)]
                for j in range(5):
                    ang = 2 * math.pi * j / 5
                    cv.ellipse(fx + math.cos(ang) * r - 0.7 * r, fy + math.sin(ang) * r - 0.7 * r,
                               fx + math.cos(ang) * r + 0.7 * r, fy + math.sin(ang) * r + 0.7 * r, petal)
                cv.ellipse(fx - 0.5 * r, fy - 0.5 * r, fx + 0.5 * r, fy + 0.5 * r, (1.0, 0.8, 0.25))
        _both(cv, x, size, tuft)
        x += rng.uniform(0.18, 0.4) * w
    layer = _finish(cv.done(), u, time, 0.0, (1, 1, 1), True)

    def paint(frame: np.ndarray, t: float, scroll: float) -> None:
        _blend(frame, _window(layer, scroll * 1.3, w))

    return paint


# --------------------------------------------------------------------------- new places

def _sunflower(cv: Canvas, x, ground, height, u):
    cv.lines([(x, ground), (x + 0.01 * u, ground - 0.5 * height), (x, ground - height)], (0.3, 0.6, 0.25), 0.012 * u)
    for side, y in ((-1, 0.45), (1, 0.62)):
        cv.poly([(x, ground - y * height), (x + side * 0.07 * u, ground - y * height - 0.03 * u),
                 (x + side * 0.05 * u, ground - y * height + 0.01 * u)], (0.36, 0.68, 0.3))
    r = 0.055 * u
    cy = ground - height
    for j in range(12):
        a = 2 * math.pi * j / 12
        px, py = x + math.cos(a) * r * 1.05, cy + math.sin(a) * r * 1.05
        cv.ellipse(px - 0.55 * r, py - 0.55 * r, px + 0.55 * r, py + 0.55 * r, (1.0, 0.8, 0.15))
    cv.ellipse(x - 0.62 * r, cy - 0.62 * r, x + 0.62 * r, cy + 0.62 * r, (0.55, 0.33, 0.15))


def _pot_house(cv: Canvas, x, ground, size, night):
    """Peeko's home: a big upside-down flowerpot with a round door."""
    terracotta, rim = (0.88, 0.5, 0.32), (0.8, 0.42, 0.26)
    w0, w1, h = 0.5 * size, 0.36 * size, 0.62 * size
    cv.poly([(x - w0, ground), (x + w0, ground), (x + w1, ground - h), (x - w1, ground - h)], terracotta)
    cv.rect(x - w1 - 0.05 * size, ground - h - 0.12 * size, x + w1 + 0.05 * size, ground - h + 0.02 * size, rim,
            radius=0.03 * size)
    cv.lines([(x, ground - h - 0.12 * size), (x + 0.03 * size, ground - h - 0.3 * size)], (0.3, 0.6, 0.25), 0.03 * size)
    for side in (-1, 1):
        a = math.radians(90 - side * 50)
        tip = (x + 0.03 * size + math.cos(a) * 0.2 * size, ground - h - 0.3 * size - math.sin(a) * 0.1 * size)
        cv.poly([(x + 0.03 * size, ground - h - 0.3 * size), ((x + tip[0]) / 2 + 0.015 * size, tip[1] - 0.06 * size),
                 tip, ((x + tip[0]) / 2, tip[1] + 0.04 * size)], (0.46, 0.8, 0.33))
    cv.d.chord([(x - 0.13 * size) * S, (ground - 0.42 * size) * S, (x + 0.13 * size) * S, (ground - 0.15 * size) * S],
               180, 360, fill=_c((0.55, 0.32, 0.2), cv.tint), outline=_c((0.3, 0.18, 0.1), cv.tint), width=cv.line * S)
    cv.rect(x - 0.13 * size, ground - 0.29 * size, x + 0.13 * size, ground, (0.55, 0.32, 0.2))
    cv.ellipse(x + 0.07 * size, ground - 0.15 * size, x + 0.1 * size, ground - 0.12 * size, (1.0, 0.85, 0.3))
    glow = (1.0, 0.86, 0.45) if night else (0.7, 0.88, 1.0)
    for side in (-1, 1):
        wx = x + side * 0.27 * size
        cv.ellipse(wx - 0.07 * size, ground - 0.47 * size, wx + 0.07 * size, ground - 0.33 * size, glow, lit=night)
        cv.lines([(wx - 0.07 * size, ground - 0.4 * size), (wx + 0.07 * size, ground - 0.4 * size)], (0.98, 0.95, 0.9),
                 0.012 * size)
    for k in range(3):
        cv.lines([(x - w0 + 0.05 * size, ground - (0.1 + 0.17 * k) * size),
                  (x - w0 + 0.12 * size + 0.04 * k * size, ground - (0.1 + 0.17 * k) * size)], rim, 0.012 * size)


def _fence(cv: Canvas, y, u, colour=(0.98, 0.96, 0.9)):
    cv.rect(-5, y - 0.15 * u, cv.w + 5, y - 0.13 * u, colour)
    cv.rect(-5, y - 0.07 * u, cv.w + 5, y - 0.05 * u, colour)
    for x in np.arange(0, cv.w, 0.075 * u):
        cv.poly([(x, y), (x, y - 0.18 * u), (x + 0.02 * u, y - 0.21 * u), (x + 0.04 * u, y - 0.18 * u), (x + 0.04 * u, y)],
                colour)


def _garden(layer, W, u, ground, seed, night):
    def far(cv):
        line = _hills(cv, ground - 0.26 * u, 0.05 * u, (0.6, 0.82, 0.52), seed)
        for x in np.linspace(0, cv.w, 7, endpoint=False):
            _both(cv, x, 0.15 * u, lambda xx: _round_tree(cv, xx, _top(line, xx) + 0.02 * u, 0.18 * u,
                                                          (0.42, 0.72, 0.38), seed=int(xx)))
    layer(far, 0.15)

    def mid(cv):
        _hills(cv, ground - 0.09 * u, 0.02 * u, (0.48, 0.76, 0.4), seed + 1, (1, 2))
        _fence(cv, ground - 0.05 * u, u)
        rng = np.random.default_rng(seed + 2)
        house = 0.13 * cv.w  # in sight when a scene starts
        for x in np.arange(0.05 * cv.w, cv.w, 0.4 * u):
            if abs(x - house) < 0.42 * u or abs(x - house - cv.w) < 0.42 * u or rng.random() < 0.3:
                continue
            hgt = rng.uniform(0.27, 0.38) * u
            for dx in ((0.0,) if rng.random() < 0.5 else (0.0, 0.07 * u)):
                _both(cv, x + dx, 0.1 * u, lambda xx, hgt=hgt - dx: _sunflower(cv, xx, ground - 0.04 * u, hgt, u))
        _both(cv, house, 0.4 * u, lambda xx: _pot_house(cv, xx, ground - 0.02 * u, 0.62 * u, night))
    layer(mid, 0.5)

    def near(cv):
        _outdoor_ground(cv, ground, (0.42, 0.74, 0.32), seed, flowers=50)
        rng = np.random.default_rng(seed + 3)
        for x in np.arange(0.62 * cv.w, 0.84 * cv.w, 0.07 * u):  # stepping stones to the door
            y = ground + rng.uniform(0.02, 0.08) * u
            cv.ellipse(x, y, x + 0.05 * u, y + 0.022 * u, (0.85, 0.83, 0.78))
        for k in range(2):  # the vegetable patch
            x0 = (0.2 + 0.18 * k) * cv.w
            cv.rect(x0, ground + 0.03 * u, x0 + 0.28 * u, ground + 0.09 * u, (0.55, 0.36, 0.22), radius=0.02 * u)
            for x in np.arange(x0 + 0.03 * u, x0 + 0.27 * u, 0.05 * u):
                for a in (-30, 0, 30):
                    tip = (x + 0.03 * u * math.sin(math.radians(a)), ground + 0.02 * u - 0.04 * u)
                    cv.poly([(x - 0.006 * u, ground + 0.045 * u), tip, (x + 0.006 * u, ground + 0.045 * u)],
                            (0.35, 0.72, 0.3), outline=False)
    layer(near, 1.0)


def _swing_set(cv, x, ground, u):
    post = (0.85, 0.3, 0.25)
    top = ground - 0.42 * u
    for side in (-1, 1):
        cv.lines([(x + side * 0.22 * u, top), (x + side * 0.3 * u, ground)], post, 0.022 * u)
        cv.lines([(x + side * 0.22 * u, top), (x + side * 0.14 * u, ground)], post, 0.022 * u)
    cv.lines([(x - 0.24 * u, top), (x + 0.24 * u, top)], post, 0.026 * u)
    for side in (-1, 1):
        sx = x + side * 0.1 * u
        cv.lines([(sx - 0.04 * u, top), (sx - 0.04 * u, ground - 0.1 * u)], (0.4, 0.35, 0.3), 0.006 * u)
        cv.lines([(sx + 0.04 * u, top), (sx + 0.04 * u, ground - 0.1 * u)], (0.4, 0.35, 0.3), 0.006 * u)
        cv.rect(sx - 0.06 * u, ground - 0.11 * u, sx + 0.06 * u, ground - 0.085 * u, (1.0, 0.8, 0.2), radius=0.01 * u)


def _slide(cv, x, ground, u):
    top = ground - 0.34 * u
    cv.lines([(x, ground), (x, top)], (0.35, 0.55, 0.85), 0.02 * u)
    cv.lines([(x + 0.1 * u, ground), (x + 0.1 * u, top)], (0.35, 0.55, 0.85), 0.02 * u)
    for k in range(1, 6):
        y = ground - k * 0.06 * u
        cv.lines([(x, y), (x + 0.1 * u, y)], (0.35, 0.55, 0.85), 0.012 * u)
    cv.rect(x - 0.01 * u, top - 0.02 * u, x + 0.2 * u, top + 0.02 * u, (1.0, 0.75, 0.2))
    pts = [(x + 0.2 * u + t * 0.42 * u, top + (ground - 0.03 * u - top) * (1 - (1 - t) ** 2.2)) for t in
           np.linspace(0, 1, 16)]
    cv.lines(pts, (0.95, 0.3, 0.3), 0.05 * u)
    cv.lines([(px, py - 0.022 * u) for px, py in pts], (1.0, 0.55, 0.5), 0.01 * u)


def _playground(layer, u, ground, seed):
    def far(cv):
        line = _hills(cv, ground - 0.2 * u, 0.03 * u, (0.55, 0.8, 0.5), seed, (1, 2))
        for x in np.arange(0, cv.w, 0.16 * u):
            _both(cv, x, 0.15 * u, lambda xx: _round_tree(cv, xx, _top(line, xx) + 0.03 * u, 0.26 * u,
                                                          (0.36, 0.66, 0.34), seed=int(xx)))
    layer(far, 0.15)

    def mid(cv):
        for x in (0.1 * cv.w, 0.6 * cv.w):
            _both(cv, x, 0.4 * u, lambda xx: _swing_set(cv, xx, ground - 0.03 * u, u))
        for x in (0.32 * cv.w, 0.82 * cv.w):
            _both(cv, x, 0.7 * u, lambda xx: _slide(cv, xx, ground - 0.03 * u, u))
    layer(mid, 0.5)

    def near(cv):
        _outdoor_ground(cv, ground, (0.44, 0.74, 0.34), seed, flowers=25)
        for x in (0.45 * cv.w, 0.95 * cv.w):
            def sandbox(xx):
                cv.rect(xx, ground + 0.02 * u, xx + 0.4 * u, ground + 0.1 * u, (0.7, 0.48, 0.28), radius=0.015 * u)
                cv.rect(xx + 0.02 * u, ground + 0.035 * u, xx + 0.38 * u, ground + 0.085 * u, (0.98, 0.88, 0.62),
                        outline=False)
                cv.poly([(xx + 0.26 * u, ground + 0.07 * u), (xx + 0.32 * u, ground + 0.07 * u),
                         (xx + 0.31 * u, ground + 0.02 * u), (xx + 0.27 * u, ground + 0.02 * u)], (0.3, 0.6, 0.95))
            _both(cv, x, 0.45 * u, sandbox)
    layer(near, 1.0)


def _kitchen(layer, u, ground, time):
    night = time == "night"

    def room(cv):
        wall = (0.86, 0.95, 0.9)
        cv.rect(-5, -5, cv.w + 5, ground, wall, outline=False)
        for y in np.arange(ground - 0.3 * u, ground - 0.06 * u, 0.05 * u):  # tiles behind the counter
            cv.lines([(0, y), (cv.w, y)], (0.78, 0.88, 0.84), 0.003 * u)
        for x in np.arange(0, cv.w, 0.05 * u):
            cv.lines([(x, ground - 0.3 * u), (x, ground - 0.06 * u)], (0.78, 0.88, 0.84), 0.003 * u)
        for y0 in np.arange(ground, cv.h, 0.06 * u):  # chequered floor
            for k, x in enumerate(np.arange(0, cv.w, 0.06 * u)):
                if (k + int((y0 - ground) / (0.06 * u))) % 2 == 0:
                    cv.rect(x, y0, x + 0.06 * u, y0 + 0.06 * u, (0.95, 0.88, 0.78), outline=False)
                else:
                    cv.rect(x, y0, x + 0.06 * u, y0 + 0.06 * u, (0.82, 0.68, 0.55), outline=False)
        sky_top, sky_bottom = SKY[time]
        for wx in (0.12 * cv.w, 0.62 * cv.w):
            def window(xx):
                x0, y0, ww, wh = xx, 0.1 * cv.h, 0.32 * u, 0.24 * u
                cv.rect(x0 - 0.015 * u, y0 - 0.015 * u, x0 + ww + 0.015 * u, y0 + wh + 0.015 * u, (1, 1, 1))
                cv.rect(x0, y0, x0 + ww, y0 + wh, sky_bottom if not night else sky_top, lit=True)
                cv.lines([(x0 + ww / 2, y0), (x0 + ww / 2, y0 + wh)], (1, 1, 1), 0.01 * u)
                for side in (0, 1):
                    cx = x0 - 0.03 * u if side == 0 else x0 + ww - 0.06 * u
                    cv.poly([(cx, y0 - 0.03 * u), (cx + 0.09 * u, y0 - 0.03 * u), (cx + 0.05 * u, y0 + wh + 0.04 * u),
                             (cx + 0.01 * u, y0 + wh + 0.04 * u)], (1.0, 0.6, 0.55))
            _both(cv, wx, 0.4 * u, window)
        for cx in (0.38 * cv.w, 0.88 * cv.w):  # cupboards and counter
            def counter(xx):
                cv.rect(xx, ground - 0.66 * u, xx + 0.5 * u, ground - 0.5 * u, (0.98, 0.78, 0.5))
                for k in range(3):
                    kx = xx + (0.08 + 0.16 * k) * u
                    cv.ellipse(kx, ground - 0.59 * u, kx + 0.02 * u, ground - 0.57 * u, (0.6, 0.4, 0.25))
                cv.rect(xx - 0.01 * u, ground - 0.24 * u, xx + 0.51 * u, ground - 0.21 * u, (0.95, 0.95, 0.95))
                cv.rect(xx, ground - 0.21 * u, xx + 0.5 * u, ground + 0.02 * u, (0.98, 0.78, 0.5))
                cv.lines([(xx + 0.25 * u, ground - 0.2 * u), (xx + 0.25 * u, ground + 0.01 * u)], (0.85, 0.62, 0.38),
                         0.006 * u)
                cv.lines([(xx + 0.32 * u, ground - 0.24 * u), (xx + 0.32 * u, ground - 0.31 * u),
                          (xx + 0.38 * u, ground - 0.31 * u)], (0.7, 0.72, 0.76), 0.012 * u)
                for k, colour in enumerate(((0.95, 0.35, 0.3), (1.0, 0.85, 0.3), (0.5, 0.8, 0.4))):
                    jx = xx + (0.04 + 0.06 * k) * u
                    cv.rect(jx, ground - 0.31 * u, jx + 0.045 * u, ground - 0.24 * u, colour, radius=0.01 * u)
            _both(cv, cx, 0.55 * u, counter)
        for fx in (0.04 * cv.w,):  # the fridge
            def fridge(xx):
                cv.rect(xx, ground - 0.62 * u, xx + 0.24 * u, ground + 0.02 * u, (0.92, 0.96, 1.0), radius=0.03 * u)
                cv.lines([(xx, ground - 0.4 * u), (xx + 0.24 * u, ground - 0.4 * u)], (0.7, 0.75, 0.85), 0.006 * u)
                cv.rect(xx + 0.19 * u, ground - 0.55 * u, xx + 0.205 * u, ground - 0.45 * u, (0.6, 0.65, 0.75))
                cv.ellipse(xx + 0.05 * u, ground - 0.56 * u, xx + 0.09 * u, ground - 0.52 * u, (0.95, 0.4, 0.45))
            _both(cv, fx, 0.3 * u, fridge)
    tint_in = {"day": (1, 1, 1), "sunset": (1.0, 0.92, 0.84), "night": (0.66, 0.66, 0.82)}[time]
    layer(room, 0.25, tint_in)
