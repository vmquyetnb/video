"""Things drawn by the program for the cartoons: props a character holds or
that appear on the lesson board (fruit, a toothbrush, a book, shapes...), and
two little friends (a bird and a bunny) who can be happy or sad.

Each drawing is made in a unit box (0..1, y down) at twice the size and
scaled down; :func:`prop_image` gives a Pillow image (for drawing into the
character), :func:`prop_sprite` a :class:`Sprite` (for pasting onto frames).
"""

from __future__ import annotations

import functools
import math
from typing import Callable, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .showtime import Sprite

F32 = np.float32
OUTLINE = (41, 33, 56, 255)
RED, ORANGE, YELLOW = (232, 62, 58), (247, 146, 43), (255, 214, 64)
GREEN, DARK_GREEN, BLUE = (106, 196, 84), (52, 140, 64), (66, 140, 236)
PURPLE, PINK, BROWN = (150, 96, 214), (246, 140, 178), (150, 96, 58)
WHITE, CREAM = (255, 255, 255), (255, 246, 228)


def _c(rgb, a=255):
    return tuple(rgb[:3]) + (a,)


class Pen:
    """Drawing in a unit box on a square canvas of ``size`` pixels."""

    def __init__(self, d: ImageDraw.ImageDraw, size: int):
        self.d, self.s = d, size
        self.w = max(2, int(round(size / 30)))  # outline width

    def p(self, pts):
        return [(x * self.s, y * self.s) for x, y in pts]

    def ellipse(self, x0, y0, x1, y1, fill, outline=True):
        self.d.ellipse([x0 * self.s, y0 * self.s, x1 * self.s, y1 * self.s], fill=_c(fill),
                       outline=OUTLINE if outline else None, width=self.w if outline else 0)

    def circle(self, cx, cy, r, fill, outline=True):
        self.ellipse(cx - r, cy - r, cx + r, cy + r, fill, outline)

    def poly(self, pts, fill, outline=True):
        pts = self.p(pts)
        self.d.polygon(pts, fill=_c(fill))
        if outline:
            self.d.line(pts + [pts[0]], fill=OUTLINE, width=self.w, joint="curve")

    def rect(self, x0, y0, x1, y1, fill, radius=0.0, outline=True):
        self.d.rounded_rectangle([x0 * self.s, y0 * self.s, x1 * self.s, y1 * self.s], radius=radius * self.s,
                                 fill=_c(fill), outline=OUTLINE if outline else None,
                                 width=self.w if outline else 0)

    def line(self, pts, fill=OUTLINE, width=1.0):
        self.d.line(self.p(pts), fill=_c(fill), width=max(1, int(self.w * width)), joint="curve")

    def arc(self, cx, cy, rx, ry, a0, a1, fill=OUTLINE, width=1.0):
        pts = [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
               for a in np.linspace(a0, a1, 24)]
        self.line(pts, fill, width)


def _star_pts(cx, cy, r, inner=0.45, points=5, turn=-90.0):
    pts = []
    for i in range(points * 2):
        rr = r * (1.0 if i % 2 == 0 else inner)
        a = math.radians(turn + 180.0 * i / points)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _heart_pts(cx, cy, r):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 60, endpoint=False):
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + r * x / 17, cy - r * y / 17))
    return pts


def _face(p: Pen, cx, cy, r, happy=True):
    """Two dots and a smile (or a frown) for food and friends."""
    for dx in (-0.35, 0.35):
        p.circle(cx + dx * r, cy - 0.1 * r, 0.11 * r, (40, 30, 50), outline=False)
    if happy:
        p.arc(cx, cy + 0.1 * r, 0.3 * r, 0.25 * r, 20, 160, width=0.8)
    else:
        p.arc(cx, cy + 0.45 * r, 0.25 * r, 0.18 * r, 200, 340, width=0.8)


# --------------------------------------------------------------------------- props

def _apple(p: Pen):
    p.line([(0.5, 0.3), (0.54, 0.1)], BROWN, 1.6)
    p.poly([(0.55, 0.16), (0.72, 0.06), (0.86, 0.12), (0.7, 0.22)], GREEN)
    p.ellipse(0.12, 0.24, 0.62, 0.92, RED)
    p.ellipse(0.38, 0.24, 0.88, 0.92, RED)
    p.ellipse(0.2, 0.3, 0.8, 0.9, RED, outline=False)
    p.ellipse(0.26, 0.36, 0.38, 0.52, (255, 190, 180), outline=False)


def _banana(p: Pen):
    outer = [(0.14 + 0.72 * t, 0.22 + 0.62 * math.sin(math.pi * t) ** 0.7) for t in np.linspace(0, 1, 24)]
    inner = [(0.14 + 0.72 * t, 0.22 + 0.3 * math.sin(math.pi * t) ** 1.6) for t in np.linspace(1, 0, 24)]
    p.poly(outer + inner, YELLOW)
    p.line([(0.3, 0.5), (0.5, 0.64), (0.7, 0.5)], (230, 180, 40), 0.8)
    p.line([(0.1, 0.17), (0.16, 0.24)], BROWN, 2.2)
    p.circle(0.86, 0.23, 0.025, BROWN, outline=False)


def _carrot(p: Pen):
    for dx, a in ((-0.08, -30), (0.0, 0), (0.08, 30)):
        x, y = 0.5 + dx, 0.24
        tip = (x + 0.18 * math.sin(math.radians(a)), y - 0.2 * math.cos(math.radians(a)))
        p.poly([(x - 0.05, y), tip, (x + 0.05, y)], GREEN)
    p.poly([(0.3, 0.24), (0.7, 0.24), (0.66, 0.4), (0.52, 0.94), (0.48, 0.94), (0.34, 0.4)], ORANGE)
    for y in (0.42, 0.58, 0.74):
        p.line([(0.42 + (y - 0.4) * 0.1, y), (0.52, y + 0.02)], (200, 100, 30), 0.8)


def _broccoli(p: Pen):
    p.poly([(0.42, 0.5), (0.58, 0.5), (0.62, 0.94), (0.38, 0.94)], (170, 220, 120))
    for cx, cy, r in ((0.3, 0.42, 0.16), (0.7, 0.42, 0.16), (0.5, 0.28, 0.2), (0.38, 0.22, 0.13),
                      (0.64, 0.22, 0.13), (0.5, 0.48, 0.15)):
        p.circle(cx, cy, r, DARK_GREEN)


def _strawberry(p: Pen):
    p.poly([(0.5, 0.94), (0.16, 0.45), (0.2, 0.3), (0.36, 0.24), (0.5, 0.28), (0.64, 0.24), (0.8, 0.3),
            (0.84, 0.45)], RED)
    for x, y in ((0.35, 0.42), (0.5, 0.4), (0.65, 0.42), (0.42, 0.58), (0.58, 0.58), (0.5, 0.74),
                 (0.3, 0.5), (0.7, 0.5)):
        p.ellipse(x - 0.018, y - 0.03, x + 0.018, y + 0.03, YELLOW, outline=False)
    p.poly([(0.3, 0.22), (0.42, 0.16), (0.5, 0.06), (0.58, 0.16), (0.7, 0.22), (0.58, 0.28), (0.5, 0.24),
            (0.42, 0.28)], GREEN)


def _cookie(p: Pen):
    p.circle(0.5, 0.5, 0.4, (222, 170, 100))
    for x, y in ((0.36, 0.38), (0.6, 0.32), (0.66, 0.56), (0.42, 0.62), (0.52, 0.48), (0.3, 0.55)):
        p.circle(x, y, 0.045, (90, 54, 34), outline=False)


def _candy(p: Pen):
    p.poly([(0.3, 0.5), (0.06, 0.32), (0.06, 0.68)], PINK)
    p.poly([(0.7, 0.5), (0.94, 0.32), (0.94, 0.68)], PINK)
    p.circle(0.5, 0.5, 0.22, (255, 90, 140))
    p.arc(0.5, 0.5, 0.13, 0.13, 200, 340, WHITE, 1.4)
    p.arc(0.5, 0.5, 0.13, 0.13, 20, 160, WHITE, 1.4)


def _toothbrush(p: Pen):
    p.rect(0.04, 0.5, 0.66, 0.62, BLUE, radius=0.06)
    p.rect(0.62, 0.52, 0.96, 0.62, BLUE, radius=0.04)
    for x in np.linspace(0.68, 0.92, 6):
        p.line([(x, 0.52), (x, 0.42)], WHITE, 1.4)
        p.line([(x, 0.52), (x, 0.42)], (170, 200, 230), 0.4)
    p.ellipse(0.66, 0.3, 0.94, 0.44, (120, 220, 230))
    p.ellipse(0.72, 0.32, 0.86, 0.39, WHITE, outline=False)


def _cup(p: Pen):
    p.poly([(0.22, 0.14), (0.78, 0.14), (0.7, 0.92), (0.3, 0.92)], (225, 240, 250))
    p.poly([(0.25, 0.36), (0.75, 0.36), (0.7, 0.9), (0.3, 0.9)], (110, 190, 245), outline=False)
    p.line([(0.3, 0.2), (0.34, 0.8)], WHITE, 1.2)


def _soap(p: Pen):
    p.rect(0.12, 0.4, 0.88, 0.82, PINK, radius=0.14)
    p.ellipse(0.24, 0.48, 0.6, 0.6, (255, 200, 222), outline=False)
    for cx, cy, r in ((0.3, 0.26, 0.09), (0.52, 0.18, 0.12), (0.74, 0.28, 0.08)):
        p.circle(cx, cy, r, (215, 240, 255))
        p.circle(cx - 0.3 * r, cy - 0.3 * r, 0.25 * r, WHITE, outline=False)


def _towel(p: Pen):
    p.rect(0.14, 0.2, 0.86, 0.84, (90, 190, 220), radius=0.06)
    for y in (0.3, 0.74):
        p.line([(0.16, y), (0.84, y)], WHITE, 1.6)
    p.rect(0.14, 0.2, 0.86, 0.34, (60, 160, 200), radius=0.06)


def _book(p: Pen):
    p.poly([(0.04, 0.3), (0.5, 0.36), (0.96, 0.3), (0.96, 0.8), (0.5, 0.88), (0.04, 0.8)], RED)
    p.poly([(0.08, 0.24), (0.5, 0.32), (0.5, 0.82), (0.08, 0.74)], CREAM)
    p.poly([(0.92, 0.24), (0.5, 0.32), (0.5, 0.82), (0.92, 0.74)], CREAM)
    for k in range(4):
        y = 0.38 + 0.08 * k
        p.line([(0.16, y), (0.42, y + 0.04)], (170, 160, 150), 0.6)
        p.line([(0.58, y + 0.04), (0.84, y)], (170, 160, 150), 0.6)


def _ball(p: Pen):
    p.circle(0.5, 0.5, 0.4, WHITE)
    box = [0.1 * p.s, 0.1 * p.s, 0.9 * p.s, 0.9 * p.s]
    for k, colour in enumerate((RED, YELLOW, BLUE)):
        p.d.pieslice(box, -90 + 120 * k, -30 + 120 * k, fill=_c(colour))
    p.d.ellipse(box, outline=OUTLINE, width=p.w)
    p.circle(0.5, 0.5, 0.08, WHITE)


def _teddy(p: Pen):
    tan = (196, 140, 90)
    p.ellipse(0.25, 0.5, 0.75, 0.96, tan)
    for x in (0.28, 0.72):
        p.circle(x, 0.18, 0.09, tan)
    p.circle(0.5, 0.34, 0.24, tan)
    p.ellipse(0.4, 0.38, 0.6, 0.52, (236, 206, 170))
    p.circle(0.5, 0.41, 0.03, (40, 30, 30), outline=False)
    for x in (0.42, 0.58):
        p.circle(x, 0.3, 0.025, (40, 30, 30), outline=False)
    p.ellipse(0.36, 0.64, 0.64, 0.88, (236, 206, 170), outline=False)


def _umbrella(p: Pen):
    p.line([(0.5, 0.2), (0.5, 0.86)], (90, 80, 100), 1.6)
    p.arc(0.43, 0.86, 0.07, 0.07, 0, 180, (90, 80, 100), 1.6)
    top = [(0.5 + 0.46 * math.cos(a), 0.46 - 0.4 * math.sin(a)) for a in np.linspace(0, math.pi, 30)]
    scallops = []
    for k in range(4, 0, -1):
        x0, x1 = 0.04 + 0.23 * (k - 1), 0.04 + 0.23 * k
        scallops += [((x0 + x1) / 2 + (x1 - x0) / 2 * math.cos(a), 0.46 - 0.05 * math.sin(a))
                     for a in np.linspace(0, math.pi, 8)]
    p.poly(top + scallops, RED)
    for k in (1, 3):
        x = 0.04 + 0.23 * k
        p.poly([(0.5, 0.06), (x - 0.115, 0.46), (x + 0.115, 0.46)], WHITE, outline=False)
    p.line([(0.5, 0.06), (0.5, 0.0)], (90, 80, 100), 1.6)


def _balloon(p: Pen):
    p.line([(0.5, 0.6), (0.46, 0.75), (0.54, 0.88), (0.5, 1.0)], (90, 80, 100), 0.6)
    p.poly([(0.46, 0.62), (0.54, 0.62), (0.5, 0.56)], RED)
    p.ellipse(0.26, 0.02, 0.74, 0.6, RED)
    p.ellipse(0.34, 0.1, 0.44, 0.24, (255, 170, 170), outline=False)


def _flower(p: Pen):
    p.line([(0.5, 0.45), (0.5, 0.96)], DARK_GREEN, 1.6)
    p.poly([(0.5, 0.75), (0.7, 0.62), (0.78, 0.7), (0.56, 0.8)], GREEN)
    for k in range(5):
        a = math.radians(-90 + 72 * k)
        p.circle(0.5 + 0.17 * math.cos(a), 0.32 + 0.17 * math.sin(a), 0.13, PINK)
    p.circle(0.5, 0.32, 0.1, YELLOW)


def _gift(p: Pen):
    p.rect(0.14, 0.36, 0.86, 0.92, (110, 170, 245))
    p.rect(0.1, 0.28, 0.9, 0.42, (90, 150, 230))
    p.rect(0.44, 0.28, 0.56, 0.92, YELLOW)
    p.ellipse(0.24, 0.1, 0.5, 0.3, YELLOW)
    p.ellipse(0.5, 0.1, 0.76, 0.3, YELLOW)


def _star(p: Pen):
    p.poly(_star_pts(0.5, 0.54, 0.44), YELLOW)


def _heart(p: Pen):
    p.poly(_heart_pts(0.5, 0.52, 0.42), (240, 72, 110))


def _circle(p: Pen):
    p.circle(0.5, 0.5, 0.4, BLUE)


def _square(p: Pen):
    p.rect(0.12, 0.12, 0.88, 0.88, GREEN, radius=0.03)


def _triangle(p: Pen):
    p.poly([(0.5, 0.08), (0.94, 0.88), (0.06, 0.88)], ORANGE)


def _sun(p: Pen):
    for k in range(10):
        a = math.radians(36 * k)
        p.line([(0.5 + 0.32 * math.cos(a), 0.5 + 0.32 * math.sin(a)),
                (0.5 + 0.46 * math.cos(a), 0.5 + 0.46 * math.sin(a))], ORANGE, 1.8)
    p.circle(0.5, 0.5, 0.27, YELLOW)
    _face(p, 0.5, 0.5, 0.2)


def _rainbow(p: Pen):
    for k, colour in enumerate((RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)):
        r = 0.44 - 0.05 * k
        p.d.pieslice([(0.5 - r) * p.s, (0.72 - r) * p.s, (0.5 + r) * p.s, (0.72 + r) * p.s], 180, 360,
                     fill=_c(colour))
    r = 0.14
    p.d.pieslice([(0.5 - r) * p.s, (0.72 - r) * p.s, (0.5 + r) * p.s, (0.72 + r) * p.s], 180, 360,
                 fill=(0, 0, 0, 0))
    for cx in (0.13, 0.87):
        for dx, dy, r in ((-0.06, 0.0, 0.08), (0.05, -0.02, 0.09), (0.0, 0.03, 0.08)):
            p.circle(cx + dx, 0.72 + dy, r, WHITE, outline=False)


def _light(lit: str) -> Callable[[Pen], None]:
    def draw(p: Pen):
        p.rect(0.47, 0.6, 0.53, 0.98, (90, 90, 100))
        p.rect(0.28, 0.02, 0.72, 0.64, (60, 60, 72), radius=0.06)
        for k, (name, on) in enumerate((("red", RED), ("yellow", YELLOW), ("green", (60, 210, 90)))):
            if name == lit:
                p.circle(0.5, 0.13 + 0.19 * k, 0.11, tuple(int(0.5 * c + 128) for c in on), outline=False)
                p.circle(0.5, 0.13 + 0.19 * k, 0.085, on)
            else:
                p.circle(0.5, 0.13 + 0.19 * k, 0.085, tuple(int(c * 0.25) + 30 for c in on))
    return draw


def _tooth(p: Pen):
    p.poly([(0.2, 0.24), (0.36, 0.12), (0.5, 0.18), (0.64, 0.12), (0.8, 0.24), (0.78, 0.6), (0.68, 0.92),
            (0.58, 0.92), (0.5, 0.7), (0.42, 0.92), (0.32, 0.92), (0.22, 0.6)], WHITE)
    _face(p, 0.5, 0.42, 0.22)
    p.poly(_star_pts(0.78, 0.14, 0.08, 0.35, 4), YELLOW, outline=False)


def _bubble(p: Pen):
    p.circle(0.5, 0.5, 0.4, (200, 236, 255))
    p.arc(0.5, 0.5, 0.3, 0.3, 200, 260, WHITE, 1.6)
    p.circle(0.36, 0.34, 0.05, WHITE, outline=False)


def _moon(p: Pen):
    outer = [(0.5 + 0.4 * math.cos(a), 0.5 + 0.4 * math.sin(a)) for a in np.linspace(-1.9, 1.9, 30)]
    inner = [(0.32 + 0.34 * math.cos(a), 0.5 + 0.36 * math.sin(a)) for a in np.linspace(1.3, -1.3, 30)]
    p.poly(outer + inner, (255, 236, 150))


PROPS: Dict[str, Callable[[Pen], None]] = {
    "apple": _apple, "banana": _banana, "carrot": _carrot, "broccoli": _broccoli,
    "strawberry": _strawberry, "cookie": _cookie, "candy": _candy, "toothbrush": _toothbrush,
    "cup": _cup, "soap": _soap, "towel": _towel, "book": _book, "ball": _ball, "teddy": _teddy,
    "umbrella": _umbrella, "balloon": _balloon, "flower": _flower, "gift": _gift, "star": _star,
    "heart": _heart, "circle": _circle, "square": _square, "triangle": _triangle, "sun": _sun,
    "moon": _moon, "rainbow": _rainbow, "red light": _light("red"), "green light": _light("green"),
    "tooth": _tooth, "bubble": _bubble}

# How a prop is held: (size as a share of the character's height, the point
# of the drawing in the hand (unit box), turned by degrees).
HOLD: Dict[str, Tuple[float, Tuple[float, float], float]] = {
    "toothbrush": (0.3, (0.16, 0.56), 0.0), "umbrella": (0.85, (0.5, 0.9), 0.0),
    "balloon": (0.6, (0.5, 0.99), 0.0), "book": (0.3, (0.5, 0.62), 0.0),
    "rainbow": (0.3, (0.5, 0.72), 0.0), "red light": (0.3, (0.5, 0.9), 0.0),
    "green light": (0.3, (0.5, 0.9), 0.0), "towel": (0.22, (0.5, 0.3), 0.0),
    "flower": (0.22, (0.5, 0.88), 0.0), "teddy": (0.24, (0.5, 0.66), 0.0),
    "cup": (0.17, (0.5, 0.6), 0.0)}
DEFAULT_HOLD = (0.17, (0.5, 0.55), 0.0)


def hold(name: str) -> Tuple[float, Tuple[float, float], float]:
    return HOLD.get(name, DEFAULT_HOLD)


def _draw(fn: Callable[[Pen], None], size: int) -> Image.Image:
    big = Image.new("RGBA", (2 * size, 2 * size), (0, 0, 0, 0))
    fn(Pen(ImageDraw.Draw(big), 2 * size))
    return big.resize((size, size), Image.LANCZOS)


@functools.lru_cache(maxsize=256)
def prop_image(name: str, size: int, mirrored: bool = False) -> Image.Image:
    """``name`` drawn in a square of ``size`` pixels (Pillow RGBA)."""
    img = _draw(PROPS[name], max(8, int(size)))
    return img.transpose(Image.FLIP_LEFT_RIGHT) if mirrored else img


@functools.lru_cache(maxsize=128)
def prop_sprite(name: str, size: int) -> Sprite:
    return Sprite(np.asarray(prop_image(name, size)).astype(F32) / 255.0)


# --------------------------------------------------------------------------- friends

def _bird(happy: bool) -> Callable[[Pen], None]:
    def draw(p: Pen):
        blue = (90, 160, 240)
        p.line([(0.4, 0.86), (0.38, 0.97)], ORANGE, 1.2)
        p.line([(0.58, 0.86), (0.6, 0.97)], ORANGE, 1.2)
        p.poly([(0.8, 0.56), (0.98, 0.42), (0.94, 0.62)], blue)  # tail
        p.ellipse(0.12, 0.3, 0.86, 0.9, blue)
        p.ellipse(0.26, 0.56, 0.62, 0.86, (200, 226, 255), outline=False)
        p.ellipse(0.46, 0.5, 0.78, 0.7, (70, 130, 220))  # wing
        p.poly([(0.16, 0.5), (0.02, 0.56), (0.16, 0.6)], ORANGE)  # beak
        if happy:
            p.arc(0.3, 0.48, 0.05, 0.04, 200, 340, width=1.0)
            p.ellipse(0.18, 0.56, 0.28, 0.62, (255, 160, 190), outline=False)
        else:
            p.circle(0.3, 0.47, 0.04, (40, 30, 50), outline=False)
            p.line([(0.24, 0.39), (0.36, 0.42)], width=0.8)
            p.ellipse(0.26, 0.52, 0.31, 0.6, (140, 200, 255), outline=False)  # a tear
    return draw


def _bunny(happy: bool) -> Callable[[Pen], None]:
    def draw(p: Pen):
        fur, inner = (250, 248, 244), (250, 180, 200)
        droop = 0.0 if happy else 0.1
        for x, lean in ((0.38, -0.05 - droop), (0.56, 0.05 + droop)):
            p.poly([(x - 0.05, 0.3), (x - 0.04 + lean, 0.04), (x + 0.04 + lean, 0.04), (x + 0.05, 0.3)], fur)
            p.poly([(x - 0.02, 0.27), (x - 0.015 + lean, 0.08), (x + 0.015 + lean, 0.08), (x + 0.02, 0.27)],
                   inner, outline=False)
        p.ellipse(0.26, 0.56, 0.78, 0.98, fur)  # body
        p.circle(0.8, 0.82, 0.07, fur)  # tail
        p.ellipse(0.24, 0.24, 0.72, 0.64, fur)  # head
        p.ellipse(0.32, 0.86, 0.46, 0.98, fur)
        p.ellipse(0.5, 0.86, 0.64, 0.98, fur)
        p.circle(0.48, 0.48, 0.03, inner, outline=False)
        for x in (0.38, 0.58):
            if happy:
                p.arc(x, 0.42, 0.04, 0.035, 200, 340, width=0.9)
            else:
                p.circle(x, 0.42, 0.03, (40, 30, 50), outline=False)
        if happy:
            p.arc(0.48, 0.5, 0.06, 0.05, 20, 160, width=0.8)
            for x in (0.32, 0.64):
                p.ellipse(x - 0.04, 0.48, x + 0.04, 0.53, (255, 190, 210), outline=False)
        else:
            p.arc(0.48, 0.57, 0.05, 0.035, 200, 340, width=0.8)
            p.ellipse(0.34, 0.46, 0.38, 0.53, (140, 200, 255), outline=False)
    return draw


FRIENDS = {"bird": (_bird, 0.42), "bunny": (_bunny, 0.6)}  # drawing, height (share of the hero's)


def friend_kind(value: str) -> Tuple[str, bool]:
    """("bunny", happy) from "bunny" or "sad bunny"; ("", True) for none."""
    words = str(value or "").lower().split()
    kind = next((w for w in words if w in FRIENDS), "")
    return kind, "sad" not in words


@functools.lru_cache(maxsize=32)
def friend_sprite(kind: str, happy: bool, size: int) -> Sprite:
    draw, _ = FRIENDS[kind]
    img = _draw(draw(happy), max(8, int(size)))
    return Sprite(np.asarray(img).astype(F32) / 255.0)
