"""The cast of "Peeko & Friends", drawn by the program frame by frame.

Four friends live in Sunny Garden:

* **Peeko**, a little lilac sprout in yellow rain boots who wants to learn
  everything. Its two leaves show how it feels and spin like a propeller when
  it flies; every time it learns something new, a flower blooms on its sprout.
* **Lulu**, a shy white bunny in a yellow beret who loves painting.
* **Kiko**, a chatty little bluebird in round glasses who thinks he knows
  everything (and often gets it a little bit wrong).
* **Bobo**, a big, gentle bear cub with a red bow tie who is always hungry.

They share one rig: a soft round body with light and shade, a face with
shining eyes, rubber-hose arms and legs, and what makes each one itself (a
sprout, long ears, a beak and wings, round bear ears...). They walk, wave,
clap, point, brush their teeth, wash their hands, eat and read; they blink,
look around, move their mouths with their voices and turn towards where they
go. They take the same poses as :class:`cartoon.Puppet` (``x``, ``lift``,
squash and stretch, ``angle``, ``lean``) plus ``yaw``, how far they have
turned from the viewer.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import props as PR
from .showtime import Sprite

F32 = np.float32
RGB = Tuple[int, int, int]
S = 2  # drawn at twice the size, then scaled down
WHITE = (255, 255, 255)
PUPIL = (32, 22, 46)
MOUTH = (120, 34, 62)
TONGUE = (250, 128, 140)
LEAF = (118, 206, 84)
LEAF_DARK = (74, 160, 70)
STEM = (88, 168, 74)
PETAL = (255, 140, 180)
CENTRE = (255, 214, 64)


@dataclass(frozen=True)
class Design:
    """What makes a character itself (sizes in shares of its own height)."""

    name: str
    about: str
    voice: str  # macOS voice and pitch
    pitch: float
    size: float  # height compared with Peeko's
    body: RGB
    shade: RGB
    light: RGB
    belly: RGB
    line: RGB  # outlines
    hand: RGB
    iris: RGB
    cheek: RGB
    body_c: float = 0.5  # the body: centre height, half width, half height
    body_rx: float = 0.3
    body_ry: float = 0.31
    egg: float = 0.07  # narrower at the top (+) or the bottom (-)
    belly_y: float = 0.31
    belly_r: Tuple[float, float] = (0.16, 0.1)
    eye_y: float = 0.56
    eye_dx: float = 0.095
    eye_r: Tuple[float, float] = (0.062, 0.078)
    mouth_y: float = 0.41
    hip: Tuple[float, float] = (0.1, 0.24)
    foot_y: float = 0.045
    limb: float = 0.05  # arm and leg thickness
    top: str = "sprout"  # sprout, ears, bear, crest
    face: str = ""  # bunny (nose, teeth), bear (muzzle, nose), beak
    feet: str = "boots"  # boots, paws, bird
    extra: str = ""  # beret, glasses, bowtie
    tail: str = ""  # cotton, bear, feathers
    accent: RGB = (255, 202, 56)  # boots, beret, bow tie...
    hover: float = 0.0  # flies this high when it stands still (share of height)
    lashes: bool = False


PEEKO = Design(
    "Peeko", "a little lilac sprout in yellow rain boots who wants to learn everything about the world. "
    "Its two leaves spin like a propeller so it can fly, and every time it learns something new, a flower "
    "blooms on its sprout", "Samantha", 1.25, 1.0,
    body=(176, 150, 242), shade=(138, 110, 214), light=(212, 198, 255), belly=(232, 225, 255),
    line=(66, 44, 116), hand=(196, 176, 250), iris=(112, 76, 170), cheek=(246, 156, 206),
    accent=(255, 202, 56))
LULU = Design(
    "Lulu", "a shy white bunny in a yellow beret who loves painting and colours", "Karen", 1.32, 0.98,
    body=(255, 246, 248), shade=(232, 206, 220), light=(255, 255, 255), belly=(255, 233, 242),
    line=(140, 86, 118), hand=(255, 236, 242), iris=(84, 168, 226), cheek=(255, 170, 200),
    body_c=0.35, body_rx=0.25, body_ry=0.26, egg=0.08, belly_y=0.24, belly_r=(0.13, 0.08),
    eye_y=0.42, eye_dx=0.085, eye_r=(0.056, 0.07), mouth_y=0.315, hip=(0.085, 0.14), foot_y=0.035,
    limb=0.045, top="ears", face="bunny", feet="paws", extra="beret", tail="cotton", accent=(255, 196, 60),
    lashes=True)
KIKO = Design(
    "Kiko", "a chatty little bluebird in round glasses who thinks he knows everything, and is often a little "
    "bit wrong", "Tessa", 1.5, 0.6,
    body=(96, 172, 252), shade=(58, 126, 222), light=(150, 206, 255), belly=(255, 238, 190),
    line=(30, 66, 138), hand=(96, 172, 252), iris=(60, 46, 40), cheek=(255, 160, 190),
    body_c=0.56, body_rx=0.36, body_ry=0.36, egg=0.0, belly_y=0.45, belly_r=(0.2, 0.17),
    eye_y=0.64, eye_dx=0.13, eye_r=(0.085, 0.1), mouth_y=0.5, hip=(0.1, 0.22), foot_y=0.03,
    limb=0.07, top="crest", face="beak", feet="bird", extra="glasses", tail="feathers",
    accent=(255, 160, 50), hover=0.22)
BOBO = Design(
    "Bobo", "a big, gentle bear cub with a red bow tie who is always hungry", "Rishi", 0.84, 1.14,
    body=(190, 132, 86), shade=(150, 98, 60), light=(214, 162, 116), belly=(242, 214, 172),
    line=(86, 52, 32), hand=(176, 120, 76), iris=(70, 44, 28), cheek=(236, 136, 120),
    body_c=0.46, body_rx=0.37, body_ry=0.37, egg=-0.05, belly_y=0.26, belly_r=(0.2, 0.13),
    eye_y=0.6, eye_dx=0.11, eye_r=(0.05, 0.062), mouth_y=0.44, hip=(0.13, 0.15), foot_y=0.04,
    limb=0.07, top="bear", face="bear", feet="paws", extra="bowtie", tail="bear", accent=(226, 58, 60))

CAST: Dict[str, Design] = {d.name.lower(): d for d in (PEEKO, LULU, KIKO, BOBO)}
NAME = PEEKO.name
ABOUT = f"{PEEKO.name} is {PEEKO.about}."
FRIENDS = ("Lulu", "Kiko", "Bobo")


def design(name: str) -> Optional[Design]:
    return CAST.get(str(name or "").strip().lower())


@dataclass
class Act:
    """What the character is doing, for the parts a pose does not say."""

    action: str = "idle"
    u: float = 0.0  # seconds into the action
    seconds: float = 1.0
    t: float = 0.0  # seconds into the scene (blinking, small wiggles)
    level: float = 0.0  # how loud its own voice is now, 0..1
    holds: str = ""  # a prop in its hand
    bloom: float = 0.0  # the flower on Peeko's sprout, 0..1


# --------------------------------------------------------------------------- poses

# The pose targets are worked out for Peeko's body, then fitted to each design.
P_SHOULDER = (0.26, 0.46)
P_FOOT_Y = 0.045
P_MOUTH_Y = 0.41


@dataclass
class _Look:
    """Where the limbs are and how the face looks (forward coordinates)."""

    hands: List[Tuple[float, float]]  # back, front
    feet: List[Tuple[float, float]]
    eyes: str = "open"
    gaze: Tuple[float, float] = (0.0, 0.0)
    brows: str = ""
    mouth: str = "smile"
    open: float = 0.0
    leaves: float = 30.0  # degrees above the horizontal (ears and crest follow it too)
    propeller: Optional[float] = None
    tip: float = 0.0  # the sprout bends forward (+) or back
    prop_at: Optional[Tuple[float, float]] = None
    prop_flip: bool = False
    behind: bool = False  # the back arm is behind the body
    flap: float = 0.0  # wings


def _mix(a: Tuple[float, float], b: Tuple[float, float], k: float) -> Tuple[float, float]:
    return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _blinking(t: float, offset: float = 0.0) -> bool:
    """A blink every three or four seconds, at uneven times."""
    t += offset
    period = 3.3
    k = int(t // period)
    at = (k * 0.37) % 1.0 * 1.5
    return at <= t - k * period < at + 0.13


def blink_offset(d: Design) -> float:
    """Each character blinks at its own times."""
    return sum(map(ord, d.name)) % 7 * 0.41


def _pose_look(act: Act, yaw: float, comp: float, d: Design = PEEKO) -> _Look:
    """The limbs and face for ``act`` (worked out for Peeko, fitted to ``d``)."""
    a, u, dur, t, level = act.action, act.u, act.seconds, act.t, act.level
    p = u / max(dur, 1e-6)
    sway = 0.012 * math.sin(t * 2.1)
    rest = [(-0.37, 0.27 + sway), (0.37, 0.27 - sway)]
    look = _Look(hands=list(rest), feet=[(-0.12, P_FOOT_Y), (0.12, P_FOOT_Y)])
    face_x = 0.15 * math.sin(yaw)
    mouth_y = 0.5 + (d.mouth_y - d.body_c) * 0.31 / d.body_ry
    mouth_x = face_x * 1.05

    if a in ("walk", "run", "enter", "exit"):
        fast = a in ("run", "exit")
        f = 3.4 if fast else 2.2
        phase = math.pi * f * u + math.pi / 2
        stride, up = (0.14, 0.09) if fast else (0.1, 0.06)
        for i, side in enumerate((-1, 1)):
            q = phase + i * math.pi
            look.feet[i] = (0.06 * side + stride * math.sin(q), P_FOOT_Y + up * max(0.0, math.cos(q)) - comp)
            swing = -(0.11 if fast else 0.07) * math.sin(q)
            look.hands[i] = (0.33 * side + swing, (0.36 if fast else 0.27) + 0.02 * abs(math.sin(q)))
        look.mouth = "grin" if fast else "smile"
        look.gaze = (0.7, 0.0)
        look.leaves, look.tip = (12.0, -0.05) if fast else (24.0, -0.025)
        look.flap = 0.5 + 0.5 * math.sin(u * 14)
    elif a == "jump":
        crouch = [(-0.34, 0.2), (0.34, 0.2)], [(-0.15, P_FOOT_Y), (0.15, P_FOOT_Y)]
        air = [(-0.36, 0.8), (0.36, 0.8)], [(-0.1, P_FOOT_Y + 0.06), (0.1, P_FOOT_Y + 0.06)]
        land = [(-0.42, 0.44), (0.42, 0.44)], [(-0.15, P_FOOT_Y), (0.15, P_FOOT_Y)]
        if p < 0.2:
            k, (h0, f0), (h1, f1) = _smooth(p / 0.2), (rest, look.feet), crouch
        elif p < 0.5:
            k, (h0, f0), (h1, f1) = _smooth((p - 0.2) / 0.15), crouch, air
        elif p < 0.78:
            k, (h0, f0), (h1, f1) = _smooth((p - 0.5) / 0.28), air, land
        else:
            k, (h0, f0), (h1, f1) = _smooth((p - 0.78) / 0.22), land, (rest, look.feet)
        look.hands = [_mix(h0[i], h1[i], k) for i in range(2)]
        look.feet = [_mix(f0[i], f1[i], k) for i in range(2)]
        look.eyes = "happy" if 0.25 < p < 0.8 else "open"
        look.mouth = "grin"
        look.leaves = 30 + 30 * math.sin(math.pi * min(p / 0.8, 1.0))
        look.flap = 1.0 if 0.2 < p < 0.8 else 0.0
    elif a == "happy":
        for i, side in enumerate((-1, 1)):
            look.hands[i] = (0.33 * side + 0.05 * math.sin(t * 10 + i), 0.8 + 0.03 * math.sin(t * 10 + 2 * i))
        look.eyes, look.mouth, look.leaves = "happy", "grin", 50 + 8 * math.sin(t * 9)
        look.flap = 0.5 + 0.5 * math.sin(t * 16)
    elif a == "wave":
        k = _smooth(u / 0.3)
        look.hands[1] = _mix(rest[1], (0.44 + 0.07 * math.sin(u * 11), 0.76), k)
        look.mouth, look.leaves = "grin", 42 + 6 * math.sin(u * 11)
    elif a == "dance":
        beat = u * 2.4
        k = 0.5 - 0.5 * math.cos(math.pi * beat)
        up, down = (0.36, 0.8), (0.44, 0.34)
        swap = int(beat) % 2
        for i, side in enumerate((-1, 1)):
            hi, lo = (side * up[0], up[1]), (side * down[0], down[1])
            look.hands[i] = _mix(hi, lo, k) if (i + swap) % 2 else _mix(lo, hi, k)
            look.feet[i] = (0.12 * side, P_FOOT_Y + 0.05 * max(0.0, math.sin(math.pi * (beat + i))))
        look.eyes, look.mouth, look.leaves = "happy", "grin", 45 + 15 * math.sin(math.pi * beat)
        look.flap = k
    elif a == "sad":
        k = _smooth(u / 0.6)
        look.hands = [_mix(rest[i], (0.2 * side, 0.18), k) for i, side in enumerate((-1, 1))]
        look.eyes, look.brows, look.mouth = "sad", "sad", "frown"
        look.gaze = (0.0, -0.7)
        look.leaves = 30 - 60 * k
    elif a == "scared":
        jitter = 0.008 * math.sin(u * 70)
        look.hands = [(0.25 * side + jitter, 0.43 + jitter) for side in (-1, 1)]
        look.feet = [(-0.08, P_FOOT_Y), (0.08, P_FOOT_Y)]
        look.eyes, look.brows, look.mouth = "wide", "sad", "wobble"
        look.leaves = 72 + 8 * math.sin(u * 60)
    elif a == "surprised":
        k = _smooth(u / 0.2)
        look.hands = [_mix(rest[i], (0.47 * side, 0.66), k) for i, side in enumerate((-1, 1))]
        look.eyes, look.brows, look.mouth, look.leaves = "wide", "up", "o", 30 + 45 * k
        look.flap = k
    elif a == "think":
        k = _smooth(u / 0.4)
        look.hands[1] = _mix(rest[1], (face_x + 0.06, mouth_y - 0.05), k)
        look.hands[0] = _mix(rest[0], (-0.32, 0.31), k)
        look.gaze = (0.5, 0.9)
        look.brows, look.mouth = "think", "side"
        look.tip = 0.03 * math.sin(u * 1.5)
    elif a == "sleep":
        k = _smooth(u / 0.8)
        look.hands = [_mix(rest[i], (0.3 * side, 0.2), k) for i, side in enumerate((-1, 1))]
        look.eyes, look.mouth = "closed", "snore"
        look.open = 0.5 + 0.5 * math.sin(u * 2.2)
        look.leaves = 30 - 35 * k
    elif a == "float":
        for i, side in enumerate((-1, 1)):
            look.hands[i] = (0.47 * side, 0.52 + 0.05 * math.sin(t * 3 + i * math.pi))
            look.feet[i] = (0.08 * side + 0.02 * math.sin(t * 2 + i), P_FOOT_Y + 0.02)
        look.eyes, look.mouth = "happy", "grin"
        look.propeller = t * 2 * math.pi * 3.2
        look.flap = 0.5 + 0.5 * math.sin(t * 18)
    elif a == "spin":
        look.hands = [(0.46 * side, 0.56) for side in (-1, 1)]
        look.eyes, look.mouth, look.leaves = "happy", "grin", 40.0
    elif a == "talk":
        look.hands[1] = (0.38 + 0.04 * math.sin(u * 3), 0.33 + 0.12 * level + 0.03 * math.sin(u * 5))
        look.mouth, look.open = "talk", level
        look.flap = 0.3 * level
    elif a == "clap":
        apart = 0.03 + 0.14 * abs(math.cos(math.pi * 2.5 * u))
        k = _smooth(u / 0.2)
        cx = face_x * 0.5
        look.hands = [_mix(rest[0], (cx - apart, 0.47), k), _mix(rest[1], (cx + apart, 0.47), k)]
        look.eyes, look.mouth, look.leaves = "happy", "grin", 48 + 6 * math.sin(u * 16)
    elif a in ("point", "count"):
        k = _smooth(u / 0.3)
        look.hands[1] = _mix(rest[1], (0.5, 0.64 + 0.01 * math.sin(u * 4)), k)
        look.hands[0] = _mix(rest[0], (-0.32, 0.31), k)
        look.gaze = (0.8, 0.5)
        look.mouth, look.leaves = ("grin", 45.0) if a == "count" else ("smile", 35.0)
    elif a == "brush":
        k = _smooth(u / 0.3)
        look.hands[1] = _mix(rest[1], (mouth_x + 0.17 + 0.025 * math.sin(u * 16), mouth_y + 0.015), k)
        look.mouth, look.eyes = "teeth", ("happy" if int(u * 1.5) % 3 == 2 else "open")
        look.prop_flip = True
    elif a == "wash":
        cx = face_x * 0.5
        for i, side in enumerate((-1, 1)):
            w = 9 * u + i * math.pi
            look.hands[i] = (cx + side * (0.06 + 0.03 * math.cos(w)), 0.3 + 0.03 * math.sin(w))
        look.eyes, look.mouth = "happy", "grin"
        look.gaze = (0.0, -0.6)
    elif a == "eat":
        cycle = (u % 1.1) / 1.1
        k = math.sin(math.pi * cycle) ** 2
        look.hands[1] = _mix(rest[1], (mouth_x + 0.09, mouth_y + 0.04), k)
        look.mouth = "o" if k > 0.75 else "chew"
        look.open = 0.5 + 0.5 * math.sin(u * 13)
        look.eyes = "happy" if k < 0.4 and u > 0.6 else "open"
    elif a == "read":
        k = _smooth(u / 0.4)
        cx = face_x * 0.6
        look.hands = [_mix(rest[0], (cx - 0.13, 0.31), k), _mix(rest[1], (cx + 0.13, 0.31), k)]
        look.prop_at = (cx, 0.33)
        look.gaze = (0.1 * math.sin(u * 1.2), -0.9)
    elif a == "yawn":
        k = math.sin(math.pi * p) ** 0.7
        look.hands = [_mix(rest[i], (0.3 * side, 0.9), k) for i, side in enumerate((-1, 1))]
        look.eyes = "closed" if k > 0.35 else "open"
        look.mouth, look.open = "o", 1.6 * k
        look.leaves = 30 + 30 * k
    elif a == "angry":
        jitter = 0.006 * math.sin(u * 50)
        look.hands = [(0.36 * side + jitter, 0.2 + jitter) for side in (-1, 1)]
        look.feet[int(u * 3) % 2] = (look.feet[int(u * 3) % 2][0], P_FOOT_Y + 0.03 * abs(math.sin(math.pi * 3 * u)))
        look.brows, look.mouth = "angry", "grit"
        look.leaves = 78 + 6 * math.sin(u * 40)
        look.flap = abs(math.sin(u * 12))
    elif a == "listen":
        look.gaze = (0.7, 0.1)
        look.hands[0] = _mix(rest[0], (-0.3, 0.3), 0.5)
    else:  # idle, look
        look.gaze = (0.6, 0.0) if a == "look" else (0.25 * math.sin(t * 0.7), 0.0)
    if level > 0.1 and look.mouth not in ("talk", "o", "teeth", "chew", "snore"):
        look.mouth, look.open = "talk", level  # talking while it does something else
    if act.holds == "umbrella" and a in ("idle", "walk", "run", "enter", "exit", "look", "talk", "sad",
                                         "think", "surprised", "listen"):
        look.hands[1] = (0.16, 0.6)
    look.behind = math.sin(yaw) > 0.45 and a not in ("clap", "wash", "read")
    # fit to the design
    kx, ky = d.body_rx / 0.3, d.body_ry / 0.31
    fit = lambda pt: (pt[0] * kx, d.body_c + (pt[1] - 0.5) * ky)  # noqa: E731
    look.hands = [fit(h) for h in look.hands]
    if look.prop_at is not None:
        look.prop_at = fit(look.prop_at)
    leg = (d.hip[1] - d.foot_y) / (0.24 - P_FOOT_Y)
    look.feet = [(f[0] * d.hip[0] / 0.1, d.foot_y + (f[1] - P_FOOT_Y) * leg) for f in look.feet]
    return look


# --------------------------------------------------------------------------- drawing

class _Pen:
    """Collects shapes in forward coordinates (shares of the height), then
    draws them on a small canvas around the character."""

    def __init__(self, L: np.ndarray, origin: np.ndarray, unit: float, heading: int, line: RGB):
        self.L, self.o, self.unit, self.heading, self.line = L, origin, unit, heading, line
        self.ops: list = []
        self.ow = max(1.5, 0.012 * unit)  # outline width, pixels

    def T(self, pts) -> np.ndarray:
        return np.asarray(pts, np.float64).reshape(-1, 2) @ self.L.T + self.o

    def shape(self, pts, fill, outline: bool = True) -> None:
        self.ops.append(("poly", self.T(pts), fill, self.ow if outline else 0.0, self.line))

    def stroke(self, pts, colour, width: float, closed: bool = False) -> None:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        self.ops.append(("line", self.T(pts), colour, max(1.0, width * self.unit)))

    def dot(self, x: float, y: float, r: float, fill, outline: bool = True) -> None:
        self.ellipse(x, y, r, r, fill, outline)

    @staticmethod
    def oval(x, y, rx, ry, n: int = 36, turn: float = 0.0) -> np.ndarray:
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        px, py = rx * np.cos(a), ry * np.sin(a)
        c, s = math.cos(turn), math.sin(turn)
        return np.stack([x + px * c - py * s, y + px * s + py * c], axis=1)

    def ellipse(self, x, y, rx, ry, fill, outline: bool = True, n: int = 36, turn: float = 0.0) -> None:
        self.shape(self.oval(x, y, rx, ry, n, turn), fill, outline)

    def hose(self, a, b, bend: float, width: float, fill) -> None:
        """A rubber-hose limb from ``a`` to ``b``, bowed sideways by ``bend``."""
        a, b = np.array(a, float), np.array(b, float)
        d = b - a
        c = (a + b) / 2 + np.array([-d[1], d[0]]) * bend
        s = np.linspace(0, 1, 14)[:, None]
        pts = (1 - s) ** 2 * a + 2 * (1 - s) * s * c + s ** 2 * b
        w = width * self.unit
        self.ops.append(("line", self.T(pts), self.line, w + 2 * self.ow))
        self.ops.append(("line", self.T(pts), fill, w))
        self.dot(a[0], a[1], (w / 2 + self.ow) / self.unit, fill, outline=False)

    def image(self, img: Image.Image, at: Tuple[float, float], grip: Tuple[float, float], angle: float) -> None:
        self.ops.append(("img", img, self.T([at])[0], grip, angle))

    def blit(self, frame: np.ndarray, alpha: float) -> None:
        H, W = frame.shape[:2]
        boxes = []
        for op in self.ops:
            if op[0] == "img":
                r = max(op[1].size) / S * 1.5
                boxes.append(np.array([op[2] - r, op[2] + r]))
            else:
                m = op[3] + 2
                boxes.append(np.array([op[1].min(axis=0) - m, op[1].max(axis=0) + m]))
        lo = np.floor(np.min([b[0] for b in boxes], axis=0)).astype(int)
        hi = np.ceil(np.max([b[1] for b in boxes], axis=0)).astype(int)
        x0, y0 = max(0, lo[0]), max(0, lo[1])
        x1, y1 = min(W, hi[0]), min(H, hi[1])
        if x1 <= x0 or y1 <= y0:
            return
        img = Image.new("RGBA", ((x1 - x0) * S, (y1 - y0) * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        off = np.array([x0, y0], float)
        for op in self.ops:
            if op[0] == "poly":
                pts = [tuple(p) for p in (op[1] - off) * S]
                d.polygon(pts, fill=tuple(op[2]) + (255,))
                if op[3]:
                    d.line(pts + [pts[0]], fill=tuple(op[4]) + (255,), width=max(1, int(round(op[3] * S))),
                           joint="curve")
            elif op[0] == "line":
                pts = [tuple(p) for p in (op[1] - off) * S]
                width = max(1, int(round(op[3] * S)))
                d.line(pts, fill=tuple(op[2]) + (255,), width=width, joint="curve")
                for p in (pts[0], pts[-1]):  # round ends
                    r = width / 2
                    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=tuple(op[2]) + (255,))
            else:
                _, prop, at, grip, angle = op
                turned = prop.rotate(angle, resample=Image.BICUBIC, expand=True)
                gx, gy = (grip[0] - 0.5) * prop.width, (grip[1] - 0.5) * prop.height
                a = math.radians(angle)
                rx, ry = gx * math.cos(a) + gy * math.sin(a), -gx * math.sin(a) + gy * math.cos(a)
                px = (at[0] - x0) * S - turned.width / 2 - rx
                py = (at[1] - y0) * S - turned.height / 2 - ry
                img.alpha_composite(turned, (int(round(px)), int(round(py))))
        small = np.asarray(img.reduce(S)).astype(F32) / 255.0
        small = small[:y1 - y0, :x1 - x0]
        a = small[..., 3:] * alpha
        region = frame[y0:y0 + small.shape[0], x0:x0 + small.shape[1]]
        region *= 1.0 - a
        region += small[..., :3] * a


def _leaf(pen: _Pen, base, angle: float, length: float, width: float = 0.065, fill=LEAF, vein=STEM) -> None:
    a = math.radians(angle)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    s = np.linspace(0, 1, 16)
    half = width / 2 * np.sin(np.pi * s) ** 0.8
    base = np.array(base)
    side1 = base + np.outer(s * length, d) + np.outer(half, n)
    side2 = base + np.outer(s * length, d) - np.outer(half, n)
    pen.shape(np.vstack([side1, side2[::-1]]), fill)
    if abs(length) > 0.03 and vein:
        pen.stroke([base, base + d * length * 0.75], vein, 0.008)


def _mixc(a: RGB, b: RGB, k: float) -> RGB:
    return tuple(int(round(x + (y - x) * k)) for x, y in zip(a, b))


def _eye(pen: _Pen, d: Design, x: float, y: float, style: str, gaze, inner: int, look_x: float) -> None:
    rx, ry = d.eye_r
    if style in ("happy", "closed"):
        a = np.linspace(math.radians(20), math.radians(160), 12) if style == "happy" else \
            np.linspace(math.radians(200), math.radians(340), 12)
        cy = y - 0.01 if style == "happy" else y + 0.02
        pen.stroke(np.stack([x + rx * np.cos(a), cy + ry * 0.55 * np.sin(a)], axis=1), PUPIL, 0.02)
        if d.lashes and style == "closed":
            pen.stroke([(x - inner * rx * 0.9, cy - 0.01), (x - inner * rx * 1.25, cy - 0.03)], PUPIL, 0.012)
        return
    k = 1.12 if style == "wide" else 1.0
    pen.ellipse(x, y, rx * k, ry * k, WHITE)
    small = 0.7 if style == "wide" else 1.0
    px = x + (gaze[0] * 0.18 * rx + look_x * 0.25 * rx)
    py = y + gaze[1] * 0.2 * ry - 0.04 * ry
    pen.ellipse(px, py, 0.7 * rx * small, 0.76 * ry * small, d.iris, outline=False)
    pen.ellipse(px, py, 0.42 * rx * small, 0.48 * ry * small, PUPIL, outline=False)
    pen.dot(px + 0.26 * rx * small, py + 0.3 * ry * small, 0.24 * rx * small, WHITE, outline=False)
    pen.dot(px - 0.22 * rx * small, py - 0.32 * ry * small, 0.11 * rx * small, WHITE, outline=False)
    if style == "sad":  # a heavy upper lid, lower on the outside
        outer = -inner
        lid = [(x + outer * rx * 1.15, y + ry * 1.15), (x + outer * rx * 1.15, y + 0.2 * ry),
               (x + inner * rx * 1.15, y + 0.6 * ry), (x + inner * rx * 1.15, y + ry * 1.15)]
        pen.shape(lid, d.body, outline=False)
        pen.stroke([(x + outer * rx, y + 0.2 * ry), (x + inner * rx, y + 0.6 * ry)], pen.line, 0.012)
    if d.lashes:
        outer = -inner
        for k2 in (0, 1):
            ang = math.radians(60 + 25 * k2)
            bx, by = x + outer * rx * math.cos(ang), y + ry * math.sin(ang)
            pen.stroke([(bx, by), (bx + outer * 0.03, by + 0.025)], PUPIL, 0.01)


def _brow(pen: _Pen, d: Design, x: float, style: str, inner: int, front: bool) -> None:
    if not style:
        return
    yy = d.eye_y + d.eye_r[1] + 0.025
    w = d.eye_r[0] * 0.8
    if style == "sad":
        a, b = (x + inner * w, yy + 0.03), (x - inner * w, yy - 0.005)
    elif style == "up":
        a, b = (x - w, yy + 0.035), (x + w, yy + 0.035)
    elif style == "think":
        a, b = ((x - w, yy + 0.04), (x + w, yy + 0.05)) if front else ((x - w, yy), (x + w, yy - 0.005))
    elif style == "angry":
        a, b = (x + inner * w, yy - 0.012), (x - inner * w, yy + 0.022)
    else:
        return
    pen.stroke([a, b], pen.line, 0.016)


def _mouth(pen: _Pen, x: float, y: float, style: str, amount: float, scale: float = 1.0) -> None:
    def arc(cx, cy, rx, ry, a0, a1, n=14):
        a = np.linspace(math.radians(a0), math.radians(a1), n)
        return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)

    k = scale
    if style == "talk" and amount < 0.12:
        style = "smile"
    if style == "smile":
        pen.stroke(arc(x, y + 0.025 * k, 0.05 * k, 0.035 * k, 200, 340), pen.line, 0.014)
    elif style in ("grin", "teeth"):
        top = [(x - 0.062 * k, y + 0.012 * k), (x + 0.062 * k, y + 0.012 * k)]
        bottom = arc(x, y + 0.012 * k, 0.062 * k, 0.058 * k, 0, -180, 18)
        pen.shape(np.vstack([top, bottom]), MOUTH)
        pen.shape(arc(x, y - 0.026 * k, 0.032 * k, 0.016 * k, 0, -180, 10), TONGUE, outline=False)
        if style == "teeth":
            pen.shape([(x - 0.052 * k, y + 0.008 * k), (x + 0.052 * k, y + 0.008 * k), (x + 0.046 * k, y - 0.012 * k),
                       (x - 0.046 * k, y - 0.012 * k)], WHITE, outline=False)
    elif style == "talk":
        ry = (0.012 + 0.042 * min(amount, 1.0)) * k
        pen.ellipse(x, y, 0.042 * k * (1 - 0.15 * amount), ry, MOUTH)
        if ry > 0.025 * k:
            pen.ellipse(x, y - ry * 0.55, 0.024 * k, ry * 0.32, TONGUE, outline=False)
    elif style in ("o", "snore"):
        r = (0.02 + 0.03 * min(amount, 2.0) * (1.0 if style == "o" else 0.3)) * k
        pen.ellipse(x, y - 0.005 * k, r * 0.85, r, MOUTH)
    elif style == "frown":
        pen.stroke(arc(x, y - 0.03 * k, 0.045 * k, 0.03 * k, 20, 160), pen.line, 0.014)
    elif style == "wobble":
        xs = np.linspace(x - 0.05 * k, x + 0.05 * k, 7)
        ys = y + 0.008 * k * np.array([1, -1, 1, -1, 1, -1, 1])
        pen.stroke(np.stack([xs, ys], axis=1), pen.line, 0.012)
    elif style == "side":
        pen.stroke([(x - 0.015 * k, y + 0.005 * k), (x + 0.04 * k, y + 0.016 * k)], pen.line, 0.014)
    elif style == "grit":  # clenched teeth
        pen.shape([(x - 0.05 * k, y + 0.014 * k), (x + 0.05 * k, y + 0.014 * k), (x + 0.045 * k, y - 0.016 * k),
                   (x - 0.045 * k, y - 0.016 * k)], WHITE)
        for dx in (-0.017, 0.017):
            pen.stroke([(x + dx * k, y + 0.012 * k), (x + dx * k, y - 0.014 * k)], pen.line, 0.007)
        pen.stroke([(x - 0.046 * k, y - 0.001 * k), (x + 0.046 * k, y - 0.001 * k)], pen.line, 0.007)
    elif style == "chew":
        pen.stroke(arc(x, y + 0.02 * k, 0.035 * k, (0.012 + 0.02 * amount) * k, 200, 340), pen.line, 0.014)


def _beak(pen: _Pen, d: Design, x: float, y: float, style: str, amount: float) -> None:
    """Kiko's beak: it opens to talk, to eat and when surprised."""
    gap = {"talk": 0.05 * min(amount, 1.0) if amount > 0.12 else 0.0, "grin": 0.03, "o": 0.045, "teeth": 0.03,
           "grit": 0.0,
           "chew": 0.02 * amount, "snore": 0.01, "wobble": 0.012}.get(style, 0.0)
    orange, dark = d.accent, _mixc(d.accent, (120, 50, 20), 0.35)
    if gap > 0.004:
        pen.ellipse(x, y - 0.01, 0.045, 0.012 + gap * 0.6, MOUTH, outline=False)
    pen.shape([(x - 0.06, y + 0.012), (x + 0.06, y + 0.012), (x, y - 0.035 + 0.01)], orange)
    pen.shape([(x - 0.045, y - 0.005 - gap), (x + 0.045, y - 0.005 - gap), (x, y - 0.06 - gap)], dark)
    if style == "frown":
        pen.stroke([(x - 0.07, y - 0.035), (x - 0.05, y - 0.045)], pen.line, 0.01)


def _arm(pen: _Pen, d: Design, i: int, look: _Look) -> None:
    side = -1 if i == 0 else 1
    kx, ky = d.body_rx / 0.3, d.body_ry / 0.31
    sh = (P_SHOULDER[0] * side * kx, d.body_c + (P_SHOULDER[1] - 0.5) * ky)
    hand = look.hands[i]
    if d.feet == "bird":  # wings: short, flapping
        tip = (hand[0], hand[1] + 0.08 * look.flap)
        reach = (tip[0] - sh[0], tip[1] - sh[1])
        length = math.hypot(*reach)
        ang = math.degrees(math.atan2(reach[1], reach[0]))
        _leaf(pen, (sh[0] - 0.02 * side, sh[1]), ang, max(0.2, min(length + 0.06, 0.34)), 0.16, d.shade, None)
        _leaf(pen, (sh[0] - 0.02 * side, sh[1] + 0.01), ang + 12 * side, max(0.16, min(length, 0.28)), 0.11, d.body,
              None)
        return
    reach = math.hypot(hand[0] - sh[0], hand[1] - sh[1])
    bend = 0.28 * max(0.0, 1.0 - reach / 0.36) + 0.06
    dd = (hand[0] - sh[0], hand[1] - sh[1])
    n = (-dd[1], dd[0])
    if n[0] * side - 0.6 * n[1] < 0:
        bend = -bend
    pen.hose(sh, hand, bend, d.limb, d.body)
    pen.dot(hand[0], hand[1], d.limb * 0.95, d.hand)
    if d.feet == "paws":
        pen.dot(hand[0] + 0.012 * side, hand[1] - 0.008, d.limb * 0.42, _mixc(d.hand, (255, 255, 255), 0.45), False)


def _feet(pen: _Pen, d: Design, look: _Look, sy: float, cy: float) -> None:
    for i, side in enumerate((-1, 1)):
        hip = (d.hip[0] * side * (0.6 + 0.4 * abs(cy)), d.hip[1])
        foot = look.feet[i]
        toe = 0.014 * (side * (1 - abs(sy)) + sy)
        if d.feet == "bird":
            pen.hose(hip, (foot[0], foot[1] + 0.03), 0.04 * side, 0.025, d.accent)
            for ang in (-40, 0, 40):
                a = math.radians(ang)
                pen.stroke([(foot[0], foot[1] + 0.03), (foot[0] + 0.06 * math.sin(a) + toe, foot[1] + 0.005)],
                           d.accent, 0.022)
            continue
        pen.hose(hip, (foot[0], foot[1] + 0.02), 0.08 * side, d.limb, d.body)
        if d.feet == "boots":
            pen.ellipse(foot[0] + toe, foot[1], 0.068, 0.044, d.accent)
            pen.stroke([(foot[0] + toe - 0.05, foot[1] - 0.03), (foot[0] + toe + 0.05, foot[1] - 0.03)],
                       _mixc(d.accent, (160, 80, 10), 0.45), 0.014)
            pen.shape([(foot[0] - 0.044, foot[1] + 0.02), (foot[0] + 0.044, foot[1] + 0.02),
                       (foot[0] + 0.04, foot[1] + 0.075), (foot[0] - 0.04, foot[1] + 0.075)], d.accent)
            pen.ellipse(foot[0] + toe - 0.025, foot[1] + 0.012, 0.014, 0.009, (255, 240, 190), outline=False)
        else:
            pen.ellipse(foot[0] + toe, foot[1], 0.07 * d.limb / 0.05, 0.042 * d.limb / 0.05, d.hand)
            pen.ellipse(foot[0] + toe, foot[1] - 0.006, 0.03 * d.limb / 0.05, 0.018 * d.limb / 0.05,
                        _mixc(d.hand, (255, 230, 220), 0.5), outline=False)


def _top_behind(pen: _Pen, d: Design, look: _Look, act: Act, sy: float) -> None:
    """What sticks out of the top of the head, drawn before the body."""
    top = d.body_c + d.body_ry
    wiggle = 6 * math.sin(act.t * 2.3)
    if d.top == "sprout":
        tip = (look.tip + 0.012 * math.sin(act.t * 1.7), top + 0.09)
        stem = [(0.0, top - 0.04), (tip[0] * 0.4 + 0.008, top + 0.04), tip]
        s = np.linspace(0, 1, 10)[:, None]
        a, c, b = (np.array(p) for p in stem)
        pts = (1 - s) ** 2 * a + 2 * (1 - s) * s * c + s ** 2 * b
        pen.stroke(pts, pen.line, 0.024 + 2 * pen.ow / pen.unit)
        pen.stroke(pts, STEM, 0.024)
        if look.propeller is not None:
            for k in (0, 1):
                length = 0.13 * math.cos(look.propeller + k * math.pi)
                _leaf(pen, tip, 8.0 if length >= 0 else 172.0, max(abs(length), 0.04))
            ring = np.linspace(0, 2 * math.pi, 40)
            pen.stroke(np.stack([tip[0] + 0.15 * np.cos(ring), tip[1] + 0.015 + 0.028 * np.sin(ring)], axis=1),
                       (150, 210, 130), 0.009)
        else:
            _leaf(pen, tip, 180.0 - look.leaves + wiggle, 0.13)
            _leaf(pen, tip, look.leaves + 6 * math.sin(act.t * 2.3 + 1.0), 0.13)
        if act.bloom > 0.01:
            k = act.bloom
            c0 = (tip[0], tip[1] + 0.03 * k)
            for j in range(5):
                ang = math.radians(90 + 72 * j + 40 * act.t)
                pen.dot(c0[0] + 0.032 * k * math.cos(ang), c0[1] + 0.032 * k * math.sin(ang), 0.024 * k, PETAL)
            pen.dot(c0[0], c0[1], 0.018 * k, CENTRE)
    elif d.top == "ears":  # long bunny ears: up when happy, down when sad
        perk = (look.leaves - 30) / 40  # -1.5 .. 1
        for side in (-1, 1):
            base = (0.075 * side + 0.25 * math.sin(look.tip) + sy * 0.03, top - 0.05)
            lean = 90 - side * (10 - 8 * perk) - side * max(0.0, -perk) * 55 + (wiggle if side > 0 else -wiggle) * 0.5
            if side > 0 and perk > -0.2:
                lean -= 14  # one ear flops a little: Lulu's look
            _leaf(pen, base, lean, 0.4, 0.13, d.body, None)
            a = math.radians(lean)
            inner = (base[0] + math.cos(a) * 0.04, base[1] + math.sin(a) * 0.04)
            _leaf_plain(pen, inner, lean, 0.31, 0.065, _mixc(d.cheek, (255, 255, 255), 0.25))
    elif d.top == "bear":
        for side in (-1, 1):
            cx, cy = 0.25 * side * d.body_rx / 0.37 + sy * 0.03, top - 0.06 + 0.01 * (look.leaves - 30) / 30
            pen.dot(cx, cy, 0.085, d.body)
            pen.dot(cx, cy - 0.005, 0.048, _mixc(d.belly, d.cheek, 0.25), outline=False)
    elif d.top == "crest":
        for k, ang in enumerate((105, 80, 128)):
            _leaf(pen, (0.02 * (k - 1) - 0.04 * sy, top - 0.03), ang - (look.leaves - 30) * 0.4 + wiggle * 0.6,
                  0.16 - 0.03 * k, 0.07, d.shade if k else d.body, None)


def _leaf_plain(pen: _Pen, base, angle: float, length: float, width: float, fill) -> None:
    a = math.radians(angle)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    s = np.linspace(0, 1, 14)
    half = width / 2 * np.sin(np.pi * s) ** 0.8
    base = np.array(base)
    pen.shape(np.vstack([base + np.outer(s * length, d) + np.outer(half, n),
                         (base + np.outer(s * length, d) - np.outer(half, n))[::-1]]), fill, outline=False)


def _tail(pen: _Pen, d: Design, sy: float) -> None:
    x = -0.92 * d.body_rx * sy
    y = d.body_c - 0.45 * d.body_ry
    if d.tail == "cotton":
        pen.dot(x, y, 0.06, WHITE)
    elif d.tail == "bear":
        pen.dot(x, y, 0.05, d.body)
    elif d.tail == "feathers":
        for ang in (190, 205, 220):
            _leaf(pen, (x * 0.8, y), ang if sy >= 0 else 180 - ang + 180, 0.18, 0.08, d.shade, None)


def _body(pen: _Pen, d: Design) -> None:
    a = np.linspace(0, 2 * math.pi, 72, endpoint=False)

    def egg(cx, cy, rx, ry):
        return np.stack([cx + rx * np.cos(a) * (1 - d.egg * np.sin(a)), cy + ry * np.sin(a)], axis=1)

    light_x = -0.035 * pen.heading  # the light comes from the upper left of the screen
    pen.shape(egg(0.0, d.body_c, d.body_rx, d.body_ry), d.shade, outline=False)
    pen.shape(egg(light_x * d.body_rx / 0.3, d.body_c + 0.022, d.body_rx * 0.95, d.body_ry * 0.95), d.body,
              outline=False)
    pen.ellipse(light_x * 3 * d.body_rx / 0.3 - 0.02 * pen.heading, d.body_c + 0.58 * d.body_ry,
                0.26 * d.body_rx, 0.12 * d.body_ry, _mixc(d.body, d.light, 0.7), outline=False,
                turn=0.5 * pen.heading)
    pen.stroke(egg(0.0, d.body_c, d.body_rx, d.body_ry), pen.line, pen.ow / pen.unit, closed=True)


def _face(pen: _Pen, d: Design, look: _Look, yaw: float, act: Act) -> None:
    sy, cy = math.sin(yaw), math.cos(yaw)
    face_x = 0.5 * d.body_rx * sy
    pen.ellipse(face_x * 0.6, d.belly_y, d.belly_r[0] * (0.7 + 0.3 * abs(cy)), d.belly_r[1], d.belly,
                outline=False)
    spread = d.eye_dx * (0.62 + 0.38 * cy)
    blink = _blinking(act.t, blink_offset(d)) and look.eyes in ("open", "wide", "sad")
    for side in (-1, 1):
        ex = face_x + side * spread
        if side < 0 and sy > 0.8:
            continue  # turned away: the far eye is hidden
        if not (side < 0 and sy > 0.55):
            pen.ellipse(ex + side * (d.eye_r[0] + 0.03), d.eye_y - d.eye_r[1] - 0.012, 0.034, 0.021,
                        _mixc(d.cheek, d.body, 0.25), outline=False)
        _eye(pen, d, ex, d.eye_y, "closed" if blink else look.eyes, look.gaze, -side, sy)
        _brow(pen, d, ex, look.brows, -side, side > 0)
    mx, my = face_x * 1.05, d.mouth_y
    if d.face == "bear":
        pen.ellipse(mx, my + 0.035, 0.12, 0.085, d.belly, outline=False)
        pen.ellipse(mx, my + 0.075, 0.04, 0.026, (60, 36, 30), outline=False)
        pen.dot(mx - 0.012, my + 0.082, 0.009, (140, 110, 100), outline=False)
        _mouth(pen, mx, my - 0.005, look.mouth, look.open, 0.85)
    elif d.face == "bunny":
        _mouth(pen, mx, my - 0.012, look.mouth, look.open, 0.8)
        if look.mouth in ("smile", "side", "chew", "frown"):
            pen.shape([(mx - 0.018, my - 0.002), (mx + 0.018, my - 0.002), (mx + 0.016, my - 0.03),
                       (mx - 0.016, my - 0.03)], WHITE)  # two front teeth
        pen.ellipse(mx, my + 0.03, 0.022, 0.015, (246, 128, 160), outline=False)
        for side in (-1, 1):
            for k in (0, 1):
                pen.stroke([(mx + side * 0.05, my + 0.022 - 0.012 * k), (mx + side * 0.12, my + 0.03 - 0.03 * k)],
                           _mixc(pen.line, (255, 255, 255), 0.35), 0.006)
    elif d.face == "beak":
        _beak(pen, d, mx, my, look.mouth, look.open)
    else:
        _mouth(pen, mx, my, look.mouth, look.open)
    if d.extra == "glasses":
        for side in (-1, 1):
            if side < 0 and sy > 0.8:
                continue
            ex = face_x + side * spread
            pen.stroke(_Pen.oval(ex, d.eye_y, d.eye_r[0] * 1.35, d.eye_r[1] * 1.2), (40, 40, 60), 0.014, closed=True)
        pen.stroke([(face_x - spread + d.eye_r[0] * 1.3, d.eye_y + 0.01), (face_x + spread - d.eye_r[0] * 1.3,
                                                                            d.eye_y + 0.01)], (40, 40, 60), 0.014)


def _front_extra(pen: _Pen, d: Design, yaw: float) -> None:
    sy = math.sin(yaw)
    face_x = 0.5 * d.body_rx * sy
    top = d.body_c + d.body_ry
    if d.extra == "beret":
        x = 0.03 + face_x * 0.3
        pen.ellipse(x, top - 0.005, 0.15, 0.05, d.accent, turn=-0.12)
        pen.ellipse(x - 0.02, top + 0.015, 0.1, 0.022, _mixc(d.accent, (255, 255, 255), 0.4), outline=False,
                    turn=-0.12)
        pen.stroke([(x + 0.01, top + 0.04), (x + 0.02, top + 0.07)], d.line, 0.016)
    elif d.extra == "bowtie":
        x, y = face_x * 0.75, d.mouth_y - 0.1
        red = d.accent
        pen.shape([(x, y), (x - 0.085, y + 0.045), (x - 0.085, y - 0.045)], red)
        pen.shape([(x, y), (x + 0.085, y + 0.045), (x + 0.085, y - 0.045)], red)
        pen.dot(x, y, 0.024, _mixc(red, (0, 0, 0), 0.2))
        for k in (-1, 1):
            pen.dot(x + 0.05 * k, y + 0.012, 0.009, (255, 230, 230), outline=False)


class Character:
    """One of the cast, drawn in a pose (the same interface as :class:`cartoon.Puppet`)."""

    faces = "front"

    def __init__(self, height: int, d: Design = PEEKO):
        self.design = d
        self.h = int(height)
        self.w = int(2 * d.body_rx * height * 1.05)
        sh = np.zeros((int(self.w * 0.3) + 8, int(self.w * 1.0) + 8), F32)
        cv2.ellipse(sh, (sh.shape[1] // 2, sh.shape[0] // 2), (int(self.w * 0.42), int(self.w * 0.1)), 0, 0,
                    360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), self.w * 0.03 + 1)
        self.shadow = Sprite(np.dstack([np.zeros(sh.shape + (3,), F32), sh[..., None] * 0.28]))

    def mirror(self, heading: int) -> float:
        return 1.0

    def _matrix(self, pose, heading: int, ground: float) -> Tuple[np.ndarray, np.ndarray]:
        """(L, origin): screen = origin + L @ (x, y) for forward coordinates in pixels."""
        a = math.radians(pose.angle)
        ca, sa = math.cos(a), math.sin(a)
        k = -pose.lean * heading
        A = np.array([[pose.sx * pose.scale, k * pose.sy * pose.scale], [0.0, pose.sy * pose.scale]])
        R = np.array([[ca, sa], [-sa, ca]])
        L = R @ A @ np.array([[float(heading), 0.0], [0.0, -1.0]])
        return L, np.array([pose.x, ground - pose.lift])

    def head(self, pose, ground: float) -> Tuple[float, float]:
        """Roughly where the top of the head is."""
        L, o = self._matrix(pose, 1, ground)
        _, y = o + L @ np.array([0.0, 0.97 * self.h])
        return pose.x, float(y)

    def hand(self, pose, heading: int, ground: float, act: Optional[Act] = None) -> Tuple[float, float]:
        """Where the front hand is on screen."""
        look = _pose_look(act or Act(), getattr(pose, "yaw", 0.0), 0.0, self.design)
        L, o = self._matrix(pose, heading, ground)
        x, y = o + L @ (np.array(look.hands[1]) * self.h)
        return float(x), float(y)

    def draw(self, frame: np.ndarray, pose, heading: int, ground: float, act: Optional[Act] = None) -> None:
        if pose.alpha <= 0.01:
            return
        act = act or Act()
        d, h = self.design, self.h
        lifted = 1.0 - min(1.0, pose.lift / (1.5 * h))
        self.shadow.paste(frame, pose.x, ground + 0.01 * h, scale=pose.scale * (0.55 + 0.45 * lifted),
                          opacity=pose.alpha * lifted)
        yaw = getattr(pose, "yaw", 0.0)
        walking = act.action in ("walk", "run", "enter", "exit")
        comp = pose.lift / (h * pose.sy * pose.scale) if walking and not d.hover else 0.0
        look = _pose_look(act, yaw, comp, d)
        L, origin = self._matrix(pose, heading, ground)
        pen = _Pen(L * h, origin, h * math.sqrt(abs(np.linalg.det(L))), heading, d.line)
        sy, cy = math.sin(yaw), math.cos(yaw)
        if d.tail:
            _tail(pen, d, sy)
        if look.behind:
            _arm(pen, d, 0, look)
        _feet(pen, d, look, sy, cy)
        _top_behind(pen, d, look, act, sy)
        _body(pen, d)
        if cy > -0.05:
            _face(pen, d, look, yaw, act)
        _front_extra(pen, d, yaw)
        if act.holds and act.holds in PR.PROPS:
            size, grip, turn = PR.hold(act.holds)
            flip = look.prop_flip
            img = PR.prop_image(act.holds, int(size * pen.unit * S / max(d.size, 0.6) ** 0.5),
                                flip != (pen.heading < 0))
            if flip:
                grip = (1.0 - grip[0], grip[1])
            if look.prop_at is not None:
                where, grip = look.prop_at, (0.5, 0.55)
            else:
                where = look.hands[1]
            pen.image(img, where, grip, turn * pen.heading)
        for i in ((1,) if look.behind else (0, 1)):
            _arm(pen, d, i, look)
        pen.blit(frame, pose.alpha)


def Peeko(height: int) -> Character:  # noqa: N802 - the hero, by name
    return Character(height, PEEKO)


# --------------------------------------------------------------------------- pictures

def portrait(size: Tuple[int, int] = (1000, 380), background=(0.93, 0.96, 0.9)) -> bytes:
    """Peeko in four poses on a plain background, as PNG bytes."""
    from .cartoon import Pose
    W, H = size
    frame = np.ones((H, W, 3), F32) * np.array(background, F32)
    rig = Peeko(int(0.66 * H))
    ground = 0.92 * H
    poses = [("wave", 0.6, 0.12, 0.0, 1), ("happy", 0.3, 0.05, 0.0, 1), ("think", 1.0, 0.4, 0.0, -1),
             ("float", 0.4, 0.1, 0.1, 1)]
    for k, (action, u, yaw, lift, heading) in enumerate(poses):
        pose = Pose((0.12 + 0.74 * k / (len(poses) - 1)) * W, lift=lift * H, yaw=yaw)
        rig.draw(frame, pose, heading, ground, Act(action, u, 2.0, 1.0 + k, 0.0, "", 1.0 if k == 1 else 0.0))
    return _png(frame)


def lineup(size: Tuple[int, int] = (1000, 400), background=(0.93, 0.96, 0.9),
           actions: Tuple[str, ...] = ("wave", "happy", "talk", "wave")) -> bytes:
    """The whole cast side by side, as PNG bytes."""
    from .cartoon import Pose
    W, H = size
    frame = np.ones((H, W, 3), F32) * np.array(background, F32)
    ground = 0.93 * H
    unit = 0.62 * H
    for k, (d, action) in enumerate(zip((LULU, PEEKO, KIKO, BOBO), actions)):
        rig = Character(int(unit * d.size), d)
        pose = Pose((0.13 + 0.25 * k) * W, lift=d.hover * rig.h, yaw=0.1)
        rig.draw(frame, pose, 1 if k < 2 else -1, ground, Act(action, 0.6, 2.0, 1.3 + k, 0.6 if action == "talk" else 0))
    return _png(frame)


def _png(frame: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out, "PNG")
    return out.getvalue()
