"""Animal videos: each animal is drawn in pencil from the user's photo while
its facts are written beside it and its narration plays; the video moves on to
the next animal only when the drawing is done, all of its facts have been
shown and its narration has ended.

Each scene is staged like a real drawing session: the animal itself is drawn
first on blank paper in a close-up that follows the pencil, then the scenery
around it; the facts appear as handwriting next to numbered badges, numbers in
them get a highlighter stroke, the name is written (and underlined) when the
drawing is done, and the pencil is heard scratching the paper. Around the
animals: an intro hook, a closing gallery of every drawing and a like /
subscribe / comment ending (see :mod:`showtime`).

Usage:  python -m pencil_timelapse.animals animals.json -o animals.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import analysis as A
from . import music as MU
from .config import RenderConfig
from .encoder import FFmpegWriter
from .errors import InputError, PencilError
from .imaging import load_image, to_uint8
from .media import SAMPLE_RATE, decode_audio, write_wav
from .pencil import PencilRig, make_pencil
from .pipeline import plan_animation
from .players import (HAND_FONTS, PLAIN_FONTS, _add_soundtrack, _font, _has_glyphs, compose,
                      make_layout, prepare_portrait)
from .renderer import FrameRenderer
from .showtime import (BOLD_FONTS, INK, Clip, ShowOptions, Sprite, _f, _phase, _rgb255, _u8, back,
                       intro_clip, outro_clip, smooth, sound_track, sweep_clip, text_sprite,
                       zoom_view)
from .sketch import PAPER, graphite_drawing

F32 = np.float32
ProgressFn = Callable[[float, str], None]
Box = Tuple[int, int, int, int]
MUTED = (0.45, 0.43, 0.4)
HIGHLIGHT = (1.0, 0.84, 0.25)
QUESTION = "Which animal should I draw next?"
# Numbers (with their unit) get a highlighter stroke: "8 kilometres", "40,000", "270 độ".
NUMBER = re.compile(r"\d[\d.,]*\d|\d")
UNIT = re.compile(r"\s?(%|°|km/h|mph|km|kilometres?|kilometers?|metres?|meters?|kg|tonnes?|tons?|"
                  r"cm|mm|m|years?|days?|hours?|degrees?|độ|tấn|năm|mét|giờ|ngày|tuổi)(?!\w)",
                  re.IGNORECASE)

NAME_TIME = 1.7  # to write and underline the name
END_HOLD = 0.9


@dataclass
class Animal:
    name: str
    photo: str  # path to the user's photo
    facts: List[str] = field(default_factory=list)
    voice: Optional[str] = None  # optional narration, from the moment the drawing starts


@dataclass
class AnimalsResult:
    output_path: str
    image_path: str  # the closing gallery
    duration: float
    animals: List[str]
    elapsed: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class SceneLayout:
    draw_box: Box  # where the animal is drawn
    head: Tuple[int, int]  # top-left of the "#2 / 5" line
    name: Tuple[int, int]  # top-left of the name
    facts: Box  # where the facts are written
    name_size: int
    fact_size: int


def scene_layout(width: int, height: int) -> SceneLayout:
    """Drawing on the left and text on the right (landscape), or the drawing
    on top and the text below (portrait and square)."""
    W, H = width, height
    if W > H * 1.2:
        draw = (int(0.035 * W), int(0.08 * H), int(0.53 * W), int(0.84 * H))
        x = int(0.6 * W)
        return SceneLayout(draw, (x, int(0.11 * H)), (x, int(0.16 * H)),
                           (x, int(0.33 * H), int(0.37 * W), int(0.59 * H)),
                           int(0.085 * H), int(0.046 * H))
    draw = (int(0.05 * W), int(0.045 * H), int(0.9 * W), int(0.5 * H) if H > W else int(0.55 * H))
    top = draw[1] + draw[3]
    x = int(0.07 * W)
    unit = min(W, H)
    return SceneLayout(draw, (x, top + int(0.02 * H)), (x, top + int(0.05 * H)),
                       (x, top + int(0.06 * H) + int(0.11 * unit), int(0.86 * W),
                        H - top - int(0.06 * H) - int(0.11 * unit) - int(0.03 * H)),
                       int(0.085 * unit), int(0.042 * unit))


def write_seconds(text: str) -> float:
    """How long the pencil takes to write ``text``."""
    return float(np.clip(0.045 * len(text), 0.4, 2.0))


def reading_seconds(fact: str) -> float:
    """How long a fact is the newest on screen: the pencil writes it, then it is read."""
    return max(2.5, 1.2 + 0.38 * len(fact.split()), 0.9 + 1.15 * write_seconds(fact) + 1.1)


def _paper(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    grain = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(F32), (0, 0), 0.7)
    return np.clip(PAPER[None, None, :] * (1.0 + 0.008 * grain[..., None]), 0.0, 1.0)


# --------------------------------------------------------------------------- text pieces

def _hand(text: str, size: int) -> Tuple[Tuple[str, ...], int]:
    """The handwriting font, or a plain one (slightly smaller) when it lacks a letter."""
    if _has_glyphs(_font(HAND_FONTS, size), text):
        return HAND_FONTS, size
    return PLAIN_FONTS, int(size * 0.88)


@dataclass
class _Text:
    sprite: Sprite
    ink: Tuple[float, float]  # where the writing starts and ends, from the sprite's left
    spans: List[Tuple[float, float]]  # numbers to highlight, from the sprite's left


def _text(text: str, size: int, colour=INK) -> _Text:
    """A line of handwriting and the positions of the numbers in it."""
    paths, size = _hand(text, size)
    font, big = _font(paths, size), _font(paths, size * 2)
    pad = int(size * 0.15) + 2
    asc, desc = font.getmetrics()
    w, h = int(font.getlength(text)) + 2 * pad + 2, asc + desc + 2 * pad
    img = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad * 2, pad * 2), text, font=big, fill=_rgb255(colour))
    sprite = Sprite(np.asarray(img.resize((w, h), Image.LANCZOS)).astype(F32) / 255.0)
    spans = []
    for m in NUMBER.finditer(text):
        unit = UNIT.match(text, m.end())
        end = unit.end() if unit else m.end()
        spans.append((pad + font.getlength(text[:m.start()]), pad + font.getlength(text[:end])))
    return _Text(sprite, (pad, pad + font.getlength(text)), spans)


def _wrap(text: str, size: int, width: int) -> List[str]:
    paths, size = _hand(text, size)
    font = _font(paths, size)
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.getlength(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def _render(draw_fn, size: Tuple[int, int]) -> Sprite:
    """Draw with Pillow at twice the size, then scale down (smooth edges)."""
    w, h = size
    img = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img), 2)
    return Sprite(np.asarray(img.resize((w, h), Image.LANCZOS)).astype(F32) / 255.0)


def _badge(number: int, size: int) -> Sprite:
    """A pencil circle with the fact's number in it."""
    d = int(size * 1.05)

    def draw(dr: ImageDraw.ImageDraw, s: int) -> None:
        line = max(2, int(size * 0.07)) * s
        dr.ellipse([line, line, d * s - line, d * s - line], outline=_rgb255(INK), width=line)
        dr.arc([line // 2, line * 2, d * s - line, d * s - line // 2], 200, 330,
               fill=_rgb255(INK), width=max(1, line // 2))  # a second, looser pass of the pencil
        font = _font(HAND_FONTS, int(size * 0.62) * s)
        label = str(number)
        asc, desc = font.getmetrics()
        dr.text(((d * s - font.getlength(label)) / 2, (d * s - asc - desc * 0.3) / 2), label,
                font=font, fill=_rgb255(INK))

    return _render(draw, (d, d))


def _underline(width: float, size: int) -> Sprite:
    """A quick wavy pencil stroke under the name."""
    w, h = int(width) + 8, max(8, int(size * 0.22))

    def draw(dr: ImageDraw.ImageDraw, s: int) -> None:
        xs = np.linspace(4, w - 4, 40)
        ys = h / 2 + np.sin(xs / w * 2.4 * math.pi) * h * 0.18 + (xs / w - 0.5) * h * 0.12
        dr.line([(x * s, y * s) for x, y in zip(xs, ys)], fill=_rgb255(INK),
                width=max(2, int(size * 0.06)) * s, joint="curve")

    return _render(draw, (w, h))


def _marker(width: float, height: float) -> Sprite:
    """A highlighter stroke, slightly slanted."""
    w, h = max(4, int(width)), max(4, int(height))

    def draw(dr: ImageDraw.ImageDraw, s: int) -> None:
        dr.polygon([(0, h * 0.18 * s), (w * s, 0), (w * s, h * 0.86 * s), (0, h * s)],
                   fill=_rgb255(HIGHLIGHT, 0.55))

    return _render(draw, (w, h))


# --------------------------------------------------------------------------- sound

def _scratch(rig: PencilRig, seconds: float, loud: float = 0.11) -> np.ndarray:
    """The pencil scratching the paper: louder when the tip moves fast on the
    paper, silent when it is lifted."""
    n = int(seconds * SAMPLE_RATE)
    if n <= 0:
        return np.zeros(0, F32)
    ctrl = np.arange(0.0, seconds, 1.0 / 400.0)
    state = np.array([rig.state(t) for t in ctrl])
    speed = np.hypot(np.gradient(state[:, 0]), np.gradient(state[:, 1])) * 400.0
    contact = np.clip(1.0 - 2.0 * state[:, 2], 0.0, 1.0)
    on = contact > 0.5
    ref = float(np.percentile(speed[on], 90)) if on.any() else 1.0
    env = np.clip(speed / max(ref, 1e-6), 0.0, 1.0) ** 0.7 * contact
    env = np.interp(np.arange(n) / SAMPLE_RATE, ctrl, env)
    rng = np.random.default_rng(17)
    spec = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    spec[(freq < 1200) | (freq > 7500)] = 0
    noise = np.fft.irfft(spec, n)
    noise /= np.abs(noise).max() + 1e-9
    steps = np.arange(0, n, 300)
    grain = 0.65 + 0.35 * np.interp(np.arange(n), steps, rng.random(len(steps)))
    return (loud * noise * env * grain).astype(F32)


# --------------------------------------------------------------------------- the text

@dataclass
class _Line:
    words: str
    text: _Text
    x: int
    y: int


@dataclass
class _Fact:
    badge: Sprite
    at: Tuple[int, int]  # badge top-left
    lines: List[_Line]
    page: int


def _facts(facts: List[str], box: Box, size: int) -> List[_Fact]:
    """Each fact as a numbered badge and wrapped lines; facts that do not fit
    start a new page."""
    x, y, w, h = box
    step, gap = int(size * 1.32), int(size * 0.6)
    indent = int(size * 1.35)
    out, page, cursor = [], 0, y
    for k, fact in enumerate(facts):
        lines = _wrap(fact, size, w - indent)
        need = len(lines) * step
        if cursor > y and cursor + need > y + h:
            page, cursor = page + 1, y
        placed = [_Line(line, _text(line, size), x + indent, cursor + j * step)
                  for j, line in enumerate(lines)]
        out.append(_Fact(_badge(k + 1, size), (x, cursor + int(size * 0.12)), placed, page))
        cursor += need + gap
    return out


def _timings(animal: Animal, voice_seconds: float, first: float) -> List[float]:
    """How long each fact is the newest on screen. With a narration the facts
    are spread over it, from ``first`` (when the first fact appears)."""
    reading = [reading_seconds(f) for f in animal.facts]
    if not voice_seconds or not animal.facts:
        return reading
    left = voice_seconds - first
    chars = [max(len(f), 1) for f in animal.facts]
    return [max(write_seconds(f) + 1.0, left * c / sum(chars)) for f, c in zip(animal.facts, chars)]


def _done(t: float, t0: float, t1: float) -> float:
    return min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)


def _written(text: _Text, share: float) -> float:
    """How much of a line's sprite shows when ``share`` of its writing is done."""
    if share >= 1.0:
        return 1.0
    if share <= 0.0:
        return 0.0
    return (text.ink[0] + share * (text.ink[1] - text.ink[0])) / text.sprite.size[0]


class _SceneText:
    """Everything written beside an animal, as a function of the scene's time.

    The facts are written from the first stroke of the drawing (and with the
    narration), each with a numbered badge and its numbers highlighted. The
    name is written when the drawing is done if the audience is asked to
    guess, otherwise first.
    """

    def __init__(self, animal: Animal, number: str, layout: SceneLayout, drawing: float,
                 voice: float, guess: bool, prompt: str = "Guess the animal?", where: str = ""):
        self.layout = layout
        self.head = text_sprite(number, BOLD_FONTS, int(layout.name_size * 0.38), MUTED)
        # ``where`` (a place's location) follows the number once the name is written.
        self.where = text_sprite(f"·  {where.upper()}", BOLD_FONTS, int(layout.name_size * 0.38),
                                 MUTED, max_width=int(0.9 * layout.facts[2])) if where else None
        self.guess = _text(prompt, int(layout.name_size * 0.8)).sprite if guess else None
        self.name = _text(animal.name, layout.name_size)
        nx, ny = layout.name
        self.under = _underline(self.name.ink[1] - self.name.ink[0], layout.name_size)
        self.under_at = (nx + self.name.ink[0] - 4, ny + int(self.name.sprite.size[1] * 0.86))
        self.named = drawing if guess else 0.0  # when the name is written
        first = 0.8 if guess else NAME_TIME
        self.facts = _facts(animal.facts, layout.facts, layout.fact_size)
        timings = _timings(animal, voice, first)
        self.starts = [first + sum(timings[:i]) for i in range(len(timings))]
        self.writes: List[List[Tuple[float, float]]] = []  # per fact, per line
        self.marked: List[float] = []  # when each fact's numbers are highlighted
        for fact, start, span in zip(self.facts, self.starts, timings):
            each = [write_seconds(ln.words) for ln in fact.lines]
            scale = min(1.0, max(0.4, span - 1.0) / max(sum(each) + 0.1 * (len(each) - 1), 1e-6))
            t, lines = start + 0.25, []
            for d in each:
                lines.append((t, t + d * scale))
                t += d * scale + 0.1
            self.writes.append(lines)
            self.marked.append(t)
        self.markers = [[[_marker(b - a + 6, ln.text.sprite.size[1] * 0.5) for a, b in ln.text.spans]
                         for ln in fact.lines] for fact in self.facts]
        self.end = max(self.named + NAME_TIME, first + sum(timings) + END_HOLD,
                       voice + END_HOLD, drawing + 1.0)
        self.sounds = [(self.named + 0.25, "pop")] + [(s + 0.1, "pop") for s in self.starts]
        self.sounds += [(m, "swish") for m, fact in zip(self.marked, self.facts)
                        if any(ln.text.spans for ln in fact.lines)]

    def paint(self, frame: np.ndarray, t: float) -> None:
        lay = self.layout
        self.head.paste_left(frame, *lay.head)
        if self.guess is not None and t < self.named + 0.25:
            self.guess.paste_left(frame, *lay.name, opacity=0.75 * (1.0 - _done(t, self.named,
                                                                                   self.named + 0.25)))
        self.name.sprite.paste_left(frame, *lay.name, reveal=_written(
            self.name, _done(t, self.named + 0.25, self.named + 1.05)))
        self.under.paste_left(frame, *self.under_at,
                              reveal=_done(t, self.named + 1.12, self.named + 1.38))
        if self.where is not None:
            self.where.paste_left(frame, lay.head[0] + self.head.size[0], lay.head[1],
                                  reveal=_done(t, self.named + 1.15, self.named + 1.6))
        current = max([i for i, s in enumerate(self.starts) if s <= t], default=-1)
        for i, fact in enumerate(self.facts[:current + 1]):
            fade = 1.0
            if fact.page != self.facts[current].page:  # the previous page is wiped away
                if fact.page != self.facts[current - 1].page:
                    continue
                fade = 1.0 - _done(t, self.starts[current], self.starts[current] + 0.3)
                if fade <= 0.0:
                    continue
            b = _phase(t, self.starts[i] + 0.1, 0.3)
            fact.badge.paste(frame, fact.at[0] + fact.badge.size[0] / 2,
                             fact.at[1] + fact.badge.size[1] / 2, scale=back(b),
                             opacity=min(1.0, 3 * b) * fade)
            swipe = _done(t, self.marked[i], self.marked[i] + 0.3)
            for j, ln in enumerate(fact.lines):
                for (a, _), marker in zip(ln.text.spans, self.markers[i][j]):
                    marker.paste_left(frame, ln.x + a - 3, ln.y + ln.text.sprite.size[1] * 0.36,
                                      reveal=swipe, opacity=fade)
                ln.text.sprite.paste_left(frame, ln.x, ln.y, opacity=fade,
                                          reveal=_written(ln.text, _done(t, *self.writes[i][j])))


def _finished_clip(page: np.ndarray, text: _SceneText, start: float, box: Box, fps: int) -> Clip:
    """The rest of a scene once the drawing is done: light runs over the
    graphite while the remaining text is written."""
    base = _f(page)
    x, y, w, h = box
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    slant = (xx + 0.5 * yy) / (w + 0.5 * h)
    seconds = text.end - start
    count = int(round(seconds * fps))

    def frame_at(t: float) -> np.ndarray:
        frame = base.copy()
        g = _phase(t, 0.05, 0.9)
        if 0.0 < g < 1.0:
            band = np.exp(-((slant - (g * 1.4 - 0.2)) / 0.07) ** 2)[..., None]
            region = frame[y:y + h, x:x + w]
            region += band * 0.5 * (1.0 - region) * (region < 0.85)
        text.paint(frame, start + t)
        return _u8(np.clip(frame, 0.0, 1.0))

    def frames() -> Iterator[np.ndarray]:
        for i in range(count):
            yield frame_at(i / fps)

    return Clip(seconds, frames, [], last=lambda: frame_at((count - 1) / fps))


# --------------------------------------------------------------------------- drawing

def _animal_mask(photo: np.ndarray) -> Optional[np.ndarray]:
    """Soft mask of the animal (the main subject, near the middle) so it can be
    drawn before the scenery; None when it cannot be told apart."""
    h, w = photo.shape[:2]
    s = min(1.0, 400.0 / max(h, w))
    small = cv2.resize(photo, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    rect = (int(0.06 * sw), int(0.06 * sh), int(0.88 * sw), int(0.9 * sh))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.convertScaleAbs(small[..., ::-1], alpha=255.0), mask, rect, bgd, fgd, 5,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    found = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    found = cv2.morphologyEx(found, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(found)
    if count < 2:
        return None
    found = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    flood = np.pad(1 - found, 1, constant_values=1)
    cv2.floodFill(flood, None, (0, 0), 2)
    found[flood[1:-1, 1:-1] == 1] = 1
    if not 0.06 <= float(found.mean()) <= 0.8:
        return None
    k = max(3, int(0.03 * max(sh, sw))) | 1
    found = cv2.dilate(found, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    soft = A.blur(found.astype(F32), 0.015 * max(sh, sw) + 0.5)
    return cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)


def _camera(rig: PencilRig, seconds: float, fps: int, box: Box, zoom: float = 1.4
            ) -> List[Tuple[float, float, float]]:
    """A close-up inside the drawing that follows the pencil smoothly and
    pulls back at the end: the window's top-left and zoom, per frame."""
    x, y, w, h = box
    n = int(round(seconds * fps))
    times = np.arange(n) / fps
    tip = np.array([rig.state(t)[:2] for t in times]).reshape(-1, 2)
    tip[:, 0] = np.clip(tip[:, 0], x + 0.2 * w, x + 0.8 * w)
    tip[:, 1] = np.clip(tip[:, 1], y + 0.2 * h, y + 0.8 * h)
    sigma = max(1.0, 0.8 * fps)
    kernel = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    follow = np.stack([np.convolve(np.pad(tip[:, i], pad, mode="edge"), kernel, "valid")[:n]
                       for i in range(2)], axis=1)
    out = []
    for t, (cx, cy) in zip(times, follow):
        close = smooth((t - 0.3) / 1.2) * (1.0 - smooth((t - (seconds - 2.2)) / 1.6))
        z = 1.0 + (zoom - 1.0) * close
        sx = float(np.clip(cx - w / (2 * z), x, x + w - w / z))
        sy = float(np.clip(cy - h / (2 * z), y, y + h - h / z))
        out.append((sx, sy, z))
    return out


def _close_up(frame: np.ndarray, box: Box, window: Tuple[float, float, float],
              edge: np.ndarray, paper: np.ndarray) -> None:
    """Magnify ``window`` (top-left, zoom) of the drawing into its box, in place;
    the rest of the page, with the text, stays as it is."""
    sx, sy, z = window
    if z <= 1.001:
        return
    x, y, w, h = box
    m = np.array([[z, 0, -(sx - x) * z], [0, z, -(sy - y) * z]], np.float32)
    region = cv2.warpAffine(np.ascontiguousarray(frame[y:y + h, x:x + w]), m, (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    keep = (1.0 - (1.0 - edge) * min(1.0, (z - 1.0) / 0.15))[..., None]
    frame[y:y + h, x:x + w] = region * keep + paper[y:y + h, x:x + w] * (1.0 - keep)


def _moved(rig: PencilRig, box: Box, windows: List[Tuple[float, float, float]], fps: int
           ) -> PencilRig:
    """The pencil where it appears in the close-up."""
    x, y = box[0], box[1]
    frames = np.arange(len(windows)) / fps
    sx, sy, z = (np.interp(rig.times, frames, [wnd[i] for wnd in windows]) for i in range(3))
    return PencilRig(rig.sprite, rig.times, x + (rig.x - sx) * z, y + (rig.y - sy) * z, rig.lift,
                     rig.angle, rig.end_time, rig.k, rig.shutter, rig.shadow_strength)


def render_parts(parts: List[tuple], scene: dict, voices: List[Optional[str]],
                 music: Optional[str], sound_effects: bool, layout: SceneLayout,
                 paper: np.ndarray, fps: int, output_path: str, report: ProgressFn,
                 crf: int = 20, preset: str = "fast", duck: bool = False) -> float:
    """Write ``parts`` in order and lay the sound under them; returns the length.

    A part is ``("draw", start, end, seconds, label, text, offset, follow)`` (the
    pencil drawing ``end`` over ``start`` while ``text`` is painted at
    ``offset`` + t, in a close-up following the pencil when ``follow``),
    ``("clip", Clip, label)`` or ``("still", frame, seconds, label)``.
    ``scene`` maps the index of the part that begins a scene to (the index in
    ``voices``, its text): the narration and the text's sounds start there.
    With ``duck`` the music is turned down while someone speaks.
    """
    height, width = paper.shape[:2]
    bx, by, bw, bh = layout.draw_box
    yy, xx = np.mgrid[0:bh, 0:bw].astype(F32)
    edge = A.smoothstep(np.minimum.reduce([xx, yy, bw - 1 - xx, bh - 1 - yy]) / (0.04 * min(bw, bh)))
    music_track = decode_audio(music) if music else None

    def length(part: tuple) -> float:
        return part[1].seconds if part[0] == "clip" else (part[3] if part[0] == "draw" else part[2])

    counts = [int(round(length(pt) * fps)) for pt in parts]
    n_total = sum(counts)
    total = n_total / fps

    tmp = tempfile.mkdtemp(prefix=".show-", dir=os.path.dirname(output_path))
    done, sounds, scratches, voice_at, said = 0, [], [], {}, []
    try:
        tmp_video = os.path.join(tmp, "video.mp4")
        with FFmpegWriter(tmp_video, width, height, fps, crf, preset, total_frames=n_total) as writer:
            for si, (part, count) in enumerate(zip(parts, counts)):
                kind = part[0]
                label = part[4] if kind == "draw" else (part[2] if kind == "clip" else part[3])
                report(0.12 + 0.83 * done / n_total, label)
                at = done / fps
                if si in scene:  # a scene begins: its narration and its text start now
                    i, text = scene[si]
                    voice_at[i] = at
                    sounds += [(at + t, name) for t, name in text.sounds]
                if kind == "draw":
                    _, start, end, seconds, label, text, offset, follow = part
                    cfg = RenderConfig(width=width, height=height, fps=fps, duration=seconds,
                                       hold=0.0, start_mode="image", align=False,
                                       match_paper=False, construction_lines=False, seed=21 + si)
                    plan = plan_animation(start, end, cfg, focus_region=layout.draw_box)
                    pencil, plan.pencil = plan.pencil, None  # drawn last, above the text
                    windows = None
                    if pencil is not None:
                        scratches.append((at, _scratch(pencil, seconds)))
                        if follow:
                            windows = _camera(pencil, seconds, fps, layout.draw_box)
                            pencil = _moved(pencil, layout.draw_box, windows, fps)
                    for k, t, frame in FrameRenderer(plan).iter_frames(fps, count):
                        f = _f(frame)
                        if windows is not None:
                            _close_up(f, layout.draw_box, windows[min(k, len(windows) - 1)], edge,
                                      paper)
                        text.paint(f, offset + t)
                        if pencil is not None:
                            pencil.draw(f, t)
                        writer.write(_u8(np.clip(f, 0.0, 1.0)))
                        if k % 15 == 0:
                            report(0.12 + 0.83 * (done + k) / n_total, label)
                    del plan
                elif kind == "clip":
                    sounds += [(at + t, name) for t, name in part[1].sounds]
                    said += [(at + t, samples) for t, samples in part[1].speech]
                    for k, frame in zip(range(count), part[1].frames()):
                        writer.write(frame)
                        if k % 15 == 0:
                            report(0.12 + 0.83 * (done + k) / n_total, label)
                else:
                    for _ in range(count):
                        writer.write(part[1])
                done += count

        report(0.96, "Adding the sound")
        mix = np.zeros(int(math.ceil(total * SAMPLE_RATE)) + 1, F32)
        speech = np.zeros_like(mix)
        for i, voice in enumerate(voices):
            if voice and i in voice_at:
                said.append((voice_at[i], decode_audio(voice)))
        for when, clip in said:
            at = int(when * SAMPLE_RATE)
            end = min(len(mix), at + len(clip))
            if at < end:
                speech[at:end] += clip[:end - at]
        if music_track is not None and len(music_track):
            loud = 0.3 if said else 0.8
            bed = np.tile(music_track, int(math.ceil(len(mix) / len(music_track))))[:len(mix)]
            if duck and said:
                bed = MU.duck(bed.astype(F32), speech)
            mix += bed * loud
        mix += speech
        if sound_effects:
            quiet = 0.7 if music_track is not None else 1.0
            mix += sound_track(sounds, total)[:len(mix)] * quiet
            for at, samples in scratches:
                start = int(at * SAMPLE_RATE)
                end = min(len(mix), start + len(samples))
                mix[start:end] += samples[:end - start] * quiet
        peak = float(np.abs(mix).max()) if mix.size else 0.0
        if peak > 0.98:
            mix *= 0.98 / peak
        if peak > 0.0:
            wav = os.path.join(tmp, "sound.wav")
            write_wav(wav, mix)
            mixed = os.path.join(tmp, "with_sound.mp4")
            _add_soundtrack(tmp_video, mixed, total, wav, total)
            tmp_video = mixed
        os.replace(tmp_video, output_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return total


# --------------------------------------------------------------------------- video

def render_animals(animals: List[Animal], title: str, output_path: str, width: int = 1920,
                   height: int = 1080, fps: int = 30, drawing_seconds: float = 10.0,
                   music: Optional[str] = None, guess: bool = True,
                   show: Optional[ShowOptions] = None, progress: Optional[ProgressFn] = None,
                   crf: int = 20, preset: str = "fast") -> AnimalsResult:
    """Draw ``animals`` one after another with their facts, and save the video
    and the closing gallery picture (PNG).

    Drawing, writing and narration go together: from an animal's first stroke
    its narration plays and its facts are written beside the drawing. The
    video stays with an animal until its drawing is done, its facts have all
    been shown and its narration has ended. With ``guess`` the name is hidden
    ("Guess the animal?") until the drawing is finished. ``music`` plays under
    the whole video, quieter when the animals have narration. With 8 seconds
    or more of drawing time the animal is drawn first, in a close-up that
    follows the pencil, and the scenery after it.
    """
    report = progress or (lambda f, m: None)
    started = time.perf_counter()
    show = show or ShowOptions(question=QUESTION)
    if not animals:
        raise InputError("Add at least one animal.")
    if len(animals) > 20:
        raise InputError("Add at most 20 animals.")
    if not 4.0 <= drawing_seconds <= 60.0:
        raise InputError("Drawing time must be between 4 and 60 seconds.")
    RenderConfig(width=width, height=height, fps=fps).validate()
    for a in animals:
        if not a.name.strip():
            raise InputError("Every animal needs a name.")
    voices = [len(decode_audio(a.voice)) / SAMPLE_RATE if a.voice else 0.0 for a in animals]
    if music:
        decode_audio(music, rate=8000)  # fail now, not after the drawing
    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    n = len(animals)
    layout = scene_layout(width, height)
    paper = _paper(width, height)
    blank_u8 = to_uint8(paper)
    bx, by, bw, bh = layout.draw_box

    # ---- drawings: the animal alone, the whole scene, and one for the closing gallery
    gallery_layout = make_layout(n, width, height)
    stages, gallery = [], []
    for i, a in enumerate(animals):
        report(0.02 + 0.1 * i / n, f"Preparing {a.name} ({i + 1}/{n})")
        photo = load_image(a.photo, f"Photo of {a.name}")
        framed = prepare_portrait(photo, bw / bh, None, "whole", seed=i)
        page = paper.copy()
        page[by:by + bh, bx:bx + bw] = np.minimum(
            page[by:by + bh, bx:bx + bw], graphite_drawing(framed, (bw, bh), seed=i, border=0.025))
        mask = _animal_mask(framed) if drawing_seconds >= 8.0 else None
        if mask is not None:
            first = paper.copy()
            first[by:by + bh, bx:bx + bw] += (page[by:by + bh, bx:bx + bw] -
                                              first[by:by + bh, bx:bx + bw]) * \
                cv2.resize(mask, (bw, bh))[..., None]
            stages.append([first, page])
        else:
            stages.append([page])
        gx, gy, gw, gh = gallery_layout.cells[i]
        gallery.append(graphite_drawing(prepare_portrait(photo, gw / gh, None, "whole", seed=i),
                                        (gw, gh), seed=i, border=0.025))
    names = [a.name for a in animals]
    final = to_uint8(compose(gallery_layout, gallery, names, title, n))

    # ---- the parts of the video, in order
    parts: List[tuple] = []
    if show.intro:
        hook = (show.hook.strip() or title or "Amazing animals").upper()
        kicker = f"{n} ANIMAL{'S' if n > 1 else ''}  ·  1 PENCIL"
        if guess:  # do not give the animals away: a blurred glimpse only
            teaser = _u8(cv2.GaussianBlur(_f(final), (0, 0), 0.012 * width))
            parts.append(("clip", intro_clip(teaser, blank_u8, hook, kicker,
                                             "Guess them before I finish drawing!", fps), "Intro"))
        else:
            parts.append(("clip", intro_clip(final, blank_u8, hook, kicker,
                                             "Watch them come to life in pencil", fps), "Intro"))
    scene = {}  # index of the first drawing part of each animal: the animal and its text
    for i, a in enumerate(animals):
        text = _SceneText(a, f"#{i + 1} / {n}", layout, drawing_seconds, voices[i], guess)
        scene[len(parts)] = (i, text)
        if len(stages[i]) == 2:  # the animal first, in a close-up; then the scenery
            first = max(4.0, round(0.62 * drawing_seconds, 1))
            second = drawing_seconds - first
            if second < 4.0:
                first, second = drawing_seconds - 4.0, 4.0
            parts.append(("draw", paper, stages[i][0], first, f"Drawing {a.name}", text, 0.0, True))
            parts.append(("draw", stages[i][0], stages[i][1], second,
                          f"Drawing {a.name}: the scenery", text, first, False))
        else:
            parts.append(("draw", paper, stages[i][0], drawing_seconds, f"Drawing {a.name}",
                          text, 0.0, False))
        rest = _finished_clip(to_uint8(stages[i][-1]), text, drawing_seconds, layout.draw_box, fps)
        parts.append(("clip", rest, f"{a.name}: the facts"))
        target = blank_u8 if i + 1 < n else final
        parts.append(("clip", sweep_clip(rest.last(), target, fps),
                      "Next animal" if i + 1 < n else "Gallery"))
    parts.append(("still", final, 3.0, "Gallery"))
    if show.outro:
        parts.append(("clip", outro_clip(final, show.question.strip() or QUESTION, [], fps),
                      "Like, subscribe, comment"))

    total = render_parts(parts, scene, [a.voice for a in animals], music, show.sound_effects,
                         layout, paper, fps, output_path, report, crf, preset)

    image_path = os.path.splitext(output_path)[0] + ".png"
    Image.fromarray(final).save(image_path)
    report(1.0, "Done")
    return AnimalsResult(output_path, image_path, total, names, time.perf_counter() - started)


# --------------------------------------------------------------------------- CLI

def load_script(path: str) -> Tuple[str, List[Animal]]:
    """A JSON script: {"title": "...", "animals": [{"name", "photo", "facts", "voice"}]}
    with file paths relative to the script."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))

    def where(value: Optional[str]) -> Optional[str]:
        return os.path.join(base, value) if value else None

    animals = [Animal(str(a["name"]), where(a["photo"]), [str(f) for f in a.get("facts", [])],
                      where(a.get("voice"))) for a in data.get("animals", [])]
    return str(data.get("title", "")), animals


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pencil-animals",
                                description="Draw animals in pencil with their facts, one after "
                                            "another (simulated drawing).")
    p.add_argument("script", help='JSON: {"title", "animals": [{"name", "photo", "facts": [...], '
                                  '"voice"}]}')
    p.add_argument("-o", "--output", required=True, help="output MP4 (a PNG is saved next to it)")
    p.add_argument("--size", default="1920x1080", help="resolution, e.g. 1080x1920 for vertical")
    p.add_argument("--seconds", type=float, default=10.0, help="drawing time per animal")
    p.add_argument("--music", help="music under the whole video")
    p.add_argument("--no-guess", action="store_true", help="show the name while drawing")
    p.add_argument("--hook", default="", help="intro headline (default: the title)")
    p.add_argument("--question", default=QUESTION, help="question asked at the end")
    for flag in ("intro", "outro", "sound-effects"):
        p.add_argument(f"--no-{flag}", action="store_true", help=f"leave out the {flag}")
    args = p.parse_args(argv)
    try:
        title, animals = load_script(args.script)
        w, h = (int(v) for v in args.size.lower().split("x"))
        show = ShowOptions(not args.no_intro, True, not args.no_outro, not args.no_sound_effects,
                           args.hook, args.question)
        result = render_animals(animals, title, args.output, w, h, drawing_seconds=args.seconds,
                                music=args.music, guess=not args.no_guess, show=show)
    except (PencilError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {result.output_path} ({result.duration:.0f} s) and {result.image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
