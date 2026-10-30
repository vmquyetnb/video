"""Pet diaries: a day in a pet's life, drawn in pencil and told by the pet.

Each moment of the day (a photo or a video clip) is drawn in a sketchbook
while the pet's thought is written in a bubble beside it and read in the
pet's voice; the time of day is written on top with a little clock, and a
paw print walks along a day bar from sunrise to bedtime. When the drawing is
done it comes to life: colour washes over the graphite, the real clip grows
to the whole screen as the pet's camera, and the pet's punchline appears word
by word, then paw prints walk across to the next page.

Built to be watched to the end:

* a cold open: the day's big moment plays, freezes with a record scratch,
  turns to pencil and the pet asks how it ended up there ("Let me
  explain."); the day rewinds to the morning;
* a "What happens next?" guess with a 3-2-1 countdown before the big moment;
* the day bar shows how much of the day is left;
* a sketchbook of the whole day and a question for the comments at the end.
In tall videos the text stays clear of the bottom of the screen, where
Shorts puts its own buttons and captions.

Usage:  python -m pencil_timelapse.pets diary.json -o diary.mp4
        python -m pencil_timelapse.pets --sample biscuit -o biscuit.mp4
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
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from . import music as MU
from .animals import (_animal_mask, _badge, _done, _finished_clip, _marker, _paper, _text,
                      _underline, _wrap, _written, render_parts)
from .config import RenderConfig
from .errors import InputError, PencilError, RenderError
from .imaging import load_image, to_uint8
from .media import SAMPLE_RATE, write_wav
from .players import prepare_portrait
from .showtime import (BOLD_FONTS, DISPLAY_FONTS, HAND_FONTS, INK, WHITE, Clip, Sprite, _f, _phase,
                       _pill, _render, _rgb255, _u8, back, outro_clip, smooth, text_sprite)
from .sketch import PAPER, graphite_drawing
from .travel import VIDEO_TYPES, Footage, _grid, postcard_wall, video_frame, video_seconds

F32 = np.float32
Box = Tuple[int, int, int, int]
ProgressFn = Callable[[float, str], None]

QUESTION = "What does your pet do all day?"
HOOK = "Wait... how did I end up here?"
EXPLAIN = "Let me explain."
ASK = "What do you think happens next?"
KINDS = ("dog", "cat", "rabbit", "hamster", "bird", "other")
VOICES = {"dog": ("Samantha", 1.1), "cat": ("Karen", 1.25), "rabbit": ("Tessa", 1.3),
          "hamster": ("Tessa", 1.45), "bird": ("Moira", 1.35), "other": ("Samantha", 1.15)}
RATE = 175  # words a minute: a little slower than the narrator, for small listeners
REVEAL = 3.0  # seconds of the real clip, at least
WASH, MELT, GROW, PAW = 0.7, 0.35, 0.6, 0.9
THOUGHT_AT = 1.3  # the bubble opens and the pet starts thinking aloud
PLAY, DRAIN, REWIND = 1.7, 0.5, 0.9  # the cold open
MUTED = (0.45, 0.43, 0.4)
GREEN = (0.22, 0.62, 0.33)
SAID = (1.0, 0.86, 0.18)  # the word being said in a caption
REC = (0.9, 0.16, 0.13)


@dataclass
class Moment:
    time: str  # "7:00 AM"
    title: str  # "Breakfast"
    thought: str  # what the pet thinks while it is drawn
    media: str = ""  # a photo or a video clip
    punch: str = ""  # what the pet says when the drawing comes to life
    credit: str = ""
    best: bool = False  # the day's big moment: the cold open and the guess
    search: str = ""  # words to find a photo or clip of it

    @property
    def is_video(self) -> bool:
        return os.path.splitext(self.media)[1].lower() in VIDEO_TYPES


@dataclass
class Diary:
    pet: str
    kind: str = "dog"
    title: str = ""
    hook: str = HOOK
    guess: List[str] = field(default_factory=list)  # answers to "What happens next?"
    answer: str = ""  # the right one
    question: str = QUESTION
    moments: List[Moment] = field(default_factory=list)

    def best(self) -> int:
        """The big moment: the one marked, else the last."""
        return next((i for i, m in enumerate(self.moments) if m.best), len(self.moments) - 1)

    def heading(self) -> str:
        return self.title.strip() or f"A Day in the Life of {self.pet}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Diary":
        moments = [Moment(time=str(m.get("time") or ""), title=str(m.get("title") or ""),
                          thought=str(m.get("thought") or ""), media=str(m.get("media") or ""),
                          punch=str(m.get("punch") or ""), credit=str(m.get("credit") or ""),
                          best=bool(m.get("best")), search=str(m.get("search") or ""))
                   for m in data.get("moments") or [] if isinstance(m, dict)]
        guess = data.get("guess") or []
        if isinstance(guess, str):
            guess = guess.splitlines()
        return cls(pet=str(data.get("pet") or "").strip(), kind=str(data.get("kind") or "dog"),
                   title=str(data.get("title") or ""), hook=str(data.get("hook") or HOOK),
                   guess=[str(g).strip() for g in guess if str(g).strip()],
                   answer=str(data.get("answer") or "").strip(),
                   question=str(data.get("question") or QUESTION), moments=moments)


@dataclass
class DiaryResult:
    output_path: str
    image_path: str  # the sketchbook of the day
    credits_path: Optional[str]
    duration: float
    moments: List[str]
    elapsed: float
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- the time of day

_CLOCK = re.compile(r"^(\d{1,2})(?:[:.h](\d{2}))?\s*(?:([ap])\.?\s*m?\.?)?$", re.IGNORECASE)
_HOURS = {"dawn": 6.0, "sunrise": 6.0, "morning": 8.0, "breakfast": 7.5, "noon": 12.0,
          "midday": 12.0, "lunch": 12.5, "afternoon": 15.0, "sunset": 18.5, "dinner": 18.5,
          "evening": 19.0, "bedtime": 21.5, "night": 22.0, "midnight": 24.0}


def hour_of(text: str) -> Optional[float]:
    """"7:00 AM" -> 7.0, "19:30" -> 19.5, "noon" -> 12.0; None when unknown."""
    t = text.strip().lower()
    if t in _HOURS:
        return _HOURS[t]
    m = _CLOCK.match(t)
    if not m:
        return None
    h, mins = int(m.group(1)), int(m.group(2) or 0)
    if mins > 59 or h > 24:
        return None
    if m.group(3):
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if m.group(3) == "p" else 0)
    return h + mins / 60


def day_shares(moments: List[Moment]) -> List[float]:
    """Where each moment lies on the day bar (6 AM = 0, 11 PM = 1); moments
    without a readable time are spread evenly."""
    n = len(moments)
    out = []
    for i, m in enumerate(moments):
        h = hour_of(m.time)
        if h is None:
            out.append((i + 0.5) / n)
        else:
            out.append(float(np.clip(((h + 24 if h < 5 else h) - 6.0) / 17.0, 0.0, 1.0)))
    return out


def is_night(text: str) -> bool:
    h = hour_of(text)
    return h is not None and (h >= 19.5 or h < 5.5)


# --------------------------------------------------------------------------- the pet's voice

def pet_voice(kind: str, voice: str = "", pitch: float = 0.0) -> Tuple[Optional[str], float]:
    """The speech voice and pitch for a kind of pet (None when the computer
    cannot read aloud)."""
    from .cartoon import voices
    have = voices()
    if not have:
        return None, 1.0
    default, p = VOICES.get(kind, VOICES["other"])
    name = voice if voice in have else (default if default in have else have[0])
    return name, float(np.clip(pitch or p, 0.6, 1.8))


class _Speaker:
    """Lines read in the pet's voice, as samples; None for every line when the
    voice cannot be used (and the reason in ``problems``)."""

    def __init__(self, voice: Optional[str], pitch: float, tmp: str):
        self.voice, self.pitch, self.tmp = voice, pitch, tmp
        self.problems: List[str] = []
        self.count = 0

    def say(self, text: str) -> Optional[np.ndarray]:
        if not self.voice or not text.strip():
            return None
        from .cartoon import speak
        self.count += 1
        try:
            return speak(text, self.voice, self.tmp, f"pet{self.count}", self.pitch, RATE)
        except RenderError as exc:
            self.problems.append(f"The pet's voice is off: {exc}")
            self.voice = None
            return None


def _seconds(samples: Optional[np.ndarray]) -> float:
    return 0.0 if samples is None else len(samples) / SAMPLE_RATE


# --------------------------------------------------------------------------- layout

@dataclass
class DiaryLayout:
    draw_box: Box  # where each moment is drawn
    x: int  # the text column
    width: int
    unit: int
    counter_y: int
    time_y: int
    title_y: int
    bubble: Box  # room for the thought
    bar_y: int
    time_size: int
    title_size: int
    text_size: int
    wide: bool


def diary_layout(width: int, height: int) -> DiaryLayout:
    """The drawing on the left and the text on the right (landscape), or the
    drawing on top and the text below it, all above the bottom 16 % of the
    screen where Shorts shows its own captions (portrait and square)."""
    W, H = width, height
    u = min(W, H)
    wide = W > H * 1.2
    if wide:
        box = (int(0.035 * W), int(0.08 * H), int(0.53 * W), int(0.84 * H))
        x, top, w, bottom = int(0.6 * W), int(0.1 * H), int(0.36 * W), int(0.93 * H)
        text = int(0.046 * H)
    else:
        bottom = int(0.84 * H) if H > W * 1.2 else int(0.97 * H)
        bh = int(min(0.47 * H, bottom - 0.56 * u - 0.06 * H))
        box = (int(0.05 * W), int(0.04 * H), int(0.9 * W), bh)
        x, top, w = int(0.07 * W), box[1] + bh + int(0.015 * H), int(0.86 * W)
        text = int(0.048 * u)  # read on a phone
    bubble_y = top + int(0.225 * u)
    bubble = (x, bubble_y, w, max(int(0.12 * u), bottom - int(0.105 * u) - bubble_y))
    return DiaryLayout(box, x, w, u, top, top + int(0.045 * u), top + int(0.14 * u), bubble,
                       bottom - int(0.045 * u), int(0.075 * u), int(0.052 * u), text, wide)


# --------------------------------------------------------------------------- little drawings

def _stroke(size: float) -> int:
    return max(2, int(size * 0.07))


def paw_sprite(size: int, colour=INK, alpha: float = 0.9) -> Sprite:
    """A paw print, toes up."""
    def draw(d, s):
        c = _rgb255(colour, alpha)
        S = size * s
        d.ellipse([0.22 * S, 0.45 * S, 0.78 * S, 0.95 * S], fill=c)
        for cx, cy, r in ((0.16, 0.38, 0.11), (0.37, 0.18, 0.12), (0.63, 0.18, 0.12), (0.84, 0.38, 0.11)):
            d.ellipse([(cx - r) * S, (cy - r * 1.25) * S, (cx + r) * S, (cy + r * 1.25) * S], fill=c)
    return _render(draw, (size, size))


def clock_sprite(hour: Optional[float], size: int) -> Sprite:
    """A pencil clock face showing ``hour``."""
    def draw(d, s):
        S, line = size * s, _stroke(size) * s
        ink = _rgb255(INK)
        d.ellipse([line, line, S - line, S - line], fill=_rgb255((1.0, 0.99, 0.95)), outline=ink, width=line)
        c = S / 2
        for k in range(12):
            a = k / 12 * 2 * math.pi
            r0, r1 = 0.36 * S, 0.42 * S
            d.line([(c + r0 * math.sin(a), c - r0 * math.cos(a)), (c + r1 * math.sin(a), c - r1 * math.cos(a))],
                   fill=ink, width=max(1, line // 2))
        h = 7.0 if hour is None else hour
        for share, length, width in (((h % 12) / 12, 0.22, line), ((h % 1.0), 0.32, max(1, line * 2 // 3))):
            a = share * 2 * math.pi
            d.line([(c, c), (c + length * S * math.sin(a), c - length * S * math.cos(a))], fill=ink, width=width)
        d.ellipse([c - line, c - line, c + line, c + line], fill=ink)
    return _render(draw, (size, size))


def sun_sprite(size: int) -> Sprite:
    def draw(d, s):
        S, line = size * s, _stroke(size) * s
        c = S / 2
        for k in range(8):
            a = k / 8 * 2 * math.pi
            d.line([(c + 0.32 * S * math.cos(a), c + 0.32 * S * math.sin(a)),
                    (c + 0.47 * S * math.cos(a), c + 0.47 * S * math.sin(a))], fill=_rgb255(INK), width=line)
        d.ellipse([0.24 * S, 0.24 * S, 0.76 * S, 0.76 * S], fill=_rgb255((1.0, 0.84, 0.3)),
                  outline=_rgb255(INK), width=line)
    return _render(draw, (size, size))


def moon_sprite(size: int) -> Sprite:
    def draw(d, s):
        S, line = size * s, _stroke(size) * s
        d.ellipse([0.12 * S, 0.12 * S, 0.88 * S, 0.88 * S], fill=_rgb255((0.98, 0.93, 0.7)),
                  outline=_rgb255(INK), width=line)
    img = _render(draw, (size, size))
    # the crescent: a bite out of the full moon
    yy, xx = np.mgrid[0:size, 0:size].astype(F32)
    bite = ((xx - 0.72 * size) ** 2 + (yy - 0.4 * size) ** 2) < (0.36 * size) ** 2
    img.alpha[bite] = 0.0
    return img


def _rec_dot(size: int) -> Sprite:
    def draw(d, s):
        d.ellipse([0, 0, size * s - 1, size * s - 1], fill=_rgb255(REC))
    return _render(draw, (size, size))


def _check(d, x, y, s, colour) -> None:
    d.line([(x + 0.1 * s, y + 0.55 * s), (x + 0.4 * s, y + 0.85 * s), (x + 0.95 * s, y + 0.15 * s)],
           fill=colour, width=max(2, int(s * 0.16)), joint="curve")


def _fitted_pill(text: str, height: int, width: int, fill, text_fill, icon=None, outline=None) -> Sprite:
    """A pill no wider than ``width`` (smaller letters when needed)."""
    pill = _pill(text, height, fill, text_fill, icon, outline)
    while pill.size[0] > width and height > 16:
        height = int(height * 0.9)
        pill = _pill(text, height, fill, text_fill, icon, outline)
    return pill


def _bubble_sprite(w: int, h: int) -> Sprite:
    """A thought bubble: paper inside, a doubled pencil line around."""
    def draw(d, s):
        line = max(2, int(0.006 * min(w * 3, h * 6))) * s
        r = min(w, h) * 0.22 * s
        d.rounded_rectangle([line, line, w * s - line, h * s - line], radius=r,
                            fill=_rgb255((1.0, 0.995, 0.975), 0.92), outline=_rgb255(INK), width=line)
        d.rounded_rectangle([line * 2.4, line * 1.6, w * s - line * 1.4, h * s - line * 2.2], radius=r * 0.96,
                            outline=_rgb255(INK, 0.35), width=max(1, line // 2))
    return _render(draw, (w, h))


def _circle_sprite(size: int) -> Sprite:
    def draw(d, s):
        line = max(2, int(size * 0.1)) * s
        d.ellipse([line, line, size * s - line, size * s - line], fill=_rgb255((1.0, 0.995, 0.975), 0.92),
                  outline=_rgb255(INK), width=line)
    return _render(draw, (size, size))


# --------------------------------------------------------------------------- the day bar

class DayBar:
    """A pencil line from sunrise to bedtime with a dot for every moment; a
    paw print walks to the current one."""

    def __init__(self, shares: List[float], x: int, y: int, width: int, unit: int):
        self.shares, self.y = shares, y
        icon = int(0.045 * unit)
        self.sun, self.moon = sun_sprite(icon), moon_sprite(icon)
        self.x0, self.x1 = x + icon * 0.6, x + width - icon * 0.6
        self.left, self.right = x + icon * 0.6, x + width - icon * 0.6
        self.line_x0, self.line_x1 = x + icon * 1.35, x + width - icon * 1.35
        self.weight = max(2, int(0.004 * unit))
        dot = int(0.018 * unit)
        self.done = _render(lambda d, s: d.ellipse([0, 0, dot * s - 1, dot * s - 1], fill=_rgb255(INK)), (dot, dot))
        self.todo = _render(lambda d, s: d.ellipse([s, s, dot * s - s, dot * s - s], outline=_rgb255(INK, 0.6),
                                                   width=max(1, self.weight * s // 2)), (dot, dot))
        self.paw = paw_sprite(int(0.05 * unit))

    def at(self, share: float) -> float:
        return self.line_x0 + share * (self.line_x1 - self.line_x0)

    def paint(self, frame: np.ndarray, current: int, t: float, appear: float = 1.0) -> None:
        if appear <= 0.0:
            return
        y = int(self.y)
        x0, x1 = int(self.line_x0), int(self.line_x1)
        region = frame[y - self.weight // 2:y - self.weight // 2 + self.weight, x0:x1]
        region += (np.asarray(INK, F32) - region) * 0.75 * appear
        self.sun.paste(frame, self.left, y, opacity=appear)
        self.moon.paste(frame, self.right, y, opacity=appear)
        for k, share in enumerate(self.shares):
            (self.done if k < current else self.todo).paste(frame, self.at(share), y, opacity=appear)
        before = self.shares[current - 1] if current > 0 else 0.0
        walk = smooth(_phase(t, 0.3, 0.8))
        px = self.at(before + (self.shares[current] - before) * walk)
        hop = math.sin(math.pi * walk) * 0.012 * (self.line_x1 - self.line_x0) if 0 < walk < 1 else 0.0
        self.paw.paste(frame, px, y - self.paw.size[1] * 0.55 - hop, angle=-90, opacity=appear)


# --------------------------------------------------------------------------- the text of a moment

class _DiaryText:
    """Everything written beside a moment, as a function of the scene's time:
    the counter, the time with its clock, the title, the pet's thought in a
    bubble (written while it is read aloud) and the day bar. Used by
    :func:`animals.render_parts` like the animals' facts."""

    def __init__(self, m: Moment, index: int, count: int, pet: str, lay: DiaryLayout, drawing: float,
                 voice: float, bar: DayBar):
        self.lay, self.index, self.bar = lay, index, bar
        u = lay.unit
        self.counter = text_sprite(f"{pet.upper()}'S DAY  ·  {index + 1} / {count}", BOLD_FONTS,
                                   int(0.03 * u), MUTED, tracking=0.08, max_width=lay.width)
        self.clock = clock_sprite(hour_of(m.time), int(lay.time_size * 1.05))
        self.time = _text(m.time.strip() or f"Moment {index + 1}", lay.time_size)
        self.time_x = lay.x + int(self.clock.size[0] * 1.2)
        self.mark = _marker(self.time.ink[1] - self.time.ink[0] + 12, lay.time_size * 0.55)
        size = lay.title_size
        self.title = _text(m.title.strip(), size)
        while self.title.sprite.size[0] > lay.width and size > 14:
            size = int(size * 0.9)
            self.title = _text(m.title.strip(), size)
        self.under = _underline(self.title.ink[1] - self.title.ink[0], size) if m.title.strip() else None
        self.lines: List[Tuple[object, int, int]] = []
        self.bubble = None
        thought = m.thought.strip()
        if thought:
            bx, by, bw, bh = lay.bubble
            pad = int(0.035 * u)
            size = lay.text_size
            while True:
                words = _wrap(thought, size, bw - 2 * pad)
                step = int(size * 1.3)
                if len(words) * step + 2 * pad <= bh or size <= 16:
                    break
                size = int(size * 0.9)
            height = min(bh, len(words) * step + 2 * pad + int(0.2 * size))
            self.bubble = _bubble_sprite(bw, height)
            self.bubble_at = (bx + bw / 2, by + height / 2)
            top = by + (height - len(words) * step) / 2 - int(0.08 * size)
            self.lines = [(_text(line, size), bx + pad, int(top + k * step)) for k, line in enumerate(words)]
            dots = [int(0.022 * u), int(0.015 * u)]
            if lay.wide:  # little circles trailing towards the drawing
                spots = [(bx - 0.018 * u, by + height * 0.55), (bx - 0.045 * u, by + height * 0.66)]
            else:
                spots = [(bx + bw * 0.86, by - 0.02 * u), (bx + bw * 0.9, by - 0.048 * u)]
            self.dots = [(_circle_sprite(s), p) for s, p in zip(dots, spots)]
        chars = [max(1, ln[0].ink[1] - ln[0].ink[0]) for ln in self.lines]
        span = max(voice * 0.92, 0.055 * len(thought), 0.6) if thought else 0.0
        t = THOUGHT_AT + 0.15
        self.writes = []
        for c in chars:
            d = span * c / sum(chars)
            self.writes.append((t, t + d))
            t += d
        self.end = max(drawing + 0.8, THOUGHT_AT + voice + 0.7, t + 1.0)
        self.sounds = [(0.08, "pop"), (0.55, "swish")]
        if self.bubble is not None:
            self.sounds.append((THOUGHT_AT - 0.1, "pop"))

    def paint(self, frame: np.ndarray, t: float) -> None:
        lay = self.lay
        self.counter.paste_left(frame, lay.x, lay.counter_y)
        c = _phase(t, 0.05, 0.3)
        cy = lay.time_y + self.time.sprite.size[1] / 2
        self.clock.paste(frame, lay.x + self.clock.size[0] / 2, cy, scale=back(c), opacity=min(1.0, 3 * c))
        self.mark.paste_left(frame, self.time_x + self.time.ink[0] - 6, cy - self.mark.size[1] * 0.15,
                             reveal=_done(t, 0.55, 0.85))
        self.time.sprite.paste_left(frame, self.time_x, lay.time_y, reveal=_written(self.time, _done(t, 0.1, 0.55)))
        self.title.sprite.paste_left(frame, lay.x, lay.title_y, reveal=_written(self.title, _done(t, 0.6, 1.2)))
        if self.under is not None:
            self.under.paste_left(frame, lay.x + self.title.ink[0] - 4,
                                  lay.title_y + int(self.title.sprite.size[1] * 0.86), reveal=_done(t, 1.2, 1.45))
        if self.bubble is not None:
            for k, (dot, (dx, dy)) in enumerate(self.dots):
                a = _phase(t, THOUGHT_AT - 0.4 + 0.12 * k, 0.2)
                dot.paste(frame, dx, dy, scale=back(a), opacity=min(1.0, 3 * a))
            b = _phase(t, THOUGHT_AT - 0.15, 0.3)
            self.bubble.paste(frame, *self.bubble_at, scale=0.6 + 0.4 * back(b), opacity=min(1.0, 3 * b))
            if b >= 1.0:
                for (text, x, y), (t0, t1) in zip(self.lines, self.writes):
                    text.sprite.paste_left(frame, x, y, reveal=_written(text, _done(t, t0, t1)))
        self.bar.paint(frame, self.index, t)


# --------------------------------------------------------------------------- captions

class _Captions:
    """Short-video captions: the words appear one by one as they are said,
    the word being said in yellow, two lines at a time."""

    def __init__(self, text: str, W: int, H: int, start: float, seconds: float, y: float):
        u = min(W, H)
        size = int(0.064 * u)
        stroke = max(3, size // 9)
        words = text.split()
        self.white = [text_sprite(w, BOLD_FONTS, size, WHITE, stroke=stroke, stroke_fill=(0.05, 0.05, 0.05))
                      for w in words]
        self.said = [text_sprite(w, BOLD_FONTS, size, SAID, stroke=stroke, stroke_fill=(0.05, 0.05, 0.05))
                     for w in words]
        chars = [len(w) + 1 for w in words]
        total = max(seconds, 0.3 * len(words))
        self.times = [start + total * sum(chars[:k]) / max(sum(chars), 1) for k in range(len(words))]
        limit = (0.62 if W > H * 1.2 else 0.86) * W
        gap = size * 0.3
        rows: List[List[int]] = [[]]
        width = 0.0
        for k, sp in enumerate(self.white):
            w = sp.size[0] - 2 * stroke
            if rows[-1] and width + gap + w > limit:
                rows.append([])
                width = 0.0
            width += (gap if rows[-1] else 0) + w
            rows[-1].append(k)
        self.place: Dict[int, Tuple[int, float, float]] = {}  # word -> (page, x, y)
        line_h = size * 1.25
        for r, row in enumerate(rows):
            widths = [self.white[k].size[0] - 2 * stroke for k in row]
            x = W / 2 - (sum(widths) + gap * (len(row) - 1)) / 2
            for k, w in zip(row, widths):
                self.place[k] = (r // 2, x + w / 2, y + (r % 2) * line_h)
                x += w + gap

    def paint(self, frame: np.ndarray, t: float) -> None:
        now = max([k for k, at in enumerate(self.times) if at <= t], default=-1)
        if now < 0:
            return
        page = self.place[now][0]
        for k in range(now + 1):
            p, x, y = self.place[k]
            if p != page:
                continue
            pop = back(_phase(t, self.times[k], 0.12))
            (self.said if k == now else self.white)[k].paste(frame, x, y, scale=0.7 + 0.3 * pop)


# --------------------------------------------------------------------------- pieces of video

class _Vlog:
    """The pet camera's look over real footage: the time in a pill (top left)
    and a blinking red REC dot with the camera's name (top right)."""

    def __init__(self, W: int, H: int, time_label: str, cam: str):
        u = min(W, H)
        self.W, self.H = W, H
        self.top = (0.05 if W > H * 1.2 else 0.07) * H  # below the top buttons in Shorts
        self.pill = _pill(time_label, int(0.062 * u), WHITE, INK) if time_label else None
        self.cam = text_sprite(cam, BOLD_FONTS, int(0.032 * u), WHITE, stroke=2, stroke_fill=(0.1, 0.1, 0.1),
                               tracking=0.12)
        self.dot = _rec_dot(int(0.026 * u))
        self.margin = 0.035 * W

    def paint(self, frame: np.ndarray, t: float, opacity: float = 1.0) -> None:
        if self.pill is not None:
            self.pill.paste(frame, self.margin + self.pill.size[0] / 2, self.top + self.pill.size[1] / 2,
                            opacity=0.92 * opacity)
        x = self.W - self.margin - self.cam.size[0] / 2
        self.cam.paste(frame, x, self.top + self.cam.size[1] / 2, opacity=opacity)
        if (t % 1.0) < 0.6:
            self.dot.paste(frame, x - self.cam.size[0] / 2 - self.dot.size[0], self.top + self.cam.size[1] / 2,
                           opacity=opacity)


class _PawWipe:
    """Paw prints walk across the screen while the next page is uncovered behind them."""

    def __init__(self, W: int, H: int, count: int = 7):
        u = min(W, H)
        self.paw = paw_sprite(int(0.1 * u), alpha=0.85)
        self.ring = paw_sprite(int(0.118 * u), colour=(1.0, 1.0, 1.0), alpha=0.7)
        self.W = W
        k = np.arange(count)
        self.xs = (0.08 + 0.84 * k / (count - 1)) * W
        self.ys = (0.62 - 0.22 * k / (count - 1)) * H + np.where(k % 2 == 0, 0.045, -0.045) * u
        self.angle = math.degrees(math.atan2(-1.0, 0.22 * H / W))
        self.times = 0.62 * k / (count - 1)
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        self.diag = (xx + 0.25 * (H - yy)) / (W + 0.25 * H)

    def apply(self, frame: np.ndarray, target: np.ndarray, p: float) -> np.ndarray:
        front = smooth(min(1.0, p / 0.85)) * 1.1 - 0.05
        m = np.clip((front - self.diag) / 0.04 + 0.5, 0.0, 1.0)[..., None]
        out = frame * (1.0 - m) + target * m
        fade = 1.0 - _phase(p, 0.8, 0.2)
        for x, y, t0 in zip(self.xs, self.ys, self.times):
            a = _phase(p, t0, 0.1)
            if a > 0:
                look = dict(scale=0.6 + 0.4 * back(a), opacity=min(1.0, 3 * a) * fade, angle=self.angle)
                self.ring.paste(out, x, y, **look)
                self.paw.paste(out, x, y, **look)
        return out

    def sounds(self, at: float, seconds: float) -> List[Tuple[float, str]]:
        return [(at + t * seconds, "step") for t in self.times]


def _wash(base: np.ndarray, colour: np.ndarray, box: Box, seed: int):
    """A function painting colour over the graphite in ``box`` up to share p."""
    x, y, w, h = box
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    wobble = cv2.resize(rng.standard_normal((6, 8)).astype(F32), (w, h), interpolation=cv2.INTER_CUBIC)
    sweep = (xx + 0.6 * yy) / (w + 0.6 * h) + 0.05 * wobble

    def paint(p: float) -> np.ndarray:
        frame = base.copy()
        m = np.clip((p - sweep) / 0.05 + 0.5, 0.0, 1.0)[..., None]
        region = frame[y:y + h, x:x + w]
        region[:] = region * (1.0 - m) + colour * m
        region += np.exp(-((sweep - p) / 0.018) ** 2)[..., None] * 0.3 * (1.0 - region)
        return frame
    return paint


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from .travel import _cover
    return _cover(img, max(w, 2), max(h, 2))


def alive_clip(page: np.ndarray, colour: np.ndarray, footage: Footage, lay: DiaryLayout, fps: int,
               hold: float, vlog: _Vlog, caption: str, said: Optional[np.ndarray], target: np.ndarray,
               seed: int = 0) -> Clip:
    """The drawing comes to life: colour washes over the graphite, the real
    clip grows to the whole screen as the pet's camera and the punchline is
    captioned word by word; then paw prints walk on to ``target``."""
    H, W = page.shape[:2]
    base = _f(page)
    x, y, w, h = lay.draw_box
    colour = np.clip(cv2.resize(colour, (w, h), interpolation=cv2.INTER_AREA), 0.0, 1.0)
    wash = _wash(base, colour, lay.draw_box, seed)
    shown = WASH + MELT + GROW
    total = shown + hold + PAW
    tgt = _f(target)
    shade = np.clip((np.arange(H, dtype=F32) / H - 0.45) / 0.55, 0.0, 1.0) ** 1.4 * 0.6
    caption_y = (0.8 if lay.wide else 0.66) * H
    caps = _Captions(caption, W, H, shown + 0.15, _seconds(said), caption_y) if caption.strip() else None
    paws = _PawWipe(W, H)

    def frames() -> Iterator[np.ndarray]:
        feed = footage.frames()
        try:
            for i in range(int(round(total * fps))):
                t = i / fps
                if t < WASH:
                    yield _u8(np.clip(wash(smooth(t / WASH) * 1.15 - 0.06), 0.0, 1.0))
                    continue
                real = next(feed)
                if t < WASH + MELT:
                    frame = base.copy()
                    k = smooth((t - WASH) / MELT)
                    frame[y:y + h, x:x + w] = colour * (1.0 - k) + _fit(real, w, h) * k
                elif t < shown:
                    k = smooth((t - WASH - MELT) / GROW)
                    rx, ry = int(round(x * (1 - k))), int(round(y * (1 - k)))
                    rw, rh = int(round(w + (W - w) * k)), int(round(h + (H - h) * k))
                    rw, rh = min(rw, W - rx), min(rh, H - ry)
                    frame = base * (1.0 - 0.35 * k)
                    frame[ry:ry + rh, rx:rx + rw] = _fit(real, rw, rh)
                else:
                    u = t - shown
                    frame = real * (1.0 - shade * (1.0 if caps else 0.4))[:, None, None]
                    vlog.paint(frame, u, _phase(u, 0.0, 0.25))
                    if caps is not None:
                        caps.paint(frame, t)
                    if u > hold:
                        frame = paws.apply(frame, tgt, (u - hold) / PAW)
                yield _u8(np.clip(frame, 0.0, 1.0))
        finally:
            feed.close()

    sounds = [(0.05, "sparkle"), (WASH + MELT + 0.05, "swish"), (shown + 0.05, "pop")]
    sounds += paws.sounds(shown + hold, PAW)
    speech = [(shown + 0.15, said)] if said is not None else []
    return Clip(total, frames, sounds, speech=speech)


def cold_open_clip(footage: Footage, still: np.ndarray, vlog: _Vlog, hook: str, hook_said: Optional[np.ndarray],
                   explain_said: Optional[np.ndarray], blank: np.ndarray, fps: int, seed: int = 0) -> Clip:
    """The day's big moment plays, freezes with a record scratch and turns to
    pencil; the pet asks how it got there and the day rewinds."""
    H, W = blank.shape[:2]
    u = min(W, H)
    wide = W > H * 1.2
    paper = _f(blank)
    hook_at = PLAY + 0.3
    explain_at = hook_at + max(_seconds(hook_said), 1.4) + 0.2
    rewind_at = explain_at + max(_seconds(explain_said), 0.9) + 0.25
    total = rewind_at + REWIND
    lines = _wrap_display(hook.upper(), int(0.11 * u), int(0.9 * W))
    hook_s = [text_sprite(line, DISPLAY_FONTS, int(0.11 * u), WHITE, stroke=max(3, u // 110),
                          stroke_fill=INK, max_width=int(0.92 * W)) for line in lines]
    explain_s = text_sprite(EXPLAIN, HAND_FONTS, int(0.085 * u), INK, max_width=int(0.8 * W))
    strip = _render(lambda d, s: d.rounded_rectangle(
        [0, 0, (explain_s.size[0] + 0.06 * u) * s, explain_s.size[1] * 1.2 * s], radius=0.03 * u * s,
        fill=_rgb255((1.0, 0.99, 0.95), 0.9)), (int(explain_s.size[0] + 0.06 * u), int(explain_s.size[1] * 1.2)))
    arrows = _render(lambda d, s: [d.polygon([((ox + 0.5) * 0.09 * u * s, 0), (ox * 0.09 * u * s, 0.045 * u * s),
                                               ((ox + 0.5) * 0.09 * u * s, 0.09 * u * s)], fill=_rgb255(WHITE))
                                   for ox in (0.0, 0.5)], (int(0.1 * u), int(0.09 * u)))
    hook_y = (0.2 if wide else 0.26) * H
    rng = np.random.default_rng(seed)
    bands = rng.uniform(-1, 1, 24).astype(F32)
    state: Dict[str, np.ndarray] = {}

    def frozen() -> Tuple[np.ndarray, np.ndarray]:
        if "sketch" not in state:
            last = state.get("last", _fit(still, W, H))
            state["frame"] = last
            drawn = graphite_drawing(last, (W, H), seed=seed, border=0.0)
            state["sketch"] = drawn
        return state["frame"], state["sketch"]

    def frames() -> Iterator[np.ndarray]:
        feed = footage.frames()
        try:
            for i in range(int(round(total * fps))):
                t = i / fps
                if t < PLAY:
                    real = next(feed)
                    state["last"] = real
                    frame = real.copy()
                    vlog.paint(frame, t)
                    yield _u8(np.clip(frame, 0.0, 1.0))
                    continue
                real, sketch = frozen()
                k = smooth(_phase(t, PLAY, DRAIN))
                frame = real * (1.0 - k) + sketch * k
                z = 1.0 + 0.05 * smooth(_phase(t, PLAY, 1.2))
                if z > 1.0001:
                    m = np.array([[z, 0, W / 2 - z * W / 2], [0, z, H / 2 - z * H / 2]], np.float32)
                    frame = cv2.warpAffine(frame, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                flash = 1.0 - _phase(t, PLAY, 0.12)
                if flash < 1.0:
                    frame = frame + (1.0 - frame) * 0.7 * flash
                for n, sp in enumerate(hook_s):
                    a = _phase(t, hook_at + 0.12 * n, 0.25)
                    sp.paste(frame, W / 2, hook_y + n * sp.size[1] * 0.92, scale=1.0 + 0.5 * (1.0 - back(a)),
                             opacity=min(1.0, 2 * a))
                e = _phase(t, explain_at, 0.3)
                if e > 0:
                    ey = (0.78 if wide else 0.7) * H
                    strip.paste(frame, W / 2, ey, opacity=e)
                    explain_s.paste_left(frame, W / 2 - explain_s.size[0] / 2, ey - explain_s.size[1] / 2,
                                         reveal=_done(t, explain_at, explain_at + 0.6))
                r = _phase(t, rewind_at, REWIND)
                if r > 0:  # the tape winds back: torn bands, grey, then blank paper
                    shift = (np.repeat(bands, int(math.ceil(H / len(bands))))[:H] * 0.06 * W * math.sin(math.pi * r))
                    rows = np.arange(H)
                    out = np.empty_like(frame)
                    for y0 in range(0, H, max(1, H // len(bands))):
                        y1 = min(H, y0 + max(1, H // len(bands)))
                        out[y0:y1] = np.roll(frame[y0:y1], int(shift[y0]), axis=1)
                    grey = out.mean(axis=2, keepdims=True)
                    out = out * (1 - 0.6 * r) + grey * 0.6 * r
                    out[(rows % 6) < 2] *= 0.85
                    arrows.paste(out, W / 2, H / 2, opacity=min(1.0, 4 * r) * (1.0 - _phase(r, 0.7, 0.3)))
                    w = smooth(_phase(r, 0.55, 0.45))
                    frame = out * (1.0 - w) + paper * w
                yield _u8(np.clip(frame, 0.0, 1.0))
        finally:
            feed.close()

    speech = []
    if hook_said is not None:
        speech.append((hook_at + 0.05, hook_said))
    if explain_said is not None:
        speech.append((explain_at + 0.05, explain_said))
    sounds = [(PLAY - 0.02, "scratch"), (hook_at, "thud"), (explain_at, "pop"), (rewind_at, "rewind")]
    return Clip(total, frames, sounds, speech=speech)


def _wrap_display(text: str, size: int, width: int) -> List[str]:
    from .showtime import _wrap as wrap
    return wrap(text, DISPLAY_FONTS, size, width)


def title_clip(blank: np.ndarray, heading: str, pet: str, fps: int, seconds: float = 2.6) -> Clip:
    """The diary's title written on the first page, with a paw stamp."""
    H, W = blank.shape[:2]
    u = min(W, H)
    paper = _f(blank)
    lines = _wrap(heading, int(0.1 * u), int(0.86 * W))
    texts = [_text(line, int(0.1 * u)) for line in lines]
    kicker = text_sprite(f"TOLD BY {pet.upper()}  ·  DRAWN IN PENCIL", BOLD_FONTS, int(0.034 * u), MUTED,
                         tracking=0.12, max_width=int(0.9 * W))
    paw = paw_sprite(int(0.12 * u), alpha=0.85)
    step = int(0.12 * u)
    top = H * 0.46 - len(lines) * step / 2
    each = 0.9 / max(len(lines), 1)

    def frame_at(t: float) -> np.ndarray:
        frame = paper.copy()
        for k, text in enumerate(texts):
            x = W / 2 - (text.ink[1] - text.ink[0]) / 2 - text.ink[0]
            text.sprite.paste_left(frame, x, top + k * step, reveal=_written(text, _done(t, 0.1 + k * each,
                                                                                        0.1 + (k + 1) * each)))
        kicker.paste(frame, W / 2, top + len(lines) * step + 0.05 * u, opacity=_phase(t, 1.0, 0.3))
        a = _phase(t, 1.25, 0.2)
        paw.paste(frame, W / 2 + 0.36 * W * (1 if W > H else 0.6), top - 0.04 * u, scale=1.0 + 0.6 * (1 - back(a)),
                  opacity=min(1.0, 3 * a), angle=-20)
        fade = _phase(t, seconds - 0.35, 0.35)
        return paper * fade + frame * (1.0 - fade)

    def frames() -> Iterator[np.ndarray]:
        for i in range(int(round(seconds * fps))):
            yield _u8(np.clip(frame_at(i / fps), 0.0, 1.0))

    return Clip(seconds, frames, [(0.12, "swish"), (1.25, "thud")])


def guess_clip(blank: np.ndarray, page: np.ndarray, lay: DiaryLayout, options: List[str], answer: int,
               said: Optional[np.ndarray], fps: int) -> Clip:
    """"What happens next?": the coming drawing blurred, the answers as pills,
    a 3-2-1 countdown and the right answer turning green."""
    H, W = blank.shape[:2]
    u = lay.unit
    paper = _f(blank)
    x, y, w, h = lay.draw_box
    hidden = _f(page).copy()
    region = hidden[y:y + h, x:x + w]
    region[:] = PAPER + (cv2.GaussianBlur(region, (0, 0), 0.035 * u) - PAPER) * 0.45
    size = lay.time_size
    question = _text("What happens next?", size)
    while question.sprite.size[0] > lay.width and size > 16:
        size = int(size * 0.9)
        question = _text("What happens next?", size)
    mark = text_sprite("?", DISPLAY_FONTS, int(0.3 * u), INK)
    ph = int(0.088 * u)
    letters = "ABCD"
    pills = [_fitted_pill(f"{letters[k]}   {o}", ph, lay.width, WHITE, INK, outline=INK) for k, o in enumerate(options)]
    right = _fitted_pill(f"{letters[answer]}   {options[answer]}", pills[answer].size[1], lay.width + ph, GREEN,
                         WHITE, icon=_check, outline=GREEN)
    gap = int(0.022 * u)
    top = lay.title_y
    count = [_badge(n, int(0.16 * u)) for n in (3, 2, 1)]
    start = 1.5
    reveal = start + 3.0
    total = reveal + 1.6

    def frame_at(t: float) -> np.ndarray:
        frame = paper + (hidden - paper) * _phase(t, 0.0, 0.3)
        question.sprite.paste_left(frame, lay.x, lay.time_y, reveal=_written(question, _done(t, 0.15, 0.85)))
        for k, pill in enumerate(pills):
            a = _phase(t, 0.7 + 0.18 * k, 0.25)
            py = top + k * (pill.size[1] + gap) + pill.size[1] / 2
            if t >= reveal and k == answer:
                b = _phase(t, reveal, 0.25)
                right.paste(frame, lay.x + right.size[0] / 2, py, scale=1.0 + 0.12 * math.sin(math.pi * b))
            else:
                dim = 0.4 if t >= reveal else 1.0
                pill.paste(frame, lay.x + pill.size[0] / 2 - (1 - back(a)) * 0.05 * W, py, opacity=min(1.0, 3 * a) * dim)
        cx, cy = x + w / 2, y + h / 2
        if t < start:
            a = _phase(t, 0.3, 0.3)
            mark.paste(frame, cx, cy, scale=0.5 + 0.5 * back(a), opacity=min(1.0, 3 * a))
        elif t < reveal:
            n = min(2, int(t - start))
            a = _phase(t, start + n, 0.2)
            count[n].paste(frame, cx, cy, scale=0.6 + 0.4 * back(a), opacity=min(1.0, 4 * a))
        fade = _phase(t, total - 0.4, 0.4)
        return paper * fade + frame * (1.0 - fade)

    def frames() -> Iterator[np.ndarray]:
        for i in range(int(round(total * fps))):
            yield _u8(np.clip(frame_at(i / fps), 0.0, 1.0))

    sounds = [(0.15, "swish")] + [(0.7 + 0.18 * k, "pop") for k in range(len(pills))]
    sounds += [(start + n, "tick") for n in range(3)] + [(reveal, "ding")]
    speech = [(0.2, said)] if said is not None else []
    return Clip(total, frames, sounds, speech=speech)


# --------------------------------------------------------------------------- the video

def render_pets(diary: Diary, output_path: str, width: int = 1920, height: int = 1080, fps: int = 30,
                drawing_seconds: float = 8.0, reveal_seconds: float = REVEAL, music: Optional[str] = None,
                tune: Optional[str] = "happy", voice: str = "", pitch: float = 0.0, guess: bool = True,
                cold_open: bool = True, outro: bool = True, sound_effects: bool = True,
                progress: Optional[ProgressFn] = None, crf: int = 20, preset: str = "fast") -> DiaryResult:
    """Draw ``diary``'s moments one after another, each told by the pet and
    brought to life; save the video, the sketchbook of the day (PNG) and the
    photo credits.

    ``music`` is the user's own music; without it ``tune`` ("happy", "calm"
    or None) is played, turned down while the pet speaks. ``voice`` and
    ``pitch`` choose the pet's voice (by default from its kind).
    """
    report = progress or (lambda f, m: None)
    started = time.perf_counter()
    moments = diary.moments
    if not diary.pet.strip():
        raise InputError("Give the pet a name.")
    if not moments:
        raise InputError("Add at least one moment of the day.")
    if len(moments) > 12:
        raise InputError("Add at most 12 moments.")
    if not 5.0 <= drawing_seconds <= 30.0:
        raise InputError("Drawing time must be between 5 and 30 seconds.")
    if not 1.5 <= reveal_seconds <= 10.0:
        raise InputError("The real clip must last between 1.5 and 10 seconds.")
    for k, m in enumerate(moments):
        if not m.media:
            raise InputError(f"Moment {k + 1} needs a photo or a video clip.")
        if not (m.thought.strip() or m.title.strip()):
            raise InputError(f"Moment {k + 1} needs a title or a thought.")
    RenderConfig(width=width, height=height, fps=fps).validate()
    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    W, H = width, height
    n = len(moments)
    lay = diary_layout(W, H)
    paper = _paper(W, H)
    blank = to_uint8(paper)
    bx, by, bw, bh = lay.draw_box
    best = diary.best()
    options = [o for o in diary.guess if o][:4]
    answer = next((k for k, o in enumerate(options) if o.lower() == diary.answer.lower()), -1)
    if answer < 0 and diary.answer and len(options) < 4:
        options.append(diary.answer)
        answer = len(options) - 1
    asking = guess and len(options) >= 2 and answer >= 0
    warnings: List[str] = []
    tmp = tempfile.mkdtemp(prefix=".pets-", dir=out_dir)
    try:
        name, pitch = pet_voice(diary.kind, voice, pitch)
        speaker = _Speaker(name, pitch, tmp)
        if name is None:
            warnings.append("This computer cannot read aloud, so the pet's lines are only written.")
        shares = day_shares(moments)
        bar = DayBar(shares, lay.x, lay.bar_y, lay.width, lay.unit)
        cam = f"{diary.pet.upper()} CAM"

        # ---- each moment: its picture, drawing, voice and footage
        stages, colours, footage, stills, texts, voices, punches, holds, gallery = [], [], [], [], [], [], [], [], []
        cells = _grid(n, W, H, int(0.16 * H), 4 / 3)
        for i, m in enumerate(moments):
            report(0.01 + 0.08 * i / n, f"Preparing {m.title or m.time} ({i + 1}/{n})")
            thought = speaker.say(m.thought)
            punch = speaker.say(m.punch)
            hold = max(reveal_seconds, _seconds(punch) + 0.9)
            need = WASH + MELT + GROW + hold + PAW + 0.3
            start = 0.0
            if m.is_video:
                length = video_seconds(m.media)
                start = float(np.clip(length - need, 0.0, 1.0))
                still = video_frame(m.media, start)
            else:
                still = load_image(m.media, f"Photo of {m.title or m.time}")
            framed = prepare_portrait(still, bw / bh, None, "whole", seed=i)
            page = paper.copy()
            page[by:by + bh, bx:bx + bw] = np.minimum(page[by:by + bh, bx:bx + bw],
                                                      graphite_drawing(framed, (bw, bh), seed=i, border=0.025))
            mask = _animal_mask(framed) if drawing_seconds >= 8.0 else None
            if mask is not None:
                first = paper.copy()
                first[by:by + bh, bx:bx + bw] += (page[by:by + bh, bx:bx + bw] - first[by:by + bh, bx:bx + bw]) * \
                    cv2.resize(mask, (bw, bh))[..., None]
                stages.append([to_uint8(first), to_uint8(page)])
            else:
                stages.append([to_uint8(page)])
            colours.append(framed)
            stills.append(still)
            footage.append(Footage(m.media, m.is_video, still, W, H, fps, start, need))
            path = None
            if thought is not None:  # the bubble opens a moment after the drawing starts
                path = os.path.join(tmp, f"thought_{i}.wav")
                write_wav(path, np.concatenate([np.zeros(int(THOUGHT_AT * SAMPLE_RATE), F32), thought]))
            voices.append(path)
            texts.append(_DiaryText(m, i, n, diary.pet, lay, drawing_seconds, _seconds(thought), bar))
            punches.append(punch)
            holds.append(hold)
            gx, gy, gw, gh = cells[i]
            gallery.append(graphite_drawing(prepare_portrait(still, gw / gh, None, "whole", seed=i), (gw, gh),
                                            seed=i, border=0.025))
        labels = [" · ".join(p for p in (m.time.strip(), m.title.strip()) if p) for m in moments]
        final = postcard_wall(gallery, labels, diary.heading(), paper)

        # ---- the parts of the video, in order
        parts: List[tuple] = []
        if cold_open:
            report(0.1, "The cold open")
            b = moments[best]
            hook = diary.hook.strip() or HOOK
            parts.append(("clip", cold_open_clip(footage[best], stills[best], _Vlog(W, H, b.time.strip(), cam),
                                                 hook, speaker.say(hook), speaker.say(EXPLAIN), blank, fps),
                          "The cold open"))
        parts.append(("clip", title_clip(blank, diary.heading(), diary.pet, fps), "The title"))
        scene = {}
        for i, m in enumerate(moments):
            if i == best and asking:
                parts.append(("clip", guess_clip(blank, stages[i][-1], lay, options, answer, speaker.say(ASK), fps),
                              "What happens next?"))
            text = texts[i]
            scene[len(parts)] = (i, text)
            if len(stages[i]) == 2:  # the pet first, in a close-up; then the rest of the picture
                first = max(4.0, round(0.62 * drawing_seconds, 1))
                second = drawing_seconds - first
                if second < 4.0:
                    first, second = drawing_seconds - 4.0, 4.0
                parts.append(("draw", paper, stages[i][0], first, f"Drawing {m.title or m.time}", text, 0.0, True))
                parts.append(("draw", stages[i][0], stages[i][1], second, f"Drawing {m.title or m.time}: the rest",
                              text, first, False))
            else:
                parts.append(("draw", paper, stages[i][0], drawing_seconds, f"Drawing {m.title or m.time}", text,
                              0.0, False))
            rest = _finished_clip(stages[i][-1], text, drawing_seconds, lay.draw_box, fps)
            parts.append(("clip", rest, f"{m.title or m.time}: the thought"))
            target = blank if i + 1 < n else final
            parts.append(("clip", alive_clip(rest.last(), colours[i], footage[i], lay, fps, holds[i],
                                             _Vlog(W, H, m.time.strip(), cam), m.punch, punches[i], target, seed=i),
                          f"{m.title or m.time} comes to life"))
        parts.append(("still", final, 3.0, "The sketchbook"))
        credits = [f"{label}: {m.credit}" for label, m in zip(labels, moments) if m.credit]
        if outro:
            parts.append(("clip", outro_clip(final, diary.question.strip() or QUESTION,
                                             [c.split(" (")[0] for c in credits], fps, credit_label="Photos & videos"),
                          "Like, subscribe, comment"))
        warnings += speaker.problems

        music_path = music
        if not music and tune in MU.MOODS:
            seconds = sum(p[1].seconds if p[0] == "clip" else (p[3] if p[0] == "draw" else p[2]) for p in parts)
            music_path = os.path.join(tmp, "tune.wav")
            write_wav(music_path, MU.background(seconds + 2.0, tune))
        total = render_parts(parts, scene, voices, music_path, sound_effects, lay, paper, fps, output_path, report,
                             crf, preset, duck=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    image_path = os.path.splitext(output_path)[0] + ".png"
    Image.fromarray(final).save(image_path)
    credits_path = None
    if credits:
        credits_path = os.path.splitext(output_path)[0] + ".credits.txt"
        with open(credits_path, "w", encoding="utf-8") as fh:
            fh.write("Photos and videos (redrawn in pencil):\n" + "\n".join(credits) + "\n")
    report(1.0, "Done")
    return DiaryResult(output_path, image_path, credits_path, total, labels, time.perf_counter() - started, warnings)


# --------------------------------------------------------------------------- ready-made diaries

SAMPLES: Dict[str, Dict] = {
    "biscuit": {
        "pet": "Biscuit", "kind": "dog", "title": "A Day in the Life of Biscuit",
        "hook": HOOK, "question": "What does YOUR pet do all day?",
        "guess": ["Biscuit runs away", "Biscuit takes a bath", "Biscuit eats the soap"],
        "answer": "Biscuit takes a bath",
        "moments": [
            {"time": "7:00 AM", "title": "Wake-up call", "search": "corgi bed",
             "thought": "The sun is up, so the human should be up too. I will stare until it works.",
             "punch": "It worked. It always works."},
            {"time": "8:00 AM", "title": "Breakfast", "search": "corgi eating",
             "thought": "Same food as yesterday. And the day before. My favourite!",
             "punch": "Gone in nine seconds. A new record!"},
            {"time": "10:00 AM", "title": "Walk time", "search": "corgi walking",
             "thought": "The leash! The leash means outside, and outside means EVERYTHING!",
             "punch": "I smelled forty-two trees today."},
            {"time": "1:00 PM", "title": "Garden patrol", "search": "corgi puppy garden",
             "thought": "Something in the garden smells suspicious. I must investigate. With my nose.",
             "punch": "It was a leaf. A very suspicious leaf."},
            {"time": "3:00 PM", "title": "Nap", "search": "corgi sleeping",
             "thought": "Resting my eyes. Not sleeping. Just resting.",
             "punch": "Okay. I was sleeping."},
            {"time": "6:00 PM", "title": "The B-word", "search": "corgi bath", "best": True,
             "thought": "The human said a word. That word was bath.",
             "punch": "And that is how I ended up here."},
        ]},
    "pepper": {
        "pet": "Pepper", "kind": "cat", "title": "A Day in the Life of Pepper",
        "hook": "Why am I like this? Let me start from the beginning.",
        "question": "Is your cat a morning cat or a midnight cat?",
        "guess": ["Pepper goes to sleep", "Pepper gets the zoomies", "Pepper eats dinner"],
        "answer": "Pepper gets the zoomies",
        "moments": [
            {"time": "6:00 AM", "title": "The alarm", "search": "tabby cat bed",
             "thought": "The human sleeps too much. Time to sit on their face.",
             "punch": "Breakfast is served. Good human."},
            {"time": "9:00 AM", "title": "Window patrol", "search": "tabby cat window",
             "thought": "A bird is outside. I am a fearsome hunter.",
             "punch": "The bird did not even notice me. Rude."},
            {"time": "12:00 PM", "title": "Box day", "search": "cat in box",
             "thought": "A new box! It is too small. It is perfect.",
             "punch": "I live here now."},
            {"time": "3:00 PM", "title": "Sunbeam", "search": "tabby cat sleeping sun",
             "thought": "The sun made a warm square, just for me.",
             "punch": "Do not disturb. Ever."},
            {"time": "11:00 PM", "title": "Midnight run", "search": "tabby cat playing", "best": True,
             "thought": "It is night. My legs need to run. Right now.",
             "punch": "Who knocked the plant over? Not me."},
        ]},
}


def sample(name: str) -> Diary:
    if name not in SAMPLES:
        raise InputError(f"Unknown sample diary {name!r}; choose one of: {', '.join(SAMPLES)}.")
    return Diary.from_dict(json.loads(json.dumps(SAMPLES[name])))


def find_media(diary: Diary, keys: Dict[str, str], cache: str,
               progress: Optional[ProgressFn] = None) -> List[str]:
    """Look up a photo or clip for every moment that has none (video clips
    first when a Pexels or Pixabay key is given), download it and note its
    credit; returns the problems met."""
    from .sources import download, search_media
    report = progress or (lambda f, m: None)
    used, problems = set(), []
    for i, m in enumerate(diary.moments):
        if m.media:
            continue
        query = m.search.strip() or f"{diary.kind} {m.title}".strip()
        pick = None
        # the libraries fail now and then: try again, then with fewer words
        for attempt, words in enumerate((query, query, " ".join(query.split()[:2]))):
            report(i / len(diary.moments), f"Looking for '{words}'")
            found, errors = search_media(words, keys, videos=True)
            pick = next((f for f in found if f.url not in used), None)
            if pick is not None:
                break
            problems += errors
            time.sleep(2.0 * attempt)
        if pick is None:
            problems.append(f"Nothing found for '{query}'.")
            continue
        used.add(pick.url)
        m.media = download(pick, cache)
        m.credit = pick.credit()
    return problems


# --------------------------------------------------------------------------- CLI

def load_script(path: str) -> Diary:
    """A JSON diary: {"pet", "kind", "title", "hook", "guess", "answer",
    "question", "moments": [{"time", "title", "thought", "punch", "media",
    "best", "credit", "search"}]} with file paths relative to it."""
    with open(path, encoding="utf-8") as fh:
        diary = Diary.from_dict(json.load(fh))
    base = os.path.dirname(os.path.abspath(path))
    for m in diary.moments:
        if m.media:
            m.media = os.path.join(base, m.media)
    return diary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pencil-pets",
                                description="A day in a pet's life, drawn in pencil and told by the pet.")
    p.add_argument("script", nargs="?", help="JSON diary (see load_script)")
    p.add_argument("--sample", choices=sorted(SAMPLES), help="a ready-made diary; its photos and clips are "
                                                             "looked up online (Openverse, and Pexels / "
                                                             "Pixabay with PEXELS_API_KEY / PIXABAY_API_KEY)")
    p.add_argument("-o", "--output", required=True, help="output MP4 (a PNG is saved next to it)")
    p.add_argument("--size", default="1920x1080", help="resolution, e.g. 1080x1920 for Shorts")
    p.add_argument("--seconds", type=float, default=8.0, help="drawing time per moment")
    p.add_argument("--reveal", type=float, default=REVEAL, help="seconds of each real clip, at least")
    p.add_argument("--music", help="your own music under the whole video")
    p.add_argument("--tune", default="happy", choices=("happy", "calm", "none"), help="built-in music")
    p.add_argument("--voice", default="", help="macOS voice for the pet (default: by kind)")
    p.add_argument("--pitch", type=float, default=0.0, help="voice pitch, 0.6-1.8 (default: by kind)")
    for flag in ("guess", "cold-open", "outro", "sound-effects"):
        p.add_argument(f"--no-{flag}", action="store_true", help=f"leave out the {flag.replace('-', ' ')}")
    args = p.parse_args(argv)
    try:
        if args.sample:
            from .sources import keys_from
            diary = sample(args.sample)
            cache = os.path.join(os.path.dirname(os.path.abspath(args.output)), ".pet-media")
            for problem in find_media(diary, keys_from(), cache, lambda f, m: print(m, flush=True)):
                print(f"warning: {problem}", file=sys.stderr)
        elif args.script:
            diary = load_script(args.script)
        else:
            p.error("give a diary script or --sample")
        w, h = (int(v) for v in args.size.lower().split("x"))
        result = render_pets(diary, args.output, w, h, drawing_seconds=args.seconds, reveal_seconds=args.reveal,
                             music=args.music, tune=None if args.tune == "none" else args.tune, voice=args.voice,
                             pitch=args.pitch, guess=not args.no_guess, cold_open=not args.no_cold_open,
                             outro=not args.no_outro, sound_effects=not args.no_sound_effects)
    except (PencilError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"Saved {result.output_path} ({result.duration:.0f} s) and {result.image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
