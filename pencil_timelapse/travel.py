"""Travel videos: famous places drawn in pencil one after another.

Each place is drawn from its photo (or from a picture of its video clip)
while its facts are written beside the drawing and its narration plays. When
the drawing and the facts are done the drawing comes to life: colour runs
over the graphite, the real photo or clip grows to fill the screen with the
place's name, where it lies (typed coordinates and a map card with a pin) and
a passport stamp, then a paper plane flies on to the next place.

Around the places: an intro on a pencil world map where every destination
is pinned and the route is flown, a wall of postcards of all the drawings,
and a like / subscribe / comment ending.

Usage:  python -m pencil_timelapse.travel trip.json -o trip.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import worldmap as WM
from .animals import (_SceneText, _animal_mask, _finished_clip, _paper, render_parts,
                      scene_layout)
from .config import RenderConfig
from .encoder import find_ffmpeg
from .errors import InputError, PencilError
from .imaging import load_image, to_uint8
from .media import SAMPLE_RATE, decode_audio
from .players import prepare_portrait
from .showtime import (BOLD_FONTS, DISPLAY_FONTS, HAND_FONTS, INK, WHITE, Clip, ShowOptions,
                       Sprite, _f, _font, _phase, _render, _rgb255, _u8, back, outro_clip, smooth,
                       text_sprite, zoom_view)
from .sketch import PAPER, crop_to, graphite_drawing

F32 = np.float32
QUESTION = "Which place should I draw next?"
PROMPT = "Guess the place?"
VIDEO_TYPES = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")
STAMP_INK = (0.8, 0.12, 0.13)
REVEAL = 3.2  # seconds the place is shown in colour, full screen
# The steps of a reveal: colour over the graphite, the real picture fading in,
# the picture growing to the whole screen; then the plane to the next place.
WASH, MELT, GROW, FLY = 0.9, 0.4, 0.9, 1.0


@dataclass
class Place:
    name: str
    media: str  # a photo or a video clip
    facts: List[str] = field(default_factory=list)
    location: str = ""  # "Paris, France"
    country: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    voice: Optional[str] = None  # optional narration, from the moment the drawing starts
    credit: str = ""  # who took the photo or clip, for the credits

    @property
    def is_video(self) -> bool:
        return os.path.splitext(self.media)[1].lower() in VIDEO_TYPES

    @property
    def where(self) -> Optional[Tuple[float, float]]:
        return None if self.lat is None or self.lon is None else (self.lon, self.lat)


@dataclass
class TravelResult:
    output_path: str
    image_path: str  # the postcard wall
    credits_path: Optional[str]
    duration: float
    places: List[str]
    elapsed: float
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- video clips

def video_seconds(path: str) -> float:
    """How long a video clip is; InputError when it has no picture."""
    proc = subprocess.run([find_ffmpeg(), "-hide_banner", "-nostdin", "-i", path],
                          capture_output=True, text=True, timeout=60)
    found = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not found or "Video:" not in proc.stderr:
        raise InputError(f"Could not read the video {os.path.basename(path)}.")
    h, m, s = found.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def video_frame(path: str, at: float) -> np.ndarray:
    """The picture of a clip at ``at`` seconds (float RGB)."""
    proc = subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
                           "-ss", f"{max(at, 0.0):.3f}", "-i", path, "-frames:v", "1",
                           "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                          capture_output=True, timeout=120)
    img = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(f"Could not read a picture from the video {os.path.basename(path)}.")
    return img[..., ::-1].astype(F32) / 255.0


def _cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """``img`` cropped to the shape of ``width`` x ``height`` and resized to it."""
    img = crop_to(img, width / height)
    interp = cv2.INTER_AREA if img.shape[1] > width else cv2.INTER_CUBIC
    return np.clip(cv2.resize(img, (width, height), interpolation=interp), 0.0, 1.0)


class Footage:
    """The place in colour filling a ``width`` x ``height`` frame: the video
    clip from ``start`` (its last picture held if it ends), or the photo with a
    slow push in."""

    def __init__(self, media: str, is_video: bool, still: np.ndarray, width: int, height: int,
                 fps: int, start: float = 0.0, seconds: float = 6.0):
        self.media, self.is_video, self.start = media, is_video, start
        self.width, self.height, self.fps, self.seconds = width, height, fps, seconds
        self.cover = None if is_video else _cover(still, width, height)

    def frames(self) -> Iterator[np.ndarray]:
        W, H = self.width, self.height
        if not self.is_video:
            n = 0
            while True:
                z = 1.0 + 0.07 * smooth(n / (self.seconds * self.fps))
                yield zoom_view(self.cover, W / 2, H / 2, z)
                n += 1
        cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{self.start:.3f}", "-i", self.media, "-an",
               "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                      f"fps={self.fps},format=rgb24",
               "-f", "rawvideo", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        size, last = W * H * 3, None
        try:
            while True:
                raw = proc.stdout.read(size) if proc.stdout else b""
                if len(raw) == size:
                    last = np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(F32) / 255.0
                elif last is None:
                    raise InputError(f"Could not play the video {os.path.basename(self.media)}.")
                yield last
        finally:
            proc.kill()
            proc.wait()


# --------------------------------------------------------------------------- pieces

def _glow(sprite: Sprite, radius: float, colour=PAPER, strength: float = 0.85) -> Sprite:
    """A soft halo the shape of ``sprite``, to keep text readable over a map."""
    r = max(1, int(radius))
    alpha = np.pad(sprite.alpha[..., 0], r * 2)
    alpha = cv2.dilate(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)))
    alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), r * 0.8) * strength, 0.0, 1.0)
    rgba = np.concatenate([np.broadcast_to(np.asarray(colour, F32), alpha.shape + (3,)),
                           alpha[..., None]], axis=2)
    return Sprite(rgba)


def _with_glow(frame: np.ndarray, sprite: Sprite, glow: Sprite, cx: float, cy: float,
               scale: float = 1.0, opacity: float = 1.0) -> None:
    glow.paste(frame, cx, cy, scale=scale, opacity=opacity)
    sprite.paste(frame, cx, cy, scale=scale, opacity=opacity)


def stamp_sprite(country: str, number: str, size: int) -> Sprite:
    """A passport stamp in red ink: VISITED, the country and the place's number."""
    w, h = int(size * 2.1), size
    rng = np.random.default_rng(len(country) + size)

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        ink = _rgb255(STAMP_INK)
        line = max(2, int(size * 0.05)) * s
        d.rounded_rectangle([line, line, w * s - line, h * s - line], radius=h * 0.16 * s,
                            outline=ink, width=line)
        inset = line * 2.6
        d.rounded_rectangle([inset, inset, w * s - inset, h * s - inset], radius=h * 0.1 * s,
                            outline=ink, width=max(1, line // 2))
        for text, fonts, k, y in (("VISITED", BOLD_FONTS, 0.15, 0.2),
                                  (country.upper(), DISPLAY_FONTS, 0.36, 0.5),
                                  (number, BOLD_FONTS, 0.12, 0.79)):
            font = _font(fonts, int(h * k * s))
            while font.getlength(text) > (w - 4.2 * line / s) * s * 0.92 and font.size > 8:
                font = _font(fonts, int(font.size * 0.92))
            asc, desc = font.getmetrics()
            d.text(((w * s - font.getlength(text)) / 2, y * h * s - (asc + desc) / 2), text,
                   font=font, fill=ink)

    sprite = _render(draw, (w, h))
    # Ink does not take everywhere: a few specks and a lighter patch.
    speck = cv2.GaussianBlur(rng.random((h, w)).astype(F32), (0, 0), max(0.6, size / 160))
    patch = cv2.resize(rng.random((4, 8)).astype(F32), (w, h), interpolation=cv2.INTER_CUBIC)
    keep = np.clip((speck - 0.18) * 6.0, 0.0, 1.0) * (0.82 + 0.18 * np.clip(patch, 0.0, 1.0))
    sprite.alpha *= np.clip(keep, 0.0, 1.0)[..., None] * 0.96
    return sprite


def _pin_icon(size: int, colour=WHITE) -> Sprite:
    return WM.pin_sprite(size, colour)


def coordinates(lat: float, lon: float) -> str:
    return (f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}   "
            f"{abs(lon):.4f}° {'E' if lon >= 0 else 'W'}")


def _fly(width: int, height: int) -> np.ndarray:
    """The paper plane's path across the screen, low left to high right."""
    return WM.arc((-0.12 * width, 0.86 * height), (1.12 * width, 0.14 * height), bend=0.12)


def _rising(width: int, height: int) -> np.ndarray:
    """0 at the bottom left to 1 at the top right, across the screen."""
    yy, xx = np.mgrid[0:height, 0:width].astype(F32)
    return (xx + 0.55 * (height - yy)) / (width + 0.55 * height)


def fly_over(frame: np.ndarray, target: np.ndarray, p: float, plane: Sprite, path: np.ndarray,
             rising: np.ndarray, thickness: float) -> np.ndarray:
    """A paper plane flying across, the next page following it in."""
    q = smooth(p)
    gone = np.clip((q * 1.12 - 0.06 - rising) / 0.03 + 0.5, 0.0, 1.0)[..., None]
    out = frame * (1.0 - gone) + target * gone
    trail, head, angle = WM.along(path, q * 1.1)
    WM.dashed(out, trail, thickness, dash=thickness * 5)
    plane.paste(out, float(head[0]), float(head[1]), angle=angle)
    return out


# --------------------------------------------------------------------------- clips

def map_intro_clip(world: np.ndarray, pins: Sequence[Tuple[float, float]], blank: np.ndarray,
                   headline: str, kicker: str, tagline: str, fps: int) -> Clip:
    """The pencil world map: a headline, then every destination pinned in turn
    with the route flown by a paper plane; an eraser sweep and "Let's go!"."""
    H, W = world.shape[:2]
    art, paper = _f(world), _f(blank)
    unit = min(W, H)
    k_s = text_sprite(kicker, BOLD_FONTS, int(0.045 * unit), INK, tracking=0.2, max_width=int(0.9 * W))
    h_s = text_sprite(headline, DISPLAY_FONTS, int(0.15 * unit), INK, max_width=int(0.92 * W))
    t_s = text_sprite(tagline, HAND_FONTS, int(0.065 * unit), INK, max_width=int(0.9 * W))
    go = text_sprite("Let's go!", HAND_FONTS, int(0.15 * unit), INK, max_width=int(0.8 * W))
    glows = [_glow(s, 0.02 * unit) for s in (k_s, h_s, t_s)]
    pin = WM.pin_sprite(int(0.06 * unit))
    plane = WM.plane_sprite(int(0.1 * unit))
    n = len(pins)
    step = min(0.42, 2.6 / max(n, 1))
    drops = [1.0 + i * step for i in range(n)]
    routes = [WM.arc(a, b, bend=0.22) for a, b in zip(pins, pins[1:])]
    show, sweep, total = 4.4, 0.8, 6.3
    centre = np.array([W / 2, H / 2], F32)
    head_y = 0.2 * H if W > H else 0.16 * H

    def titled(t: float) -> np.ndarray:
        z = 1.08 - 0.08 * smooth(t / show)
        frame = zoom_view(art, W / 2, H / 2, z).copy()
        if t < 0.4:
            frame = paper + (frame - paper) * smooth(t / 0.4)

        def moved(p) -> np.ndarray:
            return (np.asarray(p, F32) - centre) * z + centre

        for i, route in enumerate(routes):
            share = _phase(t, drops[i] + 0.12, step)
            if share > 0.0:
                part, head, angle = WM.along(moved(route), share)
                WM.dashed(frame, part, max(2.0, 0.004 * unit), dash=0.016 * unit)
                if share < 1.0:
                    plane.paste(frame, float(head[0]), float(head[1]), angle=angle)
        for i, p in enumerate(pins):
            d = _phase(t, drops[i], 0.3)
            if d > 0.0:
                x, y = moved(p)
                pin.paste(frame, x, y - pin.size[1] / 2 - (1.0 - back(d)) * 0.1 * H,
                          opacity=min(1.0, 3 * d))
        a = _phase(t, 0.1, 0.3)
        _with_glow(frame, k_s, glows[0], W / 2, head_y - 0.1 * unit, opacity=a)
        b = _phase(t, 0.3, 0.28)
        _with_glow(frame, h_s, glows[1], W / 2, head_y, scale=1.0 + 0.5 * (1.0 - back(b)),
                   opacity=min(1.0, 2 * b))
        _with_glow(frame, t_s, glows[2], W / 2, H - 0.09 * unit if W > H else 0.8 * H,
                   opacity=_phase(t, 1.3, 0.4))
        return frame

    def frames() -> Iterator[np.ndarray]:
        last = titled(show)
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        diag = (xx + 0.45 * yy) / (W + 0.45 * H)
        path = _fly(W, H)
        for i in range(int(round(total * fps))):
            t = i / fps
            if t < show:
                yield _u8(np.clip(titled(t), 0.0, 1.0))
            elif t < show + sweep:
                p = smooth((t - show) / sweep) * 1.08 - 0.04
                gone = np.clip((p - diag) / 0.04 + 0.5, 0.0, 1.0)[..., None]
                streak = np.exp(-((diag - p) / 0.015) ** 2)[..., None] * 0.35
                yield _u8(np.clip(last * (1 - gone) + paper * gone + streak, 0.0, 1.0))
            else:
                frame = paper.copy()
                u = t - show - sweep
                fade = 1.0 - _phase(t, total - 0.3, 0.3)
                go.paste(frame, W / 2, 0.5 * H, scale=0.55 + 0.45 * back(_phase(u, 0.05, 0.35)),
                         opacity=min(1.0, _phase(u, 0.05, 0.12)) * fade)
                trail, head, angle = WM.along(path, _phase(u, 0.1, total - show - sweep - 0.2))
                WM.dashed(frame, trail, max(2.0, 0.004 * unit), dash=0.02 * unit, opacity=0.6 * fade)
                plane.paste(frame, float(head[0]), float(head[1]), angle=angle, opacity=fade)
                yield _u8(frame)

    sounds = [(0.32, "thud"), (1.1, "swish")] + [(d, "pop") for d in drops]
    sounds += [(show, "whoosh"), (show + sweep + 0.08, "pop"), (show + sweep + 0.1, "swish")]
    return Clip(total, frames, sounds)


def reveal_clip(page: np.ndarray, colour: np.ndarray, footage: Footage, place: Place,
                box: Tuple[int, int, int, int], target: np.ndarray, number: str, fps: int,
                hold: float = REVEAL, card: Optional[Tuple[Sprite, Tuple[float, float]]] = None
                ) -> Clip:
    """The drawing comes to life: colour washes over the graphite, the real
    picture grows to the whole screen with the place's name, location and a
    passport stamp, then a paper plane flies on to ``target``."""
    H, W = page.shape[:2]
    unit = min(W, H)
    base = _f(page)
    x, y, w, h = box
    colour = np.clip(cv2.resize(colour, (w, h), interpolation=cv2.INTER_AREA), 0.0, 1.0)
    rng = np.random.default_rng(len(place.name))
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    wobble = cv2.resize(rng.standard_normal((6, 8)).astype(F32), (w, h), interpolation=cv2.INTER_CUBIC)
    sweep = (xx + 0.6 * yy) / (w + 0.6 * h) + 0.05 * wobble
    wide = W > H * 1.2
    shade = np.clip((np.arange(H, dtype=F32) / H - 0.5) / 0.5, 0.0, 1.0) ** 1.3 * 0.68
    name_s = text_sprite(place.name.upper(), DISPLAY_FONTS, int(0.12 * unit), WHITE,
                         max_width=int((0.6 if wide else 0.88) * W))
    loc_s = text_sprite(place.location, BOLD_FONTS, int(0.042 * unit), WHITE,
                        max_width=int(0.6 * W)) if place.location else None
    pin_s = _pin_icon(int(0.05 * unit), (0.95, 0.35, 0.3))
    coord_s = (text_sprite(coordinates(place.lat, place.lon), BOLD_FONTS, int(0.03 * unit),
                           (0.86, 0.86, 0.84), tracking=0.06)
               if place.where else None)
    country = place.country or (place.location.split(",")[-1].strip() if place.location else place.name)
    stamp = stamp_sprite(country, number, int(0.2 * unit))
    stamp_glow = _glow(stamp, 0.006 * unit, colour=(1.0, 1.0, 1.0), strength=0.3)
    pin = WM.pin_sprite(int(0.055 * unit))
    plane = WM.plane_sprite(int(0.13 * unit))
    fly_path = _fly(W, H)
    rising = _rising(W, H)
    left = 0.05 * W
    name_y = (0.72 if wide else 0.76) * H
    stamp_at = (W - 0.17 * W, 0.76 * H) if wide else (0.74 * W, 0.6 * H)
    card_at = None
    if card is not None:
        cw, ch = card[0].size
        card_at = (W - 0.03 * W - cw / 2, 0.05 * H + ch / 2)
    shown = WASH + MELT + GROW
    total = shown + hold + FLY
    tgt = _f(target)

    def fit(img: np.ndarray, rw: int, rh: int) -> np.ndarray:
        return _cover(img, max(rw, 2), max(rh, 2))

    def dressed(frame: np.ndarray, u: float) -> None:
        """The name, location, map card and stamp over the full-screen picture."""
        frame *= (1.0 - shade)[:, None, None]
        a = _phase(u, 0.1, 0.35)
        name_s.paste_left(frame, left - 0.04 * W * (1.0 - smooth(a)), name_y - name_s.size[1] / 2,
                          opacity=a)
        row = name_y + name_s.size[1] * 0.5
        if loc_s is not None:
            b = _phase(u, 0.35, 0.3)
            pin_s.paste(frame, left + pin_s.size[0] / 2, row + loc_s.size[1] / 2, opacity=b)
            loc_s.paste_left(frame, left + pin_s.size[0] * 1.15, row, reveal=smooth(b) if b < 1 else 1.0)
            row += loc_s.size[1] * 1.05
        if coord_s is not None:
            coord_s.paste_left(frame, left, row, reveal=_phase(u, 0.6, 0.8))
        if card is not None and card_at is not None:
            sprite, (px, py) = card
            c = _phase(u, 0.2, 0.45)
            cx, cy = card_at[0], card_at[1] - (1.0 - back(c)) * 0.3 * H
            angle = 3.0
            sprite.paste(frame, cx, cy, angle=angle, opacity=min(1.0, 2 * c))
            dx, dy = px - sprite.size[0] / 2, py - sprite.size[1] / 2
            r = math.radians(angle)
            tip = (cx + dx * math.cos(r) + dy * math.sin(r), cy - dx * math.sin(r) + dy * math.cos(r))
            d = _phase(u, 0.75, 0.3)
            if d > 0:
                pin.paste(frame, tip[0], tip[1] - pin.size[1] / 2 - (1.0 - back(d)) * 0.08 * H,
                          opacity=min(1.0, 3 * d))
        s = _phase(u, 1.1, 0.22)
        if s > 0:
            look = dict(scale=1.0 + 0.7 * (1.0 - smooth(s)), angle=-12.0, opacity=min(1.0, 2.5 * s))
            stamp_glow.paste(frame, *stamp_at, **look)
            stamp.paste(frame, *stamp_at, **look)

    def frames() -> Iterator[np.ndarray]:
        feed = footage.frames()
        try:
            for i in range(int(round(total * fps))):
                t = i / fps
                if t < WASH:
                    frame = base.copy()
                    p = smooth(t / WASH) * 1.15 - 0.06
                    m = np.clip((p - sweep) / 0.05 + 0.5, 0.0, 1.0)[..., None]
                    region = frame[y:y + h, x:x + w]
                    region[:] = region * (1.0 - m) + colour * m
                    region += np.exp(-((sweep - p) / 0.018) ** 2)[..., None] * 0.3 * (1.0 - region)
                    yield _u8(np.clip(frame, 0.0, 1.0))
                    continue
                real = next(feed)
                if t < WASH + MELT:
                    frame = base.copy()
                    k = smooth((t - WASH) / MELT)
                    frame[y:y + h, x:x + w] = colour * (1.0 - k) + fit(real, w, h) * k
                elif t < shown:
                    k = smooth((t - WASH - MELT) / GROW)
                    rx, ry = int(round(x * (1 - k))), int(round(y * (1 - k)))
                    rw = int(round(w + (W - w) * k))
                    rh = int(round(h + (H - h) * k))
                    rw, rh = min(rw, W - rx), min(rh, H - ry)
                    frame = base * (1.0 - 0.35 * k)
                    frame[ry:ry + rh, rx:rx + rw] = fit(real, rw, rh)
                else:
                    frame = real.copy()
                    u = t - shown
                    dressed(frame, u)
                    if u > hold:
                        frame = fly_over(frame, tgt, (u - hold) / FLY, plane, fly_path, rising,
                                         max(2.0, 0.004 * unit))
                yield _u8(np.clip(frame, 0.0, 1.0))
        finally:
            feed.close()

    sounds = [(0.05, "sparkle"), (WASH + MELT + 0.05, "swish"), (shown + 0.12, "swish"),
              (shown + 1.12, "thud"), (shown + hold, "whoosh")]
    if card is not None:
        sounds.append((shown + 0.78, "pop"))
    return Clip(total, frames, sounds)


# --------------------------------------------------------------------------- the postcards

def _grid(count: int, width: int, height: int, top: int, aspect: float
          ) -> List[Tuple[int, int, int, int]]:
    """Cells for ``count`` cards of ``aspect`` (with room for a caption) under ``top``."""
    margin = int(0.035 * min(width, height))
    avail_w, avail_h = width - 2 * margin, height - top - margin
    best = None
    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        cw = min(avail_w / cols - margin, (avail_h / rows - margin) / 1.2 * aspect)
        if cw > 0 and (best is None or cw > best[0]):
            best = (cw, cols, rows)
    cw, cols, rows = best
    ch = cw / aspect
    cells = []
    row_h = avail_h / rows
    for i in range(count):
        r, c = divmod(i, cols)
        in_row = min(cols, count - r * cols)
        cx = width / 2 + (c - (in_row - 1) / 2) * (cw + margin)
        cy = top + r * row_h + row_h / 2 - 0.1 * ch
        cells.append((int(cx - cw / 2), int(cy - ch / 2), int(cw), int(ch)))
    return cells


def postcard_wall(drawings: List[np.ndarray], names: List[str], title: str,
                  background: np.ndarray) -> np.ndarray:
    """All the drawings as postcards with their names, on a faded pencil map."""
    H, W = background.shape[:2]
    unit = min(W, H)
    page = PAPER + (background - PAPER) * 0.35
    top = int(0.16 * H) if title else int(0.05 * H)
    rng = np.random.default_rng(9)
    for (x, y, w, h), drawing, name in zip(_grid(len(drawings), W, H, top, 4 / 3), drawings, names):
        border = max(3, int(0.045 * w))
        cap = int(0.2 * h)
        cw, ch = w + 2 * border, h + 2 * border + cap
        card = np.ones((ch, cw, 4), F32)
        card[..., :3] = (0.99, 0.985, 0.97)
        card[border:border + h, border:border + w, :3] = cv2.resize(drawing, (w, h),
                                                                     interpolation=cv2.INTER_AREA)
        label = text_sprite(name, HAND_FONTS, int(cap * 0.62), INK, max_width=int(w * 0.95))
        label.paste(card[..., :3], cw / 2, border + h + cap / 2)
        shadow = np.zeros((ch + 4 * border, cw + 4 * border, 4), F32)
        shadow[..., :3] = 0.1
        shadow[2 * border:2 * border + ch, 2 * border:2 * border + cw, 3] = 0.35
        shadow[..., 3] = cv2.GaussianBlur(shadow[..., 3], (0, 0), border * 0.8)
        angle = float(rng.uniform(-3.0, 3.0))
        Sprite(shadow).paste(page, x + w / 2 + 0.25 * border, y + (h + cap) / 2 + 0.6 * border,
                             angle=angle)
        Sprite(card).paste(page, x + w / 2, y + (h + cap) / 2, angle=angle)
    if title:
        t_s = text_sprite(title, HAND_FONTS, int(0.09 * unit), INK, max_width=int(0.9 * W))
        _with_glow(page, t_s, _glow(t_s, 0.02 * unit), W / 2, top * 0.55)
    return to_uint8(page)


# --------------------------------------------------------------------------- video

def render_travel(places: List[Place], title: str, output_path: str, width: int = 1920,
                  height: int = 1080, fps: int = 30, drawing_seconds: float = 10.0,
                  music: Optional[str] = None, guess: bool = True,
                  show: Optional[ShowOptions] = None, reveal_seconds: float = REVEAL,
                  progress=None, crf: int = 20, preset: str = "fast") -> TravelResult:
    """Draw ``places`` one after another with their facts and bring each to
    life; save the video, the postcard wall (PNG) and the photo credits.

    As in :func:`animals.render_animals` the drawing, the facts and the
    narration of a place start together, and the video stays with a place
    until all three are done; with ``guess`` its name is hidden ("Guess the
    place?") until the drawing is finished. Then its photo or clip is shown
    in colour for ``reveal_seconds``.
    """
    report = progress or (lambda f, m: None)
    started = time.perf_counter()
    show = show or ShowOptions(question=QUESTION)
    if not places:
        raise InputError("Add at least one place.")
    if len(places) > 20:
        raise InputError("Add at most 20 places.")
    if not 4.0 <= drawing_seconds <= 60.0:
        raise InputError("Drawing time must be between 4 and 60 seconds.")
    if not 1.5 <= reveal_seconds <= 15.0:
        raise InputError("The colour reveal must last between 1.5 and 15 seconds.")
    RenderConfig(width=width, height=height, fps=fps).validate()
    for p in places:
        if not p.name.strip():
            raise InputError("Every place needs a name.")
    voices = [len(decode_audio(p.voice)) / SAMPLE_RATE if p.voice else 0.0 for p in places]
    if music:
        decode_audio(music, rate=8000)  # fail now, not after the drawing
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = len(places)
    layout = scene_layout(width, height)
    paper = _paper(width, height)
    blank_u8 = to_uint8(paper)
    bx, by, bw, bh = layout.draw_box
    footage_seconds = WASH + MELT + GROW + reveal_seconds + FLY

    # ---- the pictures: each place drawn (the landmark first when it can be told apart)
    stages, colours, footage, cards, gallery = [], [], [], [], []
    cell = _grid(n, width, height, int(0.16 * height), 4 / 3)
    for i, p in enumerate(places):
        report(0.01 + 0.09 * i / n, f"Preparing {p.name} ({i + 1}/{n})")
        start = 0.0
        if p.is_video:
            length = video_seconds(p.media)
            start = float(np.clip(length - footage_seconds - 0.3, 0.0, 1.0))
            still = video_frame(p.media, start)
        else:
            still = load_image(p.media, f"Photo of {p.name}")
        framed = prepare_portrait(still, bw / bh, None, "whole", seed=i)
        colours.append(framed)
        page = paper.copy()
        page[by:by + bh, bx:bx + bw] = np.minimum(
            page[by:by + bh, bx:bx + bw], graphite_drawing(framed, (bw, bh), seed=i, border=0.025))
        mask = _animal_mask(framed) if drawing_seconds >= 8.0 else None
        if mask is not None:
            first = paper.copy()
            first[by:by + bh, bx:bx + bw] += (page[by:by + bh, bx:bx + bw] -
                                              first[by:by + bh, bx:bx + bw]) * \
                cv2.resize(mask, (bw, bh))[..., None]
            stages.append([to_uint8(first), to_uint8(page)])
        else:
            stages.append([to_uint8(page)])
        footage.append(Footage(p.media, p.is_video, still, width, height, fps, start,
                               footage_seconds))
        if p.where:
            cw = int(0.3 * width) if width > height * 1.2 else int(0.42 * width)
            cards.append(WM.map_card(p.lon, p.lat, cw, int(cw * 0.68), seed=i))
        else:
            cards.append(None)
        gw, gh = cell[i][2], cell[i][3]
        gallery.append(graphite_drawing(prepare_portrait(still, gw / gh, None, "whole", seed=i),
                                        (gw, gh), seed=i, border=0.025))
    names = [p.name for p in places]
    report(0.1, "Drawing the world map")
    points = [p.where for p in places if p.where]
    view = WM.fit_view(points, width / height, top=0.3, bottom=0.12)  # room for the headline
    world = WM.pencil_map(view, width, height, seed=5)
    final = postcard_wall(gallery, names, title, world)

    # ---- the parts of the video, in order
    parts: List[tuple] = []
    if show.intro:
        hook = (show.hook.strip() or title or "Wonders of the world").upper()
        kicker = f"{n} PLACE{'S' if n > 1 else ''}  ·  1 PENCIL"
        tagline = "Can you guess them all?" if guess else "Around the world in pencil"
        pins = [view.xy(lon, lat, width, height) for lon, lat in points]
        parts.append(("clip", map_intro_clip(to_uint8(world), pins, blank_u8, hook, kicker, tagline,
                                             fps), "Intro"))
    scene = {}
    for i, p in enumerate(places):
        number = f"#{i + 1} / {n}"
        text = _SceneText(p, number, layout, drawing_seconds, voices[i], guess, prompt=PROMPT,
                          where=p.location)
        scene[len(parts)] = (i, text)
        if len(stages[i]) == 2:  # the landmark first, in a close-up; then its surroundings
            first = max(4.0, round(0.62 * drawing_seconds, 1))
            second = drawing_seconds - first
            if second < 4.0:
                first, second = drawing_seconds - 4.0, 4.0
            parts.append(("draw", paper, stages[i][0], first, f"Drawing {p.name}", text, 0.0, True))
            parts.append(("draw", stages[i][0], stages[i][1], second,
                          f"Drawing {p.name}: the surroundings", text, first, False))
        else:
            parts.append(("draw", paper, stages[i][0], drawing_seconds, f"Drawing {p.name}",
                          text, 0.0, False))
        rest = _finished_clip(stages[i][-1], text, drawing_seconds, layout.draw_box, fps)
        parts.append(("clip", rest, f"{p.name}: the facts"))
        target = blank_u8 if i + 1 < n else final
        parts.append(("clip", reveal_clip(rest.last(), colours[i], footage[i], p, layout.draw_box,
                                          target, f"No. {i + 1}", fps, reveal_seconds, cards[i]),
                      f"{p.name} comes to life"))
    parts.append(("still", final, 3.0, "Postcards"))
    credits = [f"{p.name}: {p.credit}" for p in places if p.credit]
    if show.outro:
        parts.append(("clip", outro_clip(final, show.question.strip() or QUESTION,
                                         [c.split(" (")[0] for c in credits], fps,
                                         credit_label="Photos & videos"),
                      "Like, subscribe, comment"))

    total = render_parts(parts, scene, [p.voice for p in places], music, show.sound_effects,
                         layout, paper, fps, output_path, report, crf, preset)
    image_path = os.path.splitext(output_path)[0] + ".png"
    Image.fromarray(final).save(image_path)
    credits_path = None
    if credits:
        credits_path = os.path.splitext(output_path)[0] + ".credits.txt"
        with open(credits_path, "w", encoding="utf-8") as fh:
            fh.write("Photos and videos (redrawn in pencil):\n" + "\n".join(credits) + "\n"
                     "\nMap data: Natural Earth (public domain).\n")
    report(1.0, "Done")
    return TravelResult(output_path, image_path, credits_path, total, names,
                        time.perf_counter() - started)


# --------------------------------------------------------------------------- CLI

def load_script(path: str) -> Tuple[str, List[Place]]:
    """A JSON script: {"title", "places": [{"name", "media", "facts", "location",
    "country", "lat", "lon", "voice", "credit"}]} with file paths relative to it."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))

    def where(value: Optional[str]) -> Optional[str]:
        return os.path.join(base, value) if value else None

    places = []
    for p in data.get("places", []):
        lat, lon = p.get("lat"), p.get("lon")
        places.append(Place(str(p["name"]), where(p["media"]), [str(f) for f in p.get("facts", [])],
                            str(p.get("location") or ""), str(p.get("country") or ""),
                            None if lat is None else float(lat), None if lon is None else float(lon),
                            where(p.get("voice")), str(p.get("credit") or "")))
    return str(data.get("title", "")), places


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pencil-travel",
                                description="Draw famous places in pencil, one after another, "
                                            "and bring each to life (simulated drawing).")
    p.add_argument("script", help='JSON: {"title", "places": [{"name", "media", "facts": [...], '
                                  '"location", "country", "lat", "lon", "voice", "credit"}]}')
    p.add_argument("-o", "--output", required=True, help="output MP4 (a PNG is saved next to it)")
    p.add_argument("--size", default="1920x1080", help="resolution, e.g. 1080x1920 for vertical")
    p.add_argument("--seconds", type=float, default=10.0, help="drawing time per place")
    p.add_argument("--reveal", type=float, default=REVEAL, help="seconds of each place in colour")
    p.add_argument("--music", help="music under the whole video")
    p.add_argument("--no-guess", action="store_true", help="show the name while drawing")
    p.add_argument("--hook", default="", help="intro headline (default: the title)")
    p.add_argument("--question", default=QUESTION, help="question asked at the end")
    for flag in ("intro", "outro", "sound-effects"):
        p.add_argument(f"--no-{flag}", action="store_true", help=f"leave out the {flag}")
    args = p.parse_args(argv)
    try:
        title, places = load_script(args.script)
        w, h = (int(v) for v in args.size.lower().split("x"))
        show = ShowOptions(not args.no_intro, True, not args.no_outro, not args.no_sound_effects,
                           args.hook, args.question)
        result = render_travel(places, title, args.output, w, h, drawing_seconds=args.seconds,
                               music=args.music, guess=not args.no_guess, show=show,
                               reveal_seconds=args.reveal)
    except (PencilError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {result.output_path} ({result.duration:.0f} s) and {result.image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
