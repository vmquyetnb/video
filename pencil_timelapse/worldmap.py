"""A pencil world map for the travel videos: the land shaded in graphite on
the paper, places pinned on it and a paper plane flying between them.

The coastlines are Natural Earth's 1:50m land (public domain), rasterised
once into ``assets/land.png`` (equirectangular, 4096 x 2048, west to east from
180° W, north to south from 90° N). ``python -m pencil_timelapse.worldmap
ne_50m_land.geojson`` rebuilds it.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import ImageDraw

from . import analysis as A
from .showtime import INK, Sprite, _render, _rgb255
from .sketch import PAPER, graphite_drawing

F32 = np.float32
LAND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "land.png")
PIN = (0.86, 0.2, 0.16)
_LAND: List[np.ndarray] = []
_LOCK = threading.Lock()


def land_mask() -> np.ndarray:
    """1 on land, 0 at sea (float32, 2048 x 4096)."""
    with _LOCK:
        if not _LAND:
            mask = cv2.imread(LAND_PATH, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"The world map is missing: {LAND_PATH}")
            _LAND.append(mask.astype(F32) / 255.0)
        return _LAND[0]


@dataclass(frozen=True)
class MapView:
    """The part of the world shown, in degrees; ``east`` may pass 180 (the
    view then continues past the date line)."""

    west: float
    south: float
    east: float
    north: float

    def xy(self, lon: float, lat: float, width: int, height: int) -> Tuple[float, float]:
        """Where (``lon``, ``lat``) falls on a ``width`` x ``height`` map of this view."""
        lon = (lon - self.west) % 360.0 + self.west
        return ((lon - self.west) / (self.east - self.west) * width,
                (self.north - lat) / (self.north - self.south) * height)


def world_view(aspect: float) -> MapView:
    """The whole world (without Antarctica) on a page of ``aspect``; the
    page's shape leaves paper above and below or beside it."""
    south, north = -56.0, 80.0
    span_x, span_y = 360.0, north - south
    if span_x / span_y > aspect:
        span_y = span_x / aspect
    else:
        span_x = span_y * aspect
    cx, cy = 10.0, (south + north) / 2 + max(0.0, (span_y - (north - south)) * 0.15)
    return MapView(cx - span_x / 2, cy - span_y / 2, cx + span_x / 2, cy + span_y / 2)


def fit_view(points: Sequence[Tuple[float, float]], aspect: float, pad: float = 14.0,
             min_span: float = 46.0, top: float = 0.0, bottom: float = 0.0) -> MapView:
    """A view of ``points`` (lon, lat) with room around them, or the whole
    world when they are spread across it. ``top`` and ``bottom`` are shares
    of the page kept free of points (for a headline)."""
    if not points:
        return world_view(aspect)
    lons = sorted(p[0] % 360.0 for p in points)
    gaps = [(lons[(i + 1) % len(lons)] - lons[i]) % 360.0 or 360.0 for i in range(len(lons))]
    i = int(np.argmax(gaps))  # the points lie between this widest gap's two ends
    west = lons[(i + 1) % len(lons)]
    span_lon = 360.0 - gaps[i] if len(lons) > 1 else 0.0
    lats = [p[1] for p in points]
    span_x = max(span_lon + 2 * pad, min_span)
    span_y = max(max(lats) - min(lats) + 2 * pad, min_span / aspect)
    span_y /= 1.0 - top - bottom
    if span_x / span_y < aspect:
        span_x = span_y * aspect
    else:
        span_y = span_x / aspect
    if span_x > 250.0 or span_y > 150.0:
        return world_view(aspect)
    cx = (west + span_lon / 2 + 180.0) % 360.0 - 180.0
    # The points sit in the middle of the part of the page left free.
    mid = (max(lats) + min(lats)) / 2 + (top - bottom) / 2 * span_y
    cy = float(np.clip(mid, -84.0 + span_y / 2, 84.0 - span_y / 2))
    return MapView(cx - span_x / 2, cy - span_y / 2, cx + span_x / 2, cy + span_y / 2)


def pencil_map(view: MapView, width: int, height: int, seed: int = 0,
               grid: bool = True) -> np.ndarray:
    """The land of ``view`` drawn in pencil on the paper (float RGB)."""
    land_full = land_mask()
    mh, mw = land_full.shape
    lon = view.west + (np.arange(width, dtype=np.float64) + 0.5) / width * (view.east - view.west)
    lat = view.north - (np.arange(height, dtype=np.float64) + 0.5) / height * (view.north - view.south)
    deg_px = (view.east - view.west) / width  # degrees per output pixel
    src = land_full
    shrink = deg_px / (360.0 / mw)
    if shrink > 1.5:  # a wide view: average the mask first rather than skip pixels
        src = cv2.resize(land_full, (int(mw / shrink), int(mh / shrink)), interpolation=cv2.INTER_AREA)
    sh, sw = src.shape
    mx = (((lon + 180.0) % 360.0) / 360.0 * sw - 0.5).astype(F32)
    my = ((90.0 - lat) / 180.0 * sh - 0.5).astype(F32)
    land = cv2.remap(src, np.tile(mx, (height, 1)), np.repeat(my[:, None], width, axis=1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    land[(lat > 89.5) | (lat < -60.0)] = 0.0  # no Antarctica: it would fill the bottom
    px = max(1.0, min(width, height) / 500.0)
    land = A.blur(land, 0.5 * px)
    tone = 1.0 - 0.3 * land
    if grid:  # faint parallels and meridians every 30°, on the sea only
        near = np.minimum(np.abs((lon + 15.0) % 30.0 - 15.0)[None, :] / deg_px,
                          np.abs((lat + 15.0) % 30.0 - 15.0)[:, None] / deg_px)
        dash = (np.add.outer(np.arange(height), np.arange(width)) // int(6 * px)) % 2
        tone -= 0.1 * np.clip(1.2 - near / px, 0.0, 1.0) * dash * (1.0 - land)
    rgb = np.repeat(np.clip(tone, 0.0, 1.0)[..., None], 3, axis=2).astype(F32)
    return graphite_drawing(rgb, (width, height), seed=seed, border=0.0)


# --------------------------------------------------------------------------- pieces

def pin_sprite(size: int, colour=PIN) -> Sprite:
    """A map pin ``size`` pixels tall; its tip is at the bottom centre."""
    w = int(size * 0.72)

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        r = w * 0.44 * s
        cx, cy = w * s / 2, r + 2 * s
        line = max(1, int(size * 0.05)) * s
        d.polygon([(cx - r * 0.82, cy + r * 0.5), (cx, size * s - 2 * s), (cx + r * 0.82, cy + r * 0.5)],
                  fill=_rgb255(colour))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_rgb255(colour),
                  outline=_rgb255(tuple(c * 0.7 for c in colour)), width=line)
        d.ellipse([cx - r * 0.38, cy - r * 0.38, cx + r * 0.38, cy + r * 0.38], fill=(255, 255, 255, 255))

    return _render(draw, (w, size))


def plane_sprite(size: int) -> Sprite:
    """A paper plane ``size`` pixels long, pointing right."""
    h = int(size * 0.62)
    top = [(1.0, 0.42), (0.0, 0.0), (0.3, 0.46)]
    keel = [(1.0, 0.42), (0.3, 0.46), (0.24, 0.62)]
    low = [(1.0, 0.42), (0.24, 0.62), (0.08, 0.98)]

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        pad = 3 * s
        k = (size * s - 2 * pad, h * s - 2 * pad)
        line = max(2, int(size * 0.03)) * s
        for pts, fill in ((low, 0.86), (keel, 0.66), (top, 0.99)):
            d.polygon([(pad + x * k[0], pad + y * k[1]) for x, y in pts],
                      fill=_rgb255((fill, fill, fill * 0.98)), outline=_rgb255(INK), width=line)

    return _render(draw, (size, h))


def arc(a: Tuple[float, float], b: Tuple[float, float], bend: float = 0.2,
        steps: int = 80) -> np.ndarray:
    """A flight path from ``a`` to ``b`` bulging upwards (points, steps x 2)."""
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    d = b - a
    normal = np.array([d[1], -d[0]], F32)
    if normal[1] > 0:
        normal = -normal
    ctrl = (a + b) / 2 + normal * bend
    t = np.linspace(0.0, 1.0, steps, dtype=F32)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b


def along(path: np.ndarray, share: float) -> Tuple[np.ndarray, float, float]:
    """The part of ``path`` travelled at ``share`` (0..1), and the position and
    heading (degrees, for :meth:`Sprite.paste`) of its head as (x, y, angle)."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    goal = min(max(share, 0.0), 1.0) * dist[-1]
    k = int(np.searchsorted(dist, goal, side="right"))
    k = min(max(k, 1), len(path) - 1)
    f = (goal - dist[k - 1]) / max(seg[k - 1], 1e-6)
    head = path[k - 1] + (path[k] - path[k - 1]) * f
    dx, dy = path[k] - path[k - 1]
    return np.vstack([path[:k], head[None, :]]), head, -math.degrees(math.atan2(dy, dx))


def dashed(frame: np.ndarray, points: np.ndarray, thickness: float, colour=INK,
           dash: float = 14.0, opacity: float = 0.85) -> None:
    """Draw a dashed line through ``points`` onto ``frame`` (float RGB), in place."""
    if len(points) < 2:
        return
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    seg = np.hypot(*np.diff(points, axis=0).T)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    t = max(1, int(round(thickness)))
    for (x0, y0), (x1, y1), d0 in zip(points[:-1], points[1:], dist[:-1]):
        if int(d0 // dash) % 2 == 0:
            cv2.line(mask, (int(x0 * 4), int(y0 * 4)), (int(x1 * 4), int(y1 * 4)), 255, t,
                     cv2.LINE_AA, shift=2)
    a = mask.astype(F32)[..., None] / 255.0 * opacity
    frame *= 1.0 - a
    frame += np.asarray(colour, F32) * a


def map_card(lon: float, lat: float, width: int, height: int, seed: int = 0
             ) -> Tuple[Sprite, Tuple[float, float]]:
    """A small paper card with the map around (``lon``, ``lat``), and where the
    place is on the card."""
    border = max(4, int(0.05 * height))
    iw, ih = width - 2 * border, height - 2 * border
    span = 40.0
    view = MapView(lon - span / 2, lat - span / 2 * ih / iw, lon + span / 2, lat + span / 2 * ih / iw)
    if view.north > 88.0 or view.south < -88.0:
        shift = (88.0 - view.north) if view.north > 88.0 else (-88.0 - view.south)
        view = MapView(view.west, view.south + shift, view.east, view.north + shift)
    inner = pencil_map(view, iw, ih, seed=seed, grid=False)
    card = np.ones((height, width, 4), F32)
    card[..., :3] = PAPER * 1.02
    card[border:border + ih, border:border + iw, :3] = inner
    edge = max(1, height // 120)
    card[:edge, :, :3] = card[-edge:, :, :3] = 0.72
    card[:, :edge, :3] = card[:, -edge:, :3] = 0.72
    px, py = view.xy(lon, lat, iw, ih)
    return Sprite(np.clip(card, 0.0, 1.0)), (border + px, border + py)


# --------------------------------------------------------------------------- build

def build_land_mask(geojson_path: str, out_path: str = LAND_PATH, width: int = 4096) -> None:
    """Rasterise Natural Earth land polygons into the map mask."""
    with open(geojson_path, encoding="utf-8") as fh:
        data = json.load(fh)
    height = width // 2
    mask = np.zeros((height, width), np.uint8)

    def pixels(ring) -> np.ndarray:
        a = np.asarray(ring, np.float64)[:, :2]
        xy = np.stack([(a[:, 0] + 180.0) / 360.0 * width, (90.0 - a[:, 1]) / 180.0 * height], 1)
        return np.round(xy * 8).astype(np.int32)

    for feature in data["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            cv2.fillPoly(mask, [pixels(polygon[0])], 255, cv2.LINE_8, shift=3)
            for hole in polygon[1:]:
                cv2.fillPoly(mask, [pixels(hole)], 0, cv2.LINE_8, shift=3)
    cv2.imwrite(out_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 9, cv2.IMWRITE_PNG_BILEVEL, 1])


if __name__ == "__main__":
    build_land_mask(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else LAND_PATH)
