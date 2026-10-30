"""Simulated pencil overlay that follows the strokes as they are drawn.

The pencil is a visual overlay only. It is moved along the order in which the
software reveals the drawing (see :mod:`pencil_timelapse.plan`); it does not
reproduce how a real artist moved.

The pencil is rendered analytically per pixel (hexagonal lacquered body,
sharpened cedar cone with a scalloped paint edge, polished graphite point),
casts a shadow that is sharp at the tip and softer further up, and is drawn
with motion blur like footage from a real camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import PENCIL_STYLES

F32 = np.float32

PENCIL_ANGLE = 43.0  # the pencil leans into the page from the lower right
LENGTH = 600.0  # reference pixels (1080 px wide video)
RADIUS = 15.0  # half-width of the hexagonal body
CONE_END = 58.0  # where the sharpened cone meets the paint (at the hexagon ridges)
LEAD_END = 11.5

# Paint colour, gloss strength and how matte the finish is, per style.
_STYLES = {
    "black": dict(paint=(0.085, 0.088, 0.095), gloss=0.55, sheen=0.10, grain=0.0),
    "yellow": dict(paint=(0.97, 0.70, 0.10), gloss=0.40, sheen=0.06, grain=0.0),
    "natural": dict(paint=(0.80, 0.60, 0.40), gloss=0.12, sheen=0.03, grain=1.0),
}
_CEDAR = np.array([0.90, 0.72, 0.53], F32)
_GRAPHITE = np.array([0.20, 0.20, 0.215], F32)
# Light in the pencil's cross-section frame: (across the pencil, towards the camera).
_LIGHT = np.array([0.45, 0.89])
_LIGHT /= np.linalg.norm(_LIGHT)


@dataclass
class PencilSprite:
    rgba: np.ndarray  # premultiplied colour + alpha, float32 (h, w, 4)
    shadow: np.ndarray  # soft alpha used for the cast shadow, float32 (h, w)
    tip: Tuple[float, float]  # graphite point in sprite pixels


# --------------------------------------------------------------------------- sprite

def make_pencil(k: float, style: str = "black", seed: int = 5) -> PencilSprite:
    """Render the pencil sprite. ``k`` = size factor (1.0 for a 1080 px wide video)."""
    if style not in PENCIL_STYLES:
        raise ValueError(f"unknown pencil style {style!r}")
    look = _STYLES[style]
    ss = 2  # supersampling for clean edges
    pad = 10.0
    a = math.radians(PENCIL_ANGLE)
    axis = np.array([math.cos(a), math.sin(a)])
    side = np.array([-math.sin(a), math.cos(a)])
    ox, oy = (pad + RADIUS) * k, pad * k  # the graphite point, in output pixels
    w = int(math.ceil((LENGTH * axis[0] + 2 * RADIUS + 2 * pad) * k))
    h = int(math.ceil((LENGTH * axis[1] + 2 * RADIUS + 2 * pad) * k))

    # (u, v): distance along the pencil from the tip and across it, reference px.
    yy, xx = np.mgrid[0:h * ss, 0:w * ss].astype(F32)
    dx, dy = (xx + 0.5) / ss - ox, (yy + 0.5) / ss - oy
    u = (dx * axis[0] + dy * axis[1]) / k
    v = (dx * side[0] + dy * side[1]) / k
    rng = np.random.default_rng(seed)

    # --- sharpened cone and scalloped paint edge (hexagon cut by a cone) -----
    vn = np.clip(v / RADIUS, -1.0, 1.0)
    av = np.abs(vn)
    across = np.where(av < 0.5, vn, (av - 0.75) * 2.0)  # position on the facet, -0.5..0.5
    rho = np.sqrt(0.75 + across ** 2)  # distance of the surface from the axis (hexagon)
    paint_start = CONE_END - 8.0 * (1.0 - rho) / (1.0 - math.sqrt(0.75))
    cone_r = 4.4 + (RADIUS - 4.4) * np.clip((u - LEAD_END) / (CONE_END - LEAD_END), 0.0, 1.0)
    lead_r = 0.7 + 3.9 * np.clip(u / LEAD_END, 0.0, 1.0)

    def edge(dist: np.ndarray) -> np.ndarray:  # signed distance (px) -> coverage
        return np.clip(dist * k + 0.5, 0.0, 1.0)

    inside_len = edge(np.minimum(u + 0.8, LENGTH - u))
    lead_m = edge(lead_r - np.abs(v)) * edge(LEAD_END + 0.6 - u) * inside_len
    cone_m = edge(cone_r - np.abs(v)) * edge(u - LEAD_END + 0.6) * edge(paint_start - u)
    body_m = edge(RADIUS - np.abs(v)) * edge(u - paint_start) * edge(LENGTH - u)

    # --- graphite: polished, slightly metallic ------------------------------
    cn = np.clip(v / np.maximum(lead_r, 1e-3), -1.0, 1.0)
    nz = np.sqrt(1.0 - cn ** 2)
    diffuse = np.clip(cn * _LIGHT[0] + nz * _LIGHT[1], 0.0, 1.0)
    lead_rgb = _GRAPHITE * (0.45 + 0.75 * diffuse[..., None])
    lead_rgb += (0.55 * np.exp(-((cn - 0.35) / 0.18) ** 2) * np.clip(u / 3.0, 0, 1))[..., None]

    # --- cedar cone: grain converging to the tip, matte ----------------------
    cn = np.clip(v / np.maximum(cone_r, 1e-3), -1.0, 1.0)
    nz = np.sqrt(1.0 - cn ** 2)
    diffuse = np.clip(cn * _LIGHT[0] + nz * _LIGHT[1], 0.0, 1.0)
    grain_line = _noise1d(rng, cn * 60.0 + 0.04 * u, 0.9)
    fibre = _noise1d(rng, cn * 260.0, 0.6)
    wood = _CEDAR * (0.58 + 0.5 * diffuse)[..., None]
    wood = wood * (1.0 + 0.10 * grain_line + 0.05 * fibre)[..., None]
    wood *= (1.0 - 0.10 * np.clip((u - 40.0) / 18.0, 0, 1))[..., None]  # darker near the paint

    # --- hexagonal body: lit facets, glossy lacquer, highlights on the ridges -
    facet = np.where(vn < -0.5, -1, np.where(vn > 0.5, 1, 0))
    normals = {-1: (-0.866, 0.5), 0: (0.0, 1.0), 1: (0.866, 0.5)}
    light_level = np.zeros_like(u)
    for f, (nx, nzz) in normals.items():
        light_level = np.where(facet == f, 0.22 + 0.78 * max(0.0, nx * _LIGHT[0] + nzz * _LIGHT[1]),
                               light_level)
    light_level += 0.06 * across * np.sign(facet + 0.5)  # gentle gradient within a facet
    paint = np.asarray(look["paint"], F32)
    body = paint * light_level[..., None]
    if look["grain"]:
        stripes = _noise1d(rng, vn * 45.0 + 0.002 * u, 1.0)
        body *= (1.0 + 0.09 * stripes)[..., None]
    # Broad gloss on the facet facing the light, thin highlights on rounded ridges.
    body += (look["gloss"] * 0.55 * np.exp(-((vn - 0.18) / 0.16) ** 2) * (facet == 0))[..., None]
    ridges = np.exp(-((av - 0.5) / 0.035) ** 2) * np.where(vn > 0, 1.0, 0.45)
    body += (look["gloss"] * 0.75 * ridges)[..., None]
    body += (look["sheen"] * np.exp(-((vn - 0.75) / 0.12) ** 2))[..., None]
    body *= (1.0 - 0.35 * np.clip((av - 0.9) / 0.1, 0.0, 1.0))[..., None]  # silhouette falloff
    lacquer = cv2.GaussianBlur(rng.standard_normal(u.shape).astype(F32), (0, 0), 1.2 * ss * k)
    body *= (1.0 + 0.025 * lacquer)[..., None]
    end_band = np.clip((u - (LENGTH - 5.0)) / 5.0, 0.0, 1.0)  # flat end catches less light
    body *= (1.0 - 0.25 * end_band)[..., None]

    rgb = np.zeros(u.shape + (3,), F32)
    alpha = np.zeros(u.shape, F32)
    for mask, colour in ((body_m, body), (cone_m, wood), (lead_m, lead_rgb)):
        rgb = rgb * (1.0 - mask)[..., None] + np.clip(colour, 0.0, 1.0) * mask[..., None]
        alpha = alpha * (1.0 - mask) + mask
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    u_out = cv2.resize(u, (w, h), interpolation=cv2.INTER_AREA)
    return PencilSprite(np.dstack([rgb, alpha]).astype(F32), _soft_shadow(alpha, u_out, k), (ox, oy))


def _noise1d(rng: np.random.Generator, coord: np.ndarray, contrast: float) -> np.ndarray:
    """Smooth 1-D value noise evaluated at ``coord`` (used for wood grain)."""
    table = cv2.GaussianBlur(rng.standard_normal((1, 4096)).astype(F32), (0, 0), 1.5)[0]
    table /= table.std() + 1e-6
    x = np.mod(coord, 4095.0)
    i = x.astype(np.int64)
    t = x - i
    return (table[i] * (1 - t) + table[i + 1] * t) * contrast


def _soft_shadow(alpha: np.ndarray, u: np.ndarray, k: float) -> np.ndarray:
    """Shadow that is crisp where the tip touches the paper and grows softer
    (and lighter) towards the raised end of the pencil."""
    height = np.clip(u / LENGTH, 0.0, 1.0)
    sigmas = [0.7, 2.0, 4.5, 8.0, 12.0]
    stack = [cv2.GaussianBlur(alpha, (0, 0), s * k) for s in sigmas]
    target = 0.7 + 11.3 * height
    shadow = np.zeros_like(alpha)
    for i in range(len(sigmas) - 1):
        lo, hi = sigmas[i], sigmas[i + 1]
        wgt = np.clip((target - lo) / (hi - lo), 0.0, 1.0)
        sel = (target >= lo) & (target <= hi)
        shadow = np.where(sel, stack[i] * (1 - wgt) + stack[i + 1] * wgt, shadow)
    shadow *= 1.0 - 0.45 * height
    # A small, dark contact shadow right where the graphite touches the page.
    contact = np.exp(-(u / 6.0) ** 2) * cv2.GaussianBlur(alpha, (0, 0), 1.2 * k)
    return np.clip(np.maximum(shadow, 1.4 * contact), 0.0, 1.0).astype(F32)


# --------------------------------------------------------------------------- motion + drawing

@dataclass
class PencilRig:
    """Pre-computed pencil trajectory plus the sprite, drawn on top of frames."""

    sprite: PencilSprite
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    lift: np.ndarray  # 0 = tip on the paper, 1 = lifted between strokes
    angle: np.ndarray  # small extra rotation in degrees
    end_time: float
    k: float
    shutter: float  # exposure time of one frame (seconds), for motion blur
    shadow_strength: float = 0.42

    def state(self, t: float) -> Tuple[float, float, float, float]:
        return tuple(float(np.interp(t, self.times, arr))  # type: ignore[return-value]
                     for arr in (self.x, self.y, self.lift, self.angle))

    def _matrices(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        x, y, lift, angle = self.state(t)
        sp = self.sprite
        theta = math.radians(angle)

        def matrix(scale: float, ox: float, oy: float) -> np.ndarray:
            c, s = math.cos(theta) * scale, math.sin(theta) * scale
            tx, ty = sp.tip
            return np.array([[c, -s, x + ox - (c * tx - s * ty)],
                             [s, c, y + oy - (s * tx + c * ty)]], np.float64)

        # Lifting the pencil brings it towards the camera: slightly bigger, with
        # its shadow falling further away. The shadow is scaled about the tip,
        # which touches the paper.
        return (matrix(1.0 + 0.03 * lift, 0.0, 0.0),
                matrix(1.05 + 0.04 * lift, (3 + 12 * lift) * self.k, (5 + 16 * lift) * self.k))

    def draw(self, frame: np.ndarray, t: float) -> None:
        """Composite the pencil and its shadow onto ``frame`` (float RGB) in place."""
        if t < 0.0 or t >= self.end_time:
            return
        H, W = frame.shape[:2]
        h, w = self.sprite.shadow.shape
        # Motion blur: average several positions within the frame's exposure.
        # The streak is capped by how far the tip actually wanders during the
        # exposure (back-and-forth hatching can return close to its start), so
        # very fast scribbling never turns into a wide translucent smear.
        cap = 20.0 * self.k
        t0, t1 = max(0.0, t - 0.5 * self.shutter), min(self.end_time, t + 0.5 * self.shutter)
        centre = np.array(self.state(t)[:2])
        for _ in range(2):
            probe = np.array([self.state(u)[:2] for u in np.linspace(t0, t1, 16)])
            spread = float(np.max(np.hypot(*(probe - centre).T)))
            if spread <= cap:
                break
            half = 0.5 * (t1 - t0) * cap / spread
            t0, t1 = max(0.0, t - half), min(self.end_time, t + half)
        path = float(np.hypot(*np.diff(probe, axis=0).T).sum())
        steps = int(np.clip(math.ceil(min(path, 2 * cap) / (2.0 * self.k + 0.5)), 1, 12))
        samples = [self._matrices(t)] if steps == 1 else [
            self._matrices(t0 + (t1 - t0) * (i + 0.5) / steps) for i in range(steps)]
        corners = np.array([(0, 0), (w, 0), (0, h), (w, h)], np.float64)
        box = np.concatenate([corners @ m[:, :2].T + m[:, 2] for pair in samples for m in pair])
        x0 = max(0, int(math.floor(box[:, 0].min())) - 2)
        y0 = max(0, int(math.floor(box[:, 1].min())) - 2)
        x1 = min(W, int(math.ceil(box[:, 0].max())) + 2)
        y1 = min(H, int(math.ceil(box[:, 1].max())) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        size = (x1 - x0, y1 - y0)
        shift = np.array([[0, 0, x0], [0, 0, y0]], np.float64)
        rgba = np.zeros((size[1], size[0], 4), F32)
        shadow = np.zeros((size[1], size[0]), F32)
        for m_pencil, m_shadow in samples:
            rgba += cv2.warpAffine(self.sprite.rgba, m_pencil - shift, size, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
            shadow += cv2.warpAffine(self.sprite.shadow, m_shadow - shift, size,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgba /= len(samples)
        shadow /= len(samples)
        roi = frame[y0:y1, x0:x1]
        roi *= (1.0 - self.shadow_strength * shadow)[..., None]
        roi *= 1.0 - rgba[..., 3:4]
        roi += rgba[..., :3]


def build_rig(plan, sprite: PencilSprite, fps: float = 30.0, rate: float = 1920.0) -> PencilRig:
    """Plan where the pencil tip is at every moment.

    Every pass of the drawing is a sequence of strokes, so the tip simply
    traces the stroke being drawn at each moment (lifting slightly between
    strokes) - graphite therefore appears exactly where the tip touches the
    paper. The path is lightly smoothed without lag; the pencil enters from
    the lower right and moves away once the last stroke is done.
    """
    W, H = plan.width, plan.height
    k = min(W, H) / 1080.0
    finish1 = plan.finish[1]
    sources: List = [l for l in plan.layers if l.path_xy is not None and l.channel == "draw"]
    sources += [ov for ov in plan.overlays if ov.path_xy is not None]
    # Leave once the last stroke is done, before the frame settles on the end image.
    leave = min(max((src.t1 for src in sources), default=plan.finish[0]), finish1 - 0.05)
    exit_end = leave + 0.6 * max(finish1 - leave, 0.1)
    times = np.arange(0.0, exit_end + 1.0 / rate, 1.0 / rate)
    n = len(times)
    px, py = np.full(n, np.nan), np.full(n, np.nan)
    lift = np.ones(n)
    best = np.full(n, -np.inf)
    for src in sources:
        if src.t1 - src.t0 < 1e-3:
            continue
        sel = (times > src.t0) & (times < src.t1)
        if not sel.any():
            continue
        # The tip is at the leading edge of the stroke being revealed.
        s = (times[sel] - src.t0) / (src.t1 - src.t0) * (1.0 + src.feather) - 0.3 * src.feather
        s = np.clip(s, 0.0, 1.0)
        # Minor passes may run in the background of a stage: follow the one
        # adding the most graphite per second.
        score = getattr(src, "work", 0.0) / max(src.t1 - src.t0, 1e-3) + 1e-6
        win = score > best[sel]
        if not win.any():
            continue
        idx = np.flatnonzero(sel)[win]
        sw = s[win]
        best[idx] = score
        px[idx] = np.interp(sw, src.path_tau, src.path_xy[:, 0])
        py[idx] = np.interp(sw, src.path_tau, src.path_xy[:, 1])
        j = np.clip(np.searchsorted(src.path_tau, sw), 1, len(src.path_tau) - 1)
        lift[idx] = (src.path_ids[j] != src.path_ids[j - 1]).astype(np.float64)

    # Off-screen at the lower right before the first stroke and after the last.
    off = (1.12 * W + 40 * k, 1.04 * H + 40 * k)
    gone = times >= exit_end - 1e-9
    px[0], py[0] = off
    px[gone], py[gone] = off
    leaving = (times >= leave) & ~gone
    px[leaving], py[leaving] = np.nan, np.nan
    valid = ~np.isnan(px)
    px = np.interp(times, times[valid], px[valid])
    py = np.interp(times, times[valid], py[valid])
    # Sampled densely and barely smoothed: in a sped-up drawing the tip can
    # cover several hatching strokes per frame and must stay on each of them.
    px, py = _smooth(px, 0.0015 * rate), _smooth(py, 0.0015 * rate)
    lift = np.clip(_smooth(lift, 0.02 * rate), 0.0, 1.0)

    # The pencil tilts a little as it reaches across the page and trails
    # slightly behind fast sideways movements, like a hand-held pencil.
    vx = np.gradient(_smooth(px, 0.04 * rate)) * rate / max(k, 1e-6)
    drag = np.clip(-vx / 900.0, -1.0, 1.0) * 3.0
    angle = -6.0 * (px / W - 0.5) - 3.0 * (py / H - 0.5) + _smooth(drag, 0.05 * rate)
    return PencilRig(sprite, times, px, py, lift, angle, exit_end, k, shutter=0.5 / fps)


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing (no lag)."""
    if sigma <= 0.5:
        return values
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")
