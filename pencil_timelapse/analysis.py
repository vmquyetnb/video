"""Computer-vision primitives used to plan a plausible drawing order.

Everything here is deterministic: the same inputs and seed always give the
same strokes, regions and timings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

F32 = np.float32


# --------------------------------------------------------------------------- small helpers

def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return smoothstep((x - lo) / max(hi - lo, 1e-6))


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def robust_max(values: np.ndarray, q: float = 99.5, sample: int = 200_000) -> float:
    flat = values.ravel()
    flat = flat[flat > 0]
    if flat.size == 0:
        return 1.0
    if flat.size > sample:
        flat = flat[:: flat.size // sample]
    value = float(np.percentile(flat, q))
    return value if value > 1e-9 else 1.0


def smooth_noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """1-D smooth random signal with unit standard deviation (hand wobble)."""
    radius = max(1, int(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    signal = np.convolve(rng.standard_normal(n + 2 * radius), kernel / kernel.sum(), "valid")[:n]
    return signal / (signal.std() + 1e-9)


def uniformize(x: np.ndarray) -> np.ndarray:
    """Map a roughly Gaussian field to roughly uniform values in (0, 1)."""
    z = (x - x.mean()) / (x.std() + 1e-6)
    return (1.0 / (1.0 + np.exp(-1.702 * z))).astype(F32)


def resize_to(img: np.ndarray, width: int, height: int, interp: int = cv2.INTER_AREA) -> np.ndarray:
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=interp)


# --------------------------------------------------------------------------- stroke flow

def orientation_field(lum: np.ndarray, scale: float, default_angle_deg: float = 55.0
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direction pencil strokes should follow at every pixel.

    Uses the edge-tangent of the smoothed structure tensor where the image has
    a clear direction (hair, contours) and falls back to a diagonal hatching
    angle in flat areas. Returns unit vectors ``(vx, vy)`` and the coherence.
    """
    g = blur(lum, 1.2 * scale)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    rho = 5.0 * scale
    jxx, jyy, jxy = blur(gx * gx, rho), blur(gy * gy, rho), blur(gx * gy, rho)
    # Doubled-angle representation of the tangent (gradient rotated by 90 deg).
    a = jyy - jxx
    b = -2.0 * jxy
    mag = np.sqrt(a * a + b * b) + 1e-12
    trace = jxx + jyy + 1e-12
    coherence = (mag / trace).astype(F32)
    energy = ramp(trace, 0.0, robust_max(trace, 90.0))
    w = smoothstep(coherence * 1.4) * energy
    theta = -math.radians(default_angle_deg)  # y axis points down: negative = up-right
    ux = w * (a / mag) + (1.0 - w) * math.cos(2 * theta)
    uy = w * (b / mag) + (1.0 - w) * math.sin(2 * theta)
    ux, uy = blur(ux.astype(F32), 2.0 * scale), blur(uy.astype(F32), 2.0 * scale)
    ang = 0.5 * np.arctan2(uy, ux)
    return np.cos(ang).astype(F32), np.sin(ang).astype(F32), coherence


def rotate_field(vx: np.ndarray, vy: np.ndarray, degrees: float) -> Tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return (vx * c - vy * s).astype(F32), (vx * s + vy * c).astype(F32)


def line_integral_convolution(vx: np.ndarray, vy: np.ndarray, textures: Sequence[np.ndarray],
                              length: int, step: float) -> List[np.ndarray]:
    """Smear noise textures along the flow field (LIC) to create stroke streaks.

    Streamlines are traced once and shared by all textures.
    """
    h, w = vx.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=F32), np.arange(h, dtype=F32))
    acc = [t.astype(F32).copy() for t in textures]
    wsum = 1.0
    for sign in (1.0, -1.0):
        px, py = xs.copy(), ys.copy()
        dx, dy = vx * sign, vy * sign
        for i in range(1, length + 1):
            px += dx * step
            py += dy * step
            nx = cv2.remap(vx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ny = cv2.remap(vy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # Orientation fields are sign-ambiguous: keep marching the same way.
            flip = np.where(nx * dx + ny * dy < 0, -1.0, 1.0).astype(F32)
            norm = np.sqrt(nx * nx + ny * ny) + 1e-6
            dx, dy = nx * flip / norm, ny * flip / norm
            weight = 0.5 * (1.0 + math.cos(math.pi * i / (length + 1)))
            for k, tex in enumerate(textures):
                acc[k] += weight * cv2.remap(tex, px, py, cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT)
            wsum += weight
    return [a / wsum for a in acc]


# --------------------------------------------------------------------------- lines

def line_mask(ink: np.ndarray, sigma: float, hi: float, lo: float,
              boost: Optional[np.ndarray] = None) -> np.ndarray:
    """Binary mask of line-like structures in an ink map.

    Combines a centre-surround ridge detector (thin dark lines) with gradient
    magnitude (boundaries of dark masses) and applies hysteresis thresholds.
    Both responses are divided by the local ink level so that graphite grain
    inside dark masses is not mistaken for lines.
    """
    a = blur(ink, sigma)
    surround = blur(ink, 3.0 * sigma)
    norm = 1.0 / (surround + 0.12)
    ridge = np.maximum(a - surround, 0.0) * norm
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy) * blur(norm, 2.0 * sigma)
    strength = np.maximum(ridge / robust_max(ridge), 0.7 * edge / robust_max(edge))
    if boost is not None:
        strength = strength * boost
    weak = (strength > lo).astype(np.uint8)
    strong = strength > hi
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels]


def thin(mask: np.ndarray, max_iter: int = 40) -> np.ndarray:
    """Zhang-Suen thinning (vectorized). Returns a uint8 0/1 skeleton."""
    img = (mask > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            ring = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbours = sum(p.astype(np.uint8) for p in ring)
            transitions = sum(((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.uint8)
                              for i in range(8))
            if step == 0:
                cond = ((p2 & p4 & p6) == 0) & ((p4 & p6 & p8) == 0)
            else:
                cond = ((p2 & p4 & p8) == 0) & ((p2 & p6 & p8) == 0)
            remove = (img == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & cond
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return img


def extract_strokes(skeleton: np.ndarray, min_len: int) -> List[np.ndarray]:
    """Turn a 1-px skeleton into continuous polylines (x, y float32 arrays)."""
    h, w = skeleton.shape
    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    visited = np.zeros(h * w, bool)
    strokes: List[np.ndarray] = []
    for contour in contours:
        pts = contour[:, 0, :]
        lin = pts[:, 1] * w + pts[:, 0]
        # A contour around a 1-px curve walks out and back; keep first visits only.
        _, first = np.unique(lin, return_index=True)
        first.sort()
        pts, lin = pts[first], lin[first]
        fresh = ~visited[lin]
        visited[lin] = True
        pts = pts[fresh]
        if len(pts) < min_len:
            continue
        jumps = np.abs(np.diff(pts, axis=0)).max(axis=1) > 1
        for seg in np.split(pts, np.flatnonzero(jumps) + 1):
            if len(seg) >= min_len:
                strokes.append(seg.astype(F32))
    return strokes


def polyline_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    d = np.diff(p, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def smooth_polyline(p: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1 or len(p) < 3:
        return p
    k = 2 * radius + 1
    kernel = np.ones(k, F32) / k
    padded = np.pad(p, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, 0], kernel, "valid"),
                     np.convolve(padded[:, 1], kernel, "valid")], axis=1).astype(F32)


def resample_polyline(p: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    if len(p) < 2:
        return p.astype(F32)
    d = np.diff(p, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    if cum[-1] < 1e-6:
        return p[:1].astype(F32)
    s = np.linspace(0.0, cum[-1], max(2, int(math.ceil(cum[-1] / spacing)) + 1))
    return np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], axis=1).astype(F32)


def follows_flow(stroke: np.ndarray, vx: np.ndarray, vy: np.ndarray, coherence: np.ndarray,
                 min_alignment: float = 0.75, coherent: float = 0.4) -> bool:
    """False for strokes that wander across a strongly directional area (such
    as squiggles traced from grain inside hair) instead of following it."""
    if len(stroke) < 9:
        return True
    h, w = vx.shape
    t = stroke[4:] - stroke[:-4]
    t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-6
    mid = stroke[2:-2]
    xi = np.clip(np.rint(mid[:, 0]).astype(np.int64), 0, w - 1)
    yi = np.clip(np.rint(mid[:, 1]).astype(np.int64), 0, h - 1)
    c = coherence[yi, xi]
    if float(c.mean()) < coherent:
        return True
    dot = np.abs(t[:, 0] * vx[yi, xi] + t[:, 1] * vy[yi, xi])
    return float((dot * c).sum() / (c.sum() + 1e-6)) >= min_alignment


def order_strokes(strokes: List[np.ndarray], focus_points: Sequence[Tuple[float, float]],
                  diag: float, w_focus: float, w_length: float,
                  start: Tuple[float, float]) -> List[np.ndarray]:
    """Greedy "pen path": pick the next stroke by travel distance, closeness to
    the focal points and stroke length; strokes may be drawn in reverse."""
    n = len(strokes)
    if n == 0:
        return []
    heads = np.array([s[0] for s in strokes], dtype=np.float64)
    tails = np.array([s[-1] for s in strokes], dtype=np.float64)
    centres = np.array([s.mean(axis=0) for s in strokes], dtype=np.float64)
    lengths = np.array([polyline_length(s) for s in strokes], dtype=np.float64)
    focus = np.array(focus_points, dtype=np.float64).reshape(-1, 2)
    focus_dist = np.min(np.hypot(centres[:, None, 0] - focus[None, :, 0],
                                 centres[:, None, 1] - focus[None, :, 1]), axis=1) / diag
    length_term = np.log1p(lengths / (0.01 * diag))
    length_term /= max(length_term.max(), 1e-6)
    static = w_focus * focus_dist - w_length * length_term
    used = np.zeros(n, bool)
    pos = np.array(start, dtype=np.float64)
    ordered: List[np.ndarray] = []
    for _ in range(n):
        dh = np.hypot(heads[:, 0] - pos[0], heads[:, 1] - pos[1])
        dt = np.hypot(tails[:, 0] - pos[0], tails[:, 1] - pos[1])
        cost = np.minimum(dh, dt) / diag + static
        cost[used] = np.inf
        i = int(np.argmin(cost))
        used[i] = True
        stroke = strokes[i][::-1] if dt[i] < dh[i] else strokes[i]
        ordered.append(stroke)
        pos = stroke[-1].astype(np.float64)
    return ordered


def stroke_point_times(strokes: List[np.ndarray], length_exponent: float = 0.85,
                       gap_ratio: float = 0.15, taper: float = 0.35
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give every point a normalized time (strokes grow from start to end) and
    a radius factor that tapers both ends like a real pencil line."""
    points, taus, radii, _ = stroke_point_times_with_ids(strokes, length_exponent, gap_ratio, taper)
    return points, taus, radii


def stroke_point_times_with_ids(strokes: List[np.ndarray], length_exponent: float = 0.85,
                                gap_ratio: float = 0.15, taper: float = 0.35
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`stroke_point_times` but also returns each stroke's start time."""
    lengths = np.array([max(polyline_length(s), 1.0) for s in strokes])
    durations = lengths ** length_exponent
    gap = gap_ratio * float(np.median(durations))
    total = durations.sum() + gap * max(len(strokes) - 1, 0)
    points, taus, radii, starts = [], [], [], []
    t = 0.0
    for stroke, dur, length in zip(strokes, durations, lengths):
        starts.append(t / total)
        if len(stroke) > 1:
            d = np.diff(stroke, axis=0)
            u = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))]) / length
        else:
            u = np.zeros(1)
        u = np.clip(u, 0.0, 1.0)
        points.append(stroke)
        taus.append((t + u * dur) / total)
        radii.append(1.0 - taper + taper * np.sin(np.pi * u))
        t += dur + gap
    return (np.concatenate(points).astype(F32), np.concatenate(taus).astype(F32),
            np.concatenate(radii).astype(F32), np.array(starts, F32))


def tube_field(points: np.ndarray, taus: np.ndarray, radii: np.ndarray,
               shape: Tuple[int, int], radius: float, with_distance: bool = False):
    """Spread stroke-point times into a tube around the strokes.

    Every pixel within ``radius`` of a stroke takes the time of its nearest
    stroke point (via a labelled distance transform), so lines grow along
    their length. Returns flat pixel indices, times and soft coverage (and the
    distance to the stroke centre line with ``with_distance=True``).
    """
    h, w = shape
    xi = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, w - 1)
    yi = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, h - 1)
    lin = yi * w + xi
    order = np.lexsort((taus, lin))  # earliest time wins on shared pixels
    lin_sorted = lin[order]
    first = np.ones(len(order), bool)
    first[1:] = lin_sorted[1:] != lin_sorted[:-1]
    sel = order[first]
    seed_idx = lin[sel]
    src = np.full(h * w, 255, np.uint8)
    src[seed_idx] = 0
    dist, labels = cv2.distanceTransformWithLabels(src.reshape(h, w), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    lab_flat, d_flat = labels.ravel(), dist.ravel()
    lut_tau = np.zeros(int(labels.max()) + 1, F32)
    lut_rad = np.zeros_like(lut_tau)
    lut_tau[lab_flat[seed_idx]] = taus[sel]
    lut_rad[lab_flat[seed_idx]] = radii[sel]
    cand = np.flatnonzero(d_flat <= radius + 1.0)
    lab = lab_flat[cand]
    cover = np.clip(lut_rad[lab] * radius + 0.5 - d_flat[cand], 0.0, 1.0)
    keep = cover > 0.02
    if with_distance:
        return (cand[keep], lut_tau[lab[keep]], cover[keep].astype(F32),
                d_flat[cand][keep].astype(F32))
    return cand[keep], lut_tau[lab[keep]], cover[keep].astype(F32)


# --------------------------------------------------------------------------- regions

def segment_regions(ink: np.ndarray, tone: np.ndarray, rows: int, cols: int,
                    iterations: int = 15) -> Tuple[np.ndarray, int]:
    """Split the drawing into compact "work areas" with k-means on position
    and tone, initialised from a grid so the result is deterministic."""
    h, w = ink.shape
    s = 1.0 / max(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ink > 0.02
    if mask.sum() < rows * cols * 30:
        mask = np.ones_like(mask, dtype=bool)
    feats = np.stack([xx[mask] * s, yy[mask] * s, tone[mask] * 0.3], axis=1).astype(F32)
    cell = (np.minimum(yy[mask] * rows // h, rows - 1) * cols
            + np.minimum(xx[mask] * cols // w, cols - 1))
    _, init = np.unique(cell, return_inverse=True)
    k = int(init.max()) + 1
    centres = np.stack([feats[init == j].mean(axis=0) for j in range(k)])
    labels = init
    for _ in range(iterations):
        d = ((feats[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for j in range(k):
            sel = labels == j
            if sel.any():
                centres[j] = feats[sel].mean(axis=0)
    full = np.full((h, w), -1, np.int32)
    full[mask] = labels
    if (full < 0).any():  # paper pixels join their nearest work area
        src = np.where(full >= 0, 0, 255).astype(np.uint8)
        _, near = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5,
                                                  labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(full >= 0)
        lut = np.zeros(int(near.max()) + 1, np.int32)
        lut[near[ys, xs]] = full[ys, xs]
        full = lut[near]
    full = cv2.medianBlur(full.astype(np.uint8), 5).astype(np.int32)
    return full, k


def region_order(labels: np.ndarray, k: int, weight: np.ndarray, darkness: np.ndarray,
                 focus: Tuple[float, float], mode: str) -> Tuple[List[int], list]:
    """Order in which an artist would visit the work areas.

    ``mode="focus"`` starts near the focal point (eyes/face) and travels to
    neighbouring areas; ``mode="dark"`` visits the darkest areas first, the
    way shadows are usually blocked in. Returns region ids in order and the
    per-region statistics (mask, centre, darkness, principal axis).
    """
    h, w = labels.shape
    diag = math.hypot(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    stats: list = []
    for j in range(k):
        m = labels == j
        if not m.any():
            stats.append(None)
            continue
        px, py = xx[m], yy[m]
        wt = weight[m] + 1e-3
        cx, cy = float((px * wt).sum() / wt.sum()), float((py * wt).sum() / wt.sum())
        dark = float((darkness[m] * wt).sum() / wt.sum())
        cov = np.cov(np.stack([px - cx, py - cy]), aweights=wt) if m.sum() > 2 else np.eye(2)
        if not np.all(np.isfinite(cov)):
            cov = np.eye(2)
        vals, vecs = np.linalg.eigh(np.atleast_2d(cov))
        stats.append((m, cx, cy, dark, vecs[:, int(np.argmax(vals))]))
    valid = [j for j in range(k) if stats[j] is not None]
    centres = np.array([[stats[j][1], stats[j][2]] for j in valid])
    darks = np.array([stats[j][3] for j in valid])
    focus_d = np.hypot(centres[:, 0] - focus[0], centres[:, 1] - focus[1]) / diag
    if mode == "dark":
        rel = (darks - darks.min()) / max(darks.max() - darks.min(), 1e-6)
        score = 0.8 * (1.0 - rel) + 0.4 * focus_d
    else:
        score = focus_d
    remaining = list(range(len(valid)))
    order = [remaining.pop(int(np.argmin(score)))]
    while remaining:
        cur = centres[order[-1]]
        rem = np.array(remaining)
        cost = np.hypot(centres[rem, 0] - cur[0], centres[rem, 1] - cur[1]) / diag + 0.6 * score[rem]
        order.append(remaining.pop(int(np.argmin(cost))))
    return [valid[i] for i in order], stats


def region_schedule(labels: np.ndarray, k: int, weight: np.ndarray, darkness: np.ndarray,
                    focus: Tuple[float, float], mode: str, passes: int = 1,
                    pass_index: int = 0) -> np.ndarray:
    """Time field that visits work areas one after another (see :func:`region_order`).

    With ``passes > 1`` each area's time slot is split into that many parts and
    this field covers part ``pass_index``, so several passes can finish one
    area before moving on to the next. Inside an area the reveal sweeps along
    its principal axis, starting from the side closest to the previous area.
    """
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    order, stats = region_order(labels, k, weight, darkness, focus, mode)
    tau = np.zeros((h, w), F32)
    prev = np.array(focus, dtype=np.float64)
    n = len(order)
    for rank, j in enumerate(order):
        m, cx, cy, _, axis = stats[j]
        proj = (xx[m] - cx) * axis[0] + (yy[m] - cy) * axis[1]
        lo, hi = np.percentile(proj, [2, 98])
        u = np.clip((proj - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        start_hi = (prev[0] - cx) * axis[0] + (prev[1] - cy) * axis[1] > 0
        if start_hi:
            u = 1.0 - u
        tau[m] = (rank + (pass_index + 0.92 * u) / passes) / n
        end_proj = lo if start_hi else hi
        prev = np.array([cx + axis[0] * end_proj, cy + axis[1] * end_proj])
    return blur(tau, 0.012 * max(h, w) + 1.0)


def hatch_strokes(mask: np.ndarray, angle: float, spacing: float, rng: np.random.Generator,
                  stroke_len: Tuple[float, float] = (50.0, 140.0), step: float = 1.5,
                  bow: float = 1.5) -> List[np.ndarray]:
    """Parallel hatching strokes that cover ``mask``, in the order a hand draws them.

    Lines at ``angle`` (radians) and ``spacing`` px are clipped to the mask and
    cut into strokes of varied length. Strokes are drawn in blocks: a band of
    short side-by-side strokes going back and forth, then the next band, the
    way an artist hatches a patch and moves on. Each stroke is slightly bowed
    like a wrist-pivoted pencil mark.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return []
    h, w = mask.shape
    u = np.array([math.cos(angle), math.sin(angle)])
    n = np.array([-u[1], u[0]])
    pu, pn = xs * u[0] + ys * u[1], xs * n[0] + ys * n[1]
    offsets = np.arange(pn.min() + 0.5 * spacing, pn.max() + 0.5 * spacing, spacing)
    offsets = offsets + rng.uniform(-0.12, 0.12, len(offsets)) * spacing
    pos = np.arange(pu.min() - step, pu.max() + 2 * step, step)
    gx = pos[None, :] * u[0] + offsets[:, None] * n[0]
    gy = pos[None, :] * u[1] + offsets[:, None] * n[1]
    xi, yi = np.rint(gx).astype(np.int64), np.rint(gy).astype(np.int64)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    inside = np.zeros(gx.shape, bool)
    inside[ok] = mask[yi[ok], xi[ok]]

    band = 0.5 * (stroke_len[0] + stroke_len[1]) / step  # band length in samples
    pieces = []  # (band, line, start, end)
    for i in range(len(offsets)):
        row = inside[i].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row, [0]])))
        shift = rng.uniform(0.0, 0.6) * band  # stagger band edges between lines
        for a, b in zip(edges[0::2], edges[1::2]):
            while b - a >= 2:
                k = int((a + shift) // band)
                cut = int((k + 1) * band - shift + rng.uniform(-0.2, 0.2) * band)
                if cut - a < 0.3 * band:  # no stubs at the start of a run
                    cut += int(band)
                end = b if cut >= b or b - cut < 0.3 * band else cut
                pieces.append((k, i, a, end))
                a = end
    pieces.sort(key=lambda p: (p[0], p[1] if p[0] % 2 == 0 else -p[1], p[2]))
    strokes: List[np.ndarray] = []
    for j, (_, i, a, b) in enumerate(pieces):
        idx = np.arange(a, b) if j % 2 == 0 else np.arange(b - 1, a - 1, -1)  # back and forth
        if len(idx) < 2:
            continue
        # Hand-made stroke ends: some overshoot the area, some stop short, so
        # the edge of a shaded patch is ragged instead of a clean outline.
        d = np.array([gx[i, idx[-1]] - gx[i, idx[0]], gy[i, idx[-1]] - gy[i, idx[0]]])
        d /= np.hypot(*d) + 1e-9
        e0, e1 = rng.uniform(-0.3, 1.0, 2) * spacing
        head = np.array([gx[i, idx[0]], gy[i, idx[0]]]) - d * e0
        tail = np.array([gx[i, idx[-1]], gy[i, idx[-1]]]) + d * e1
        count = max(2, int(np.hypot(*(tail - head)) / step) + 1)
        t = np.linspace(0.0, 1.0, count)
        line = head[None, :] + (tail - head)[None, :] * t[:, None]
        sag = bow * rng.uniform(0.4, 1.0) * np.sin(np.pi * t)
        pts = line + n[None, :] * sag[:, None]
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        strokes.append(pts.astype(F32))
    return strokes


def equalize_times(tau: np.ndarray, weight: np.ndarray, mix: float = 0.85) -> np.ndarray:
    """Re-time a layer so its weighted amount of revealed ink grows steadily.

    Areas with little left to draw (for example because the start image
    already contains them) automatically take less time.
    """
    tau = tau.astype(F32)
    if tau.size == 0:
        return tau
    lo, hi = float(tau.min()), float(tau.max())
    base = (tau - lo) / max(hi - lo, 1e-9)
    w = weight.astype(np.float64)
    total = w.sum()
    if total <= 1e-6:
        return base.astype(F32)
    order = np.argsort(tau, kind="stable")
    cw = np.cumsum(w[order])
    eq = np.empty_like(base)
    eq[order] = (cw - 0.5 * w[order]) / total
    return np.clip(mix * eq + (1.0 - mix) * base, 0.0, 1.0).astype(F32)


# --------------------------------------------------------------------------- focus

@dataclass
class Focus:
    """Where the eye of the viewer (and the artist) goes first."""

    centre: Tuple[float, float]
    points: List[Tuple[float, float]]
    method: str
    face: Optional[Tuple[int, int, int, int]] = None
    eyes: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"method": self.method, "centre": [round(v, 1) for v in self.centre],
                "face": list(self.face) if self.face else None,
                "eyes": [list(e) for e in self.eyes]}


def _cascade(name: str):
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    base = getattr(getattr(cv2, "data", None), "haarcascades", "")
    clf = cv2.CascadeClassifier(base + name) if base else None
    return None if clf is None or clf.empty() else clf


def detect_focus(lum: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Focus:
    """Locate the face and eyes with OpenCV Haar cascades when possible,
    otherwise fall back to the area with the densest fine detail.

    ``region`` (x, y, w, h) limits the search to that part of the picture.
    """
    if region is not None:
        x, y, w, h = region
        f = detect_focus(lum[y:y + h, x:x + w])

        def moved(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
            return (box[0] + x, box[1] + y, box[2], box[3])

        return Focus((f.centre[0] + x, f.centre[1] + y), [(px + x, py + y) for px, py in f.points],
                     f.method, moved(f.face) if f.face else None, [moved(e) for e in f.eyes])
    h, w = lum.shape
    s = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(lum, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(cv2.convertScaleAbs(small, alpha=255.0))
    sh, sw = gray.shape
    face = None
    for name, neighbours in (("haarcascade_frontalface_default.xml", 6),
                             ("haarcascade_frontalface_alt2.xml", 4),
                             ("haarcascade_profileface.xml", 5)):
        clf = _cascade(name)
        if clf is None:
            continue
        min_side = int(0.12 * min(sh, sw))
        found = clf.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=neighbours,
                                     minSize=(min_side, min_side))
        if len(found):
            face = max(found, key=lambda r: r[2] * r[3])
            break
    if face is not None:
        fx, fy, fw, fh = (int(v) for v in face)
        eyes: List[Tuple[int, int, int, int]] = []
        clf = _cascade("haarcascade_eye.xml")
        if clf is not None:
            roi = gray[fy:fy + int(fh * 0.62), fx:fx + fw]
            found = clf.detectMultiScale(roi, scaleFactor=1.08, minNeighbors=5,
                                         minSize=(max(8, fw // 10),) * 2,
                                         maxSize=(max(9, fw // 2),) * 2)
            for ex, ey, ew, eh in sorted(found, key=lambda r: -r[2] * r[3]):
                cx = fx + ex + ew / 2
                if all(abs(cx - (e[0] + e[2] / 2)) > fw * 0.18 for e in eyes):
                    eyes.append((fx + ex, fy + ey, ew, eh))
                if len(eyes) == 2:
                    break
        inv = 1.0 / s
        face_full = (int(fx * inv), int(fy * inv), int(fw * inv), int(fh * inv))
        eyes_full = [(int(x * inv), int(y * inv), int(ew * inv), int(eh * inv)) for x, y, ew, eh in eyes]
        centre = ((fx + fw / 2) * inv, (fy + fh * 0.42) * inv)
        points = [(x + ew / 2, y + eh / 2) for x, y, ew, eh in eyes_full] or [centre]
        method = "face + eyes" if eyes_full else "face"
        return Focus(centre, points, method, face_full, eyes_full)

    # Fallback: densest fine detail, with a gentle prior towards the upper centre
    # where portrait faces usually sit.
    detail = np.abs(cv2.Laplacian(blur(small, 1.0), cv2.CV_32F, ksize=3))
    detail = blur(detail, 0.06 * max(sh, sw))
    yy, xx = np.mgrid[0:sh, 0:sw].astype(F32)
    prior = np.exp(-(((xx - 0.5 * sw) / (0.35 * sw)) ** 2 + ((yy - 0.38 * sh) / (0.32 * sh)) ** 2))
    score = detail * (0.35 + prior)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    centre = (x / s, y / s)
    return Focus(centre, [centre], "detail density")
