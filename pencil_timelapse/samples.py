"""Procedural sample images: a synthetic graphite portrait plus start states.

Used by the test-suite and the demo script so the project can be exercised
without shipping or downloading any real person's likeness.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

F32 = np.float32


class _Sketch:
    def __init__(self, width: int, height: int, seed: int):
        self.w, self.h = width, height
        self.rng = np.random.default_rng(seed)
        self.k = width / 900.0
        self.lines = np.zeros((height, width), F32)  # contour/feature lines
        self.tone = np.zeros((height, width), F32)  # shading

    def _paint(self, target: str, draw, value: float) -> None:
        tmp = np.zeros((self.h, self.w), np.uint8)
        draw(tmp)
        layer = tmp.astype(F32) / 255.0 * value
        if target == "lines":
            np.maximum(self.lines, layer, out=self.lines)
        else:
            self.tone += layer * (1.0 - self.tone)

    def polyline(self, pts, value: float, thickness: float, target: str = "lines") -> None:
        arr = np.round(np.asarray(pts, np.float64) * 16).astype(np.int32).reshape(-1, 1, 2)
        t = max(1, int(round(thickness * self.k)))
        self._paint(target,
                    lambda img: cv2.polylines(img, [arr], False, 255, t, cv2.LINE_AA, shift=4), value)

    def arc(self, centre, axes, angle, a0, a1, value, thickness, target="lines") -> None:
        pts = cv2.ellipse2Poly((int(centre[0]), int(centre[1])), (int(axes[0]), int(axes[1])),
                               int(angle), int(a0), int(a1), 3)
        self.polyline(pts, value, thickness, target)

    def blob(self, centre, axes, angle, value, softness) -> None:
        tmp = np.zeros((self.h, self.w), np.uint8)
        cv2.ellipse(tmp, (int(centre[0]), int(centre[1])), (int(axes[0]), int(axes[1])),
                    angle, 0, 360, 255, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(tmp.astype(F32) / 255.0, (0, 0), softness * self.k) * value
        self.tone += layer * (1.0 - self.tone)

    def hatch(self, mask: np.ndarray, angle_deg: float, spacing: float, value: float,
              jitter: float = 0.3) -> None:
        tmp = np.zeros((self.h, self.w), np.uint8)
        a = math.radians(angle_deg)
        d = np.array([math.cos(a), -math.sin(a)])
        n = np.array([-d[1], d[0]])
        diag = math.hypot(self.w, self.h)
        steps = int(diag / (spacing * self.k))
        for i in range(-steps, steps):
            off = i * spacing * self.k + self.rng.normal(0, jitter * spacing * self.k)
            c = np.array([self.w / 2, self.h / 2]) + n * off
            length = diag * 0.6
            p = c - d * length * self.rng.uniform(0.4, 1.0)
            q = c + d * length * self.rng.uniform(0.4, 1.0)
            cv2.line(tmp, tuple(np.int32(p)), tuple(np.int32(q)), int(255 * self.rng.uniform(0.6, 1)),
                     max(1, int(round(self.k))), cv2.LINE_AA)
        layer = tmp.astype(F32) / 255.0 * mask * value
        self.tone += layer * (1.0 - self.tone)


def _bezier(p0, p1, p2, p3, n=40) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def synthetic_portrait(width: int = 900, height: int = 1350, seed: int = 3) -> Dict[str, np.ndarray]:
    """Return ``{"end", "blank", "partial"}`` uint8 RGB images of a stylised
    graphite portrait, a blank sheet and an outline-only start state."""
    sk = _Sketch(width, height, seed)
    rng = sk.rng
    W, H, k = width, height, sk.k
    cx, cy = 0.5 * W, 0.40 * H
    fw, fh = 0.20 * W, 0.17 * H  # face half-width / half-height
    outline_parts: List[Tuple] = []

    def outline(fn, *args, **kw):
        outline_parts.append((fn, args, kw))
        fn(*args, **kw)

    # Head and jaw
    outline(sk.arc, (cx, cy), (fw, fh), 0, -10, 190, 0.62, 2.6)
    # Ears
    outline(sk.arc, (cx - fw * 0.98, cy + fh * 0.05), (fw * 0.14, fh * 0.2), 0, 80, 280, 0.5, 2.0)
    outline(sk.arc, (cx + fw * 0.98, cy + fh * 0.05), (fw * 0.14, fh * 0.2), 0, -100, 100, 0.5, 2.0)
    # Neck and shoulders
    outline(sk.polyline, [(cx - fw * 0.45, cy + fh * 0.92), (cx - fw * 0.5, cy + fh * 1.55)], 0.55, 2.2)
    outline(sk.polyline, [(cx + fw * 0.45, cy + fh * 0.92), (cx + fw * 0.5, cy + fh * 1.55)], 0.55, 2.2)
    outline(sk.polyline, _bezier((cx - fw * 0.5, cy + fh * 1.55), (cx - fw * 1.4, cy + fh * 1.6),
                                 (cx - fw * 2.2, cy + fh * 1.9), (0.02 * W, 0.97 * H)), 0.55, 2.4)
    outline(sk.polyline, _bezier((cx + fw * 0.5, cy + fh * 1.55), (cx + fw * 1.4, cy + fh * 1.6),
                                 (cx + fw * 2.2, cy + fh * 1.9), (0.98 * W, 0.97 * H)), 0.55, 2.4)

    # Eyes
    for side in (-1, 1):
        ex, ey = cx + side * fw * 0.42, cy - fh * 0.08
        outline(sk.arc, (ex, ey), (fw * 0.24, fh * 0.11), 0, 195, 345, 0.9, 2.6)
        outline(sk.arc, (ex, ey - fh * 0.01), (fw * 0.22, fh * 0.07), 0, 15, 165, 0.5, 1.6)
        sk.blob((ex, ey - fh * 0.005), (fw * 0.09, fw * 0.09), 0, 0.55, 1.2)
        sk.blob((ex, ey - fh * 0.005), (fw * 0.045, fw * 0.045), 0, 0.95, 0.8)
        sk.blob((ex + side * fw * 0.03, ey - fh * 0.03), (fw * 0.018, fw * 0.018), 0, -0.6, 0.6)
        for j in range(9):  # lashes
            a = math.radians(200 + j * 17)
            bx = ex + math.cos(a) * fw * 0.24
            by = ey + math.sin(a) * fh * 0.11
            sk.polyline([(bx, by), (bx + side * 4 * k * math.cos(a + 0.3), by - 7 * k)], 0.8, 1.2)
        # Eyebrow: many short strokes along an arc
        for j in range(26):
            u = j / 25
            bx = ex - side * fw * 0.27 + side * u * fw * 0.56
            by = ey - fh * 0.26 - math.sin(u * math.pi) * fh * 0.07
            sk.polyline([(bx, by + 5 * k), (bx + side * 9 * k, by - 3 * k)], 0.75, 1.4)
        sk.hatch(_ellipse_mask(W, H, (ex, ey - fh * 0.15), (fw * 0.28, fh * 0.1), 4 * k), 35, 4.0, 0.25)

    # Nose
    outline(sk.polyline, _bezier((cx - fw * 0.08, cy - fh * 0.05), (cx - fw * 0.12, cy + fh * 0.2),
                                 (cx - fw * 0.15, cy + fh * 0.3), (cx - fw * 0.1, cy + fh * 0.42)), 0.45, 1.8)
    outline(sk.arc, (cx, cy + fh * 0.42), (fw * 0.16, fh * 0.07), 0, 10, 170, 0.6, 2.0)
    sk.blob((cx - fw * 0.09, cy + fh * 0.44), (fw * 0.05, fh * 0.025), 0, 0.85, 1.5)
    sk.blob((cx + fw * 0.09, cy + fh * 0.44), (fw * 0.05, fh * 0.025), 0, 0.85, 1.5)
    sk.blob((cx - fw * 0.13, cy + fh * 0.22), (fw * 0.06, fh * 0.2), 0, 0.3, 8)

    # Mouth
    outline(sk.polyline, _bezier((cx - fw * 0.32, cy + fh * 0.64), (cx - fw * 0.1, cy + fh * 0.6),
                                 (cx + fw * 0.1, cy + fh * 0.6), (cx + fw * 0.32, cy + fh * 0.64)), 0.85, 2.2)
    sk.blob((cx, cy + fh * 0.58), (fw * 0.26, fh * 0.05), 0, 0.45, 3)
    sk.blob((cx, cy + fh * 0.71), (fw * 0.22, fh * 0.06), 0, 0.3, 4)

    # Shading: shadow side of the face, under the jaw and on the neck
    face_mask = _ellipse_mask(W, H, (cx, cy), (fw, fh), 6 * k)
    shadow = face_mask * np.clip((np.arange(W)[None, :] - cx) / fw + 0.2, 0, 1) ** 1.2
    sk.hatch(shadow.astype(F32), 60, 5.0, 0.45)
    sk.tone += cv2.GaussianBlur(shadow.astype(F32), (0, 0), 10 * k) * 0.25 * (1 - sk.tone)
    neck = _ellipse_mask(W, H, (cx, cy + fh * 1.2), (fw * 0.5, fh * 0.32), 10 * k)
    sk.hatch(neck, 120, 4.5, 0.55)
    sk.blob((cx - fw * 0.55, cy - fh * 0.35), (fw * 0.15, fh * 0.2), 0, 0.2, 10)

    # Hair: a dark mass plus hundreds of flowing strands
    hair = np.zeros((H, W), np.uint8)
    poly = np.int32([
        (cx - fw * 1.15, cy + fh * 0.9), (cx - fw * 1.25, cy - fh * 0.3), (cx - fw * 0.9, cy - fh * 1.05),
        (cx, cy - fh * 1.35), (cx + fw * 0.9, cy - fh * 1.05), (cx + fw * 1.25, cy - fh * 0.3),
        (cx + fw * 1.15, cy + fh * 0.9), (cx + fw * 1.02, cy + fh * 0.1), (cx + fw * 0.7, cy - fh * 0.65),
        (cx, cy - fh * 0.8), (cx - fw * 0.7, cy - fh * 0.65), (cx - fw * 1.02, cy + fh * 0.1)])
    cv2.fillPoly(hair, [poly], 255, cv2.LINE_AA)
    hair_mask = cv2.GaussianBlur(hair.astype(F32) / 255.0, (0, 0), 6 * k)
    sk.tone += hair_mask * 0.5 * (1 - sk.tone)
    for _ in range(420):
        side = rng.choice([-1, 1])
        u = rng.uniform(0, 1)
        top = (cx + side * fw * rng.uniform(0, 0.6), cy - fh * (1.3 - 0.25 * u))
        end = (cx + side * fw * rng.uniform(1.0, 1.2), cy + fh * rng.uniform(0.2, 0.95))
        c1 = (top[0] + side * fw * rng.uniform(0.3, 0.7), top[1] + fh * rng.uniform(-0.05, 0.15))
        c2 = (end[0] + side * fw * rng.uniform(0.05, 0.3), end[1] - fh * rng.uniform(0.3, 0.7))
        sk.polyline(_bezier(top, c1, c2, end, 60), rng.uniform(0.45, 0.95), rng.uniform(0.8, 1.6), "tone")
    outline(sk.polyline, poly[[0, 1, 2, 3, 4, 5, 6]], 0.6, 2.4)

    # Background hatching on one side
    bg = np.zeros((H, W), F32)
    bg[:, int(cx + fw * 1.35):] = 1.0
    bg *= np.clip(1.2 - np.arange(H)[:, None] / H, 0, 1)
    sk.hatch(cv2.GaussianBlur(bg, (0, 0), 25 * k), 70, 7.0, 0.3)

    paper = np.array([0.958, 0.951, 0.932], F32)
    grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(F32), (0, 0), 1.2 * k)
    grain = 0.5 + 0.5 * grain / (grain.std() + 1e-6)
    surface = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(F32), (0, 0), 0.7)

    def compose(darkness: np.ndarray) -> np.ndarray:
        d = cv2.GaussianBlur(np.clip(darkness, 0, 1), (0, 0), 0.6 * k)
        d = np.clip(d * (0.82 + 0.36 * np.clip(grain, 0, 1)), 0, 1)
        base = paper[None, None, :] * (1.0 + 0.012 * surface[..., None])
        return np.clip(base * (1.0 - 0.9 * d[..., None]) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    lines, tone = sk.lines.copy(), sk.tone.copy()
    end = compose(1.0 - (1.0 - lines) * (1.0 - np.clip(tone, 0, 1)))
    blank = compose(np.zeros((H, W), F32))

    sk.lines[:] = 0
    sk.tone[:] = 0
    for fn, args, kw in outline_parts:
        fn(*args, **kw)
    partial = compose(sk.lines * 0.75)
    return {"end": end, "blank": blank, "partial": partial}


def _ellipse_mask(W: int, H: int, centre, axes, soft: float) -> np.ndarray:
    tmp = np.zeros((H, W), np.uint8)
    cv2.ellipse(tmp, (int(centre[0]), int(centre[1])), (int(axes[0]), int(axes[1])), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(tmp.astype(F32) / 255.0, (0, 0), max(soft, 0.5))


def write_samples(out_dir: str, width: int = 900, height: int = 1350, seed: int = 3) -> Dict[str, str]:
    """Write sample PNGs and return their paths."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, img in synthetic_portrait(width, height, seed).items():
        path = os.path.join(out_dir, f"sample_{name}.png")
        Image.fromarray(img).save(path)
        paths[name] = path
    return paths
