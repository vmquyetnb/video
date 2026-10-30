"""Per-frame compositing of a :class:`~pencil_timelapse.plan.DrawingPlan`."""

from __future__ import annotations

import os
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .plan import DrawingPlan, Layer

F32 = np.float32


def _reveal(tau: np.ndarray, t: float, t0: float, t1: float, feather: float) -> np.ndarray:
    """Smooth per-pixel progress of a reveal front sweeping through ``tau``.

    At ``t0`` nothing is revealed; at ``t1`` every pixel (tau <= 1) is fully
    revealed. Each pixel eases in over ``feather`` of the window.
    """
    s = (t - t0) / (t1 - t0) * (1.0 + feather)
    r = np.subtract(s, tau, dtype=F32)
    r *= 1.0 / feather
    np.clip(r, 0.0, 1.0, out=r)
    r2 = r * r
    r *= -2.0
    r += 3.0
    r *= r2
    return r


def _ease(t: float, t0: float, t1: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3.0 - 2.0 * x)


class FrameRenderer:
    """Renders frames at arbitrary times. Thread-safe; frames are independent."""

    def __init__(self, plan: DrawingPlan):
        self.plan = plan
        H, W = plan.height, plan.width
        self.shape = (H, W)
        self.npix = H * W
        diff = plan.graphite - plan.start
        self.start = plan.start
        self.draw_delta = np.minimum(diff, 0.0)
        self.erase_delta = np.maximum(diff, 0.0)
        self.end = plan.end
        self.end_u8 = plan.end_u8
        self.finish_start, self.finish_end = plan.finish
        self.draw_layers = sorted((l for l in plan.layers if l.channel == "draw"), key=lambda l: l.t1)
        self.erase_layers = sorted((l for l in plan.layers if l.channel == "erase"), key=lambda l: l.t1)
        self._base_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ progress maps

    def _completed_base(self, channel: str, layers: List[Layer], count: int) -> np.ndarray:
        """Max-combined alpha of the first ``count`` (finished) layers, cached."""
        key = (channel, count)
        with self._lock:
            cached = self._base_cache.get(key)
        if cached is not None:
            return cached
        base = np.zeros(self.npix, F32)
        for layer in layers[:count]:
            if layer.idx is None:
                np.maximum(base, layer.alpha, out=base)
            else:
                base[layer.idx] = np.maximum(base[layer.idx], layer.alpha)
        with self._lock:
            # Keep the cache small: frames are rendered roughly in time order.
            stale = [k for k in self._base_cache if k[0] == channel and k[1] < count - 1]
            for k in stale:
                del self._base_cache[k]
            self._base_cache[key] = base
        return base

    def _progress(self, channel: str, layers: List[Layer], t: float) -> np.ndarray:
        done = 0
        while done < len(layers) and layers[done].t1 <= t:
            done += 1
        p = self._completed_base(channel, layers, done).copy()
        for layer in layers[done:]:
            if t <= layer.t0:
                continue
            contrib = _reveal(layer.tau, t, layer.t0, layer.t1, layer.feather)
            contrib *= layer.alpha
            if layer.idx is None:
                np.maximum(p, contrib, out=p)
            else:
                p[layer.idx] = np.maximum(p[layer.idx], contrib)
        return p

    # ------------------------------------------------------------------ frames

    def render(self, t: float) -> np.ndarray:
        """Return the frame at time ``t`` (seconds) as uint8 RGB."""
        if t >= self.finish_end:
            return self.end_u8
        H, W = self.shape
        e = _ease(t, self.finish_start, self.finish_end)

        pd = self._progress("draw", self.draw_layers, t)
        if e > 0.0:
            pd *= 1.0 - e
            pd += e
        if self.plan.grain is not None:
            g = pd * (1.0 - pd)
            g *= self.plan.grain
            pd += g
            np.clip(pd, 0.0, 1.0, out=pd)
        frame = self.draw_delta * pd.reshape(H, W, 1)
        frame += self.start

        if self.erase_layers or e > 0.0:
            pe = (self._progress("erase", self.erase_layers, t) if self.erase_layers
                  else np.zeros(self.npix, F32))
            if e > 0.0:
                pe *= 1.0 - e
                pe += e
            frame += self.erase_delta * pe.reshape(H, W, 1)

        for ov in self.plan.overlays:  # guide marks / covered highlights, erased while finishing
            if t <= ov.t0 or t >= ov.fade_t1 or e >= 1.0:
                continue
            grow = _reveal(ov.tau, t, ov.t0, ov.t1, ov.feather)
            if t > ov.fade_t0:
                grow *= 1.0 - _reveal(ov.fade_tau, t, ov.fade_t0, ov.fade_t1, ov.fade_feather)
            grow *= ov.darkness
            grow *= 1.0 - e
            flat = frame.reshape(-1, 3)
            flat[ov.idx] *= (1.0 - grow)[:, None]

        if e > 0.0:
            frame += (self.end - frame) * e
        if self.plan.pencil is not None:
            self.plan.pencil.draw(frame, t)
        return cv2.convertScaleAbs(frame, alpha=255.0)

    def stage_at(self, t: float) -> str:
        for st in self.plan.stages:
            if st.key == "hold":
                continue
            if st.start <= t < st.end:
                return st.title
        return "Final artwork (hold)" if t >= self.finish_end else self.plan.stages[-1].title

    def iter_frames(self, fps: float, count: int, workers: int = 0
                    ) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield ``(index, time, frame)`` in order, rendering ahead on a thread pool.

        Frames in the final hold are the exact end image and are not recomputed.
        """
        if workers <= 0:
            workers = max(1, min(4, (os.cpu_count() or 2) // 2))
        times = [i / fps for i in range(count)]
        if workers == 1:
            for i, t in enumerate(times):
                yield i, t, self.render(t)
            return
        pending: Deque[Tuple[int, float, Optional[Future]]] = deque()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="frame") as pool:
            for i, t in enumerate(times):
                fut = pool.submit(self.render, t) if t < self.finish_end else None
                pending.append((i, t, fut))
                while len(pending) > 2 * workers:
                    j, tj, f = pending.popleft()
                    yield j, tj, (f.result() if f is not None else self.end_u8)
            while pending:
                j, tj, f = pending.popleft()
                yield j, tj, (f.result() if f is not None else self.end_u8)
