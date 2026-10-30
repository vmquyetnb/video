"""High-level API: prepare an image pair, plan the animation, render the MP4."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import RenderConfig
from .encoder import FFmpegWriter
from .errors import RenderError
from .errors import InputError
from .imaging import (ImageSource, Placement, colourfulness, estimate_alignment, estimate_paper_color,
                      fit_to_canvas, high_frequency_std, ink_fraction, load_image, paper_canvas,
                      paper_texture, to_uint8, warp_to_canvas)
from .pencil import build_rig, make_pencil
from .plan import DrawingPlan, build_plan
from .renderer import FrameRenderer

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]

# Share of the progress bar used by each phase.
_PREPARE, _PLAN = 0.04, 0.16


@dataclass
class PreparedPair:
    """Start and end images normalized onto the same output canvas."""

    start: np.ndarray
    end: np.ndarray
    end_u8: np.ndarray
    paper: np.ndarray
    tooth: np.ndarray
    info: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RenderResult:
    output_path: str
    width: int
    height: int
    fps: int
    frames: int
    duration: float
    elapsed: float
    encoder: str
    info: Dict
    warnings: List[str]

    def to_dict(self) -> Dict:
        return {"output_path": self.output_path, "width": self.width, "height": self.height,
                "fps": self.fps, "frames": self.frames, "duration": self.duration,
                "elapsed": round(self.elapsed, 2), "encoder": self.encoder, "info": self.info,
                "warnings": self.warnings}


def prepare_pair(start_rgb: Optional[np.ndarray], end_rgb: np.ndarray,
                 cfg: RenderConfig) -> PreparedPair:
    """Fit both images onto the canvas, align the start to the end and match paper tone.

    With ``start_rgb=None`` the drawing starts on a blank sheet of the same
    paper as the end image.
    """
    W, H = cfg.width, cfg.height
    info: Dict = {}
    warnings: List[str] = []
    scale = min(W, H) / 1080.0
    surface, tooth = paper_texture(H, W, cfg.seed, scale)
    amplitude = 0.02 * cfg.paper_texture

    paper_end = estimate_paper_color(end_rgb)
    end_canvas, end_place = fit_to_canvas(end_rgb, paper_canvas(H, W, paper_end, surface, amplitude))
    info["end_placement"] = end_place.to_dict()
    info["paper_color"] = [round(float(c), 3) for c in paper_end]
    end_u8 = to_uint8(end_canvas)
    end_exact = end_u8.astype(np.float32) / 255.0

    if start_rgb is None:
        info.update(start_mode="blank paper", start_is_blank=True,
                    alignment="not needed (blank paper)")
        start_canvas = paper_canvas(H, W, paper_end, surface, amplitude)
        return PreparedPair(start_canvas, end_exact, end_u8, paper_end, tooth, info, warnings)

    info["start_mode"] = "start image"
    if colourfulness(start_rgb) > 0.10 or ink_fraction(start_rgb, estimate_paper_color(start_rgb)) > 0.35:
        warnings.append(
            "The start image looks like a photograph rather than an earlier stage of the drawing, "
            "so the video morphs that photo into the drawing. Choose 'Blank white paper' as the "
            "start to watch the portrait being drawn on an empty page.")
    paper_start = estimate_paper_color(start_rgb)
    start_bg = paper_canvas(H, W, paper_start, surface, amplitude)
    start_canvas, start_place = fit_to_canvas(start_rgb, start_bg)
    info["start_placement"] = start_place.to_dict()
    _aspect_warning(start_place, end_place, warnings)

    if cfg.align:
        alignment = estimate_alignment(start_canvas, end_canvas, cfg.seed)
        info["alignment"] = alignment.message
        if alignment.applied and alignment.matrix is not None:
            start_canvas = warp_to_canvas(start_canvas, alignment.matrix, start_bg)
    else:
        info["alignment"] = "disabled"

    if cfg.match_paper:
        gain = np.clip(paper_end / np.maximum(paper_start, 1e-3), 0.6, 1.6)
        if np.max(np.abs(gain - 1.0)) > 0.01:
            start_canvas = np.clip(start_canvas * gain[None, None, :], 0.0, 1.0)
            info["paper_match_gain"] = [round(float(g), 3) for g in gain]

    # A perfectly flat digital start (e.g. pure white) gets a natural paper surface.
    flat_start = high_frequency_std(start_rgb) < 0.004 and ink_fraction(start_rgb, paper_start) < 0.002
    if flat_start and amplitude > 0:  # the padding around it is already textured
        p = start_place
        region = (slice(p.y, p.y + p.height), slice(p.x, p.x + p.width))
        shade = 1.0 + amplitude * surface[region]
        start_canvas[region] = np.clip(start_canvas[region] * shade[..., None], 0.0, 1.0)
    info["start_is_blank"] = bool(ink_fraction(start_rgb, paper_start) < 0.002)
    return PreparedPair(np.ascontiguousarray(start_canvas, dtype=np.float32), end_exact, end_u8,
                        paper_end, tooth, info, warnings)


def _aspect_warning(start: Placement, end: Placement, warnings: List[str]) -> None:
    ra = start.source_width / start.source_height
    rb = end.source_width / end.source_height
    if abs(ra - rb) / rb > 0.03:
        warnings.append(
            f"The start ({start.source_width}x{start.source_height}) and end "
            f"({end.source_width}x{end.source_height}) images have different aspect ratios; each "
            "was fitted to the canvas separately, so the drawing may shift slightly.")


def plan_animation(start: Optional[ImageSource], end: ImageSource,
                   cfg: Optional[RenderConfig] = None,
                   progress: Optional[ProgressFn] = None,
                   focus_region: Optional[Tuple[int, int, int, int]] = None) -> DrawingPlan:
    """Load, prepare and plan without encoding (useful for previews and tests).

    ``start`` is ignored (and may be ``None``) when ``cfg.start_mode`` is ``"blank"``.
    ``focus_region`` limits the face search to part of the output frame.
    """
    cfg = (cfg or RenderConfig()).validate()
    report = progress or (lambda f, m: None)
    report(0.0, "Loading images")
    if cfg.start_mode == "image":
        if start is None:
            raise InputError("Start image is missing. Upload one, or choose 'Blank white paper' "
                             "as the start.")
        start_rgb: Optional[np.ndarray] = load_image(start, "Start image")
    else:
        start_rgb = None
    end_rgb = load_image(end, "End image")
    report(0.01, "Normalizing and aligning images")
    pair = prepare_pair(start_rgb, end_rgb, cfg)
    report(_PREPARE, "Analyzing the drawing")
    plan = build_plan(pair.start, pair.end, pair.end_u8, pair.paper, pair.tooth, cfg,
                      progress=lambda f, m: report(_PREPARE + (_PLAN - _PREPARE) * f, m),
                      focus_region=focus_region)
    plan.info.update(pair.info)
    plan.warnings[:0] = pair.warnings
    if cfg.show_pencil:
        k = min(cfg.width, cfg.height) / 1080.0 * cfg.pencil_scale
        plan.pencil = build_rig(plan, make_pencil(k, cfg.pencil_style), cfg.fps)
    return plan


def render_video(start: Optional[ImageSource], end: ImageSource, output_path: str,
                 config: Optional[RenderConfig] = None,
                 progress: Optional[ProgressFn] = None) -> RenderResult:
    """Render the pencil-drawing timelapse to ``output_path`` (MP4).

    ``progress(fraction, message)`` is called regularly with fraction in [0, 1].
    The output file only appears once encoding has fully succeeded.
    """
    cfg = (config or RenderConfig()).validate()
    report = progress or (lambda f, m: None)
    started = time.perf_counter()
    plan = plan_animation(start, end, cfg, report)
    renderer = FrameRenderer(plan)

    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise RenderError(f"Cannot create the output folder '{out_dir}': {exc}") from None
    tmp_dir = tempfile.mkdtemp(prefix=".pencil-render-", dir=out_dir)
    tmp_path = os.path.join(tmp_dir, "video.mp4")
    count = cfg.frame_count
    try:
        with FFmpegWriter(tmp_path, cfg.width, cfg.height, cfg.fps, cfg.crf, cfg.preset,
                          total_frames=count, hold_start=renderer.finish_end) as writer:
            last_report = 0.0
            for i, t, frame in renderer.iter_frames(cfg.fps, count, cfg.workers):
                writer.write(frame)
                now = time.perf_counter()
                if now - last_report > 0.1 or i == count - 1:
                    last_report = now
                    report(_PLAN + (0.985 - _PLAN) * (i + 1) / count,
                           f"{renderer.stage_at(t)} - frame {i + 1}/{count}")
            encoder = writer.codec
            report(0.99, "Finalizing MP4")
        os.replace(tmp_path, output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    elapsed = time.perf_counter() - started
    report(1.0, "Done")
    log.info("Rendered %s (%d frames) in %.1fs", output_path, count, elapsed)
    info = dict(plan.info)
    info["config"] = cfg.to_dict()
    return RenderResult(output_path, cfg.width, cfg.height, cfg.fps, count, count / cfg.fps,
                        elapsed, encoder, info, list(plan.warnings))
