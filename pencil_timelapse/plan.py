"""Build the layered reveal plan that drives every frame of the animation.

Model
-----
``S`` is the normalized start image, ``E`` the normalized end image and
``G`` a graphite-style rendition of ``E`` (neutral grey tones over the paper
colour plus subtle simulated pencil streaks). Every frame is

    frame = S + p_draw * min(G - S, 0) + p_erase * max(G - S, 0)

where ``p_draw`` and ``p_erase`` are per-pixel progress maps in [0, 1] built
from timed *layers*. Every layer is a sequence of pencil strokes (contour
lines, feature lines, hatching, cross-hatching, tonal fill): a pixel darkens
when the stroke passing over it is drawn, so graphite only appears where the
pencil tip is. Light guide marks are separate *overlays* that are erased
again, as are temporarily shaded-over highlights. Pixels the start image
already shares with the result need no graphite, so existing artwork is never
redrawn.

The stages follow an artist's chronological order (``config.STAGE_PLAN``):
construction strokes, contour strokes, feature strokes, shading strokes and
finishing strokes, ending on the exact end image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import analysis as A
from .config import STAGE_PLAN, RenderConfig, Stage, build_timeline
from .imaging import luminance

F32 = np.float32
ProgressFn = Callable[[float, str], None]

STAGE_ORDER = {key: i for i, (key, _, _) in enumerate(STAGE_PLAN)}
# Stages whose time may be handed to a neighbour when they have nothing to draw.
MERGEABLE = ("contours", "features", "shading")
WORK_STAGES = MERGEABLE + ("finishing",)
LAST_WORK_END = 0.62  # fraction of the finishing stage used for the last strokes
SETTLE_SPAN = (0.55, 0.70)  # fraction of the finishing stage used to settle on the end image
# How long a pixel takes to darken once the stroke reaches it, as a fraction of
# its pass: a fraction of a frame, so marks land right under the pencil tip.
STROKE_FEATHER = 0.004


@dataclass
class Layer:
    """One timed reveal pass over a set of pixels."""

    name: str
    stage: str
    channel: str  # "draw" or "erase"
    idx: Optional[np.ndarray]  # flat pixel indices, or None for every pixel
    tau: np.ndarray  # normalized activation time of each pixel within the window
    alpha: np.ndarray  # how far towards the target the pass takes each pixel
    feather: float  # softness of the reveal front (fraction of the window)
    t0: float = 0.0
    t1: float = 0.0
    work: float = 0.0  # graphite this layer adds (paces the schedule and the pencil)
    # Where the pencil is while this layer is revealed: positions indexed by tau.
    path_xy: Optional[np.ndarray] = None
    path_tau: Optional[np.ndarray] = None
    path_ids: Optional[np.ndarray] = None  # stroke id per point (line layers)
    path_dir: Optional[np.ndarray] = None  # stroke direction per point (area layers)
    group: Optional[str] = None  # layers of one group share a time slot and pencil path
    seq: int = 0  # order within its stage

    @property
    def size(self) -> int:
        return int(self.tau.size)


@dataclass
class Overlay:
    """Light guide marks (proportion ticks, construction lines) erased at the end."""

    idx: np.ndarray
    tau: np.ndarray
    darkness: np.ndarray
    fade_tau: np.ndarray
    stage: str = "construction"
    lead: float = 0.0  # where in its stage it starts (fraction of the stage)
    share: float = 0.55  # how much of the stage it takes
    t0: float = 0.0
    t1: float = 0.0
    fade_t0: float = 0.0
    fade_t1: float = 0.0
    feather: float = 0.03
    fade_feather: float = 0.25
    path_xy: Optional[np.ndarray] = None
    path_tau: Optional[np.ndarray] = None
    path_ids: Optional[np.ndarray] = None


@dataclass
class DrawingPlan:
    width: int
    height: int
    start: np.ndarray  # S, float32 (H, W, 3)
    graphite: np.ndarray  # G
    end: np.ndarray  # E as float32 (exactly end_u8 / 255)
    end_u8: np.ndarray
    grain: Optional[np.ndarray]  # flat graphite-grain modulation, or None
    layers: List[Layer]
    overlays: List[Overlay]
    stages: List[Stage]  # effective (possibly adapted) stage spans
    finish: Tuple[float, float]
    info: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    pencil: Optional[object] = None  # pencil_timelapse.pencil.PencilRig, attached after planning


@dataclass(frozen=True)
class _LineSpec:
    name: str
    factor: int  # analysis downsampling factor at 1080 px width
    sigma: float
    hi: float
    lo: float
    min_len: int
    radius: float  # tube radius in output pixels at 1080 px width
    w_focus: float
    w_length: float
    max_strokes: int
    follow_flow: bool  # drop strokes that cut across strongly directional areas
    face_boost: bool


LINE_SPECS = (
    _LineSpec("outlines", 4, 1.6, 0.30, 0.13, 10, 2.2, 0.35, 0.6, 140, False, False),
    _LineSpec("structure", 2, 1.3, 0.27, 0.11, 8, 1.7, 0.6, 0.15, 160, False, True),
    _LineSpec("fine", 1, 1.0, 0.22, 0.09, 6, 1.2, 0.4, 0.05, 800, True, True),
)


@dataclass
class _Landmarks:
    """Proportion landmarks an artist would mark before drawing."""

    top: float
    bottom: float
    left: float
    right: float
    cx: float
    eye_y: float
    tilt: float = 0.0  # radians, from the eye line
    face: Optional[Tuple[int, int, int, int]] = None


def build_plan(start: np.ndarray, end: np.ndarray, end_u8: np.ndarray, paper: np.ndarray,
               tooth: np.ndarray, cfg: RenderConfig,
               progress: Optional[ProgressFn] = None,
               focus_region: Optional[Tuple[int, int, int, int]] = None) -> DrawingPlan:
    """Analyse the image pair and produce a :class:`DrawingPlan`.

    ``focus_region`` (x, y, w, h) is where to look for the face, when the
    picture holds several and only one of them is being drawn.
    """

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    H, W = end.shape[:2]
    npix = H * W
    s = min(W, H) / 1080.0
    diag = math.hypot(W, H)
    rng = np.random.default_rng(cfg.seed)
    warnings: List[str] = []
    info: Dict = {}

    # ---- graphite rendition G ------------------------------------------------
    report(0.0, "Analyzing tones and stroke directions")
    lum_e = luminance(end)
    paper_y = float(np.dot(paper, [0.299, 0.587, 0.114]))
    # Not clipped at 1: keeps the end image's own paper grain in G.
    rel = np.clip(lum_e / max(paper_y, 1e-3), 0.0, 1.25)
    dark = np.clip(1.0 - rel, 0.0, 1.0)
    vx, vy, coherence = A.orientation_field(lum_e, s)

    report(0.12, "Tracing pencil stroke flow")
    lic_len = max(4, round(9 * s))
    lic_step = 1.4 * max(s, 0.5)
    noise_a = A.blur(rng.random((H, W), dtype=F32), 0.8 * s)
    noise_b = A.blur(rng.random((H, W), dtype=F32), 0.8 * s)
    dots = A.blur((rng.random((H, W)) < 0.05).astype(F32), 0.6 * s)
    dots_x = A.blur((rng.random((H, W)) < 0.05).astype(F32), 0.6 * s)
    lic_a, lic_dots, lic_b = A.line_integral_convolution(vx, vy, [noise_a, dots, noise_b],
                                                         lic_len, lic_step)
    cvx, cvy = A.rotate_field(vx, vy, 62.0)
    (lic_cross,) = A.line_integral_convolution(cvx, cvy, [dots_x], lic_len, lic_step)
    tex_a, tex_b = A.uniformize(lic_a), A.uniformize(lic_b)
    hatch_1, hatch_2 = A.uniformize(lic_dots), A.uniformize(lic_cross)
    del lic_a, lic_b, lic_dots, lic_cross, noise_a, noise_b, dots, dots_x, cvx, cvy

    # Simulated pencil streaks only where new graphite goes; artwork the start
    # image already contains is left exactly as it is.
    lum_s = luminance(start)
    need_e = np.clip((lum_s - lum_e) / np.maximum(lum_s, 0.05), 0.0, 1.0)
    amp = 0.12 * cfg.stroke_strength
    streaks = amp * (tex_a - 0.5) * 4.0 * dark * (1.0 - dark) * A.ramp(need_e, 0.02, 0.1)
    rel_g = np.clip(rel - streaks, 0.0, 1.25)
    graphite = np.clip(rel_g[..., None] * paper[None, None, :], 0.0, 1.0).astype(F32)
    del need_e, streaks

    # ---- what still has to be drawn / erased -------------------------------
    lum_g = luminance(graphite)
    need = np.clip((lum_s - lum_g) / np.maximum(lum_s, 0.05), 0.0, 1.0).astype(F32)
    need_flat = need.ravel()
    # Share of the final graphite the start already has (paper grain ignored).
    ink_g = np.clip((paper_y - lum_g) / max(paper_y, 1e-3), 0.0, 1.0)
    ink_s = np.clip((paper_y - lum_s) / max(paper_y, 1e-3), 0.0, 1.0)
    significant = (ink_g > 0.08) & (ink_s > 0.08)
    existing = float(np.minimum(ink_s, ink_g)[significant].sum()) / max(
        float(ink_g[ink_g > 0.08].sum()), 1e-6)
    info["existing_artwork_fraction"] = round(min(max(existing, 0.0), 1.0), 3)
    if existing > 0.03:
        warnings.append(
            f"The start image already contains about {existing:.0%} of the final graphite; "
            "the animation continues from it instead of starting over.")

    # ---- study the subject: face, proportions, feature zone ------------------
    report(0.3, "Studying the face and proportions")
    focus = A.detect_focus(lum_e, focus_region)
    info["focus"] = focus.to_dict()
    marks = _landmarks(focus, need, W, H)
    zone = _feature_zone(focus, marks, W, H, s)
    boost = None
    if focus.face is not None:
        fx, fy, fw, fh = focus.face
        ell = np.zeros((H, W), np.uint8)
        cv2.ellipse(ell, (int(fx + fw / 2), int(fy + fh * 0.55)), (int(fw * 0.62), int(fh * 0.72)),
                    0, 0, 360, 255, -1)
        boost = 1.0 + 0.6 * A.blur(ell.astype(F32) / 255.0, 0.05 * fw)

    # ---- line strokes at three scales ---------------------------------------
    layers: List[Layer] = []
    occupied = np.zeros(npix, bool)
    start_pos = (marks.cx, marks.top)
    stroke_sets: Dict[str, List[np.ndarray]] = {}
    for li, spec in enumerate(LINE_SPECS):
        report(0.36 + 0.1 * li, f"Extracting {spec.name} lines")
        f = max(1, round(spec.factor * s))
        sw_, sh_ = max(16, W // f), max(16, H // f)
        ink_small = A.resize_to(need, sw_, sh_)
        boost_small = A.resize_to(boost, sw_, sh_) if (boost is not None and spec.face_boost) else None
        mask = A.line_mask(ink_small, spec.sigma, spec.hi, spec.lo, boost_small)
        mask &= ~(A.resize_to(occupied.reshape(H, W).astype(F32), sw_, sh_) > 0.5)
        strokes = A.extract_strokes(A.thin(mask), spec.min_len)
        fx_, fy_ = W / sw_, H / sh_
        strokes = [A.resample_polyline(A.smooth_polyline(
            st * np.array([fx_, fy_], F32) + np.array([(fx_ - 1) / 2, (fy_ - 1) / 2], F32),
            2 if f > 1 else 1)) for st in strokes]
        if spec.follow_flow:
            strokes = [st for st in strokes if A.follows_flow(st, vx, vy, coherence)]
        # Small curly loops (texture traced as outlines) look like scribbles, not contours.
        strokes = [st for st in strokes if not _curly(st, 0.06 * min(W, H))]
        if len(strokes) > spec.max_strokes:
            strokes = sorted(strokes, key=len, reverse=True)[:spec.max_strokes]
        stroke_sets[spec.name] = strokes
        # Mark what this scale covers so finer scales add only new lines.
        for st in strokes:
            pts = np.clip(np.rint(st).astype(np.int64), 0, [W - 1, H - 1])
            occupied[pts[:, 1] * W + pts[:, 0]] = True
        grown = cv2.dilate(occupied.reshape(H, W).astype(np.uint8),
                           np.ones((3, 3), np.uint8), iterations=max(1, round(spec.radius * s)))
        occupied = grown.ravel().astype(bool)
    info["strokes"] = {k: len(v) for k, v in stroke_sets.items()}

    def line_layer(name: str, stage: str, strokes: List[np.ndarray], spec: _LineSpec,
                   gain: float, feather: float, seq: int, ordered: bool = False) -> None:
        if not strokes:
            return
        if not ordered:
            strokes = A.order_strokes(strokes, focus.points, diag, spec.w_focus, spec.w_length,
                                      start_pos)
        pts, taus, radii, starts = A.stroke_point_times_with_ids(strokes)
        idx, tau, cover = A.tube_field(pts, taus, radii, (H, W), spec.radius * max(s, 0.5))
        keep = need_flat[idx] > 0.004
        idx, tau, cover = idx[keep], tau[keep], cover[keep]
        if idx.size == 0:
            return
        # Pencil pressure varies from stroke to stroke: no uniform outlines.
        pressure = (0.82 + 0.3 * rng.random(len(strokes))).astype(F32)
        ids = np.clip(np.searchsorted(starts, tau, side="right") - 1, 0, len(strokes) - 1)
        alpha = np.clip(gain * cover * pressure[ids], 0.0, 1.0).astype(F32)
        layers.append(Layer(name, stage, "draw", idx, tau, alpha, feather,
                            work=float((alpha * need_flat[idx]).sum()), path_xy=pts, path_tau=taus,
                            path_ids=np.searchsorted(starts, taus, side="right") - 1, seq=seq))

    outline_spec, structure_spec, fine_spec = LINE_SPECS
    # 3-6 s: controlled, delicate contour lines, then the facial structure.
    line_layer("contour lines", "contours", stroke_sets["outlines"], outline_spec, 0.6, STROKE_FEATHER,
               0)
    line_layer("structure lines", "contours", stroke_sets["structure"], structure_spec, 0.72,
               STROKE_FEATHER, 1)
    # 6-9 s: eyes, nose and lips, the iris, then hair strands along their flow.
    # The shorter strands wait for the finishing stage.
    # (Shorter strokes are left to the tonal passes, which cover every pixel.)
    fine = stroke_sets["fine"]
    in_zone = [st for st in fine if _zone_share(st, zone, W, H) > 0.5][:150]
    outside = sorted((st for st in fine if _zone_share(st, zone, W, H) <= 0.5), key=len,
                     reverse=True)
    line_layer("feature lines", "features", in_zone, fine_spec, 0.95, STROKE_FEATHER, 0)
    line_layer("iris strokes", "features", _iris_strokes(focus, rng), fine_spec, 0.9,
               STROKE_FEATHER, 1, ordered=True)
    line_layer("hair strands", "features", outside[:160], fine_spec, 0.9, STROKE_FEATHER, 3)
    line_layer("fine strands", "finishing", outside[160:260], fine_spec, 1.0, STROKE_FEATHER, 0)

    # ---- work areas ---------------------------------------------------------
    report(0.7, "Planning hatching and shading strokes")
    rs = 320.0 / max(H, W)
    sw_, sh_ = max(8, round(W * rs)), max(8, round(H * rs))
    need_s = A.resize_to(need, sw_, sh_)
    tone_s = A.resize_to(A.blur(dark.astype(F32), 3.0 * s), sw_, sh_)
    rows, cols = (4, 3) if H > 1.3 * W else ((3, 4) if W > 1.3 * H else (3, 3))
    labels, k = A.segment_regions(need_s, tone_s, rows, cols)
    focus_s = (focus.centre[0] * sw_ / W, focus.centre[1] * sh_ / H)
    labels_full = cv2.medianBlur(
        cv2.resize(labels.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST), 7)
    info["regions"] = k
    region_angle = _region_angles(labels, k, A.resize_to(vx, sw_, sh_), A.resize_to(vy, sw_, sh_),
                                  need_s)
    tau_dark = cv2.resize(A.region_schedule(labels, k, need_s, tone_s, focus_s, "dark"), (W, H),
                          interpolation=cv2.INTER_LINEAR).ravel()

    dark_flat = dark.ravel()
    tex_a_flat = tex_a.ravel()
    deepest = A.ramp(dark_flat, 0.45, 0.6)
    sp = max(s, 0.5)
    # Shade continuous areas (from smoothed tone, without tiny islands); the
    # last specks of paper grain settle in the final second.
    need_soft = A.blur(need, 2.5 * sp)
    dark_soft = A.blur(dark.astype(F32), 2.5 * sp)
    min_area = int((40 * sp) ** 2)
    toned = _clean_mask(need_soft > 0.05, 3 * sp, min_area)
    shadowed = _clean_mask((dark_soft > 0.38) & toned, 3 * sp, min_area)
    filled = _clean_mask(need_soft > 0.02, 3 * sp, min_area)

    def fill_group(name_stage_group: Tuple[str, str], areas: List[Tuple[np.ndarray, float]],
                   passes: List[dict], seq: int) -> None:
        """Hatching passes drawn as strokes, area by area (all passes of an area
        before the next). Each pass: name, mask, angle offset, spacing, stroke
        length, alpha(dark, profile, pressure, idx)."""
        stage, group = name_stage_group
        strokes: List[np.ndarray] = []
        owner: List[int] = []
        for area_mask, angle in areas:
            for pi, ps in enumerate(passes):
                m = ps["mask"] & area_mask
                if m.sum() < 12:
                    continue
                st = A.hatch_strokes(m, angle + ps["angle"], ps["spacing"] * sp, rng,
                                     tuple(v * sp for v in ps["length"]))
                strokes += st
                owner += [pi] * len(st)
        if not strokes:
            return
        pts, taus, radii, starts = A.stroke_point_times_with_ids(strokes, length_exponent=1.0,
                                                                 gap_ratio=0.05, taper=0.25)
        point_owner = np.repeat(np.array(owner), [len(st) for st in strokes])
        path_ids = np.searchsorted(starts, taus, side="right") - 1
        pressure = (0.85 + 0.25 * rng.random(len(strokes))).astype(F32)
        for pi, ps in enumerate(passes):
            sel = point_owner == pi
            if not sel.any():
                continue
            radius = 0.82 * ps["spacing"] * sp  # neighbouring strokes overlap: no gaps
            idx, tau, cover, dist = A.tube_field(pts[sel], taus[sel], radii[sel], (H, W), radius,
                                                 with_distance=True)
            keep = need_flat[idx] > 0.004
            idx, tau, cover, dist = idx[keep], tau[keep], cover[keep], dist[keep]
            if idx.size == 0:
                continue
            ids = np.clip(np.searchsorted(starts, tau, side="right") - 1, 0, len(strokes) - 1)
            profile = np.clip(1.0 - (dist / radius) ** 2, 0.0, 1.0)
            alpha = np.clip(ps["alpha"](dark_flat[idx], profile, pressure[ids], idx) * cover,
                            0.0, 1.0).astype(F32)
            layers.append(Layer(ps["name"], stage, "draw", idx, tau, alpha, STROKE_FEATHER,
                                work=float((alpha * need_flat[idx]).sum()), path_xy=pts,
                                path_tau=taus, path_ids=path_ids, group=group, seq=seq))

    def areas_in(order: List[int], extra: Optional[np.ndarray] = None):
        out = []
        for r in order:
            m = labels_full == r
            if extra is not None:
                m &= extra
            if m.any():
                out.append((m, region_angle[r]))
        return out

    # 6-9 s: small hatching that gives the eyes, nostrils and lips their tone.
    zone_mask = (zone > 0.4) & (need > 0.03)
    fill_group(("features", "feature tone"), _zone_areas(zone_mask, focus, vx, vy), [dict(
        name="feature tone", mask=zone_mask, angle=0.0, spacing=6.0, length=(14.0, 36.0),
        alpha=lambda d, prof, pr, idx: 0.85 * (0.6 + 0.4 * prof) * pr)], seq=2)

    # 9-12 s: layered parallel hatching, cross-hatching in the darker areas
    # with more pressure, then a broad tonal fill - one area at a time.
    focus_order, _ = A.region_order(labels, k, need_s, tone_s, focus_s, "focus")
    fill_group(("shading", "shading"), areas_in(focus_order), [
        dict(name="parallel hatching", mask=toned, angle=0.0, spacing=20.0, length=(110.0, 240.0),
             alpha=lambda d, prof, pr, idx: (0.34 + 0.22 * d) * (0.55 + 0.45 * prof) * pr
             * (0.85 + 0.3 * tex_a_flat[idx])),
        dict(name="cross-hatching", mask=shadowed, angle=math.radians(62), spacing=20.0,
             length=(100.0, 220.0),
             alpha=lambda d, prof, pr, idx: (0.55 + 0.3 * d) * (0.55 + 0.45 * prof) * pr),
        dict(name="tonal fill", mask=filled, angle=math.radians(-28), spacing=30.0,
             length=(140.0, 280.0),
             alpha=lambda d, prof, pr, idx: 1.0 - 0.25 * deepest[idx]),
    ], seq=0)

    # 12-15 s: deepest shadows (darkest areas first), then marks the result
    # doesn't have are erased.
    dark_order, _ = A.region_order(labels, k, need_s, tone_s, focus_s, "dark")
    deepest_mask = _clean_mask(A.ramp(dark_soft, 0.45, 0.6) > 0.1, 3 * sp, min_area)
    fill_group(("finishing", "deepest"), areas_in(dark_order, deepest_mask), [dict(
        name="deepest shadows", mask=deepest_mask, angle=math.radians(40), spacing=18.0,
        length=(100.0, 220.0), alpha=lambda d, prof, pr, idx: np.ones_like(d))], seq=1)
    excess_flat = luminance(np.clip(graphite - start, 0.0, 1.0)).ravel()
    erase = A.ramp(excess_flat, 0.03, 0.08)
    eidx = np.flatnonzero(erase > 0.01)
    if eidx.size:  # small paper-grain differences are left to the final cross-fade
        et = A.equalize_times(tau_dark[eidx], erase[eidx] * excess_flat[eidx] + 1e-4)
        layers.append(Layer("erasing", "finishing", "erase", eidx.astype(np.int64), et,
                            erase[eidx].astype(F32), 0.15,
                            work=float((erase[eidx] * excess_flat[eidx]).sum()), seq=2))
    info["erase_needed"] = bool(eidx.size)

    # ---- guide marks and eraser-lifted highlights ---------------------------
    overlays: List[Overlay] = []
    fade_field = _distance_to(focus.points, W, H).ravel()
    fade_field = (fade_field / max(float(fade_field.max()), 1e-6)).astype(F32)
    if cfg.construction_lines:
        ticks = _overlay(_tick_marks(marks, W, H, s), (H, W), s, rng, fade_field, (0.18, 0.28))
        if ticks is not None:
            ticks.stage, ticks.lead, ticks.share = "construction", 0.12, 0.13
            overlays.append(ticks)
        guides = _construction_lines(marks, stroke_sets["outlines"], diag, s, rng)
        lines = _overlay(guides, (H, W), s, rng, fade_field, (0.12, 0.24))
        if lines is not None:
            lines.stage, lines.lead, lines.share = "construction", 0.25, 0.75
            overlays.append(lines)
    highlights = _highlight_cover(lum_e, dark, layers, fade_field, s)

    # ---- timeline -----------------------------------------------------------
    report(0.93, "Scheduling the timeline")
    works = _stage_work(layers, need_flat, excess_flat, npix)
    stages, finish = _adapt_stages(build_timeline(cfg.duration, cfg.hold), works, npix, warnings)
    info["stage_work"] = {k: round(v, 4) for k, v in works.items()}
    spans = {st.key: st for st in stages}
    last = spans["finishing"]
    _schedule(layers, overlays, spans, {"finishing": last.start + LAST_WORK_END * last.length})
    for layer in layers:
        if layer.t1 - layer.t0 < 1e-3:
            layer.t0, layer.t1 = finish
    for ov in overlays:  # guide marks are erased while finishing
        ov.fade_t0, ov.fade_t1 = last.start, last.start + 0.55 * last.length
    if highlights is not None:  # shaded over with the hatching, lifted with the eraser
        shading = [l for l in layers if l.group == "shading"]
        if shading:
            highlights.t0, highlights.t1 = shading[0].t0, shading[0].t1
            highlights.fade_t0 = last.start + 0.1 * last.length
            highlights.fade_t1 = last.start + 0.5 * last.length
            overlays.append(highlights)
    overlays = [ov for ov in overlays
                if ov.t1 - ov.t0 > 1e-3 and ov.fade_t1 - ov.fade_t0 > 1e-3]

    grain = None
    if cfg.stroke_strength > 0:
        kappa = 0.35 * cfg.stroke_strength
        grain = (kappa * 4.0 * (A.uniformize(tooth) - 0.5)).ravel().astype(F32)

    info["layers"] = [{"name": l.name, "stage": l.stage, "pixels": l.size,
                       "start": round(l.t0, 3), "end": round(l.t1, 3)} for l in layers]
    info["stages"] = [{"key": st.key, "title": st.title, "start": round(st.start, 3),
                       "end": round(st.end, 3)} for st in stages]
    report(1.0, "Plan ready")
    return DrawingPlan(W, H, start, graphite, end, end_u8, grain, layers, overlays, stages, finish,
                       info, warnings)


# --------------------------------------------------------------------------- landmarks

def _landmarks(focus: A.Focus, need: np.ndarray, W: int, H: int) -> _Landmarks:
    """Head extents and eye line from the detected face, or the extent of the
    drawing when no face is found."""
    if focus.face is not None:
        fx, fy, fw, fh = focus.face
        cx = fx + fw / 2
        tilt = 0.0
        if len(focus.eyes) == 2:
            (ax, ay, aw, ah), (bx, by, bw, bh) = sorted(focus.eyes)
            p, q = (ax + aw / 2, ay + ah / 2), (bx + bw / 2, by + bh / 2)
            eye_y = (p[1] + q[1]) / 2
            tilt = math.atan2(q[1] - p[1], q[0] - p[0])
        else:
            eye_y = fy + 0.42 * fh
        return _Landmarks(top=max(0.0, fy - 0.32 * fh), bottom=min(H - 1.0, fy + 1.04 * fh),
                          left=max(0.0, cx - 0.56 * fw), right=min(W - 1.0, cx + 0.56 * fw),
                          cx=cx, eye_y=eye_y, tilt=tilt, face=focus.face)
    ys, xs = np.nonzero(need > 0.08)
    if len(xs) < 50:
        return _Landmarks(0.2 * H, 0.8 * H, 0.25 * W, 0.75 * W, 0.5 * W, 0.4 * H)
    step = max(1, len(xs) // 200_000)
    xs, ys = xs[::step], ys[::step]
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    return _Landmarks(float(y0), float(y1), float(x0), float(x1), float(focus.centre[0]),
                      float(focus.centre[1]))


def _feature_zone(focus: A.Focus, marks: _Landmarks, W: int, H: int, s: float) -> np.ndarray:
    """Soft mask around the eyes, brows, nose and mouth."""
    zone = np.zeros((H, W), np.uint8)
    if marks.face is not None:
        fx, fy, fw, fh = marks.face
        eyes = [(x + w / 2, y + h / 2, w) for x, y, w, h in focus.eyes] or [
            (fx + 0.3 * fw, fy + 0.4 * fh, 0.22 * fw), (fx + 0.7 * fw, fy + 0.4 * fh, 0.22 * fw)]
        for ex, ey, ew in eyes:
            cv2.ellipse(zone, (int(ex), int(ey - 0.08 * ew)), (int(0.85 * ew), int(0.6 * ew)),
                        0, 0, 360, 255, -1)
        cv2.ellipse(zone, (int(fx + fw / 2), int(fy + 0.64 * fh)), (int(0.17 * fw), int(0.2 * fh)),
                    0, 0, 360, 255, -1)
        cv2.ellipse(zone, (int(fx + fw / 2), int(fy + 0.84 * fh)), (int(0.27 * fw), int(0.1 * fh)),
                    0, 0, 360, 255, -1)
        soft = 0.04 * fw
    else:  # without a face there are no eyes/nose/mouth to single out
        return np.zeros((H, W), F32)
    return A.blur(zone.astype(F32) / 255.0, max(1.0, soft))


def _zone_share(stroke: np.ndarray, zone: np.ndarray, W: int, H: int) -> float:
    pts = np.clip(np.rint(stroke).astype(np.int64), 0, [W - 1, H - 1])
    return float(zone[pts[:, 1], pts[:, 0]].mean())


def _distance_to(points, W: int, H: int) -> np.ndarray:
    src = np.full((H, W), 255, np.uint8)
    for px, py in points:
        src[int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))] = 0
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def _curly(stroke: np.ndarray, short: float) -> bool:
    """A short stroke that curls back on itself (its extent is small for its length)."""
    length = A.polyline_length(stroke)
    if length >= 3 * short:
        return False
    extent = float(np.hypot(*(stroke.max(axis=0) - stroke.min(axis=0))))
    return extent < 0.45 * length or (length < short and extent < 0.7 * length)


def _clean_mask(mask: np.ndarray, radius: float, min_area: int) -> np.ndarray:
    """Close small gaps, drop specks and islands smaller than ``min_area``."""
    r = max(1, int(round(radius)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _flow_angle(vx: np.ndarray, vy: np.ndarray, weight: np.ndarray) -> float:
    """Mean stroke direction (directions are +/-, so average doubled angles)."""
    w = weight.astype(np.float64) + 1e-6
    c2 = float((w * (vx * vx - vy * vy)).sum())
    s2 = float((w * 2 * vx * vy).sum())
    return 0.5 * math.atan2(s2, c2)


def _region_angles(labels: np.ndarray, k: int, vx: np.ndarray, vy: np.ndarray,
                   weight: np.ndarray) -> Dict[int, float]:
    """Hatching direction per work area: along hair flow and contours where the
    image has a clear direction, a comfortable diagonal elsewhere."""
    out = {}
    for r in range(k):
        m = labels == r
        out[r] = _flow_angle(vx[m], vy[m], weight[m]) if m.any() else math.radians(55)
    return out


def _zone_areas(zone_mask: np.ndarray, focus: A.Focus, vx: np.ndarray, vy: np.ndarray):
    """The separate parts of the feature zone (each eye, nose, mouth), nearest
    to the focal points first, each with its own hatching direction."""
    count, labels = cv2.connectedComponents(zone_mask.astype(np.uint8), connectivity=8)
    areas = []
    for c in range(1, count):
        m = labels == c
        ys, xs = np.nonzero(m)
        d = min(math.hypot(xs.mean() - px, ys.mean() - py) for px, py in focus.points)
        areas.append((d, m, _flow_angle(vx[m], vy[m], np.ones(len(xs)))))
    areas.sort(key=lambda a: a[0])
    return [(m, angle) for _, m, angle in areas]


def _iris_strokes(focus: A.Focus, rng: np.random.Generator) -> List[np.ndarray]:
    """Short radial strokes around each detected pupil."""
    strokes: List[np.ndarray] = []
    for x, y, w, h in focus.eyes:
        cx, cy = x + w / 2, y + h / 2 + 0.04 * h
        inner, outer = 0.07 * w, 0.19 * w
        for a in np.linspace(0, 2 * np.pi, 18, endpoint=False) + rng.uniform(0, 0.3):
            r0 = inner * rng.uniform(0.9, 1.2)
            r1 = outer * rng.uniform(0.8, 1.0)
            line = np.array([[cx + r0 * math.cos(a), cy + r0 * math.sin(a)],
                             [cx + r1 * math.cos(a), cy + r1 * math.sin(a)]], F32)
            strokes.append(A.resample_polyline(line))
    return strokes


def _highlight_cover(lum_e: np.ndarray, dark: np.ndarray, layers: List[Layer],
                     fade_field: np.ndarray, s: float) -> Optional[Overlay]:
    """Small highlights inside toned areas (catch-lights, hair sheen): first
    shaded over by the hatching like the area around them, then lifted out
    with the eraser while finishing."""
    hatching = next((l for l in layers if l.name == "parallel hatching"), None)
    if hatching is None:
        return None
    local = A.blur(lum_e, 5.0 * max(s, 0.5))
    lift = np.clip((lum_e - local) / 0.12, 0.0, 1.0) * A.ramp(A.blur(dark, 5.0 * max(s, 0.5)),
                                                               0.25, 0.45)
    cover = np.clip(1.0 - local / np.maximum(lum_e, 1e-3), 0.0, 0.7) * lift
    idx = np.flatnonzero(cover.ravel() > 0.03)
    if idx.size == 0:
        return None
    tau = np.full(lum_e.size, 0.5, F32)
    tau[hatching.idx] = hatching.tau
    return Overlay(idx.astype(np.int64), tau[idx], cover.ravel()[idx].astype(F32), fade_field[idx],
                   stage="shading", feather=STROKE_FEATHER, fade_feather=0.3)


# --------------------------------------------------------------------------- guide marks

def _tick_marks(marks: _Landmarks, W: int, H: int, s: float) -> List[np.ndarray]:
    """Short dashes at the top, bottom and sides of the head and on the eye line."""
    half = 18.0 * s
    c, sn = math.cos(marks.tilt), math.sin(marks.tilt)

    def dash(x: float, y: float, horizontal: bool) -> np.ndarray:
        d = np.array([c, sn]) if horizontal else np.array([-sn, c])
        return np.stack([(x, y) - d * half, (x, y) + d * half]).astype(F32)

    return [dash(marks.cx, marks.top, True), dash(marks.cx, marks.bottom, True),
            dash(marks.left, marks.eye_y, False), dash(marks.right, marks.eye_y, False),
            dash(marks.cx, marks.eye_y, True)]


def _construction_lines(marks: _Landmarks, outlines: List[np.ndarray], diag: float, s: float,
                        rng: np.random.Generator) -> List[np.ndarray]:
    """Loose proportion lines: head oval, centre axis, eye/nose/mouth lines and
    neck when a face was found, plus simplified straight guides along the main
    outlines (hair mass, shoulders)."""
    guides: List[np.ndarray] = []
    c, sn = math.cos(marks.tilt), math.sin(marks.tilt)
    across, down = np.array([c, sn]), np.array([-sn, c])
    if marks.face is not None:
        fx, fy, fw, fh = marks.face
        cy = (marks.top + marks.bottom) / 2
        ax, ay = 0.56 * fw, (marks.bottom - marks.top) / 2
        t = np.linspace(0, 2 * np.pi, 120)
        oval = np.stack([marks.cx + ax * np.cos(t) * c - ay * np.sin(t) * sn,
                         cy + ax * np.cos(t) * sn + ay * np.sin(t) * c], 1)
        guides += [oval[:68], oval[60:]]  # two overlapping arcs, like a quick hand-drawn oval
        centre = np.array([marks.cx, marks.eye_y])
        guides.append(np.stack([centre - down * (marks.eye_y - marks.top + 0.1 * fh),
                                centre + down * (marks.bottom - marks.eye_y + 0.12 * fh)]))
        for y, width in ((marks.eye_y, 0.62 * fw), (fy + 0.68 * fh, 0.2 * fw),
                         (fy + 0.84 * fh, 0.28 * fw)):
            p = centre + down * (y - marks.eye_y)
            guides.append(np.stack([p - across * width, p + across * width]))
        for side in (-1, 1):  # jaw: from below the ear to the chin
            ear = centre + across * side * 0.52 * fw + down * 0.3 * (marks.bottom - marks.eye_y)
            guides.append(np.stack([ear, np.array([marks.cx, marks.bottom])]))
        for side in (-1, 1):  # neck
            top = np.array([marks.cx + side * 0.27 * fw, marks.bottom - 0.08 * fh])
            guides.append(np.stack([top, top + np.array([side * 0.03 * fw, 0.5 * fh])]))
    else:
        top, bottom = np.array([marks.cx, marks.top]), np.array([marks.cx, marks.bottom])
        guides.append(np.stack([top, bottom]))
        width = 0.35 * (marks.right - marks.left)
        p = np.array([marks.cx, marks.eye_y])
        guides.append(np.stack([p - across * width, p + across * width]))
    for st in outlines:  # straight block-in segments along the big shapes
        if A.polyline_length(st) < 0.03 * diag:
            continue
        approx = cv2.approxPolyDP(st.reshape(-1, 1, 2), 0.012 * diag, False)[:, 0, :].astype(F32)
        for p, q in zip(approx[:-1], approx[1:]):
            if float(np.hypot(*(q - p))) >= 0.01 * diag:
                guides.append(np.stack([p, q]))
    # Exploratory strokes: most guides are gone over again, slightly off the first line.
    explored: List[np.ndarray] = []
    for g in guides:
        g = np.asarray(g, np.float64)
        explored.append(g)
        if len(g) >= 2 and rng.random() < 0.7:
            d = g[-1] - g[0]
            normal = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
            a, b = sorted(rng.uniform(0.0, 1.0, 2))
            a, b = 0.3 * a, 0.7 + 0.3 * b  # a partial second pass
            i0, i1 = int(a * (len(g) - 1)), max(int(b * (len(g) - 1)), int(a * (len(g) - 1)) + 1)
            part = g[i0:i1 + 1] if len(g) > 2 else np.stack([g[0] + d * a, g[0] + d * b])
            explored.append(part + normal * rng.normal(0.0, 2.5 * s))
    return explored


def _overlay(polylines: List[np.ndarray], shape: Tuple[int, int], s: float,
             rng: np.random.Generator, fade_field: np.ndarray,
             pressure_range: Tuple[float, float]) -> Optional[Overlay]:
    """Light, slightly wobbly, overshooting pencil marks along ``polylines``."""
    H, W = shape
    strokes: List[np.ndarray] = []
    for line in polylines:
        line = np.asarray(line, np.float64)
        if len(line) < 2:
            continue
        d0, d1 = line[1] - line[0], line[-1] - line[-2]
        n0, n1 = np.hypot(*d0), np.hypot(*d1)
        if n0 < 1e-6 or n1 < 1e-6:
            continue
        length = A.polyline_length(line.astype(F32))
        ext = min(0.07 * length, 25.0 * s)  # quick strokes overshoot their ends a little
        line = np.vstack([line[:1] - d0 / n0 * ext, line, line[-1:] + d1 / n1 * ext])
        line = A.resample_polyline(line.astype(F32), 1.0).astype(np.float64)
        if len(line) < 2:
            continue
        tangent = np.gradient(line, axis=0)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
        normal /= np.hypot(normal[:, 0], normal[:, 1])[:, None] + 1e-9
        wobble = A.smooth_noise(rng, len(line), max(8.0, len(line) / 6.0)) * 1.2 * s
        strokes.append((line + normal * wobble[:, None]).astype(F32))
    if not strokes:
        return None
    pts, taus, _, starts = A.stroke_point_times_with_ids(strokes, gap_ratio=0.3, taper=0.0)
    idx, tau, cover = A.tube_field(pts, taus, np.ones_like(taus), (H, W), 0.6 * max(s, 0.6))
    if idx.size == 0:
        return None
    lo, hi = pressure_range
    pressure = (lo + (hi - lo) * rng.random(len(strokes))).astype(F32)
    # Strokes occupy consecutive time slots, so a pixel's time identifies its stroke.
    ids = np.clip(np.searchsorted(starts, tau, side="right") - 1, 0, len(strokes) - 1)
    return Overlay(idx, tau, (cover * pressure[ids]).astype(F32), fade_field[idx].astype(F32),
                   path_xy=pts, path_tau=taus,
                   path_ids=np.searchsorted(starts, taus, side="right") - 1)


# --------------------------------------------------------------------------- scheduling

def _schedule(layers: List[Layer], overlays: List[Overlay], spans: Dict[str, Stage],
              limits: Dict[str, float]) -> None:
    """Run the passes of each stage one after another.

    A pencil can only be in one place, so new graphite only appears where its
    tip is. Guide-mark overlays take a fixed part of their stage; layers share
    the rest in proportion to the graphite they add (line passes get at least
    a minimum share). Layers of one group (for example interleaved tonal
    passes) share one slot. Only stroke-less passes that add almost nothing
    (such as erasing a few specks) run in the background over the whole stage.
    """
    buckets: Dict[Tuple[float, float], Tuple[List[Overlay], List[Layer]]] = {}

    def bucket(stage: str):
        span = spans[stage]
        return buckets.setdefault((span.start, limits.get(stage, span.end)), ([], []))

    for ov in overlays:
        bucket(ov.stage)[0].append(ov)
    for layer in sorted(layers, key=lambda l: (STAGE_ORDER[l.stage], l.seq)):
        bucket(layer.stage)[1].append(layer)
    for (start, end), (ovs, members) in buckets.items():
        length = end - start
        if length < 1e-3:
            continue
        t = start
        for ov in ovs:
            ov.t0 = start + ov.lead * length
            ov.t1 = ov.t0 + ov.share * length
            t = max(t, ov.t1)
        units: List[List[Layer]] = []
        for layer in members:
            same = [u for u in units if layer.group and u[0].group == layer.group]
            if same:
                same[0].append(layer)
            else:
                units.append([layer])
        total = sum(l.work for l in members) or 1.0
        slots: List[Tuple[List[Layer], float]] = []
        for unit in units:
            work = sum(l.work for l in unit)
            has_strokes = unit[0].channel == "draw" and unit[0].path_xy is not None
            if has_strokes or work >= 0.002 * total:
                # Line passes add little graphite but are what the eye follows:
                # give them enough time to be drawn at a believable pace.
                floor = 0.3 if unit[0].path_dir is None else 0.12
                slots.append((unit, max(work / total, floor)))
            else:  # adds almost nothing and has no strokes: let it run over the whole stage
                for l in unit:
                    l.t0, l.t1 = t, end
        norm = sum(w for _, w in slots) or 1.0
        available = end - t
        for unit, weight in slots:
            dt = available * weight / norm
            for l in unit:
                l.t0, l.t1 = t, t + dt
            t += dt


def _front_path(idx: np.ndarray, tau: np.ndarray, weight: np.ndarray, width: int,
                vx: np.ndarray, vy: np.ndarray, bins: int = 240
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre of the reveal front over time for an area layer, plus the local
    stroke direction there (averaged as doubled angles: directions are +/-)."""
    b = np.minimum((tau * bins).astype(np.int64), bins - 1)
    w = weight.astype(np.float64) + 1e-6
    total = np.bincount(b, w, bins)
    xs, ys = (idx % width).astype(np.float64), (idx // width).astype(np.float64)
    cx = np.bincount(b, w * xs, bins)
    cy = np.bincount(b, w * ys, bins)
    fx, fy = vx[idx].astype(np.float64), vy[idx].astype(np.float64)
    c2 = np.bincount(b, w * (fx * fx - fy * fy), bins)
    s2 = np.bincount(b, w * 2 * fx * fy, bins)
    ok = total > 1e-6
    centres = np.arange(bins)
    out_tau = (centres + 0.5) / bins
    x = np.interp(centres, centres[ok], cx[ok] / total[ok])
    y = np.interp(centres, centres[ok], cy[ok] / total[ok])
    ang = 0.5 * np.arctan2(np.interp(centres, centres[ok], s2[ok]),
                           np.interp(centres, centres[ok], c2[ok]))
    return (out_tau.astype(F32), np.stack([x, y], 1).astype(F32),
            np.stack([np.cos(ang), np.sin(ang)], 1).astype(F32))


def _stage_work(layers: List[Layer], need: np.ndarray, excess: np.ndarray, npix: int
                ) -> Dict[str, float]:
    """Graphite (and erasing) each stage adds beyond earlier stages."""
    covered = np.zeros(npix, F32)
    works: Dict[str, float] = {}
    for key in WORK_STAGES:
        stage_max = np.zeros(npix, F32)
        erase_work = 0.0
        for layer in layers:
            if layer.stage != key:
                continue
            if layer.channel == "erase":
                weights = excess if layer.idx is None else excess[layer.idx]
                erase_work += float((layer.alpha * weights).sum())
                continue
            if layer.idx is None:
                np.maximum(stage_max, layer.alpha, out=stage_max)
            else:
                stage_max[layer.idx] = np.maximum(stage_max[layer.idx], layer.alpha)
        gain = np.maximum(stage_max - covered, 0.0)
        works[key] = float((gain * need).sum()) + erase_work
        np.maximum(covered, stage_max, out=covered)
    works["remaining"] = float(((1.0 - covered) * need).sum())
    return works


def _adapt_stages(nominal: List[Stage], works: Dict[str, float], npix: int,
                  warnings: List[str]) -> Tuple[List[Stage], Tuple[float, float]]:
    """Give the time of stages with (almost) nothing to draw to the next stage.

    Construction and finishing keep their time. Returns the stages and the
    window in which the frame settles on the exact end image.
    """
    by_key = {st.key: st for st in nominal}
    last = by_key["finishing"]
    finish = (last.start + SETTLE_SPAN[0] * last.length,
              last.start + SETTLE_SPAN[1] * last.length)
    if sum(works[k] for k in WORK_STAGES) < 5e-4 * npix:
        warnings.append("The start and end images are nearly identical, so there is very little "
                        "to animate; the video is mostly a gentle cross-fade.")
        drawing_end = last.end
        stages = [Stage(st.key, st.title, 0.0, drawing_end) if st.key != "hold" else st
                  for st in nominal]
        return stages, (0.0, finish[1])
    total = sum(works[k] for k in MERGEABLE)
    empty = [works[k] < 0.02 * total for k in MERGEABLE]
    spans = {k: [by_key[k].start, by_key[k].end] for k in MERGEABLE}
    merged_into: Dict[str, str] = {}
    for i, key in enumerate(MERGEABLE):
        if not empty[i]:
            continue
        later = [j for j in range(i + 1, len(MERGEABLE)) if not empty[j]]
        earlier = [j for j in range(i) if not empty[j]]
        if later:
            target = MERGEABLE[later[0]]
            spans[target][0] = min(spans[target][0], spans[key][0])
        elif earlier:
            target = MERGEABLE[earlier[-1]]
            spans[target][1] = max(spans[target][1], spans[key][1])
        else:
            continue
        merged_into[key] = target
    for key, target in merged_into.items():
        spans[key] = list(spans[target])
    if merged_into:
        titles = ", ".join(by_key[k].title.lower() for k in merged_into)
        warnings.append(f"Little or nothing is left to draw for: {titles}. That time was given "
                        "to the neighbouring stage.")
    stages = [Stage(k, by_key[k].title, spans[k][0], spans[k][1]) if k in spans else by_key[k]
              for k in (st.key for st in nominal)]
    return stages, finish
