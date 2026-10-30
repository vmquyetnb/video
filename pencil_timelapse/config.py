"""Render settings, validation and the five-stage animation timeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Tuple

from .errors import ConfigError

RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "1080x1920": (1080, 1920),
    "720x1280": (720, 1280),
    "540x960": (540, 960),
}
X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow")
START_MODES = ("blank", "image")  # draw on empty paper, or continue from the start image
PENCIL_STYLES = ("black", "yellow", "natural")

MIN_DIM = 160
MAX_DIM = 3840
MAX_PIXELS = 2160 * 3840
MIN_FPS, MAX_FPS = 10, 60
MIN_DURATION, MAX_DURATION = 4.0, 120.0
MAX_HOLD = 10.0
MIN_DRAWING_TIME = 3.0


@dataclass
class RenderConfig:
    """All user-adjustable settings for one render."""

    width: int = 1080
    height: int = 1920
    fps: int = 30
    duration: float = 15.0
    hold: float = 0.0  # extra seconds on the exact end image after the finishing stage
    align: bool = True
    match_paper: bool = True
    paper_texture: float = 0.5
    stroke_strength: float = 0.5
    construction_lines: bool = True
    seed: int = 7
    crf: int = 18
    preset: str = "medium"
    workers: int = 0  # 0 = choose automatically
    start_mode: str = "blank"
    show_pencil: bool = True  # simulated pencil that follows the strokes
    pencil_scale: float = 1.0
    pencil_style: str = "black"

    @property
    def frame_count(self) -> int:
        return int(round(self.duration * self.fps))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> "RenderConfig":
        problems: List[str] = []
        for name in ("width", "height"):
            value = getattr(self, name)
            if not MIN_DIM <= value <= MAX_DIM:
                problems.append(f"{name} must be between {MIN_DIM} and {MAX_DIM} px (got {value})")
            elif value % 2:
                problems.append(f"{name} must be an even number for H.264 video (got {value})")
        if self.width * self.height > MAX_PIXELS:
            problems.append(
                f"resolution {self.width}x{self.height} is too large; the maximum is 2160x3840 pixels"
            )
        if not MIN_FPS <= self.fps <= MAX_FPS:
            problems.append(f"FPS must be between {MIN_FPS} and {MAX_FPS} (got {self.fps})")
        if not MIN_DURATION <= self.duration <= MAX_DURATION:
            problems.append(
                f"duration must be between {MIN_DURATION:g} and {MAX_DURATION:g} seconds "
                f"(got {self.duration:g})"
            )
        if not 0.0 <= self.hold <= MAX_HOLD:
            problems.append(f"final hold must be between 0 and {MAX_HOLD:g} seconds (got {self.hold:g})")
        elif self.duration - self.hold < MIN_DRAWING_TIME:
            problems.append(
                f"duration minus the final hold must leave at least {MIN_DRAWING_TIME:g} seconds "
                f"of drawing (got {self.duration - self.hold:g})"
            )
        for name in ("paper_texture", "stroke_strength"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                problems.append(f"{name.replace('_', ' ')} must be between 0 and 1 (got {value:g})")
        if not 0 <= self.crf <= 51:
            problems.append(f"CRF quality must be between 0 and 51 (got {self.crf})")
        if self.preset not in X264_PRESETS:
            problems.append(f"encoder preset must be one of {', '.join(X264_PRESETS)}")
        if not 0 <= self.workers <= 32:
            problems.append("workers must be between 0 (auto) and 32")
        if self.start_mode not in START_MODES:
            problems.append("start mode must be 'blank' (empty paper) or 'image' (your start image)")
        if self.pencil_style not in PENCIL_STYLES:
            problems.append(f"pencil style must be one of {', '.join(PENCIL_STYLES)}")
        if not 0.4 <= self.pencil_scale <= 2.0:
            problems.append(f"pencil size must be between 0.4 and 2 (got {self.pencil_scale:g})")
        if problems:
            raise ConfigError("Invalid settings: " + "; ".join(problems) + ".")
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RenderConfig":
        """Build a config from loosely typed values such as HTML form fields.

        ``resolution`` may be a preset key ("1080x1920") or "custom", in which
        case ``width``/``height`` are read instead.
        """
        cfg = cls()
        problems: List[str] = []

        def get(name: str) -> Any:
            value = data.get(name)
            if isinstance(value, str):
                value = value.strip()
                return value if value != "" else None
            return value

        def parse(name: str, kind: type) -> None:
            raw = get(name)
            if raw is None:
                return
            try:
                if kind is bool:
                    value: Any = raw if isinstance(raw, bool) else str(raw).lower() in (
                        "1", "true", "yes", "on")
                elif kind is int:
                    as_float = float(raw)
                    if not as_float.is_integer():
                        raise ValueError
                    value = int(as_float)
                else:
                    value = kind(raw)
            except (TypeError, ValueError):
                problems.append(f"{name.replace('_', ' ')} must be a valid {kind.__name__} (got {raw!r})")
                return
            setattr(cfg, name, value)

        resolution = get("resolution")
        if resolution and resolution != "custom":
            text = str(resolution).lower().replace("×", "x").replace(" ", "")
            if text in RESOLUTION_PRESETS:
                cfg.width, cfg.height = RESOLUTION_PRESETS[text]
            else:
                try:
                    w, h = text.split("x")
                    cfg.width, cfg.height = int(w), int(h)
                except ValueError:
                    problems.append(f"resolution must look like 1080x1920 (got {resolution!r})")
        else:
            parse("width", int)
            parse("height", int)

        for name, kind in (
            ("fps", int), ("duration", float), ("hold", float), ("align", bool),
            ("match_paper", bool), ("paper_texture", float), ("stroke_strength", float),
            ("construction_lines", bool), ("seed", int), ("crf", int), ("preset", str),
            ("workers", int), ("start_mode", str), ("show_pencil", bool), ("pencil_scale", float),
            ("pencil_style", str),
        ):
            parse(name, kind)
        if problems:
            raise ConfigError("Invalid settings: " + "; ".join(problems) + ".")
        return cfg.validate()


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


# (key, title, relative length): an artist's chronological order. With the
# default 15 s video the units are exactly one second: construction 0-3 s,
# contours 3-6 s, features 6-9 s, shading 9-12 s and finishing 12-15 s (the
# last stage ends with the pencil moving away and a short hold on the artwork).
STAGE_PLAN = (
    ("construction", "Construction strokes", 3.0),
    ("contours", "Contour strokes", 3.0),
    ("features", "Feature strokes", 3.0),
    ("shading", "Shading strokes", 3.0),
    ("finishing", "Finishing strokes", 3.0),
)


def build_timeline(duration: float, hold: float) -> List[Stage]:
    """Return the nominal stages followed by the final hold."""
    drawing_end = duration - hold
    unit = drawing_end / sum(units for _, _, units in STAGE_PLAN)
    stages: List[Stage] = []
    t = 0.0
    for i, (key, title, units) in enumerate(STAGE_PLAN):
        end = drawing_end if i == len(STAGE_PLAN) - 1 else t + units * unit
        stages.append(Stage(key, title, t, end))
        t = end
    stages.append(Stage("hold", "Final artwork (hold)", drawing_end, duration))
    return stages
