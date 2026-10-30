"""Deterministic pencil-drawing timelapse generator.

Animates a finished pencil portrait being drawn, on a blank page or from a
partly finished start image, with an optional pencil that follows the strokes.
It uses classical image processing (OpenCV/NumPy) and FFmpeg. The strokes and
the pencil are *simulated*: the software plans a plausible drawing order from
the images, it does not reconstruct the artist's real hand movements.
"""

from .config import RenderConfig, build_timeline
from .errors import ConfigError, InputError, PencilError, RenderError
from .pipeline import plan_animation, render_video

__all__ = ["RenderConfig", "build_timeline", "render_video", "plan_animation", "PencilError",
           "InputError", "ConfigError", "RenderError"]
__version__ = "1.0.0"
