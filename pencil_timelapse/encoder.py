"""FFmpeg discovery and a streaming raw-frame -> MP4 (H.264) writer."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from .errors import RenderError

INSTALL_HINT = (
    "Install it with your package manager (macOS: 'brew install ffmpeg', Ubuntu/Debian: "
    "'sudo apt install ffmpeg', Windows: 'winget install ffmpeg') or run "
    "'pip install imageio-ffmpeg', which ships a bundled binary. You can also point the "
    "PENCIL_FFMPEG environment variable at an ffmpeg executable."
)


@functools.lru_cache(maxsize=None)
def find_ffmpeg() -> str:
    """Locate an ffmpeg executable: $PENCIL_FFMPEG, then PATH, then imageio-ffmpeg."""
    env = os.environ.get("PENCIL_FFMPEG")
    if env:
        if os.path.isfile(env) and os.access(env, os.X_OK):
            return env
        raise RenderError(f"PENCIL_FFMPEG points to '{env}', which is not an executable file.")
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    raise RenderError("FFmpeg was not found. " + INSTALL_HINT)


@functools.lru_cache(maxsize=None)
def available_encoders(ffmpeg: str) -> frozenset:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderError(f"Could not run FFmpeg at '{ffmpeg}': {exc}") from None
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] == "V":
            names.add(parts[1])
    return frozenset(names)


def ffmpeg_status() -> dict:
    """Non-throwing summary used by the web UI."""
    try:
        exe = find_ffmpeg()
        encoders = available_encoders(exe)
        codec = next((c for c in ("libx264", "h264_videotoolbox", "libopenh264", "mpeg4")
                      if c in encoders), None)
        return {"ok": codec is not None, "path": exe, "encoder": codec,
                "message": "" if codec else "FFmpeg has no usable video encoder."}
    except RenderError as exc:
        return {"ok": False, "path": None, "encoder": None, "message": str(exc)}


class FFmpegWriter:
    """Pipe RGB24 frames into FFmpeg and produce a web-friendly MP4."""

    def __init__(self, path: str, width: int, height: int, fps: float, crf: int = 18,
                 preset: str = "medium", ffmpeg: Optional[str] = None,
                 total_frames: Optional[int] = None, hold_start: Optional[float] = None):
        self.path = path
        self.width, self.height = width, height
        self.frames = 0
        exe = ffmpeg or find_ffmpeg()
        encoders = available_encoders(exe)
        cmd: List[str] = [
            exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-framerate", f"{fps:g}", "-i", "pipe:0", "-an",
        ]
        # Keyframes re-encode the paper grain and make it "pulse", so use few of
        # them and never let one fall inside the static final hold.
        gop = max(1, int(round(fps * 10)))
        keyframes: List[str] = []
        if total_frames and total_frames <= fps * 30:
            gop = total_frames
        elif hold_start is not None:
            keyframes = ["-force_key_frames", f"{hold_start:.4f}"]
        if "libx264" in encoders:
            self.codec = "libx264"
            cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                    "-profile:v", "high", "-g", str(gop)] + keyframes
        elif "h264_videotoolbox" in encoders:
            self.codec = "h264_videotoolbox"
            cmd += ["-c:v", "h264_videotoolbox", "-b:v", _bitrate(width, height, fps)]
        elif "libopenh264" in encoders:
            self.codec = "libopenh264"
            cmd += ["-c:v", "libopenh264", "-b:v", _bitrate(width, height, fps)]
        elif "mpeg4" in encoders:
            self.codec = "mpeg4"
            cmd += ["-c:v", "mpeg4", "-q:v", "2"]
        else:
            raise RenderError("FFmpeg has no H.264 or MPEG-4 encoder available. " + INSTALL_HINT)
        cmd += [
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-color_range", "tv", "-movflags", "+faststart", path,
        ]
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                          stderr=self._stderr)
        except OSError as exc:
            self._stderr.close()
            raise RenderError(f"Could not start FFmpeg: {exc}") from None

    def _error_text(self) -> str:
        try:
            self._stderr.seek(0)
            text = self._stderr.read().decode("utf-8", "replace").strip()
        except Exception:
            text = ""
        lines = text.splitlines()[-6:]
        return " | ".join(lines) if lines else "no error output"

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise RenderError(f"Internal error: frame has shape {frame.shape} {frame.dtype}, "
                              f"expected ({self.height}, {self.width}, 3) uint8.")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).data)
        except (BrokenPipeError, OSError):
            self._proc.wait()
            raise RenderError(f"FFmpeg stopped unexpectedly: {self._error_text()}") from None
        self.frames += 1

    def close(self) -> None:
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            code = self._proc.wait()
            if code != 0:
                raise RenderError(f"FFmpeg failed with exit code {code}: {self._error_text()}")
        finally:
            self._stderr.close()

    def abort(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait()
        finally:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.close()
            except OSError:
                pass
            self._stderr.close()

    def __enter__(self) -> "FFmpegWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def _bitrate(width: int, height: int, fps: float) -> str:
    return f"{max(2, int(width * height * fps * 0.12 / 1_000_000))}M"
