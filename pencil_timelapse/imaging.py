"""Image loading, validation, canvas normalization, paper texture and alignment."""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import InputError

F32 = np.float32

ALLOWED_FORMATS = {"PNG", "JPEG", "MPO", "WEBP", "BMP", "TIFF"}
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
FORMAT_HINT = "Supported formats: PNG, JPEG, WEBP, BMP and TIFF."
MIN_SIDE = 128
MAX_SOURCE_PIXELS = 80_000_000

# Pillow raises DecompressionBombError above twice this value and warns above it.
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS

ImageSource = Union[str, "os.PathLike[str]", bytes, bytearray, np.ndarray, Image.Image]


# --------------------------------------------------------------------------- loading

def load_image(source: ImageSource, label: str = "Image") -> np.ndarray:
    """Load an image as a float32 RGB array in [0, 1].

    Accepts a path, raw bytes, a PIL image or a NumPy array. EXIF rotation is
    applied and transparency is flattened onto white paper.
    """
    if isinstance(source, np.ndarray):
        return _check_size(_array_to_rgb(source, label), label)
    return _check_size(_pil_to_rgb(_open_image(source, label)), label)


def _open_image(source: ImageSource, label: str) -> Image.Image:
    """Open, fully decode, validate and EXIF-rotate an image file."""
    if isinstance(source, Image.Image):
        img = source
    else:
        if isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise InputError(f"{label}: the uploaded file is empty.")
            fp: Union[str, io.BytesIO] = io.BytesIO(bytes(source))
        else:
            fp = os.fspath(source)
            if not os.path.isfile(fp):
                raise InputError(f"{label}: file not found: {fp}")
            if os.path.getsize(fp) == 0:
                raise InputError(f"{label}: the file is empty.")
        try:
            img = Image.open(fp)
            fmt = img.format
            img.load()
        except Image.DecompressionBombError:
            raise InputError(
                f"{label}: the image is too large (limit {MAX_SOURCE_PIXELS // 1_000_000} megapixels)."
            ) from None
        except UnidentifiedImageError:
            raise InputError(f"{label}: the file is not a readable image. {FORMAT_HINT}") from None
        except (OSError, ValueError, SyntaxError) as exc:
            raise InputError(
                f"{label}: the image file appears to be corrupted or truncated ({exc})."
            ) from None
        if fmt not in ALLOWED_FORMATS:
            raise InputError(f"{label}: unsupported image format '{fmt}'. {FORMAT_HINT}")

    w, h = img.size
    if w * h > MAX_SOURCE_PIXELS:
        raise InputError(
            f"{label}: the image is too large ({w}x{h}); the limit is "
            f"{MAX_SOURCE_PIXELS // 1_000_000} megapixels."
        )
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # malformed EXIF must not block an otherwise valid image
        pass
    return img


def _check_size(rgb: np.ndarray, label: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise InputError(
            f"{label}: the image is too small ({w}x{h}). The shorter side must be at least "
            f"{MIN_SIDE} px."
        )
    return rgb


def _pil_to_rgb(img: Image.Image) -> np.ndarray:
    mode = img.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        arr = np.asarray(img).astype(F32)
        peak = 65535.0 if arr.max() > 255 else 255.0
        if mode == "F" and arr.max() <= 1.0:
            peak = 1.0
        gray = np.clip(arr / peak, 0.0, 1.0)
        return np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2))
    has_alpha = mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = np.asarray(img.convert("RGBA")).astype(F32) / 255.0
        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3] * alpha + (1.0 - alpha)
        return np.ascontiguousarray(rgb, dtype=F32)
    return np.asarray(img.convert("RGB")).astype(F32) / 255.0


def _array_to_rgb(arr: np.ndarray, label: str) -> np.ndarray:
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)):
        raise InputError(f"{label}: expected a grayscale, RGB or RGBA array, got shape {arr.shape}.")
    if arr.dtype == np.uint8:
        data = arr.astype(F32) / 255.0
    elif arr.dtype == np.uint16:
        data = arr.astype(F32) / 65535.0
    else:
        data = np.clip(arr.astype(F32), 0.0, 1.0)
    if data.ndim == 2:
        data = data[..., None]
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    elif data.shape[2] == 4:
        alpha = data[..., 3:4]
        data = data[..., :3] * alpha + (1.0 - alpha)
    return np.ascontiguousarray(data, dtype=F32)


# --------------------------------------------------------------------------- tone helpers

def luminance(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=F32), cv2.COLOR_RGB2GRAY)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(np.clip(rgb, 0.0, 1.0), alpha=255.0)


def _downscale(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def estimate_paper_color(rgb: np.ndarray) -> np.ndarray:
    """Estimate the paper colour as the median of the brightest 15 % of pixels."""
    small = _downscale(rgb, 512)
    y = luminance(small)
    sel = small[y >= np.percentile(y, 85)]
    color = np.median(sel, axis=0) if len(sel) else np.ones(3, F32)
    return np.clip(color, 0.2, 1.0).astype(F32)


def high_frequency_std(rgb: np.ndarray) -> float:
    """Standard deviation of fine detail (full-resolution centre crop); near zero
    for flat digital images, higher for photographed or scanned paper."""
    h, w = rgb.shape[:2]
    ch, cw = min(h, 768), min(w, 768)
    crop = rgb[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]
    y = luminance(np.ascontiguousarray(crop))
    return float(np.std(y - cv2.GaussianBlur(y, (0, 0), 3.0)))


def colourfulness(rgb: np.ndarray) -> float:
    """Mean channel spread: about 0 for graphite on paper, high for colour photos."""
    small = _downscale(rgb, 512)
    return float(np.mean(small.max(axis=2) - small.min(axis=2)))


def ink_fraction(rgb: np.ndarray, paper: np.ndarray, threshold: float = 0.12) -> float:
    """Fraction of pixels noticeably darker than the paper."""
    y = luminance(_downscale(rgb, 768))
    paper_y = float(np.dot(paper, [0.299, 0.587, 0.114]))
    return float(np.mean(y < paper_y * (1.0 - threshold)))


# --------------------------------------------------------------------------- paper

def paper_texture(height: int, width: int, seed: int, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Procedural paper surface.

    Returns ``(surface, tooth)``: two zero-mean, unit-variance float32 fields.
    ``surface`` mixes fine grain, tooth, fibres and soft mottling; ``tooth``
    is the medium-scale relief that catches graphite.
    """
    rng = np.random.default_rng(seed + 101)

    def noise() -> np.ndarray:
        return rng.standard_normal((height, width), dtype=F32)

    def z(a: np.ndarray) -> np.ndarray:
        a = a - a.mean()
        return (a / (a.std() + 1e-6)).astype(F32)

    fine = cv2.GaussianBlur(noise(), (0, 0), 0.55 * scale + 0.25)
    tooth = cv2.GaussianBlur(noise(), (0, 0), 1.6 * scale)
    fibres = (cv2.GaussianBlur(noise(), (0, 0), sigmaX=5.0 * scale, sigmaY=0.7 * scale)
              + cv2.GaussianBlur(noise(), (0, 0), sigmaX=0.7 * scale, sigmaY=5.0 * scale))
    coarse = rng.standard_normal((height // 40 + 3, width // 40 + 3), dtype=F32)
    mottle = cv2.resize(cv2.GaussianBlur(coarse, (0, 0), 1.5), (width, height),
                        interpolation=cv2.INTER_CUBIC)
    surface = 0.6 * z(fine) + 0.45 * z(tooth) + 0.2 * z(fibres) + 0.12 * z(mottle)
    return z(surface), z(tooth)


def paper_canvas(height: int, width: int, color: np.ndarray, surface: np.ndarray,
                 amplitude: float) -> np.ndarray:
    shade = 1.0 + amplitude * surface
    return np.clip(shade[..., None] * color[None, None, :], 0.0, 1.0).astype(F32)


# --------------------------------------------------------------------------- canvas fit

@dataclass(frozen=True)
class Placement:
    """Where a source image landed on the output canvas."""

    x: int
    y: int
    width: int
    height: int
    scale: float
    source_width: int
    source_height: int

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height,
                "scale": round(self.scale, 4), "source_width": self.source_width,
                "source_height": self.source_height}


def fit_to_canvas(rgb: np.ndarray, background: np.ndarray) -> Tuple[np.ndarray, Placement]:
    """Scale ``rgb`` to fit inside ``background`` (aspect ratio preserved) and centre it."""
    H, W = background.shape[:2]
    h, w = rgb.shape[:2]
    s = min(W / w, H / h)
    nw, nh = min(W, max(1, round(w * s))), min(H, max(1, round(h * s)))
    if (nw, nh) == (w, h):
        resized = rgb
    else:
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LANCZOS4
        resized = np.clip(cv2.resize(rgb, (nw, nh), interpolation=interp), 0.0, 1.0)
    canvas = background.copy()
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, Placement(x0, y0, nw, nh, s, w, h)


# --------------------------------------------------------------------------- alignment

@dataclass
class Alignment:
    applied: bool
    message: str
    matrix: Optional[np.ndarray] = None
    inliers: int = 0


def estimate_alignment(src: np.ndarray, dst: np.ndarray, seed: int = 0) -> Alignment:
    """Estimate a similarity transform that registers ``src`` onto ``dst``.

    Uses ORB features + RANSAC. Only small, well-supported corrections are
    accepted so that unrelated or blank start images are left untouched.
    """
    H, W = dst.shape[:2]
    s = min(1.0, 1000.0 / max(H, W))
    size = (max(1, round(W * s)), max(1, round(H * s)))

    def prep(rgb: np.ndarray) -> np.ndarray:
        g = cv2.resize(luminance(rgb), size, interpolation=cv2.INTER_AREA)
        return cv2.equalizeHist(cv2.convertScaleAbs(g, alpha=255.0))

    a, b = prep(src), prep(dst)
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=10)
    kp1, d1 = orb.detectAndCompute(a, None)
    kp2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None or len(kp1) < 30 or len(kp2) < 30:
        return Alignment(False, "skipped: the start image has too few distinctive features")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 25:
        return Alignment(False, f"skipped: only {len(good)} reliable feature matches")
    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(seed)
    M, mask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                          maxIters=4000, confidence=0.995)
    if M is None or mask is None:
        return Alignment(False, "skipped: no consistent transform found")
    inliers = int(mask.sum())
    if inliers < 20 or inliers < 0.25 * len(good):
        return Alignment(False, f"skipped: weak geometric agreement ({inliers} inliers)", inliers=inliers)
    scale = math.sqrt(abs(np.linalg.det(M[:, :2])))
    angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    M_full = M.copy()
    M_full[:, 2] /= s
    shift = math.hypot(M_full[0, 2] + (M_full[0, 0] - 1) * W / 2 + M_full[0, 1] * H / 2,
                       M_full[1, 2] + M_full[1, 0] * W / 2 + (M_full[1, 1] - 1) * H / 2)
    if not (0.85 <= scale <= 1.18 and abs(angle) <= 10.0 and shift <= 0.12 * math.hypot(W, H)):
        return Alignment(False, f"skipped: implausible correction (scale {scale:.2f}, "
                                f"rotation {angle:.1f} deg)", inliers=inliers)
    if abs(scale - 1) < 0.002 and abs(angle) < 0.1 and shift < 0.75:
        return Alignment(False, "not needed: images are already aligned", inliers=inliers)
    return Alignment(True, f"applied (scale {scale:.3f}, rotation {angle:.2f} deg, "
                           f"shift {shift:.1f} px, {inliers} inliers)", M_full.astype(F32), inliers)


def warp_to_canvas(rgb: np.ndarray, matrix: np.ndarray, background: np.ndarray) -> np.ndarray:
    H, W = background.shape[:2]
    warped = cv2.warpAffine(rgb, matrix, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    coverage = cv2.warpAffine(np.ones((H, W), F32), matrix, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # The warp leaves black borders; pre-multiplied blending fills them with paper.
    return np.clip(warped + background * (1.0 - coverage[..., None]), 0.0, 1.0).astype(F32)
