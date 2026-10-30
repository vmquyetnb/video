import io

import numpy as np
import pytest
from PIL import Image

from pencil_timelapse import InputError
from pencil_timelapse.imaging import (estimate_alignment, estimate_paper_color, fit_to_canvas,
                                      load_image, warp_to_canvas)


def _png_bytes(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def test_load_rgb_png_roundtrip():
    arr = (np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8)
    out = load_image(_png_bytes(arr), "Start image")
    assert out.shape == (200, 300, 3) and out.dtype == np.float32
    assert np.array_equal(np.round(out * 255).astype(np.uint8), arr)


def test_transparency_is_flattened_onto_white():
    rgba = np.zeros((200, 200, 4), np.uint8)
    rgba[..., 3] = 0  # fully transparent
    out = load_image(_png_bytes(rgba), "End image")
    assert np.allclose(out, 1.0)


def test_grayscale_and_16bit_images():
    gray = np.full((150, 150), 128, np.uint8)
    assert load_image(_png_bytes(gray), "x").shape == (150, 150, 3)
    deep = np.full((150, 150), 65535, np.uint16)
    buf = io.BytesIO()
    Image.fromarray(deep).save(buf, format="PNG")
    assert np.allclose(load_image(buf.getvalue(), "x"), 1.0)


@pytest.mark.parametrize("payload, fragment", [
    (b"", "empty"),
    (b"definitely not an image", "not a readable image"),
])
def test_unreadable_files_raise_clear_errors(payload, fragment):
    with pytest.raises(InputError) as err:
        load_image(payload, "Start image")
    assert "Start image" in str(err.value) and fragment in str(err.value)


def test_truncated_file_is_reported():
    data = _png_bytes(np.zeros((300, 300, 3), np.uint8))
    with pytest.raises(InputError, match="End image"):
        load_image(data[: len(data) // 3], "End image")


def test_too_small_image_is_rejected():
    with pytest.raises(InputError, match="too small"):
        load_image(_png_bytes(np.zeros((64, 400, 3), np.uint8)), "End image")


def test_unsupported_format_is_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200)).save(buf, format="GIF")
    with pytest.raises(InputError, match="unsupported image format"):
        load_image(buf.getvalue(), "Start image")


def test_missing_path():
    with pytest.raises(InputError, match="not found"):
        load_image("/no/such/file.png", "Start image")


def test_fit_preserves_aspect_ratio_and_centres():
    img = np.zeros((500, 1000, 3), np.float32)
    bg = np.ones((1920, 1080, 3), np.float32)
    canvas, place = fit_to_canvas(img, bg)
    assert (place.width, place.height) == (1080, 540)
    assert (place.x, place.y) == (0, 690)
    assert canvas[690:1230].max() == 0.0 and canvas[:690].min() == 1.0


def test_fit_without_resize_is_exact():
    rng = np.random.default_rng(1)
    img = rng.random((480, 270, 3)).astype(np.float32)
    canvas, place = fit_to_canvas(img, np.ones((480, 270, 3), np.float32))
    assert place.scale == 1.0 and np.array_equal(canvas, img)


def test_paper_color_estimate(portrait):
    paper = estimate_paper_color(portrait["end"].astype(np.float32) / 255)
    assert np.allclose(paper, [0.958, 0.951, 0.932], atol=0.02)


def test_alignment_recovers_a_small_shift(portrait):
    end = portrait["end"].astype(np.float32) / 255
    h, w = end.shape[:2]
    shifted = np.roll(np.roll(end, 9, axis=1), -6, axis=0)
    result = estimate_alignment(shifted, end, seed=1)
    assert result.applied, result.message
    assert result.matrix[0, 2] == pytest.approx(-9, abs=1.5)
    assert result.matrix[1, 2] == pytest.approx(6, abs=1.5)
    fixed = warp_to_canvas(shifted, result.matrix, np.ones_like(end))
    core = (slice(20, h - 20), slice(20, w - 20))
    assert np.abs(fixed[core] - end[core]).mean() < np.abs(shifted[core] - end[core]).mean() * 0.3


def test_alignment_skips_blank_start(portrait):
    blank = np.full((675, 450, 3), 0.95, np.float32)
    result = estimate_alignment(blank, portrait["end"].astype(np.float32) / 255)
    assert not result.applied
