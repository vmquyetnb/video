import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pencil_timelapse.samples import synthetic_portrait  # noqa: E402


@pytest.fixture(scope="session")
def portrait():
    """Small synthetic portrait set: {"end", "blank", "partial"} uint8 RGB arrays."""
    return synthetic_portrait(450, 675, seed=3)


@pytest.fixture(scope="session")
def sample_files(portrait, tmp_path_factory):
    from PIL import Image

    folder = tmp_path_factory.mktemp("samples")
    paths = {}
    for name, img in portrait.items():
        path = folder / f"{name}.png"
        Image.fromarray(img).save(path)
        paths[name] = str(path)
    return paths


def small_config(**overrides):
    from pencil_timelapse import RenderConfig

    values = dict(width=270, height=480, fps=12, duration=15.0, hold=2.0, preset="ultrafast", workers=2,
                  start_mode="image", show_pencil=False)
    values.update(overrides)
    return RenderConfig(**values).validate()


@pytest.fixture
def cfg_small():
    return small_config()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
