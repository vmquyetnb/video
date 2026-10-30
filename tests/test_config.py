import pytest

from pencil_timelapse import ConfigError, RenderConfig, build_timeline


def test_default_timeline_matches_specification():
    cfg = RenderConfig()
    stages = build_timeline(cfg.duration, cfg.hold)
    spans = [(s.key, round(s.start, 6), round(s.end, 6)) for s in stages]
    assert spans == [
        ("construction", 0.0, 3.0),
        ("contours", 3.0, 6.0),
        ("features", 6.0, 9.0),
        ("shading", 9.0, 12.0),
        ("finishing", 12.0, 15.0),
        ("hold", 15.0, 15.0),
    ]


def test_timeline_scales_with_duration_and_hold():
    stages = build_timeline(32.0, 2.0)
    assert stages[0].end == pytest.approx(6.0)
    assert stages[-2].key == "finishing" and stages[-2].end == pytest.approx(30.0)
    assert stages[-1].start == pytest.approx(30.0) and stages[-1].end == pytest.approx(32.0)


def test_defaults_are_valid_and_vertical():
    cfg = RenderConfig().validate()
    assert (cfg.width, cfg.height, cfg.fps, cfg.duration) == (1080, 1920, 30, 15.0)
    assert cfg.frame_count == 450
    assert cfg.start_mode == "blank" and cfg.show_pencil is True


@pytest.mark.parametrize("kwargs, fragment", [
    ({"width": 1081}, "even"),
    ({"height": 100}, "between"),
    ({"fps": 0}, "FPS"),
    ({"fps": 120}, "FPS"),
    ({"duration": 2.0}, "duration"),
    ({"hold": 12.0}, "hold"),
    ({"duration": 5.0, "hold": 3.0}, "at least"),
    ({"paper_texture": 2.0}, "paper texture"),
    ({"preset": "insane"}, "preset"),
    ({"width": 3840, "height": 3840}, "too large"),
    ({"start_mode": "photo"}, "start mode"),
    ({"pencil_scale": 5.0}, "pencil size"),
    ({"pencil_style": "red"}, "pencil style"),
])
def test_invalid_settings_are_rejected(kwargs, fragment):
    with pytest.raises(ConfigError) as err:
        RenderConfig(**kwargs).validate()
    assert fragment.lower() in str(err.value).lower()


def test_from_mapping_parses_form_values():
    cfg = RenderConfig.from_mapping({
        "resolution": "720x1280", "fps": "24", "duration": "20", "hold": "1.5",
        "align": "false", "match_paper": "true", "stroke_strength": "0.3", "crf": "20",
    })
    assert (cfg.width, cfg.height, cfg.fps, cfg.duration, cfg.hold) == (720, 1280, 24, 20.0, 1.5)
    assert cfg.align is False and cfg.match_paper is True
    assert cfg.stroke_strength == pytest.approx(0.3) and cfg.crf == 20


def test_from_mapping_custom_resolution_and_errors():
    cfg = RenderConfig.from_mapping({"resolution": "custom", "width": "600", "height": "1000"})
    assert (cfg.width, cfg.height) == (600, 1000)
    with pytest.raises(ConfigError, match="fps"):
        RenderConfig.from_mapping({"fps": "fast"})
    with pytest.raises(ConfigError, match="resolution"):
        RenderConfig.from_mapping({"resolution": "huge"})
