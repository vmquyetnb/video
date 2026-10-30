import os

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import psnr, small_config
from pencil_timelapse import InputError, plan_animation, render_video
from pencil_timelapse.imaging import to_uint8
from pencil_timelapse.renderer import FrameRenderer


def _frames(plan, cfg):
    renderer = FrameRenderer(plan)
    return renderer, [f for _, _, f in renderer.iter_frames(cfg.fps, cfg.frame_count, workers=2)]


@pytest.fixture(scope="module")
def blank_run(sample_files):
    cfg = small_config()
    plan = plan_animation(sample_files["blank"], sample_files["end"], cfg)
    renderer, frames = _frames(plan, cfg)
    return cfg, plan, renderer, frames


def test_last_two_seconds_hold_the_exact_end_image(blank_run):
    cfg, plan, renderer, frames = blank_run
    hold_frames = int(round(cfg.hold * cfg.fps))
    for frame in frames[-hold_frames:]:
        assert np.array_equal(frame, plan.end_u8)
    # The drawing settles on the end image during the finishing stage, before the hold.
    assert renderer.finish_end <= 13.0


def test_end_canvas_is_the_resized_end_image(blank_run, portrait):
    cfg, plan, _, _ = blank_run
    place = plan.info["end_placement"]
    region = plan.end_u8[place["y"]:place["y"] + place["height"], place["x"]:place["x"] + place["width"]]
    expected = cv2.resize(portrait["end"].astype(np.float32) / 255, (place["width"], place["height"]),
                          interpolation=cv2.INTER_AREA)
    assert np.abs(region.astype(np.int16) - to_uint8(expected).astype(np.int16)).max() <= 1


def test_first_frame_is_the_start_image(blank_run):
    _, plan, _, frames = blank_run
    assert np.array_equal(frames[0], to_uint8(plan.start))


def test_no_abrupt_jumps_between_frames(blank_run):
    _, _, _, frames = blank_run
    diffs = [np.abs(frames[i + 1].astype(np.int16) - frames[i].astype(np.int16)).mean()
             for i in range(len(frames) - 1)]
    # 12 fps test video: even the busiest frame changes by < 3 grey levels on average.
    assert max(diffs) < 3.0, max(diffs)


def test_drawing_converges_stage_by_stage(blank_run):
    cfg, plan, renderer, _ = blank_run
    end = plan.end_u8.astype(np.float32)
    errors = [np.abs(renderer.render(t).astype(np.float32) - end).mean()
              for t in (0.0, 3.0, 6.0, 9.0, 12.0, 13.0)]
    assert all(a > b for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] == 0.0


def test_outline_stage_reveals_lines_not_areas(blank_run):
    _, plan, renderer, _ = blank_run
    start = renderer.render(0.0).astype(np.int16)
    changed_3s = (np.abs(renderer.render(3.0).astype(np.int16) - start).max(axis=2) > 12).mean()
    changed_12s = (np.abs(renderer.render(12.0).astype(np.int16) - start).max(axis=2) > 12).mean()
    assert changed_3s < 0.08, changed_3s
    assert changed_12s > 3 * changed_3s


def test_partial_start_is_continued_not_restarted(sample_files, blank_run):
    cfg = small_config()
    plan = plan_animation(sample_files["partial"], sample_files["end"], cfg)
    renderer = FrameRenderer(plan)
    start = to_uint8(plan.start).astype(np.int16)
    lum = start.mean(axis=2)
    existing_lines = lum < np.percentile(lum, 3)  # the outlines already on the page
    for t in (0.5, 1.5, 3.0, 6.0):
        frame = renderer.render(t).astype(np.int16)
        # Existing artwork is never wiped: those pixels only stay or get darker.
        assert (frame[existing_lines] - start[existing_lines]).max() <= 2
    blank_plan = blank_run[1]
    def drawn(info):  # graphite the animation still adds
        return sum(v for k, v in info["stage_work"].items() if k != "remaining")

    assert drawn(plan.info) < drawn(blank_plan.info)
    assert plan.info["existing_artwork_fraction"] > 0.03
    assert np.array_equal(renderer.render(13.0), plan.end_u8)


def test_rendering_is_deterministic(sample_files):
    cfg = small_config()
    a = FrameRenderer(plan_animation(sample_files["blank"], sample_files["end"], cfg))
    b = FrameRenderer(plan_animation(sample_files["blank"], sample_files["end"], cfg))
    for t in (1.0, 4.0, 7.5, 10.0, 12.5):
        assert np.array_equal(a.render(t), b.render(t))


def test_marks_missing_from_the_end_image_are_erased(portrait, tmp_path):
    start = portrait["blank"].copy()
    cv2.line(start, (40, 600), (400, 640), (40, 40, 40), 6)  # a stray mark
    path = tmp_path / "stray.png"
    Image.fromarray(start).save(path)
    cfg = small_config()
    plan = plan_animation(str(path), portrait["end"], cfg)
    assert plan.info["erase_needed"]
    renderer = FrameRenderer(plan)
    assert np.array_equal(renderer.render(13.0), plan.end_u8)


def test_identical_images_produce_a_gentle_crossfade(portrait):
    cfg = small_config()
    plan = plan_animation(portrait["end"], portrait["end"], cfg)
    assert any("nearly identical" in w for w in plan.warnings)
    renderer = FrameRenderer(plan)
    assert psnr(renderer.render(5.0), plan.end_u8) > 35


def test_different_aspect_ratios_warn_but_render(portrait):
    cfg = small_config()
    wide = cv2.resize(portrait["blank"], (600, 400))
    plan = plan_animation(wide, portrait["end"], cfg)
    assert any("aspect ratios" in w for w in plan.warnings)
    frame = FrameRenderer(plan).render(6.0)
    assert frame.shape == (480, 270, 3)


def test_render_video_end_to_end(sample_files, tmp_path):
    cfg = small_config(duration=8.0, hold=2.0, fps=12)
    out = tmp_path / "nested" / "drawing.mp4"
    progress = []
    result = render_video(sample_files["blank"], sample_files["end"], str(out), cfg,
                          lambda f, m: progress.append((f, m)))
    assert out.exists() and out.stat().st_size > 10_000
    assert result.frames == 96 and result.duration == pytest.approx(8.0)
    assert [p for p, _ in progress] == sorted(p for p, _ in progress)
    assert progress[-1][0] == 1.0
    assert os.listdir(out.parent) == ["drawing.mp4"]  # temporary files removed

    cap = cv2.VideoCapture(str(out))
    decoded = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    assert len(decoded) == 96
    assert decoded[0].shape == (480, 270, 3)
    plan = plan_animation(sample_files["blank"], sample_files["end"], cfg)
    assert psnr(decoded[-1], plan.end_u8) > 35


def test_render_video_reports_bad_input_without_leaving_files(tmp_path, sample_files):
    out = tmp_path / "x.mp4"
    with pytest.raises(InputError, match="Start image"):
        render_video(str(tmp_path / "missing.png"), sample_files["end"], str(out), small_config())
    assert not out.exists()


def test_face_detection_path_drives_focus(monkeypatch, sample_files):
    """Exercise the face/eye branch with a stub cascade (the synthetic face is
    too stylised for OpenCV's Haar cascades)."""
    from pencil_timelapse import analysis

    class Stub:
        def __init__(self, rects):
            self.rects = rects

        def detectMultiScale(self, image, **kwargs):
            return np.array(self.rects)

    h, w = 480, 270  # detection runs at full size for small canvases
    face = [int(w * 0.3), int(h * 0.25), int(w * 0.4), int(h * 0.25)]
    eyes = [[8, 20, 20, 14], [64, 20, 20, 14]]
    stubs = {"haarcascade_frontalface_default.xml": Stub([face]), "haarcascade_eye.xml": Stub(eyes)}
    monkeypatch.setattr(analysis, "_cascade", lambda name: stubs.get(name))

    plan = plan_animation(sample_files["blank"], sample_files["end"], small_config())
    focus = plan.info["focus"]
    assert focus["method"] == "face + eyes"
    assert focus["face"] == face and len(focus["eyes"]) == 2
    renderer = FrameRenderer(plan)
    assert np.array_equal(renderer.render(13.0), plan.end_u8)



def test_blank_paper_start(portrait):
    cfg = small_config(start_mode="blank")
    plan = plan_animation(None, portrait["end"], cfg)
    renderer = FrameRenderer(plan)
    first = renderer.render(0.0).astype(np.float32)
    paper = np.array(plan.info["paper_color"]) * 255
    assert np.abs(first.reshape(-1, 3).mean(0) - paper).max() < 3  # an empty sheet of the same paper
    assert first.std() < 6
    assert plan.info["start_mode"] == "blank paper"
    assert np.array_equal(renderer.render(13.0), plan.end_u8)


def test_image_mode_requires_a_start_image(portrait):
    with pytest.raises(InputError, match="Blank white paper"):
        plan_animation(None, portrait["end"], small_config(start_mode="image"))


def test_photo_like_start_is_flagged(portrait):
    rng = np.random.default_rng(0)
    photo = (rng.random((675, 450, 3)) * [255, 120, 60]).astype(np.uint8)
    plan = plan_animation(photo, portrait["end"], small_config())
    assert any("photograph" in w for w in plan.warnings)


@pytest.fixture(scope="module")
def pencil_run(sample_files):
    cfg = small_config(start_mode="blank", show_pencil=True)
    plan = plan_animation(None, sample_files["end"], cfg)
    return plan, FrameRenderer(plan)


def test_pencil_is_off_page_at_start_and_gone_before_hold(pencil_run):
    plan, renderer = pencil_run
    assert np.array_equal(renderer.render(0.0), to_uint8(plan.start))
    rig = plan.pencil
    assert rig.end_time < renderer.finish_end
    assert np.array_equal(renderer.render(renderer.finish_end), plan.end_u8)


def test_marks_appear_where_the_pencil_tip_passes(pencil_run):
    """Every mark added during a frame lies on the path the tip traced in that frame."""
    plan, renderer = pencil_run
    rig = plan.pencil
    finish0 = plan.finish[0]
    plan.pencil = None  # measure the drawing itself, without the overlay
    try:
        shares = []
        for t in np.arange(3.2, finish0 - 0.05, 0.25):
            before = renderer.render(t - 1 / 12).astype(np.float32).mean(axis=2)
            after = renderer.render(t).astype(np.float32).mean(axis=2)
            ys, xs = np.nonzero(np.clip(before - after, 0, None) > 3)
            if len(xs) < 10:
                continue
            path = np.array([rig.state(u)[:2] for u in np.linspace(t - 1 / 12, t, 80)])
            dist = np.min(np.hypot(xs[:, None] - path[None, :, 0], ys[:, None] - path[None, :, 1]),
                          axis=1)
            shares.append(np.mean(dist < 0.03 * plan.width + 4))
    finally:
        plan.pencil = rig
    assert len(shares) > 10
    assert np.mean(shares) > 0.85, np.mean(shares)


def test_pencil_is_visible_while_drawing(pencil_run):
    plan, renderer = pencil_run
    rig = plan.pencil
    # A moment when the whole pencil is inside the frame.
    t = next(t for t in np.arange(3.0, 11.0, 0.25)
             if rig.state(t)[0] < 0.5 * plan.width and rig.state(t)[1] < 0.6 * plan.height)
    with_pencil = renderer.render(t)
    plan.pencil = None
    try:
        without = renderer.render(t)
    finally:
        plan.pencil = rig
    changed = np.abs(with_pencil.astype(np.int16) - without.astype(np.int16)).max(axis=2) > 20
    x, y, _, _ = rig.state(t)
    ys, xs = np.nonzero(changed)
    assert changed.sum() > 200
    assert np.hypot(xs - x, ys - y).min() < 6  # the graphite point sits on the drawing position



@pytest.mark.parametrize("style", ["black", "yellow", "natural"])
def test_pencil_styles_render(style):
    from pencil_timelapse.pencil import make_pencil

    sprite = make_pencil(0.5, style)
    alpha = sprite.rgba[..., 3]
    assert alpha.max() > 0.99 and alpha.min() == 0.0
    tx, ty = sprite.tip
    assert alpha[int(ty), int(tx)] > 0.3  # the graphite point is where the tip is declared


def test_fast_pencil_motion_is_blurred(pencil_run):
    plan, _ = pencil_run
    rig = plan.pencil
    # The fastest moment at which the whole pencil is inside the frame.
    times = [t for t in np.arange(1.0, 11.0, 0.05)
             if rig.state(t)[0] < 0.55 * plan.width and rig.state(t)[1] < 0.6 * plan.height]
    speeds = [np.hypot(*np.subtract(rig.state(t + 0.02)[:2], rig.state(t - 0.02)[:2]))
              for t in times]
    t = times[int(np.argmax(speeds))]
    sharp_rig = type(rig)(**{**rig.__dict__, "shutter": 0.0})
    blurred = np.zeros((plan.height, plan.width, 3), np.float32) + 0.9
    sharp = blurred.copy()
    rig.draw(blurred, t)
    sharp_rig.draw(sharp, t)
    # Motion blur spreads the pencil over more pixels, with softer coverage.
    assert (np.abs(blurred - 0.9).max(axis=2) > 0.01).sum() > (np.abs(sharp - 0.9).max(axis=2) > 0.01).sum()



def test_default_video_ends_on_the_exact_end_image(portrait):
    from pencil_timelapse import RenderConfig

    cfg = RenderConfig(width=270, height=480, fps=12, start_mode="blank", workers=1)
    plan = plan_animation(None, portrait["end"], cfg)
    renderer = FrameRenderer(plan)
    last = (cfg.frame_count - 1) / cfg.fps
    assert renderer.finish_end <= last
    assert np.array_equal(renderer.render(last), plan.end_u8)
    assert plan.pencil.end_time < renderer.finish_end


def test_artist_order_of_stages(sample_files):
    cfg = small_config(start_mode="blank", hold=0.0)
    plan = plan_animation(None, sample_files["end"], cfg)
    spans = {s.key: s for s in plan.stages}
    renderer = FrameRenderer(plan)
    blank = renderer.render(0.0).astype(np.float32)

    def changed(t, level=10):
        diff = np.abs(renderer.render(t).astype(np.float32) - blank).max(axis=2)
        return float((diff > level).mean())

    # Construction: light, loose guide lines only - no strong marks yet.
    assert changed(spans["construction"].end) > 0.002
    assert changed(spans["construction"].end, level=80) < 0.002
    # Contours are line work; the big tonal areas come later.
    assert changed(spans["contours"].end) < 0.10
    assert changed(spans["shading"].end) > 2 * changed(spans["contours"].end)
    # Guide marks and covered highlights are cleaned up before the end.
    assert all(ov.fade_t1 <= renderer.finish_end for ov in plan.overlays)
    stage_of = {l.name: l.stage for l in plan.layers}
    assert stage_of["contour lines"] == "contours"
    assert stage_of["parallel hatching"] == stage_of["cross-hatching"] == "shading"
    assert stage_of["deepest shadows"] == "finishing"
