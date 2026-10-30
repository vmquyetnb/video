import io
import json
import os

import numpy as np
import pytest

from pencil_timelapse import animals as AN
from pencil_timelapse.errors import InputError
from pencil_timelapse.media import write_wav
from pencil_timelapse.showtime import ShowOptions

PLAIN = ShowOptions(intro=False, highlights=True, outro=False, sound_effects=False)


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (1080, 1080)])
def test_scene_layout_fits(size):
    W, H = size
    lay = AN.scene_layout(W, H)
    x, y, w, h = lay.draw_box
    fx, fy, fw, fh = lay.facts
    assert 0 <= x and x + w <= W and 0 <= y and y + h <= H
    assert 0 <= fx and fx + fw <= W and fy + fh <= H and fh > 0.15 * H
    assert x + w <= fx or y + h <= fy  # the text never covers the drawing


def test_facts_are_wrapped_and_paged():
    facts = [f"Fact number {i} is a rather long sentence about this animal and its habits."
             for i in range(12)]
    placed = AN._facts(facts, (100, 300, 600, 500), 40)
    assert len(placed) == 12 and placed[-1].page > 0
    for fact in placed:
        for line in fact.lines:
            assert line.x + line.text.ink[1] <= 100 + 600 + 10  # wrapped to the column
            assert 300 <= line.y <= 800


def test_reading_time_grows_with_the_fact():
    assert AN.reading_seconds("Short.") == 2.5
    assert AN.reading_seconds(" ".join(["word"] * 20)) > AN.reading_seconds(" ".join(["word"] * 8))


def _animals(sample_files):
    return [AN.Animal("Lion", sample_files["end"], ["Lions live in prides.",
                                                    "A roar can be heard 8 km away."]),
            AN.Animal("Owl", sample_files["end"], ["Owls can turn their heads 270 degrees."])]


def test_each_animal_stays_until_its_facts_are_shown(tmp_path, sample_files):
    team = _animals(sample_files)
    labels = []
    result = AN.render_animals(team, "Animals", str(tmp_path / "a.mp4"), 480, 270, fps=10,
                               drawing_seconds=4, show=PLAIN, preset="ultrafast",
                               progress=lambda f, m: labels.append(m))
    # Each scene lasts as long as the longest of: the drawing then the name, and the
    # facts (written from the first stroke on); then the sweep. Then the gallery.
    facts = [sum(AN.reading_seconds(f) for f in a.facts) for a in team]
    expected = sum(max(4 + AN.NAME_TIME, 0.8 + f + AN.END_HOLD) + 0.8 for f in facts) + 3.0
    assert result.duration == pytest.approx(expected, abs=0.25)
    assert labels.index("Drawing Owl") > labels.index("Lion: the facts")
    assert os.path.getsize(result.output_path) > 1000 and os.path.exists(result.image_path)


def test_narration_starts_with_the_drawing_and_sets_the_scene_length(tmp_path, sample_files):
    import subprocess
    from pencil_timelapse.encoder import find_ffmpeg

    voice = tmp_path / "voice.wav"
    t = np.arange(int(20 * 44100)) / 44100.0
    write_wav(str(voice), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    team = [AN.Animal("Lion", sample_files["end"], ["Lions live in prides."], str(voice))]
    result = AN.render_animals(team, "", str(tmp_path / "a.mp4"), 480, 270, fps=10,
                               drawing_seconds=4, show=PLAIN, preset="ultrafast")
    # The 20 s narration runs from the first stroke; the scene waits for it to end.
    assert result.duration == pytest.approx(20 + AN.END_HOLD + 0.8 + 3.0, abs=0.25)
    raw = subprocess.run([find_ffmpeg(), "-v", "error", "-i", result.output_path, "-vn", "-ac",
                          "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True).stdout
    sound = np.frombuffer(raw, np.int16) / 32768.0
    assert np.sqrt((sound[int(0.4 * 8000):int(1.5 * 8000)] ** 2).mean()) > 0.05  # while drawing
    assert np.sqrt((sound[int(22 * 8000):] ** 2).mean()) < 0.01  # over by the gallery


def test_full_show_with_intro_and_ending(tmp_path, sample_files):
    result = AN.render_animals(_animals(sample_files), "Wild", str(tmp_path / "a.mp4"), 480, 270,
                               fps=10, drawing_seconds=4, preset="ultrafast")
    plain = AN.render_animals(_animals(sample_files), "Wild", str(tmp_path / "b.mp4"), 480, 270,
                              fps=10, drawing_seconds=4, preset="ultrafast", show=PLAIN)
    assert result.duration == pytest.approx(plain.duration + 5.0 + 7.0, abs=0.25)


def test_input_errors(tmp_path, sample_files):
    with pytest.raises(InputError):
        AN.render_animals([], "", str(tmp_path / "a.mp4"))
    with pytest.raises(InputError):
        AN.render_animals([AN.Animal(" ", sample_files["end"])], "", str(tmp_path / "a.mp4"))
    with pytest.raises(InputError):
        AN.render_animals(_animals(sample_files), "", str(tmp_path / "a.mp4"), drawing_seconds=2)


def test_script_file(tmp_path, sample_files):
    script = tmp_path / "animals.json"
    script.write_text(json.dumps({"title": "Zoo", "animals": [
        {"name": "Lion", "photo": os.path.relpath(sample_files["end"], tmp_path),
         "facts": ["Big cat."]}]}))
    title, animals = AN.load_script(str(script))
    assert title == "Zoo" and animals[0].facts == ["Big cat."]
    assert os.path.samefile(animals[0].photo, sample_files["end"])


def test_numbers_get_highlighted():
    text = AN._text("A roar can be heard 8 kilometres away, 270 độ and 40,000 muscles.", 40)
    assert len(text.spans) == 3
    assert all(a < b for a, b in text.spans) and text.spans[0][1] < text.spans[1][0]
    assert AN._text("No numbers here.", 40).spans == []


def test_animal_is_found_before_the_scenery():
    import cv2
    rng = np.random.default_rng(5)
    photo = np.clip(0.75 + 0.08 * rng.standard_normal((300, 400, 3)), 0, 1).astype(np.float32)
    cv2.ellipse(photo, (200, 160), (90, 70), 0, 0, 360, (0.25, 0.2, 0.15), -1)
    mask = AN._animal_mask(photo)
    assert mask is not None and mask.shape == (300, 400)
    assert mask[160, 200] > 0.9 and mask[10, 10] < 0.1 and mask[290, 390] < 0.1


def test_pencil_is_heard_only_on_the_paper():
    from pencil_timelapse.pencil import PencilRig, make_pencil
    times = np.arange(0.0, 4.0, 1 / 240)
    x = np.interp(times, [0, 1, 2, 2.5, 4], [900, 100, 400, 900, 900])
    y = np.interp(times, [0, 1, 2, 2.5, 4], [700, 200, 200, 700, 700])
    lift = np.interp(times, [0, 0.95, 1.0, 2.0, 2.05, 4], [1, 1, 0, 0, 1, 1])
    rig = PencilRig(make_pencil(0.5), times, x, y, lift, np.zeros_like(times), 4.0, 0.5, 0.05)
    sound = AN._scratch(rig, 4.0)
    rate = 44100
    assert np.abs(sound[: int(0.5 * rate)]).max() == 0.0  # still lifted
    assert np.abs(sound[int(1.2 * rate):int(1.8 * rate)]).max() > 0.01  # writing
    assert np.abs(sound[int(3.0 * rate):]).max() == 0.0  # gone


def test_long_drawing_time_draws_the_animal_first(tmp_path, sample_files):
    labels = []
    team = _animals(sample_files)[:1]
    result = AN.render_animals(team, "", str(tmp_path / "a.mp4"), 480, 270, fps=10,
                               drawing_seconds=8, show=PLAIN, preset="ultrafast",
                               progress=lambda f, m: labels.append(m))
    facts = sum(AN.reading_seconds(f) for f in team[0].facts)
    scene = max(8 + AN.NAME_TIME, 0.8 + facts + AN.END_HOLD)
    assert result.duration == pytest.approx(scene + 0.8 + 3.0, abs=0.25)
    assert "Drawing Lion" in labels


def test_facts_are_written_while_the_animal_is_drawn():
    animal = AN.Animal("Lion", "x.jpg", ["Lions live in prides.", "A roar carries 8 km."])
    lay = AN.scene_layout(960, 540)
    text = AN._SceneText(animal, "#1 / 1", lay, 10.0, 0.0, guess=True)
    assert text.starts[0] < 1.0 and text.starts[1] < 10.0  # during the drawing
    assert text.named == 10.0  # the name only when the drawing is done
    fx, fy, fw, fh = lay.facts
    blank = np.ones((540, 960, 3), np.float32)
    early, later = blank.copy(), blank.copy()
    text.paint(early, 0.5)
    text.paint(later, 2.5)
    assert early[fy:fy + fh, fx:fx + fw].min() > 0.9  # nothing written yet
    assert later[fy:fy + fh, fx:fx + fw].min() < 0.5  # the first fact, mid-drawing
    nx, ny = lay.name
    assert later[ny:ny + lay.name_size, nx:nx + 300].min() < 0.9  # "Guess the animal?"
