import json
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from pencil_timelapse import media
from pencil_timelapse.errors import InputError
from pencil_timelapse.media import SAMPLE_RATE, find_pauses, write_wav
from pencil_timelapse.sketch import PAPER, photo_to_pencil
from pencil_timelapse.story import (Captions, Scene, StoryScript, caption_at, plan_timing,
                                    render_story, subtitle_chunks, template_from_text, _font)


def _tone(seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


PARA_1 = ("An là một cậu bé học lớp ba. Mỗi sáng, An vội vàng chạy đến trường và thường quên "
          "rửa tay sau khi chơi ngoài sân.")
PARA_2 = ("Một hôm, An bị đau bụng và phải nghỉ học. Bác sĩ khám cho An và giải thích rằng vi "
          "khuẩn có thể bám trên đôi tay của chúng ta.")


def test_template_keeps_paragraphs_as_scenes():
    data = template_from_text(f"Chuyện của An\n\n{PARA_1}\n\n\n{PARA_2}")
    assert data["title"] == "Chuyện của An"
    assert [s["text"] for s in data["scenes"]] == [PARA_1, PARA_2]
    assert "doctor" in data["scenes"][1]["image_query"]


def test_story_pasted_as_one_block_is_split_at_sentences():
    from pencil_timelapse.story import SCENE_LENGTHS, split_story
    sentences = [f"Câu số {i} kể về việc rửa tay sạch sẽ trước khi ăn cơm cùng gia đình." for i in range(20)]
    title, scenes = split_story("Rửa tay\n" + " ".join(sentences), SCENE_LENGTHS["medium"])
    assert title == "Rửa tay"
    assert len(scenes) >= 5
    assert all(s.endswith(".") for s in scenes)  # cut at sentence ends
    assert all(len(s) <= 1.3 * SCENE_LENGTHS["medium"] for s in scenes)
    assert " ".join(scenes) == " ".join(sentences)  # nothing lost or reordered
    _, short = split_story(" ".join(sentences), SCENE_LENGTHS["short"])
    assert len(short) > len(scenes)


def test_tiny_paragraphs_join_the_previous_scene():
    from pencil_timelapse.story import split_story
    _, scenes = split_story(f"{PARA_1}\n\nVậy đó.\n\n{PARA_2}")
    assert scenes == [f"{PARA_1} Vậy đó.", PARA_2]


def test_scenes_without_keywords_use_the_story_subject():
    data = template_from_text("Bác sĩ Lan khám cho bệnh nhân. Bác sĩ dặn uống thuốc.\n\n"
                              "Mọi chuyện rồi cũng qua, như một giấc mơ dài.")
    assert data["scenes"][-1]["image_query"].startswith("doctor")


def test_script_validation(tmp_path):
    with pytest.raises(InputError, match="scenes"):
        StoryScript.from_dict({"title": "x", "scenes": []})
    with pytest.raises(InputError, match="image_query"):
        StoryScript.from_dict({"scenes": [{"text": "a"}]})
    with pytest.raises(InputError, match="not found"):
        StoryScript.from_dict({"scenes": [{"text": "a", "image": "missing.jpg"}]}, str(tmp_path))


def test_vietnamese_subtitles_are_split_into_short_captions():
    font = _font(40)
    text = ("Cô giáo hướng dẫn cả lớp rửa tay đúng cách: làm ướt tay, xoa xà phòng ít nhất hai "
            "mươi giây, rồi rửa sạch dưới vòi nước. Từ đó, An luôn rửa tay trước khi ăn.")
    chunks = subtitle_chunks(text, font, 600)
    assert len(chunks) > 1
    assert " ".join(chunks).split() == text.split()  # nothing lost or reordered
    from pencil_timelapse.story import wrap_text
    assert all(len(wrap_text(c, font, 600)) <= 2 for c in chunks)


def test_captions_follow_the_speech():
    chunks = ["một", "hai hai hai"]
    assert caption_at(chunks, (10.0, 14.0), 5.0) == ("", 0.0)
    assert caption_at(chunks, (10.0, 14.0), 10.5)[0] == "một"
    assert caption_at(chunks, (10.0, 14.0), 13.5)[0] == "hai hai hai"


def test_pauses_are_found():
    audio = np.concatenate([_tone(1.0), _silence(0.6), _tone(1.0), _silence(0.1), _tone(0.5)])
    pauses = find_pauses(audio)
    assert len(pauses) == 1
    assert pauses[0][0] == pytest.approx(1.0, abs=0.05) and pauses[0][1] == pytest.approx(1.6, abs=0.05)


def test_single_recording_is_split_at_natural_pauses():
    scenes = [Scene("a" * 40, "q"), Scene("b" * 40, "q")]
    script = StoryScript("", scenes)
    # The proportional split would be at 2.15 s; the real pause is at 1.8-2.3 s.
    narration = np.concatenate([_tone(1.8), _silence(0.5), _tone(2.0)])
    timings, track = plan_timing(script, [None, None], narration, offset=0.0)
    assert timings[1].speech[0] == pytest.approx(0.5 + 2.05, abs=0.06)
    assert len(track) >= len(narration)


def test_per_scene_recordings_set_scene_length():
    scenes = [Scene("a", "q"), Scene("b", "q")]
    timings, track = plan_timing(StoryScript("", scenes), [_tone(6.0), _tone(2.0)], None, 3.0)
    assert timings[0].start == 3.0
    assert timings[0].duration == pytest.approx(0.5 + 6.0 + 0.9, abs=0.01)
    assert timings[1].duration == pytest.approx(5.0)  # minimum scene length
    assert np.abs(track[int(3.6 * SAMPLE_RATE)]) > 0  # first clip plays after the title


def test_photo_to_pencil_is_a_graphite_drawing(portrait):
    photo = cv2.cvtColor(portrait["end"], cv2.COLOR_RGB2BGR)[..., ::-1].astype(np.float32) / 255
    out = photo_to_pencil(photo, (400, 240))
    assert out.shape == (240, 400, 3)
    rgb_spread = np.abs(out - out.mean(axis=2, keepdims=True)).max()
    assert rgb_spread < 0.03  # monochrome on warm paper
    assert np.allclose(out[1, 1], PAPER, atol=0.03)  # edges fade into the paper
    assert out.min() < 0.5  # real graphite tones


def test_image_search_prefers_relevant_photos_and_caches(tmp_path, monkeypatch):
    results = [
        {"title": "The boy's camp book (1914)", "url": "https://example.org/u0.jpg", "width": 3000, "height": 2000,
         "mime": "image/jpeg", "author": "Internet Archive Book Images", "licence": "PD",
         "source": "s0"},
        {"title": "Boy washing his hands", "url": "https://example.org/u1.jpg", "width": 2400, "height": 1600,
         "mime": "image/jpeg", "author": "A. Person", "licence": "CC BY 4.0", "source": "s1"},
        {"title": "Hand washing sign", "url": "https://example.org/u2.jpg", "width": 4000, "height": 2250,
         "mime": "image/jpeg", "author": "B", "licence": "CC0", "source": "s2"},
    ]
    calls = []
    monkeypatch.setattr(media, "search_commons", lambda q: calls.append(q) or results)
    monkeypatch.setattr(media, "search_openverse", lambda q: [])
    buf = cv2.imencode(".jpg", np.full((10, 10, 3), 128, np.uint8))[1].tobytes()
    monkeypatch.setattr(media, "_get", lambda url, timeout=30: buf)
    path, credit = media.find_image("boy washing hands", str(tmp_path), 16 / 9)
    assert credit.source_url == "s1" and credit.author == "A. Person"
    assert os.path.exists(path)
    again, _ = media.find_image("boy washing hands", str(tmp_path), 16 / 9)
    assert again == path and calls == ["boy washing hands"]  # served from the cache


def test_render_short_story_with_per_scene_audio(tmp_path, portrait):
    for name in ("end", "partial"):
        Image.fromarray(portrait[name]).save(tmp_path / f"{name}.png")
    write_wav(str(tmp_path / "s1.wav"), np.concatenate([_tone(1.5), _silence(0.3), _tone(1.0)]))
    write_wav(str(tmp_path / "s2.wav"), _tone(2.0, 330))
    data = {"title": "Câu chuyện thử", "scenes": [
        {"text": "Cảnh thứ nhất của câu chuyện.", "image": "end.png", "audio": "s1.wav"},
        {"text": "Cảnh thứ hai.", "image": "partial.png", "audio": "s2.wav"}],
        "width": 480, "height": 270, "fps": 12}
    (tmp_path / "story.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    script = StoryScript.load(str(tmp_path / "story.json"))
    out = tmp_path / "out" / "story.mp4"
    result = render_story(script, str(out), preset="ultrafast")
    assert out.exists() and result.credits_path is None
    assert os.listdir(out.parent) == ["story.mp4"]  # temporary files cleaned up
    # title card + two scenes (each shorter than the 5 s minimum); no credits for local images
    expected = 3.5 + 5.0 + 5.0
    assert result.duration == pytest.approx(expected, abs=0.1)
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert abs(frames - round(expected * 12)) <= 2
    from pencil_timelapse.encoder import find_ffmpeg
    import subprocess
    info = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(out)], capture_output=True,
                          text=True).stderr
    assert "Audio: aac" in info and "480x270" in info


def test_keyword_suggestions():
    from pencil_timelapse.keywords import suggest_query
    q = suggest_query("Một hôm, An bị đau bụng và phải nghỉ học. Bác sĩ khám cho An.")
    assert q.split(" | ")[0] == "stomach ache doctor"
    assert "studying" not in q  # "nghỉ học" (missing school) is not "studying"
    assert suggest_query("Các em học sinh trồng cây quanh sân trường.").startswith("children students")
    assert suggest_query("Xin chào") == ""


def test_archive_rate_limit_is_retried_then_reported(monkeypatch):
    import io
    import urllib.error
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=30):
        calls.append(req.full_url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, io.BytesIO())
        return Resp(b"ok")

    monkeypatch.setattr(media.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(media, "MIN_INTERVAL", 0.0)
    assert media._get("https://example.org/x") == b"ok" and len(calls) == 3

    calls.clear()
    monkeypatch.setattr(media.urllib.request, "urlopen",
                        lambda req, timeout=30: (_ for _ in ()).throw(
                            urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO())))
    with pytest.raises(media.RateLimited):
        media._get("https://example.org/y")


def test_english_story_gets_keywords_and_characters():
    from pencil_timelapse.keywords import character_roles, suggest_queries
    story = ("The Boy Who Was Called Stupid\n\n"
             "Daniel was twelve years old, and he was the worst student in his class. "
             "His classmates laughed at him every day.\n\n"
             "One rainy evening, Daniel found an old man sitting outside a small repair shop. "
             "A beautiful wooden clock lay broken beside him.\n\n"
             "The old man was Mr. Thomas, a retired engineer. Then Mr. Thomas smiled.\n\n"
             "[Ending – 4:15–4:50]\nDaniel thought for a moment about what intelligence means.")
    assert character_roles(story) == {"Daniel": "student", "Thomas": "old man"}
    data = template_from_text(story)
    assert data["title"] == "The Boy Who Was Called Stupid"
    assert all(s["image_query"] for s in data["scenes"])
    assert not any("Ending" in s["text"] for s in data["scenes"])
    first = [s["image_query"].split(" | ")[0] for s in data["scenes"]]
    assert first[0].startswith("student") and "old man" in first[1]
    # A scene that only says "he" continues with the last character mentioned.
    queries = suggest_queries(["Daniel was the worst student in his class. Daniel tried hard.",
                               "He thought about it for a long time."])
    assert queries[1].startswith("student thinking")
