import io
import json
import os
import subprocess
import time

import numpy as np
import pytest

from pencil_timelapse import pets as P
from pencil_timelapse.ai_writer import AIError, parse_diary
from pencil_timelapse.encoder import find_ffmpeg
from pencil_timelapse.errors import InputError


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A 6 s moving test pattern."""
    path = str(tmp_path_factory.mktemp("clip") / "clip.mp4")
    subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=6", "-pix_fmt", "yuv420p", path], check=True)
    return path


@pytest.fixture
def quiet(monkeypatch):
    """No speech synthesiser: the pet's lines are only written (fast tests)."""
    monkeypatch.setattr(P, "pet_voice", lambda kind, voice="", pitch=0.0: (None, 1.0))


def test_time_of_day():
    assert P.hour_of("7:00 AM") == 7.0 and P.hour_of("7 pm") == 19.0 and P.hour_of("19:30") == 19.5
    assert P.hour_of("12:00 AM") == 0.0 and P.hour_of("noon") == 12.0 and P.hour_of("7am") == 7.0
    assert P.hour_of("later") is None and P.hour_of("13 PM") is None and P.hour_of("7:75") is None
    moments = [P.Moment("6:00 AM", "a", "x"), P.Moment("soon", "b", "x"), P.Moment("11 PM", "c", "x")]
    shares = P.day_shares(moments)
    assert shares[0] == 0.0 and shares[2] == 1.0 and 0.0 < shares[1] < 1.0
    assert P.is_night("11:00 PM") and not P.is_night("3:00 PM")


def test_diary_from_dict_and_best():
    d = P.Diary.from_dict({"pet": " Mochi ", "kind": "cat", "guess": "A\nB\n", "moments": [
        {"time": "7 AM", "title": "Up", "thought": "Hi"}, {"time": "9 PM", "title": "Zoom", "best": True},
        "junk"]})
    assert d.pet == "Mochi" and d.guess == ["A", "B"] and len(d.moments) == 2 and d.best() == 1
    assert d.heading() == "A Day in the Life of Mochi" and d.hook == P.HOOK
    d.moments[1].best = False
    assert d.best() == 1  # the last one when none is marked
    assert P.sample("pepper").moments[-1].best and P.sample("biscuit").guess
    with pytest.raises(InputError):
        P.sample("nobody")


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (1080, 1080), (640, 360)])
def test_layout_keeps_text_beside_the_drawing(size):
    W, H = size
    lay = P.diary_layout(W, H)
    x, y, w, h = lay.draw_box
    bx, by, bw, bh = lay.bubble
    assert 0 <= x and x + w <= W and 0 <= y and y + h <= H
    assert bh >= 0.1 * lay.unit and lay.bar_y < H
    if lay.wide:
        assert lay.x > x + w
    else:
        assert lay.counter_y > y + h and lay.time_y < lay.title_y < by
    if H > W * 1.2:  # Shorts: nothing important under its own captions and buttons
        assert by + bh <= 0.84 * H and lay.bar_y <= 0.84 * H


def test_ai_diary_is_checked():
    d = parse_diary(json.dumps({"title": "T", "guess": ["runs", "naps"], "answer": "bath", "moments": [
        {"time": "7 AM", "title": "Up", "thought": "Morning!", "best": True},
        {"time": "8 PM", "title": "Bath", "thought": "Oh no.", "best": True}, {"nothing": 1}]}))
    assert [m["best"] for m in d["moments"]] == [True, False] and d["answer"] == "bath" and "bath" in d["guess"]
    d = parse_diary('```json\n{"moments": [{"title": "Nap"}]}\n```')
    assert d["moments"][0]["best"] and d["moments"][0]["search"] == "Nap"
    with pytest.raises(AIError):
        parse_diary('{"moments": []}')
    with pytest.raises(AIError):
        parse_diary("not json")


def test_captions_follow_the_words():
    caps = P._Captions("Gone in nine seconds, a new record!", 640, 360, 1.0, 2.0, 280)
    assert caps.times[0] == 1.0 and caps.times == sorted(caps.times) and caps.times[-1] < 3.0
    frame = np.zeros((360, 640, 3), np.float32)
    caps.paint(frame, 0.5)
    assert frame.max() == 0.0  # nothing before the first word
    caps.paint(frame, 2.9)
    assert frame.max() > 0.5


def test_pet_diary_video(tmp_path, sample_files, clip, quiet):
    diary = P.Diary(pet="Biscuit", kind="dog", guess=["Runs away", "Takes a bath"], answer="Takes a bath",
                    moments=[P.Moment("7:00 AM", "Wake-up call", "The sun is up. I will stare until it works.",
                                      sample_files["end"], punch="It worked.", credit="Ann (CC BY 2.0, Flickr)"),
                             P.Moment("6:00 PM", "The B-word", "The human said a word.", clip,
                                      punch="That is how I ended up here.", best=True)])
    out = str(tmp_path / "diary.mp4")
    seen = []
    result = P.render_pets(diary, out, 480, 270, fps=10, drawing_seconds=5, reveal_seconds=1.5,
                           progress=lambda f, m: seen.append(m))
    assert os.path.getsize(out) > 10_000 and os.path.exists(result.image_path)
    assert result.moments == ["7:00 AM · Wake-up call", "6:00 PM · The B-word"]
    assert any("read aloud" in w for w in result.warnings)
    assert "The cold open" in seen and "What happens next?" in seen and "Like, subscribe, comment" in seen
    with open(result.credits_path) as fh:
        assert "Ann" in fh.read()
    assert 30 < result.duration < 90
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".pets-")]


def test_pet_diary_portrait_without_extras(tmp_path, sample_files, quiet):
    diary = P.Diary(pet="Pepper", kind="cat", moments=[P.Moment("noon", "Box day", "", sample_files["end"])])
    out = str(tmp_path / "short.mp4")
    seen = []
    result = P.render_pets(diary, out, 270, 480, fps=10, drawing_seconds=5, reveal_seconds=1.5, tune=None,
                           guess=False, cold_open=False, outro=False, sound_effects=False,
                           progress=lambda f, m: seen.append(m))
    assert os.path.exists(out) and result.credits_path is None
    assert "The cold open" not in seen and "What happens next?" not in seen


def test_pet_diary_input_errors(tmp_path, sample_files):
    ok = [P.Moment("7 AM", "Up", "Hi", sample_files["end"])]
    with pytest.raises(InputError):
        P.render_pets(P.Diary(pet="", moments=ok), str(tmp_path / "a.mp4"))
    with pytest.raises(InputError):
        P.render_pets(P.Diary(pet="Rex"), str(tmp_path / "a.mp4"))
    with pytest.raises(InputError):
        P.render_pets(P.Diary(pet="Rex", moments=[P.Moment("7 AM", "Up", "Hi")]), str(tmp_path / "a.mp4"))
    with pytest.raises(InputError):
        P.render_pets(P.Diary(pet="Rex", moments=ok), str(tmp_path / "a.mp4"), drawing_seconds=2)
    with pytest.raises(InputError):
        P.render_pets(P.Diary(pet="Rex", moments=ok * 13), str(tmp_path / "a.mp4"))


def test_find_media_fills_missing_pictures(tmp_path, monkeypatch):
    from pencil_timelapse import sources
    from pencil_timelapse.sources import Media
    asked = []

    def fake(query, keys, videos=True):
        asked.append(query)
        return [Media("photo", f"https://live.staticflickr.com/1/{len(asked)}.jpg", "t", 1024, 768, "Dog", "Ann",
                      "CC BY 2.0", "https://flickr.com/x", "Flickr")], []

    monkeypatch.setattr(sources, "search_media", fake)
    monkeypatch.setattr(sources, "download", lambda media, cache: os.path.join(cache, media.url[-5:]))
    diary = P.sample("biscuit")
    diary.moments[0].media = "mine.mp4"
    problems = P.find_media(diary, {}, str(tmp_path))
    assert not problems and len(asked) == len(diary.moments) - 1 and diary.moments[0].media == "mine.mp4"
    assert all(m.media for m in diary.moments) and "Ann" in diary.moments[1].credit


# --------------------------------------------------------------------------- the web page

@pytest.fixture
def client(tmp_path):
    from app import create_app
    app = create_app(str(tmp_path / "runtime"))
    app.config["TESTING"] = True
    yield app.test_client()


def _wait(c, job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}").get_json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError("render did not finish")


def test_pets_page_and_ai(client, monkeypatch):
    import app as app_module
    html = client.get("/pets").get_data(as_text=True)
    assert "Pet diary in pencil" in html and "Load sample: Biscuit" in html and "What happens next?" in html
    assert 'href="/pets"' in client.get("/travel").get_data(as_text=True)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = client.post("/api/pets/write", json={"provider": "gemini", "pet": "Rex"})
    assert res.status_code == 400 and "API key" in res.get_json()["error"]
    asked = {}

    def fake(provider, key, pet, kind, idea, count, model):
        asked.update(pet=pet, kind=kind, count=count)
        return {"title": "Rex", "moments": [{"time": "7 AM", "title": "Up", "thought": "Hi", "best": True}]}

    monkeypatch.setattr(app_module.ai_writer, "write_diary", fake)
    res = client.post("/api/pets/write", json={"provider": "openai", "key": "sk", "pet": "Rex", "kind": "cat",
                                               "count": 4})
    assert res.status_code == 200 and res.get_json()["moments"][0]["title"] == "Up"
    assert asked == {"pet": "Rex", "kind": "cat", "count": 4}


def test_pets_video_job(client, sample_files, quiet):
    with open(sample_files["end"], "rb") as fh:
        photo = fh.read()
    data = {"pet": "Biscuit", "kind": "dog", "resolution": "480x270", "seconds": 5, "reveal": 1.5,
            "guess": "Runs away\nTakes a bath", "answer": "Takes a bath", "tune": "calm",
            "moments": [{"time": "7 AM", "title": "Up", "thought": "Morning!", "punch": "Yes.", "best": True,
                         "upload": "picture_1"}]}
    form = {"data": json.dumps(data), "picture_1": (io.BytesIO(photo), "dog.png")}
    res = client.post("/api/pets/render", data=form, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(client, res.get_json()["id"])
    assert job["state"] == "done", job.get("error")
    assert job["kind"] == "pets" and job["info"]["moments"] == ["7 AM · Up"] and job["has_image"]
    assert "pet-diary-" in client.get(f"/api/jobs/{job['id']}/download").headers["Content-Disposition"]


def test_pets_input_errors(client):
    def post(data):
        return client.post("/api/pets/render", data={"data": json.dumps(data)}, content_type="multipart/form-data")

    one = [{"time": "7 AM", "title": "Up", "thought": "Hi", "media": None}]
    assert "between 1 and 12" in post({"pet": "Rex", "moments": []}).get_json()["error"]
    assert "name" in post({"pet": " ", "moments": one}).get_json()["error"]
    assert "Choose a photo" in post({"pet": "Rex", "moments": one}).get_json()["error"]
    bad = [{"time": "7 AM", "title": "", "thought": ""}]
    assert "title or a thought" in post({"pet": "Rex", "moments": bad}).get_json()["error"]
    media = {"kind": "photo", "url": "https://example.com/a.jpg"}
    assert "supported library" in post({"pet": "Rex", "moments": [dict(one[0], media=media)]}).get_json()["error"]
