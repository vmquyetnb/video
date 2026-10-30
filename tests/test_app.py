import io
import os
import time

import pytest

from app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "runtime"))
    app.config["TESTING"] = True
    yield app.test_client(), tmp_path / "runtime"
    app.extensions["pencil_jobs"].shutdown()


def _files(sample_files, start="blank"):
    with open(sample_files[start], "rb") as a, open(sample_files["end"], "rb") as b:
        return {"start": (io.BytesIO(a.read()), "start.png"), "end": (io.BytesIO(b.read()), "end.png")}


SMALL = {"resolution": "custom", "width": "270", "height": "480", "fps": "12", "duration": "6",
         "hold": "1", "preset": "ultrafast", "align": "true", "construction_lines": "false",
         "start_mode": "image", "show_pencil": "false"}


def test_index_page(client):
    c, _ = client
    res = c.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Pencil Timelapse Studio" in html and "Simulated effect" in html


def test_missing_file_is_reported(client, sample_files):
    c, _ = client
    data = dict(SMALL)
    data["start"] = _files(sample_files)["start"]
    res = c.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "End image is missing" in res.get_json()["error"]


def test_corrupt_upload_is_reported(client, sample_files):
    c, _ = client
    data = dict(SMALL, **_files(sample_files))
    data["start"] = (io.BytesIO(b"not an image at all"), "start.png")
    res = c.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "Start image" in res.get_json()["error"]


def test_invalid_settings_are_reported(client, sample_files):
    c, _ = client
    data = dict(SMALL, fps="5", **_files(sample_files))
    res = c.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "FPS" in res.get_json()["error"]


def test_full_job_lifecycle(client, sample_files):
    c, runtime = client
    data = dict(SMALL, **_files(sample_files, "partial"))
    res = c.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job_id = res.get_json()["id"]

    deadline = time.time() + 120
    seen = []
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}").get_json()
        seen.append(job["progress"])
        if job["state"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert job["state"] == "done", job.get("error")
    assert job["progress"] == 1.0 and seen == sorted(seen)
    assert job["info"]["frames"] == 72
    assert not os.listdir(runtime / "uploads")  # uploads cleaned up

    video = c.get(f"/api/jobs/{job_id}/video")
    assert video.status_code == 200 and video.mimetype == "video/mp4"
    body = video.get_data()
    assert body[4:8] == b"ftyp"
    ranged = c.get(f"/api/jobs/{job_id}/video", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and len(ranged.get_data()) == 100
    download = c.get(f"/api/jobs/{job_id}/download")
    assert "attachment" in download.headers["Content-Disposition"]


def test_unknown_job(client):
    c, _ = client
    assert c.get("/api/jobs/doesnotexist").status_code == 404
    assert c.get("/api/jobs/doesnotexist/video").status_code == 404


def _wait(c, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}").get_json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError("render did not finish")


def test_blank_paper_needs_only_the_end_image(client, sample_files):
    c, _ = client
    data = dict(SMALL, start_mode="blank", show_pencil="true")
    data["end"] = _files(sample_files)["end"]
    res = c.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"])
    assert job["state"] == "done", job.get("error")
    assert job["info"]["start_mode"] == "blank paper"


def test_start_image_required_in_image_mode(client, sample_files):
    c, _ = client
    data = dict(SMALL)
    data["end"] = _files(sample_files)["end"]
    res = c.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "Blank white paper" in res.get_json()["error"]


def test_all_tabs(client):
    c, _ = client
    for path in ("/", "/story", "/players", "/animals", "/travel", "/cartoon"):
        html = c.get(path).get_data(as_text=True)
        assert 'href="/story"' in html and "Vẽ chân dung" in html and "Video câu chuyện" in html
        assert 'href="/players"' in html and "Football players" in html
        assert 'href="/animals"' in html and ">Animals<" in html
        assert 'href="/travel"' in html and ">Travel<" in html
        assert 'href="/cartoon"' in html and ">Cartoon<" in html
    assert "Chia thành cảnh" in c.get("/story").get_data(as_text=True)


def test_story_is_split_into_scenes_with_keywords(client):
    c, _ = client
    text = ("Chuyện của An\n\nMột hôm, An bị đau bụng và phải nghỉ học. Bác sĩ khám cho An và "
            "giải thích rằng vi khuẩn có thể bám trên tay.\n\nTối hôm đó, cả nhà quây quần ăn "
            "cơm và An kể cho mọi người nghe về những điều bác sĩ đã dặn.")
    data = c.post("/api/story/scenes", json={"text": text}).get_json()
    assert data["title"] == "Chuyện của An"
    assert len(data["scenes"]) == 2
    assert "doctor" in data["scenes"][0]["image_query"]
    short = c.post("/api/story/scenes", json={"text": text.replace("\n\n", " "),
                                              "length": "short"}).get_json()
    assert len(short["scenes"]) >= 2
    assert c.post("/api/story/scenes", json={"text": "  "}).status_code == 400


def _wav_bytes(seconds=2.0):
    import numpy as np
    from pencil_timelapse.media import SAMPLE_RATE, write_wav
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    path = os.path.join(os.path.dirname(__file__), "_tmp_story.wav")
    write_wav(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    with open(path, "rb") as fh:
        content = fh.read()
    os.remove(path)
    return content


def test_story_video_job(client, sample_files):
    import json
    c, runtime = client
    with open(sample_files["end"], "rb") as a, open(sample_files["partial"], "rb") as b:
        pictures = (a.read(), b.read())
    script = {"title": "Thử", "scenes": [{"text": "Cảnh một.", "image_query": ""},
                                         {"text": "Cảnh hai.", "image_query": ""}]}
    data = {"script": json.dumps(script), "resolution": "480x270", "preview_voice": "false",
            "scene_0_image": (io.BytesIO(pictures[0]), "a.png"),
            "scene_1_image": (io.BytesIO(pictures[1]), "b.png"),
            "narration": (io.BytesIO(_wav_bytes(3.0)), "voice.wav")}
    res = c.post("/api/story", data=data, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=180)
    assert job["state"] == "done", job.get("error")
    assert job["kind"] == "story" and job["info"]["scenes"] == 2
    video = c.get(f"/api/jobs/{job['id']}/video")
    assert video.status_code == 200 and video.get_data()[4:8] == b"ftyp"
    assert "story-" in c.get(f"/api/jobs/{job['id']}/download").headers["Content-Disposition"]
    assert c.get(f"/api/jobs/{job['id']}/credits").status_code == 404  # own pictures: no credits
    assert not os.listdir(runtime / "uploads")


def test_story_input_errors(client):
    import json
    c, _ = client
    no_picture = {"script": json.dumps({"scenes": [{"text": "Một cảnh.", "image_query": ""}]})}
    res = c.post("/api/story", data=no_picture, content_type="multipart/form-data")
    assert res.status_code == 400 and "image" in res.get_json()["error"]
    bad_audio = {"script": json.dumps({"scenes": [{"text": "Một cảnh.", "image_query": "doctor"}]}),
                 "narration": (io.BytesIO(b"not audio"), "voice.m4a")}
    res = c.post("/api/story", data=bad_audio, content_type="multipart/form-data")
    assert res.status_code == 400 and "Narration" in res.get_json()["error"]
    res = c.post("/api/story", data={"script": "{}"}, content_type="multipart/form-data")
    assert res.status_code == 400


def test_story_image_picker(client, monkeypatch, sample_files):
    import json
    import app as app_module
    from pencil_timelapse import media
    c, _ = client
    cand = {"url": "https://example.org/doctor.jpg", "thumb": "https://example.org/t.jpg",
            "title": "Doctor measuring blood pressure", "author": "A. Person",
            "licence": "CC BY 4.0", "source": "https://example.org/page", "mime": "image/jpeg",
            "width": 2000, "height": 1300}
    monkeypatch.setattr(app_module, "search_candidates", lambda q, aspect, limit=8: ([cand], []))
    data = c.get("/api/story/images?q=doctor").get_json()
    assert data["candidates"][0]["thumb"] == "https://example.org/t.jpg"
    assert c.get("/api/story/images?q=").status_code == 400

    with open(sample_files["end"], "rb") as fh:
        picture = fh.read()
    monkeypatch.setattr(media, "_get", lambda url, timeout=30: picture)
    script = {"title": "", "scenes": [{"text": "Bác sĩ đo huyết áp.", "image_query": "doctor",
                                       "image_choice": cand}]}
    res = c.post("/api/story", data={"script": json.dumps(script), "resolution": "480x270",
                                     "preview_voice": "false"},
                 content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=180)
    assert job["state"] == "done", job.get("error")
    assert job["info"]["credits"][0].startswith("Doctor measuring blood pressure - A. Person")
    credits = c.get(f"/api/jobs/{job['id']}/credits")
    assert credits.status_code == 200 and b"https://example.org/page" in credits.get_data()


PLAYER = {"name": "Cristiano Ronaldo", "qid": "Q11571", "image_file": "CR.jpg", "fame": 190,
          "thumb": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/CR.jpg/240px-CR.jpg",
          "photo_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/CR.jpg/1280px-CR.jpg",
          "author": "A. Photographer", "licence": "CC BY-SA 4.0",
          "source": "https://commons.wikimedia.org/wiki/File:CR.jpg"}


def test_players_page_is_in_english(client):
    c, _ = client
    html = c.get("/players").get_data(as_text=True)
    assert "Football players in pencil" in html and "Find players" in html
    assert '<option value="Portugal">' in html and "Simulated drawing" in html


def test_players_search(client, monkeypatch):
    import app as app_module
    from pencil_timelapse.media import RateLimited
    from pencil_timelapse.players import Player
    c, _ = client
    found = [Player(**PLAYER), Player("No Photo", "Q2", "x.jpg", 3)]
    monkeypatch.setattr(app_module, "find_players", lambda country, limit=20: found)
    data = c.get("/api/players?country=Portugal").get_json()
    assert [p["name"] for p in data["players"]] == ["Cristiano Ronaldo"]  # only with a photo
    assert c.get("/api/players?country=").status_code == 400

    def limited(country, limit=20):
        raise RateLimited("slow down")

    monkeypatch.setattr(app_module, "find_players", limited)
    assert c.get("/api/players?country=Portugal").status_code == 429


def test_players_render_input_errors(client):
    c, _ = client

    def post(**overrides):
        body = {"players": [PLAYER], "title": "PORTUGAL", "resolution": "320x180", "seconds": 4}
        body.update(overrides)
        return c.post("/api/players/render", json=body)

    elsewhere = dict(PLAYER, photo_url="https://example.org/cr.jpg")
    assert post(players=[elsewhere]).status_code == 400
    assert post(players=[]).status_code == 400
    assert post(players=[PLAYER] * 13).status_code == 400
    assert post(resolution="huge").status_code == 400
    assert post(seconds=2).status_code == 400
    assert post(players=[dict(PLAYER, fame="many")]).status_code == 400


def test_players_video_job(client, monkeypatch, sample_files):
    from pencil_timelapse import players
    from pencil_timelapse.media import ImageCredit
    c, _ = client

    def fake_choose(player, cache_dir):
        return sample_files["end"], ImageCredit(player.name, player.author, player.licence,
                                                player.source, sample_files["end"]), None

    monkeypatch.setattr(players, "choose_portrait", fake_choose)
    res = c.post("/api/players/render", json={"players": [PLAYER], "title": "PORTUGAL",
                                              "resolution": "320x180", "seconds": 4,
                                              "intro": False, "highlights": False, "outro": False,
                                              "sound_effects": False})
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=180)
    assert job["state"] == "done", job.get("error")
    assert job["kind"] == "players" and job["has_image"] and job["has_credits"]
    assert job["info"]["players"] == ["Cristiano Ronaldo"]
    picture = c.get(f"/api/jobs/{job['id']}/image")
    assert picture.status_code == 200 and picture.mimetype == "image/png"
    credits = c.get(f"/api/jobs/{job['id']}/credits").get_data(as_text=True)
    assert "A. Photographer, CC BY-SA 4.0" in credits
    video = c.get(f"/api/jobs/{job['id']}/download")
    assert video.status_code == 200 and "players-" in video.headers["Content-Disposition"]


def test_players_from_uploaded_photos_with_sound(client, sample_files):
    import json
    c, _ = client
    with open(sample_files["end"], "rb") as fh:
        photo = fh.read()
    data = {"players": [{"name": "My Striker", "upload": "photo_1"}], "title": "",
            "resolution": "320x180", "seconds": 4, "background": "shade"}
    form = {"data": json.dumps(data), "photo_1": (io.BytesIO(photo), "striker.png"),
            "audio": (io.BytesIO(_wav_bytes(30.0)), "music.wav")}
    res = c.post("/api/players/render", data=form, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=240)
    assert job["state"] == "done", job.get("error")
    assert job["info"]["players"] == ["My Striker"] and not job["has_credits"]
    assert job["info"]["duration"] == 30.0  # fitted to the sound
    assert c.get(f"/api/jobs/{job['id']}/download").status_code == 200


def test_players_upload_errors(client):
    import json
    c, _ = client

    def post(data, **files):
        form = {"data": json.dumps(data)}
        form.update(files)
        return c.post("/api/players/render", data=form, content_type="multipart/form-data")

    one = {"players": [{"name": "A", "upload": "photo_1"}], "resolution": "320x180", "seconds": 4}
    assert post(one).status_code == 400  # the photo is missing
    assert post(one, photo_1=(io.BytesIO(b"not an image"), "a.jpg")).status_code == 400
    assert post(dict(one, players=[{"name": "", "upload": "photo_1"}])).status_code == 400
    assert post(dict(one, background="neon")).status_code == 400


def test_players_request_cannot_choose_a_local_file(client, monkeypatch):
    c, _ = client
    seen = []
    jobs = c.application.extensions["pencil_jobs"]
    monkeypatch.setattr(jobs, "submit_players", lambda players, *a, **k: seen.extend(players)
                        or jobs._new_job("players", {}))
    res = c.post("/api/players/render", json={"players": [dict(PLAYER, local_file="/etc/passwd")],
                                              "resolution": "320x180", "seconds": 4})
    assert res.status_code == 202 and seen[0].local_file == ""


def test_animals_page_and_video(client, sample_files):
    import json
    c, _ = client
    html = c.get("/animals").get_data(as_text=True)
    assert "Animals in pencil" in html and 'href="/players"' in html
    with open(sample_files["end"], "rb") as fh:
        photo = fh.read()
    data = {"animals": [{"name": "Lion", "facts": ["Lions live in prides."], "photo": "photo_1"}],
            "title": "Zoo", "resolution": "480x270", "seconds": 4, "intro": False, "outro": False}
    form = {"data": json.dumps(data), "photo_1": (io.BytesIO(photo), "lion.png"),
            "music": (io.BytesIO(_wav_bytes(3.0)), "music.wav")}
    res = c.post("/api/animals/render", data=form, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=240)
    assert job["state"] == "done", job.get("error")
    assert job["kind"] == "animals" and job["info"]["animals"] == ["Lion"] and job["has_image"]
    assert "animals-" in c.get(f"/api/jobs/{job['id']}/download").headers["Content-Disposition"]


def test_animals_input_errors(client):
    import json
    c, _ = client

    def post(animals, **files):
        form = {"data": json.dumps({"animals": animals, "resolution": "480x270", "seconds": 4})}
        form.update(files)
        return c.post("/api/animals/render", data=form, content_type="multipart/form-data")

    assert post([]).status_code == 400
    assert post([{"name": "Lion", "photo": "photo_1"}]).status_code == 400  # no photo sent
    assert post([{"name": "", "photo": "photo_1"}],
                photo_1=(io.BytesIO(b"x"), "a.jpg")).status_code == 400
    assert post([{"name": "Lion", "photo": "photo_1"}],
                photo_1=(io.BytesIO(b"not an image"), "a.jpg")).status_code == 400


def test_animals_with_a_recording_per_photo(client, sample_files):
    import json
    c, _ = client
    with open(sample_files["end"], "rb") as fh:
        photo = fh.read()
    data = {"animals": [{"name": "Lion", "facts": ["Lions live in prides."], "photo": "photo_1",
                         "voice": "voice_1"},
                        {"name": "Owl", "facts": ["Owls hunt at night."], "photo": "photo_2"}],
            "resolution": "480x270", "seconds": 4, "intro": False, "outro": False}
    form = {"data": json.dumps(data), "photo_1": (io.BytesIO(photo), "lion.png"),
            "photo_2": (io.BytesIO(photo), "owl.png"),
            "voice_1": (io.BytesIO(_wav_bytes(12.0)), "lion.wav")}
    res = c.post("/api/animals/render", data=form, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=240)
    assert job["state"] == "done", job.get("error")
    from pencil_timelapse import animals as AN
    # Lion: the 12 s recording from the first stroke, a pause, the sweep; Owl: the drawing
    # and its name, or its fact if that takes longer, then the sweep; then the 3 s gallery.
    lion = 12.0 + AN.END_HOLD + 0.8
    owl = max(4 + AN.NAME_TIME,
              0.8 + AN.reading_seconds("Owls hunt at night.") + AN.END_HOLD) + 0.8
    assert job["info"]["duration"] == pytest.approx(lion + owl + 3.0, abs=0.3)
    bad = dict(form, data=json.dumps(data), photo_1=(io.BytesIO(photo), "lion.png"),
               photo_2=(io.BytesIO(photo), "owl.png"), voice_1=(io.BytesIO(b"noise"), "lion.m4a"))
    assert c.post("/api/animals/render", data=bad,
                  content_type="multipart/form-data").status_code == 400


def test_travel_script_is_written_by_ai(client, monkeypatch):
    import app as app_module
    c, _ = client
    html = c.get("/travel").get_data(as_text=True)
    assert "Travel in pencil" in html and "Write the script" in html and "Pexels" in html
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = c.post("/api/travel/write", json={"provider": "gemini", "theme": "Wonders"})
    assert res.status_code == 400 and "API key" in res.get_json()["error"]
    asked = {}

    def fake(provider, key, places, theme, count, facts, model):
        asked.update(provider=provider, key=key, places=places, theme=theme, count=count)
        return {"title": "Wonders", "hook": "WONDERS", "question": "Next?", "places": [
            {"name": "Petra", "location": "Jordan", "country": "Jordan", "lat": 30.3, "lon": 35.4,
             "facts": ["Carved into rock."], "search": "Petra"}]}

    monkeypatch.setattr(app_module.ai_writer, "write_script", fake)
    res = c.post("/api/travel/write", json={"provider": "openai", "key": "sk", "places": "Petra\nRome",
                                            "count": 2})
    assert res.status_code == 200 and res.get_json()["places"][0]["name"] == "Petra"
    assert asked["places"] == ["Petra", "Rome"] and asked["key"] == "sk"

    def broken(*args):
        raise app_module.ai_writer.AIError("Gemini did not accept the API key.")

    monkeypatch.setattr(app_module.ai_writer, "write_script", broken)
    res = c.post("/api/travel/write", json={"provider": "gemini", "key": "bad", "theme": "x"})
    assert res.status_code == 502 and "did not accept" in res.get_json()["error"]


def test_travel_media_search(client, monkeypatch):
    import app as app_module
    from pencil_timelapse.sources import Media
    c, _ = client
    seen = {}

    def fake(query, keys, videos=True):
        seen.update(query=query, keys=keys)
        return [Media("photo", "https://live.staticflickr.com/1/a.jpg", "t", 1024, 768, "Tower",
                      "Ann", "CC BY 2.0", "https://flickr.com/x", "Flickr")], []

    monkeypatch.setattr(app_module, "search_media", fake)
    res = c.post("/api/travel/media", json={"q": "Eiffel Tower", "keys": {"pexels": "PK"}})
    assert res.status_code == 200 and res.get_json()["media"][0]["provider"] == "Flickr"
    assert seen["keys"].get("pexels") == "PK"
    assert c.post("/api/travel/media", json={"q": " "}).status_code == 400
    monkeypatch.setattr(app_module, "search_media", lambda q, k, videos=True: ([], ["Openverse: down"]))
    assert c.post("/api/travel/media", json={"q": "x"}).status_code == 502


def test_travel_video_job(client, sample_files):
    import json
    c, _ = client
    with open(sample_files["end"], "rb") as fh:
        photo = fh.read()
    data = {"places": [{"name": "Tower", "facts": ["It is 330 metres tall."], "location": "Paris, France",
                        "lat": 48.86, "lon": 2.29, "upload": "picture_1"}],
            "title": "Trip", "resolution": "480x270", "seconds": 4, "reveal": 2,
            "intro": True, "outro": True}
    form = {"data": json.dumps(data), "picture_1": (io.BytesIO(photo), "tower.png")}
    res = c.post("/api/travel/render", data=form, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=300)
    assert job["state"] == "done", job.get("error")
    assert job["kind"] == "travel" and job["info"]["places"] == ["Tower"] and job["has_image"]
    assert "travel-" in c.get(f"/api/jobs/{job['id']}/download").headers["Content-Disposition"]
    assert "travel-" in c.get(f"/api/jobs/{job['id']}/image").headers["Content-Disposition"]


def test_travel_input_errors(client):
    import json
    c, _ = client

    def post(places, **extra):
        form = {"data": json.dumps(dict({"places": places, "resolution": "480x270", "seconds": 4},
                                        **extra))}
        return c.post("/api/travel/render", data=form, content_type="multipart/form-data")

    assert post([]).status_code == 400
    assert post([{"name": "Tower"}]).status_code == 400  # no picture
    foreign = {"kind": "photo", "url": "https://example.com/tower.jpg"}
    res = post([{"name": "Tower", "media": foreign}])
    assert res.status_code == 400 and "supported library" in res.get_json()["error"]
    flickr = {"kind": "photo", "url": "https://live.staticflickr.com/1/a.jpg"}
    assert post([{"name": "Tower", "media": flickr}], reveal=40).status_code == 400
    bad_clip = {"data": json.dumps({"places": [{"name": "Tower", "upload": "picture_1"}],
                                    "resolution": "480x270", "seconds": 4}),
                "picture_1": (io.BytesIO(b"not a video"), "tower.mp4")}
    assert c.post("/api/travel/render", data=bad_clip,
                  content_type="multipart/form-data").status_code == 400


def _cartoon_picture():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (240, 300), (255, 255, 255))
    ImageDraw.Draw(img).ellipse([40, 40, 200, 280], fill=(240, 140, 60), outline=(40, 30, 50), width=5)
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def test_cartoon_page_cutout_and_story(client, monkeypatch):
    import app as app_module
    c, _ = client
    html = c.get("/cartoon").get_data(as_text=True)
    assert "Cartoon stories" in html and "Use the sample story" in html and "underwater" in html
    res = c.post("/api/cartoon/cutout", data={"picture": (io.BytesIO(_cartoon_picture()), "pip.png")},
                 content_type="multipart/form-data")
    assert res.status_code == 200 and res.mimetype == "image/png"
    from PIL import Image
    cut = Image.open(io.BytesIO(res.data))
    assert cut.mode == "RGBA" and cut.getpixel((2, 2))[3] == 0
    assert c.post("/api/cartoon/cutout", data={}, content_type="multipart/form-data").status_code == 400
    sample = c.post("/api/cartoon/story", json={"sample": True, "name": "Momo"}).get_json()
    assert "Momo" in sample["title"] and len(sample["scenes"]) >= 5
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = c.post("/api/cartoon/story", json={"provider": "gemini", "name": "Momo"})
    assert res.status_code == 400 and "API key" in res.get_json()["error"]
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return {"title": "T", "question": "", "scenes": [{"setting": "beach"}]}

    monkeypatch.setattr(app_module.ai_writer, "write_story", fake)
    assert c.post("/api/cartoon/story", json={"provider": "gemini", "key": "k"}).get_json()["title"] == "T"
    assert "Bobo" in seen["about"] and "sprout" not in seen["about"]  # the friends, but not Peeko
    c.post("/api/cartoon/story", json={"provider": "gemini", "key": "k", "peeko": True})
    assert "sprout" in seen["about"] and "Lulu" in seen["about"]  # the AI is told who everyone is


def test_peeko_and_episodes(client):
    c, _ = client
    html = c.get("/cartoon").get_data(as_text=True)
    assert "Peeko" in html and "Ready-made episodes" in html and "Brush, Brush, Brush!" in html
    res = c.get("/api/cartoon/peeko.png")
    assert res.status_code == 200 and res.mimetype == "image/png" and len(res.data) > 10000
    res = c.get("/api/cartoon/cast.png")
    assert res.status_code == 200 and res.mimetype == "image/png" and len(res.data) > 10000
    assert "Sunny Garden" in html and "1. Hello, Sunny Garden!" in html
    season = c.post("/api/cartoon/story", json={"episode": "s1e04-lulus-missing-colors", "peeko": True}).get_json()
    assert season["hook"] and season["episode"] == 4 and season["season"] == 12 and season["next"]
    assert any(ln["who"] == "Lulu" for sc in season["scenes"] for ln in sc["lines"])
    story = c.post("/api/cartoon/story", json={"episode": "counting-apples", "peeko": True}).get_json()
    assert story["title"] == "Peeko Counts Apples" and story["lesson"]
    assert any(s["count"] == 5 and s["item"] == "apple" for s in story["scenes"])
    renamed = c.post("/api/cartoon/story", json={"episode": "counting-apples", "name": "Momo"}).get_json()
    assert "Momo" in renamed["title"] and "Peeko" not in str(renamed)
    assert c.post("/api/cartoon/story", json={"episode": "nope"}).status_code == 404


def test_cartoon_video_job(client):
    import json
    c, _ = client
    story = {"title": "Pip", "scenes": [{"setting": "beach", "time": "sunset", "narration": "Pip ran.",
                                         "actions": ["run", "jump"], "say": "Yay!", "effect": "hearts"}]}
    data = {"name": "Pip", "faces": "front", "story": story, "resolution": "480x270", "intro": False,
            "outro": False}
    form = {"data": json.dumps(data), "picture": (io.BytesIO(_cartoon_picture()), "pip.png")}
    res = c.post("/api/cartoon/render", data=form, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=300)
    assert job["state"] == "done", job.get("error")
    assert job["kind"] == "cartoon" and job["info"]["scenes"] == 1 and job["has_image"]
    assert "cartoon-" in c.get(f"/api/jobs/{job['id']}/download").headers["Content-Disposition"]


def test_peeko_video_job(client):
    import json
    c, _ = client
    story = {"title": "Count", "lesson": "Count with me.", "scenes": [
        {"setting": "room", "narration": "Let's count.", "actions": ["point"], "say": "Three!", "board": "Count!",
         "item": "star", "count": 3, "holds": "book", "friend": "bunny"},
        {"setting": "kitchen", "narration": "", "cast": "Bobo", "lines": "Bobo: Snack time! (eat)\nYes!"}]}
    data = {"character": "peeko", "story": story, "resolution": "480x270", "intro": False, "outro": False,
            "series": False}
    res = c.post("/api/cartoon/render", data={"data": json.dumps(data)}, content_type="multipart/form-data")
    assert res.status_code == 202, res.get_json()
    job = _wait(c, res.get_json()["id"], timeout=300)
    assert job["state"] == "done", job.get("error")
    assert job["config"]["character"] == "peeko" and job["config"]["name"] == "Peeko"


def test_cartoon_input_errors(client):
    import json
    c, _ = client
    good = {"setting": "beach", "narration": "Hi.", "actions": ["run"]}

    def post(picture=True, **data):
        body = dict({"story": {"scenes": [good]}, "resolution": "480x270"}, **data)
        form = {"data": json.dumps(body)}
        if picture:
            form["picture"] = (io.BytesIO(_cartoon_picture()), "pip.png")
        return c.post("/api/cartoon/render", data=form, content_type="multipart/form-data")

    assert post(picture=False).status_code == 400
    assert post(picture=False, character="peeko", story={"scenes": []}).status_code == 400
    assert post(story={"scenes": []}).status_code == 400
    assert post(story={"scenes": [{"setting": "beach"}]}).status_code == 400  # nothing to tell
    assert post(faces="up").status_code == 400
    assert post(pitch=5).status_code == 400
    assert post(narrator="Nobody At All").status_code in (400, 202)
