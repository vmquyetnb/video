import io
import json
import os
import subprocess
import urllib.error

import numpy as np
import pytest

from pencil_timelapse import ai_writer as AI
from pencil_timelapse import sources as SR
from pencil_timelapse import travel as T
from pencil_timelapse import worldmap as WM
from pencil_timelapse.animals import END_HOLD, NAME_TIME, reading_seconds
from pencil_timelapse.encoder import find_ffmpeg
from pencil_timelapse.errors import InputError
from pencil_timelapse.showtime import ShowOptions

PLAIN = ShowOptions(intro=False, highlights=True, outro=False, sound_effects=False)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A 6 s moving test pattern."""
    path = str(tmp_path_factory.mktemp("clip") / "clip.mp4")
    subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=6", "-pix_fmt", "yuv420p", path],
                   check=True)
    return path


class _Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _answers(monkeypatch, module, replies, seen=None):
    """Answer ``urlopen`` calls by URL substring with JSON (or an HTTP error code)."""
    def fake(req, timeout=0):
        url = req.full_url if hasattr(req, "full_url") else req
        if seen is not None:
            seen.append(req)
        for part, reply in replies.items():
            if part in url:
                if isinstance(reply, int):
                    raise urllib.error.HTTPError(url, reply, "error", {},
                                                 io.BytesIO(b'{"error": {"message": "nope"}}'))
                return _Answer(json.dumps(reply).encode() if not isinstance(reply, bytes) else reply)
        raise AssertionError(f"unexpected request {url}")
    monkeypatch.setattr(module.urllib.request, "urlopen", fake)


# --------------------------------------------------------------------------- the map

def test_places_are_on_the_map_and_clear_of_the_headline():
    paris, rome, madrid = (2.29, 48.86), (12.49, 41.89), (-3.7, 40.4)
    view = WM.fit_view([paris, rome, madrid], 16 / 9, top=0.3, bottom=0.12)
    for lon, lat in (paris, rome, madrid):
        x, y = view.xy(lon, lat, 1920, 1080)
        assert 0 < x < 1920 and 0.3 * 1080 < y < 0.88 * 1080
    assert view.east - view.west < 120  # a close view of Europe, not the world
    # Sydney, San Francisco and Paris: a view across the Pacific, over the date line.
    pacific = WM.fit_view([paris, (151.2, -33.9), (-122.4, 37.8)], 16 / 9)
    for lon, lat in (paris, (151.2, -33.9), (-122.4, 37.8), (178.0, -17.0)):
        x, y = pacific.xy(lon, lat, 1920, 1080)
        assert 0 < x < 1920 and 0 < y < 1080
    # Places all around the world: the whole world.
    spread = [paris, (139.7, 35.7), (-122.4, 37.8), (-43.2, -22.9), (151.2, -33.9), (18.4, -33.9)]
    assert WM.fit_view(spread, 16 / 9) == WM.world_view(16 / 9)


def test_pencil_map_shades_the_land_only():
    view = WM.MapView(-20.0, 0.0, 40.0, 40.0)  # Sahara and the Atlantic
    img = WM.pencil_map(view, 300, 200, seed=1)
    assert img.shape == (200, 300, 3)
    land = img[100:180, 150:250].mean()  # Africa
    sea = img[100:180, 5:40].mean()  # the Atlantic off West Africa
    assert land < sea - 0.03


def test_land_mask_is_built_from_polygons(tmp_path):
    square = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
        [[0, 0], [90, 0], [90, 45], [0, 45], [0, 0]]]}}
    src = tmp_path / "land.geojson"
    src.write_text(json.dumps({"type": "FeatureCollection", "features": [square]}))
    out = str(tmp_path / "land.png")
    WM.build_land_mask(str(src), out, width=360)
    import cv2
    mask = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (180, 360) and mask[60, 220] == 255 and mask[60, 100] == 0


def test_plane_follows_its_path():
    path = WM.arc((0, 100), (200, 100), bend=0.2)
    part, head, angle = WM.along(path, 0.5)
    assert head[1] < 100 and abs(head[0] - 100) < 5  # over the top of the arc
    assert abs(angle) < 5  # flying level, to the right
    assert WM.along(path, 1.0)[1][0] == pytest.approx(200, abs=0.5)


# --------------------------------------------------------------------------- photo libraries

def test_openverse_results_skip_wikimedia_drawings_and_repeats(monkeypatch):
    seen = []

    def r(title, author, url="https://live.staticflickr.com/1/a_b.jpg"):
        return {"title": title, "creator": author, "url": url, "thumbnail": url, "width": 1024,
                "height": 768, "license": "by", "license_version": "2.0", "source": "flickr",
                "foreign_landing_url": "https://www.flickr.com/photos/x/1"}

    _answers(monkeypatch, SR, {"api.openverse.org": {"results": [
        r("Eiffel Tower", "ann"), r("Eiffel Tower 2", "ann"), r("Eiffel Tower 3", "ann"),
        r("Map of Paris", "bob"), r("Tower", "cy", url="https://example.com/x.jpg"),
        r("Night tower", "dee")]}}, seen)
    found = SR.search_openverse("Eiffel Tower")
    assert [m.title for m in found] == ["Eiffel Tower", "Eiffel Tower 2", "Night tower"]
    assert found[0].licence == "CC BY 2.0" and found[0].provider == "Flickr"
    url = seen[0].full_url
    assert "excluded_source=wikimedia" in url and "license_type=commercial%2Cmodification" in url
    assert "page_size=20" in url


def test_pexels_and_pixabay_give_photos_and_clips(monkeypatch):
    seen = []
    _answers(monkeypatch, SR, {
        "api.pexels.com/v1/": {"photos": [{"width": 4000, "height": 2600, "alt": "Tower at dusk",
                                           "photographer": "Pat", "url": "https://www.pexels.com/photo/1/",
                                           "src": {"large2x": "https://images.pexels.com/p/1.jpg?w=1880",
                                                   "medium": "https://images.pexels.com/p/1.jpg?h=350"}}]},
        "api.pexels.com/videos/": {"videos": [{
            "url": "https://www.pexels.com/video/aerial-view-of-paris-1234/", "image": "https://images.pexels.com/v.jpg",
            "duration": 14, "user": {"name": "Vid"}, "video_files": [
                {"file_type": "video/mp4", "width": 3840, "height": 2160, "link": "https://videos.pexels.com/4k.mp4"},
                {"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "https://videos.pexels.com/hd.mp4"},
                {"file_type": "video/mp4", "width": 640, "height": 360, "link": "https://videos.pexels.com/sd.mp4"}]}]},
        "pixabay.com/api/videos/": {"hits": [{"pageURL": "https://pixabay.com/videos/x", "tags": "paris",
                                              "duration": 2, "user": "Short", "videos": {
                                                  "medium": {"url": "https://cdn.pixabay.com/v.mp4", "width": 1280,
                                                             "height": 720, "thumbnail": "https://cdn.pixabay.com/t.jpg"}}}]},
        "pixabay.com/api/": {"hits": [{"largeImageURL": "https://pixabay.com/get/abc.jpg",
                                       "webformatURL": "https://cdn.pixabay.com/w.jpg", "imageWidth": 5000,
                                       "imageHeight": 3000, "tags": "eiffel, tower", "user": "Pix",
                                       "pageURL": "https://pixabay.com/photos/1"}]},
        "api.openverse.org": 500,
    }, seen)
    found, problems = SR.search_media("Eiffel Tower", {"pexels": "PK", "pixabay": "XK"})
    kinds = [(m.kind, m.provider) for m in found]
    # Clips first; the 2 s Pixabay clip is too short; Openverse failing does not stop the rest.
    assert kinds == [("video", "Pexels"), ("photo", "Pexels"), ("photo", "Pixabay")]
    assert found[0].url == "https://videos.pexels.com/hd.mp4" and found[0].title == "Aerial view of paris"
    assert any("Openverse" in p for p in problems)
    pexels = [r for r in seen if "api.pexels.com" in r.full_url]
    assert all(r.get_header("Authorization") == "PK" for r in pexels)
    assert "Pexels License" in found[1].credit()


def test_only_known_libraries_are_downloaded(monkeypatch, tmp_path):
    with pytest.raises(InputError):
        SR.Media.from_dict({"kind": "photo", "url": "https://example.com/x.jpg"})
    with pytest.raises(InputError):
        SR.Media.from_dict({"kind": "photo", "url": "http://live.staticflickr.com/x.jpg"})
    with pytest.raises(InputError):
        SR.Media.from_dict({"kind": "photo", "url": "https://staticflickr.com.evil.net/x.jpg"})
    media = SR.Media.from_dict({"kind": "photo", "url": "https://live.staticflickr.com/1/a.jpg",
                                "title": "Tower", "author": "Ann", "licence": "CC BY 2.0",
                                "provider": "Flickr"})
    _answers(monkeypatch, SR, {"staticflickr.com": b"JPEGDATA"})
    path = SR.download(media, str(tmp_path))
    assert open(path, "rb").read() == b"JPEGDATA" and path.endswith(".jpg")
    assert SR.download(media, str(tmp_path)) == path  # cached
    assert media.credit() == "Tower - Ann (CC BY 2.0, Flickr)"


# --------------------------------------------------------------------------- AI writer

ANSWER = {"title": "Wonders", "hook": "seven wonders", "question": "Which next?", "places": [
    {"name": "Petra", "location": "Ma'an, Jordan", "country": "Jordan", "lat": 30.3285, "lon": 35.4444,
     "facts": ["Carved into rock over 2,000 years ago.", "Two.", "Three.", "Four."], "search": "Petra Treasury"},
    {"name": "Nowhere", "lat": 200, "lon": "east", "facts": []},
    {"facts": ["no name"]}]}


def test_script_is_checked_and_trimmed():
    text = "```json\n" + json.dumps(ANSWER) + "\n```"
    script = AI.parse_script(text, facts=3)
    assert script["hook"] == "SEVEN WONDERS" and len(script["places"]) == 2
    petra, nowhere = script["places"]
    assert petra["facts"] == ["Carved into rock over 2,000 years ago.", "Two.", "Three."]
    assert petra["lat"] == 30.3285 and nowhere["lat"] is None and nowhere["lon"] is None
    with pytest.raises(AI.AIError):
        AI.parse_script("Sorry, I cannot help.")


def test_gpt_writes_the_script(monkeypatch):
    seen = []
    reply = {"choices": [{"message": {"content": json.dumps(ANSWER)}}]}
    _answers(monkeypatch, AI, {"api.openai.com/v1/chat/completions": reply}, seen)
    script = AI.write_script("openai", "sk-test", places=["Petra"], facts=2)
    assert script["places"][0]["name"] == "Petra"
    req = seen[0]
    body = json.loads(req.data)
    assert req.get_header("Authorization") == "Bearer sk-test"
    assert body["model"] == AI.PROVIDERS["openai"] and body["response_format"] == {"type": "json_object"}
    assert "1. Petra" in body["messages"][1]["content"]


def test_gemini_writes_the_script(monkeypatch):
    seen = []
    reply = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                   {"text": json.dumps(ANSWER)}]}}]}
    _answers(monkeypatch, AI, {"generativelanguage.googleapis.com": reply}, seen)
    script = AI.write_script("gemini", "g-key", theme="Wonders of the Middle East", count=3,
                             model="gemini-2.5-flash")
    assert script["title"] == "Wonders"
    req = seen[0]
    assert "models/gemini-2.5-flash:generateContent" in req.full_url and "g-key" not in req.full_url
    assert req.get_header("X-goog-api-key") == "g-key"
    body = json.loads(req.data)
    assert "3 most famous" in body["contents"][0]["parts"][0]["text"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_ai_problems_are_explained(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(InputError, match="API key"):
        AI.write_script("openai", "", places=["Petra"])
    with pytest.raises(InputError):
        AI.write_script("claude", "k", places=["Petra"])
    with pytest.raises(InputError):
        AI.write_script("gemini", "k", places=["Petra"], model="bad model/../x y")
    _answers(monkeypatch, AI, {"googleapis.com": 401})
    with pytest.raises(AI.AIError, match="did not accept the API key"):
        AI.write_script("gemini", "k", places=["Petra"])
    _answers(monkeypatch, AI, {"googleapis.com": 404})
    with pytest.raises(AI.AIError, match="no model called"):
        AI.write_script("gemini", "k", places=["Petra"], model="gemini-9")


# --------------------------------------------------------------------------- the video

def test_coordinates_and_stamp():
    assert T.coordinates(48.8584, 2.2945) == "48.8584° N   2.2945° E"
    assert T.coordinates(-13.1631, -72.545) == "13.1631° S   72.5450° W"
    stamp = T.stamp_sprite("Peru", "No. 3", 120)
    assert stamp.size == (252, 120) and 0.05 < float(stamp.alpha.mean()) < 0.6


def test_places_are_drawn_then_brought_to_life(tmp_path, sample_files, clip):
    trip = [T.Place("Tower", sample_files["end"], ["It is 330 metres tall."], "Paris, France", "France",
                    48.8584, 2.2945, credit="Tower - Ann (CC BY 2.0, Flickr)"),
            T.Place("Pattern", clip, ["A clip plays when the drawing is done."], "Lima, Peru")]
    labels = []
    result = T.render_travel(trip, "Trip", str(tmp_path / "t.mp4"), 480, 270, fps=10,
                             drawing_seconds=4, show=PLAIN, reveal_seconds=2.0, preset="ultrafast",
                             progress=lambda f, m: labels.append(m))
    reveal = T.WASH + T.MELT + T.GROW + 2.0 + T.FLY
    expected = sum(max(4 + NAME_TIME, 0.8 + reading_seconds(p.facts[0]) + END_HOLD) + reveal
                   for p in trip) + 3.0
    assert result.duration == pytest.approx(expected, abs=0.3)
    assert labels.index("Pattern comes to life") > labels.index("Drawing Pattern")
    assert os.path.exists(result.image_path)
    assert "Tower - Ann (CC BY 2.0, Flickr)" in open(result.credits_path).read()
    # The clip moves while it is shown full screen: two moments of it differ.
    start = result.duration - 3.0 - reveal + T.WASH + T.MELT + T.GROW
    frames = [T.video_frame(result.output_path, start + dt) for dt in (0.2, 1.4)]
    assert float(np.abs(frames[0] - frames[1]).mean()) > 0.02


def test_travel_input_errors(tmp_path, sample_files):
    one = [T.Place("Tower", sample_files["end"])]
    with pytest.raises(InputError):
        T.render_travel([], "", str(tmp_path / "a.mp4"))
    with pytest.raises(InputError):
        T.render_travel(one, "", str(tmp_path / "a.mp4"), drawing_seconds=2)
    with pytest.raises(InputError):
        T.render_travel(one, "", str(tmp_path / "a.mp4"), reveal_seconds=30)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(InputError):
        T.video_seconds(str(bad))


def test_script_file(tmp_path, sample_files):
    script = tmp_path / "trip.json"
    script.write_text(json.dumps({"title": "Trip", "places": [
        {"name": "Tower", "media": sample_files["end"], "facts": ["Tall."], "lat": 1.5, "lon": 2}]}))
    title, places = T.load_script(str(script))
    assert title == "Trip" and places[0].where == (2.0, 1.5) and not places[0].is_video
