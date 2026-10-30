import json
import os
import urllib.parse

import cv2
import numpy as np
import pytest

from pencil_timelapse import players as P
from pencil_timelapse.errors import InputError
from pencil_timelapse.media import ImageCredit
from pencil_timelapse.showtime import ShowOptions

PLAIN = ShowOptions(intro=False, highlights=False, outro=False, sound_effects=False)


def _fake_wikimedia(monkeypatch, responses):
    """Answer ``players._get`` from {url fragment: JSON-able value}."""
    calls = []

    def fake_get(url, timeout=30.0, retries=4):
        calls.append(url)
        for fragment, value in responses.items():
            if fragment in url:
                return json.dumps(value).encode()
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(P, "_get", fake_get)
    return calls


def test_find_team_picks_the_senior_mens_team(monkeypatch):
    _fake_wikimedia(monkeypatch, {"wbsearchentities": {"search": [
        {"id": "Q1", "label": "Portugal women's national football team",
         "description": "women's national association football team representing Portugal"},
        {"id": "Q2", "label": "Portugal national under-21 football team",
         "description": "national association football team"},
        {"id": "Q267245", "label": "Portugal men's national football team",
         "description": "men's national association football team representing Portugal"},
    ]}})
    assert P.find_team("Portugal") == ("Q267245", "Portugal men's national football team")


def test_unknown_country_is_reported(monkeypatch):
    _fake_wikimedia(monkeypatch, {"wbsearchentities": {"search": []}})
    with pytest.raises(InputError, match="Atlantis"):
        P.find_team("Atlantis")


def test_find_players_reads_wikidata_and_commons(monkeypatch):
    calls = _fake_wikimedia(monkeypatch, {
        "wbsearchentities": {"search": [{"id": "Q267245", "label": "Portugal",
                                         "description": "men's national football team"}]},
        "query.wikidata.org": {"results": {"bindings": [
            {"p": {"value": "http://www.wikidata.org/entity/Q11571"},
             "pLabel": {"value": "Cristiano Ronaldo"},
             "image": {"value": "http://commons.wikimedia.org/wiki/Special:FilePath/CR%207.jpg"},
             "fame": {"value": "190"}},
            {"p": {"value": "http://www.wikidata.org/entity/Q134183"},
             "pLabel": {"value": "Eusébio"},
             "image": {"value": "http://commons.wikimedia.org/wiki/Special:FilePath/Eusebio.jpg"},
             "fame": {"value": "120"}},
        ]}},
        "commons.wikimedia.org/w/api.php": {"query": {"pages": {
            "1": {"title": "File:CR 7.jpg", "imageinfo": [{
                "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/CR_7.jpg/1280px-CR_7.jpg",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:CR_7.jpg",
                "extmetadata": {"Artist": {"value": "<a href='x'>A. Photographer</a>"},
                                "LicenseShortName": {"value": "CC BY-SA 4.0"}}}]},
            "-1": {"title": "File:Eusebio.jpg", "missing": ""},
        }}},
    })
    found = P.find_players("Portugal", limit=5)
    assert [p.name for p in found] == ["Cristiano Ronaldo", "Eusébio"]
    cr = found[0]
    assert cr.qid == "Q11571" and cr.fame == 190 and cr.image_file == "CR 7.jpg"
    assert cr.photo_url.endswith("/1280px-CR_7.jpg") and cr.thumb.endswith("/240px-CR_7.jpg")
    assert cr.author == "A. Photographer" and cr.licence == "CC BY-SA 4.0"
    assert found[1].photo_url == ""  # no photo information: left out by the web app
    query = urllib.parse.unquote_plus(calls[1])
    assert "p:P54/ps:P54 wd:Q267245" in query and "LIMIT 5" in query


def test_thumbnail_urls_for_scoring():
    base = "https://upload.wikimedia.org/wikipedia/commons/"
    assert P._small(base + "thumb/2/26/A.jpg/1280px-A.jpg") == base + "thumb/2/26/A.jpg/500px-A.jpg"
    assert P._small(base + "1/15/Big.jpg?x=1", 3000) == base + "thumb/1/15/Big.jpg/500px-Big.jpg"
    assert P._small(base + "5/55/Small.jpg", 0) == base + "5/55/Small.jpg"  # size unknown: as is


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (1080, 1080)])
def test_layout_fits_the_frame_without_overlaps(size):
    W, H = size
    for count in range(1, 13):
        layout = P.make_layout(count, W, H)
        assert len(layout.cells) == len(layout.name_boxes) == count
        boxes = []
        for (x, y, w, h), (nx, ny, nw, nh) in zip(layout.cells, layout.name_boxes):
            assert 0 <= x and x + w <= W and layout.title_box[3] <= y and y + h <= H
            assert abs(w / h - 0.8) < 0.02 and w % 2 == 0 and h % 2 == 0
            assert ny == y + h and ny + nh <= H  # the name sits under its portrait
            boxes.append((x, y, w, h))
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
                        or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def test_compose_adds_the_portraits_one_by_one():
    layout = P.make_layout(3, 960, 540)
    drawings = [np.full((h, w, 3), 0.3, np.float32) for _, _, w, h in layout.cells]
    names = ["Player One", "Player Two", "Éder"]

    def mean_in(page, i):
        x, y, w, h = layout.cells[i]
        return float(page[y:y + h, x:x + w].mean())

    blank = P.compose(layout, drawings, names, "PORTUGAL", 0, show_title=False)
    first = P.compose(layout, drawings, names, "PORTUGAL", 1)
    full = P.compose(layout, drawings, names, "PORTUGAL", 3)
    assert blank.shape == (540, 960, 3) and blank.min() > 0.85  # only paper
    title = slice(layout.title_box[1], layout.title_box[1] + layout.title_box[3])
    assert first[title].min() < 0.5 < blank[title].min()  # the title is written
    assert mean_in(first, 0) < 0.4 and mean_in(first, 1) > 0.85
    assert all(mean_in(full, i) < 0.4 for i in range(3))
    x, y, w, h = layout.name_boxes[2]
    assert full[y:y + h, x:x + w].min() < 0.5 < first[y:y + h, x:x + w].min()


SKIN, HAIR, SHIRT = (0.85, 0.65, 0.5), (0.25, 0.15, 0.08), (0.15, 0.15, 0.2)


def _studio_photo(h=600, w=480, face=(180, 160, 120, 150)):
    """Head (with hair and ears) and shoulders in front of a busy striped background."""
    img = np.zeros((h, w, 3), np.float32)
    img[:, :] = (0.8, 0.1, 0.1)
    img[:, ::16] = (0.1, 0.1, 0.9)  # stripes: a busy background
    fx, fy, fw, fh = face
    cx, cy = fx + fw // 2, fy + fh // 2
    cv2.rectangle(img, (cx - 150, fy + fh + 30), (cx + 150, h), SHIRT, -1)
    cv2.rectangle(img, (cx - 28, cy), (cx + 28, fy + fh + 40), SKIN, -1)  # neck
    cv2.ellipse(img, (cx, cy - fh // 4), (fw // 2 + 14, fh // 2 + 10), 0, 180, 360, HAIR, -1)
    for side in (-1, 1):  # ears
        cv2.ellipse(img, (cx + side * (fw // 2 + 6), cy), (12, 22), 0, 0, 360, SKIN, -1)
    cv2.ellipse(img, (cx, cy), (fw // 2 + 8, fh // 2 + 12), 0, 0, 360, SKIN, -1)
    return img


def _count(img, colour, tol=0.05, inner=0):
    """Pixels of ``colour``; with ``inner``, only those at least that far from its edge."""
    mask = (np.abs(img - np.array(colour, np.float32)).max(axis=2) < tol).astype(np.uint8)
    if inner:
        mask = cv2.erode(mask, np.ones((2 * inner + 1, 2 * inner + 1), np.uint8))
    return int(mask.sum())


def _rel(face, img):
    h, w = img.shape[:2]
    return (face[0] / w, face[1] / h, face[2] / w, face[3] / h)


def test_shaded_background_keeps_the_whole_head():
    img = _studio_photo()
    out = P.prepare_portrait(img, 0.8, _rel((180, 160, 120, 150), img), "shade")
    h, w = out.shape[:2]
    assert abs(w / h - 0.8) < 0.01
    # Hair, face, ears and shirt are all kept (apart from their softened outline).
    for colour in (HAIR, SKIN):
        assert _count(out, colour) >= _count(img, colour, inner=5)
    assert _count(out, SHIRT) >= 0.85 * _count(img, SHIRT, inner=5)  # framed at the chest
    corner = out[: h // 8, : w // 8]
    assert 0.4 < corner.mean() < 0.9 and corner.std() < 0.05  # stripes shaded over evenly
    assert _count(out, (0.1, 0.1, 0.9), 0.1) < 0.02 * _count(img, (0.1, 0.1, 0.9), 0.1)


def test_photo_background_is_kept_when_asked():
    img = _studio_photo()
    out = P.prepare_portrait(img, 0.8, _rel((180, 160, 120, 150), img), "photo")
    assert _count(out, (0.1, 0.1, 0.9)) > 0.7 * _count(img, (0.1, 0.1, 0.9))
    assert _count(out, HAIR) >= _count(img, HAIR, inner=5)


def test_close_up_gets_shaded_paper_around_it():
    img = _studio_photo(300, 260, face=(70, 90, 120, 150))[40:, :]  # hair close to the top
    out = P.prepare_portrait(img, 0.8, _rel((70, 50, 120, 150), img), "shade")
    assert out.shape[0] > img.shape[0]  # framed with room around the head, not cropped
    assert _count(out, HAIR) >= _count(img, HAIR, inner=5)
    assert out[:5].mean() < 0.95  # the extra space is shaded, not left white


def test_without_a_clear_face_the_whole_photo_is_kept(monkeypatch):
    monkeypatch.setattr(P, "_face_in", lambda rgb: ((100, 80, 90, 90), False))
    img = np.random.default_rng(1).random((400, 400, 3)).astype(np.float32)
    out = P.prepare_portrait(img, 0.8)
    assert out.shape[:2] == (500, 400)
    assert np.allclose(out[60:380], img[10:330])


def test_graphite_drawing_shows_strokes_and_paper():
    from pencil_timelapse.sketch import PAPER, graphite_drawing
    grey = np.full((500, 400, 3), 0.55, np.float32)
    drawn = graphite_drawing(grey, (400, 500), seed=1)
    inner = drawn[60:-60, 60:-60].mean(axis=2)
    assert inner.std() > 0.04  # hatching, not a flat filter
    assert inner.min() > 0.18  # graphite is a grey, never black
    white = graphite_drawing(np.ones((500, 400, 3), np.float32), (400, 500), seed=1)
    assert np.abs(white - PAPER).max() < 0.05


def test_choose_portrait_takes_the_best_photo_and_falls_back(monkeypatch, tmp_path):
    player = P.Player("Test Player", "Q1", "a.jpg", 10, photo_url="https://upload.wikimedia.org/a")
    cands = [{"url": f"https://upload.wikimedia.org/{k}.jpg", "title": k, "width": 2000,
              "bonus": 0.0, "mime": "image/jpeg", "source": k} for k in ("good", "best", "poor")]
    monkeypatch.setattr(P, "candidate_photos", lambda p: cands)
    sizes = {"good": 200, "best": 300, "poor": 100}

    def fake_get(url, timeout=30.0):
        key = url.rsplit("/", 1)[1].split(".")[0]
        return cv2.imencode(".png", np.zeros((sizes[key], sizes[key], 3), np.uint8))[1].tobytes()

    monkeypatch.setattr(P, "_get", fake_get)
    monkeypatch.setattr(P, "portrait_score",
                        lambda thumb, width=0: (thumb.shape[0] / 300.0, (0.3, 0.2, 0.4, 0.3)))
    tried = []

    def fake_download(cand, cache_dir):
        tried.append(cand["source"])
        if cand["source"] == "best":
            raise OSError("connection reset")
        return "/tmp/x.jpg", ImageCredit(cand["title"], "", "", cand["source"], "/tmp/x.jpg")

    monkeypatch.setattr(P, "download_candidate", fake_download)
    path, credit, face = P.choose_portrait(player, str(tmp_path))
    assert tried == ["best", "good"] and credit.source_url == "good"
    assert credit.title == "Test Player" and face == (0.3, 0.2, 0.4, 0.3)


def test_render_players_video_picture_and_credits(tmp_path, monkeypatch, sample_files):
    def fake_choose(player, cache_dir):
        return sample_files["end"], ImageCredit(player.name, "A. Person", "CC BY 4.0",
                                                "https://commons.wikimedia.org/x",
                                                sample_files["end"]), None

    monkeypatch.setattr(P, "choose_portrait", fake_choose)
    team = [P.Player("First Player", "Q1", "a.jpg", 5), P.Player("Second Player", "Q2", "b.jpg", 4)]
    progress = []
    result = P.render_players(team, "TESTLAND", str(tmp_path / "out" / "team.mp4"), 320, 180,
                              fps=10, seconds_per_player=4, preset="ultrafast", show=PLAIN,
                              progress=lambda f, m: progress.append(f))
    assert result.duration == pytest.approx(4 + 2 * 4 + 3 + 4)
    assert result.players == ["First Player", "Second Player"]
    assert os.path.getsize(result.output_path) > 1000
    picture = cv2.imread(result.image_path)
    assert picture.shape == (180, 320, 3)
    text = open(result.credits_path, encoding="utf-8").read()
    assert "First Player - A. Person, CC BY 4.0" in text
    assert progress[-1] == 1.0 and progress == sorted(progress)
    assert [n for n in os.listdir(tmp_path / "out") if n.startswith(".players-")] == []


def test_render_players_checks_its_input(tmp_path):
    with pytest.raises(InputError):
        P.render_players([], "X", str(tmp_path / "a.mp4"))
    with pytest.raises(InputError):
        P.render_players([P.Player(str(i), "Q", "f", 1) for i in range(13)], "X",
                         str(tmp_path / "a.mp4"))
    with pytest.raises(InputError):
        P.render_players([P.Player("A", "Q", "f", 1)], "X", str(tmp_path / "a.mp4"),
                         seconds_per_player=2)


def test_focus_search_stays_in_the_portrait_being_drawn(portrait):
    from pencil_timelapse import analysis as A
    from pencil_timelapse.imaging import luminance

    face = portrait["end"]
    h, w = face.shape[:2]
    page = np.full((h, 3 * w, 3), 245, np.uint8)
    page[:, :w] = face  # a finished portrait on the left, the new one on the right
    page[:, 2 * w:] = face
    focus = A.detect_focus(luminance(page.astype(np.float32) / 255.0), (2 * w, 0, w, h))
    assert 2 * w <= focus.centre[0] <= 3 * w
    assert all(2 * w <= x <= 3 * w for x, _ in focus.points)
    assert focus.face is None or focus.face[0] >= 2 * w


def test_render_players_from_own_photos_with_sound(tmp_path, sample_files):
    import subprocess
    from pencil_timelapse.encoder import find_ffmpeg
    from pencil_timelapse.media import write_wav

    sound = tmp_path / "music.wav"
    t = np.arange(int(20.0 * 44100)) / 44100.0
    write_wav(str(sound), 0.3 * np.sin(2 * np.pi * 440 * t))
    team = [P.Player(n, "", "", 0, local_file=sample_files["end"]) for n in ("One", "Two")]
    result = P.render_players(team, "MY TEAM", str(tmp_path / "team.mp4"), 320, 180, fps=10,
                              preset="ultrafast", audio=str(sound), show=PLAIN)
    # 4 s title + 2 players + 3 s hold = the 20 s of music; no credits card for own photos
    assert result.duration == pytest.approx(20.0)
    assert result.credits_path is None
    info = subprocess.run([find_ffmpeg(), "-i", result.output_path], capture_output=True,
                          text=True).stderr
    assert "Audio: aac" in info and "Duration: 00:00:20" in info


def test_own_photo_path_is_not_sent_to_the_browser():
    player = P.Player("A", "Q", "f.jpg", 1, local_file="/secret/photo.jpg")
    assert "local_file" not in player.to_dict()


def test_names_from_file_names():
    assert P.name_from_filename("cristiano_ronaldo-2.jpg") == "Cristiano Ronaldo"
    assert P.name_from_filename("/x/Luís Figo (1).png") == "Luís Figo"


def test_full_show_intro_highlights_and_call_to_action(tmp_path, sample_files):
    import subprocess
    from pencil_timelapse.encoder import find_ffmpeg

    team = [P.Player(n, "", "", 0, local_file=sample_files["end"]) for n in ("One", "Two")]
    labels = []
    result = P.render_players(team, "MY TEAM", str(tmp_path / "show.mp4"), 320, 180, fps=10,
                              seconds_per_player=4, preset="ultrafast",
                              progress=lambda f, m: labels.append(m))
    # intro 5 + title 4 + 2 x (teaser 1.4 + drawing 4 + highlight 2.6) + hold 2 + ending 7
    assert result.duration == pytest.approx(34.0)
    for step in ("Intro", "Who's first?", "Who's next?", "One", "Two", "Like, subscribe, comment"):
        assert step in labels
    info = subprocess.run([find_ffmpeg(), "-i", result.output_path], capture_output=True,
                          text=True).stderr
    assert "Audio: aac" in info  # the sound effects
    assert "Duration: 00:00:34" in info


def test_sound_effects_are_placed_in_time():
    from pencil_timelapse.showtime import RATE, sound_track
    track = sound_track([(1.0, "pop"), (2.5, "click")], 4.0)
    assert len(track) >= 4 * RATE and np.abs(track[: int(0.9 * RATE)]).max() == 0.0
    assert np.abs(track[RATE:RATE + 2000]).max() > 0.1
    assert np.abs(track[int(2.5 * RATE):int(2.5 * RATE) + 500]).max() > 0.05


def test_dark_hair_in_front_of_a_dark_crowd_is_kept():
    """Sharp dark hair against a dark, blurred crowd: the hair stays, the crowd goes."""
    rng = np.random.default_rng(4)
    h, w, face = 600, 480, (180, 170, 120, 150)
    fx, fy, fw, fh = face
    cx, cy = fx + fw // 2, fy + fh // 2
    crowd = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 9) * 0.5
    img = np.clip(crowd - 0.1, 0.0, 1.0)
    hair = np.zeros((h, w), np.uint8)
    cv2.ellipse(hair, (cx, cy - 20), (fw // 2 + 22, fh // 2 + 45), 0, 0, 360, 1, -1)
    strands = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 0.8)
    img[hair == 1] = (0.1 + 0.25 * (strands[hair == 1] - 0.5))[:, None] * np.ones(3)
    cv2.rectangle(img, (cx - 150, fy + fh + 30), (cx + 150, h), (0.75, 0.1, 0.12), -1)
    cv2.rectangle(img, (cx - 28, cy), (cx + 28, fy + fh + 40), SKIN, -1)
    cv2.ellipse(img, (cx, cy + 8), (fw // 2 + 6, fh // 2 + 6), 0, 0, 360, SKIN, -1)
    out = P.prepare_portrait(img, 0.8, _rel(face, img), "shade")
    crop, _, fbox = P._frame(img, 0.8, face)
    framed, inside, _ = P._frame(np.dstack([hair] * 3).astype(np.float32), 0.8, face)
    shifted = ((framed[..., 0] * inside) > 0.5).astype(np.uint8)
    top_hair = cv2.erode(shifted, np.ones((9, 9), np.uint8)).astype(bool)
    top_hair[fbox[1] + fbox[3] // 3:] = False  # the hair above the face
    assert np.abs(out[top_hair] - crop[top_hair]).mean() < 0.03  # kept as in the photo
    ring = cv2.dilate(shifted, np.ones((61, 61), np.uint8)).astype(bool) & ~cv2.dilate(
        shifted, np.ones((31, 31), np.uint8)).astype(bool)
    ring[fbox[1]:] = False
    assert out[ring].mean() > crop[ring].mean() + 0.3  # the crowd above it is shaded over


def test_whole_photo_is_drawn_untouched():
    rng = np.random.default_rng(2)
    exact = rng.random((500, 400, 3)).astype(np.float32)  # already 4:5
    assert np.array_equal(P.prepare_portrait(exact, 0.8), exact)
    narrow = rng.random((500, 340, 3)).astype(np.float32)
    out = P.prepare_portrait(narrow, 0.8, background="whole")
    assert out.shape[:2] == (500, 400)
    assert np.array_equal(out[:, 30:370], narrow)  # nothing cut, nothing covered
    assert out[:, :30].std() < 0.5 * narrow.std()  # the strips beside it are a soft continuation
