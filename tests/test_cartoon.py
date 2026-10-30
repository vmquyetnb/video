import json
import math
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pencil_timelapse import ai_writer as AI
from pencil_timelapse import cartoon as C
from pencil_timelapse import scenery as SC
from pencil_timelapse.errors import InputError


def _character(path, transparent=False, background=(255, 255, 255)):
    """A round orange character with white eyes, on a plain background."""
    img = Image.new("RGBA" if transparent else "RGB", (300, 360),
                    (0, 0, 0, 0) if transparent else background)
    d = ImageDraw.Draw(img)
    d.ellipse([60, 60, 240, 340], fill=(240, 140, 60), outline=(40, 30, 50), width=6)
    for cx in (120, 180):
        d.ellipse([cx - 22, 130, cx + 22, 180], fill=(255, 255, 255), outline=(40, 30, 50), width=4)
        d.ellipse([cx - 9, 145, cx + 9, 170], fill=(30, 20, 40))
    img.save(path)
    return str(path)


@pytest.fixture
def pip(tmp_path):
    return _character(tmp_path / "pip.png")


def test_character_is_cut_out_of_a_plain_background(pip):
    rgba = C.cut_out(pip)
    h, w = rgba.shape[:2]
    assert 280 <= h <= 310 and 180 <= w <= 210  # cropped to the character
    alpha = rgba[..., 3]
    assert alpha[h // 2, w // 2] > 0.99 and alpha[2, 2] < 0.01  # body kept, corner gone
    eye = rgba[155 - 60 + int(0.02 * 360), 120 - 15 - 60 + int(0.02 * 360)]
    assert eye[3] > 0.99  # the white of the eyes stays: it is inside the character


def test_transparent_picture_keeps_its_own_outline(tmp_path):
    rgba = C.cut_out(_character(tmp_path / "t.png", transparent=True))
    assert rgba[..., 3].min() == 0.0 and rgba[..., 3].max() == 1.0


def test_unusable_pictures_are_refused(tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), (255, 255, 255)).save(blank)
    with pytest.raises(InputError):
        C.cut_out(str(blank))
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a picture")
    with pytest.raises(InputError):
        C.cut_out(str(junk))


def test_story_values_are_cleaned():
    story = C.Story.from_dict({"title": "T", "scenes": [
        {"setting": "castle", "time": "noon", "weather": "storm", "narration": "  Hello   there. ",
         "actions": "wave, fly, dig, count", "say": "Hi!", "effect": "fireworks", "holds": "Apples",
         "friend": "a sad bunny", "board": "  RED   and BLUE ", "item": "spaceship", "count": 40}]})
    shot = story.scenes[0]
    assert (shot.setting, shot.time, shot.weather, shot.effect) == ("meadow", "day", "clear", "none")
    assert shot.actions == ["wave", "float"] and shot.narration == "Hello there."
    assert (shot.holds, shot.cast, shot.board, shot.item, shot.count) == ("apple", ["sad Lulu"], "RED and BLUE",
                                                                          "", 10)  # the old bunny is Lulu now


def test_scene_lasts_for_the_narration_then_the_line():
    shot = C.Shot("meadow", actions=["enter", "walk", "jump"], narration="x", say="Hello!")
    timed = C._timed(shot, narration_s=6.0, line_s=1.5)
    names = [a for a, _ in timed]
    assert names == ["enter", "walk", "jump", "talk"]
    before = sum(s for a, s in timed if a != "talk")
    assert before == pytest.approx(6.0 + 0.7 + C.IRIS)  # the walk stretches to fill the narration
    assert timed[-1][1] == pytest.approx(1.5 + 0.6)
    quiet = C._timed(C.Shot("meadow", actions=["jump"], narration="x"), 4.0, 0.0)
    assert [a for a, _ in quiet] == ["jump", "idle"]  # something to fill the time


def test_walking_moves_the_world_and_look_turns_round():
    beats = C.Performance.plan([("walk", 2.0), ("look", 0.8), ("idle", 1.0)], 1000, 380)
    show = C.Performance(beats, 1000, 600, 250)
    _, heading, scroll0, _, _ = show.at(0.0)
    _, _, scroll1, _, _ = show.at(1.9)
    assert scroll1 > scroll0 + 300 and heading == 1
    _, heading, _, beat, _ = show.at(3.5)
    assert beat.action == "idle" and heading == -1
    jump = C.Performance(C.Performance.plan([("jump", 1.0)], 1000, 500), 1000, 600, 250)
    lifts = [jump.at(t)[0].lift for t in np.linspace(0, 0.99, 30)]
    assert max(lifts) > 100 and lifts[0] == 0  # it leaves the ground


def test_puppet_is_drawn_where_it_stands(pip):
    puppet = C.Puppet(C.cut_out(pip), 200, "right")
    frame = np.zeros((300, 400, 3), np.float32)
    puppet.draw(frame, C.Pose(100), 1, 260)
    ys, xs = np.nonzero(frame.sum(axis=2) > 0.3)
    assert abs(xs.mean() - 100) < 10 and ys.max() <= 262 and ys.min() >= 55
    assert puppet.mirror(-1) == -1.0 and puppet.mirror(1) == 1.0  # walking left: mirrored
    assert C.Puppet(C.cut_out(pip), 200, "front").mirror(-1) == 1.0
    leaning = np.zeros_like(frame)
    puppet.draw(leaning, C.Pose(100, lean=0.2), 1, 260)
    top = lambda f: np.nonzero(f[60:90].sum(axis=2) > 0.3)[1].mean()  # noqa: E731
    assert top(leaning) > top(frame) + 15  # the top leans along the heading


def test_every_backdrop_can_be_drawn():
    for setting in SC.SETTINGS:
        bd = SC.build(setting, "night" if setting == "city" else "day", "rain", 160, 90)
        frame = bd.background(1.0, 37.0)
        bd.foreground(frame, 1.0, 37.0)
        assert frame.shape == (90, 160, 3) and 0.0 <= frame.min() and frame.max() <= 1.0
        assert 0.7 * 90 < bd.ground < 90
    assert SC.build("space", "day", "clear", 160, 90).floating


def test_story_written_by_ai(monkeypatch):
    seen = []

    def fake(key, model, prompt, timeout, system):
        seen.append((prompt, system))
        return json.dumps({"title": "Pip and the Snow", "question": "Where next?", "scenes": [
            {"setting": "snow", "time": "day", "weather": "snow", "narration": "Pip saw snow.",
             "actions": ["enter", "surprised"], "say": "Wow!", "effect": "stars"},
            {"setting": "nowhere", "narration": "", "say": ""}]})

    monkeypatch.setattr(AI, "_gemini", fake)
    story = AI.write_story("gemini", "k", "Pip", "Pip sees snow", 4)
    assert story["title"] == "Pip and the Snow" and len(story["scenes"]) == 1
    assert story["scenes"][0]["actions"] == ["enter", "surprised"]
    prompt, system = seen[0]
    assert "Write 4 scenes" in prompt and "underwater" in prompt and "teaser" in system


@pytest.mark.skipif(not shutil.which("say"), reason="needs the macOS speech synthesiser")
def test_voices_are_read_and_pitched(tmp_path):
    plain = C.speak("Hello there.", C.CHARACTER_VOICE, str(tmp_path), "a")
    high = C.speak("Hello there.", C.CHARACTER_VOICE, str(tmp_path), "b", pitch=1.3)
    assert len(plain) > 0.4 * 44100
    assert abs(len(high) - len(plain)) < 0.15 * len(plain)  # same length, higher voice
    assert "Samantha" in C.voices()


def test_cartoon_video(tmp_path, pip):
    story = C.Story("Pip's Day", [
        C.Shot("meadow", "day", "clear", "Pip went for a walk.", ["enter", "walk", "jump"], "Hello!", "hearts"),
        C.Shot("space", "night", "clear", "", ["float"], "Wow, space!", "stars")])
    labels = []
    result = C.render_cartoon(pip, "Pip", story, str(tmp_path / "c.mp4"), 320, 180, fps=10,
                              preset="ultrafast", intro=True, outro=False,
                              progress=lambda f, m: labels.append(m))
    assert labels.index("Title") < labels.index("Scene 1") < labels.index("Scene 2") < labels.index("The End")
    assert 4.2 + 3.2 + 2 * (2.2 + 2 * C.IRIS) < result.duration < 40
    assert (tmp_path / "c.png").exists() and result.scenes == 2


def test_script_file(tmp_path, pip):
    script = tmp_path / "s.json"
    script.write_text(json.dumps({"character": "pip.png", "name": "Pip", "faces": "left",
                                  "scenes": [{"setting": "beach", "narration": "Hi.", "actions": ["run"]}]}))
    picture, name, faces, story = C.load_script(str(script))
    assert picture.endswith("pip.png") and faces == "left" and story.scenes[0].actions == ["run"]
    sample = C.sample_story("Momo")
    assert "Momo" in sample.title and len(sample.scenes) >= 5
    assert {s.setting for s in sample.scenes} <= set(SC.SETTINGS)


# --------------------------------------------------------------------------- Peeko, props, the chalkboard, episodes

from pencil_timelapse import episodes as EP  # noqa: E402
from pencil_timelapse import cast as PK  # noqa: E402
from pencil_timelapse import props as PR  # noqa: E402


def _drawn(action, u=0.5, level=0.0, holds="", yaw=0.1, t=1.0, heading=1, bloom=0.0):
    frame = np.zeros((300, 300, 3), np.float32)
    PK.Peeko(200).draw(frame, C.Pose(150, yaw=yaw), heading, 280, PK.Act(action, u, 2.0, t, level, holds, bloom))
    return frame


def test_peeko_moves_its_limbs_and_face():
    steps = [_drawn("walk", u, yaw=0.8) for u in (0.0, 0.11, 0.22)]
    legs = [f[230:290] for f in steps]
    assert np.abs(legs[0] - legs[1]).mean() > 0.01 and np.abs(legs[1] - legs[2]).mean() > 0.01  # legs swing
    quiet, loud = _drawn("talk", level=0.0), _drawn("talk", level=1.0)
    mouth = (slice(185, 215), slice(130, 170))
    assert np.abs(quiet[mouth] - loud[mouth]).mean() > 0.02  # the mouth opens with the voice
    beside_head = lambda f: (f[105:145].sum(axis=2) > 0.3)[:, np.r_[0:100, 200:300]].sum()  # noqa: E731
    assert beside_head(_drawn("happy")) > 100 and beside_head(_drawn("idle")) < 20  # arms up
    open_eyes = _drawn("idle", t=1.0)
    blink_t = next(t / 100 for t in range(0, 400) if PK._blinking(t / 100, PK.blink_offset(PK.PEEKO)))
    assert np.abs(open_eyes - _drawn("idle", t=blink_t)).sum() > 1.0  # it blinks
    flower = _drawn("idle", bloom=1.0)
    assert (np.abs(flower - open_eyes).sum(axis=2) > 0.2)[80:110].sum() > 30  # a flower blooms on its sprout


def test_peeko_holds_props_and_turns():
    plain, apple = _drawn("idle"), _drawn("idle", holds="apple")
    red = (apple[..., 0] > 0.7) & (apple[..., 1] < 0.35)
    assert red.sum() > 50 and ((plain[..., 0] > 0.7) & (plain[..., 1] < 0.35)).sum() < 10
    right, left = _drawn("walk", yaw=0.8, heading=1), _drawn("walk", yaw=0.8, heading=-1)
    assert np.abs(right - left[:, ::-1]).mean() < 0.02  # walking the other way: mirrored
    back = _drawn("spin", yaw=math.pi)
    assert (back.max(axis=2) > 0.95).sum() < (_drawn("idle").max(axis=2) > 0.95).sum()  # no eyes from behind


def test_every_prop_and_friend_is_drawn():
    for name in PR.PROPS:
        sprite = PR.prop_sprite(name, 64)
        assert sprite.alpha.max() > 0.9 and 0.05 < sprite.alpha.mean() < 0.9, name
    for value in C.FRIEND_VALUES[1:]:
        kind, happy = PR.friend_kind(value)
        assert kind and happy == ("sad" not in value)
        assert PR.friend_sprite(kind, happy, 80).alpha.max() > 0.9


def test_counting_comes_after_the_narration():
    shot = C.Shot("room", actions=["walk"], narration="x", say="Five!", count=5)
    timed = C._timed(shot, 3.0, 1.0, count_s=4.0)
    assert [a for a, _ in timed] == ["walk", "count", "talk"] and timed[1][1] == 4.0
    beats = C.Performance.plan(timed, 1000, 300, span=(0.1, 0.42), world=False)
    assert beats[0].bg == 0.0 and C.Performance._after(beats[0], 1000, (0.1, 0.42))[0] <= 420  # stays left
    board = C.Board("RED and BLUE", "apple", 5, (500, 40, 950, 330), 0.4, [2.0 + k for k in range(5)])
    assert len(board.items) == 5 and len(board.lines) == 1
    assert board.lines[0][0][0].rgb[..., 0].max() > 0.9  # "RED" in red
    frame = np.zeros((360, 1000, 3), np.float32)
    board.draw(frame, 3.3)  # two apples up
    two = frame.copy()
    board.draw(frame, 6.5)
    assert np.abs(frame - two).sum() > 10  # more popped up


def test_episodes_are_ready_to_animate():
    assert len(EP.EPISODES) >= 10 and {e["group"] for e in EP.EPISODES} == set(EP.GROUPS)
    for e in EP.EPISODES:
        story = EP.episode(e["id"], "Momo")
        assert story.lesson and story.question and "{name}" not in str(story.to_dict())
        for raw, shot in zip(e["scenes"], story.scenes):
            assert len(shot.actions) == len(raw["actions"]), (e["id"], raw["actions"])  # every action known
            assert shot.holds == raw.get("holds", "") and shot.item == raw.get("item", ""), e["id"]
            assert shot.narration or shot.say or shot.lines
    titles = {e["id"]: e["title"] for e in EP.catalogue("Momo")}
    assert titles["counting-apples"] == "Momo Counts Apples" and titles["s1e09-peeko-feels-grumpy"] == "Momo Feels Grumpy"
    assert EP.catalogue()[0]["episode"] == 1  # the season comes first


def test_ai_is_told_about_props_and_counting(monkeypatch):
    seen = []
    monkeypatch.setattr(AI, "_openai", lambda key, model, prompt, timeout, system: seen.append(prompt) or json.dumps(
        {"title": "T", "lesson": "Share.", "scenes": [{"narration": "Hi.", "holds": "apple", "count": 3,
                                                       "item": "apple", "board": "Count!"}]}))
    story = AI.write_story("openai", "k", "Peeko", "sharing", 3, about=PK.ABOUT)
    assert story["lesson"] == "Share." and story["scenes"][0]["count"] == 3
    assert "toothbrush" in seen[0] and "sprout" in seen[0] and "count" in seen[0]


def test_peeko_video(tmp_path):
    story = C.Story("Count", [C.Shot("room", "night", "clear", "Let's count.", ["point"], "Three!", "stars",
                                     "book", "bird", "Count!", "star", 3)], lesson="Count every day.")
    labels = []
    result = C.render_cartoon(None, "", story, str(tmp_path / "p.mp4"), 320, 180, fps=10, preset="ultrafast",
                              intro=True, outro=False, progress=lambda f, m: labels.append(m))
    assert "Drawing Peeko" in labels and result.scenes == 1 and (tmp_path / "p.png").exists()


# --------------------------------------------------------------------------- Peeko & Friends: the cast, conversations, music, the season

from pencil_timelapse import music as MU  # noqa: E402
from pencil_timelapse import season1 as S1  # noqa: E402


def _friend(d, action="idle", level=0.0, t=1.0):
    frame = np.zeros((320, 320, 3), np.float32)
    rig = PK.Character(int(200 * d.size), d)
    rig.draw(frame, C.Pose(160, lift=d.hover * rig.h, yaw=0.1), 1, 300, PK.Act(action, 0.6, 2.0, t, level))
    return frame


def test_every_friend_is_drawn_and_talks():
    for d in PK.CAST.values():
        quiet, loud = _friend(d, "talk", 0.0), _friend(d, "talk", 1.0)
        assert (quiet.sum(axis=2) > 0.3).mean() > 0.05, d.name
        assert np.abs(quiet - loud).sum() > 5.0, d.name  # the mouth (or beak) opens with the voice
        assert np.abs(_friend(d, "angry") - _friend(d, "idle")).sum() > 50.0, d.name
    assert len({d.voice for d in PK.CAST.values()}) == 4  # everyone sounds different


def test_conversations_and_quizzes_are_read():
    story = C.Story.from_dict({"hook": True, "scenes": [
        {"narration": "", "cast": "Lulu, sleepy bobo, Peeko, Nobody", "quiz": "apple, candy", "answer": "apples",
         "lines": "Lulu: Hi! (wave)\nHello!\nNarrator: Later...\nBobo: Yawn. (dance badly)"},
        {"narration": "x", "quiz": ["apple", "candy"], "answer": "ball"}]})
    first, second = story.scenes
    assert first.cast == ["Lulu", "sleepy Bobo"] and story.hook
    assert [(ln.who, ln.do) for ln in first.lines] == [("Lulu", "wave"), ("", ""), ("Narrator", ""), ("Bobo", "")]
    assert (first.quiz, first.answer) == (["apple", "candy"], "apple") and second.quiz == []  # no right answer
    assert not C.Story.from_dict({"hook": True, "scenes": [{"narration": "x"}]}).hook  # a teaser needs more
    timed, at = C._timeline(first, 0.0, [(False, 1.0, "wave"), (True, 2.0, ""), (False, 1.0, "")], quiz=True)
    assert [a for a, _ in timed[at - 2:]] == ["think", "happy", "listen", "talk", "listen"]


def test_quiz_board_shows_the_answer():
    board = C.Board("Which one?", "", 0, (500, 40, 950, 330), 0.2, [], ["candy", "apple"], "apple", 1.0)
    assert len(board.choices) == 2
    before, after = np.zeros((360, 1000, 3), np.float32), np.zeros((360, 1000, 3), np.float32)
    board.draw(before, 1.5)
    board.draw(after, 1.0 + C.QUIZ_COUNT + 0.5)
    x, y = int(board.choices[1][0]), int(board.choices[1][1])
    ring = after[y - 60:y + 60, x - 60:x + 60]
    assert (ring[..., 1] - ring[..., 0] > 0.3).sum() > (before[y - 60:y + 60, x - 60:x + 60][..., 1] -
                                                        before[y - 60:y + 60, x - 60:x + 60][..., 0] > 0.3).sum()


def test_music_is_made_and_steps_aside_for_voices():
    tune = MU.background(12.0, "happy")
    assert abs(len(tune) / MU.RATE - 12.0) < 0.01 and 0.45 < np.abs(tune).max() <= 0.51
    assert np.abs(MU.background(5.0, "calm")).max() > 0.1
    theme, marks = MU.jingle()
    assert 6.0 < len(theme) / MU.RATE < 9.0 and marks == sorted(marks) and np.abs(theme).max() <= 0.81
    voice = np.zeros_like(tune)
    voice[MU.RATE * 4:MU.RATE * 6] = 0.5
    ducked = MU.duck(tune, voice)
    assert np.abs(ducked[MU.RATE * 5 - 2000:MU.RATE * 5]).max() < 0.7 * np.abs(tune[MU.RATE * 5 - 2000:MU.RATE * 5]).max()
    assert np.allclose(ducked[:MU.RATE], tune[:MU.RATE])


def test_season_one_is_complete():
    assert len(S1.EPISODES) == 12
    names = {d.name for d in PK.CAST.values()}
    for k, raw in enumerate(S1.EPISODES, 1):
        story = EP.episode(raw["id"])
        assert story.hook and story.episode == k and story.season == 12 and story.lesson and story.question
        assert story.next or k == 12
        assert 7 <= len(story.scenes) <= 10 and not story.scenes[0].narration  # a teaser first
        quizzes = sum(bool(s.quiz) for s in story.scenes)
        joins = sum(bool(s.count) for s in story.scenes) + quizzes
        assert joins >= 2, raw["id"]  # children join in at least twice
        for s in story.scenes:
            present = {c.split()[-1] for c in s.cast}
            for ln in s.lines:
                assert ln.who in present | {"", "Narrator"}, (raw["id"], ln)
            assert present <= names
            assert s.narration or s.lines


def test_series_episode_video(tmp_path):
    story = C.Story("Snack Time", [
        C.Shot("kitchen", "day", "clear", "", ["surprised"], cast=["Bobo"],
               lines=[C.Line("Bobo", "Snack!", "eat"), C.Line("", "Wait!")]),
        C.Shot("garden", "day", "clear", "Which one is healthy?", ["think"], board="Healthy?", cast=["Kiko"],
               quiz=["apple", "candy"], answer="apple", lines=[C.Line("Kiko", "The apple!")])],
        lesson="Eat fruit.", hook=True, episode=2, season=12, next="Kiko Counts the Stars", music="happy")
    labels = []
    result = C.render_cartoon(None, "", story, str(tmp_path / "s.mp4"), 320, 180, fps=10, preset="ultrafast",
                              outro=False, series=True, progress=lambda f, m: labels.append(m))
    assert labels.index("Scene 1") < labels.index("Series intro") < labels.index("Title") < labels.index("Scene 2")
    assert result.duration > 7.6 + 4.2 and (tmp_path / "s.png").exists()
