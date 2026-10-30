"""Cartoon stories: a hero and its friends act out a story in backdrops drawn by
the program (:mod:`scenery`), with a narrator, conversations in speech bubbles
(each character in its own voice), subtitles, music (:mod:`music`) and sound
effects.

The hero is either Peeko, drawn by the program with moving arms, legs and face
(:mod:`cast`, with its friends Lulu, Kiko and Bobo), or cut out of the user's
picture and animated as a whole, like a cut-out puppet: it walks and runs with
a bounce (the world scrolls past), jumps with squash and stretch, dances,
sways, trembles, droops, floats, spins and turns round; small effects (hearts,
music notes, a rain cloud, Zzz, an exclamation mark...) show how it feels. Scenes open and close with an
iris, as in old cartoons; around the story are a title with the character
dropping in, "The End" (with the story's lesson read out) and a like /
subscribe / comment ending.

For learning stories a scene can show a chalkboard with a word, a picture or
things to count (the narrator counts them one by one as they pop up), give
the character something to hold (:mod:`props`), bring friends along who talk,
and ask a picture quiz with a 3-2-1 countdown. A series episode opens with a
teaser scene and the show's opening (logo, friends, theme tune) and ends with
the lesson, a flower for the season's collection and a look at the next one.

The narration is read by the macOS speech synthesiser (``say``) in English;
each character speaks in its own voice and pitch.

Usage:  python -m pencil_timelapse.cartoon story.json -o story.mp4
        python -m pencil_timelapse.cartoon --season -o season1/
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import cast as CA
from . import props as PR
from . import scenery as SC
from .config import RenderConfig
from .encoder import FFmpegWriter, find_ffmpeg
from .errors import InputError, PencilError, RenderError
from .media import SAMPLE_RATE, decode_audio, write_wav
from .players import _add_soundtrack
from .showtime import (BOLD_FONTS, INK, WHITE, Sprite, _font, _phase, _pill, _render, _rgb255, _u8, back,
                       outro_clip, smooth, sound_track, text_sprite)

F32 = np.float32
ProgressFn = Callable[[float, str], None]
SUP = "/System/Library/Fonts/Supplemental/"
TITLE_FONTS = (SUP + "ChalkboardSE.ttc#2", SUP + "Comic Sans MS Bold.ttf", SUP + "Arial Rounded Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/comicbd.ttf")
SPEECH_FONTS = (SUP + "ChalkboardSE.ttc#1", SUP + "Comic Sans MS.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/comic.ttf")
CAPTION_FONTS = (SUP + "Arial Rounded Bold.ttf",) + BOLD_FONTS
NARRATOR = "Daniel"
CHARACTER_VOICE = "Samantha"
NARRATOR_RATE = 165  # words a minute: a little slower than usual, for children
GOOD_VOICES = ("Daniel", "Samantha", "Karen", "Moira", "Tessa", "Rishi")  # listed first
QUESTION = "What should {name} do next?"
OUTLINE = (0.16, 0.13, 0.22)

# action: (usual seconds, can be stretched to fill time)
ACTIONS: Dict[str, Tuple[float, bool]] = {
    "idle": (1.5, True), "walk": (2.6, True), "run": (1.8, True), "enter": (1.9, False),
    "exit": (1.7, False), "jump": (1.1, False), "dance": (2.4, True), "happy": (1.6, False),
    "sad": (2.4, True), "scared": (2.0, True), "surprised": (1.2, False), "think": (2.2, True),
    "sleep": (3.0, True), "float": (2.6, True), "spin": (1.0, False), "look": (0.8, False),
    "wave": (1.8, True), "clap": (1.6, False), "point": (1.6, True), "brush": (2.6, True),
    "wash": (2.6, True), "eat": (2.2, True), "read": (2.6, True), "yawn": (1.8, False), "angry": (2.0, True),
    "talk": (1.5, False), "count": (1.0, False), "listen": (1.0, True)}
INTERNAL = ("talk", "count", "listen")  # actions the scene adds by itself
ALIASES = {"laugh": "happy", "cheer": "happy", "smile": "happy", "cry": "sad", "hug": "happy",
           "fly": "float", "swim": "float", "hop": "jump", "skip": "walk", "wander": "walk",
           "explore": "walk", "rest": "sleep", "nap": "sleep", "fear": "scared", "shiver": "scared",
           "wonder": "think", "turn": "look", "twirl": "spin", "celebrate": "dance", "arrive": "enter",
           "leave": "exit", "stand": "idle", "wait": "idle", "speak": "talk", "say": "talk",
           "shout": "talk", "rush": "run", "hurry": "run", "gasp": "surprised", "shock": "surprised",
           "chew": "eat", "bite": "eat", "drink": "eat", "munch": "eat", "scrub": "wash", "rinse": "wash",
           "show": "point", "stretch": "yawn", "applaud": "clap", "study": "read", "greet": "wave",
           "hello": "wave", "bye": "wave", "goodbye": "wave", "grumpy": "angry", "mad": "angry",
           "stomp": "angry", "cross": "angry"}
EFFECTS = ("none", "hearts", "stars", "music", "confetti", "idea", "question", "exclaim")
# Feelings that come with an action.
ACTION_EFFECT = {"happy": "sparkles", "sad": "raincloud", "scared": "sweat", "surprised": "exclaim",
                 "think": "question", "sleep": "zzz", "dance": "music", "wash": "soap", "angry": "steam"}
NUMBERS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")
FRIEND_VALUES = ("", "bird", "sad bird", "bunny", "sad bunny")  # old scripts


# --------------------------------------------------------------------------- the story

@dataclass
class Line:
    """Something said in a scene: by the hero, a friend (Lulu, Kiko, Bobo) or the narrator."""

    who: str
    say: str
    do: str = ""  # what the speaker does while saying it (an action), or nothing


MOODS = ("sad", "scared", "sleepy", "happy")  # how a friend is when it is not speaking


@dataclass
class Shot:
    """One scene of the story."""

    setting: str = "meadow"
    time: str = "day"
    weather: str = "clear"
    narration: str = ""
    actions: List[str] = field(default_factory=lambda: ["idle"])
    say: str = ""  # a line the hero says (speech bubble, its voice), before the other lines
    effect: str = "none"
    holds: str = ""  # a prop in the hero's hand
    friend: str = ""  # old scripts: "bird", "bunny", "sad bunny" (now Kiko and Lulu)
    board: str = ""  # words on the chalkboard
    item: str = ""  # a picture on the chalkboard
    count: int = 0  # how many of ``item`` to count (numbers alone without an item)
    cast: List[str] = field(default_factory=list)  # friends in the scene: "Lulu", "sad Bobo"...
    lines: List[Line] = field(default_factory=list)  # the conversation, after the narration
    quiz: List[str] = field(default_factory=list)  # 2 or 3 pictures to choose from
    answer: str = ""  # the right one

    def cleaned(self) -> "Shot":
        """The scene with every value one the animation knows."""
        acts = []
        for a in self.actions or []:
            a = ALIASES.get(str(a).strip().lower(), str(a).strip().lower())
            if a in ACTIONS and a not in ("count", "listen"):
                acts.append(a)
        try:
            count = int(self.count or 0)
        except (TypeError, ValueError):
            count = 0
        cast = [c for c in (_cast_name(c) for c in self.cast or []) if c]
        kind, happy = PR.friend_kind(self.friend)  # old scripts' bird and bunny
        if kind:
            name = {"bird": "Kiko", "bunny": "Lulu"}[kind]
            cast.append(name if happy else f"sad {name}")
        seen, unique = set(), []
        for c in cast:
            if c.split()[-1] not in seen:
                seen.add(c.split()[-1])
                unique.append(c)
        lines = []
        for ln in self.lines or []:
            text = re.sub(r"\s+", " ", ln.say or "").strip()[:140]
            if not text:
                continue
            who = _speaker(ln.who)
            do = ALIASES.get(str(ln.do or "").strip().lower(), str(ln.do or "").strip().lower())
            lines.append(Line(who, text, do if do in ACTIONS and do not in INTERNAL else ""))
        quiz = [q for q in (_prop_name(q) for q in self.quiz or []) if q][:3]
        answer = _prop_name(self.answer)
        if len(quiz) < 2 or answer not in quiz:
            quiz, answer = [], ""
        return Shot(self.setting if self.setting in SC.SETTINGS else "meadow",
                    self.time if self.time in SC.TIMES else "day",
                    self.weather if self.weather in SC.WEATHER else "clear",
                    re.sub(r"\s+", " ", self.narration or "").strip()[:400], acts[:6] or ["idle"],
                    re.sub(r"\s+", " ", self.say or "").strip()[:140],
                    self.effect if self.effect in EFFECTS else "none",
                    _prop_name(self.holds), "",
                    re.sub(r"\s+", " ", self.board or "").strip()[:40], _prop_name(self.item),
                    max(0, min(count, 10)), unique[:3], lines[:12], quiz, answer)


def _cast_name(value) -> str:
    """"sad lulu" -> "sad Lulu"; "" for someone unknown."""
    words = str(value or "").strip().lower().split()
    if not words or not CA.design(words[-1]) or words[-1] == CA.NAME.lower():
        return ""
    mood = words[0] if len(words) > 1 and words[0] in MOODS else ""
    return f"{mood} {CA.design(words[-1]).name}".strip()


def _speaker(value) -> str:
    """A friend's name, "Narrator", or "" for the hero."""
    v = str(value or "").strip()
    if v.lower() in ("narrator", "the narrator"):
        return "Narrator"
    d = CA.design(v)
    return d.name if d and d.name != CA.NAME else ""


def parse_lines(text: str) -> List[Line]:
    """Lines written as "Lulu: Hello! (wave)", one per line; no name = the hero."""
    out = []
    for raw in str(text or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        who, sep, said = raw.partition(":")
        if not sep or len(who) > 20:
            who, said = "", raw
        do = ""
        m = re.search(r"\(([a-zA-Z ]+)\)\s*$", said)
        if m:
            do, said = m.group(1).strip(), said[:m.start()].strip()
        out.append(Line(who.strip(), said.strip(), do))
    return out


def _prop_name(value) -> str:
    v = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    v = {"traffic light": "red light", "stop light": "red light", "toothpaste": "toothbrush",
         "glass": "cup", "water": "cup", "candies": "candy"}.get(v, v)
    if v not in PR.PROPS and v.endswith("s") and v[:-1] in PR.PROPS:
        v = v[:-1]  # "apples" -> "apple"
    return v if v in PR.PROPS else ""


@dataclass
class Story:
    title: str
    scenes: List[Shot]
    question: str = ""
    lesson: str = ""  # what the story teaches, read out at the end
    hook: bool = False  # the first scene is a teaser shown before the series intro
    episode: int = 0  # its number in the season (0: not part of one)
    season: int = 0  # how many episodes the season has (the flower collection)
    next: str = ""  # what happens next time, read out at the end
    music: str = "happy"  # the background tune: happy, calm or none

    def to_dict(self) -> Dict:
        return {"title": self.title, "question": self.question, "lesson": self.lesson, "hook": self.hook,
                "episode": self.episode, "season": self.season, "next": self.next, "music": self.music,
                "scenes": [asdict(s) for s in self.scenes]}

    @classmethod
    def from_dict(cls, data: Dict) -> "Story":
        scenes = []
        for s in data.get("scenes") or []:
            if not isinstance(s, dict):
                continue
            acts = s.get("actions") or []
            if isinstance(acts, str):
                acts = re.split(r"[,\s]+", acts)
            raw_lines = s.get("lines") or []
            if isinstance(raw_lines, str):
                lines = parse_lines(raw_lines)
            else:
                lines = [Line(str(ln.get("who") or ""), str(ln.get("say") or ""), str(ln.get("do") or ""))
                         for ln in raw_lines if isinstance(ln, dict)]
            cast = s.get("cast") or []
            if isinstance(cast, str):
                cast = [c for c in re.split(r"\s*,\s*", cast) if c]
            quiz = s.get("quiz") or []
            if isinstance(quiz, str):
                quiz = [q for q in re.split(r"\s*,\s*", quiz) if q]
            scenes.append(Shot(str(s.get("setting") or "meadow"), str(s.get("time") or "day"),
                               str(s.get("weather") or "clear"), str(s.get("narration") or ""),
                               [str(a) for a in acts if str(a).strip()], str(s.get("say") or ""),
                               str(s.get("effect") or "none"), str(s.get("holds") or ""),
                               str(s.get("friend") or ""), str(s.get("board") or ""),
                               str(s.get("item") or ""), s.get("count") or 0, [str(c) for c in cast],
                               lines, [str(q) for q in quiz], str(s.get("answer") or "")).cleaned())

        def number(key, top):
            try:
                return max(0, min(int(data.get(key) or 0), top))
            except (TypeError, ValueError):
                return 0

        music = str(data.get("music") or "happy")
        return cls(str(data.get("title") or "")[:60], scenes, str(data.get("question") or "")[:90],
                   re.sub(r"\s+", " ", str(data.get("lesson") or "")).strip()[:140],
                   bool(data.get("hook")) and len(scenes) > 1, number("episode", 99), number("season", 99),
                   re.sub(r"\s+", " ", str(data.get("next") or "")).strip()[:90],
                   music if music in ("happy", "calm", "none") else "happy")

    def has_words(self) -> bool:
        return all(s.narration or s.say or s.lines for s in self.scenes)


def sample_story(name: str) -> Story:
    """A ready-made little adventure, to try the tab without an AI key."""
    n = name.strip() or "Pip"
    return Story(f"{n}'s Big Adventure", [
        Shot("room", "day", "clear", f"This is {n}. One sunny morning, {n} woke up with a big idea.",
             ["sleep", "surprised", "happy"], "Today I'm going on an adventure!", "idea"),
        Shot("meadow", "day", "clear", f"{n} skipped over the green hills and waved hello to the birds.",
             ["enter", "walk", "jump"], "What a beautiful day!", "none"),
        Shot("forest", "day", "leaves", "Deep in the forest, the leaves began to rustle.",
             ["walk", "scared", "look"], "W-who's there?", "none"),
        Shot("forest", "day", "leaves", f"It was only the wind! {n} laughed and felt brave again.",
             ["happy", "dance"], "Ha! I'm not scared at all!", "music"),
        Shot("beach", "sunset", "clear", f"At last {n} reached the sea, glowing in the evening sun.",
             ["run", "jump", "happy"], "I did it! I found the sea!", "hearts"),
        Shot("meadow", "night", "clear", f"When the stars came out, {n} lay down on the soft grass.",
             ["walk", "think", "sleep"], "", "stars"),
        Shot("space", "night", "clear", f"And in a dream, {n} floated among the planets.",
             ["float", "spin", "happy"], "See you on the next adventure!", "stars"),
    ], QUESTION.format(name=n))


def load_script(path: str) -> Tuple[Optional[str], str, str, Story]:
    """A JSON script: {"character": "pip.png", "name": "Pip", "faces": "right",
    "title", "question", "lesson", "scenes": [...]}, the picture's path
    relative to it; no character (or "peeko") stars Peeko."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    character = str(data.get("character") or "").strip()
    if not character or character.lower() == CA.NAME.lower():
        return None, str(data.get("name") or CA.NAME), "front", Story.from_dict(data)
    picture = os.path.join(os.path.dirname(os.path.abspath(path)), character)
    return picture, str(data.get("name") or "Pip"), str(data.get("faces") or "right"), Story.from_dict(data)


# --------------------------------------------------------------------------- the character

def cut_out(path, longest: int = 1400) -> np.ndarray:
    """The character alone (float RGBA, cropped to it); ``path`` may be a file object.

    A transparent background is used as it is. Otherwise the background is
    whatever plain colour surrounds the character and reaches the picture's
    edges (white areas inside the character, like its eyes, stay); a busy
    background is separated with GrabCut.
    """
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    except Exception:
        raise InputError("The character picture could not be read. Use a PNG or JPG.") from None
    scale = min(1.0, longest / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    rgba = np.asarray(img).astype(F32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    if min(alpha.shape) < 32:
        raise InputError("The character picture is too small.")
    if (alpha < 0.98).mean() > 0.03:
        mask = alpha
        background = None
    else:
        mask, background = _separate(rgb)
    ys, xs = np.nonzero(mask > 0.05)
    if len(ys) < 0.01 * mask.size:
        raise InputError("Could not find the character in the picture. Use a PNG with a transparent "
                         "background, or a picture on a plain background.")
    if background is not None:  # take the background colour out of the soft edge
        a = np.clip(mask, 1e-3, 1.0)[..., None]
        rgb = np.where(mask[..., None] < 0.999, np.clip((rgb - background * (1 - a)) / a, 0, 1), rgb)
    m = int(0.02 * max(mask.shape))
    y0, y1 = max(0, ys.min() - m), min(mask.shape[0], ys.max() + 1 + m)
    x0, x1 = max(0, xs.min() - m), min(mask.shape[1], xs.max() + 1 + m)
    return np.dstack([rgb, mask])[y0:y1, x0:x1].astype(F32)


def _separate(rgb: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = rgb.shape[:2]
    edge = np.concatenate([rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3),
                           rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)])
    bg = np.median(edge, axis=0)
    plain = float((np.linalg.norm(edge - bg, axis=1) < 0.12).mean())
    if plain > 0.6:
        tol = 0.1 if plain > 0.9 else 0.14
        near = (np.linalg.norm(rgb - bg, axis=2) < tol).astype(np.uint8)
        count, labels = cv2.connectedComponents(near, connectivity=4)
        touching = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        background = np.isin(labels, touching[touching > 0] if 0 in touching else touching) & (near > 0)
        fg = (~background).astype(np.uint8)
    else:
        mask = np.zeros((h, w), np.uint8)
        rect = (int(0.04 * w), int(0.04 * h), int(0.92 * w), int(0.92 * h))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.convertScaleAbs(rgb[..., ::-1], alpha=255.0), mask, rect, bgd, fgd, 5,
                    cv2.GC_INIT_WITH_RECT)
        fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        bg = None
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.nonzero(areas >= 0.03 * areas.max())[0]
        fg = np.isin(labels, keep).astype(np.uint8)
    soft = cv2.GaussianBlur(fg.astype(F32), (0, 0), 0.8)
    return np.clip((soft - 0.1) / 0.8, 0.0, 1.0), (None if bg is None else bg.astype(F32))


@dataclass
class Pose:
    x: float  # feet, from the left
    lift: float = 0.0  # above the ground, pixels
    sx: float = 1.0  # squash and stretch
    sy: float = 1.0
    angle: float = 0.0  # degrees, counter-clockwise
    lean: float = 0.0  # the top leans by this share of the height (towards the heading)
    turn: float = 1.0  # 1 facing the heading, through 0 (edge on) to -1 (turned round)
    alpha: float = 1.0
    scale: float = 1.0
    yaw: float = 0.0  # radians turned from the viewer towards the heading (drawn characters)


class Puppet:
    """The cut-out character, drawn in a pose."""

    def __init__(self, rgba: np.ndarray, height: int, faces: str = "right"):
        h, w = rgba.shape[:2]
        k = height / h
        self.rgba = cv2.resize(rgba, (max(2, int(w * k)), height),
                               interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        self.h, self.w = self.rgba.shape[:2]
        self.faces = faces
        sh = np.zeros((int(self.w * 0.3) + 8, int(self.w * 1.0) + 8), F32)
        cv2.ellipse(sh, (sh.shape[1] // 2, sh.shape[0] // 2), (int(self.w * 0.42), int(self.w * 0.1)), 0, 0,
                    360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), self.w * 0.03 + 1)
        self.shadow = Sprite(np.dstack([np.zeros(sh.shape + (3,), F32), sh[..., None] * 0.28]))

    def mirror(self, heading: int) -> float:
        """-1 when the picture must be flipped to look along ``heading``."""
        if self.faces == "left":
            return -1.0 if heading > 0 else 1.0
        if self.faces == "right":
            return -1.0 if heading < 0 else 1.0
        return 1.0

    def head(self, pose: Pose, ground: float) -> Tuple[float, float]:
        """Roughly where the top of the head is."""
        return pose.x, ground - pose.lift - self.h * pose.sy * pose.scale

    def hand(self, pose: Pose, heading: int, ground: float, act=None) -> Tuple[float, float]:
        """Where a prop is held: beside the body, on the side it looks to."""
        side = heading if self.faces != "front" else 1
        return pose.x + side * 0.42 * self.w * pose.scale, ground - pose.lift - 0.4 * self.h * pose.sy

    def draw(self, frame: np.ndarray, pose: Pose, heading: int, ground: float,
             act: Optional[CA.Act] = None) -> None:
        H, W = frame.shape[:2]
        if pose.alpha <= 0.01:
            return
        if act is not None and act.holds:
            self._draw(frame, pose, heading, ground)
            size, _, _ = PR.hold(act.holds)
            x, y = self.hand(pose, heading, ground)
            PR.prop_sprite(act.holds, int(size * self.h)).paste(
                frame, x, y - 0.1 * size * self.h, opacity=pose.alpha, angle=pose.angle)
            return
        self._draw(frame, pose, heading, ground)

    def _draw(self, frame: np.ndarray, pose: Pose, heading: int, ground: float) -> None:
        H, W = frame.shape[:2]
        lifted = 1.0 - min(1.0, pose.lift / (1.5 * self.h))
        self.shadow.paste(frame, pose.x, ground + 0.01 * self.h, scale=pose.scale * (0.55 + 0.45 * lifted),
                          opacity=pose.alpha * lifted)
        flip = self.mirror(heading) * pose.turn
        flip = math.copysign(max(abs(flip), 0.04), flip)
        sx, sy = pose.sx * pose.scale * flip, pose.sy * pose.scale
        a = math.radians(pose.angle)
        ca, sa = math.cos(a), math.sin(a)
        k = -pose.lean * heading  # shear: the top (y = -h) moves by lean * h along the heading
        # sprite -> feet origin -> scale -> shear -> rotate -> place
        A = np.array([[sx, k * sy], [0.0, sy]], np.float64)
        R = np.array([[ca, sa], [-sa, ca]], np.float64)
        M2 = R @ A
        t = np.array([pose.x, ground - pose.lift]) - M2 @ np.array([self.w / 2, self.h])
        corners = np.array([[0, 0], [self.w, 0], [0, self.h], [self.w, self.h]], np.float64) @ M2.T + t
        x0, y0 = np.floor(corners.min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 2
        fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        M = np.hstack([M2, (t - np.array([fx0, fy0]))[:, None]]).astype(np.float32)
        part = cv2.warpAffine(self.rgba, M, (fx1 - fx0, fy1 - fy0), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = part[..., 3:] * pose.alpha
        region = frame[fy0:fy1, fx0:fx1]
        region *= 1.0 - alpha
        region += part[..., :3] * alpha


# --------------------------------------------------------------------------- acting

@dataclass
class Beat:
    action: str
    start: float
    seconds: float
    x0: float  # where the character starts this beat
    heading: int
    scroll0: float
    v: float = 0.0  # walking: on-screen speed, pixels a second
    bg: float = 0.0  # walking: scenery speed
    voice: Optional[Callable[[float], float]] = None  # speaking: how loud, by time into the beat


# How far a drawn character turns from the viewer towards where it is going.
YAW = {"walk": 0.8, "run": 0.85, "enter": 0.8, "exit": 0.85, "think": 0.4, "point": 0.45, "count": 0.45,
       "read": 0.15, "talk": 0.05, "wave": 0.05, "happy": 0.05, "dance": 0.05, "clap": 0.05, "listen": 0.4}
SPAN = (0.15, 0.85)  # where the character may walk, shares of the width


class Performance:
    """What the character does in one scene, second by second."""

    def __init__(self, beats: List[Beat], width: int, height: int, char_h: int,
                 voice_level: Optional[Callable[[float], float]] = None, floating: bool = False,
                 span: Tuple[float, float] = SPAN, count_step: float = 1.0):
        self.beats, self.W, self.H, self.h = beats, width, height, char_h
        self.level = voice_level or (lambda t: 0.0)
        self.floating = floating
        self.span = span
        self.count_step = count_step  # seconds between numbers when counting
        self.end = beats[-1].start + beats[-1].seconds if beats else 0.0

    @staticmethod
    def plan(actions: List[Tuple[str, float]], width: int, start_x: float,
             span: Tuple[float, float] = SPAN, world: bool = True) -> List[Beat]:
        """Lay ``actions`` ((name, seconds)) end to end, following where the
        character goes and which way it looks. Walking scrolls the scenery;
        with ``world`` False (someone else stands in the scene) the scenery
        stays and the character walks across, up to the edge of ``span``."""
        beats, t, x, heading, scroll = [], 0.0, start_x, 1, 0.0
        for name, seconds in actions:
            if name == "enter":
                x = -0.2 * width
            v, bg = Performance._speeds(name, width)
            if name in ("walk", "run") and not world:
                room = span[1] * width - x if heading > 0 else x - span[0] * width
                v, bg = float(np.clip(room / max(seconds, 1e-6), 0.0, 2.0 * v)), 0.0
            beats.append(Beat(name, t, seconds, x, heading, scroll, v, bg))
            x, heading, scroll = Performance._after(beats[-1], width, span)
            t += seconds
        return beats

    @staticmethod
    def _speeds(name: str, width: int) -> Tuple[float, float]:
        """(on-screen speed, scenery speed) in pixels per second."""
        return {"walk": (0.05 * width, 0.22 * width), "run": (0.08 * width, 0.55 * width),
                "enter": (0.3 * width, 0.0), "exit": (0.45 * width, 0.0)}.get(name, (0.0, 0.0))

    @staticmethod
    def _enter_x(width: int, span: Tuple[float, float]) -> float:
        return min(0.36, 0.5 * (span[0] + span[1])) * width

    @staticmethod
    def _after(b: Beat, width: int, span=SPAN) -> Tuple[float, int, float]:
        if b.action == "enter":
            return Performance._enter_x(width, span), b.heading, b.scroll0
        if b.action == "exit":
            return 1.3 * width, b.heading, b.scroll0
        if b.action in ("walk", "run"):
            return (float(np.clip(b.x0 + b.heading * b.v * b.seconds, span[0] * width, span[1] * width)),
                    b.heading, b.scroll0 + b.heading * b.bg * b.seconds)
        if b.action == "look":
            return b.x0, -b.heading, b.scroll0
        return b.x0, b.heading, b.scroll0

    def beat_at(self, t: float) -> Beat:
        for b in self.beats:
            if t < b.start + b.seconds:
                return b
        return self.beats[-1]

    def at(self, t: float) -> Tuple[Pose, int, float, Beat, float]:
        """The pose, heading and scenery scroll at ``t``, the beat and its local time."""
        b = self.beat_at(t)
        u = min(max(t - b.start, 0.0), b.seconds)
        pose, heading, scroll = self._pose(b, u)
        i = self.beats.index(b)
        if i > 0 and u < 0.3:  # turn smoothly from where the last action left off
            prev = self.beats[i - 1]
            before = self._pose(prev, prev.seconds)[0].yaw
            before = (before + math.pi) % (2 * math.pi) - math.pi
            pose.yaw = before + (pose.yaw - before) * smooth(u / 0.3)
        if self.floating and b.action not in ("float", "jump"):
            pose.lift += self.h * (0.04 + 0.03 * math.sin(t * 1.6))  # weightless: a gentle drift
        return pose, heading, scroll, b, u

    def _pose(self, b: Beat, u: float) -> Tuple[Pose, int, float]:
        pose, heading, scroll = self._body(b, u)
        p = u / max(b.seconds, 1e-6)
        if b.action == "look":
            pose.yaw = 0.3 * abs(math.cos(math.pi * smooth(p)))
        elif b.action == "spin":
            pose.yaw = 0.12 + 2 * math.pi * smooth(p)
        else:
            pose.yaw = YAW.get(b.action, 0.12)
        return pose, heading, scroll

    def _body(self, b: Beat, u: float) -> Tuple[Pose, int, float]:
        h, W = self.h, self.W
        lo, hi = self.span[0] * W, self.span[1] * W
        d, p = b.seconds, u / max(b.seconds, 1e-6)
        x, heading, scroll = b.x0, b.heading, b.scroll0
        breathe = 1.0 + 0.012 * math.sin(u * 2 * math.pi * 0.55)
        a = b.action
        if a in ("walk", "run", "enter", "exit"):
            v, bg = b.v, b.bg
            if a == "enter":
                end = self._enter_x(W, self.span)
                x = b.x0 + (end - b.x0) * smooth(p) if d else end
            elif a == "exit":
                x = b.x0 + (1.3 * W - b.x0) * p ** 1.3
            else:
                x = float(np.clip(b.x0 + heading * v * u, lo, hi))
                scroll = b.scroll0 + heading * bg * u
            fast = a in ("run", "exit")
            f = 3.4 if fast else 2.2  # steps per second
            step = abs(math.sin(math.pi * f * u))
            return Pose(x, lift=step * h * (0.07 if fast else 0.04), sy=1.0 - 0.05 * (1 - step) + 0.03 * step,
                        sx=1.0 + 0.04 * (1 - step), angle=math.sin(math.pi * f * u) * (3 if fast else 4),
                        lean=0.1 if fast else 0.03), heading, scroll
        if a == "jump":
            if p < 0.2:  # crouch
                k = smooth(p / 0.2)
                return Pose(x, sx=1 + 0.12 * k, sy=1 - 0.16 * k), heading, scroll
            if p < 0.78:  # up and down
                q = (p - 0.2) / 0.58
                return Pose(x, lift=4 * q * (1 - q) * 0.5 * h, sx=0.92 + 0.08 * q, sy=1.12 - 0.12 * q,
                            angle=-6 * math.sin(math.pi * q) * heading), heading, scroll
            k = math.sin(math.pi * (p - 0.78) / 0.22)  # land
            return Pose(x, sx=1 + 0.14 * k, sy=1 - 0.18 * k), heading, scroll
        if a == "dance":
            beat = u * 2.4
            bounce = abs(math.sin(math.pi * beat))
            turn = math.cos(math.pi * smooth((beat % 4) / 4) * 2) if int(beat) % 4 == 3 else 1.0
            return Pose(x, lift=bounce * 0.06 * h, sy=1 - 0.06 * (1 - bounce), sx=1 + 0.05 * (1 - bounce),
                        angle=10 * math.sin(math.pi * beat), turn=turn), heading, scroll
        if a == "happy":
            hop = abs(math.sin(math.pi * u * 2.5)) if u < 1.2 else 0.0
            return Pose(x, lift=hop * 0.12 * h, sy=1 + 0.05 * hop, sx=1 - 0.03 * hop,
                        angle=5 * math.sin(u * 7)), heading, scroll
        if a == "sad":
            k = smooth(min(u / 0.6, 1.0))
            return Pose(x, sy=1 - 0.08 * k, sx=1 + 0.04 * k, lean=-0.06 * k,
                        angle=3 * k * math.sin(u * 1.2) - 2 * k * heading), heading, scroll
        if a == "scared":
            jitter = math.sin(u * 2 * math.pi * 11) * 0.006 * W + math.sin(u * 2 * math.pi * 7.3) * 0.003 * W
            return Pose(x + jitter, sy=0.94, sx=1.02, lean=-0.04, angle=2 * math.sin(u * 40)), heading, scroll
        if a == "surprised":
            q = min(p / 0.35, 1.0)
            k = math.sin(math.pi * q) if p < 0.35 else 0.0
            return Pose(x, lift=k * 0.12 * h, sy=1 + 0.12 * k, sx=1 - 0.06 * k), heading, scroll
        if a == "think":
            return Pose(x, sy=breathe, lean=0.035 * math.sin(u * 1.4), angle=4 * math.sin(u * 0.9)), heading, scroll
        if a == "sleep":
            k = smooth(min(u / 0.8, 1.0))
            return Pose(x, sy=1 - 0.05 * k + 0.025 * math.sin(u * 2.2), angle=9 * k * heading,
                        lean=-0.02 * k), heading, scroll
        if a == "float":
            return Pose(x + 0.03 * W * math.sin(u * 0.9), lift=h * (0.18 + 0.06 * math.sin(u * 1.8)),
                        angle=8 * math.sin(u * 1.3), sy=breathe), heading, scroll
        if a == "spin":
            q = smooth(p)
            return Pose(x, lift=math.sin(math.pi * p) * 0.08 * h, turn=math.cos(2 * math.pi * q)), heading, scroll
        if a == "look":
            q = smooth(p)
            return Pose(x, turn=math.cos(math.pi * q), lift=math.sin(math.pi * p) * 0.03 * h), -heading if q > 0.5 else heading, scroll
        if a == "talk":
            v = b.voice(u) if b.voice else self.level(u)
            return Pose(x, lift=v * 0.025 * h, sy=1 + 0.04 * v, sx=1 - 0.015 * v,
                        angle=2.5 * math.sin(u * 3)), heading, scroll
        if a == "wave":
            return Pose(x, sy=breathe, angle=3 * math.sin(u * 3)), heading, scroll
        if a == "clap":
            c = abs(math.cos(math.pi * 2.5 * u))
            return Pose(x, lift=(1 - c) * 0.025 * h, sy=1 + 0.02 * (1 - c)), heading, scroll
        if a == "point":
            return Pose(x, sy=breathe, lean=0.03), heading, scroll
        if a == "count":  # a little hop with every number
            q = (u - COUNT_LEAD) / max(self.count_step, 0.1)
            hop = math.sin(math.pi * (q % 1.0)) if q >= 0 else 0.0
            return Pose(x, lift=hop * 0.05 * h, sy=1 + 0.03 * hop, lean=0.03), heading, scroll
        if a == "brush":
            return Pose(x, sy=breathe, angle=1.5 * math.sin(u * 16)), heading, scroll
        if a == "wash":
            return Pose(x, sy=breathe, angle=2 * math.sin(u * 9)), heading, scroll
        if a == "eat":
            return Pose(x, sy=1 + 0.015 * math.sin(u * 13)), heading, scroll
        if a == "read":
            return Pose(x, sy=breathe, lean=-0.01), heading, scroll
        if a == "yawn":
            k = math.sin(math.pi * p)
            return Pose(x, sy=1 + 0.07 * k, sx=1 - 0.035 * k), heading, scroll
        if a == "angry":  # stomping and shaking
            stomp = abs(math.sin(math.pi * 3 * u))
            return Pose(x + 0.003 * W * math.sin(u * 31), lift=stomp * 0.02 * h, sy=1 - 0.04 * (1 - stomp),
                        sx=1 + 0.02 * (1 - stomp), angle=2 * math.sin(u * 14)), heading, scroll
        return Pose(x, sy=breathe, angle=1.5 * math.sin(u * 1.1)), heading, scroll


# --------------------------------------------------------------------------- effects

def _shape(draw_fn, w: int, h: int) -> Sprite:
    return _render(lambda d, s: draw_fn(d, s, w, h), (w, h))


def _heart(size: int, colour=(0.95, 0.3, 0.42)) -> Sprite:
    def draw(d: ImageDraw.ImageDraw, s, w, h):
        r = w * 0.26 * s
        d.ellipse([w * 0.06 * s, h * 0.1 * s, w * 0.06 * s + 2 * r, h * 0.1 * s + 2 * r], fill=_rgb255(colour))
        d.ellipse([w * 0.94 * s - 2 * r, h * 0.1 * s, w * 0.94 * s, h * 0.1 * s + 2 * r], fill=_rgb255(colour))
        d.polygon([(w * 0.08 * s, h * 0.42 * s), (w * 0.5 * s, h * 0.95 * s), (w * 0.92 * s, h * 0.42 * s),
                   (w * 0.5 * s, h * 0.3 * s)], fill=_rgb255(colour))
    return _shape(draw, size, size)


def _star(size: int, colour=(1.0, 0.86, 0.25), points: int = 4) -> Sprite:
    def draw(d, s, w, h):
        c = w * s / 2
        pts = []
        for i in range(points * 2):
            r = c * (1.0 if i % 2 == 0 else 0.32)
            a = math.pi * i / points - math.pi / 2
            pts.append((c + r * math.cos(a), c + r * math.sin(a)))
        d.polygon(pts, fill=_rgb255(colour))
    return _shape(draw, size, size)


def _note(size: int) -> Sprite:
    def draw(d, s, w, h):
        line = max(2, int(w * 0.09)) * s
        d.ellipse([w * 0.08 * s, h * 0.66 * s, w * 0.5 * s, h * 0.95 * s], fill=_rgb255(OUTLINE))
        d.line([(w * 0.46 * s, h * 0.8 * s), (w * 0.46 * s, h * 0.08 * s)], fill=_rgb255(OUTLINE), width=line)
        d.line([(w * 0.46 * s, h * 0.1 * s), (w * 0.9 * s, h * 0.3 * s)], fill=_rgb255(OUTLINE), width=line)
    return _shape(draw, size, size)


def _drop(size: int) -> Sprite:
    def draw(d, s, w, h):
        d.ellipse([w * 0.15 * s, h * 0.38 * s, w * 0.85 * s, h * 0.98 * s], fill=_rgb255((0.55, 0.8, 1.0)))
        d.polygon([(w * 0.18 * s, h * 0.6 * s), (w * 0.5 * s, h * 0.02 * s), (w * 0.82 * s, h * 0.6 * s)],
                  fill=_rgb255((0.55, 0.8, 1.0)))
    return _shape(draw, size, int(size * 1.3))


def _cloud(size: int) -> Sprite:
    def draw(d, s, w, h):
        for cx, cy, r in ((0.3, 0.55, 0.22), (0.52, 0.42, 0.28), (0.72, 0.56, 0.2), (0.5, 0.62, 0.22)):
            d.ellipse([(cx - r) * w * s, (cy - r) * h * s * 1.6, (cx + r) * w * s, (cy + r) * h * s * 1.6],
                      fill=_rgb255((0.55, 0.58, 0.65)))
    return _shape(draw, size, int(size * 0.55))


def _bulb(size: int) -> Sprite:
    def draw(d, s, w, h):
        d.ellipse([w * 0.2 * s, h * 0.1 * s, w * 0.8 * s, h * 0.66 * s], fill=_rgb255((1.0, 0.9, 0.3)),
                  outline=_rgb255(OUTLINE), width=max(2, int(w * 0.05)) * s)
        d.rectangle([w * 0.36 * s, h * 0.64 * s, w * 0.64 * s, h * 0.82 * s], fill=_rgb255((0.6, 0.6, 0.65)),
                     outline=_rgb255(OUTLINE), width=max(1, int(w * 0.03)) * s)
    return _shape(draw, size, int(size * 1.0))


def _glyph(text: str, size: int, colour=WHITE) -> Sprite:
    return text_sprite(text, TITLE_FONTS, size, colour, stroke=max(2, size // 12), stroke_fill=OUTLINE)


class Effects:
    """Little drawings that show feelings, around the character."""

    def __init__(self, char_h: int, width: int, height: int):
        u = char_h
        self.W, self.H = width, height
        self.heart = _heart(int(0.16 * u))
        self.star = _star(int(0.14 * u))
        self.spark = _star(int(0.1 * u), (1.0, 1.0, 0.85))
        self.note = _note(int(0.15 * u))
        self.drop = _drop(int(0.07 * u))
        self.cloud = _cloud(int(0.5 * u))
        self.bulb = _bulb(int(0.2 * u))
        self.bang = _glyph("!", int(0.3 * u), (1.0, 0.85, 0.2))
        self.what = _glyph("?", int(0.3 * u), (0.6, 0.85, 1.0))
        self.zzz = _glyph("Z", int(0.14 * u), (0.85, 0.9, 1.0))
        self.soap = PR.prop_sprite("bubble", int(0.09 * u))
        rng = np.random.default_rng(3)
        self.confetti = [(rng.uniform(0, width), rng.uniform(-height, 0), rng.uniform(0.25, 0.5) * height,
                          rng.uniform(0, 6), [(0.95, 0.3, 0.4), (1.0, 0.8, 0.2), (0.3, 0.7, 1.0),
                                              (0.4, 0.85, 0.4), (0.8, 0.45, 0.95)][rng.integers(5)])
                         for _ in range(70)]
        self.u = u

    def draw(self, frame: np.ndarray, name: str, t: float, head: Tuple[float, float], body: float,
             fade: float = 1.0) -> None:
        """``t`` seconds into effect ``name``; ``head`` is the top of the head and
        ``body`` the height of the character on screen."""
        hx, hy = head
        u = self.u
        if fade <= 0:
            return
        if name == "hearts":
            for i in range(3):
                q = ((t * 0.6 + i / 3) % 1.0)
                x = hx + (i - 1) * 0.3 * u + 0.06 * u * math.sin(t * 3 + i)
                self.heart.paste(frame, x, hy - q * 0.6 * u, scale=0.6 + 0.6 * q,
                                 opacity=fade * min(1.0, 4 * q) * (1 - q))
        elif name in ("stars", "sparkles"):
            sprite = self.star if name == "stars" else self.spark
            for i in range(5):
                a = i * 1.26 + t * 0.4
                x = hx + math.cos(a) * 0.55 * u
                y = hy + body * 0.45 + math.sin(a) * 0.55 * body
                k = 0.5 + 0.5 * math.sin(t * 6 + i * 2)
                sprite.paste(frame, x, y, scale=0.5 + 0.7 * k, opacity=fade * k, angle=t * 60)
        elif name == "music":
            for i in range(3):
                q = ((t * 0.5 + i / 3) % 1.0)
                x = hx + 0.35 * u + 0.15 * u * math.sin(t * 2 + i * 2) - i * 0.12 * u
                self.note.paste(frame, x, hy - 0.1 * u - q * 0.5 * u, opacity=fade * min(1, 4 * q) * (1 - q),
                                angle=15 * math.sin(t * 4 + i))
        elif name == "exclaim":
            k = back(_phase(t, 0.0, 0.25))
            self.bang.paste(frame, hx + 0.25 * u, hy - 0.2 * u, scale=k, opacity=fade, angle=-10)
        elif name == "question":
            self.what.paste(frame, hx + 0.3 * u, hy - 0.22 * u - 0.03 * u * math.sin(t * 3),
                            scale=back(_phase(t, 0.0, 0.3)), opacity=fade, angle=10 * math.sin(t * 2))
        elif name == "zzz":
            for i in range(3):
                q = ((t * 0.45 + i / 3) % 1.0)
                self.zzz.paste(frame, hx + 0.2 * u + q * 0.3 * u, hy - q * 0.5 * u, scale=0.5 + 0.8 * q,
                               opacity=fade * min(1, 4 * q) * (1 - q))
        elif name == "soap":
            for i in range(6):
                q = ((t * 0.5 + i / 6) % 1.0)
                x = hx + (i - 2.5) * 0.09 * u + 0.04 * u * math.sin(t * 2 + i)
                self.soap.paste(frame, x, hy + 0.6 * body - q * 0.7 * u, scale=0.5 + 0.7 * q,
                                opacity=fade * min(1.0, 5 * q) * (1 - q) ** 0.5)
        elif name == "steam":  # puffs from both sides of the head
            for i in range(4):
                q = ((t * 0.9 + i / 4) % 1.0)
                side = -1 if i % 2 else 1
                self.cloud.paste(frame, hx + side * (0.22 + 0.15 * q) * u, hy + 0.12 * u - q * 0.25 * u,
                                 scale=0.25 + 0.35 * q, opacity=fade * 0.8 * (1 - q))
        elif name == "sweat":
            for i in range(2):
                q = ((t * 0.9 + i / 2) % 1.0)
                self.drop.paste(frame, hx + (0.3 if i else -0.25) * u, hy + 0.1 * u + q * 0.25 * u,
                                opacity=fade * (1 - q))
        elif name == "raincloud":
            cx, cy = hx, hy - 0.32 * u
            self.cloud.paste(frame, cx, cy, opacity=fade * 0.95)
            for i in range(6):
                q = ((t * 1.6 + i / 6) % 1.0)
                x = int(cx - 0.18 * u + i * 0.07 * u)
                y = int(cy + 0.08 * u + q * 0.3 * u)
                cv2.line(frame, (x, y), (x - 2, y + int(0.05 * u)), (0.55, 0.75, 1.0), max(2, int(u / 120)),
                         cv2.LINE_AA)
        elif name == "idea":
            k = back(_phase(t, 0.0, 0.3))
            for i in range(6):  # rays
                a = -math.pi / 2 + (i - 2.5) * 0.4
                r0, r1 = 0.17 * u, 0.24 * u + 0.02 * u * math.sin(t * 8)
                cx, cy = hx, hy - 0.25 * u
                cv2.line(frame, (int(cx + math.cos(a) * r0), int(cy + math.sin(a) * r0)),
                         (int(cx + math.cos(a) * r1), int(cy + math.sin(a) * r1)), (1.0, 0.85, 0.3),
                         max(2, int(u / 70)), cv2.LINE_AA)
            self.bulb.paste(frame, hx, hy - 0.25 * u, scale=k, opacity=fade)
        elif name == "confetti":
            for x0, y0, speed, p, colour in self.confetti:
                y = (y0 + speed * t) % (self.H * 1.1) - 0.05 * self.H
                x = x0 + 0.02 * self.W * math.sin(t * 2 + p)
                w = max(3, int(0.012 * u * (1 + abs(math.sin(t * 5 + p)))))
                cv2.rectangle(frame, (int(x), int(y)), (int(x) + w, int(y) + max(3, int(0.02 * u))),
                              tuple(c * fade + 0.5 * (1 - fade) for c in colour), -1)
        elif name == "dust":
            for i in range(3):
                q = min(t / 0.6, 1.0)
                r = int(0.05 * u + 0.1 * u * q)
                x = int(hx + (i - 1) * 0.25 * u * (0.5 + q))
                if q < 1:
                    overlay = frame.copy()
                    cv2.circle(overlay, (x, int(body)), r, (0.9, 0.88, 0.82), -1, cv2.LINE_AA)
                    frame += (overlay - frame) * 0.5 * (1 - q) * fade


# --------------------------------------------------------------------------- words

def _wrap(text: str, fonts, size: int, width: float) -> List[str]:
    font = _font(fonts, size)
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.getlength(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


class Bubble:
    """A speech bubble whose words appear as they are said."""

    def __init__(self, text: str, unit: int, max_width: float):
        size = int(0.05 * unit)
        self.lines = _wrap(text, SPEECH_FONTS, size, max_width)
        self.sprites = [text_sprite(line, SPEECH_FONTS, size, OUTLINE) for line in self.lines]
        pad = int(0.6 * size)
        self.w = int(max(s.size[0] for s in self.sprites) + 2 * pad)
        self.h = int(sum(s.size[1] * 0.92 for s in self.sprites) + 1.6 * pad)
        self.tail = int(0.9 * size)
        self.pad = pad
        self.shapes = {side: self._shape(side) for side in (-1, 1)}
        self.chars = [len(line) for line in self.lines]

    def _shape(self, side: int) -> Sprite:
        w, h, tail = self.w, self.h, self.tail

        def draw(d: ImageDraw.ImageDraw, s: int) -> None:
            line = max(2, int(h * 0.03)) * s
            d.rounded_rectangle([line, line, w * s - line, h * s - line], radius=h * 0.3 * s,
                                fill=(255, 255, 255, 255), outline=_rgb255(OUTLINE), width=line)
            tx = w * (0.22 if side > 0 else 0.78) * s
            pts = [(tx - tail * 0.5 * s, h * s - 2 * line), (tx + (-tail if side > 0 else tail) * 0.9 * s,
                                                              (h + tail) * s - line),
                   (tx + tail * 0.5 * s, h * s - 2 * line)]
            d.polygon(pts, fill=(255, 255, 255, 255))
            d.line(pts[:2], fill=_rgb255(OUTLINE), width=line)
            d.line(pts[1:], fill=_rgb255(OUTLINE), width=line)

        return _render(draw, (w, h + tail))

    def draw(self, frame: np.ndarray, head: Tuple[float, float], said: float, pop: float,
             fade: float = 1.0, room: Optional[Tuple[float, float, float]] = None) -> None:
        """``said``: the share of the line spoken so far; ``pop``: 0..1 appearing;
        ``fade``: 1..0 going away (the words go with the bubble); ``room``:
        (left, right, top) the bubble must keep within (pixels)."""
        H, W = frame.shape[:2]
        hx, hy = head
        left, right, top = room or (0.02 * W, 0.98 * W, 0.03 * H)
        side = 1 if hx < (left + right) / 2 + 0.05 * W else -1  # the bubble on the roomier side
        x = hx + (0.05 * self.w if side > 0 else -1.05 * self.w)
        x = float(np.clip(x, left, max(left, right - self.w)))
        side = 1 if x + 0.5 * self.w > hx else -1
        y = float(np.clip(hy - self.h - self.tail - 0.02 * H, top, H - self.h - self.tail))
        shape = self.shapes[side]
        shape.paste(frame, x + self.w / 2, y + (self.h + self.tail) / 2, scale=0.4 + 0.6 * back(pop),
                    opacity=min(1.0, 2 * pop) * fade)
        if pop < 1.0 or fade <= 0.0:
            return
        left = said * sum(self.chars)
        ty = y + self.pad * 0.8
        for sprite, n in zip(self.sprites, self.chars):
            share = min(max(left / max(n, 1), 0.0), 1.0)
            sprite.paste_left(frame, x + self.pad, ty, reveal=share if share < 1 else 1.0, opacity=fade)
            left -= n
            ty += sprite.size[1] * 0.92


class Captions:
    """Subtitles for the narration, a sentence or two at a time."""

    def __init__(self, text: str, seconds: float, start: float, width: int, height: int):
        unit = min(width, height)
        self.size = int(0.042 * unit)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        chunks: List[str] = []
        for sentence in sentences:
            lines = _wrap(sentence, CAPTION_FONTS, self.size, 0.86 * width)
            for i in range(0, len(lines), 2):
                chunks.append(" ".join(lines[i:i + 2]))
        total = sum(len(c) for c in chunks) or 1
        self.items, t = [], start
        for chunk in chunks:
            span = seconds * len(chunk) / total
            lines = _wrap(chunk, CAPTION_FONTS, self.size, 0.86 * width)
            sprites = [text_sprite(line, CAPTION_FONTS, self.size, WHITE, stroke=max(2, self.size // 9),
                                   stroke_fill=(0.08, 0.07, 0.1)) for line in lines]
            self.items.append((t, t + span, sprites))
            t += span
        self.W, self.H = width, height

    def draw(self, frame: np.ndarray, t: float) -> None:
        for t0, t1, sprites in self.items:
            if t0 <= t < t1 + 0.25:
                a = min(_phase(t, t0, 0.15), 1.0 - _phase(t, t1, 0.25))
                y = self.H - 0.06 * self.H - (len(sprites) - 1) * self.size * 1.25
                for s in sprites:
                    s.paste(frame, self.W / 2, y, opacity=a)
                    y += self.size * 1.25


# --------------------------------------------------------------------------- the chalkboard

CHALK = (0.97, 0.96, 0.9)
COLOUR_WORDS = {"red": (1.0, 0.38, 0.35), "orange": (1.0, 0.64, 0.22), "yellow": (1.0, 0.9, 0.3),
                "green": (0.47, 0.9, 0.42), "blue": (0.45, 0.7, 1.0), "purple": (0.8, 0.58, 1.0),
                "pink": (1.0, 0.64, 0.82), "brown": (0.82, 0.58, 0.38), "grey": (0.78, 0.78, 0.8),
                "gray": (0.78, 0.78, 0.8)}
COUNT_LEAD = 0.35  # from the start of counting to the first number
QUIZ_COUNT = 3.2  # the countdown before a quiz's answer shows
QUIZ_REVEAL = 1.6  # then the answer


def board_box(W: int, H: int, ground: float, char_h: int) -> Tuple[float, float, float, float]:
    """Where the chalkboard goes: on the right in a wide frame, else above the character."""
    if W > 1.1 * H:
        return 0.5 * W, 0.07 * H, 0.95 * W, 0.58 * H
    x0, x1, y0 = 0.06 * W, 0.94 * W, 0.05 * H
    bh = min(0.62 * (x1 - x0), ground - char_h - 0.05 * H - y0)
    return x0, y0, x1, y0 + max(bh, 0.16 * H)


class Board:
    """A chalkboard with words (colour names in their own colour), a picture,
    or things popping up one by one to be counted."""

    def __init__(self, text: str, item: str, count: int, box: Tuple[float, float, float, float],
                 appear: float, count_times: List[float], quiz: Sequence[str] = (), answer: str = "",
                 quiz_at: float = 0.0, fade_at: Optional[float] = None):
        x0, y0, x1, y1 = box
        self.box, self.appear, self.times = box, appear, list(count_times)
        self.quiz_at, self.fade_at, self.answer = quiz_at, fade_at, answer
        if quiz:
            item, count = "", 0  # the choices take the board's picture space
        bw, bh = x1 - x0, y1 - y0
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.slate = self._slate(int(bw), int(bh))
        pad = 0.1 * min(bw, bh)
        inner = bw - 2 * pad
        pictures = bool(item) or count > 0 or bool(quiz)
        size = 0.12 if quiz else 0.14 if pictures else 0.2
        self.lines = self._words(text, int(bh * size), inner) if text else []
        text_h = sum(h for _, h, _ in self.lines)
        top = y0 + pad if pictures or not self.lines else self.cy - text_h / 2
        self.text_top = top
        area0 = top + text_h + (0.04 * bh if self.lines else 0.0)
        area1 = y1 - pad
        self.items: List[Tuple[float, float, Optional[Sprite], Optional[Sprite]]] = []
        self.picture: Optional[Sprite] = None
        self.cell = 0.0
        if count > 0:
            rows = 1 if count <= 5 else 2
            per = math.ceil(count / rows)
            tall = 1.35 if item else 1.0
            cell = min(inner / per, (area1 - area0) / rows / tall)
            self.cell = cell
            area0 += max(0.0, (area1 - area0 - rows * cell * tall) / 2)  # centred in the space left
            for k in range(count):
                r, c = divmod(k, per)
                in_row = min(per, count - r * per)
                x = self.cx + (c - (in_row - 1) / 2) * cell
                y = area0 + (r + 0.5) * cell * tall - (0.15 * cell if item else 0.0)
                number = text_sprite(str(k + 1), TITLE_FONTS, int((0.3 if item else 0.7) * cell), CHALK)
                thing = PR.prop_sprite(item, int(0.8 * cell)) if item else None
                self.items.append((x, y, thing, number))
        elif item:
            size = int(0.9 * min(inner, area1 - area0))
            self.picture = PR.prop_sprite(item, size)
            self.picture_at = (self.cx, (area0 + area1) / 2)
            self.cell = size
        self.choices: List[Tuple[float, float, Sprite, str]] = []
        if quiz:
            n = len(quiz)
            cell = max(min(inner / n, (area1 - area0) * 0.95), 0.3 * min(bw, bh), 12.0)
            size = int(0.78 * cell)
            for k, name in enumerate(quiz):
                self.choices.append((self.cx + (k - (n - 1) / 2) * cell, (area0 + area1) / 2,
                                     PR.prop_sprite(name, size), name))
            self.cell = cell
            r = int(0.11 * min(bw, bh))
            self.badge_at = (x1 - pad - 0.4 * r, y0 + pad + 0.4 * r)
            self.badge = _shape(lambda d, s_, w, h: d.ellipse([2 * s_, 2 * s_, (w - 2) * s_, (h - 2) * s_],
                                                            fill=(255, 205, 60, 255), outline=_rgb255(OUTLINE),
                                                            width=max(2, w // 18) * s_), 2 * r, 2 * r)
            self.digits = [text_sprite(str(k), TITLE_FONTS, int(1.3 * r), OUTLINE) for k in (3, 2, 1)]
            ring = int(0.95 * cell)
            self.ring = _shape(lambda d, s_, w, h: d.ellipse([4 * s_, 4 * s_, (w - 4) * s_, (h - 4) * s_],
                                                           outline=(110, 230, 120, 255), width=max(3, w // 22) * s_),
                               ring, ring)
            self.cross = _shape(lambda d, s_, w, h: (d.line([(0.2 * w * s_, 0.2 * h * s_), (0.8 * w * s_, 0.8 * h * s_)],
                                                            fill=(240, 80, 80, 255), width=max(3, w // 9) * s_),
                                                     d.line([(0.8 * w * s_, 0.2 * h * s_), (0.2 * w * s_, 0.8 * h * s_)],
                                                            fill=(240, 80, 80, 255), width=max(3, w // 9) * s_)),
                                int(0.5 * cell), int(0.5 * cell))

    @staticmethod
    def _slate(w: int, h: int) -> Sprite:
        def draw(d: ImageDraw.ImageDraw, s: int) -> None:
            r, m = 0.07 * min(w, h), 0.055 * min(w, h)
            line = max(2, int(min(w, h) * 0.012)) * s
            d.rounded_rectangle([line, line, w * s - line, h * s - line], radius=r * s, fill=(166, 110, 62, 255),
                                outline=_rgb255(OUTLINE), width=line)
            d.rounded_rectangle([m * s, m * s, (w - m) * s, (h - m) * s], radius=0.5 * r * s,
                                fill=(44, 90, 70, 255), outline=(112, 70, 38, 255), width=line)
            rng = np.random.default_rng(5)
            for _ in range(7):  # old chalk, wiped away
                cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.2, 0.8) * h
                rx, ry = rng.uniform(0.08, 0.2) * w, rng.uniform(0.03, 0.07) * h
                d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], fill=(52, 99, 79, 255))
        return _render(draw, (w, h))

    @staticmethod
    def _words(text: str, size: int, width: float) -> List[Tuple[List[Sprite], float, float]]:
        """Lines of word sprites, colour names in their colour."""
        sprites = []
        for word in text.split():
            key = re.sub(r"[^a-z]", "", word.lower())
            sprites.append(text_sprite(word, TITLE_FONTS, size, COLOUR_WORDS.get(key, CHALK),
                                       max_width=int(width)))
        space = 0.25 * size
        lines, line, used = [], [], 0.0
        for sp in sprites:
            w = sp.size[0]
            if line and used + space + w > width:
                lines.append(line)
                line, used = [], 0.0
            used += (space if line else 0.0) + w
            line.append(sp)
        if line:
            lines.append(line)
        out = []
        for ln in lines[:3]:
            out.append((ln, max(sp.size[1] for sp in ln) * 0.95, sum(sp.size[0] for sp in ln) + space * (len(ln) - 1)))
        return out

    def draw(self, frame: np.ndarray, t: float) -> None:
        a = _phase(t, self.appear, 0.35)
        fade = 1.0 - _phase(t, self.fade_at, 0.4) if self.fade_at is not None else 1.0
        if a <= 0.0 or fade <= 0.0:
            return
        self.slate.paste(frame, self.cx, self.cy, scale=0.6 + 0.4 * back(a), opacity=min(1.0, 2 * a) * fade)
        if a < 1.0:
            return
        show = _phase(t, self.appear + 0.3, 0.3) * fade
        y = self.text_top
        for line, h, w in self.lines:
            x = self.cx - w / 2
            for sp in line:
                sp.paste_left(frame, x, y, opacity=show)
                x += sp.size[0] + 0.25 * sp.size[1] / 1.3
            y += h
        if self.picture is not None:
            q = _phase(t, self.appear + 0.45, 0.35)
            self.picture.paste(frame, self.picture_at[0],
                               self.picture_at[1] + 0.03 * self.cell * math.sin(2 * t),
                               scale=back(q), opacity=min(1.0, 3 * q) * fade)
        if self.choices:
            reveal = self.quiz_at + QUIZ_COUNT
            r = _phase(t, reveal, 0.35)
            for k, (x, y, sprite, name) in enumerate(self.choices):
                q = _phase(t, self.appear + 0.45 + 0.15 * k, 0.3)
                right = name == self.answer
                grow = 1.0 + (0.15 * back(r) if right else -0.12 * r)
                bob = 0.03 * self.cell * math.sin(2.5 * t + k) if r <= 0 else \
                    (-0.12 * self.cell * math.sin(math.pi * min(r * 2, 1.0)) if right else 0.0)
                if right and r > 0:
                    self.ring.paste(frame, x, y, scale=0.6 + 0.4 * back(r), opacity=min(1.0, 2 * r) * fade)
                sprite.paste(frame, x, y + bob, scale=back(q) * grow, opacity=min(1.0, 3 * q) * fade *
                             (1.0 - 0.6 * r if not right else 1.0))
                if not right and r > 0:
                    self.cross.paste(frame, x, y, scale=back(r), opacity=r * fade)
            k = int(t - self.quiz_at)
            if 0 <= t - self.quiz_at < QUIZ_COUNT and k < 3:
                pop = _phase(t - self.quiz_at - k, 0.0, 0.2)
                self.badge.paste(frame, *self.badge_at, scale=0.8 + 0.2 * back(pop), opacity=fade)
                self.digits[k].paste(frame, *self.badge_at, scale=back(pop), opacity=fade)
        last = self.times[-1] if self.times else 0.0
        for k, (x, y, thing, number) in enumerate(self.items):
            at = self.times[k] if k < len(self.times) else self.appear + 0.5 + 0.3 * k
            q = _phase(t, at, 0.22)
            if q <= 0.0:
                continue
            j = _phase(t, last + 0.55 + 0.07 * k, 0.4)
            dy = -0.22 * self.cell * math.sin(math.pi * j) if 0.0 < j < 1.0 else 0.0
            if thing is not None:
                thing.paste(frame, x, y + dy, scale=back(q), opacity=min(1.0, 3 * q) * fade)
                number.paste(frame, x, y + 0.62 * self.cell + dy, scale=back(q), opacity=min(1.0, 3 * q) * fade)
            else:
                number.paste(frame, x, y + dy, scale=back(q), opacity=min(1.0, 3 * q) * fade)



# --------------------------------------------------------------------------- voices

@functools.lru_cache(maxsize=1)
def voices() -> Tuple[str, ...]:
    """English voices of the macOS speech synthesiser (none elsewhere)."""
    say = shutil.which("say")
    if not say:
        return ()
    try:
        out = subprocess.run([say, "-v", "?"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    found = []
    for line in out.splitlines():
        m = re.match(r"(.+?)\s+(en_[A-Z]{2})\s+#", line)
        if m:
            found.append(m.group(1).strip())
    return tuple(found)


def speak(text: str, voice: str, out_dir: str, name: str, pitch: float = 1.0, rate: int = 0) -> np.ndarray:
    """``text`` read aloud by ``voice`` (mono samples), pitched up or down by ``pitch``."""
    say = shutil.which("say")
    if not say:
        raise RenderError("Reading aloud needs the macOS 'say' command.")
    aiff = os.path.join(out_dir, f"{name}.aiff")
    cmd = [say, "-v", voice, "-o", aiff] + (["-r", str(rate)] if rate else []) + [text]
    for attempt in range(3):  # the speech service now and then hangs; a second try works
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
            break
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise RenderError(f"The voice '{voice}' did not answer; please try again.")
    if proc.returncode != 0 or not os.path.exists(aiff):
        raise RenderError(f"The voice '{voice}' could not read the text: {proc.stderr.strip()[:200]}")
    if abs(pitch - 1.0) < 0.01:
        return decode_audio(aiff)
    wav = os.path.join(out_dir, f"{name}.wav")
    f = float(np.clip(pitch, 0.6, 1.8))
    proc = subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", aiff,
                           "-af", f"aresample={SAMPLE_RATE},asetrate={SAMPLE_RATE * f:.0f},"
                                  f"aresample={SAMPLE_RATE},atempo={1 / f:.4f}", wav],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RenderError(f"Could not change the voice's pitch: {proc.stderr.strip()[-200:]}")
    return decode_audio(wav)


def _level(samples: Optional[np.ndarray]) -> Callable[[float], float]:
    """How loud the voice is at a time (0..1), for the character's talking bob."""
    if samples is None or not len(samples):
        return lambda t: 0.5 + 0.5 * math.sin(t * 2 * math.pi * 4)
    hop = SAMPLE_RATE // 30
    n = len(samples) // hop
    if n == 0:
        return lambda t: 0.0
    rms = np.sqrt((samples[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1))
    env = np.convolve(rms, np.ones(3) / 3, "same")
    env /= np.percentile(env, 95) + 1e-6

    def level(t: float) -> float:
        i = int(t * 30)
        return float(np.clip(env[i], 0, 1)) if 0 <= i < len(env) else 0.0

    return level


# --------------------------------------------------------------------------- shots

def _iris(frame: np.ndarray, centre: Tuple[float, float], share: float, grid) -> np.ndarray:
    """Black outside a circle around ``centre``; ``share`` 1 = fully open."""
    if share >= 1.0:
        return frame
    H, W = frame.shape[:2]
    yy, xx = grid
    r = share * math.hypot(max(centre[0], W - centre[0]), max(centre[1], H - centre[1]))
    inside = np.clip((r - np.hypot(xx - centre[0], yy - centre[1])) / 3.0, 0.0, 1.0)[..., None]
    return frame * inside


@dataclass
class _Said:
    """A line of the conversation, timed within its scene."""

    line: Line
    start: float
    seconds: float
    audio: Optional[np.ndarray]
    level: Callable[[float], float]
    bubble: Optional[Bubble]
    captions: Optional[Captions]  # the narrator's lines are subtitles


@dataclass
class _Friend:
    name: str
    rig: "CA.Character"
    x: float
    mood: str  # sad, scared, sleepy or ""


@dataclass
class _Scene:
    shot: Shot
    backdrop: SC.Backdrop
    show: Performance
    seconds: float
    narration: Optional[np.ndarray]
    narration_at: float
    captions: Optional[Captions]
    sounds: List[Tuple[float, str]]
    effect_at: float  # when the scene's effect appears
    scale: float = 1.0  # the characters' size
    board: Optional[Board] = None
    counting: List[Tuple[float, Optional[np.ndarray]]] = field(default_factory=list)  # (time, "One!")
    said: List[_Said] = field(default_factory=list)
    friends: List[_Friend] = field(default_factory=list)
    room: Optional[Tuple[float, float, float]] = None  # where speech bubbles may go while the board shows
    board_off: float = 1e9  # when the board fades away


IRIS = 0.5
GAP = 0.6  # after each spoken line
MOOD_ACTION = {"sad": "sad", "scared": "scared", "sleepy": "sleep"}


def _timeline(shot: Shot, narration_s: float, said: List[Tuple[bool, float, str]], count_s: float = 0.0,
              quiz: bool = False) -> Tuple[List[Tuple[str, float]], int]:
    """The hero's actions with their lengths, and where the conversation starts.

    The actions before "talk" fill the narration; then come the counting, the
    quiz, the conversation (``said``: (is the hero speaking, seconds, action)),
    with the hero listening to the others, and the actions after "talk"."""
    acts = list(shot.actions)
    cut = acts.index("talk") if "talk" in acts else len(acts)
    before, after = acts[:cut], acts[cut + 1:]
    if not said:
        after = [a for a in after if a != "talk"]
    if not before:
        before = ["idle"]
    need = max(narration_s + 0.7, 2.2 if narration_s else 1.0) + IRIS
    fixed = sum(ACTIONS[a][0] for a in before if not ACTIONS[a][1])
    stretch = [a for a in before if ACTIONS[a][1]]
    if not stretch and fixed < need:
        before.append("idle")
        stretch = ["idle"]
    usual = sum(ACTIONS[a][0] for a in stretch)
    k = max(1.0, (need - fixed) / max(usual, 1e-6))
    timed = [(a, ACTIONS[a][0] * (k if ACTIONS[a][1] else 1.0)) for a in before]
    if count_s > 0:
        timed.append(("count", count_s))
    if quiz:
        timed += [("think", QUIZ_COUNT), ("happy", QUIZ_REVEAL)]
    first = len(timed)
    for hero, seconds, do in said:
        timed.append(((do or "talk") if hero else "listen", seconds + GAP))
    timed += [(a, ACTIONS[a][0]) for a in after if a != "talk"]
    return timed, first


def _timed(shot: Shot, narration_s: float, line_s: float, count_s: float = 0.0) -> List[Tuple[str, float]]:
    """The hero's actions for a scene whose only line is the hero's ``say``."""
    said = [(True, line_s, "")] if shot.say else []
    return _timeline(shot, narration_s, said, count_s)[0]


ACTION_SOUNDS = {"jump": [(0.2, "boing")], "spin": [(0.0, "whoosh")], "surprised": [(0.0, "pop")],
                 "happy": [(0.05, "sparkle")], "run": [(0.0, "whoosh")], "exit": [(0.0, "whoosh")],
                 "look": [(0.0, "swish")], "float": [(0.0, "sparkle")], "wash": [(0.0, "splash")],
                 "clap": [(0.2 + 0.4 * k, "clap") for k in range(4)], "yawn": [(0.3, "swish")]}
EFFECT_SOUNDS = {"hearts": "sparkle", "stars": "sparkle", "confetti": "pop", "idea": "ding",
                 "question": "pop", "exclaim": "pop", "music": "pop"}


def _rhythm(beat: Beat) -> List[Tuple[float, str]]:
    """Sounds that repeat through an action: footsteps, bites, scrubbing."""
    a = beat.action
    if a in ("walk", "enter", "exit", "run"):
        f = 3.4 if a in ("run", "exit") else 2.2
        return [(beat.start + i / f, "step") for i in range(1, int(beat.seconds * f))]
    if a == "eat":  # the hand reaches the mouth half way through each 1.1 s bite
        return [(beat.start + 0.55 + 1.1 * i, "crunch") for i in range(int((beat.seconds - 0.55) / 1.1) + 1)]
    if a in ("brush", "wash"):
        return [(beat.start + 0.3 + 0.7 * i, "scrub") for i in range(int((beat.seconds - 0.3) / 0.7))]
    return []


def _layout(W: int, H: int, friends: int, board: bool) -> Tuple[float, Tuple[float, float], List[float]]:
    """(the characters' size, where the hero may go, where the friends stand), shares of the width."""
    wide, tall = W > 1.1 * H, H > 1.1 * W
    scale = ((1.0, 1.0, 0.92, 0.85) if wide else (1.0, 0.85, 0.72, 0.62) if tall else (1.0, 0.9, 0.8, 0.7))[friends]
    if board and wide and friends:
        scale = min(scale, 0.8)
    if not friends:
        return scale, ((0.1, 0.42) if board and wide else SPAN), []
    slots = {1: [0.68], 2: [0.6, 0.8], 3: [0.55, 0.71, 0.87]}[friends] if wide else \
        {1: [0.7], 2: [0.56, 0.82], 3: [0.48, 0.68, 0.87]}[friends]
    hi = slots[0] - (0.17 if wide else 0.24) * scale
    return scale, (0.06 if wide else 0.08, max(hi, 0.2)), slots


# --------------------------------------------------------------------------- video

@dataclass
class CartoonResult:
    output_path: str
    image_path: str
    duration: float
    scenes: int
    elapsed: float
    warnings: List[str] = field(default_factory=list)


def _banner(text: str, unit: int, width: int, icon: str = "star", size_k: float = 0.045) -> Sprite:
    """Words on a little paper banner with a star (the lesson) or an arrow (next time)."""
    size = int(size_k * unit)
    lines = _wrap(text, SPEECH_FONTS, size, 0.72 * width)[:3]
    sprites = [text_sprite(line, SPEECH_FONTS, size, OUTLINE) for line in lines]
    pad = int(0.7 * size)
    w = int(max(sp.size[0] for sp in sprites) + 2 * pad + 1.4 * size)
    h = int(sum(sp.size[1] * 0.95 for sp in sprites) + 1.2 * pad)

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        line = max(2, int(size * 0.08)) * s
        d.rounded_rectangle([line, line, w * s - line, h * s - line], radius=0.3 * min(h, 3 * size) * s,
                            fill=(255, 250, 230, 255), outline=_rgb255(OUTLINE), width=line)
        c, r = (pad + 0.45 * size) * s, 0.5 * size * s
        if icon == "star":
            pts = []
            for i in range(10):
                rr = r * (1.0 if i % 2 == 0 else 0.45)
                a = math.pi * i / 5 - math.pi / 2
                pts.append((c + rr * math.cos(a), h * s / 2 + rr * math.sin(a)))
            d.polygon(pts, fill=(255, 205, 60, 255), outline=_rgb255(OUTLINE))
        else:
            d.polygon([(c - r, h * s / 2 - 0.8 * r), (c + r, h * s / 2), (c - r, h * s / 2 + 0.8 * r)],
                      fill=(110, 200, 255, 255), outline=_rgb255(OUTLINE))

    banner = _render(draw, (w, h))
    canvas = banner.rgb.copy()
    y = 0.6 * pad
    for sp in sprites:
        x0, y0 = int(pad + 1.3 * size), int(y)
        sw, sh = sp.size
        sh_ = min(sh, h - y0)
        a = sp.alpha[:sh_, :min(sw, w - x0)]
        region = canvas[y0:y0 + sh_, x0:x0 + a.shape[1]]
        region *= 1.0 - a
        region += sp.rgb[:sh_, :a.shape[1]] * a
        y += sh * 0.95
    return Sprite(np.dstack([canvas, banner.alpha]))


def _lesson_sprite(text: str, unit: int, width: int) -> Sprite:
    return _banner(text, unit, width)


def _blossom(size: int, on: bool) -> Sprite:
    def draw(d: ImageDraw.ImageDraw, s: int, w: int, h: int) -> None:
        c, r = w * s / 2, w * s * 0.2
        petal = (255, 140, 180, 255) if on else (255, 255, 255, 70)
        line = (110, 60, 90, 255) if on else (255, 255, 255, 150)
        for j in range(5):
            a = 2 * math.pi * j / 5 - math.pi / 2
            px, py = c + math.cos(a) * r * 1.1, c + math.sin(a) * r * 1.1
            d.ellipse([px - r, py - r, px + r, py + r], fill=petal, outline=line, width=max(1, w // 24) * s)
        d.ellipse([c - 0.75 * r, c - 0.75 * r, c + 0.75 * r, c + 0.75 * r],
                  fill=(255, 214, 64, 255) if on else (255, 255, 255, 90), outline=line, width=max(1, w // 24) * s)
    return _shape(draw, size, size)


def _vignette(W: int, H: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(F32)
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / math.sqrt(2)
    return (1.0 - 0.22 * np.clip(r, 0, 1) ** 2.4)[..., None].astype(F32)


SERIES = "Peeko & Friends"
LETTER_COLOURS = ((1.0, 0.45, 0.45), (1.0, 0.7, 0.22), (1.0, 0.86, 0.25), (0.48, 0.85, 0.42), (0.42, 0.72, 1.0),
                  (0.72, 0.55, 1.0))


def _logo_letters(text: str, size: int, colours) -> List[Tuple[Sprite, Sprite]]:
    """Each letter twice: a dark shadow and the coloured letter with a white rim."""
    out = []
    for k, ch in enumerate(text):
        if ch == " ":
            out.append((None, None))
            continue
        colour = colours[k % len(colours)]
        rim = text_sprite(ch, TITLE_FONTS, size, colour, stroke=max(3, size // 9), stroke_fill=WHITE)
        shadow = text_sprite(ch, TITLE_FONTS, size, OUTLINE, stroke=max(4, size // 7), stroke_fill=OUTLINE)
        out.append((rim, shadow))
    return out


class SeriesIntro:
    """The show's opening: the logo's letters drop in one by one on the beat,
    the friends pop up with their names, everyone waves, ta-da."""

    def __init__(self, W: int, H: int, char_h: int, hero: Optional[object], name: str, beats: List[float]):
        self.W, self.H = W, H
        unit = min(W, H)
        self.unit = unit
        self.bd = SC.build("garden", "day", "clear", W, H, seed=42)
        self.ground = self.bd.ground
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        self.cx, self.cy = 0.5 * W, 0.3 * H
        self.angle = np.arctan2(yy - self.cy, xx - self.cx)
        self.fall = np.exp(-np.hypot(xx - self.cx, yy - self.cy) / (0.8 * unit))[..., None]
        wide = W > 1.1 * H
        big, small = int((0.2 if wide else 0.19) * unit), int((0.11 if wide else 0.1) * unit)
        name = name.strip() or CA.NAME
        top = 0.17 * H if wide else 0.14 * H
        self.lines = [(_logo_letters(name, big, LETTER_COLOURS), top, big),
                      (_logo_letters("& Friends", small, LETTER_COLOURS[2:] + LETTER_COLOURS[:2]),
                       top + 0.92 * big, small)]
        letters = sum(1 for line, _, _ in self.lines for a, _ in line if a is not None)
        self.drop_at = [i * 1.0 / max(letters, 1) for i in range(letters)]
        self.beats = beats
        h = int(char_h * (0.82 if wide else 0.7))
        cast = [CA.LULU, None, CA.KIKO, CA.BOBO]  # None: the hero
        xs = (0.17, 0.39, 0.62, 0.83) if wide else (0.14, 0.38, 0.62, 0.86)
        self.who = []
        for k, (d, x) in enumerate(zip(cast, xs)):
            rig = hero if d is None else CA.Character(int(h * d.size), d)
            label = name if d is None else d.name
            colour = tuple(c / 255 for c in (d or CA.PEEKO).body) if d is not CA.LULU else (1.0, 0.72, 0.82)
            tag = _pill(label, int(0.075 * unit), colour, WHITE if d is not CA.LULU else OUTLINE, outline=OUTLINE)
            self.who.append((rig, d, x * W, tag, (2.0, 1.0, 3.0, 4.0)[k]))  # the hero first
        self.scale = h / char_h if hero is not None else 1.0
        self.seconds = 7.6

    def frame(self, t: float, vig: np.ndarray) -> np.ndarray:
        W, H, unit = self.W, self.H, self.unit
        frame = self.bd.background(t, 30 * t)
        rays = (0.5 + 0.5 * np.cos(16 * self.angle + 0.5 * t)) ** 8
        frame += (1.0 - frame) * rays[..., None] * self.fall * 0.22
        ground = self.ground
        for k, (rig, d, x, tag, at) in enumerate(self.who):
            if rig is None or t < at:
                continue
            u = t - at
            pop = _phase(u, 0.0, 0.35)
            hover = d.hover if d is not None else 0.0
            if d is not None and d.hover:  # Kiko flies in
                pose = Pose(x + (1 - smooth(pop)) * 0.4 * W, lift=(hover + (1 - pop) * 0.6) * rig.h * self._k(rig))
            else:
                pose = Pose(x, lift=max(0.0, math.sin(math.pi * min(u / 0.5, 1.0))) * 0.3 * rig.h * self._k(rig)
                            if u < 0.5 else 0.0, sy=0.85 + 0.15 * back(pop), sx=1.1 - 0.1 * back(pop))
            pose.scale = self._k(rig)
            pose.yaw = 0.08
            action = "jump" if u < 0.5 and not hover else ("dance" if t > 5.0 else "wave")
            if hover:
                pose.lift += 0.05 * rig.h * self._k(rig) * math.sin(t * 4)
            rig.draw(frame, pose, 1 if x < W / 2 else -1, ground, CA.Act(action, u, 2.0, t, 0.0))
            tag.paste(frame, x, ground + 0.07 * unit, scale=back(pop), opacity=min(1.0, 2 * pop))
        i = 0
        for line, y, size in self.lines:
            widths = [a.size[0] * 0.82 if a is not None else size * 0.3 for a, _ in line]
            x = W / 2 - sum(widths) / 2
            for (letter, shadow), w in zip(line, widths):
                if letter is None:
                    x += w
                    continue
                at = self.drop_at[i]
                i += 1
                q = _phase(t, at, 0.4)
                if q > 0:
                    yy = y - (1 - back(q, 2.2)) * 0.6 * H
                    beat = max((0.0, *[b for b in self.beats if b <= t]))
                    pulse = 1.0 + 0.06 * math.exp(-(t - beat) * 8) if t > 1.0 else 1.0
                    wob = 6 * math.sin(t * 3 + i) * (1 - math.exp(-max(t - at, 0) * 2))
                    shadow.paste(frame, x + w / 2 + 0.006 * unit, yy + 0.008 * unit, scale=pulse, angle=wob,
                                 opacity=0.85 * min(1.0, 3 * q))
                    letter.paste(frame, x + w / 2, yy, scale=pulse, angle=wob, opacity=min(1.0, 3 * q))
                x += w
        if t > 7.0:
            k = _phase(t, 7.0, 0.3)
            frame += (1.0 - frame) * 0.25 * (1 - k)  # a flash on the last chord
        if t > self.seconds - IRIS:
            frame = frame * smooth((self.seconds - t) / IRIS)
        return frame * vig

    def _k(self, rig) -> float:
        return self.scale if isinstance(rig, Puppet) or getattr(rig, "design", None) is CA.PEEKO else 1.0


def render_cartoon(picture: Optional[str], name: str, story: Story, output_path: str, width: int = 1920,
                   height: int = 1080, fps: int = 30, faces: str = "right",
                   narrator: str = NARRATOR, character_voice: str = CHARACTER_VOICE, pitch: float = 1.25,
                   music: Optional[str] = None, intro: bool = True, outro: bool = True,
                   sound_effects: bool = True, progress: Optional[ProgressFn] = None, crf: int = 20,
                   preset: str = "fast", series: bool = False) -> CartoonResult:
    """Make the cartoon: ``name`` acts out ``story``, with its friends.

    The hero is cut out of ``picture``, or with no picture it is Peeko, drawn
    by the program with moving arms, legs and face; Lulu, Kiko and Bobo join
    in when a scene's ``cast`` names them. Each scene opens with an iris; the
    narrator reads its narration while the hero acts, counts what pops up on
    the chalkboard or asks the quiz, then the characters talk in speech
    bubbles, each in its own voice. ``series`` puts the show's opening (logo,
    friends, theme tune) after the story's teaser scene (``story.hook``).
    ``faces`` is where the hero looks in its picture.
    """
    report = progress or (lambda f, m: None)
    started = time.perf_counter()
    if not story.scenes:
        raise InputError("The story needs at least one scene.")
    if len(story.scenes) > 30:
        raise InputError("The story can have at most 30 scenes.")
    if faces not in ("right", "left", "front"):
        raise InputError("The character must face right, left or the front.")
    RenderConfig(width=width, height=height, fps=fps).validate()
    W, H = width, height
    unit = min(W, H)
    wide = W > 1.1 * H
    shots = [s.cleaned() for s in story.scenes]
    char_h = int(0.42 * unit)
    if picture is None:
        name = name.strip() or CA.NAME
        report(0.01, f"Drawing {name}")
        character = CA.Peeko(char_h)
    else:
        name = name.strip() or "Pip"
        report(0.01, "Cutting out the character")
        character = Puppet(cut_out(picture), char_h, faces)
    effects = Effects(char_h, W, H)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".cartoon-", dir=os.path.dirname(output_path))
    warnings: List[str] = []
    try:
        # ---- voices
        talking = bool(shutil.which("say"))
        if not talking:
            warnings.append("No speech synthesiser here: the story is told with subtitles only.")
        if music:
            decode_audio(music, rate=8000)
        installed = set(voices())
        said_cache: Dict[Tuple[str, str, float], np.ndarray] = {}

        def read(text: str, voice: str, p: float) -> Optional[np.ndarray]:
            if not text or not talking:
                return None
            key = (text, voice, p)
            if key not in said_cache:
                said_cache[key] = speak(text, voice, tmp, f"v{len(said_cache)}", p,
                                        NARRATOR_RATE if voice == narrator else 0)
            return said_cache[key]

        def voice_of(who: str) -> Tuple[str, float]:
            if who == "Narrator":
                return narrator, 1.0
            d = CA.design(who)
            if d is None or d.name == CA.NAME:
                return character_voice, pitch
            if d.voice in installed or not installed:
                return d.voice, d.pitch
            return character_voice, d.pitch  # the friend's own voice is missing: at least its own pitch

        scenes: List[_Scene] = []
        for i, shot in enumerate(shots):
            report(0.02 + 0.08 * i / len(shots), f"Voices and scenery for scene {i + 1}")
            narration = read(shot.narration, narrator, 1.0)
            n_s = len(narration) / SAMPLE_RATE if narration is not None else \
                (0.35 * len(shot.narration.split()) if shot.narration else 0.0)
            lines = ([Line("", shot.say)] if shot.say else []) + list(shot.lines)
            audio = []
            for ln in lines:
                v, p = voice_of(ln.who or CA.NAME)
                a = read(ln.say, v, p)
                audio.append((a, len(a) / SAMPLE_RATE if a is not None else max(1.2, 0.33 * len(ln.say.split()))))
            numbers = [read(NUMBERS[k] + "!", narrator, 1.0) for k in range(shot.count)]
            longest = max([len(a) / SAMPLE_RATE for a in numbers if a is not None] or [0.5])
            step = max(0.85, longest + 0.25)
            count_s = COUNT_LEAD + shot.count * step + 0.6 if shot.count else 0.0
            backdrop = SC.build(shot.setting, shot.time, shot.weather, W, H, seed=i)
            ground = backdrop.ground
            has_board = bool(shot.board or shot.item or shot.count or shot.quiz)
            cast = [(c.split()[-1], c.split()[0] if len(c.split()) > 1 else "") for c in shot.cast]
            scale, (lo, hi), slots = _layout(W, H, len(cast), has_board)
            hero_h = char_h * scale
            start = min(0.3 if (has_board and wide) or cast else 0.38, hi) * W
            timed, first = _timeline(shot, n_s, [(not ln.who, s, ln.do) for ln, (_, s) in zip(lines, audio)],
                                     count_s, bool(shot.quiz))
            beats = Performance.plan(timed, W, start, (lo, hi), world=not cast)
            said: List[_Said] = []
            for k, (ln, (a, s_)) in enumerate(zip(lines, audio)):
                b = beats[first + k]
                level = _level(a)
                if not ln.who:
                    b.voice = level
                speaker = "Narrator" if ln.who == "Narrator" else None
                bubble = None if speaker else Bubble(ln.say, unit, 0.36 * W if wide else 0.7 * W)
                caps = Captions(ln.say, s_, b.start, W, H) if speaker else None
                said.append(_Said(ln, b.start, s_, a, level, bubble, caps))
            show = Performance(beats, W, H, int(hero_h), None, backdrop.floating, (lo, hi), step)
            seconds = show.end + IRIS + 0.15
            caps = Captions(shot.narration, n_s, IRIS * 0.6, W, H) if shot.narration else None
            sounds = []
            for b in beats:
                sounds += [(b.start + t, s) for t, s in ACTION_SOUNDS.get(b.action, []) if t < b.seconds]
                sounds += _rhythm(b)
            effect_at = beats[1].start if len(beats) > 1 and beats[1].action not in INTERNAL else IRIS
            if shot.effect in EFFECT_SOUNDS:
                sounds.append((effect_at + 0.05, EFFECT_SOUNDS[shot.effect]))
            friends = [_Friend(nm, CA.Character(int(hero_h * CA.design(nm).size), CA.design(nm)), x * W, mood)
                       for (nm, mood), x in zip(cast, slots)]
            board, counting, room, board_off = None, [], None, 1e9
            if has_board:
                box = board_box(W, H, ground, int(hero_h * 1.15 if cast else hero_h))
                if wide and cast:
                    box = (box[0], box[1], box[2], min(box[3], 0.46 * H))
                times, quiz_at = [], 0.0
                counter = next((b for b in beats if b.action == "count"), None)
                if counter is not None:
                    times = [counter.start + COUNT_LEAD + k * step for k in range(shot.count)]
                    counting = list(zip(times, numbers))
                    sounds += [(t, "pop") for t in times] + [(times[-1] + 0.55, "tada")]
                if shot.quiz:
                    quiz_at = next(b.start for b in beats if b.action == "think" and b.seconds == QUIZ_COUNT)
                    sounds += [(quiz_at + k, "tick") for k in range(3)]
                    sounds += [(quiz_at + QUIZ_COUNT, "ding"), (quiz_at + QUIZ_COUNT + 0.1, "tada")]
                if wide and any(ln.who for ln in lines) and said:
                    board_off = said[0].start - 0.2
                board = Board(shot.board, shot.item, shot.count, box, IRIS * 0.8, times, shot.quiz, shot.answer,
                              quiz_at, board_off if board_off < 1e8 else None)
                sounds.append((IRIS * 0.8, "swish"))
                room = (0.02 * W, box[0] - 0.01 * W, 0.03 * H) if wide else (0.02 * W, 0.98 * W, box[3] + 0.01 * H)
            scenes.append(_Scene(shot, backdrop, show, seconds, narration, IRIS * 0.6, caps, sounds, effect_at,
                                 scale, board, counting, said, friends, room, board_off))

        grid = np.mgrid[0:H, 0:W].astype(F32)
        vig = _vignette(W, H)
        auto_prop = {"read": "book", "brush": "toothbrush"}

        def scene_frame(sc: _Scene, t: float) -> np.ndarray:
            pose, heading, scroll, beat, u = sc.show.at(min(t, sc.show.end - 1e-3))
            pose.scale = sc.scale
            frame = sc.backdrop.background(t, scroll)
            ground = sc.backdrop.ground
            if sc.board is not None:
                sc.board.draw(frame, t)
            now = next((s for s in sc.said if s.start <= t < s.start + s.seconds + GAP * 0.5), None)
            friend_bubble = None  # drawn last, so nothing covers the words
            for fr in sc.friends:
                speaking = now is not None and now.line.who == fr.name
                if speaking:
                    action, uu, dur = now.line.do or "talk", t - now.start, now.seconds + GAP
                    voice = now.level
                else:
                    action, uu, dur, voice = MOOD_ACTION.get(fr.mood, "listen"), t, 1e6, None
                target = pose.x if now is None or not now.line.who or now.line.who == "Narrator" else \
                    next((f.x for f in sc.friends if f.name == now.line.who), pose.x)
                facing = (-1 if target < fr.x else 1) if not speaking else (-1 if pose.x < fr.x else 1)
                beat_f = Beat(action, 0.0, dur, fr.x, facing, 0.0, voice=voice)
                fp = Performance([beat_f], W, H, fr.rig.h).at(min(uu, dur - 1e-3))[0]
                fp.lift += fr.rig.design.hover * fr.rig.h * (1 + 0.15 * math.sin(t * 3 + fr.x))
                level = voice(uu) if voice else 0.0
                fr.rig.draw(frame, fp, facing, ground, CA.Act(action, uu, dur, t + fr.x * 0.01, level))
                if speaking and now.bubble is not None:
                    friend_bubble = fr.rig.head(fp, ground)
            level = beat.voice(u) if beat.voice else 0.0
            act = CA.Act(beat.action, u, beat.seconds, t, level, sc.shot.holds or auto_prop.get(beat.action, ""))
            character.draw(frame, pose, heading, ground, act)
            head = character.head(pose, ground)
            body = character.h * pose.sy * pose.scale
            feel = ACTION_EFFECT.get(beat.action)
            if feel:
                effects.draw(frame, feel, u, head, body, fade=min(1.0, 4 * u, 4 * (beat.seconds - u)))
            if beat.action in ("jump",) and u > 0.78 * beat.seconds:
                effects.draw(frame, "dust", u - 0.78 * beat.seconds, (pose.x, 0), ground)
            if sc.shot.effect not in ("none", "") and t > sc.effect_at:
                effects.draw(frame, sc.shot.effect, t - sc.effect_at, head, body,
                             fade=min(1.0, 3 * (t - sc.effect_at), 3 * (sc.seconds - t)))
            sc.backdrop.foreground(frame, t, scroll)
            for s in sc.said:
                if s.line.who == "" and s.bubble is not None:
                    _speech(frame, s, t, head, sc, W)
                elif s.captions is not None:
                    s.captions.draw(frame, t)
            if friend_bubble is not None:
                _speech(frame, now, t, friend_bubble, sc, W)
            if sc.captions is not None:
                sc.captions.draw(frame, t)
            centre = (pose.x, ground - 0.5 * body - pose.lift)
            if t < IRIS:
                frame = _iris(frame, centre, smooth(t / IRIS), grid)
            elif t > sc.seconds - IRIS:
                frame = _iris(frame, centre, smooth((sc.seconds - t) / IRIS), grid)
            return frame * vig

        def _speech(frame, s: _Said, t: float, head, sc: _Scene, W: int) -> None:
            u2 = t - s.start
            if -0.1 < u2 < s.seconds + GAP + 0.3:
                room = sc.room if t < sc.board_off else None
                s.bubble.draw(frame, head, min(1.0, u2 / max(s.seconds, 0.1)), _phase(u2, -0.1, 0.25),
                              fade=1.0 - _phase(u2, s.seconds + GAP, 0.3), room=room)

        # ---- title, the end
        hook = story.hook and len(scenes) > 1
        opening = scenes[1] if hook else scenes[0]
        title = story.title.strip() or f"{name}'s Adventure"
        title_s = text_sprite(title, TITLE_FONTS, int(0.11 * unit), WHITE, stroke=max(3, unit // 90),
                              stroke_fill=OUTLINE, max_width=int(0.9 * W))
        kicker = f"Episode {story.episode}" if story.episode else f"starring {name}"
        star_s = text_sprite(kicker, TITLE_FONTS, int(0.06 * unit), (1.0, 0.88, 0.3),
                             stroke=max(3, unit // 110), stroke_fill=OUTLINE, max_width=int(0.8 * W))
        title_voice = read(f"Episode {story.episode}. {title}!" if story.episode else f"{title}!", narrator, 1.0) \
            if series else None
        title_seconds = max(4.2, 1.2 + (len(title_voice) / SAMPLE_RATE if title_voice is not None else 0) + 1.2)
        end_s = text_sprite("The End", TITLE_FONTS, int(0.15 * unit), WHITE, stroke=max(3, unit // 80),
                            stroke_fill=OUTLINE)
        lesson = story.lesson.strip()
        lesson_s = _banner(lesson, unit, W) if lesson else None
        lesson_voice = read(lesson, narrator, 1.0) if lesson else None
        lesson_len = (len(lesson_voice) / SAMPLE_RATE if lesson_voice is not None else
                      0.33 * len(lesson.split())) if lesson else 0.0
        teaser = f"Next time: {story.next}" if story.next else ""
        next_s = _banner(teaser, unit, W, icon="arrow", size_k=0.04) if teaser else None
        next_voice = read(teaser, narrator, 1.0) if teaser else None
        next_len = (len(next_voice) / SAMPLE_RATE if next_voice is not None else 0.33 * len(teaser.split())) \
            if teaser else 0.0
        collection = story.season >= 2 and 1 <= story.episode <= story.season
        flower_px = int(min(0.055 * unit, 0.8 * W / max(story.season, 1)))
        flowers = (_blossom(flower_px, True), _blossom(flower_px, False)) if collection else None
        last = scenes[-1]
        lesson_at = 1.0
        flower_at = lesson_at + lesson_len + 0.3 if lesson else 1.0
        next_at = flower_at + (0.9 if collection else 0.0)
        end_seconds = max(3.4, (next_at + next_len + 1.4) if teaser else (flower_at + 1.6))
        end_y = 0.13 * H if lesson else 0.24 * H

        def title_frame(t: float, seconds: float) -> np.ndarray:
            frame = opening.backdrop.background(t, 0.0)
            ground = opening.backdrop.ground
            fall = 0.7
            if t < fall:  # dropping in from the sky
                q = t / fall
                pose = Pose(0.5 * W, lift=(1 - q * q) * (H + character.h), sy=1.08, sx=0.95)
                act = CA.Act("happy", t, 1.0, t)
            elif t < fall + 0.3:
                k = math.sin(math.pi * (t - fall) / 0.3)
                pose = Pose(0.5 * W, sx=1 + 0.18 * k, sy=1 - 0.22 * k)
                act = CA.Act("surprised", 0.3, 1.0, t)
            else:
                hop = abs(math.sin(math.pi * (t - fall - 0.3) * 1.6)) if t < fall + 1.6 else 0.0
                pose = Pose(0.5 * W, lift=hop * 0.08 * character.h, angle=4 * math.sin(t * 5))
                act = CA.Act("wave", t - fall - 0.3, seconds, t)
            frame *= 0.85
            character.draw(frame, pose, 1, ground, act)
            head = character.head(pose, ground)
            if t > fall:
                effects.draw(frame, "stars", t - fall, head, character.h)
            b = _phase(t, 0.9, 0.35)
            if b > 0:
                title_s.paste(frame, W / 2, 0.2 * H, scale=0.3 + 0.7 * back(b), opacity=min(1.0, 2 * b),
                              angle=2.5 * math.sin(t * 2.2))
            star_s.paste(frame, W / 2, 0.2 * H + 0.11 * unit, opacity=_phase(t, 1.5, 0.4))
            if t < IRIS and series:
                frame = _iris(frame, (0.5 * W, ground - 0.5 * character.h), smooth(t / IRIS), grid)
            if t > seconds - IRIS:
                frame = _iris(frame, (0.5 * W, ground - 0.5 * character.h), smooth((seconds - t) / IRIS), grid)
            return frame * vig

        def end_frame(t: float, seconds: float) -> np.ndarray:
            frame = last.backdrop.background(last.seconds + t, 0.0)
            ground = last.backdrop.ground
            hop = abs(math.sin(math.pi * t * 1.8))
            pose = Pose(0.5 * W, lift=hop * 0.06 * character.h + (0.08 * character.h if last.backdrop.floating else 0),
                        sy=1 + 0.03 * hop, angle=5 * math.sin(t * 4))
            bloom = smooth((t - 0.6) / 0.8) if lesson else 0.0
            character.draw(frame, pose, 1, ground, CA.Act("happy" if t < 1.4 else "wave", t, seconds, t,
                                                          bloom=bloom))
            effects.draw(frame, "confetti", t, (0, 0), 0)
            b = _phase(t, 0.3, 0.35)
            end_s.paste(frame, W / 2, end_y, scale=0.3 + 0.7 * back(b), opacity=min(1.0, 2 * b),
                        angle=3 * math.sin(t * 2))
            y = end_y + 0.1 * unit
            if lesson_s is not None:
                q = _phase(t, 0.8, 0.4)
                lesson_s.paste(frame, W / 2, y + lesson_s.size[1] / 2, scale=0.5 + 0.5 * back(q),
                               opacity=min(1.0, 2 * q))
                y += lesson_s.size[1] + 0.03 * unit
            if flowers is not None:
                gap = 1.15 * flower_px
                x0 = W / 2 - (story.season - 1) * gap / 2
                for k in range(story.season):
                    on = k < story.episode
                    newest = k == story.episode - 1
                    q = _phase(t, flower_at, 0.4) if newest else _phase(t, flower_at - 0.6 + 0.03 * k, 0.3)
                    sprite = flowers[0] if on else flowers[1]
                    sprite.paste(frame, x0 + k * gap, y + flower_px / 2, scale=(back(q) * (1.25 if newest else 1.0))
                                 if q > 0 else 0.0, opacity=min(1.0, 3 * q), angle=20 * math.sin(t * 3) if newest else 0)
            if next_s is not None:
                q = _phase(t, next_at, 0.4)
                next_s.paste(frame, W / 2, H - 0.06 * unit - next_s.size[1] / 2, scale=0.5 + 0.5 * back(q),
                             opacity=min(1.0, 2 * q))
            if t < IRIS:
                frame = _iris(frame, (0.5 * W, ground - 0.5 * character.h), smooth(t / IRIS), grid)
            return frame * vig

        # ---- the parts in order: (seconds, frame function, label)
        parts: List[Tuple[float, Callable[[float], np.ndarray], str]] = []
        events: List[Tuple[float, str]] = []
        voice_track: List[Tuple[float, np.ndarray]] = []
        jingle_at: List[Tuple[float, float]] = []  # (start, seconds) of the theme tune
        at = 0.0

        def add_scene(i: int, sc: _Scene) -> None:
            nonlocal at
            parts.append((sc.seconds, lambda t, sc=sc: scene_frame(sc, t), f"Scene {i + 1}"))
            events.extend((at + t, s) for t, s in sc.sounds)
            if sc.narration is not None:
                voice_track.append((at + sc.narration_at, sc.narration))
            for t, number in sc.counting:
                if number is not None:
                    voice_track.append((at + t, number))
            for s in sc.said:
                if s.audio is not None:
                    voice_track.append((at + s.start, s.audio))
            at += sc.seconds

        if hook:
            add_scene(0, scenes[0])
            events.append((at - 0.3, "whoosh"))
        if series:
            report(0.1, "The show's opening")
            from . import music as MU
            theme, marks = MU.jingle()
            opening_clip = SeriesIntro(W, H, char_h, character, name, marks)
            parts.append((opening_clip.seconds, lambda t: opening_clip.frame(t, vig), "Series intro"))
            jingle_at.append((at, opening_clip.seconds))
            voice_track.append((at, theme * 0.75))
            events += [(at + 1.0 + k, "pop") for k in range(4)]
            at += opening_clip.seconds
        if intro or series:
            parts.append((title_seconds, lambda t: title_frame(t, title_seconds), "Title"))
            events += [(at + 0.05, "whoosh"), (at + 0.7, "boing"), (at + 0.95, "thud"), (at + 1.0, "sparkle")]
            if title_voice is not None:
                voice_track.append((at + 1.2, title_voice))
            at += title_seconds
        for i, sc in enumerate(scenes):
            if hook and i == 0:
                continue
            add_scene(i, sc)
        parts.append((end_seconds, lambda t: end_frame(t, end_seconds), "The End"))
        events += [(at + 0.3, "thud"), (at + 0.35, "sparkle")]
        if lesson:
            events.append((at + 0.75, "ding"))
            if lesson_voice is not None:
                voice_track.append((at + lesson_at, lesson_voice))
        if collection:
            events += [(at + flower_at, "sparkle"), (at + flower_at + 0.05, "pop")]
        if teaser:
            events.append((at + next_at, "swish"))
            if next_voice is not None:
                voice_track.append((at + next_at + 0.2, next_voice))
        at += end_seconds
        poster = _u8(np.clip(end_frame(min(next_at - 0.1 if teaser else 2.2, end_seconds - 0.5), end_seconds),
                             0.0, 1.0))
        if outro:
            question = story.question.strip() or QUESTION.format(name=name)
            clip = outro_clip(poster, question, [], fps)
            parts.append((clip.seconds, None, "Like, subscribe, comment"))
            events += [(at + t, s) for t, s in clip.sounds]
            at += clip.seconds
        else:
            clip = None

        counts = [int(round(seconds * fps)) for seconds, _, _ in parts]
        n_total = sum(counts)
        total = n_total / fps
        tmp_video = os.path.join(tmp, "video.mp4")
        done = 0
        with FFmpegWriter(tmp_video, W, H, fps, crf, preset, total_frames=n_total) as writer:
            for (seconds, fn, label), count in zip(parts, counts):
                report(0.1 + 0.85 * done / n_total, label)
                if fn is None:
                    for k, frame in zip(range(count), clip.frames()):
                        writer.write(frame)
                else:
                    for k in range(count):
                        writer.write(_u8(np.clip(fn(k / fps), 0.0, 1.0)))
                        if k % 15 == 0:
                            report(0.1 + 0.85 * (done + k) / n_total, label)
                done += count

        report(0.96, "Adding the sound")
        n = int(math.ceil(total * SAMPLE_RATE)) + 1
        speech = np.zeros(n, F32)
        for t0, samples in voice_track:
            s0 = int(t0 * SAMPLE_RATE)
            s1 = min(n, s0 + len(samples))
            if s1 > s0:
                speech[s0:s1] += samples[:s1 - s0] * 0.95
        bed = np.zeros(n, F32)
        if music:
            tune = decode_audio(music)
            if len(tune):
                bed = np.tile(tune, int(math.ceil(n / len(tune))))[:n] * 0.5
        elif story.music in ("happy", "calm"):
            from . import music as MU
            bed = MU.background(total, story.music)[:n]
            if len(bed) < n:
                bed = np.pad(bed, (0, n - len(bed)))
        for start_s, length in jingle_at:  # the theme plays alone
            a, b = int(start_s * SAMPLE_RATE), int((start_s + length) * SAMPLE_RATE)
            ramp = int(0.4 * SAMPLE_RATE)
            bed[max(0, a - ramp):a] *= np.linspace(1, 0, len(bed[max(0, a - ramp):a]), dtype=F32)
            bed[a:b] = 0.0
            bed[b:b + ramp] *= np.linspace(0, 1, len(bed[b:b + ramp]), dtype=F32)
        if bed.any():
            from . import music as MU
            bed = MU.duck(bed, speech) * (0.3 if voice_track else 0.8)
        mix = speech + bed
        if sound_effects:
            mix += sound_track(events, total)[:n] * (0.6 if voice_track else 0.9)
        peak = float(np.abs(mix).max()) if mix.size else 0.0
        if peak > 0.98:
            mix *= 0.98 / peak
        if peak > 0.0:
            wav = os.path.join(tmp, "sound.wav")
            write_wav(wav, mix)
            mixed = os.path.join(tmp, "with_sound.mp4")
            _add_soundtrack(tmp_video, mixed, total, wav, total)
            tmp_video = mixed
        os.replace(tmp_video, output_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    image_path = os.path.splitext(output_path)[0] + ".png"
    Image.fromarray(poster).save(image_path)
    report(1.0, "Done")
    return CartoonResult(output_path, image_path, total, len(shots), time.perf_counter() - started, warnings)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    from .episodes import EPISODES, SEASON_GROUP, episode
    p = argparse.ArgumentParser(prog="pencil-cartoon",
                                description="Animate Peeko & Friends, or a character from one picture, through a "
                                            "story.")
    p.add_argument("script", nargs="?", help='JSON: {"character": "pip.png" (none for Peeko), "name", "faces", '
                                             '"title", "question", "lesson", "next", "hook", "music", "scenes": '
                                             '[{"setting", "time", "weather", "narration", "actions", "cast", '
                                             '"lines", "effect", "holds", "board", "item", "count", "quiz", '
                                             '"answer"}]}')
    p.add_argument("-o", "--output", required=True,
                   help="output MP4 (a PNG is saved next to it); with --season, a folder")
    p.add_argument("--character", help="the hero's picture (with --sample, --episode or --season; default Peeko)")
    p.add_argument("--name", default="", help="the hero's name (with --sample, --episode or --season)")
    p.add_argument("--sample", action="store_true", help="act out the built-in sample story")
    p.add_argument("--episode", choices=[e["id"] for e in EPISODES], help="act out a ready-made episode")
    p.add_argument("--season", action="store_true", help="make every episode of season 1, one after another")
    p.add_argument("--size", default="1920x1080", help="resolution, e.g. 1080x1920 for vertical")
    p.add_argument("--faces", default=None, choices=("right", "left", "front"))
    p.add_argument("--narrator", default=NARRATOR)
    p.add_argument("--voice", default=CHARACTER_VOICE, help="the hero's voice")
    p.add_argument("--pitch", type=float, default=1.25, help="the hero's voice pitch")
    p.add_argument("--music", help="your own music instead of the program's")
    for flag in ("intro", "outro", "sound-effects", "series"):
        p.add_argument(f"--no-{flag}", action="store_true", help=f"leave out the {flag}")
    args = p.parse_args(argv)
    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
        options = dict(narrator=args.narrator, character_voice=args.voice, pitch=args.pitch, music=args.music,
                       intro=not args.no_intro, outro=not args.no_outro,
                       sound_effects=not args.no_sound_effects, series=not args.no_series)
        if args.season:
            name = args.name or ("Pip" if args.character else CA.NAME)
            os.makedirs(args.output, exist_ok=True)
            for e in (e for e in EPISODES if e["group"] == SEASON_GROUP):
                out = os.path.join(args.output, f"{e['id']}.mp4")
                if os.path.exists(out):
                    print(f"Already made: {out}")
                    continue
                result = render_cartoon(args.character, name, episode(e["id"], name), out, w, h,
                                        faces=args.faces or ("right" if args.character else "front"), **options)
                print(f"Saved {result.output_path} ({result.duration:.0f} s)", flush=True)
            return 0
        if args.sample or args.episode:
            name = args.name or ("Pip" if args.character else CA.NAME)
            story = episode(args.episode, name) if args.episode else sample_story(name)
            picture, faces = args.character, "right" if args.character else "front"
        else:
            if not args.script:
                raise InputError("Give a script, or --sample, --episode or --season.")
            picture, name, faces, story = load_script(args.script)
        result = render_cartoon(picture, name, story, args.output, w, h, faces=args.faces or faces, **options)
    except (PencilError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {result.output_path} ({result.duration:.0f} s) and {result.image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
