"""Story videos: a narrated sequence of scenes, each drawn in pencil.

A *story script* (JSON) lists scenes. Each scene has its narration text (shown
as subtitles), an image (a local file, or English keywords used to find a
free-licensed photo on Wikimedia Commons) and optionally its own audio
recording. The narration can also be one recording for the whole story; the
scenes are then matched to it by text length and natural pauses.

Each photo is turned into a graphite drawing and drawn on blank paper by the
pencil animation, then held while the narration continues. Photo credits are
shown at the end and written next to the video.

    python -m pencil_timelapse.story story.json -o story.mp4
    python -m pencil_timelapse.story --template story.txt -o story.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import RenderConfig
from .encoder import FFmpegWriter, find_ffmpeg
from .errors import InputError, PencilError, RenderError
from .imaging import load_image
from .keywords import suggest_queries
from .media import (SAMPLE_RATE, ImageCredit, decode_audio, download_candidate, find_image,
                    find_pauses, speak, write_wav)
from .pipeline import plan_animation
from .renderer import FrameRenderer
from .sketch import PAPER, photo_to_pencil

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]

TITLE_SECONDS = 3.5
CREDITS_SECONDS = 5.0
CROSSFADE = 0.5
SPEECH_LEAD = 0.5  # pause before the narration of a scene starts
SPEECH_TAIL = 0.9  # pause after it, before the next scene
MIN_SCENE = 5.0
READING_RATE = 13.0  # characters per second when a scene has no recording
INK = (52, 50, 46)

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


# --------------------------------------------------------------------------- script

@dataclass
class Scene:
    text: str
    image_query: str = ""
    image: Optional[str] = None
    audio: Optional[str] = None
    image_choice: Optional[dict] = None  # a search result picked by the user


@dataclass
class StoryScript:
    title: str
    scenes: List[Scene]
    narration: Optional[str] = None  # one recording for the whole story
    music: Optional[str] = None
    music_volume: float = 0.12
    width: int = 1920
    height: int = 1080
    fps: int = 30
    base_dir: str = "."

    def path(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value if os.path.isabs(value) else os.path.join(self.base_dir, value)

    @classmethod
    def load(cls, path: str) -> "StoryScript":
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise InputError(f"Story script not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise InputError(f"Story script is not valid JSON ({exc}).") from None
        return cls.from_dict(data, os.path.dirname(os.path.abspath(path)))

    @classmethod
    def from_dict(cls, data: dict, base_dir: str = ".") -> "StoryScript":
        raw = data.get("scenes")
        if not isinstance(raw, list) or not raw:
            raise InputError("Story script: 'scenes' must be a non-empty list.")
        scenes = []
        for i, item in enumerate(raw, 1):
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                raise InputError(f"Story script: scene {i} needs a 'text'.")
            choice = item.get("image_choice")
            if choice is not None and not (isinstance(choice, dict) and
                                           str(choice.get("url", "")).startswith("https://")):
                raise InputError(f"Story script: scene {i} has an invalid 'image_choice'.")
            scene = Scene(str(item["text"]).strip(), str(item.get("image_query", "")).strip(),
                          item.get("image"), item.get("audio"), choice)
            if not scene.image and not scene.image_query and not scene.image_choice:
                raise InputError(f"Story script: scene {i} needs an 'image' file or an "
                                 "'image_query' (English keywords).")
            scenes.append(scene)
        width, height = data.get("width", 1920), data.get("height", 1080)
        script = cls(str(data.get("title", "")).strip(), scenes, data.get("narration"),
                     data.get("music"), float(data.get("music_volume", 0.12)), int(width),
                     int(height), int(data.get("fps", 30)), base_dir)
        script.check_files()
        return script

    def check_files(self) -> None:
        for label, value in [("narration", self.narration), ("music", self.music)] + [
                (f"scene {i} {kind}", getattr(sc, kind)) for i, sc in enumerate(self.scenes, 1)
                for kind in ("image", "audio")]:
            path = self.path(value)
            if path and not os.path.isfile(path):
                raise InputError(f"Story script: the {label} file was not found: {path}")


# Target scene length in characters of narration: Vietnamese is read at about
# 15 characters per second, so these are about 8, 12 and 18 seconds.
SCENE_LENGTHS = {"short": 120, "medium": 180, "long": 270}


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?…][\"”’)])\s+|(?<=[.!?…])\s+|\n+", text)
    return [s.strip() for s in parts if s.strip()]


def split_story(text: str, target: int = SCENE_LENGTHS["medium"]) -> Tuple[str, List[str]]:
    """Split a story into a title and scene texts.

    Paragraphs (separated by blank lines, or by line breaks when there are no
    blank lines) stay together when they fit. Longer paragraphs - or a story
    pasted as one block - are cut at sentence ends into scenes of about
    ``target`` characters, and very short pieces join the scene before them.
    A short first line without final punctuation is taken as the title.
    """
    text = text.replace("\r\n", "\n").strip()
    # Stage directions such as "[Ending – 4:15–4:50]" are not narration.
    text = re.sub(r"\[[^\]\n]{0,80}\]", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) == 1 and "\n" in blocks[0]:
        blocks = [line.strip() for line in blocks[0].split("\n") if line.strip()]
    title = ""
    if len(blocks) > 1 and len(blocks[0]) <= 80 and not re.search(r"[.!?…:;,]$", blocks[0]):
        title, blocks = blocks[0], blocks[1:]
    pieces: List[str] = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block)
        if len(block) <= 1.5 * target:
            pieces.append(block)
            continue
        current = ""
        for sentence in _sentences(block):
            if current and len(current) + 1 + len(sentence) > 1.15 * target \
                    and len(current) >= 0.5 * target:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)
    # A scene must last long enough for the pencil to draw its picture: short
    # pieces join the scene before them (or the next one, at the start).
    scenes: List[str] = []
    for piece in pieces:
        if scenes and len(piece) < 0.5 * target and len(scenes[-1]) + len(piece) <= 2.0 * target:
            scenes[-1] = f"{scenes[-1]} {piece}"
        else:
            scenes.append(piece)
    if len(scenes) > 1 and len(scenes[0]) < 0.5 * target:
        scenes[1] = f"{scenes[0]} {scenes[1]}"
        scenes = scenes[1:]
    return title, scenes


def template_from_text(text: str, title: str = "", length: str = "medium") -> dict:
    """A script skeleton: scenes split automatically, each with suggested image
    keywords (falling back to the story's main subject)."""
    found_title, scenes = split_story(text, SCENE_LENGTHS.get(length, SCENE_LENGTHS["medium"]))
    title = title or found_title
    queries = suggest_queries(scenes, title)
    return {"title": title, "narration": None, "music": None,
            "scenes": [{"text": s, "image_query": q, "audio": None}
                       for s, q in zip(scenes, queries)]}


# --------------------------------------------------------------------------- text rendering

def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if font.getlength(trial) <= max_width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
    return lines


def subtitle_chunks(text: str, font: ImageFont.FreeTypeFont, max_width: float,
                    max_lines: int = 2) -> List[str]:
    """Split narration into on-screen captions of at most ``max_lines`` lines,
    breaking at sentence ends where possible."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…;:])\s+", text) if s.strip()]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        trial = f"{current} {sentence}".strip()
        if len(wrap_text(trial, font, max_width)) <= max_lines:
            current = trial
            continue
        if current:
            chunks.append(current)
        lines = wrap_text(sentence, font, max_width)  # a long sentence: split it by lines
        while len(lines) > max_lines:
            chunks.append(" ".join(lines[:max_lines]))
            lines = lines[max_lines:]
        current = " ".join(lines)
    if current:
        chunks.append(current)
    return chunks


class Captions:
    """Pre-rendered subtitle strips blended onto the bottom of frames."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.font = _font(max(18, round(min(width, height) * 0.042)))
        self.max_width = 0.84 * width
        self._cache: Dict[str, Tuple[np.ndarray, int, int]] = {}

    def chunks(self, text: str) -> List[str]:
        return subtitle_chunks(text, self.font, self.max_width)

    def _strip(self, chunk: str) -> Tuple[np.ndarray, int, int]:
        if chunk in self._cache:
            return self._cache[chunk]
        lines = wrap_text(chunk, self.font, self.max_width)
        size = self.font.size
        line_h = int(size * 1.3)
        pad_x, pad_y = int(size * 0.8), int(size * 0.45)
        text_w = int(max(self.font.getlength(line) for line in lines))
        box_w, box_h = text_w + 2 * pad_x, line_h * len(lines) + 2 * pad_y
        img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, box_w - 1, box_h - 1), radius=int(size * 0.4),
                               fill=(20, 20, 20, 150))
        for i, line in enumerate(lines):
            x = (box_w - self.font.getlength(line)) / 2
            draw.text((x, pad_y + i * line_h), line, font=self.font, fill=(255, 255, 255, 255))
        rgba = np.asarray(img).astype(np.float32) / 255.0
        x0 = (self.width - box_w) // 2
        y0 = self.height - box_h - int(0.045 * self.height)
        self._cache[chunk] = (rgba, x0, y0)
        return self._cache[chunk]

    def apply(self, frame: np.ndarray, chunk: str, opacity: float) -> np.ndarray:
        if not chunk or opacity <= 0:
            return frame
        rgba, x0, y0 = self._strip(chunk)
        h, w = rgba.shape[:2]
        out = frame.copy()
        roi = out[y0:y0 + h, x0:x0 + w].astype(np.float32)
        a = rgba[..., 3:4] * opacity
        out[y0:y0 + h, x0:x0 + w] = np.clip(roi * (1 - a) + rgba[..., :3] * 255.0 * a + 0.5,
                                            0, 255).astype(np.uint8)
        return out


def paper_frame(width: int, height: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), 0.7)
    paper = PAPER[None, None, :] * (1.0 + 0.008 * grain[..., None])
    return np.clip(paper * 255.0 + 0.5, 0, 255).astype(np.uint8)


def text_card(width: int, height: int, title: str, lines: List[str], title_scale: float = 0.075
              ) -> np.ndarray:
    """Paper with a heading and optional smaller lines, centred."""
    img = Image.fromarray(paper_frame(width, height))
    draw = ImageDraw.Draw(img)
    big = _font(max(20, round(min(width, height) * title_scale)))
    small = _font(max(14, round(min(width, height) * 0.024)))
    title_lines = wrap_text(title, big, 0.8 * width) if title else []
    body = [l for line in lines for l in wrap_text(line, small, 0.86 * width)]
    heights = [int(big.size * 1.25)] * len(title_lines) + [int(small.size * 1.45)] * len(body)
    gap = int(big.size * 0.6) if title_lines and body else 0
    y = (height - sum(heights) - gap) // 2
    for line in title_lines:
        draw.text(((width - big.getlength(line)) / 2, y), line, font=big, fill=INK)
        y += int(big.size * 1.25)
    y += gap
    for line in body:
        draw.text(((width - small.getlength(line)) / 2, y), line, font=small, fill=INK)
        y += int(small.size * 1.45)
    return np.asarray(img).copy()


# --------------------------------------------------------------------------- timing

@dataclass
class SceneTiming:
    start: float  # seconds in the final video
    duration: float
    speech: Tuple[float, float]  # when this scene's narration plays (video time)


@dataclass
class StoryResult:
    output_path: str
    duration: float
    scenes: int
    credits: List[ImageCredit]
    credits_path: Optional[str]
    elapsed: float
    warnings: List[str] = field(default_factory=list)


def plan_timing(script: StoryScript, clips: List[Optional[np.ndarray]],
                narration: Optional[np.ndarray], offset: float
                ) -> Tuple[List[SceneTiming], np.ndarray]:
    """Scene start times/durations and the narration track (mono samples).

    * Per-scene recordings: each scene lasts as long as its clip plus pauses.
    * One recording: scenes split it in proportion to their text length, with
      each boundary moved to the nearest natural pause.
    * No recording: durations follow a comfortable reading pace.
    """
    rate = SAMPLE_RATE
    timings: List[SceneTiming] = []
    if narration is not None:
        total = len(narration) / rate
        lengths = np.array([max(len(s.text), 1) for s in script.scenes], np.float64)
        bounds = list(np.concatenate([[0.0], np.cumsum(lengths) / lengths.sum() * total]))
        pauses = [0.5 * (a + b) for a, b in find_pauses(narration, rate)]
        window = 0.25 * total / len(script.scenes)
        for i in range(1, len(bounds) - 1):  # snap inner boundaries to pauses, in order
            near = [p for p in pauses if abs(p - bounds[i]) <= window and p > bounds[i - 1] + 1.0]
            if near:
                bounds[i] = min(near, key=lambda p: abs(p - bounds[i]))
        lead = SPEECH_LEAD
        for i in range(len(script.scenes)):
            a, b = bounds[i], bounds[i + 1]
            start = offset + lead + a if i else offset
            end = offset + lead + (b if i < len(script.scenes) - 1 else b + SPEECH_TAIL)
            timings.append(SceneTiming(start, end - start, (offset + lead + a, offset + lead + b)))
        track = np.zeros(int((timings[-1].start + timings[-1].duration + 1.0) * rate), np.float32)
        s0 = int((offset + lead) * rate)
        track[s0:s0 + len(narration)] += narration[:len(track) - s0]
        return timings, track
    t = offset
    for scene, clip in zip(script.scenes, clips):
        speak_len = len(clip) / rate if clip is not None else len(scene.text) / READING_RATE
        duration = max(MIN_SCENE, SPEECH_LEAD + speak_len + SPEECH_TAIL)
        timings.append(SceneTiming(t, duration, (t + SPEECH_LEAD, t + SPEECH_LEAD + speak_len)))
        t += duration
    track = np.zeros(int((t + 1.0) * rate), np.float32)
    for timing, clip in zip(timings, clips):
        if clip is not None:
            s0 = int(timing.speech[0] * rate)
            track[s0:s0 + len(clip)] += clip[:len(track) - s0]
    return timings, track


def caption_at(chunks: List[str], speech: Tuple[float, float], t: float) -> Tuple[str, float]:
    """Caption shown at time ``t`` and its opacity (chunks share the speech time
    in proportion to their length, with short fades)."""
    if not chunks:
        return "", 0.0
    s0, s1 = speech
    s1 = max(s1, s0 + 0.5)
    weights = np.array([len(c) for c in chunks], np.float64)
    edges = s0 + np.concatenate([[0.0], np.cumsum(weights) / weights.sum()]) * (s1 - s0)
    if t < s0 - 0.15 or t > s1 + 0.6:
        return "", 0.0
    i = int(np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(chunks) - 1))
    fade_in = np.clip((t - (edges[i] - 0.15)) / 0.15, 0.0, 1.0) if i == 0 else 1.0
    fade_out = np.clip((s1 + 0.6 - t) / 0.3, 0.0, 1.0) if i == len(chunks) - 1 else 1.0
    return chunks[i], float(min(fade_in, fade_out))


# --------------------------------------------------------------------------- rendering

def _blend(a: np.ndarray, b: np.ndarray, x: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0 - x, b, x, 0.0)


def _scene_canvas(drawing: np.ndarray, width: int, height: int) -> np.ndarray:
    """The finished scene: the drawing on paper, leaving room for subtitles."""
    canvas = paper_frame(width, height).astype(np.float32) / 255.0
    bh, bw = drawing.shape[:2]
    x0, y0 = (width - bw) // 2, int(0.04 * height)
    canvas[y0:y0 + bh, x0:x0 + bw] = drawing
    return canvas


def render_story(script: StoryScript, output_path: str, progress: Optional[ProgressFn] = None,
                 preview_voice: Optional[str] = None, crf: int = 20, preset: str = "fast",
                 cache_dir: Optional[str] = None) -> StoryResult:
    """Render the whole story to ``output_path`` (MP4 with sound)."""
    report = progress or (lambda f, m: None)
    started = time.perf_counter()
    W, H, fps = script.width, script.height, script.fps
    RenderConfig(width=W, height=H, fps=fps).validate()
    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = cache_dir or os.path.join(script.base_dir, "story_images")
    tmp = tempfile.mkdtemp(prefix=".story-", dir=out_dir)
    warnings: List[str] = []
    try:
        # ---- images -------------------------------------------------------
        box = (int(0.86 * W) // 2 * 2, int(0.78 * H) // 2 * 2)
        credits: List[ImageCredit] = []
        drawings: List[np.ndarray] = []
        used: List[str] = []
        for i, scene in enumerate(script.scenes):
            report(0.02 + 0.13 * i / len(script.scenes), f"Preparing picture {i + 1}/{len(script.scenes)}")
            if scene.image:
                path = script.path(scene.image)
            elif scene.image_choice:
                path, credit = download_candidate(scene.image_choice, cache_dir)
                used.append(credit.source_url)
                credits.append(credit)
            else:
                path, credit = find_image(scene.image_query, cache_dir, box[0] / box[1],
                                          skip=tuple(used))
                used.append(credit.source_url)
                credits.append(credit)
            drawings.append(photo_to_pencil(load_image(path, f"Scene {i + 1} image"), box, seed=i))

        # ---- sound --------------------------------------------------------
        report(0.15, "Preparing narration")
        narration = decode_audio(script.path(script.narration)) if script.narration else None
        clips: List[Optional[np.ndarray]] = []
        for i, scene in enumerate(script.scenes):
            if scene.audio:
                clips.append(decode_audio(script.path(scene.audio)))
            elif preview_voice and narration is None:
                clip_path = os.path.join(tmp, f"voice_{i}.aiff")
                clips.append(decode_audio(speak(scene.text, preview_voice, clip_path)))
            else:
                clips.append(None)
        if narration is None and not any(c is not None for c in clips):
            warnings.append("No narration was given; scene lengths follow a reading pace.")
        offset = TITLE_SECONDS if script.title else 0.0
        timings, track = plan_timing(script, clips, narration, offset)
        story_end = timings[-1].start + timings[-1].duration
        credit_lines = [c.short() for c in credits]
        credit_cards = [credit_lines[i:i + 18] for i in range(0, len(credit_lines), 18)]
        total = story_end + CREDITS_SECONDS * len(credit_cards)
        n_frames = int(round(total * fps))

        # ---- frames -------------------------------------------------------
        captions = Captions(W, H)
        blank = paper_frame(W, H)
        tmp_video = os.path.join(tmp, "video.mp4")
        written = 0

        def frames_done(count: int, label: str) -> None:
            report(0.17 + 0.78 * min(count / n_frames, 1.0), label)

        with FFmpegWriter(tmp_video, W, H, fps, crf, preset, total_frames=n_frames) as writer:
            if script.title:
                card = text_card(W, H, script.title, [])
                n = int(round(TITLE_SECONDS * fps))
                for i in range(n):
                    t = i / fps
                    x = min(t / 0.6, 1.0, (TITLE_SECONDS - t) / CROSSFADE)
                    writer.write(_blend(blank, card, float(np.clip(x, 0.0, 1.0))))
                written += n
            for si, (scene, timing) in enumerate(zip(script.scenes, timings)):
                label = f"Scene {si + 1}/{len(script.scenes)}"
                frames_done(written, f"{label}: planning the drawing")
                canvas = _scene_canvas(drawings[si], W, H)
                draw_time = float(np.clip(0.62 * timing.duration, 4.5, 14.0))
                cfg = RenderConfig(width=W, height=H, fps=fps, duration=draw_time, hold=0.0,
                                   start_mode="blank", seed=7 + si, workers=0)
                plan = plan_animation(None, canvas, cfg)
                renderer = FrameRenderer(plan)
                chunks = captions.chunks(scene.text)
                n_scene = int(round((timing.start + timing.duration) * fps)) - written
                n_draw = min(n_scene, cfg.frame_count)
                fade_frames = int(round(CROSSFADE * fps))
                drawing_frames: Iterator = renderer.iter_frames(fps, n_draw)
                for k in range(n_scene):
                    t_local = k / fps
                    frame = next(drawing_frames)[2] if k < n_draw else plan.end_u8
                    if k >= n_scene - fade_frames:  # fade the page out for the next scene
                        frame = _blend(frame, blank, (k - (n_scene - fade_frames) + 1) / fade_frames)
                    text, alpha = caption_at(chunks, timing.speech, timing.start + t_local)
                    writer.write(captions.apply(frame, text, alpha))
                    if k % 15 == 0:
                        frames_done(written + k, f"{label}: drawing" if k < n_draw else
                                    f"{label}: narration")
                written += n_scene
                del plan, renderer
            for lines in credit_cards:
                card = text_card(W, H, "Hình ảnh / Images", lines + [
                    "", "Ảnh gốc từ Wikimedia Commons / Openverse, đã chuyển thành tranh chì."],
                    title_scale=0.045)
                n = int(round(CREDITS_SECONDS * fps))
                for i in range(n):
                    t = i / fps
                    writer.write(_blend(blank, card, float(np.clip(min(t / 0.6,
                                                                       (CREDITS_SECONDS - t) / 0.6),
                                                                   0.0, 1.0))))
                written += n
            frames_done(n_frames, "Encoding video")

        # ---- sound mix ----------------------------------------------------
        report(0.96, "Mixing sound")
        n_samples = int(total * SAMPLE_RATE)
        mix = np.zeros(n_samples, np.float32)
        mix[:min(len(track), n_samples)] += track[:n_samples]
        if script.music:
            music = decode_audio(script.path(script.music))
            if len(music):
                music = np.tile(music, int(math.ceil(n_samples / len(music))))[:n_samples]
                env = np.ones(n_samples, np.float32)
                fade_in, fade_out = int(1.5 * SAMPLE_RATE), int(3.0 * SAMPLE_RATE)
                env[:fade_in] = np.linspace(0, 1, fade_in)
                env[-fade_out:] = np.linspace(1, 0, fade_out)
                mix += music * env * script.music_volume
        peak = float(np.abs(mix).max()) if mix.size else 0.0
        if peak > 0.98:
            mix *= 0.98 / peak
        wav = os.path.join(tmp, "audio.wav")
        write_wav(wav, mix)
        _mux(tmp_video, wav, os.path.join(tmp, "final.mp4"))
        os.replace(os.path.join(tmp, "final.mp4"), output_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    credits_path = None
    if credits:
        credits_path = os.path.splitext(output_path)[0] + ".credits.txt"
        with open(credits_path, "w", encoding="utf-8") as fh:
            fh.write("Images (converted into pencil drawings):\n")
            for i, c in enumerate(credits, 1):
                fh.write(f"{i}. {c.line()}\n")
    report(1.0, "Done")
    return StoryResult(output_path, total, len(script.scenes), credits, credits_path,
                       time.perf_counter() - started, warnings)


def _mux(video: str, audio: str, out: str) -> None:
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", video,
           "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
           "-b:a", "160k", "-shortest", "-movflags", "+faststart", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"Could not add the sound track: {proc.stderr.strip()[-300:]}")


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="pencil-story",
        description="Render a narrated story video in which every scene is drawn in pencil. "
                    "Drawings and pencil movements are simulated with image processing.")
    p.add_argument("script", help="story script (JSON), or a text file with --template")
    p.add_argument("-o", "--output", required=True, help="output MP4 (or JSON with --template)")
    p.add_argument("--template", action="store_true",
                   help="turn a plain-text story into a script skeleton (one scene per paragraph)")
    p.add_argument("--preview-voice", metavar="VOICE",
                   help="read scenes without a recording with this macOS voice (e.g. Linh)")
    p.add_argument("--size", help="override the resolution, e.g. 1280x720")
    p.add_argument("--fps", type=int, help="override the frame rate")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="fast")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        if args.template:
            with open(args.script, encoding="utf-8") as fh:
                data = template_from_text(fh.read())
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            print(f"Wrote {args.output} with {len(data['scenes'])} scenes; check each "
                  "'image_query' (English keywords) before rendering.")
            return 0
        script = StoryScript.load(args.script)
        if args.size:
            script.width, script.height = (int(v) for v in args.size.lower().split("x"))
        if args.fps:
            script.fps = args.fps
        interactive = sys.stderr.isatty()

        def progress(fraction: float, message: str) -> None:
            if interactive:
                sys.stderr.write(f"\r[{'#' * int(fraction * 30):<30}] {fraction * 100:5.1f}%  "
                                 f"{message[:60]:<60}")
                sys.stderr.flush()

        result = render_story(script, args.output, progress, args.preview_voice, args.crf,
                              args.preset)
        if interactive:
            sys.stderr.write("\n")
    except PencilError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    minutes, seconds = divmod(int(round(result.duration)), 60)
    print(f"Saved {result.output_path} ({minutes}:{seconds:02d}, {result.scenes} scenes, "
          f"rendered in {result.elapsed:.0f} s)")
    if result.credits_path:
        print(f"  image credits: {result.credits_path}")
    for w in result.warnings:
        print(f"  note: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
