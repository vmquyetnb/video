"""Background render queue with progress tracking and temporary-file cleanup."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .config import RenderConfig
from .errors import InputError, PencilError
from .media import decode_audio
from .pipeline import render_video
from .animals import Animal, render_animals
from .cartoon import Story, render_cartoon
from .pets import Diary, Moment, render_pets
from .players import Player, render_players
from .showtime import ShowOptions
from .sources import Media, download
from .story import StoryScript, render_story
from .travel import VIDEO_TYPES, Place, render_travel, video_seconds

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]


@dataclass
class Job:
    id: str
    kind: str  # "portrait", "story", "players", "animals", "travel", "pets" or "cartoon"
    settings: Dict
    state: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = "Waiting in queue"
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    info: Dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    output_path: Optional[str] = None
    upload_dir: Optional[str] = None
    credits_path: Optional[str] = None
    image_path: Optional[str] = None  # finished picture (players)

    def to_dict(self, queue_position: int = 0) -> Dict:
        now = time.time()
        elapsed = ((self.finished or now) - self.started) if self.started else 0.0
        eta = None
        if self.state == "running" and self.progress > 0.2 and elapsed > 1:
            eta = max(0.0, elapsed / self.progress * (1.0 - self.progress))
        return {
            "id": self.id, "kind": self.kind, "state": self.state,
            "progress": round(self.progress, 4), "message": self.message, "error": self.error,
            "warnings": self.warnings, "queue_position": queue_position,
            "elapsed": round(elapsed, 1), "eta": round(eta, 1) if eta is not None else None,
            "config": self.settings, "info": self.info,
            "has_credits": bool(self.credits_path and os.path.exists(self.credits_path)),
            "has_image": bool(self.image_path and os.path.exists(self.image_path)),
        }


class JobManager:
    """Runs renders one (or a few) at a time on worker threads."""

    def __init__(self, work_dir: str, max_concurrent: int = 1, retention_hours: float = 6.0):
        self.work_dir = os.path.abspath(work_dir)
        self.upload_root = os.path.join(self.work_dir, "uploads")
        self.output_root = os.path.join(self.work_dir, "outputs")
        self.image_cache = os.path.join(self.work_dir, "story_images")
        self.retention = retention_hours * 3600.0
        os.makedirs(self.upload_root, exist_ok=True)
        os.makedirs(self.output_root, exist_ok=True)
        # Uploads from a previous run are never needed again.
        for name in os.listdir(self.upload_root):
            shutil.rmtree(os.path.join(self.upload_root, name), ignore_errors=True)
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="render")
        self.cleanup()

    # ------------------------------------------------------------------ submitting

    def _new_job(self, kind: str, settings: Dict) -> Job:
        self.cleanup()
        job_id = uuid.uuid4().hex[:12]
        upload_dir = os.path.join(self.upload_root, job_id)
        os.makedirs(upload_dir)
        return Job(job_id, kind, settings, upload_dir=upload_dir,
                   output_path=os.path.join(self.output_root, f"{job_id}.mp4"))

    def _start(self, job: Job, task: Callable[[ProgressFn], None]) -> Job:
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._pool.submit(self._run, job, task)
        return job

    def submit(self, start: Optional[bytes], start_name: str, end: bytes, end_name: str,
               config: RenderConfig) -> Job:
        """Queue a portrait render. ``start`` may be None when drawing on blank paper."""
        job = self._new_job("portrait", config.to_dict())
        start_path = None
        if start is not None:
            start_path = os.path.join(job.upload_dir, "start" + _safe_ext(start_name))
            with open(start_path, "wb") as fh:
                fh.write(start)
        end_path = os.path.join(job.upload_dir, "end" + _safe_ext(end_name))
        with open(end_path, "wb") as fh:
            fh.write(end)

        def task(progress: ProgressFn) -> None:
            result = render_video(start_path, end_path, job.output_path, config, progress)
            job.info = {k: v for k, v in result.info.items() if k != "config"}
            job.info.update({"frames": result.frames, "encoder": result.encoder,
                             "render_seconds": round(result.elapsed, 1),
                             "file_size": os.path.getsize(result.output_path)})
            job.warnings = result.warnings

        return self._start(job, task)

    def submit_story(self, data: Dict, files: Dict[str, Tuple[bytes, str]],
                     preview_voice: Optional[str] = None) -> Job:
        """Queue a story video.

        ``data`` is a story script whose file references are keys of ``files``
        (``{key: (content, original_name)}``). Raises :class:`InputError` for an
        invalid script or unreadable audio before anything is queued.
        """
        settings = {"width": data.get("width", 1920), "height": data.get("height", 1080),
                    "fps": data.get("fps", 30), "scenes": len(data.get("scenes") or [])}
        job = self._new_job("story", settings)
        try:
            names = {}
            for key, (content, original) in files.items():
                name = key + os.path.splitext(original or "")[1].lower()[:6]
                with open(os.path.join(job.upload_dir, name), "wb") as fh:
                    fh.write(content)
                names[key] = name

            def ref(value):
                return names.get(value) if value else None

            resolved = dict(data, narration=ref(data.get("narration")), music=ref(data.get("music")))
            resolved["scenes"] = [dict(sc, image=ref(sc.get("image")), audio=ref(sc.get("audio")))
                                  for sc in data.get("scenes") or []]
            script = StoryScript.from_dict(resolved, job.upload_dir)
            for label, value in [("Narration", script.narration), ("Music", script.music)] + [
                    (f"Scene {i} recording", sc.audio) for i, sc in enumerate(script.scenes, 1)]:
                if value:  # fail now, not after minutes of rendering
                    try:
                        decode_audio(script.path(value))
                    except InputError as exc:
                        raise InputError(f"{label}: {exc}") from None
        except Exception:
            shutil.rmtree(job.upload_dir, ignore_errors=True)
            raise

        def task(progress: ProgressFn) -> None:
            result = render_story(script, job.output_path, progress, preview_voice,
                                  cache_dir=self.image_cache)
            if result.credits_path:
                job.credits_path = os.path.join(self.output_root, f"{job.id}.credits.txt")
                os.replace(result.credits_path, job.credits_path)
            job.info = {"duration": round(result.duration, 1), "scenes": result.scenes,
                        "render_seconds": round(result.elapsed, 1),
                        "file_size": os.path.getsize(result.output_path),
                        "credits": [c.short() for c in result.credits]}
            job.warnings = result.warnings

        return self._start(job, task)

    def submit_players(self, players: List[Player], title: str, width: int, height: int,
                       seconds: float, background: str = "whole",
                       uploads: Optional[Dict[int, Tuple[bytes, str]]] = None,
                       audio: Optional[Tuple[bytes, str]] = None,
                       fit_to_audio: bool = True, show: Optional[ShowOptions] = None) -> Job:
        """Queue a footballers video: each player drawn in turn into one frame.

        ``uploads`` maps a player's index to their own photo (content, file name);
        ``audio`` is an optional sound file (content, file name). Raises
        :class:`InputError` for an unreadable sound file before anything is queued.
        """
        settings = {"width": width, "height": height, "fps": 30, "players": len(players),
                    "seconds_per_player": seconds, "title": title, "background": background,
                    "audio": bool(audio), "fit_to_audio": fit_to_audio,
                    "show": asdict(show) if show else None}
        job = self._new_job("players", settings)
        audio_path = None
        try:
            for i, (content, original) in (uploads or {}).items():
                path = os.path.join(job.upload_dir, f"photo_{i}{_safe_ext(original)}")
                with open(path, "wb") as fh:
                    fh.write(content)
                players[i].local_file = path
            if audio:
                content, original = audio
                audio_path = os.path.join(job.upload_dir,
                                          "audio" + os.path.splitext(original or "")[1].lower()[:6])
                with open(audio_path, "wb") as fh:
                    fh.write(content)
                decode_audio(audio_path, rate=8000)  # fail now, not after the drawing
        except Exception:
            shutil.rmtree(job.upload_dir, ignore_errors=True)
            raise

        def task(progress: ProgressFn) -> None:
            result = render_players(players, title, job.output_path, width, height,
                                    seconds_per_player=seconds,
                                    cache_dir=os.path.join(self.work_dir, "player_photos"),
                                    progress=progress, background=background,
                                    audio=audio_path, fit_to_audio=fit_to_audio, show=show)
            if result.credits_path:
                job.credits_path = os.path.join(self.output_root, f"{job.id}.credits.txt")
                os.replace(result.credits_path, job.credits_path)
            job.image_path = os.path.join(self.output_root, f"{job.id}.png")
            os.replace(result.image_path, job.image_path)
            job.info = {"duration": round(result.duration, 1), "players": result.players,
                        "render_seconds": round(result.elapsed, 1),
                        "file_size": os.path.getsize(result.output_path)}
            job.warnings = result.warnings

        return self._start(job, task)

    def submit_animals(self, animals: List[Dict], files: Dict[int, Tuple[Tuple[bytes, str],
                                                                       Optional[Tuple[bytes, str]]]],
                       music: Optional[Tuple[bytes, str]], title: str, width: int, height: int,
                       seconds: float, guess: bool = True,
                       show: Optional[ShowOptions] = None) -> Job:
        """Queue an animals video. ``animals`` are {name, facts}; ``files`` maps an
        animal's index to its photo and optional narration (content, file name).
        Raises :class:`InputError` for an unreadable recording before queueing."""
        settings = {"width": width, "height": height, "fps": 30, "animals": len(animals),
                    "seconds": seconds, "title": title, "guess": guess, "music": bool(music),
                    "show": asdict(show) if show else None}
        job = self._new_job("animals", settings)
        cast: List[Animal] = []
        music_path = None
        try:
            def save(name: str, item: Tuple[bytes, str], audio: bool) -> str:
                ext = os.path.splitext(item[1] or "")[1].lower()[:6] if audio else _safe_ext(item[1])
                path = os.path.join(job.upload_dir, name + ext)
                with open(path, "wb") as fh:
                    fh.write(item[0])
                if audio:
                    decode_audio(path, rate=8000)  # fail now, not after the drawing
                return path

            for i, a in enumerate(animals):
                photo, voice = files[i]
                cast.append(Animal(a["name"], save(f"photo_{i}", photo, False), list(a["facts"]),
                                   save(f"voice_{i}", voice, True) if voice else None))
            if music:
                music_path = save("music", music, True)
        except Exception:
            shutil.rmtree(job.upload_dir, ignore_errors=True)
            raise

        def task(progress: ProgressFn) -> None:
            result = render_animals(cast, title, job.output_path, width, height,
                                    drawing_seconds=seconds, music=music_path, guess=guess,
                                    show=show, progress=progress)
            job.image_path = os.path.join(self.output_root, f"{job.id}.png")
            os.replace(result.image_path, job.image_path)
            job.info = {"duration": round(result.duration, 1), "animals": result.animals,
                        "render_seconds": round(result.elapsed, 1),
                        "file_size": os.path.getsize(result.output_path)}
            job.warnings = result.warnings

        return self._start(job, task)

    def submit_travel(self, places: List[Dict], chosen: Dict[int, Media],
                      files: Dict[int, Tuple[Optional[Tuple[bytes, str]], Optional[Tuple[bytes, str]]]],
                      music: Optional[Tuple[bytes, str]], title: str, width: int, height: int,
                      seconds: float, reveal: float, guess: bool = True,
                      show: Optional[ShowOptions] = None) -> Job:
        """Queue a travel video. ``places`` are {name, facts, location, country,
        lat, lon}; a place's picture is either a photo or clip found online
        (``chosen``, downloaded when the job runs) or the user's own file
        (``files``: index -> (picture, narration), each (content, file name) or
        None). Raises :class:`InputError` for an unreadable upload before queueing."""
        settings = {"width": width, "height": height, "fps": 30, "places": len(places),
                    "seconds": seconds, "reveal": reveal, "title": title, "guess": guess,
                    "music": bool(music), "show": asdict(show) if show else None}
        job = self._new_job("travel", settings)
        trip: List[Place] = []
        music_path = None
        try:
            def save(name: str, item: Tuple[bytes, str], kind: str) -> str:
                ext = os.path.splitext(item[1] or "")[1].lower()[:6]
                if kind == "picture" and ext not in VIDEO_TYPES:
                    ext = _safe_ext(item[1])
                path = os.path.join(job.upload_dir, name + ext)
                with open(path, "wb") as fh:
                    fh.write(item[0])
                if kind == "sound":
                    decode_audio(path, rate=8000)  # fail now, not after the drawing
                elif ext in VIDEO_TYPES:
                    video_seconds(path)
                return path

            for i, p in enumerate(places):
                picture, voice = files.get(i, (None, None))
                media = save(f"picture_{i}", picture, "picture") if picture else ""
                trip.append(Place(p["name"], media, list(p["facts"]), p.get("location", ""),
                                  p.get("country", ""), p.get("lat"), p.get("lon"),
                                  save(f"voice_{i}", voice, "sound") if voice else None,
                                  chosen[i].credit() if i in chosen else ""))
            if music:
                music_path = save("music", music, "sound")
        except Exception:
            shutil.rmtree(job.upload_dir, ignore_errors=True)
            raise
        cache = os.path.join(self.work_dir, "travel_media")

        def task(progress: ProgressFn) -> None:
            for k, (i, media) in enumerate(sorted(chosen.items())):
                what = "clip" if media.kind == "video" else "photo"
                progress(0.01 * k / max(len(chosen), 1), f"Downloading the {what} of {trip[i].name}")
                trip[i].media = download(media, cache)
            result = render_travel(trip, title, job.output_path, width, height,
                                   drawing_seconds=seconds, music=music_path, guess=guess,
                                   show=show, reveal_seconds=reveal, progress=progress)
            job.image_path = os.path.join(self.output_root, f"{job.id}.png")
            os.replace(result.image_path, job.image_path)
            if result.credits_path:
                job.credits_path = os.path.join(self.output_root, f"{job.id}.credits.txt")
                os.replace(result.credits_path, job.credits_path)
            job.info = {"duration": round(result.duration, 1), "places": result.places,
                        "render_seconds": round(result.elapsed, 1),
                        "file_size": os.path.getsize(result.output_path)}
            job.warnings = result.warnings

        return self._start(job, task)

    def submit_pets(self, diary: Diary, chosen: Dict[int, Media], files: Dict[int, Tuple[bytes, str]],
                    music: Optional[Tuple[bytes, str]], width: int, height: int, seconds: float,
                    reveal: float, tune: Optional[str], voice: str, pitch: float, guess: bool = True,
                    cold_open: bool = True, outro: bool = True, sound_effects: bool = True) -> Job:
        """Queue a pet diary. A moment's picture is either a photo or clip found
        online (``chosen``, downloaded when the job runs) or the user's own file
        (``files``: index -> (content, file name)). Raises :class:`InputError`
        for an unreadable upload before queueing."""
        settings = {"width": width, "height": height, "fps": 30, "pet": diary.pet, "kind": diary.kind,
                    "moments": len(diary.moments), "seconds": seconds, "reveal": reveal, "tune": tune,
                    "voice": voice, "pitch": pitch, "music": bool(music), "guess": guess,
                    "cold_open": cold_open, "outro": outro}
        job = self._new_job("pets", settings)
        music_path = None
        try:
            def save(name: str, item: Tuple[bytes, str], kind: str) -> str:
                ext = os.path.splitext(item[1] or "")[1].lower()[:6]
                if kind == "picture" and ext not in VIDEO_TYPES:
                    ext = _safe_ext(item[1])
                path = os.path.join(job.upload_dir, name + ext)
                with open(path, "wb") as fh:
                    fh.write(item[0])
                if kind == "sound":
                    decode_audio(path, rate=8000)  # fail now, not after the drawing
                elif ext in VIDEO_TYPES:
                    video_seconds(path)
                return path

            for i, item in files.items():
                diary.moments[i].media = save(f"picture_{i}", item, "picture")
            if music:
                music_path = save("music", music, "sound")
        except Exception:
            shutil.rmtree(job.upload_dir, ignore_errors=True)
            raise
        cache = os.path.join(self.work_dir, "travel_media")

        def task(progress: ProgressFn) -> None:
            for k, (i, media) in enumerate(sorted(chosen.items())):
                what = "clip" if media.kind == "video" else "photo"
                progress(0.01 * k / max(len(chosen), 1), f"Downloading the {what} for moment {i + 1}")
                diary.moments[i].media = download(media, cache)
                diary.moments[i].credit = media.credit()
            result = render_pets(diary, job.output_path, width, height, drawing_seconds=seconds,
                                 reveal_seconds=reveal, music=music_path, tune=tune, voice=voice, pitch=pitch,
                                 guess=guess, cold_open=cold_open, outro=outro, sound_effects=sound_effects,
                                 progress=progress)
            job.image_path = os.path.join(self.output_root, f"{job.id}.png")
            os.replace(result.image_path, job.image_path)
            if result.credits_path:
                job.credits_path = os.path.join(self.output_root, f"{job.id}.credits.txt")
                os.replace(result.credits_path, job.credits_path)
            job.info = {"duration": round(result.duration, 1), "moments": result.moments,
                        "render_seconds": round(result.elapsed, 1),
                        "file_size": os.path.getsize(result.output_path)}
            job.warnings = result.warnings

        return self._start(job, task)

    def submit_cartoon(self, picture: Optional[Tuple[bytes, str]], name: str, faces: str, story: Story,
                       width: int, height: int, narrator: str, voice: str, pitch: float,
                       music: Optional[Tuple[bytes, str]] = None, intro: bool = True,
                       outro: bool = True, sound_effects: bool = True, series: bool = False) -> Job:
        """Queue a cartoon: the character in ``picture`` (content, file name;
        None for Peeko) acts out ``story``. Raises :class:`InputError` for
        unreadable music."""
        settings = {"width": width, "height": height, "fps": 30, "name": name, "faces": faces,
                    "character": "picture" if picture else "peeko",
                    "scenes": len(story.scenes), "narrator": narrator, "voice": voice, "pitch": pitch,
                    "music": bool(music), "intro": intro, "outro": outro, "series": series}
        job = self._new_job("cartoon", settings)
        try:
            path = None
            if picture is not None:
                path = os.path.join(job.upload_dir, "character" + _safe_ext(picture[1]))
                with open(path, "wb") as fh:
                    fh.write(picture[0])
            music_path = None
            if music:
                music_path = os.path.join(job.upload_dir,
                                          "music" + os.path.splitext(music[1] or "")[1].lower()[:6])
                with open(music_path, "wb") as fh:
                    fh.write(music[0])
                decode_audio(music_path, rate=8000)  # fail now, not after the animation
        except Exception:
            shutil.rmtree(job.upload_dir, ignore_errors=True)
            raise

        def task(progress: ProgressFn) -> None:
            result = render_cartoon(path, name, story, job.output_path, width, height, faces=faces,
                                    narrator=narrator, character_voice=voice, pitch=pitch,
                                    music=music_path, intro=intro, outro=outro,
                                    sound_effects=sound_effects, progress=progress, series=series)
            job.image_path = os.path.join(self.output_root, f"{job.id}.png")
            os.replace(result.image_path, job.image_path)
            job.info = {"duration": round(result.duration, 1), "scenes": result.scenes,
                        "render_seconds": round(result.elapsed, 1),
                        "file_size": os.path.getsize(result.output_path)}
            job.warnings = result.warnings

        return self._start(job, task)

    # ------------------------------------------------------------------ running

    def _run(self, job: Job, task: Callable[[ProgressFn], None]) -> None:
        job.state, job.started, job.message = "running", time.time(), "Starting"

        def progress(fraction: float, message: str) -> None:
            job.progress = max(job.progress, min(1.0, fraction))
            job.message = message

        try:
            task(progress)
            job.progress, job.message, job.state = 1.0, "Done", "done"
        except PencilError as exc:
            job.state, job.error, job.message = "error", str(exc), "Failed"
        except MemoryError:
            job.state, job.message = "error", "Failed"
            job.error = "Not enough memory for this resolution. Try a smaller output size."
        except Exception as exc:  # unexpected: keep the server alive and report it
            log.exception("Render %s failed", job.id)
            job.state, job.message = "error", "Failed"
            job.error = f"Unexpected error while rendering: {exc}"
        finally:
            job.finished = time.time()
            if job.upload_dir:
                shutil.rmtree(job.upload_dir, ignore_errors=True)
            if job.state != "done":
                for path in (job.output_path, job.credits_path, job.image_path):
                    if path and os.path.exists(path):
                        os.remove(path)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> int:
        if job.state != "queued":
            return 0
        with self._lock:
            queued = [j for j in self._order if self._jobs[j].state == "queued"]
        return queued.index(job.id) + 1 if job.id in queued else 0

    def cleanup(self) -> None:
        """Forget finished jobs older than the retention period and delete their files."""
        cutoff = time.time() - self.retention
        with self._lock:
            for job_id in list(self._order):
                job = self._jobs[job_id]
                if job.finished and job.finished < cutoff:
                    for path in (job.output_path, job.credits_path, job.image_path):
                        if path and os.path.exists(path):
                            os.remove(path)
                    del self._jobs[job_id]
                    self._order.remove(job_id)
            known = {f"{j}{suffix}" for j in self._jobs
                     for suffix in (".mp4", ".credits.txt", ".png")}
        media = os.path.join(self.work_dir, "travel_media")  # downloaded clips are large
        for name in os.listdir(media) if os.path.isdir(media) else []:
            path = os.path.join(media, name)
            try:
                if os.path.getmtime(path) < cutoff - 18 * 3600:
                    os.remove(path)
            except OSError:
                pass
        for name in os.listdir(self.output_root):
            path = os.path.join(self.output_root, name)
            try:
                stale = os.path.getmtime(path) < cutoff
            except OSError:
                continue
            if name not in known and stale:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _safe_ext(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff") else ".img"
