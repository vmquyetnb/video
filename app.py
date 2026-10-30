"""Web interface for the pencil-drawing timelapse generator.

Run:  python app.py            (then open http://127.0.0.1:8000)
      python app.py --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import shutil
import urllib.parse

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from pencil_timelapse import ai_writer, cartoon, cast, episodes, pets, props, scenery
from pencil_timelapse.animals import QUESTION as ANIMAL_QUESTION
from pencil_timelapse.config import (MAX_DURATION, MAX_FPS, MIN_DURATION, MIN_FPS,
                                     RESOLUTION_PRESETS, STAGE_PLAN, X264_PRESETS, RenderConfig)
from pencil_timelapse.encoder import ffmpeg_status
from pencil_timelapse.errors import PencilError
from pencil_timelapse.imaging import ALLOWED_EXTENSIONS, MIN_SIDE, load_image
from pencil_timelapse.jobs import JobManager
from pencil_timelapse.media import RateLimited, search_candidates
from pencil_timelapse.players import BACKGROUNDS, COUNTRIES, Player, find_players
from pencil_timelapse.showtime import ShowOptions
from pencil_timelapse.sources import Media, keys_from, search_media
from pencil_timelapse.story import template_from_text
from pencil_timelapse.travel import QUESTION as TRAVEL_QUESTION, REVEAL, VIDEO_TYPES

MAX_UPLOAD_MB = 40
MAX_AUDIO_MB = 150
MAX_VIDEO_MB = 200
MAX_SCENES = 80
STORY_RESOLUTIONS = {"1920x1080": (1920, 1080), "1280x720": (1280, 720)}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app(work_dir: str = None, max_concurrent: int = 1) -> Flask:
    app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
                static_folder=os.path.join(BASE_DIR, "static"))
    app.config["MAX_CONTENT_LENGTH"] = 600 * 1024 * 1024  # story uploads: pictures + recordings
    work_dir = work_dir or os.environ.get("PENCIL_WORK_DIR") or os.path.join(BASE_DIR, "runtime")
    jobs = JobManager(work_dir, max_concurrent=max_concurrent)
    app.extensions["pencil_jobs"] = jobs

    def error(message: str, status: int = 400):
        return jsonify({"error": message}), status

    @app.get("/")
    def index():
        defaults = RenderConfig()
        return render_template(
            "index.html", defaults=defaults, presets=list(RESOLUTION_PRESETS), x264_presets=X264_PRESETS,
            ffmpeg=ffmpeg_status(), max_upload_mb=MAX_UPLOAD_MB, min_side=MIN_SIDE,
            stages=[[title, units] for _, title, units in STAGE_PLAN],
            accept=",".join(ALLOWED_EXTENSIONS), limits={
                "min_fps": MIN_FPS, "max_fps": MAX_FPS,
                "min_duration": MIN_DURATION, "max_duration": MAX_DURATION})

    @app.get("/story")
    def story_page():
        return render_template("story.html", ffmpeg=ffmpeg_status(), max_upload_mb=MAX_UPLOAD_MB,
                               max_audio_mb=MAX_AUDIO_MB, preview_voice=bool(shutil.which("say")),
                               resolutions=list(STORY_RESOLUTIONS))

    @app.post("/api/story/scenes")
    def story_scenes():
        """Split a story into scenes automatically, with suggested image keywords."""
        body = request.get_json(silent=True) or {}
        text = body.get("text", "")
        if not str(text).strip():
            return error("Please paste the story first.")
        data = template_from_text(str(text), length=str(body.get("length", "medium")))
        if not data["scenes"]:
            return error("The story needs at least one paragraph after the title.")
        if len(data["scenes"]) > MAX_SCENES:
            return error(f"The story has {len(data['scenes'])} paragraphs; the limit is {MAX_SCENES}.")
        return jsonify(data)

    @app.get("/animals")
    def animals_page():
        return render_template("animals.html", ffmpeg=ffmpeg_status(),
                               max_upload_mb=MAX_UPLOAD_MB, max_audio_mb=MAX_AUDIO_MB)

    @app.post("/api/animals/render")
    def animals_render():
        """Start an animals video: a form with the script as JSON in ``data``
        ({animals: [{name, facts, photo, voice}], title, ...}), each animal's
        photo and optional narration as files named in it, and optional music."""
        status = ffmpeg_status()
        if not status["ok"]:
            return error(status["message"], 503)
        try:
            body = json.loads(request.form.get("data") or "{}")
        except ValueError:
            return error("Invalid request. Reload the page and try again.")
        chosen = body.get("animals") if isinstance(body, dict) else None
        if not isinstance(chosen, list) or not 1 <= len(chosen) <= 20:
            return error("Add between 1 and 20 animals.")

        def upload(field: str, label: str, limit_mb: int):
            item = request.files.get(field) if field else None
            if item is None or not item.filename:
                return None
            content = item.read()
            if len(content) > limit_mb * 1024 * 1024:
                raise PencilError(f"{label} is larger than {limit_mb} MB.")
            return content, item.filename

        animals, files = [], {}
        try:
            for i, item in enumerate(chosen):
                if not isinstance(item, dict):
                    return error("Invalid animal data. Reload the page and try again.")
                name = str(item.get("name") or "").strip()[:60]
                if not name:
                    return error(f"Animal {i + 1} needs a name.")
                facts = item.get("facts") or []
                if isinstance(facts, str):
                    facts = facts.splitlines()
                facts = [str(f).strip()[:300] for f in facts if str(f).strip()][:15]
                photo = upload(str(item.get("photo") or ""), f"The photo of {name}", MAX_UPLOAD_MB)
                if photo is None:
                    return error(f"Add a photo of {name}.")
                load_image(photo[0], f"Photo of {name}")
                voice = upload(str(item.get("voice") or ""), f"The recording for {name}",
                               MAX_AUDIO_MB)
                animals.append({"name": name, "facts": facts})
                files[i] = (photo, voice)
            music = upload("music", "The music", MAX_AUDIO_MB)
        except PencilError as exc:
            return error(str(exc))
        size = str(body.get("resolution", "1920x1080"))
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", size)
        if not match:
            return error("Unknown resolution.")
        try:
            seconds = float(body.get("seconds", 10))
        except (TypeError, ValueError):
            return error("Drawing time must be a number of seconds.")
        if not 4 <= seconds <= 60:
            return error("Drawing time must be between 4 and 60 seconds.")
        show = ShowOptions(intro=bool(body.get("intro", True)), highlights=True,
                           outro=bool(body.get("outro", True)),
                           sound_effects=bool(body.get("sound_effects", True)),
                           hook=str(body.get("hook") or "").strip()[:40],
                           question=str(body.get("question") or "").strip()[:80]
                           or ANIMAL_QUESTION)
        try:
            job = jobs.submit_animals(animals, files, music, str(body.get("title", "")).strip()[:50],
                                      int(match.group(1)), int(match.group(2)), seconds,
                                      bool(body.get("guess", True)), show)
        except PencilError as exc:
            return error(str(exc))
        return jsonify(job.to_dict(jobs.queue_position(job))), 202

    @app.get("/travel")
    def travel_page():
        return render_template("travel.html", ffmpeg=ffmpeg_status(), max_upload_mb=MAX_UPLOAD_MB,
                               max_audio_mb=MAX_AUDIO_MB, max_video_mb=MAX_VIDEO_MB,
                               models=ai_writer.PROVIDERS, question=TRAVEL_QUESTION, reveal=REVEAL,
                               env_keys={k: bool(os.environ.get(v)) for k, v in {
                                   "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
                                   "pexels": "PEXELS_API_KEY", "pixabay": "PIXABAY_API_KEY"}.items()})

    @app.post("/api/travel/write")
    def travel_write():
        """Let GPT or Gemini write the script: a theme or a list of places in,
        names, locations, coordinates, facts and photo search words out."""
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return error("Invalid request. Reload the page and try again.")
        places = body.get("places") or []
        if isinstance(places, str):
            places = places.splitlines()
        if not isinstance(places, list):
            return error("Invalid list of places.")
        try:
            script = ai_writer.write_script(
                str(body.get("provider", "")), str(body.get("key") or ""),
                [str(p)[:80] for p in places], str(body.get("theme") or "")[:200],
                int(body.get("count") or 7), int(body.get("facts") or 3),
                str(body.get("model") or ""))
        except (TypeError, ValueError):
            return error("The number of places and facts must be numbers.")
        except ai_writer.AIError as exc:
            return error(str(exc), 502)
        except PencilError as exc:
            return error(str(exc))
        return jsonify(script)

    @app.post("/api/travel/media")
    def travel_media():
        """Photos (and with a Pexels or Pixabay key, video clips) of a place."""
        body = request.get_json(silent=True) or {}
        query = str(body.get("q") or "").strip()[:100] if isinstance(body, dict) else ""
        if not query:
            return error("Type what to search for first.")
        given = body.get("keys") if isinstance(body.get("keys"), dict) else {}
        try:
            found, problems = search_media(query, keys_from({k: str(v) for k, v in given.items()}),
                                           videos=bool(body.get("videos", True)))
        except RateLimited as exc:
            return error(str(exc), 429)
        if not found and problems:
            return error("Could not search the photo libraries: " + "; ".join(problems), 502)
        return jsonify({"media": [m.to_dict() for m in found], "problems": problems})

    @app.post("/api/travel/render")
    def travel_render():
        """Start a travel video: a form with the script as JSON in ``data``
        ({places: [{name, facts, location, country, lat, lon, media | upload,
        voice}], title, ...}), the user's own pictures and narration as files
        named in it, and optional music. ``media`` is a search result."""
        status = ffmpeg_status()
        if not status["ok"]:
            return error(status["message"], 503)
        try:
            body = json.loads(request.form.get("data") or "{}")
        except ValueError:
            return error("Invalid request. Reload the page and try again.")
        chosen = body.get("places") if isinstance(body, dict) else None
        if not isinstance(chosen, list) or not 1 <= len(chosen) <= 20:
            return error("Add between 1 and 20 places.")

        def upload(field: str, label: str, limit_mb: int):
            item = request.files.get(field) if field else None
            if item is None or not item.filename:
                return None
            content = item.read()
            if len(content) > limit_mb * 1024 * 1024:
                raise PencilError(f"{label} is larger than {limit_mb} MB.")
            return content, item.filename

        def number(value, low: float, high: float):
            try:
                x = float(value)
            except (TypeError, ValueError):
                return None
            return x if low <= x <= high else None

        places, found, files = [], {}, {}
        try:
            for i, item in enumerate(chosen):
                if not isinstance(item, dict):
                    return error("Invalid place data. Reload the page and try again.")
                name = str(item.get("name") or "").strip()[:60]
                if not name:
                    return error(f"Place {i + 1} needs a name.")
                facts = item.get("facts") or []
                if isinstance(facts, str):
                    facts = facts.splitlines()
                facts = [str(f).strip()[:300] for f in facts if str(f).strip()][:15]
                field = str(item.get("upload") or "")
                own = request.files.get(field) if field else None
                is_video = bool(own and os.path.splitext(own.filename or "")[1].lower() in VIDEO_TYPES)
                picture = upload(field, f"The picture of {name}",
                                 MAX_VIDEO_MB if is_video else MAX_UPLOAD_MB)
                if picture is not None and not is_video:
                    load_image(picture[0], f"Photo of {name}")
                if picture is None:
                    if not isinstance(item.get("media"), dict):
                        return error(f"Choose a photo or video of {name}, or upload your own.")
                    found[i] = Media.from_dict(item["media"])
                voice = upload(str(item.get("voice") or ""), f"The recording for {name}",
                               MAX_AUDIO_MB)
                lat, lon = number(item.get("lat"), -90, 90), number(item.get("lon"), -180, 180)
                places.append({"name": name, "facts": facts,
                               "location": str(item.get("location") or "").strip()[:60],
                               "country": str(item.get("country") or "").strip()[:40],
                               "lat": lat if lon is not None else None,
                               "lon": lon if lat is not None else None})
                files[i] = (picture, voice)
            music = upload("music", "The music", MAX_AUDIO_MB)
        except PencilError as exc:
            return error(str(exc))
        size = str(body.get("resolution", "1920x1080"))
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", size)
        if not match:
            return error("Unknown resolution.")
        seconds = number(body.get("seconds", 10), 4, 60)
        if seconds is None:
            return error("Drawing time must be between 4 and 60 seconds.")
        reveal = number(body.get("reveal", REVEAL), 1.5, 15)
        if reveal is None:
            return error("The colour reveal must last between 1.5 and 15 seconds.")
        show = ShowOptions(intro=bool(body.get("intro", True)), highlights=True,
                           outro=bool(body.get("outro", True)),
                           sound_effects=bool(body.get("sound_effects", True)),
                           hook=str(body.get("hook") or "").strip()[:40],
                           question=str(body.get("question") or "").strip()[:80]
                           or TRAVEL_QUESTION)
        try:
            job = jobs.submit_travel(places, found, files, music,
                                     str(body.get("title", "")).strip()[:50], int(match.group(1)),
                                     int(match.group(2)), seconds, reveal,
                                     bool(body.get("guess", True)), show)
        except PencilError as exc:
            return error(str(exc))
        return jsonify(job.to_dict(jobs.queue_position(job))), 202

    @app.get("/pets")
    def pets_page():
        return render_template("pets.html", ffmpeg=ffmpeg_status(), max_upload_mb=MAX_UPLOAD_MB,
                               max_audio_mb=MAX_AUDIO_MB, max_video_mb=MAX_VIDEO_MB,
                               models=ai_writer.PROVIDERS, question=pets.QUESTION, hook=pets.HOOK,
                               reveal=pets.REVEAL, kinds=pets.KINDS, voices=list(cartoon.voices()),
                               pet_voices={k: {"voice": v, "pitch": p} for k, (v, p) in pets.VOICES.items()},
                               samples=pets.SAMPLES,
                               env_keys={k: bool(os.environ.get(v)) for k, v in {
                                   "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
                                   "pexels": "PEXELS_API_KEY", "pixabay": "PIXABAY_API_KEY"}.items()})

    @app.post("/api/pets/write")
    def pets_write():
        """Let GPT or Gemini write a pet's day: moments with the pet's thoughts,
        punchlines and search words, the teaser line and the guess."""
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return error("Invalid request. Reload the page and try again.")
        try:
            diary = ai_writer.write_diary(
                str(body.get("provider", "")), str(body.get("key") or ""), str(body.get("pet") or "")[:30],
                str(body.get("kind") or "dog")[:20], str(body.get("idea") or "")[:400],
                int(body.get("count") or 6), str(body.get("model") or ""))
        except (TypeError, ValueError):
            return error("The number of moments must be a number.")
        except ai_writer.AIError as exc:
            return error(str(exc), 502)
        except PencilError as exc:
            return error(str(exc))
        return jsonify(diary)

    @app.post("/api/pets/render")
    def pets_render():
        """Start a pet diary: a form with the diary as JSON in ``data`` ({pet,
        kind, title, hook, guess, answer, question, moments: [{time, title,
        thought, punch, best, media | upload}], ...}), the user's own photos and
        clips as files named in it, and optional music."""
        status = ffmpeg_status()
        if not status["ok"]:
            return error(status["message"], 503)
        try:
            body = json.loads(request.form.get("data") or "{}")
        except ValueError:
            return error("Invalid request. Reload the page and try again.")
        if not isinstance(body, dict):
            return error("Invalid request. Reload the page and try again.")
        items = body.get("moments")
        if not isinstance(items, list) or not 1 <= len(items) <= 12:
            return error("Add between 1 and 12 moments.")
        pet = str(body.get("pet") or "").strip()[:30]
        if not pet:
            return error("Give the pet a name.")

        def upload(field: str, label: str, limit_mb: int):
            item = request.files.get(field) if field else None
            if item is None or not item.filename:
                return None
            content = item.read()
            if len(content) > limit_mb * 1024 * 1024:
                raise PencilError(f"{label} is larger than {limit_mb} MB.")
            return content, item.filename

        def number(value, low: float, high: float):
            try:
                x = float(value)
            except (TypeError, ValueError):
                return None
            return x if low <= x <= high else None

        def text(value, limit: int) -> str:
            return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]

        moments, found, files = [], {}, {}
        try:
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    return error("Invalid moment data. Reload the page and try again.")
                m = pets.Moment(time=text(item.get("time"), 12), title=text(item.get("title"), 40),
                                thought=text(item.get("thought"), 160), punch=text(item.get("punch"), 90),
                                best=bool(item.get("best")))
                if not (m.title or m.thought):
                    return error(f"Moment {i + 1} needs a title or a thought.")
                field = str(item.get("upload") or "")
                own = request.files.get(field) if field else None
                is_video = bool(own and os.path.splitext(own.filename or "")[1].lower() in VIDEO_TYPES)
                picture = upload(field, f"The picture of moment {i + 1}", MAX_VIDEO_MB if is_video else MAX_UPLOAD_MB)
                if picture is not None and not is_video:
                    load_image(picture[0], f"Photo of moment {i + 1}")
                if picture is None:
                    if not isinstance(item.get("media"), dict):
                        return error(f"Choose a photo or clip for moment {i + 1}, or upload your own.")
                    found[i] = Media.from_dict(item["media"])
                else:
                    files[i] = picture
                moments.append(m)
            music = upload("music", "The music", MAX_AUDIO_MB)
        except PencilError as exc:
            return error(str(exc))
        guess = body.get("guess") or []
        if isinstance(guess, str):
            guess = guess.splitlines()
        diary = pets.Diary(pet=pet, kind=str(body.get("kind") or "dog") if body.get("kind") in pets.KINDS else "dog",
                           title=text(body.get("title"), 60), hook=text(body.get("hook"), 60) or pets.HOOK,
                           guess=[text(g, 40) for g in guess if text(g, 40)][:4], answer=text(body.get("answer"), 40),
                           question=text(body.get("question"), 80) or pets.QUESTION, moments=moments)
        size = str(body.get("resolution", "1080x1920"))
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", size)
        if not match:
            return error("Unknown resolution.")
        seconds = number(body.get("seconds", 8), 5, 30)
        if seconds is None:
            return error("Drawing time must be between 5 and 30 seconds.")
        reveal = number(body.get("reveal", pets.REVEAL), 1.5, 10)
        if reveal is None:
            return error("The real clip must last between 1.5 and 10 seconds.")
        tune = str(body.get("tune") or "happy")
        pitch = number(body.get("pitch", 0), 0, 1.8) or 0.0
        voice = str(body.get("voice") or "")
        if voice and voice not in cartoon.voices():
            voice = ""
        try:
            job = jobs.submit_pets(diary, found, files, music, int(match.group(1)), int(match.group(2)), seconds,
                                   reveal, tune if tune in ("happy", "calm") else None, voice, pitch,
                                   guess=bool(body.get("guess_on", True)), cold_open=bool(body.get("cold_open", True)),
                                   outro=bool(body.get("outro", True)),
                                   sound_effects=bool(body.get("sound_effects", True)))
        except PencilError as exc:
            return error(str(exc))
        return jsonify(job.to_dict(jobs.queue_position(job))), 202

    @app.get("/cartoon")
    def cartoon_page():
        found = sorted(cartoon.voices(), key=lambda v: (v not in cartoon.GOOD_VOICES, v))
        return render_template(
            "cartoon.html", ffmpeg=ffmpeg_status(), max_upload_mb=MAX_UPLOAD_MB,
            max_audio_mb=MAX_AUDIO_MB, voices=found, narrator=cartoon.NARRATOR,
            character_voice=cartoon.CHARACTER_VOICE,
            vocab={"settings": scenery.SETTINGS, "times": scenery.TIMES, "weather": scenery.WEATHER,
                   "actions": [a for a in cartoon.ACTIONS if a not in cartoon.INTERNAL],
                   "effects": cartoon.EFFECTS, "props": list(props.PROPS), "friends": list(cast.FRIENDS),
                   "moods": list(cartoon.MOODS)},
            episodes=episodes.catalogue(), groups=episodes.GROUPS, hero=cast.NAME, about=cast.ABOUT,
            friends=[(d.name, d.about) for d in cast.CAST.values() if d.name != cast.NAME],
            models=ai_writer.PROVIDERS,
            env_keys={k: bool(os.environ.get(v)) for k, v in ai_writer.ENV_KEYS.items()})

    @app.post("/api/cartoon/cutout")
    def cartoon_cutout():
        """The character cut out of its picture (PNG with transparency), to check it."""
        upload = request.files.get("picture")
        if upload is None or not upload.filename:
            return error("Choose the character's picture first.")
        content = upload.read()
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            return error(f"The picture is larger than {MAX_UPLOAD_MB} MB.", 413)
        try:
            rgba = cartoon.cut_out(io.BytesIO(content))
        except PencilError as exc:
            return error(str(exc))
        from PIL import Image
        out = io.BytesIO()
        Image.fromarray((rgba * 255 + 0.5).clip(0, 255).astype("uint8"), "RGBA").save(out, "PNG")
        out.seek(0)
        return send_file(out, mimetype="image/png", max_age=0)

    peeko_png = []

    def cartoon_about(name: str, hero: bool) -> str:
        """Who the AI writes for: the hero (Peeko or the user's own) and the friends."""
        me = f"{name} is " + (cast.PEEKO.about if hero else "the hero of the story") + "."
        friends = "\n".join(f"- {d.name}: {d.about}." for d in cast.CAST.values() if d.name != cast.NAME)
        return f"{me}\nThe hero's friends, who can appear and talk:\n{friends}"

    @app.get("/api/cartoon/peeko.png")
    def cartoon_peeko():
        """A picture of Peeko, the built-in character, in a few poses."""
        if not peeko_png:
            peeko_png.append(cast.portrait())
        return send_file(io.BytesIO(peeko_png[0]), mimetype="image/png", max_age=3600)

    cast_png = []

    @app.get("/api/cartoon/cast.png")
    def cartoon_cast():
        """Peeko and its friends side by side."""
        if not cast_png:
            cast_png.append(cast.lineup((1000, 360)))
        return send_file(io.BytesIO(cast_png[0]), mimetype="image/png", max_age=3600)

    @app.post("/api/cartoon/story")
    def cartoon_story():
        """A story for the character: a ready-made episode, the sample story,
        or one written by GPT or Gemini."""
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return error("Invalid request. Reload the page and try again.")
        hero = bool(body.get("peeko"))
        name = str(body.get("name") or "").strip()[:40] or (cast.NAME if hero else "Pip")
        if body.get("episode"):
            try:
                return jsonify(episodes.episode(str(body["episode"]), name).to_dict())
            except KeyError:
                return error("Unknown episode. Reload the page and try again.", 404)
        if body.get("sample"):
            return jsonify(cartoon.sample_story(name).to_dict())
        try:
            story = ai_writer.write_story(str(body.get("provider", "")), str(body.get("key") or ""),
                                          name, str(body.get("idea") or "")[:500],
                                          int(body.get("scenes") or 6), str(body.get("model") or ""),
                                          about=cartoon_about(name, hero))
        except (TypeError, ValueError):
            return error("The number of scenes must be a number.")
        except ai_writer.AIError as exc:
            return error(str(exc), 502)
        except PencilError as exc:
            return error(str(exc))
        return jsonify(story)

    @app.post("/api/cartoon/render")
    def cartoon_render():
        """Start a cartoon: a form with the character's ``picture`` (none for
        Peeko, ``character: "peeko"``), optional ``music`` and the rest as JSON
        in ``data`` ({character, name, faces, story, ...})."""
        status = ffmpeg_status()
        if not status["ok"]:
            return error(status["message"], 503)
        try:
            body = json.loads(request.form.get("data") or "{}")
        except ValueError:
            return error("Invalid request. Reload the page and try again.")
        if not isinstance(body, dict):
            return error("Invalid request. Reload the page and try again.")
        hero = body.get("character") == "peeko"
        picture = None
        if not hero:
            upload = request.files.get("picture")
            if upload is None or not upload.filename:
                return error("Add the character's picture, or choose Peeko.")
            content = upload.read()
            if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
                return error(f"The picture is larger than {MAX_UPLOAD_MB} MB.", 413)
            try:
                cartoon.cut_out(io.BytesIO(content))
            except PencilError as exc:
                return error(str(exc))
            picture = (content, upload.filename)
        story = cartoon.Story.from_dict(body.get("story") if isinstance(body.get("story"), dict) else {})
        story.scenes = [s for s in story.scenes if s.narration or s.say or s.lines]
        if not 1 <= len(story.scenes) <= 30:
            return error("The story needs between 1 and 30 scenes, each with a narration or a line.")
        faces = str(body.get("faces") or "right")
        if faces not in ("right", "left", "front"):
            return error("The character must face right, left or the front.")
        size = str(body.get("resolution", "1920x1080"))
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", size)
        if not match:
            return error("Unknown resolution.")
        known = set(cartoon.voices())
        narrator = str(body.get("narrator") or cartoon.NARRATOR)
        voice = str(body.get("voice") or cartoon.CHARACTER_VOICE)
        if known and (narrator not in known or voice not in known):
            return error("Unknown voice. Reload the page and choose the voices again.")
        try:
            pitch = float(body.get("pitch", 1.25))
        except (TypeError, ValueError):
            return error("The voice pitch must be a number.")
        if not 0.7 <= pitch <= 1.7:
            return error("The voice pitch must be between 0.7 and 1.7.")
        music = None
        sound = request.files.get("music")
        if sound is not None and sound.filename:
            data = sound.read()
            if len(data) > MAX_AUDIO_MB * 1024 * 1024:
                return error(f"The music is larger than {MAX_AUDIO_MB} MB.", 413)
            music = (data, sound.filename)
        story.question = story.question.strip()[:90]
        try:
            job = jobs.submit_cartoon(picture, str(body.get("name") or "").strip()[:40]
                                      or (cast.NAME if hero else "Pip"), "front" if hero else faces, story,
                                      int(match.group(1)), int(match.group(2)),
                                      narrator, voice, pitch, music, bool(body.get("intro", True)),
                                      bool(body.get("outro", True)),
                                      bool(body.get("sound_effects", True)), bool(body.get("series", False)))
        except PencilError as exc:
            return error(str(exc))
        return jsonify(job.to_dict(jobs.queue_position(job))), 202

    @app.get("/players")
    def players_page():
        return render_template("players.html", ffmpeg=ffmpeg_status(), countries=COUNTRIES,
                               max_upload_mb=MAX_UPLOAD_MB, max_audio_mb=MAX_AUDIO_MB)

    @app.get("/api/players")
    def players_search():
        """Famous players of a country's national team who have a portrait photo."""
        country = request.args.get("country", "").strip()
        if not country:
            return error("Choose a country first.")
        try:
            found = find_players(country, limit=20)
        except RateLimited:
            return error("Wikimedia is limiting requests right now. Wait a minute and try again.",
                         429)
        except PencilError as exc:
            return error(str(exc), 404)
        except Exception as exc:  # network trouble
            return error(f"Could not reach Wikidata: {exc}", 502)
        return jsonify({"country": country,
                        "players": [p.to_dict() for p in found if p.photo_url]})

    @app.post("/api/players/render")
    def players_render():
        """Start a players video. Accepts JSON, or a form with the same JSON in
        ``data`` plus the user's own photos and an optional ``audio`` file. A
        player is either a Wikimedia Commons find (``photo_url``) or one of the
        uploaded photos (``upload`` = the file field's name)."""
        status = ffmpeg_status()
        if not status["ok"]:
            return error(status["message"], 503)
        if request.files or "data" in request.form:
            try:
                body = json.loads(request.form.get("data") or "{}")
            except ValueError:
                return error("Invalid request. Reload the page and try again.")
        else:
            body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return error("Invalid request. Reload the page and try again.")
        chosen = body.get("players") or []
        if not isinstance(chosen, list) or not 1 <= len(chosen) <= 12:
            return error("Choose between 1 and 12 players.")
        players, uploads = [], {}
        for i, item in enumerate(chosen):
            if not isinstance(item, dict):
                return error("Invalid player data. Search for the players again.")
            name = str(item.get("name") or "").strip()[:60]
            if not name:
                return error("Every player needs a name.")
            upload = item.get("upload")
            if upload:
                photo = request.files.get(str(upload))
                if photo is None or not photo.filename:
                    return error(f"The photo of {name} is missing. Add it again.")
                content = photo.read()
                if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
                    return error(f"The photo of {name} is larger than {MAX_UPLOAD_MB} MB.", 413)
                try:
                    load_image(content, f"Photo of {name}")
                except PencilError as exc:
                    return error(str(exc))
                uploads[i] = (content, photo.filename)
                players.append(Player(name, "", photo.filename, 0))
                continue
            # A Commons find; ``local_file`` is never taken from the request.
            try:
                fields = {k: str(item.get(k) or "") for k in Player.__dataclass_fields__
                          if k not in ("fame", "local_file")}
                player = Player(**dict(fields, name=name, fame=int(item.get("fame") or 0)))
            except (TypeError, ValueError):
                return error("Invalid player data. Search for the players again.")
            host = urllib.parse.urlparse(player.photo_url).hostname or ""
            if not player.photo_url.startswith("https://") or not host.endswith(".wikimedia.org"):
                return error(f"The photo of {name} is not from Wikimedia Commons. "
                             "Search for the players again, or upload your own photo.")
            players.append(player)
        size = str(body.get("resolution", "1920x1080"))
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", size)
        if not match:
            return error("Unknown resolution.")
        try:
            seconds = float(body.get("seconds", 10))
        except (TypeError, ValueError):
            return error("Drawing time must be a number of seconds.")
        if not 4 <= seconds <= 60:
            return error("Drawing time per player must be between 4 and 60 seconds.")
        background = str(body.get("background", "whole"))
        if background not in BACKGROUNDS:
            return error("Unknown background choice.")
        audio = None
        sound = request.files.get("audio")
        if sound is not None and sound.filename:
            content = sound.read()
            if len(content) > MAX_AUDIO_MB * 1024 * 1024:
                return error(f"The audio file is larger than {MAX_AUDIO_MB} MB.", 413)
            audio = (content, sound.filename)
        title = str(body.get("title", "")).strip()[:40]
        show = ShowOptions(intro=bool(body.get("intro", True)),
                           highlights=bool(body.get("highlights", True)),
                           outro=bool(body.get("outro", True)),
                           sound_effects=bool(body.get("sound_effects", True)),
                           hook=str(body.get("hook") or "").strip()[:40],
                           question=str(body.get("question") or "").strip()[:80]
                           or ShowOptions.question)
        try:
            job = jobs.submit_players(players, title, int(match.group(1)), int(match.group(2)),
                                      seconds, background, uploads, audio,
                                      fit_to_audio=bool(body.get("fit_to_audio", True)), show=show)
        except PencilError as exc:
            return error(str(exc))
        return jsonify(job.to_dict(jobs.queue_position(job))), 202

    @app.get("/api/jobs/<job_id>/image")
    def job_image(job_id: str):
        job = jobs.get(job_id)
        if job is None or not job.image_path or not os.path.exists(job.image_path):
            abort(404)
        return send_file(job.image_path, mimetype="image/png", as_attachment=True,
                         download_name=f"{job.kind}-{job_id}.png", max_age=0)

    @app.get("/api/story/images")
    def story_images():
        """Free-licensed photo candidates for a scene, to pick from."""
        query = request.args.get("q", "").strip()
        if not query:
            return error("Enter English keywords first.")
        try:
            ranked, errors = search_candidates(query, 16 / 9 * 0.86 / 0.78, limit=8)
        except RateLimited:
            return error("Kho ảnh đang tạm giới hạn số lần tìm. Hãy đợi khoảng một phút rồi bấm "
                         "“Tìm lại”.", 429)
        if not ranked and errors:
            return error("Không kết nối được kho ảnh: " + errors[-1], 502)
        keys = ("url", "thumb", "title", "author", "licence", "source", "mime", "width", "height")
        return jsonify({"candidates": [{k: c.get(k) for k in keys} for c in ranked]})

    @app.post("/api/story")
    def create_story():
        status = ffmpeg_status()
        if not status["ok"]:
            return error(status["message"], 503)
        try:
            data = json.loads(request.form.get("script", ""))
            scenes = data["scenes"]
            assert isinstance(scenes, list) and scenes
        except (ValueError, KeyError, TypeError, AssertionError):
            return error("The story has no scenes yet. Paste the story and split it into scenes.")
        if len(scenes) > MAX_SCENES:
            return error(f"Too many scenes ({len(scenes)}); the limit is {MAX_SCENES}.")
        size = STORY_RESOLUTIONS.get(request.form.get("resolution", "1920x1080"))
        if size is None:
            match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", request.form.get("resolution", ""))
            if not match:
                return error("Unknown resolution.")
            size = (int(match.group(1)), int(match.group(2)))
        files = {}

        def take(field: str, label: str, limit_mb: int, image: bool):
            upload = request.files.get(field)
            if upload is None or not upload.filename:
                return None
            content = upload.read()
            if len(content) > limit_mb * 1024 * 1024:
                raise PencilError(f"{label} is larger than {limit_mb} MB.")
            if image:
                load_image(content, label)  # fail fast with a clear message
            files[field] = (content, upload.filename)
            return field

        script = {"title": str(data.get("title", "")).strip(), "width": size[0],
                  "height": size[1], "fps": 30, "scenes": [],
                  "music_volume": float(data.get("music_volume", 0.12))}
        try:
            script["narration"] = take("narration", "Narration", MAX_AUDIO_MB, False)
            script["music"] = take("music", "Background music", MAX_AUDIO_MB, False)
            for i, scene in enumerate(scenes):
                choice = scene.get("image_choice")
                script["scenes"].append({
                    "text": str(scene.get("text", "")),
                    "image_query": str(scene.get("image_query", "")),
                    "image_choice": choice if isinstance(choice, dict) else None,
                    "image": take(f"scene_{i}_image", f"Scene {i + 1} picture", MAX_UPLOAD_MB, True),
                    "audio": take(f"scene_{i}_audio", f"Scene {i + 1} recording", MAX_AUDIO_MB,
                                  False)})
            voice = "Linh" if request.form.get("preview_voice") == "true" else None
            if voice and not shutil.which("say"):
                raise PencilError("The preview voice needs macOS; upload a recording instead.")
            job = jobs.submit_story(script, files, voice)
        except PencilError as exc:
            return error(str(exc))
        return jsonify(job.to_dict(jobs.queue_position(job))), 202

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "ffmpeg": ffmpeg_status()})

    @app.post("/api/jobs")
    def create_job():
        status = ffmpeg_status()
        if not status["ok"]:
            return error(status["message"], 503)
        try:
            config = RenderConfig.from_mapping(request.form)
        except PencilError as exc:
            return error(str(exc))
        fields = [("end", "End image")]
        if config.start_mode == "image":  # blank-paper mode needs no start image
            fields.insert(0, ("start", "Start image"))
        files = {"start": (None, "")}
        for field, label in fields:
            upload = request.files.get(field)
            if upload is None or not upload.filename:
                hint = (" Or choose 'Blank white paper' as the start." if field == "start" else "")
                return error(f"{label} is missing. Please choose a file.{hint}")
            data = upload.read()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                return error(f"{label} is larger than {MAX_UPLOAD_MB} MB.", 413)
            try:
                load_image(data, label)  # fail fast with a clear message
            except PencilError as exc:
                return error(str(exc))
            files[field] = (data, upload.filename)
        job = jobs.submit(files["start"][0], files["start"][1], files["end"][0], files["end"][1],
                          config)
        return jsonify(job.to_dict(jobs.queue_position(job))), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return error("Unknown or expired job.", 404)
        return jsonify(job.to_dict(jobs.queue_position(job)))

    def _video(job_id: str, download: bool):
        job = jobs.get(job_id)
        if job is None or job.state != "done" or not os.path.exists(job.output_path or ""):
            abort(404)
        prefix = {"story": "story", "players": "players", "animals": "animals",
                  "travel": "travel", "pets": "pet-diary", "cartoon": "cartoon"}.get(job.kind, "pencil-drawing")
        name = f"{prefix}-{job_id}.mp4"
        return send_file(job.output_path, mimetype="video/mp4", conditional=True,
                         as_attachment=download, download_name=name if download else None,
                         max_age=0)

    @app.get("/api/jobs/<job_id>/video")
    def job_video(job_id: str):
        return _video(job_id, download=False)

    @app.get("/api/jobs/<job_id>/download")
    def job_download(job_id: str):
        return _video(job_id, download=True)

    @app.get("/api/jobs/<job_id>/credits")
    def job_credits(job_id: str):
        job = jobs.get(job_id)
        if job is None or not job.credits_path or not os.path.exists(job.credits_path):
            abort(404)
        return send_file(job.credits_path, mimetype="text/plain", as_attachment=True,
                         download_name=f"{job.kind}-{job_id}-image-credits.txt", max_age=0)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return error(f"Upload too large: images must be at most {MAX_UPLOAD_MB} MB and "
                     f"recordings at most {MAX_AUDIO_MB} MB.", 413)

    @app.errorhandler(404)
    def not_found(_exc):
        if request.path.startswith("/api/"):
            return error("Not found.", 404)
        return "Not found", 404

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Pencil-drawing timelapse web interface")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--work-dir", default=None, help="folder for uploads and rendered videos")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(args.work_dir)
    status = ffmpeg_status()
    if status["ok"]:
        logging.info("Using FFmpeg at %s (%s)", status["path"], status["encoder"])
    else:
        logging.warning("FFmpeg problem: %s", status["message"])
    print(f"Open http://{args.host}:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
