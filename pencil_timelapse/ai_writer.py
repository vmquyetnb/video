"""Let GPT (OpenAI) or Gemini (Google) write a travel video's script: for
each place its name, where it is, its coordinates, a few facts and the
words to search a photo with, plus the intro headline and closing question.

The API key comes from the request (typed on the page) or the environment
(``OPENAI_API_KEY``, ``GEMINI_API_KEY``). The model can be changed; the
defaults are small, quick ones.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from .errors import InputError, RenderError

PROVIDERS = {"openai": "gpt-5-mini", "gemini": "gemini-flash-latest"}
NAMES = {"openai": "OpenAI", "gemini": "Gemini"}
ENV_KEYS = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
MAX_PLACES = 20

SYSTEM = """You write scripts for YouTube videos in which famous travel destinations are drawn in \
pencil one after another while short facts about each are written beside the drawing.
Write in clear, lively English for a general audience. Only state facts that are well \
established; never invent numbers, records or dates. Prefer surprising facts with a number in \
them. No emojis, no hashtags. Answer with JSON only."""


class AIError(RenderError):
    """The AI service could not write the script."""


def _ask(places: List[str], theme: str, count: int, facts: int) -> str:
    if places:
        listed = "\n".join(f"{i}. {p}" for i, p in enumerate(places, 1))
        what = f"Write about exactly these places, in this order:\n{listed}"
    else:
        what = (f"Choose the {count} most famous and most beautiful places for this theme: "
                f"{theme.strip()}.\nOrder them to keep viewers watching, the most spectacular last.")
    return f"""{what}

Return one JSON object:
{{
  "title": "video title, at most 40 characters",
  "hook": "intro headline, at most 26 characters, punchy, in capitals",
  "question": "closing question asking viewers which place to draw next, at most 70 characters",
  "places": [
    {{
      "name": "the usual English name, at most 32 characters",
      "location": "city or region, country; at most 36 characters",
      "country": "the country, in English",
      "lat": latitude in decimal degrees (number),
      "lon": longitude in decimal degrees (number),
      "facts": [{facts} facts, each a single sentence of at most 90 characters],
      "search": "3 to 6 English words to find a photo of the place itself"
    }}
  ]
}}"""


def _post(url: str, body: dict, headers: Dict[str, str], provider: str, model: str,
          timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    name = NAMES[provider]
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            detail = ""
        detail = f" ({detail[:200]})" if detail else ""
        if exc.code in (401, 403):
            raise AIError(f"{name} did not accept the API key{detail}.") from None
        if exc.code == 404:
            raise AIError(f"{name} has no model called '{model}'{detail}. Check the model name.") from None
        if exc.code == 429:
            raise AIError(f"{name} says the quota or rate limit is reached{detail}. Wait a little "
                          "or check your plan.") from None
        if exc.code == 400:
            raise AIError(f"{name} refused the request{detail}.") from None
        raise AIError(f"{name} answered with an error ({exc.code}){detail}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AIError(f"Could not reach {name}: {getattr(exc, 'reason', exc)}") from None


def _openai(key: str, model: str, prompt: str, timeout: float, system: str = SYSTEM) -> str:
    body = {"model": model, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    if re.match(r"(gpt-5|o\d)", model):
        body["reasoning_effort"] = "low"  # facts, not puzzles: answer quickly
    data = _post("https://api.openai.com/v1/chat/completions", body,
                 {"Authorization": f"Bearer {key}"}, "openai", model, timeout)
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise AIError("OpenAI sent an answer without text.") from None


def _gemini(key: str, model: str, prompt: str, timeout: float, system: str = SYSTEM) -> str:
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7}}
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{urllib.parse.quote(model, safe='-.')}:generateContent")
    data = _post(url, body, {"x-goog-api-key": key}, "gemini", model, timeout)
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        reason = (data.get("promptFeedback") or {}).get("blockReason") if isinstance(data, dict) else None
        raise AIError("Gemini sent an answer without text" + (f" ({reason})." if reason else ".")) from None
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _number(value, low: float, high: float) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if low <= x <= high else None


def _short(value, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip(" ,.;") + "…"


def parse_script(text: str, facts: int = 3) -> Dict:
    """The script in an AI answer, checked and trimmed to what the video can show."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        raise AIError("The AI's answer was not valid JSON. Try again.") from None
    places = []
    for item in data.get("places") or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        found = [_short(f, 140) for f in item.get("facts") or [] if str(f).strip()]
        places.append({
            "name": _short(item.get("name"), 40), "location": _short(item.get("location"), 48),
            "country": _short(item.get("country"), 32),
            "lat": _number(item.get("lat"), -90, 90), "lon": _number(item.get("lon"), -180, 180),
            "facts": found[:max(1, facts)],
            "search": _short(item.get("search") or item.get("name"), 80)})
    if not places:
        raise AIError("The AI's answer had no places in it. Try again.")
    return {"title": _short(data.get("title"), 50), "hook": _short(data.get("hook"), 40).upper(),
            "question": _short(data.get("question"), 80), "places": places[:MAX_PLACES]}


def _service(provider: str, key: str, model: str) -> tuple:
    """The checked provider, key and model."""
    if provider not in PROVIDERS:
        raise InputError("Choose GPT (OpenAI) or Gemini.")
    key = (key or os.environ.get(ENV_KEYS[provider], "")).strip()
    if not key:
        raise InputError(f"Add your {NAMES[provider]} API key first.")
    model = (model or PROVIDERS[provider]).strip()
    if not re.fullmatch(r"[A-Za-z0-9._:\-/]{1,80}", model):
        raise InputError("The model name has characters a model name cannot have.")
    return provider, key, model


def write_script(provider: str, key: str = "", places: Optional[List[str]] = None, theme: str = "",
                 count: int = 7, facts: int = 3, model: str = "", timeout: float = 150.0) -> Dict:
    """Ask ``provider`` ("openai" or "gemini") for a script about ``places`` (in
    that order) or, without them, about ``count`` places chosen for ``theme``."""
    provider, key, model = _service(provider, key, model)
    places = [p.strip() for p in places or [] if p.strip()][:MAX_PLACES]
    if not places and not theme.strip():
        raise InputError("Type a theme (for example 'Wonders of Asia') or a list of places.")
    count = max(1, min(int(count), MAX_PLACES))
    facts = max(1, min(int(facts), 6))
    prompt = _ask(places, theme, count, facts)
    answer = (_openai if provider == "openai" else _gemini)(key, model, prompt, timeout)
    return parse_script(answer, facts)


# --------------------------------------------------------------------------- cartoon stories

STORY_SYSTEM = """You write episodes of a cartoon series on YouTube for children of about 3 to 6. \
Write in simple, lively English with short sentences. Give each friend their own personality and \
let them talk to each other. Start with a funny or surprising teaser moment, then a small problem, \
and a happy ending. Teach one simple thing, repeat its key words, and let the viewers join in \
(counting together, answering a picture quiz, saying a word out loud). Answer with JSON only."""


def _story_prompt(name: str, idea: str, scenes: int, about: str = "") -> str:
    from .cartoon import ACTIONS, EFFECTS, INTERNAL, MOODS
    from .props import PROPS
    from .scenery import SETTINGS, TIMES, WEATHER
    actions = ", ".join(a for a in ACTIONS if a not in INTERNAL)
    topic = f"The episode is about: {idea.strip()}" if idea.strip() else "Choose a fun story that teaches " \
        "something small."
    who = f"{about.strip()}\n" if about.strip() else ""
    return f"""The hero is called {name}.
{who}{topic}
Write {scenes} scenes. The first scene is a short teaser (no narration, two or three lines) showing a \
funny or exciting moment, shown before the show's opening. Each scene can only use these values:
- setting: {", ".join(SETTINGS)}
- time: {", ".join(TIMES)}
- weather: {", ".join(WEATHER)}
- actions (what {name} does while the narrator speaks, in order, 1 to 3 of them): {actions}
- cast: the friends in the scene (0 to 3 names); write "sad Lulu" for a friend who is sad, scared or \
sleepy ({", ".join(MOODS)})
- lines: what the characters say after the narration, in order, 1 to 4 lines of at most 70 characters: \
{{"who": "{name}" or a friend in the cast or "Narrator", "say": "...", "do": one of the actions or ""}}
- effect: {", ".join(EFFECTS)}
- holds (something in {name}'s hand, or ""): {", ".join(PROPS)}
- board: a few words written on a chalkboard (at most 30 characters; colour names are written in \
their colour), or ""
- item (a picture on the chalkboard, or ""): the same list as "holds"
- count: 0, or 2 to 10 to count the item (or plain numbers) together; the narrator then says the \
numbers one by one, so the narration only invites the viewers to count
- quiz: 2 or 3 pictures from the list for a picture quiz (the narration asks the question, a 3-2-1 \
countdown follows), or []; answer: the right picture
Use "enter" only as the first action of a scene and "exit" only as the last. Use "eat" for eating \
and drinking, "brush" for brushing teeth, "wash" for washing hands. Use the board, counting and the \
quiz in a few scenes, not all.

Return one JSON object:
{{
  "title": "the episode's title, at most 40 characters",
  "lesson": "what the episode teaches, in one short sentence a child can remember, at most 90 characters",
  "question": "a closing question for the viewers, at most 70 characters",
  "next": "a teaser title for the next episode, at most 40 characters",
  "music": "happy" or "calm" (calm for bedtime stories),
  "scenes": [
    {{
      "setting": "...", "time": "...", "weather": "...",
      "narration": "what happens, told by a narrator: 1 or 2 short sentences, at most 200 characters",
      "actions": ["...", "..."], "cast": ["..."],
      "lines": [{{"who": "...", "say": "...", "do": ""}}],
      "effect": "...", "holds": "", "board": "", "item": "", "count": 0, "quiz": [], "answer": ""
    }}
  ]
}}"""


def parse_story(text: str) -> Dict:
    """The story in an AI answer, with every value one the animation knows."""
    from .cartoon import Story
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        raise AIError("The AI's answer was not valid JSON. Try again.") from None
    if isinstance(data, dict) and data.get("scenes"):
        data.setdefault("hook", True)
    story = Story.from_dict(data if isinstance(data, dict) else {})
    story.scenes = [s for s in story.scenes if s.narration or s.say or s.lines][:30]
    if not story.scenes:
        raise AIError("The AI's answer had no scenes in it. Try again.")
    return story.to_dict()


def write_story(provider: str, key: str = "", name: str = "", idea: str = "", scenes: int = 6,
                model: str = "", timeout: float = 150.0, about: str = "") -> Dict:
    """Ask ``provider`` for a cartoon story of ``scenes`` scenes starring
    ``name``; ``about`` describes the hero (Peeko's description for Peeko)."""
    provider, key, model = _service(provider, key, model)
    name = re.sub(r"\s+", " ", name or "").strip()[:40] or "Pip"
    scenes = max(2, min(int(scenes), 12))
    prompt = _story_prompt(name, idea[:500], scenes, about[:400])
    answer = (_openai if provider == "openai" else _gemini)(key, model, prompt, timeout, STORY_SYSTEM)
    return parse_story(answer)


# --------------------------------------------------------------------------- pet diaries

DIARY_SYSTEM = """You write original, funny "day in the life" videos told by a pet in the first person, \
for a family audience on YouTube and Shorts. Each moment of the pet's day is drawn in pencil while the \
pet's thought is shown, then the real clip plays with a short punchline. Give the pet a clear, lovable \
personality; keep every line short, warm and funny (setup while drawing, payoff on the clip). Build a \
small story through the day that ends in one big moment. Write everything yourself; do not reuse other \
creators' jokes, catchphrases or titles. No emojis, no hashtags. Answer with JSON only."""


def _diary_prompt(pet: str, kind: str, idea: str, moments: int) -> str:
    topic = f"The day is about: {idea.strip()}" if idea.strip() else "Choose a typical, funny day."
    return f"""The pet is a {kind} called {pet}.
{topic}
Write {moments} moments of the day in order, from morning to evening. Mark exactly one moment, late in \
the day, as the big moment ("best": true): it is shown first as a teaser and the viewers guess what \
happens in it.

Return one JSON object:
{{
  "title": "video title, at most 45 characters",
  "hook": "what {pet} says over the frozen teaser of the big moment, at most 40 characters, a question \
that makes viewers curious",
  "question": "closing question for the comments about the viewers' own pets, at most 70 characters",
  "guess": ["three short answers to 'What happens next?' before the big moment, at most 32 characters \
each, one of them right"],
  "answer": "the right answer, exactly as written in guess",
  "moments": [
    {{
      "time": "time of day like 7:00 AM",
      "title": "two to four words",
      "thought": "what {pet} thinks while the moment is drawn, first person, at most 110 characters",
      "punch": "what {pet} says when the real clip plays: a short payoff, at most 60 characters",
      "search": "3 to 6 English words to find a stock photo or video of a {kind} doing this",
      "best": false
    }}
  ]
}}"""


def parse_diary(text: str) -> Dict:
    """The diary in an AI answer, checked and trimmed to what the video can show."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        raise AIError("The AI's answer was not valid JSON. Try again.") from None
    if not isinstance(data, dict):
        raise AIError("The AI's answer had no moments in it. Try again.")
    moments = []
    for item in data.get("moments") or []:
        if not isinstance(item, dict) or not (str(item.get("thought") or "").strip()
                                              or str(item.get("title") or "").strip()):
            continue
        moments.append({"time": _short(item.get("time"), 12), "title": _short(item.get("title"), 40),
                        "thought": _short(item.get("thought"), 140), "punch": _short(item.get("punch"), 80),
                        "search": _short(item.get("search") or item.get("title"), 80),
                        "best": bool(item.get("best"))})
    if not moments:
        raise AIError("The AI's answer had no moments in it. Try again.")
    moments = moments[:12]
    marked = [i for i, m in enumerate(moments) if m["best"]]
    for i, m in enumerate(moments):
        m["best"] = i == (marked[0] if marked else len(moments) - 1)
    guess = [_short(g, 40) for g in data.get("guess") or [] if str(g).strip()][:4]
    answer = _short(data.get("answer"), 40)
    if answer and answer not in guess and len(guess) < 4:
        guess.append(answer)
    return {"title": _short(data.get("title"), 60), "hook": _short(data.get("hook"), 60),
            "question": _short(data.get("question"), 80), "guess": guess,
            "answer": answer if answer in guess else "", "moments": moments}


def write_diary(provider: str, key: str = "", pet: str = "", kind: str = "dog", idea: str = "",
                moments: int = 6, model: str = "", timeout: float = 150.0) -> Dict:
    """Ask ``provider`` for a pet diary of ``moments`` moments told by ``pet``."""
    provider, key, model = _service(provider, key, model)
    pet = re.sub(r"\s+", " ", pet or "").strip()[:30] or "Biscuit"
    kind = re.sub(r"[^a-z ]", "", (kind or "dog").lower())[:20] or "dog"
    moments = max(3, min(int(moments), 12))
    answer = (_openai if provider == "openai" else _gemini)(key, model, _diary_prompt(pet, kind, idea[:400], moments),
                                                            timeout, DIARY_SYSTEM)
    return parse_diary(answer)
