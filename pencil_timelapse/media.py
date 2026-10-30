"""Media helpers for story videos: free-licence image search, audio handling.

Images come from Wikimedia Commons (all freely licensed, no API key) with
Openverse as a fallback. Every downloaded image is cached together with its
author and licence so the video can credit it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np

from .encoder import find_ffmpeg
from .errors import InputError, RenderError

log = logging.getLogger(__name__)

USER_AGENT = "PencilTimelapseStudio/1.0 (local story-video tool; contact: local user)"
SAMPLE_RATE = 44100
# Results whose title suggests violence or graphic content, or that are scans,
# signs and diagrams rather than photos, are skipped.
_UNSUITABLE = re.compile(r"\b(armed|gun|guns|rifle|weapon|corpse|dead body|bloody|bloodshed|gore|nude|naked|"
                         r"execution|war crime|massacre|logo|map|diagram|chart|flag|coat of arms|"
                         r"signature|icon|catalogue|catalog|journal|magazine|bulletin|"
                         r"report|newspaper|poster|advertisement|sign|banner|text|document|"
                         r"letter|manuscript|illustration|drawing|engraving|cartoon|stamp|"
                         r"coin|plan|blueprint|screenshot|distribution|infographic|"
                         r"anatomy|caricature|statue|sculpture|rack)s?\b", re.I)
_ANIMALS = re.compile(r"\b(baboon|monkey|ape|gorilla|chimpanzee|dog|puppy|cat|kitten|bird|lion|"
                      r"tiger|jaguar|leopard|cheetah|elephant|rhinoceros|rhino|hippo|hippopotamus|"
                      r"horse|foal|cow|calf|cattle|buffalo|bison|goat|sheep|lamb|pig|camel|donkey|"
                      r"wolf|fox|bear|cub|deer|zebra|giraffe|squirrel|rabbit|duck|chicken|hen|chick|"
                      r"owl|eagle|parrot|penguin|seal|whale|dolphin|shark|fish|frog|turtle|"
                      r"crocodile|lizard|snake|insect|butterfly|bee|ant|spider|wildlife|zoo)s?\b",
                      re.I)
# Buildings, institutions, monuments and famous people only fit when asked for:
# "Queen Mother" is not a photo of a mother, nor a "Teacher-Student Center" of a teacher.
_CONTEXT = re.compile(r"\b(center|centre|cathedral|church|temple|mosque|pagoda|university|college|"
                      r"auditorium|building|hall|museum|station|monument|memorial|tomb|queen|king|"
                      r"prince|princess|emperor|president|minister|senator|governor|mayor|general|"
                      r"portrait|ceremony|parade|conference|summit|election|p\.\s?\d+)s?\b", re.I)
# Stories are about people: photos that show people are preferred.
_PEOPLE = {"man", "men", "woman", "women", "child", "children", "kid", "boy", "girl", "people",
           "person", "family", "student", "teacher", "doctor", "nurse", "patient", "mother",
           "father", "elderly", "old", "senior", "baby", "grandmother", "grandfather", "parent",
           "worker", "farmer", "volunteer", "friend", "pupil", "lady", "couple"}
# Book scans on Commons usually carry the publication year in brackets.
_SCAN_TITLE = re.compile(r"\((1[5-9]\d\d)\)")
_SCAN_AUTHORS = ("internet archive book images",)


@dataclass
class ImageCredit:
    title: str
    author: str
    licence: str
    source_url: str
    file: str

    def line(self) -> str:
        who = self.author or "unknown author"
        return f"{self.title} - {who}, {self.licence}, {self.source_url}"

    def short(self) -> str:
        """Title, author and licence without the long URL (for on-screen credits)."""
        title = re.sub(r"\s*\([^)]*\d{5,}[^)]*\)", "", self.title).strip()  # drop archive ids
        title = title if len(title) <= 70 else title[:67].rstrip() + "..."
        return f"{title} - {self.author or 'unknown author'} ({self.licence})"


# --------------------------------------------------------------------------- HTTP

# Wikimedia asks API clients to send requests one at a time and to back off
# when told to (HTTP 429). All requests go through this polite gate.
MIN_INTERVAL = 0.6
_GATE = threading.Lock()
_last_request = [0.0]


class RateLimited(RenderError):
    """The image archive refused more requests for now."""


def _read(resp, url: str, limit: float) -> bytes:
    """The whole response body; gives up on a download that trickles in for
    longer than ``limit`` seconds (the socket timeout only covers each read)."""
    end = time.monotonic() + limit
    chunks = []
    while True:
        chunk = resp.read(1 << 16)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        if time.monotonic() > end:
            raise TimeoutError(f"Downloading {url} took too long.")


def _get(url: str, timeout: float = 30.0, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        with _GATE:  # one request at a time; a back-off also pauses the others
            wait = _last_request[0] + MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                    return _read(resp, url, 4.0 * timeout)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 503):
                    raise
                if attempt == retries - 1:
                    raise RateLimited("The image archive is limiting requests right now. Wait a "
                                      "minute, then search again.") from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 ** (attempt + 1)
                time.sleep(min(delay, 20.0))
            finally:
                _last_request[0] = time.monotonic()
    raise RateLimited("The image archive is limiting requests right now.")


_SEARCH_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _cached(kind: str, query: str, fetch) -> List[dict]:
    """Remember search results for this session (re-searching costs requests)."""
    key = (kind, query.strip().lower())
    with _CACHE_LOCK:
        if key in _SEARCH_CACHE:
            return _SEARCH_CACHE[key]
    result = fetch()
    with _CACHE_LOCK:
        _SEARCH_CACHE[key] = result
    return result


def _clean(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- image search

def search_commons(query: str, limit: int = 30) -> List[dict]:
    """Photos on Wikimedia Commons matching ``query``, best candidates first."""
    return _cached("commons", query, lambda: _search_commons(query, limit))


def _search_commons(query: str, limit: int) -> List[dict]:
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6", "gsrlimit": str(limit),
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "1920",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    data = json.loads(_get(url))
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        url = info.get("thumburl") or info.get("url")
        out.append({
            "title": re.sub(r"^File:", "", page.get("title", "")).rsplit(".", 1)[0],
            "url": url, "thumb": re.sub(r"/\d+px-", "/330px-", url or ""),
            "width": info.get("width", 0), "height": info.get("height", 0),
            "mime": info.get("mime", ""),
            "author": _clean(meta.get("Artist", {}).get("value", "")),
            "licence": _clean(meta.get("LicenseShortName", {}).get("value", "")) or "see source",
            "source": info.get("descriptionurl", ""),
        })
    return out


_UNREACHABLE: set = set()


def search_openverse(query: str, limit: int = 20) -> List[dict]:
    """CC-licensed images from Openverse that allow modification and commercial use."""
    if "openverse" in _UNREACHABLE:
        return []
    return _cached("openverse", query, lambda: _search_openverse(query, limit))


def _search_openverse(query: str, limit: int) -> List[dict]:
    params = {"q": query, "license_type": "commercial,modification", "page_size": str(limit)}
    try:
        data = json.loads(_get("https://api.openverse.org/v1/images/?"
                               + urllib.parse.urlencode(params), timeout=8))
    except OSError:
        _UNREACHABLE.add("openverse")  # don't wait for it again in this session
        raise
    return [{
        "title": r.get("title") or query, "url": r.get("url"),
        "thumb": r.get("thumbnail") or r.get("url"), "width": r.get("width") or 0,
        "height": r.get("height") or 0, "mime": "image/" + (r.get("filetype") or "jpeg"),
        "author": r.get("creator") or "",
        "licence": f"CC {r.get('license', '').upper()} {r.get('license_version', '')}".strip(),
        "source": r.get("foreign_landing_url") or r.get("url"),
    } for r in data.get("results", [])]


def _stems(text: str) -> set:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {re.sub(r"(ing|es|s)$", "", w) or w for w in words}


def _score(candidate: dict, aspect: float, rank: int, count: int, query: str
           ) -> Optional[Tuple[bool, float]]:
    """``(names_the_subject, score)`` or None when the result is unusable.

    Commons matches descriptions loosely, so a title that actually names what
    was searched for ("Old man walks in the park" for "walking park elderly")
    matters most, then the search rank, the shape and the size.
    """
    w, h = candidate["width"], candidate["height"]
    if not w or not h or "jpeg" not in candidate["mime"] and "png" not in candidate["mime"]:
        return None
    title = candidate["title"]
    if (_UNSUITABLE.search(title) or _SCAN_TITLE.search(title) or min(w, h) < 800
            or candidate["author"].strip().lower() in _SCAN_AUTHORS):
        return None
    if _ANIMALS.search(title) and not _ANIMALS.search(query):
        return None
    if _CONTEXT.search(title) and not _CONTEXT.search(query):
        return None
    words = _stems(title)
    wanted = _stems(query)
    overlap = len(wanted & words)
    people = 0.25 if words & {re.sub(r"(ing|es|s)$", "", p) or p for p in _PEOPLE} else 0.0
    relevance = 1.0 - rank / max(count, 1)
    shape = 1.0 - min(abs(np.log((w / h) / aspect)), 1.0)  # prefer the video's shape
    size = min(w * h / (1920 * 1080), 1.0)
    score = 0.35 * min(overlap, 3) + people + 0.6 * relevance + 0.3 * shape + 0.15 * size
    # One shared word is often a coincidence; two or more mean the photo is about it.
    return overlap >= min(2, len(wanted)), score


def search_candidates(query: str, aspect: float, skip: Tuple[str, ...] = (), limit: int = 8
                      ) -> Tuple[List[dict], List[str]]:
    """Usable free-licensed photos for ``query``, best first, plus any errors.

    ``query`` may hold alternatives separated by ``|`` ("hand washing | washing
    hands soap"), tried in order. Photos whose title names the subject come
    first; the others are only a fallback.
    """
    alternatives = [q.strip() for q in query.split("|") if q.strip()]
    named: List[Tuple[float, dict]] = []
    fallback: List[Tuple[float, dict]] = []
    errors: List[str] = []
    for q in alternatives:
        for search in (search_commons, search_openverse):
            try:
                found = search(q)
            except RateLimited:
                raise
            except Exception as exc:  # network trouble: try the next source
                errors.append(f"{search.__name__}: {exc}")
                continue
            found = [c for c in found if c["url"] and c["source"] not in skip]
            hits = 0
            for i, c in enumerate(found):
                result = _score(c, aspect, i, len(found), q)
                if result is None:
                    continue
                is_named, score = result
                (named if is_named else fallback).append((score, c))
                hits += is_named
            if hits:
                break
        if len(named) >= limit:
            break
    ranked, seen = [], set()
    for _, c in sorted(named, key=lambda x: -x[0]) + sorted(fallback, key=lambda x: -x[0]):
        if c["source"] not in seen:
            seen.add(c["source"])
            ranked.append(c)
    return ranked[:limit], errors


def download_candidate(candidate: dict, cache_dir: str) -> Tuple[str, ImageCredit]:
    """Download (once) a search result and return its local path and credit."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(candidate["url"].encode()).hexdigest()[:12]
    meta_path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            credit = ImageCredit(**json.load(fh))
        if os.path.exists(credit.file):
            return credit.file, credit
    url = candidate["url"]
    if not url.startswith("https://"):
        raise InputError(f"Refusing to download a non-HTTPS image: {url}")
    data = _get(url, timeout=60)
    ext = ".png" if "png" in candidate.get("mime", "") else ".jpg"
    path = os.path.join(cache_dir, f"{key}{ext}")
    with open(path, "wb") as fh:
        fh.write(data)
    credit = ImageCredit(candidate.get("title", ""), candidate.get("author", ""),
                         candidate.get("licence", ""), candidate.get("source", url), path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(credit), fh, ensure_ascii=False, indent=1)
    return path, credit


def find_image(query: str, cache_dir: str, aspect: float, skip: Tuple[str, ...] = ()
               ) -> Tuple[str, ImageCredit]:
    """Search, download (once) and return the best photo for ``query`` plus its credit."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(f"{query}|{aspect:.2f}|{'|'.join(skip)}".encode()).hexdigest()[:12]
    choice_path = os.path.join(cache_dir, f"query-{key}.json")
    if os.path.exists(choice_path):  # same query as before: same picture
        with open(choice_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        try:
            return download_candidate(cached, cache_dir)
        except Exception:
            pass
    ranked, errors = search_candidates(query, aspect, skip, limit=5)
    if not ranked:
        detail = f" ({'; '.join(errors[-2:])})" if errors else ""
        raise InputError(f"No suitable free-licensed image found for '{query}'{detail}. "
                         "Try other English keywords or give the scene an image file.")
    for cand in ranked:
        try:
            result = download_candidate(cand, cache_dir)
        except Exception as exc:
            errors.append(str(exc))
            continue
        with open(choice_path, "w", encoding="utf-8") as fh:
            json.dump(cand, fh, ensure_ascii=False)
        return result
    raise RenderError(f"Could not download an image for '{query}': {'; '.join(errors[-3:])}")


# --------------------------------------------------------------------------- audio

def decode_audio(path: str, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Any audio file FFmpeg can read, as mono float32 samples in [-1, 1]."""
    if not os.path.isfile(path):
        raise InputError(f"Audio file not found: {path}")
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
           "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["unknown error"]
        raise InputError(f"Could not read the audio file {os.path.basename(path)}: {msg[0]}")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def write_wav(path: str, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def find_pauses(samples: np.ndarray, rate: int = SAMPLE_RATE, min_pause: float = 0.25
                ) -> List[Tuple[float, float]]:
    """Silent stretches (start, end) in seconds - natural places to change scene."""
    win = int(0.02 * rate)
    n = len(samples) // win
    if n == 0:
        return []
    rms = np.sqrt((samples[:n * win].reshape(n, win) ** 2).mean(axis=1))
    loud = rms[rms > 1e-4]
    if loud.size == 0:
        return []
    quiet = rms < max(0.12 * float(np.median(loud)), 1e-3)
    pauses = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        if (b - a) * 0.02 >= min_pause:
            pauses.append((a * 0.02, b * 0.02))
    return pauses


def speak(text: str, voice: str, out_path: str) -> str:
    """Preview narration with the macOS speech synthesiser (offline)."""
    say = shutil.which("say")
    if not say:
        raise RenderError("Preview narration needs the macOS 'say' command; record the narration "
                          "instead.")
    proc = subprocess.run([say, "-v", voice, "-o", out_path, text], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"Speech synthesis failed: {proc.stderr.strip() or 'unknown error'}")
    return out_path
