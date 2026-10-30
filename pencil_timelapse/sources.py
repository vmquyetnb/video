"""Photos and video clips of places from free libraries other than Wikimedia.

* Openverse: Creative Commons photos, mostly from Flickr; no key needed.
  Only licences that allow commercial use and changes (a drawing is a
  change) are asked for, and Wikimedia is left out.
* Pexels and Pixabay: photos and video clips under their own free licences;
  each needs a free API key (``PEXELS_API_KEY`` / ``PIXABAY_API_KEY`` in the
  environment, or given with the request).

Every result carries its author, licence and page so the video can credit it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .errors import InputError, RenderError
from .media import USER_AGENT, RateLimited

# Files are only ever downloaded from these hosts (and their subdomains).
ALLOWED_HOSTS = ("staticflickr.com", "pexels.com", "pixabay.com", "vimeo.com", "vimeocdn.com")
PHOTO_TYPES = (".jpg", ".jpeg", ".png", ".webp")
MAX_VIDEO_MB = 300
# Openverse also has maps, posters and drawings; a travel video wants photos.
_NOT_A_PHOTO = re.compile(r"\b(logo|map|diagram|chart|flag|poster|screenshot|document|infographic|"
                          r"illustration|drawing|sketch|painting|cartoon|clipart|vector|icon|"
                          r"sticker|postcard|ticket|brochure|text|sign)s?\b", re.I)


class SourceError(RenderError):
    """A photo library refused the request (bad key, quota, outage)."""


@dataclass
class Media:
    kind: str  # "photo" or "video"
    url: str  # the file
    thumb: str
    width: int
    height: int
    title: str
    author: str
    licence: str
    source: str  # the page to credit
    provider: str  # "Flickr", "Pexels", "Pixabay"...
    duration: float = 0.0  # seconds, for a video

    def credit(self) -> str:
        """One line for the credits: what, who, licence, where."""
        what = f"{self.title} - " if self.title else ""
        return f"{what}{self.author or 'unknown author'} ({self.licence}, {self.provider})"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Media":
        """A search result sent back by the browser; only files on the known
        libraries' servers are accepted."""
        try:
            media = cls(kind=str(data["kind"]), url=str(data["url"]), thumb=str(data.get("thumb") or ""),
                        width=int(data.get("width") or 0), height=int(data.get("height") or 0),
                        title=str(data.get("title") or "")[:200], author=str(data.get("author") or "")[:120],
                        licence=str(data.get("licence") or "")[:60], source=str(data.get("source") or "")[:500],
                        provider=str(data.get("provider") or "")[:40],
                        duration=float(data.get("duration") or 0.0))
        except (KeyError, TypeError, ValueError):
            raise InputError("Invalid photo or video choice. Search again.") from None
        if media.kind not in ("photo", "video") or not allowed(media.url):
            raise InputError("That photo or video is not from a supported library. Search again.")
        return media


def allowed(url: str) -> bool:
    parts = urllib.parse.urlparse(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def keys_from(given: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """API keys: those given (from the page), else the environment's."""
    given = given or {}
    out = {}
    for name, env in (("pexels", "PEXELS_API_KEY"), ("pixabay", "PIXABAY_API_KEY")):
        value = (given.get(name) or os.environ.get(env) or "").strip()
        if value:
            out[name] = value
    return out


# --------------------------------------------------------------------------- HTTP

def _get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0,
         what: str = "The photo library") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            try:
                detail = str(json.loads(exc.read()).get("detail") or "")[:160]
            except (ValueError, AttributeError):
                detail = ""
            raise SourceError(f"{what} refused the request ({detail})" if detail else
                              f"{what} did not accept the API key.") from None
        if exc.code == 429:
            raise RateLimited(f"{what} is limiting requests right now. Wait a minute and try "
                              "again.") from None
        raise SourceError(f"{what} answered with an error ({exc.code}).") from None


def _json(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0,
          what: str = "The photo library") -> dict:
    return json.loads(_get(url, headers, timeout, what))


# --------------------------------------------------------------------------- libraries

def search_openverse(query: str, limit: int = 20) -> List[Media]:
    """CC photos (Flickr and others, not Wikimedia) that may be changed and used
    commercially; at most two by the same photographer, for variety. (Without
    an account Openverse sends at most 20 results at a time.)"""
    params = {"q": query, "license_type": "commercial,modification", "excluded_source": "wikimedia",
              "page_size": str(min(limit, 20))}
    url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(params)
    try:
        data = _json(url, timeout=75, what="Openverse")
    except SourceError as exc:  # Openverse sometimes fails once and then answers
        if "error (5" not in str(exc):
            raise
        data = _json(url, timeout=75, what="Openverse")
    out, per_author = [], {}
    for r in data.get("results", []):
        author = str(r.get("creator") or "")
        per_author[author] = per_author.get(author, 0) + 1
        if per_author[author] > 2:
            continue
        licence = str(r.get("license") or "").lower()
        name = {"cc0": "CC0", "pdm": "Public Domain"}.get(licence,
                                                          f"CC {licence.upper()} {r.get('license_version') or ''}")
        title = str(r.get("title") or "")
        url = str(r.get("url") or "")
        if not allowed(url) or _NOT_A_PHOTO.search(title):
            continue
        out.append(Media("photo", url, str(r.get("thumbnail") or url), int(r.get("width") or 0),
                         int(r.get("height") or 0), title, author, name.strip(),
                         str(r.get("foreign_landing_url") or url), str(r.get("source") or "").title()))
    return out


def search_pexels(query: str, key: str, limit: int = 15) -> List[Media]:
    params = {"query": query, "per_page": str(limit), "orientation": "landscape"}
    data = _json("https://api.pexels.com/v1/search?" + urllib.parse.urlencode(params),
                 {"Authorization": key}, what="Pexels")
    return [Media("photo", p["src"]["large2x"], p["src"].get("medium") or p["src"]["large2x"],
                  int(p.get("width") or 0), int(p.get("height") or 0), str(p.get("alt") or query),
                  str(p.get("photographer") or ""), "Pexels License", str(p.get("url") or ""), "Pexels")
            for p in data.get("photos", []) if p.get("src", {}).get("large2x")]


def _best_file(files: List[dict], width_key: str = "width") -> Optional[dict]:
    """The video file closest to Full HD without being larger."""
    usable = [f for f in files if f.get(width_key) and f.get("url" if "url" in f else "link")]
    if not usable:
        return None
    return min(usable, key=lambda f: (f[width_key] > 1920, abs(f[width_key] - 1920)))


def _title_from_page(url: str, fallback: str) -> str:
    """Pexels pages are named after the clip: ".../video/aerial-view-of-paris-1234/"."""
    slug = urllib.parse.urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    words = re.sub(r"-?\d+$", "", slug).replace("-", " ").strip()
    return words.capitalize() if words else fallback


def search_pexels_videos(query: str, key: str, limit: int = 10) -> List[Media]:
    params = {"query": query, "per_page": str(limit), "orientation": "landscape"}
    data = _json("https://api.pexels.com/videos/search?" + urllib.parse.urlencode(params),
                 {"Authorization": key}, what="Pexels")
    out = []
    for v in data.get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"]
        best = _best_file(files)
        if best is None:
            continue
        out.append(Media("video", best["link"], str(v.get("image") or ""), int(best["width"]),
                         int(best.get("height") or 0), _title_from_page(str(v.get("url") or ""), query),
                         str((v.get("user") or {}).get("name") or ""), "Pexels License",
                         str(v.get("url") or ""), "Pexels", float(v.get("duration") or 0)))
    return out


def search_pixabay(query: str, key: str, limit: int = 15) -> List[Media]:
    params = {"key": key, "q": query[:100], "image_type": "photo", "orientation": "horizontal",
              "per_page": str(max(3, limit)), "safesearch": "true"}
    data = _json("https://pixabay.com/api/?" + urllib.parse.urlencode(params), what="Pixabay")
    return [Media("photo", h["largeImageURL"], str(h.get("webformatURL") or h["largeImageURL"]),
                  int(h.get("imageWidth") or 0), int(h.get("imageHeight") or 0),
                  str(h.get("tags") or query), str(h.get("user") or ""), "Pixabay Content License",
                  str(h.get("pageURL") or ""), "Pixabay")
            for h in data.get("hits", []) if h.get("largeImageURL")]


def search_pixabay_videos(query: str, key: str, limit: int = 10) -> List[Media]:
    params = {"key": key, "q": query[:100], "per_page": str(max(3, limit)), "safesearch": "true"}
    data = _json("https://pixabay.com/api/videos/?" + urllib.parse.urlencode(params), what="Pixabay")
    out = []
    for h in data.get("hits", []):
        sizes = [v for v in (h.get("videos") or {}).values() if isinstance(v, dict)]
        best = _best_file(sizes)
        if best is None:
            continue
        thumb = best.get("thumbnail") or (f"https://i.vimeocdn.com/video/{h['picture_id']}_640x360.jpg"
                                          if h.get("picture_id") else "")
        out.append(Media("video", best["url"], str(thumb), int(best["width"]),
                         int(best.get("height") or 0), str(h.get("tags") or query),
                         str(h.get("user") or ""), "Pixabay Content License",
                         str(h.get("pageURL") or ""), "Pixabay", float(h.get("duration") or 0)))
    return out


def search_media(query: str, keys: Optional[Dict[str, str]] = None, videos: bool = True,
                 limit: int = 12) -> Tuple[List[Media], List[str]]:
    """Photos and (with a Pexels or Pixabay key) video clips for ``query``,
    best first, and the problems met on the way (one per library)."""
    keys = keys or {}
    searches: List[Tuple[str, Callable[[], List[Media]]]] = [
        ("Openverse", lambda: search_openverse(query))]
    if keys.get("pexels"):
        searches.append(("Pexels", lambda: search_pexels(query, keys["pexels"])))
        if videos:
            searches.append(("Pexels videos", lambda: search_pexels_videos(query, keys["pexels"])))
    if keys.get("pixabay"):
        searches.append(("Pixabay", lambda: search_pixabay(query, keys["pixabay"])))
        if videos:
            searches.append(("Pixabay videos", lambda: search_pixabay_videos(query, keys["pixabay"])))
    found: List[List[Media]] = []
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=len(searches)) as pool:
        futures = [(name, pool.submit(fn)) for name, fn in searches]
        for name, future in futures:
            try:
                found.append(future.result())
            except Exception as exc:  # one library failing must not hide the others
                found.append([])
                errors.append(f"{name}: {exc}")
    photos = [[m for m in group if m.kind == "photo" and min(m.width, m.height) >= 600]
              for group in found]
    clips = [[m for m in group if m.kind == "video" and min(m.width, m.height) >= 540
              and m.duration >= 4] for group in found]

    def interleave(groups: List[List[Media]]) -> List[Media]:  # the best of each library first
        out = []
        for row in range(max((len(g) for g in groups), default=0)):
            out += [g[row] for g in groups if row < len(g)]
        return out[:limit]

    return interleave(clips) + interleave(photos), errors


# --------------------------------------------------------------------------- download

def download(media: Media, cache_dir: str, timeout: float = 60.0) -> str:
    """The file of ``media`` on disk (downloaded once into ``cache_dir``)."""
    if not allowed(media.url):
        raise InputError("That photo or video is not from a supported library.")
    os.makedirs(cache_dir, exist_ok=True)
    ext = os.path.splitext(urllib.parse.urlparse(media.url).path)[1].lower()
    if media.kind == "video":
        ext = ".mp4"
    elif ext not in PHOTO_TYPES:
        ext = ".jpg"
    path = os.path.join(cache_dir, hashlib.sha1(media.url.encode()).hexdigest()[:16] + ext)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    req = urllib.request.Request(media.url, headers={"User-Agent": USER_AGENT})
    limit = MAX_VIDEO_MB * 1024 * 1024
    part = path + ".part"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:  # noqa: S310
            deadline, size = time.monotonic() + 6 * timeout, 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise InputError(f"The video is larger than {MAX_VIDEO_MB} MB.")
                if time.monotonic() > deadline:
                    raise RenderError(f"Downloading {media.url} took too long.")
                fh.write(chunk)
        os.replace(part, path)
    except urllib.error.HTTPError as exc:
        raise RenderError(f"Could not download {media.title or media.url} ({exc.code}). Pick "
                          "another photo or video.") from None
    finally:
        if os.path.exists(part):
            os.remove(part)
    return path
