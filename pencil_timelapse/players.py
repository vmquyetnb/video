"""Pencil portraits of a country's famous footballers, drawn into one frame.

Players are found on Wikidata: everyone who played for the country's men's
national team, most famous first (measured by how many Wikipedia editions have
an article about them), with a portrait photo on Wikimedia Commons. Each photo
is cropped around the face and turned into a graphite drawing; the video then
writes the country's name and draws the players one after another into a
shared composition, ending on the finished group picture.

    python -m pencil_timelapse.players Portugal -o portugal.mp4 --count 6
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
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import analysis as A
from .config import RenderConfig
from .encoder import FFmpegWriter, find_ffmpeg
from .errors import InputError, PencilError, RenderError
from .imaging import load_image, luminance, to_uint8
from .media import (ImageCredit, RateLimited, _clean, _get, decode_audio, download_candidate,
                    search_commons, write_wav)
from .pipeline import plan_animation
from .renderer import FrameRenderer
from .showtime import (ShowOptions, intro_clip, outro_clip, sound_track, spotlight_clip,
                       teaser_clip)
from .sketch import PAPER, graphite_drawing

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]
RelBox = Tuple[float, float, float, float]  # x, y, w, h as fractions of the image

# Football nations offered in the interface (any other country can be typed in).
COUNTRIES = [
    "Portugal", "Brazil", "Argentina", "France", "Germany", "Spain", "England", "Italy",
    "Netherlands", "Belgium", "Croatia", "Uruguay", "Colombia", "Mexico", "United States",
    "Poland", "Denmark", "Sweden", "Norway", "Switzerland", "Morocco", "Senegal", "Egypt",
    "Nigeria", "Japan", "South Korea", "Australia", "Vietnam", "Thailand", "Indonesia",
]

HAND_FONTS = ("/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
              "/System/Library/Fonts/Noteworthy.ttc")
PLAIN_FONTS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf")
INK = (58, 56, 52)


@dataclass
class Player:
    name: str
    qid: str
    image_file: str  # Commons file name
    fame: int  # Wikipedia editions with an article
    thumb: str = ""
    photo_url: str = ""
    author: str = ""
    licence: str = ""
    source: str = ""
    local_file: str = ""  # the user's own photo; never taken from a web request

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "local_file"}


# --------------------------------------------------------------------------- Wikidata

def _json(url: str, timeout: float = 60.0) -> dict:
    return json.loads(_get(url, timeout=timeout))


def find_team(country: str) -> Tuple[str, str]:
    """Wikidata id and label of ``country``'s senior men's national team."""
    params = {"action": "wbsearchentities", "search": f"{country} national football team",
              "language": "en", "type": "item", "limit": "10", "format": "json"}
    data = _json("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode(params))
    for item in data.get("search", []):
        desc = (item.get("description") or "").lower()
        label = (item.get("label") or "").lower()
        if ("national" in desc and ("football" in desc or "soccer" in desc)
                and not re.search(r"women|under-|u-?\d\d|youth|result|history|list|olympic|futsal|beach",
                                  desc + " " + label)):
            return item["id"], item.get("label", "")
    raise InputError(f"No men's national football team found for '{country}'. "
                     "Check the country name (in English).")


def find_players(country: str, limit: int = 15) -> List[Player]:
    """The most famous players of ``country``'s national team who have a portrait photo."""
    team, _ = find_team(country)
    query = f"""
    SELECT ?p ?pLabel (SAMPLE(?img) AS ?image) (MAX(?sl) AS ?fame) WHERE {{
      ?p p:P54/ps:P54 wd:{team}; wdt:P106 wd:Q937857; wdt:P18 ?img; wikibase:sitelinks ?sl.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} GROUP BY ?p ?pLabel ORDER BY DESC(?fame) LIMIT {int(limit)}"""
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query,
                                                                          "format": "json"})
    rows = _json(url, timeout=90)["results"]["bindings"]
    players = []
    for row in rows:
        file_name = urllib.parse.unquote(row["image"]["value"].rsplit("/", 1)[1])
        players.append(Player(row["pLabel"]["value"], row["p"]["value"].rsplit("/", 1)[1],
                              file_name, int(row["fame"]["value"])))
    if not players:
        raise InputError(f"No players with a photo were found for '{country}'.")
    _add_photo_info(players)
    return players


def _add_photo_info(players: List[Player]) -> None:
    """Thumbnail, full-size URL, author and licence for each player's photo (one request)."""
    titles = "|".join(f"File:{p.image_file}" for p in players)
    params = {"action": "query", "format": "json", "titles": titles, "prop": "imageinfo",
              "iiprop": "url|size|extmetadata", "iiurlwidth": "1280"}
    data = _json("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params))
    pages = data.get("query", {}).get("pages", {})
    normalized = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
    by_title = {}
    for page in pages.values():
        title = normalized.get(page.get("title", ""), page.get("title", ""))
        by_title[title.replace("_", " ")] = page
    for p in players:
        page = by_title.get(f"File:{p.image_file}".replace("_", " "))
        if not page or not page.get("imageinfo"):
            continue
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        p.photo_url = info.get("thumburl") or info.get("url", "")
        p.thumb = re.sub(r"/\d+px-", "/240px-", p.photo_url)
        p.author = _clean(meta.get("Artist", {}).get("value", ""))
        p.licence = _clean(meta.get("LicenseShortName", {}).get("value", "")) or "see source"
        p.source = info.get("descriptionurl", "")


# --------------------------------------------------------------------------- pictures

def _plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))


# Titles that suggest the player is not the clear subject of the photo.
_CROWDED = re.compile(r"\b(and|with|vs|v|team|squad|line-?up|celebrat\w*|fans?|group|trophy|"
                      r"statue|mural|signature|autograph|wax)\b", re.I)


def candidate_photos(player: Player, limit: int = 6) -> List[dict]:
    """Photos that could show ``player``: the Wikidata portrait plus Commons
    search results whose title names them, most promising first."""
    words = _plain(player.name).split()
    surname = words[-1] if words else ""
    out: List[dict] = []
    if player.photo_url:
        out.append({"url": player.photo_url, "title": player.image_file, "author": player.author,
                    "licence": player.licence, "source": player.source, "mime": "image/jpeg",
                    "width": 0, "height": 0, "bonus": 0.25})  # curated: preferred
    try:
        found = search_commons(f'"{player.name}" filetype:bitmap')
    except Exception:  # the Wikidata portrait alone still works
        found = []
    for c in found:
        title = _plain(c["title"])
        if surname not in title or _CROWDED.search(c["title"]) or "jpeg" not in c["mime"]:
            continue
        if min(c["width"], c["height"]) < 900 or any(o["source"] == c["source"] for o in out):
            continue
        bonus = 0.1 if "crop" in title or "portrait" in title else 0.0
        out.append(dict(c, bonus=bonus))
    return out[:limit]


def _small(url: str, width: int = 0) -> str:
    """A 500 px wide thumbnail URL for quick scoring (Commons thumbnail naming).

    Originals wider than that are asked for as a thumbnail too; a smaller or
    unknown-size original is fetched as it is.
    """
    if re.search(r"/\d+px-", url):
        return re.sub(r"/\d+px-", "/500px-", url)
    m = re.match(r"(https://upload\.wikimedia\.org/wikipedia/commons/)([0-9a-f]/[0-9a-f]{2}/)"
                 r"([^/?#]+)", url)
    if m and width > 500:
        return f"{m.group(1)}thumb/{m.group(2)}{m.group(3)}/500px-{m.group(3)}"
    return url


def _detect(gray: np.ndarray, cascade: str, neighbours: int) -> List[Tuple[int, int, int, int]]:
    clf = A._cascade(cascade)
    if clf is None:
        return []
    m = int(0.07 * min(gray.shape))
    return [tuple(int(v) for v in f) for f in
            clf.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=neighbours, minSize=(m, m))]


def _faces(gray: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], bool]:
    """Faces (x, y, w, h), largest first, and whether they are frontal.

    Turned or tilted heads (three-quarter and profile views, common in match
    photos) are found with the profile cascade, on the image and its mirror,
    and with a less strict frontal search; they do not count as frontal.
    """
    found = _detect(gray, "haarcascade_frontalface_default.xml", 6)
    frontal = bool(found)
    if not found:
        w = gray.shape[1]
        found = _detect(gray, "haarcascade_frontalface_default.xml", 3) + \
            _detect(gray, "haarcascade_profileface.xml", 5) + [
                (w - x - fw, y, fw, fh)
                for x, y, fw, fh in _detect(np.ascontiguousarray(gray[:, ::-1]),
                                            "haarcascade_profileface.xml", 5)]
    return sorted(found, key=lambda f: -f[2] * f[3]), frontal


def _face_in(rgb: np.ndarray, long_side: int = 700
             ) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    """The main face in full-image coordinates (detected on a reduced copy)
    and whether it is a clear frontal face."""
    h, w = rgb.shape[:2]
    s = min(1.0, long_side / max(h, w))
    small = cv2.resize(rgb, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    faces, frontal = _faces(cv2.equalizeHist(cv2.convertScaleAbs(luminance(small), alpha=255.0)))
    if not faces:
        return None, False
    return tuple(int(v / s) for v in faces[0]), frontal


def portrait_score(rgb: np.ndarray, full_width: int = 0
                   ) -> Tuple[float, Optional[RelBox]]:
    """How well a photo works as a drawn portrait (higher is better, < 0 unusable),
    and the clear frontal face it found as fractions of the image (x, y, w, h),
    or None for a turned head, whose detected box is too unreliable to crop to.

    Rewards one large, sharp, frontal face (both eyes found) with room for the
    shoulders, balanced exposure and a calm background; penalises group shots.
    """
    gray = cv2.equalizeHist(cv2.convertScaleAbs(luminance(rgb), alpha=255.0))
    faces, is_frontal = _faces(gray)
    if not faces:
        return -1.0, None
    h, w = gray.shape
    fx, fy, fw, fh = faces[0]
    where = (fx / w, fy / h, fw / w, fh / h) if is_frontal else None
    if min(fx, fy, w - fx - fw) < 0.15 * fw:
        return -0.5, where  # the face touches the edge: no room for a head-and-shoulders crop
    others = sum(1 for f in faces if f[2] * f[3] > 0.25 * fw * fh) - 1
    scale = (full_width or w) / w
    detail = min(fh * scale / 260.0, 1.0)  # enough pixels on the face to draw it well
    face = luminance(rgb)[fy:fy + fh, fx:fx + fw]
    sharp = min(float(cv2.Laplacian(face, cv2.CV_32F).var()) / 0.004, 1.0)
    eyes_clf = A._cascade("haarcascade_eye.xml")
    eyes = len(eyes_clf.detectMultiScale(gray[fy:fy + int(fh * 0.6), fx:fx + fw], 1.08, 5)) \
        if eyes_clf is not None and is_frontal else 0
    frontal = 1.0 if eyes >= 2 else (0.5 if eyes == 1 else (0.2 if is_frontal else 0.35))
    room = min((h - fy - fh) / (0.8 * fh), 1.0)  # space below the chin for neck and shoulders
    mean = float(face.mean())
    exposure = 1.0 - min(abs(mean - 0.5) / 0.35, 1.0)
    # Calm background around the head and shoulders: few edges, and no large
    # blurred shapes either (a team-mate's out-of-focus head, crowd colours).
    x0, x1 = max(0, fx - fw), min(w, fx + 2 * fw)
    y0, y1 = max(0, fy - fh // 2), min(h, fy + 2 * fh)
    background = 1.0 - _bust(h, w, (fx, fy, fw, fh))[y0:y1, x0:x1] > 0.9
    edges = cv2.Canny(gray[y0:y1, x0:x1], 80, 160) > 0
    busy = float(edges[background].mean()) if background.any() else 0.0
    soft = A.blur(luminance(rgb), 0.04 * fw + 1.0)[y0:y1, x0:x1]
    spread = float(soft[background].std()) if background.any() else 0.0
    calm = 0.5 * (1.0 - min(busy / 0.12, 1.0)) + 0.5 * (1.0 - min(spread / 0.25, 1.0))
    return (0.25 * detail + 0.15 * sharp + 0.25 * frontal + 0.10 * exposure + 0.15 * calm
            + 0.10 * room - 0.35 * max(others, 0)), where


def choose_portrait(player: Player, cache_dir: str
                    ) -> Tuple[str, ImageCredit, Optional[RelBox]]:
    """Download the most suitable photo of ``player`` (scored on small thumbnails)
    and return it with its credit and the face found in it (or None)."""
    scored = []
    for cand in candidate_photos(player):
        try:
            data = _get(_small(cand["url"], cand.get("width") or 0), timeout=30)
            thumb = load_image(data, cand["title"])
        except Exception:
            continue
        score, face = portrait_score(thumb, cand.get("width") or 0)
        scored.append((score + cand.get("bonus", 0.0), len(scored), cand, face))
    for _, _, cand, face in sorted(scored, key=lambda s: (-s[0], s[1])):
        try:
            path, credit = download_candidate(dict(cand, title=player.name), cache_dir)
        except RateLimited:
            raise
        except Exception as exc:  # try the next best photo
            log.warning("Could not download %s: %s", cand["url"], exc)
            continue
        return path, credit, face
    raise InputError(f"No usable photo was found for {player.name}.")


# "whole": the entire photo is drawn, nothing cut or covered (default);
# "shade": head and shoulders, the background behind the player shaded in pencil;
# "photo": head and shoulders with the photo's own background.
BACKGROUNDS = ("whole", "shade", "photo")


def prepare_portrait(rgb: np.ndarray, aspect: float, face: Optional[RelBox] = None,
                     background: str = "whole", seed: int = 0) -> np.ndarray:
    """The photo framed for a portrait cell (width / height = ``aspect``);
    see :func:`portrait_and_silhouette`."""
    return portrait_and_silhouette(rgb, aspect, face, background, seed)[0]


def portrait_and_silhouette(rgb: np.ndarray, aspect: float, face: Optional[RelBox] = None,
                            background: str = "whole", seed: int = 0
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """The photo framed for a portrait cell (width / height = ``aspect``), and
    the player's silhouette in it (0..1, for the "who's next?" teaser).

    With ``background="whole"`` the entire photo is kept as it is, fitted into
    the cell; the strips a different shape leaves are filled with a soft
    continuation of the photo's edges. Otherwise the photo is framed head and
    shoulders (the whole head, with hair and ears and room around it, at the
    same scale for every player whenever the photo allows), and with
    ``background="shade"`` everything behind the player is replaced by an even
    pencil tone. ``face`` is the face box as fractions of the image (found
    when the photo was chosen); it is searched for if omitted.
    """
    if background not in BACKGROUNDS:
        raise InputError(f"Unknown background '{background}'.")
    h, w = rgb.shape[:2]
    if face is not None:
        box, clear = (int(face[0] * w), int(face[1] * h), int(face[2] * w), int(face[3] * h)), True
    else:
        box, clear = _face_in(rgb)
    # A turned head's box is unreliable (it may sit on the chin): the whole photo is kept.
    crop, inside, fbox = _frame(rgb, aspect, box if clear and background != "whole" else None)
    if background == "whole" and box is not None and clear:
        ys, xs = np.nonzero(inside)  # the face, moved with the photo into the cell
        fbox = (box[0] + xs.min(), box[1] + ys.min(), box[2], box[3])
    subject = None
    if fbox is not None:
        ys, xs = np.nonzero(inside)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        part = _subject_mask(crop[y0:y1, x0:x1], (fbox[0] - x0, fbox[1] - y0, fbox[2], fbox[3]))
        if part is not None:
            subject = np.zeros(inside.shape, np.float32)
            subject[y0:y1, x0:x1] = part
    ch, cw = inside.shape
    if subject is not None:
        silhouette = subject
    else:  # a generic head and shoulders where the player stands
        guess = fbox or (int(0.33 * cw), int(0.2 * ch), int(0.34 * cw), int(0.27 * ch))
        silhouette = _bust(ch, cw, guess)
    if background == "whole":
        return _continue_edges(crop, inside), silhouette
    image = _shade_background(crop, inside, subject if background == "shade" else None, seed)
    return image, silhouette


def _continue_edges(crop: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Fill the parts of the cell outside the photo with a soft, blurred
    continuation of the photo's edges; the photo itself is left untouched."""
    if float(inside.min()) == 1.0:
        return crop
    h, w = inside.shape
    ys, xs = np.nonzero(inside > 0.5)
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    photo = crop[top:bottom, left:right]
    extended = cv2.copyMakeBorder(photo, top, h - bottom, left, w - right, cv2.BORDER_REPLICATE)
    soft = cv2.GaussianBlur(extended, (0, 0), 0.03 * max(h, w))
    # Fading towards the photo's average tone further away, like loose shading.
    away = cv2.distanceTransform((inside < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    far = A.smoothstep(away / (0.15 * max(h, w)))[..., None]
    fill = soft * (1.0 - 0.6 * far) + photo.mean(axis=(0, 1)) * 0.6 * far
    keep = inside[..., None]
    return (crop * keep + fill * (1.0 - keep)).astype(np.float32)


def _frame(rgb: np.ndarray, aspect: float, face: Optional[Tuple[int, int, int, int]]
           ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """The frame (crop, mask of where it shows the photo, face box inside it).

    The frame may reach beyond the photo; those parts are left for shading.
    """
    h, w = rgb.shape[:2]
    if face is None:  # no clear face: the whole photo, nothing cut
        ch = max(h, int(round(w / aspect)))
        cw = int(round(ch * aspect))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
    else:
        fx, fy, fw, fh = face
        # Head and shoulders, the head (hair to chin) about half the height; at least
        # the whole head with a margin, and no more than the photo comfortably fills.
        ch = int(round(max(2.8 * fh, 2.4 * fw / aspect, min(3.9 * fh, 1.6 * h))))
        cw = int(round(ch * aspect))
        hair_top = fy - 0.75 * fh
        y0 = int(round(hair_top - 0.1 * ch))
        x0 = int(round(fx + fw / 2 - cw / 2))
        if cw <= w:
            x0 = int(np.clip(x0, 0, w - cw))
    crop = np.ones((ch, cw, 3), np.float32)
    inside = np.zeros((ch, cw), np.float32)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + cw), min(h, y0 + ch)
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    inside[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = 1.0
    fbox = None if face is None else (face[0] - x0, face[1] - y0, face[2], face[3])
    return crop, inside, fbox


def _person(h: int, w: int, face: Tuple[int, int, int, int], loose: bool) -> np.ndarray:
    """Head and upper-body shape around a face box (uint8 0/1): a generous
    outer limit, or a slim core that is surely the player."""
    fx, fy, fw, fh = face
    a, b, neck, shoulder, base = (1.05, 1.35, 0.8, 2.3, 2.7) if loose else (0.5, 0.75, 0.3, 1.2, 1.4)
    cx = fx + fw / 2
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(cx), int(fy + 0.32 * fh)), (int(a * fw), int(b * fh)), 0, 0, 360, 1, -1)
    ny = fy + 1.0 * fh
    pts = np.array([(cx - neck * fw, ny), (cx + neck * fw, ny),
                    (cx + shoulder * fw, ny + 0.9 * fh), (cx + base * fw, h),
                    (cx - base * fw, h), (cx - shoulder * fw, ny + 0.9 * fh)], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _oval(h: int, w: int, centre: Tuple[float, float], radii: Tuple[float, float]) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(round(centre[0])), int(round(centre[1]))),
                (max(1, int(round(radii[0]))), max(1, int(round(radii[1])))), 0, 0, 360, 1, -1)
    return mask


def _subject_mask(img: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Soft mask (0..1) of the player in ``img``: head, hair, ears and body,
    with no holes; None when the player cannot be told from the background.

    GrabCut separates the player from the background by colour. The hair is
    then added by sharpness: in sports photos the player is in focus and the
    crowd behind is blurred, so dark hair against a dark crowd, which colour
    alone cannot tell apart, is kept. The head gets a natural rounded outline.
    """
    h, w = img.shape[:2]
    s = min(1.0, 480.0 / max(h, w))
    small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    fx, fy, fw, fh = (int(round(v * s)) for v in face)
    cx = fx + fw / 2
    body = _person(sh, sw, (fx, fy, fw, fh), loose=False)
    outer = _person(sh, sw, (fx, fy, fw, fh), loose=True) == 0
    sure = _oval(sh, sw, (cx, fy + 0.52 * fh), (0.34 * fw, 0.46 * fh))
    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    mask[outer] = cv2.GC_BGD
    mask[body == 1] = cv2.GC_PR_FGD
    mask[sure == 1] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.convertScaleAbs(small[..., ::-1], alpha=255.0), mask, None, bgd, fgd, 6,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    found = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    found = cv2.morphologyEx(found, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # The part joined to the face, and any part lying mostly inside the body
    # shape (a shirt cut off from the head when the neck matches the background).
    count, labels = cv2.connectedComponents(found)
    seed = labels[min(sh - 1, int(fy + 0.52 * fh)), min(sw - 1, int(cx))]
    if seed == 0:
        return None
    sizes = np.bincount(labels.ravel(), minlength=count)
    inside_body = np.bincount(labels[body == 1], minlength=count)
    keep = (inside_body >= 0.6 * sizes) & (sizes >= 0.02 * fw * fh)
    keep[0], keep[seed] = False, True
    subject = keep[labels].astype(np.uint8)

    # The head: what colour found, plus everything as sharp as the player (hair
    # strands) joined to the face, measured against how sharp the background is.
    detail = A.blur(np.abs(cv2.Laplacian(luminance(small), cv2.CV_32F)), 0.03 * fw + 1.0)
    if outer.sum() > 0.05 * outer.size:
        background = float(np.percentile(detail[outer], 75))
        sharp = detail > max(1.8 * background, 0.4 * float(np.median(detail[sure == 1])), 0.01)
        sharp = cv2.morphologyEx(sharp.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    else:  # a close-up with hardly any background to compare with: colour only
        sharp = np.zeros_like(subject)
    chin = min(sh, int(fy + 1.05 * fh))
    zone = _oval(sh, sw, (cx, fy + 0.32 * fh), (0.95 * fw, 1.25 * fh))
    zone[chin:] = 0
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(0.08 * fw)) | 1,) * 2)
    head = cv2.morphologyEx(((subject | sharp) & zone) | sure, cv2.MORPH_CLOSE, disk) & zone
    _, labels = cv2.connectedComponents(head)
    head = (labels == labels[min(sh - 1, int(fy + 0.52 * fh)), min(sw - 1, int(cx))]).astype(np.uint8)
    # A rounded outline: notches, gaps between strands of hair and straight cuts filled in.
    hull = np.zeros_like(head)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(head)), 1)
    near = cv2.dilate(head, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (max(3, int(0.2 * fw)) | 1,) * 2))
    head |= hull & near
    subject[zone == 1] = head[zone == 1]

    k = max(3, int(0.12 * fw)) | 1  # smooth the outline, close gaps in the hair
    subject = cv2.morphologyEx(subject, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Fill holes (eyes, mouth, a dark shirt): only what the border can reach is background.
    flood = np.pad(1 - subject, 1, constant_values=1)
    cv2.floodFill(flood, None, (0, 0), 2)
    subject[flood[1:-1, 1:-1] == 1] = 1
    if float(subject.mean()) < 0.06:  # hardly anything found: the separation failed
        return None
    soft = A.blur(subject.astype(np.float32), 0.006 * max(sh, sw) + 0.6)
    return cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)


def _shade_background(crop: np.ndarray, inside: np.ndarray, subject: Optional[np.ndarray],
                      seed: int) -> np.ndarray:
    """Replace the background (outside ``subject``, or outside the photo when
    there is no subject mask) with an even pencil tone: lighter right around
    the player so the outline reads clearly, deeper towards the edges."""
    h, w = inside.shape
    if subject is None and float(inside.min()) == 1.0:
        return crop  # the photo fills the frame and is drawn as it is
    # Where the photo ends inside the frame it fades out: only a sliver at its top,
    # which may run right along the hair, more at the sides and below the shoulders.
    ys, xs = np.nonzero(inside > 0.5)
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    yy, xx = np.arange(h, dtype=np.float32)[:, None], np.arange(w, dtype=np.float32)[None, :]

    def fade(dist: np.ndarray, reach: float, cut: bool) -> np.ndarray:
        return A.smoothstep(dist / reach) if cut else np.ones_like(dist)

    edge = (fade(yy - top, 0.005 * h, top > 0) * fade(bottom - 1 - yy, 0.07 * h, bottom < h)
            * fade(xx - left, 0.05 * w, left > 0) * fade(right - 1 - xx, 0.05 * w, right < w))
    edge = edge * inside
    keep = edge if subject is None else subject * edge
    near = (keep > 0.5).astype(np.uint8)
    if not near.any():
        near = inside.astype(np.uint8)
    dist = cv2.distanceTransform(1 - near, cv2.DIST_L2, 5) / (0.28 * max(h, w))
    rng = np.random.default_rng(seed + 101)
    wobble = cv2.resize(rng.standard_normal((6, 5)).astype(np.float32), (w, h),
                        interpolation=cv2.INTER_CUBIC)
    tone = 0.80 - 0.22 * A.smoothstep(dist) + 0.025 * wobble
    alpha = keep[..., None]
    return (crop * alpha + np.clip(tone, 0.0, 1.0)[..., None] * (1.0 - alpha)).astype(np.float32)


def _bust(h: int, w: int, face: Tuple[int, int, int, int]) -> np.ndarray:
    """Soft head, neck and shoulders silhouette around a face box, in [0, 1]."""
    fx, fy, fw, fh = face
    cx = fx + fw / 2
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(cx), int(fy + 0.38 * fh)), (int(0.64 * fw), int(0.96 * fh)), 0, 0, 360,
                255, -1)
    neck = fy + 0.95 * fh
    pts = np.array([(cx - 0.4 * fw, neck), (cx + 0.4 * fw, neck),
                    (cx + 1.5 * fw, neck + 0.7 * fh), (cx + 1.8 * fw, h),
                    (cx - 1.8 * fw, h), (cx - 1.5 * fw, neck + 0.7 * fh)], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.08 * fw + 0.5)


def _font(paths, size: int) -> ImageFont.FreeTypeFont:
    for path in paths:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _has_glyphs(font: ImageFont.FreeTypeFont, text: str) -> bool:
    """False if the font would draw a 'missing glyph' box for any character."""
    def render(ch: str) -> bytes:
        img = Image.new("L", (font.size * 2, font.size * 2))
        ImageDraw.Draw(img).text((0, 0), ch, font=font, fill=255)
        return img.tobytes()

    missing = render("\uE000")
    return all(render(ch) != missing for ch in set(text) if not ch.isspace())


def _label_font(text: str, size: int) -> ImageFont.FreeTypeFont:
    hand = _font(HAND_FONTS, size)
    return hand if _has_glyphs(hand, text) else _font(PLAIN_FONTS, int(size * 0.85))


@dataclass
class Layout:
    width: int
    height: int
    cells: List[Tuple[int, int, int, int]]  # portrait boxes (x, y, w, h)
    name_boxes: List[Tuple[int, int, int, int]]
    title_box: Tuple[int, int, int, int]


def make_layout(count: int, width: int, height: int) -> Layout:
    """Grid of 4:5 portraits with a name under each and a title band on top,
    choosing the number of columns that gives the largest portraits."""
    title_h = int(0.13 * height)
    margin = int(0.03 * min(width, height))
    avail_w, avail_h = width - 2 * margin, height - title_h - margin
    best = None
    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        cell_w, cell_h = avail_w / cols, avail_h / rows
        name_h = 0.16 * cell_h
        ph = min(cell_h - name_h - margin * 0.5, (cell_w - margin) / 0.8)
        if ph > 0 and (best is None or ph > best[0]):
            best = (ph, cols, rows, cell_w, cell_h, name_h)
    ph, cols, rows, cell_w, cell_h, name_h = best
    pw = int(ph * 0.8) // 2 * 2
    ph = int(ph) // 2 * 2
    # Portraits stand close together, like one group picture, centred on the page.
    step_x = min(cell_w, pw * 1.18)
    label_w = max(step_x, pw * 1.1)
    cells, names = [], []
    for i in range(count):
        r, c = divmod(i, cols)
        in_row = min(cols, count - r * cols)  # centre a shorter last row
        cx = width / 2 + (c - (in_row - 1) / 2) * step_x
        top = title_h + r * cell_h + (cell_h - ph - name_h) / 2
        cells.append((int(cx - pw / 2), int(top), pw, ph))
        names.append((int(cx - label_w / 2), int(top + ph), int(label_w), int(name_h)))
    return Layout(width, height, cells, names, (0, int(0.02 * height), width, title_h))


def compose(layout: Layout, drawings: List[np.ndarray], names: List[str], title: str,
            upto: int, show_title: bool = True) -> np.ndarray:
    """The shared page with the title and the first ``upto`` portraits and names."""
    W, H = layout.width, layout.height
    rng = np.random.default_rng(3)
    grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.7)
    page = np.clip(PAPER[None, None, :] * (1.0 + 0.008 * grain[..., None]), 0.0, 1.0)
    for i in range(upto):
        x, y, w, h = layout.cells[i]
        page[y:y + h, x:x + w] = np.minimum(page[y:y + h, x:x + w], drawings[i])
    img = Image.fromarray(to_uint8(page))
    draw = ImageDraw.Draw(img)
    if show_title and title:
        tx, ty, tw, th = layout.title_box
        font = _label_font(title, int(th * 0.62))
        while font.getlength(title) > 0.9 * W and font.size > 20:
            font = _label_font(title, int(font.size * 0.9))
        draw.text((tx + (tw - font.getlength(title)) / 2, ty + (th - font.size) / 2), title,
                  font=font, fill=INK)
    for i in range(upto):
        x, y, w, h = layout.name_boxes[i]
        font = _label_font(names[i], int(h * 0.55))
        while font.getlength(names[i]) > 0.95 * w and font.size > 12:
            font = _label_font(names[i], int(font.size * 0.9))
        draw.text((x + (w - font.getlength(names[i])) / 2, y + (h - font.size) / 2), names[i],
                  font=font, fill=INK)
    return np.asarray(img).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- video

@dataclass
class PlayersResult:
    output_path: str
    image_path: str
    credits_path: Optional[str]  # None when every photo was the user's own
    duration: float
    players: List[str]
    elapsed: float
    warnings: List[str] = field(default_factory=list)


def render_players(players: List[Player], title: str, output_path: str,
                   width: int = 1920, height: int = 1080, fps: int = 30,
                   seconds_per_player: float = 10.0, cache_dir: Optional[str] = None,
                   progress: Optional[ProgressFn] = None, crf: int = 20,
                   preset: str = "fast", background: str = "whole",
                   audio: Optional[str] = None, fit_to_audio: bool = True,
                   show: Optional[ShowOptions] = None) -> PlayersResult:
    """Draw ``players`` one after another into one frame and save the video,
    the finished picture (PNG) and the credits of photos taken from Commons.

    A player with a ``local_file`` is drawn from that photo; the others get
    the best portrait found on Wikimedia Commons. ``background`` is "whole"
    (the entire photo drawn, nothing cut), "shade" (head and shoulders, the
    background shaded in pencil) or "photo" (head and shoulders with the
    photo's own background).

    ``audio`` is a sound file (music or narration, any format FFmpeg reads)
    laid under the video. With ``fit_to_audio`` the drawing time per player
    is chosen so that the video lasts as long as the audio (the final picture
    is held longer if the players cannot be drawn slowly enough). ``show``
    switches the intro, the per-player highlights, the call-to-action ending
    and the sound effects on or off (all on by default).
    """
    show = show or ShowOptions()
    report = progress or (lambda f, m: None)
    started = time.perf_counter()
    if not players:
        raise InputError("Choose at least one player.")
    if len(players) > 12:
        raise InputError("Choose at most 12 players.")
    if not 4.0 <= seconds_per_player <= 60.0:
        raise InputError("Drawing time per player must be between 4 and 60 seconds.")
    if background not in BACKGROUNDS:
        raise InputError(f"Background must be one of: {', '.join(BACKGROUNDS)}.")
    RenderConfig(width=width, height=height, fps=fps).validate()
    audio_seconds = 0.0
    if audio:
        audio_seconds = len(decode_audio(audio, rate=8000)) / 8000.0  # also checks the file
        if audio_seconds < 1.0:
            raise InputError("The audio file is shorter than one second.")
    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = cache_dir or os.path.join(out_dir, "player_photos")
    warnings: List[str] = []

    # ---- photos -> drawings
    layout = make_layout(len(players), width, height)
    drawings, silhouettes, credits = [], [], []
    for i, p in enumerate(players):
        if p.local_file:
            report(0.02 + 0.13 * i / len(players), f"Preparing {p.name} ({i + 1}/{len(players)})")
            path, face = p.local_file, None
        else:
            report(0.02 + 0.13 * i / len(players),
                   f"Choosing the best portrait of {p.name} ({i + 1}/{len(players)})")
            path, credit, face = choose_portrait(p, cache_dir)
            credits.append(credit)
        x, y, w, h = layout.cells[i]
        photo, silhouette = portrait_and_silhouette(load_image(path, f"Photo of {p.name}"), w / h,
                                                    face, background, seed=i)
        drawings.append(graphite_drawing(photo, (w, h), seed=i,
                                         border=0.015 if background == "whole" else 0.03))
        silhouettes.append(silhouette)
    names = [p.name for p in players]
    pages = [compose(layout, drawings, names, title, k) for k in range(len(players) + 1)]
    blank = compose(layout, drawings, names, title, 0, show_title=False)

    # ---- the parts of the video, in order. Each segment looks for its face only in
    # the portrait being drawn, so construction lines never land on a finished player.
    n = len(players)
    final, blank_u8 = to_uint8(pages[-1]), to_uint8(blank)
    title_time = 4.0 if title else 0.0
    parts: List[tuple] = []
    if show.intro:
        hook = (show.hook.strip() or title or "Football legends").upper()
        kicker = f"{n} PLAYER{'S' if n > 1 else ''}  ·  1 PENCIL"
        parts.append(("clip", intro_clip(final, blank_u8, hook, kicker,
                                         "Watch them come to life in pencil", fps), "Intro"))
    if title:
        parts.append(("draw", blank, pages[0], title_time, "Writing the title", layout.title_box))
    for i in range(n):
        if show.highlights:
            label = "WHO'S FIRST?" if i == 0 else "WHO'S NEXT?"
            parts.append(("clip", teaser_clip(to_uint8(pages[i]), layout.cells[i], silhouettes[i],
                                              label, f"{i + 1} / {n}", fps),
                          "Who's first?" if i == 0 else "Who's next?"))
        parts.append(("draw", pages[i], pages[i + 1], None, f"Drawing {players[i].name}",
                      layout.cells[i]))
        if show.highlights:
            parts.append(("clip", spotlight_clip(to_uint8(pages[i + 1]), layout.cells[i],
                                                 layout.name_boxes[i], i + 1, fps), players[i].name))
    hold = 2.0 if show.outro else 3.0
    parts.append(("still", final, hold, "The finished picture"))
    if show.outro:
        question = show.question.strip() or ShowOptions.question
        parts.append(("clip", outro_clip(final, question, [c.short() for c in credits], fps),
                      "Like, subscribe, comment"))
    elif credits:
        parts.append(("credits", _credits_card(width, height, credits), 4.0, "Photo credits"))

    # ---- timing: fixed, or stretched to the length of the audio
    def length(part: tuple) -> float:
        return part[1].seconds if part[0] == "clip" else (part[3] if part[0] == "draw" else part[2])

    if audio and fit_to_audio:
        fixed = sum(length(pt) for pt in parts if not (pt[0] == "draw" and pt[3] is None))
        seconds_per_player = float(np.clip((audio_seconds - fixed) / n, 4.0, 60.0))
        extra = audio_seconds - fixed - seconds_per_player * n
        hold_index = next(k for k, pt in enumerate(parts) if pt[0] == "still")
        parts[hold_index] = ("still", final, float(np.clip(hold + extra, hold, 120.0)),
                             parts[hold_index][3])
    parts = [pt[:3] + (seconds_per_player,) + pt[4:] if pt[0] == "draw" and pt[3] is None else pt
             for pt in parts]
    counts = [int(round(length(pt) * fps)) for pt in parts]
    n_total = sum(counts)
    total = n_total / fps

    tmp = tempfile.mkdtemp(prefix=".players-", dir=out_dir)
    written, sounds = 0, []
    try:
        tmp_video = os.path.join(tmp, "video.mp4")
        with FFmpegWriter(tmp_video, width, height, fps, crf, preset, total_frames=n_total) as writer:
            for si, (part, count) in enumerate(zip(parts, counts)):
                kind, label = part[0], (part[4] if part[0] == "draw" else part[-1])
                report(0.15 + 0.8 * written / n_total, label)
                if kind == "draw":
                    _, start, end, seconds, label, region = part
                    cfg = RenderConfig(width=width, height=height, fps=fps, duration=seconds,
                                       hold=0.0, start_mode="image", align=False,
                                       match_paper=False, seed=11 + si)
                    report(0.15 + 0.8 * written / n_total, f"{label}: planning")
                    plan = plan_animation(start, end, cfg, focus_region=region)
                    renderer = FrameRenderer(plan)
                    for k, t, frame in renderer.iter_frames(fps, count):
                        writer.write(frame)
                        if k % 15 == 0:
                            report(0.15 + 0.8 * (written + k) / n_total, label)
                    del plan, renderer
                elif kind == "clip":
                    sounds += [(written / fps + at, name) for at, name in part[1].sounds]
                    for k, frame in zip(range(count), part[1].frames()):
                        writer.write(frame)
                elif kind == "still":
                    for _ in range(count):
                        writer.write(part[1])
                else:  # photo credits card
                    for k in range(count):
                        x = min(k / (0.6 * fps), 1.0, (count - k) / (0.6 * fps))
                        writer.write(cv2.addWeighted(final, 1.0 - x, part[1], x, 0.0))
                written += count
        effects = None
        if show.sound_effects and sounds:
            effects = os.path.join(tmp, "effects.wav")
            write_wav(effects, sound_track(sounds, total))
        if audio or effects:
            report(0.97, "Adding the sound")
            mixed = os.path.join(tmp, "with_sound.mp4")
            _add_soundtrack(tmp_video, mixed, total, audio, audio_seconds, effects)
            tmp_video = mixed
        os.replace(tmp_video, output_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    image_path = os.path.splitext(output_path)[0] + ".png"
    Image.fromarray(to_uint8(pages[-1])).save(image_path)
    credits_path = None
    if credits:
        credits_path = os.path.splitext(output_path)[0] + ".credits.txt"
        with open(credits_path, "w", encoding="utf-8") as fh:
            fh.write("Photos (converted into pencil drawings), from Wikimedia Commons:\n")
            for i, c in enumerate(credits, 1):
                fh.write(f"{i}. {c.line()}\n")
    report(1.0, "Done")
    return PlayersResult(output_path, image_path, credits_path, total, names,
                         time.perf_counter() - started, warnings)


def _add_soundtrack(video: str, out: str, video_seconds: float, audio: Optional[str] = None,
                    audio_seconds: float = 0.0, effects: Optional[str] = None) -> None:
    """Lay the user's ``audio`` (cut to the video with a short fade in and a fade
    out, silence after it if shorter) and the sound ``effects`` under ``video``."""
    inputs, chains, labels = ["-i", video], [], []
    if audio:
        end = min(video_seconds, audio_seconds)
        fade = min(2.5, end / 4.0)
        chains.append(f"[{len(labels) + 1}:a]atrim=0:{end:.3f},asetpts=PTS-STARTPTS,"
                      f"afade=t=in:d=0.3,afade=t=out:st={end - fade:.3f}:d={fade:.3f},"
                      f"volume={0.85 if effects else 1.0},apad=whole_dur={video_seconds:.3f},"
                      "aformat=channel_layouts=stereo[music]")
        inputs += ["-i", audio]
        labels.append("[music]")
    if effects:
        chains.append(f"[{len(labels) + 1}:a]apad=whole_dur={video_seconds:.3f},"
                      "aformat=channel_layouts=stereo[effects]")
        inputs += ["-i", effects]
        labels.append("[effects]")
    if len(labels) == 2:
        chains.append("[music][effects]amix=inputs=2:normalize=0:duration=longest,"
                      "alimiter=limit=0.95[sound]")
        labels = ["[sound]"]
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *inputs,
           "-filter_complex", ";".join(chains), "-map", "0:v:0", "-map", labels[0],
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{video_seconds:.3f}",
           "-movflags", "+faststart", out]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RenderError(f"Could not add the sound: {proc.stderr.strip()[-300:]}")


def _credits_card(width: int, height: int, credits: List[ImageCredit]) -> np.ndarray:
    from .story import text_card  # shared card layout

    lines = [f"{c.title}: {c.author or 'unknown author'} ({c.licence})" for c in credits]
    return text_card(width, height, "Photos: Wikimedia Commons", lines + [
        "", "Photos converted into pencil drawings."], title_scale=0.04)


# --------------------------------------------------------------------------- CLI

def name_from_filename(path: str) -> str:
    """A player's name guessed from a photo's file name ("cristiano_ronaldo-2.jpg"
    -> "Cristiano Ronaldo")."""
    stem = os.path.splitext(os.path.basename(path))[0]
    words = re.sub(r"[_\-.]+", " ", re.sub(r"\(.*?\)", " ", stem)).split()
    while words and words[-1].isdigit():
        words.pop()
    name = " ".join(words)
    return name.title() if name == name.lower() else name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pencil-players",
                                description="Draw footballers in pencil, together in one frame "
                                            "(simulated drawing): your own photos, or a country's "
                                            "famous players found on Wikidata.")
    p.add_argument("country", nargs="?", default="",
                   help="country name in English, e.g. Portugal (not needed with --photos)")
    p.add_argument("-o", "--output", help="output MP4 (a PNG is saved next to it)")
    p.add_argument("--photos", nargs="+", metavar="PHOTO",
                   help="your own photos, drawn in this order; name a player with "
                        "'Name=photo.jpg', otherwise the file name is used")
    p.add_argument("--title", help="title written above the players (default: the country)")
    p.add_argument("--background", choices=BACKGROUNDS, default="whole",
                   help="whole: draw the entire photo, nothing cut (default); shade: head and "
                        "shoulders, background shaded in pencil; photo: head and shoulders with "
                        "the photo's background")
    p.add_argument("--audio", help="music or narration to lay under the video")
    p.add_argument("--keep-timing", action="store_true",
                   help="with --audio: keep --seconds instead of fitting the video to the audio")
    p.add_argument("--hook", default="", help="intro headline (default: the title)")
    p.add_argument("--question", default=ShowOptions.question,
                   help="question asked at the end, before like / subscribe")
    for flag, what in (("intro", "the intro"), ("highlights", "the teaser and highlight per player"),
                       ("outro", "the like / subscribe / comment ending"),
                       ("sound-effects", "the sound effects")):
        p.add_argument(f"--no-{flag}", action="store_true", help=f"leave out {what}")
    p.add_argument("--count", type=int, default=6, help="number of players found (1-12)")
    p.add_argument("--size", default="1920x1080", help="resolution, e.g. 1080x1920 for vertical")
    p.add_argument("--seconds", type=float, default=10.0, help="drawing time per player")
    p.add_argument("--list", action="store_true", help="only list the players found")
    args = p.parse_args(argv)
    if not args.photos and not args.country:
        p.error("give a country or --photos")
    if not args.list and not args.output:
        p.error("the output file (-o) is required")
    try:
        if args.photos:
            chosen = []
            for item in args.photos:
                name, sep, path = item.partition("=")
                if not sep:
                    name, path = name_from_filename(item), item
                if not os.path.isfile(path):
                    raise InputError(f"Photo not found: {path}")
                chosen.append(Player(name.strip(), "", os.path.basename(path), 0,
                                     local_file=os.path.abspath(path)))
        else:
            found = find_players(args.country, limit=max(15, args.count))
            if args.list:
                for pl in found:
                    print(f"{pl.fame:4d}  {pl.name}")
                return 0
            chosen = [pl for pl in found if pl.photo_url][:args.count]
        w, h = (int(v) for v in args.size.lower().split("x"))
        title = args.title if args.title is not None else args.country.upper()
        interactive = sys.stderr.isatty()

        def progress(fraction: float, message: str) -> None:
            if interactive:
                sys.stderr.write(f"\r[{'#' * int(fraction * 30):<30}] {fraction * 100:5.1f}%  "
                                 f"{message[:60]:<60}")
                sys.stderr.flush()

        result = render_players(chosen, title, args.output, w, h, seconds_per_player=args.seconds,
                                progress=progress, background=args.background,
                                audio=args.audio, fit_to_audio=not args.keep_timing,
                                show=ShowOptions(not args.no_intro, not args.no_highlights,
                                                 not args.no_outro, not args.no_sound_effects,
                                                 args.hook, args.question))
        if interactive:
            sys.stderr.write("\n")
    except PencilError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {result.output_path} ({result.duration:.0f} s) and {result.image_path}")
    print("  players: " + ", ".join(result.players))
    if result.credits_path:
        print(f"  photo credits: {result.credits_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
