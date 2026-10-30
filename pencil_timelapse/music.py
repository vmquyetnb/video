"""Music made by the program, free to publish: a cheerful background tune
(plucked ukulele, a soft bass, bells and light drums), a gentle lullaby for
bedtime stories, and a short theme jingle for the series intro.

Instruments are synthesised: plucked strings with the Karplus-Strong method,
bells from a few inharmonic sine partials, drums from shaped noise.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .media import SAMPLE_RATE

RATE = SAMPLE_RATE
MOODS = ("happy", "calm")
NOTE = {n: i for i, n in enumerate(("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"))}


def freq(note: str) -> float:
    """"A4" -> 440 Hz."""
    name, octave = note[:-1], int(note[-1])
    return 440.0 * 2 ** ((NOTE[name] + 12 * (octave + 1) - 69) / 12)


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def pluck(f: float, seconds: float, rng: np.random.Generator, bright: float = 0.5) -> np.ndarray:
    """A plucked string (Karplus-Strong), a period at a time."""
    n = int(seconds * RATE)
    period = max(2, int(round(RATE / f)))
    out = np.zeros(n + period, np.float64)
    buf = rng.uniform(-1, 1, period)
    buf = buf * bright + np.convolve(buf, np.ones(3) / 3, "same") * (1 - bright)
    decay = 0.996
    for k in range(0, n + period, period):
        out[k:k + period] = buf[:len(out[k:k + period])]
        buf = decay * 0.5 * (buf + np.roll(buf, 1))
    out = out[:n]
    return out * np.minimum(1.0, np.arange(n) / (0.003 * RATE))


def bell(f: float, seconds: float, soft: float = 1.0) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    tone = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.15)
            + 0.18 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.06))
    return tone * _env(n, 0.002, 0.5 * soft)


def bass(f: float, seconds: float) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    return (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * _env(n, 0.01, 0.35)


def pad(freqs: Sequence[float], seconds: float) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    out = sum(np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * 2 * f * t) for f in freqs)
    shape = np.minimum(1.0, t / 0.6) * np.minimum(1.0, (t[-1] - t) / 0.6 + 1e-3)
    return out * shape / len(freqs)


def _noise(n: int, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    spec[(f < lo) | (f > hi)] = 0
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-9)


def kick() -> np.ndarray:
    n = int(0.3 * RATE)
    t = np.arange(n) / RATE
    f = 50 + 90 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / RATE) * _env(n, 0.001, 0.12)


def clap(rng: np.random.Generator) -> np.ndarray:
    n = int(0.18 * RATE)
    return _noise(n, rng, 800, 6000) * _env(n, 0.001, 0.05)


def hat(rng: np.random.Generator, open_: bool = False) -> np.ndarray:
    n = int((0.25 if open_ else 0.06) * RATE)
    return _noise(n, rng, 6000, 16000) * _env(n, 0.0005, 0.08 if open_ else 0.015)


def _add(track: np.ndarray, at: float, sound: np.ndarray, gain: float) -> None:
    s = int(at * RATE)
    if s >= len(track):
        return
    e = min(len(track), s + len(sound))
    track[s:e] += sound[:e - s] * gain


CHORDS = {"C": ("C", "E", "G"), "G": ("G", "B", "D"), "Am": ("A", "C", "E"), "F": ("F", "A", "C"),
          "Dm": ("D", "F", "A"), "Em": ("E", "G", "B"), "Bb": ("A#", "D", "F")}
HAPPY_MELODY = [  # one bar per line, eighth notes ("" = rest)
    "E5 - G5 - E5 D5 C5 -", "D5 - G5 - D5 - B4 -", "C5 - E5 - A5 G5 E5 -", "F5 - A5 - G5 - F5 -",
    "E5 G5 C6 - B5 - G5 -", "D5 - G5 - B5 A5 G5 -", "A5 - F5 - C5 - A4 -", "B4 - D5 - G5 - - -"]


def _chord_notes(name: str, octave: int) -> List[float]:
    notes = CHORDS[name]
    out, last = [], -1
    for n in notes:
        o = octave
        while NOTE[n] + 12 * o <= last:
            o += 1
        last = NOTE[n] + 12 * o
        out.append(freq(f"{n}{o}"))
    return out


def background(seconds: float, mood: str = "happy", seed: int = 1) -> np.ndarray:
    """A tune at least ``seconds`` long (mono, peak about 0.5), looping smoothly."""
    rng = np.random.default_rng(seed)
    if mood == "calm":
        bpm, bars, chords = 72, 8, ["F", "C", "Dm", "Bb", "F", "C", "Bb", "C"]
    else:
        bpm, bars, chords = 104, 16, ["C", "G", "Am", "F", "C", "G", "F", "G"] * 2
    beat = 60.0 / bpm
    loop_s = bars * 4 * beat
    loop = np.zeros(int(loop_s * RATE) + RATE, np.float64)
    pattern = (0, 2, 1, 2, 3, 2, 1, 2)
    for b, name in enumerate(chords):
        t0 = b * 4 * beat
        tones = _chord_notes(name, 4)
        tones.append(tones[0] * 2)
        if mood == "calm":
            _add(loop, t0, pad(_chord_notes(name, 3), 4 * beat + 0.4), 0.16)
            for k in range(8):  # a music box, slowly
                f = tones[pattern[k]] * 2
                _add(loop, t0 + k * beat / 2, bell(f, 1.6, 0.8), 0.12 if k % 2 == 0 else 0.07)
            _add(loop, t0, bass(_chord_notes(name, 2)[0], 2.5), 0.25)
            continue
        for k in range(8):  # ukulele on the eighths
            f = tones[pattern[k]]
            _add(loop, t0 + k * beat / 2 + rng.uniform(0, 0.008), pluck(f, 0.8, rng, 0.45),
                 0.22 if k % 2 == 0 else 0.15)
        root = _chord_notes(name, 2)[0]
        _add(loop, t0, bass(root, beat * 1.5), 0.35)
        _add(loop, t0 + 2 * beat, bass(root, beat * 1.5), 0.3)
        _add(loop, t0, kick(), 0.32)
        _add(loop, t0 + 2 * beat, kick(), 0.25)
        for k in range(4):
            _add(loop, t0 + (k + 0.5) * beat, hat(rng), 0.06)
        _add(loop, t0 + beat, clap(rng), 0.07)
        _add(loop, t0 + 3 * beat, clap(rng), 0.07)
        if b < 8:  # the tune, every other time round
            for k, note in enumerate(HAPPY_MELODY[b].split()):
                if note != "-":
                    _add(loop, t0 + k * beat / 2, bell(freq(note), 1.0), 0.1)
    # loop it, letting each pass ring into the next
    tail = loop[int(loop_s * RATE):].copy()
    loop = loop[:int(loop_s * RATE)]
    loop[:len(tail)] += tail
    reps = int(math.ceil(seconds / loop_s)) + 1
    out = np.tile(loop, reps)[:int(seconds * RATE) + 1]
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def jingle(seed: int = 3) -> Tuple[np.ndarray, List[float]]:
    """The series theme (about 6.6 s, peak about 0.8) and the beat times the
    picture can land on: letters dropping, friends popping in, the last chord."""
    rng = np.random.default_rng(seed)
    bpm = 120
    beat = 60.0 / bpm
    seconds = 13 * beat + 1.2
    out = np.zeros(int(seconds * RATE), np.float64)
    marks = []
    # a rising run while the letters drop in
    run = ["C5", "D5", "E5", "G5", "A5", "C6", "D6", "E6"]
    for k, note in enumerate(run):
        at = k * beat / 4
        _add(out, at, bell(freq(note), 0.6), 0.22)
        _add(out, at, pluck(freq(note) / 2, 0.5, rng, 0.6), 0.18)
        marks.append(at)
    _add(out, 0.0, _noise(int(0.9 * RATE), rng, 3000, 12000) * np.linspace(0, 1, int(0.9 * RATE)) ** 2, 0.1)
    # the hook, with drums: "Pee-ko and friends!"
    hook = [("G5", 2), ("E5", 2), ("G5", 2), ("C6", 2), ("A5", 2), ("G5", 1), ("E5", 1), ("D5", 2),
            ("C5", 1), ("E5", 1), ("G5", 2), ("C6", 4)]
    chords = ["C", "C", "F", "G", "C", "C"]
    start = 2 * beat
    at = start
    for note, length in hook:
        _add(out, at, bell(freq(note), 0.9), 0.3)
        _add(out, at, pluck(freq(note), 0.7, rng, 0.5), 0.2)
        at += length * beat / 2
    for b in range(len(chords)):
        t0 = start + b * 2 * beat
        for k, f in enumerate(_chord_notes(chords[b], 4)):
            _add(out, t0 + k * 0.02, pluck(f, 1.0, rng, 0.4), 0.16)
        _add(out, t0, bass(_chord_notes(chords[b], 2)[0], 2 * beat), 0.4)
        _add(out, t0, kick(), 0.45)
        _add(out, t0 + beat, clap(rng), 0.18)
        _add(out, t0 + beat / 2, hat(rng), 0.08)
        _add(out, t0 + 1.5 * beat, hat(rng), 0.08)
        marks.append(t0)
    end = start + len(chords) * 2 * beat
    for k, f in enumerate(_chord_notes("C", 4) + [freq("C6"), freq("E6"), freq("G6")]):  # the last chord: ta-da!
        _add(out, end + 0.015 * k, pluck(f, 1.6, rng, 0.5), 0.14)
        _add(out, end + 0.015 * k, bell(f, 1.4), 0.1)
    _add(out, end, kick(), 0.5)
    _add(out, end, hat(rng, True), 0.2)
    marks.append(end)
    return (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32), marks


def duck(music: np.ndarray, voices: np.ndarray, depth: float = 0.55) -> np.ndarray:
    """Lower the music while someone speaks (both mono, same length)."""
    n = min(len(music), len(voices))
    hop = RATE // 50
    m = n // hop
    if m == 0:
        return music
    env = np.abs(voices[:m * hop]).reshape(m, hop).max(axis=1)
    env = np.convolve(env > 0.02, np.ones(25) / 25, "same")  # half a second either side
    gain = 1.0 - depth * np.clip(env * 1.5, 0, 1)
    gain = np.repeat(gain, hop)
    out = music.copy()
    out[:len(gain)] *= gain.astype(np.float32)
    return out


def tune(mood: Optional[str], seconds: float) -> Optional[np.ndarray]:
    """The background music for a mood, or None for no music."""
    if mood not in MOODS:
        return None
    return background(seconds, mood)
