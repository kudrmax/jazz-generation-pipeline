from __future__ import annotations

ROOTS: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES: list[str] = ["maj", "min", "7", "maj7", "min7", "dim", "dim7"]

_ROOT_ALIASES: dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "C#": "C#", "D#": "D#", "F#": "F#", "G#": "G#", "A#": "A#",
}

_QUALITY_ALIASES: dict[str, str] = {
    "":     "maj",
    "m":    "min",
    "min":  "min",
    "maj":  "maj",
    "M":    "maj",
    "7":    "7",
    "M7":   "maj7",
    "maj7": "maj7",
    "m7":   "min7",
    "min7": "min7",
    "dim":  "dim",
    "dim7": "dim7",
}

_QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "dim7": [0, 3, 6, 9],
}

def parse_chord(chord_str: str) -> tuple[int, str]:
    """`"Cmaj7"` → `(0, "maj7")`. Pitch class of root (0..11), normalized quality."""
    s = chord_str.strip()
    # выделяем root: либо одна буква + опциональный # или b, либо одна буква
    if len(s) >= 2 and s[1] in "#b":
        root_raw, rest = s[:2], s[2:]
    else:
        root_raw, rest = s[:1], s[1:]
    root_norm = _ROOT_ALIASES.get(root_raw, root_raw)
    if root_norm not in ROOTS:
        raise ValueError(f"unknown root in chord {chord_str!r}: {root_raw!r}")
    root_idx = ROOTS.index(root_norm)
    quality = _QUALITY_ALIASES.get(rest)
    if quality is None:
        raise ValueError(f"unknown quality in chord {chord_str!r}: {rest!r}")
    return root_idx, quality


def chord_to_pitches(chord_str: str, octave: int = 4) -> list[int]:
    """Возвращает MIDI-pitches аккорда в указанной октаве (по умолчанию 4-я: C4=60).

    Корень кладётся как `pitch_class + 12*(octave+1)` (pretty_midi convention,
    где C-1 = 0, C4 = 60). Это означает, что **корень всегда находится в одной
    и той же MIDI-октаве**, независимо от его pitch class:
      - Cmaj7 (root_idx=0,  octave=4) → C4=60,   pitches [60, 64, 67, 71]
      - Am7   (root_idx=9,  octave=4) → A4=69,   pitches [69, 72, 76, 79]
      - Cdim7 (root_idx=0,  octave=4) → C4=60,   pitches [60, 63, 66, 69]

    При `octave=4` (default) все pitches попадают в диапазон [60, 82],
    что комфортно лежит в [40, 84] (~ E2..C6) — рабочем регистре пианино.
    Для transposition в другой регистр передавай `octave` явно.
    """
    root_idx, quality = parse_chord(chord_str)
    intervals = _QUALITY_INTERVALS[quality]
    base = root_idx + 12 * (octave + 1)
    return [base + i for i in intervals]
