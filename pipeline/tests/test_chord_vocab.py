import pytest

from pipeline.chord_vocab import (
    ROOTS, QUALITIES,
    parse_chord, chord_to_pitches,
    QUALITY_FALLBACK_TO_TRIADS, EXTENDED_FALLBACK_TO_VOCAB,
)


def test_roots_count():
    assert len(ROOTS) == 12
    assert ROOTS[0] == "C"
    assert ROOTS[11] == "B"


def test_qualities_count():
    assert len(QUALITIES) == 7
    assert "maj7" in QUALITIES
    assert "dim7" in QUALITIES


@pytest.mark.parametrize("chord_str, expected", [
    ("C",      (0,  "maj")),
    ("Cmaj",   (0,  "maj")),
    ("C#",     (1,  "maj")),
    ("Db",     (1,  "maj")),
    ("Cmaj7",  (0,  "maj7")),
    ("Cm7",    (0,  "min7")),
    ("Cmin7",  (0,  "min7")),
    ("C7",     (0,  "7")),
    ("Bbm7",   (10, "min7")),
    ("F#dim7", (6,  "dim7")),
    ("Adim",   (9,  "dim")),
])
def test_parse_chord(chord_str, expected):
    assert parse_chord(chord_str) == expected


def test_parse_chord_unknown_root():
    with pytest.raises(ValueError, match="root"):
        parse_chord("Hmaj")


def test_parse_chord_unknown_quality():
    with pytest.raises(ValueError, match="quality"):
        parse_chord("Csus2")


def test_chord_to_pitches_cmaj7():
    # C major 7: C E G B в pretty_midi convention (C4 = 60, octave=4)
    assert chord_to_pitches("Cmaj7") == [60, 64, 67, 71]


def test_chord_to_pitches_am7():
    # A min 7: A C E G — корень A4 = 69 (pretty_midi convention, octave=4).
    # Резолюция mismatch'а спеки: единая формула base = root_idx + 12*(octave+1),
    # все pitches остаются в [60, 79] ⊂ [40, 84].
    assert chord_to_pitches("Am7") == [69, 72, 76, 79]


def test_chord_to_pitches_dim7():
    # C dim 7: C Eb Gb Bbb (=A) — все интервалы по минор-3
    assert chord_to_pitches("Cdim7") == [60, 63, 66, 69]


def test_quality_fallback_to_triads():
    assert QUALITY_FALLBACK_TO_TRIADS["7"] == "maj"
    assert QUALITY_FALLBACK_TO_TRIADS["min7"] == "min"
    assert QUALITY_FALLBACK_TO_TRIADS["dim7"] == "dim"


def test_extended_fallback():
    assert EXTENDED_FALLBACK_TO_VOCAB["m7b5"] == "dim"
    assert EXTENDED_FALLBACK_TO_VOCAB["13"] == "7"
    assert EXTENDED_FALLBACK_TO_VOCAB["6"] == "maj"
