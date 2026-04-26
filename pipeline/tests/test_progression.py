import json
from pathlib import Path

import pytest

from pipeline.progression import ChordProgression


def test_progression_total_beats():
    p = ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4), ("Dm7", 2), ("G7", 2)],
        tempo=120.0,
        time_signature="4/4",
    )
    assert p.total_beats() == 12


def test_progression_num_bars_4_4():
    p = ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )
    assert p.num_bars() == 2


def test_progression_num_bars_3_4():
    p = ChordProgression(
        chords=[("Cmaj7", 3), ("Am7", 3)],
        tempo=120.0,
        time_signature="3/4",
    )
    assert p.num_bars() == 2


def test_from_json_round_trip(tmp_path: Path):
    src = tmp_path / "p.json"
    src.write_text(json.dumps({
        "tempo": 100.0,
        "time_signature": "4/4",
        "chords": [["Cmaj7", 4], ["G7", 4]],
    }))
    p = ChordProgression.from_json(src)
    assert p.tempo == 100.0
    assert p.chords == [("Cmaj7", 4), ("G7", 4)]

    dst = tmp_path / "out.json"
    p.to_json(dst)
    reloaded = ChordProgression.from_json(dst)
    assert reloaded == p


def test_from_json_rejects_unknown_field(tmp_path: Path):
    src = tmp_path / "p.json"
    src.write_text(json.dumps({
        "tempo": 100.0,
        "time_signature": "4/4",
        "chords": [["Cmaj7", 4]],
        "seed": [60, 64],
    }))
    with pytest.raises(ValueError, match="seed"):
        ChordProgression.from_json(src)
