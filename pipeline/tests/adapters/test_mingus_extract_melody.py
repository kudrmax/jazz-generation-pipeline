from pathlib import Path

import pretty_midi
import pytest

from pipeline.adapters.mingus import MingusAdapter, MingusPipelineConfig


def _make_two_track_midi(out_path: Path, melody_name: str = "Tenor Sax") -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    melody_inst = pretty_midi.Instrument(program=66, name=melody_name)
    for i, p in enumerate([60, 64, 67, 71]):
        melody_inst.notes.append(pretty_midi.Note(80, p, i * 0.5, (i + 1) * 0.5))
    pm.instruments.append(melody_inst)

    chord_inst = pretty_midi.Instrument(program=0, name="piano")
    for i, p in enumerate([60, 64, 67]):
        chord_inst.notes.append(pretty_midi.Note(60, p, 0.0, 2.0))
    pm.instruments.append(chord_inst)

    pm.write(str(out_path))


def test_extract_melody_returns_tenor_sax_track(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi)
    cfg = MingusPipelineConfig()
    melody = MingusAdapter(cfg).extract_melody(midi)
    assert melody.name == "Tenor Sax"
    assert len(melody.notes) == 4


def test_extract_melody_returns_instrument_type(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi)
    cfg = MingusPipelineConfig()
    melody = MingusAdapter(cfg).extract_melody(midi)
    assert isinstance(melody, pretty_midi.Instrument)


def test_extract_melody_raises_when_track_missing(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi, melody_name="Wrong Name")
    cfg = MingusPipelineConfig()
    with pytest.raises(ValueError, match="Tenor Sax"):
        MingusAdapter(cfg).extract_melody(midi)
