from pathlib import Path

import pretty_midi
import pytest

from pipeline.adapters.bebopnet import BebopNetAdapter, BebopNetPipelineConfig


def _cfg(tmp_path: Path) -> BebopNetPipelineConfig:
    return BebopNetPipelineConfig(
        model_dir=tmp_path / "model", repo_path=tmp_path / "repo",
    )


def _single_track_midi(out: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    melody = pretty_midi.Instrument(program=65, name="Tenor Sax")
    for i, p in enumerate([60, 64, 67, 71]):
        melody.notes.append(pretty_midi.Note(80, p, i * 0.5, (i + 1) * 0.5))
    pm.instruments.append(melody)
    pm.write(str(out))


def _empty_midi(out: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.write(str(out))


def test_extract_melody_picks_first_track(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _single_track_midi(midi)
    inst = BebopNetAdapter(_cfg(tmp_path)).extract_melody(midi)
    assert isinstance(inst, pretty_midi.Instrument)
    assert len(inst.notes) == 4


def test_extract_melody_raises_when_no_instruments(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _empty_midi(midi)
    with pytest.raises(ValueError, match="no instruments|empty|melody"):
        BebopNetAdapter(_cfg(tmp_path)).extract_melody(midi)
