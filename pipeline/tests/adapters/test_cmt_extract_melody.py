from pathlib import Path

import pretty_midi
import pytest

from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig


def _cfg(tmp_path: Path) -> CMTPipelineConfig:
    return CMTPipelineConfig(
        checkpoint_path=tmp_path / "x", hparams_path=tmp_path / "x", repo_path=tmp_path / "x",
    )


def _two_track_midi(out: Path, melody_name: str = "melody") -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    melody = pretty_midi.Instrument(program=0, name=melody_name)
    for i, p in enumerate([60, 64, 67, 71]):
        melody.notes.append(pretty_midi.Note(80, p, i * 0.5, (i + 1) * 0.5))
    pm.instruments.append(melody)
    chord = pretty_midi.Instrument(program=0, name="chord")
    for p in [60, 64, 67]:
        chord.notes.append(pretty_midi.Note(60, p, 0.0, 2.0))
    pm.instruments.append(chord)
    pm.write(str(out))


def test_extract_melody_picks_melody_track(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _two_track_midi(midi)
    inst = CMTAdapter(_cfg(tmp_path)).extract_melody(midi)
    assert inst.name == "melody"
    assert len(inst.notes) == 4
    assert isinstance(inst, pretty_midi.Instrument)


def test_extract_melody_raises_when_track_missing(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _two_track_midi(midi, melody_name="not_melody")
    with pytest.raises(ValueError, match="melody"):
        CMTAdapter(_cfg(tmp_path)).extract_melody(midi)
