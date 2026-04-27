from pathlib import Path

import pretty_midi
import pytest

from pipeline.progression import ChordProgression
from pipeline.postprocess import postprocess


def _fake_melody() -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=99, name="OriginalName")  # any non-66 program
    for i, pitch in enumerate([60, 62, 64, 65]):
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=pitch, start=i * 0.5, end=(i + 1) * 0.5
        ))
    return inst


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_postprocess_creates_two_files(tmp_path: Path):
    melody = _fake_melody()
    paths = postprocess(
        melody, _basic_progression(),
        model_name="mingus", run_id="20260427-test-aaaa",
        output_root=tmp_path, melody_program=66,
    )
    assert "melody_only" in paths and "with_chords" in paths
    assert Path(paths["melody_only"]).exists()
    assert Path(paths["with_chords"]).exists()


def test_postprocess_melody_only_has_one_instrument_with_target_program(tmp_path: Path):
    melody = _fake_melody()
    paths = postprocess(
        melody, _basic_progression(),
        model_name="mingus", run_id="20260427-test-bbbb",
        output_root=tmp_path, melody_program=66,
    )
    pm = pretty_midi.PrettyMIDI(str(paths["melody_only"]))
    assert len(pm.instruments) == 1
    assert pm.instruments[0].program == 66
    assert pm.instruments[0].name == "Melody"
    assert len(pm.instruments[0].notes) == 4


def test_postprocess_with_chords_has_two_instruments(tmp_path: Path):
    melody = _fake_melody()
    paths = postprocess(
        melody, _basic_progression(),
        model_name="mingus", run_id="20260427-test-cccc",
        output_root=tmp_path, melody_program=66,
    )
    pm = pretty_midi.PrettyMIDI(str(paths["with_chords"]))
    assert len(pm.instruments) == 2
    progs = sorted(i.program for i in pm.instruments)
    assert progs == [0, 66]  # piano chord track + melody


def test_postprocess_filename_pattern(tmp_path: Path):
    melody = _fake_melody()
    paths = postprocess(
        melody, _basic_progression(),
        model_name="bebopnet", run_id="20260427-153012-abcd1234",
        output_root=tmp_path, melody_program=66,
    )
    assert Path(paths["melody_only"]).name == "bebopnet_20260427-153012-abcd1234.mid"
    assert Path(paths["with_chords"]).name == "bebopnet_20260427-153012-abcd1234.mid"
    assert "melody_only" in str(paths["melody_only"])
    assert "with_chords" in str(paths["with_chords"])


def test_postprocess_does_not_mutate_input_melody(tmp_path: Path):
    melody = _fake_melody()
    original_program = melody.program
    original_name = melody.name
    postprocess(
        melody, _basic_progression(),
        model_name="mingus", run_id="20260427-test-dddd",
        output_root=tmp_path, melody_program=66,
    )
    assert melody.program == original_program
    assert melody.name == original_name
