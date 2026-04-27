from pathlib import Path

import pretty_midi
import pytest

from pipeline.progression import ChordProgression
from pipeline.chord_render import build_chord_instrument, render_chord_track


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4), ("Dm7", 4), ("G7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_build_chord_instrument_returns_instrument():
    inst = build_chord_instrument(_basic_progression())
    assert isinstance(inst, pretty_midi.Instrument)


def test_build_chord_instrument_program_zero():
    inst = build_chord_instrument(_basic_progression())
    assert inst.program == 0
    assert inst.is_drum is False


def test_build_chord_instrument_correct_note_count():
    """4 chord × 4-voice voicing = 16 notes."""
    inst = build_chord_instrument(_basic_progression())
    assert len(inst.notes) == 16


def test_build_chord_instrument_durations_match_progression():
    p = ChordProgression(
        chords=[("Cmaj7", 2), ("Am7", 6)],
        tempo=120.0,
        time_signature="4/4",
    )
    inst = build_chord_instrument(p)
    cmaj_notes = [n for n in inst.notes if n.start == 0.0]
    am_notes = [n for n in inst.notes if n.start > 0.0]
    assert len(cmaj_notes) == 4
    assert len(am_notes) == 4
    assert all(abs(n.end - 1.0) < 1e-6 for n in cmaj_notes)
    assert all(abs(n.start - 1.0) < 1e-6 for n in am_notes)
    assert all(abs(n.end - 4.0) < 1e-6 for n in am_notes)


def test_build_chord_instrument_does_not_write_file(tmp_path: Path):
    """build_chord_instrument should NOT touch disk — only return Instrument."""
    snapshot_before = set(tmp_path.iterdir())
    build_chord_instrument(_basic_progression())
    snapshot_after = set(tmp_path.iterdir())
    assert snapshot_before == snapshot_after


def test_render_chord_track_writes_file(tmp_path: Path):
    out = tmp_path / "chords.mid"
    render_chord_track(_basic_progression(), out)
    assert out.exists()
    pm = pretty_midi.PrettyMIDI(str(out))
    assert len(pm.instruments) == 1


def test_render_chord_track_has_one_piano_instrument(tmp_path: Path):
    out = tmp_path / "chords.mid"
    render_chord_track(_basic_progression(), out)
    pm = pretty_midi.PrettyMIDI(str(out))
    inst = pm.instruments[0]
    assert inst.program == 0  # Acoustic Grand Piano (GM 0-indexed)
    assert inst.is_drum is False


def test_render_chord_track_correct_number_of_chord_blocks(tmp_path: Path):
    """4 аккорда × 4 ноты voicing = 16 нот."""
    out = tmp_path / "chords.mid"
    render_chord_track(_basic_progression(), out)
    pm = pretty_midi.PrettyMIDI(str(out))
    notes = pm.instruments[0].notes
    # Cmaj7, Am7, Dm7, G7 — все четырёхголосные аккорды (3 интервала + корень)
    assert len(notes) == 16


def test_render_chord_track_durations_match_progression(tmp_path: Path):
    p = ChordProgression(
        chords=[("Cmaj7", 2), ("Am7", 6)],  # 2 beats + 6 beats
        tempo=120.0,
        time_signature="4/4",
    )
    out = tmp_path / "chords.mid"
    render_chord_track(p, out)
    pm = pretty_midi.PrettyMIDI(str(out))
    notes = pm.instruments[0].notes
    # каждый аккорд — 4 ноты с одинаковым start/end
    cmaj_notes = [n for n in notes if n.start == 0.0]
    am_notes   = [n for n in notes if n.start > 0.0]
    assert len(cmaj_notes) == 4
    assert len(am_notes)   == 4
    # 120 BPM → 1 beat = 0.5 sec
    # Cmaj7: 0.0 — 1.0; Am7: 1.0 — 4.0
    assert all(abs(n.end - 1.0) < 1e-6 for n in cmaj_notes)
    assert all(abs(n.start - 1.0) < 1e-6 for n in am_notes)
    assert all(abs(n.end - 4.0) < 1e-6 for n in am_notes)


def test_render_chord_track_tempo_in_midi(tmp_path: Path):
    out = tmp_path / "chords.mid"
    p = ChordProgression(chords=[("Cmaj7", 4)], tempo=100.0, time_signature="4/4")
    render_chord_track(p, out)
    pm = pretty_midi.PrettyMIDI(str(out))
    # initial_tempo читается из header
    assert abs(pm.get_tempo_changes()[1][0] - 100.0) < 1e-6 or abs(pm.estimate_tempo() - 100.0) < 5.0
