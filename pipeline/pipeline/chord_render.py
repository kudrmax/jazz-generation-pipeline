from __future__ import annotations

from pathlib import Path

import pretty_midi

from pipeline.chord_vocab import chord_to_pitches
from pipeline.progression import ChordProgression


_CHORD_VELOCITY = 70


def build_chord_instrument(progression: ChordProgression) -> pretty_midi.Instrument:
    """ChordProgression → pretty_midi.Instrument с piano block voicing.

    Возвращает Instrument (program=0 Acoustic Grand Piano, name='ChordTrack',
    is_drum=False), без записи файла на диск.

    На каждый аккорд — block voicing (root+3rd+5th+7th или triada) длительностью
    `duration_in_beats * (60 / tempo)` секунд.
    """
    inst = pretty_midi.Instrument(program=0, name="ChordTrack", is_drum=False)

    seconds_per_beat = 60.0 / progression.tempo
    cursor = 0.0
    for chord_str, beats in progression.chords:
        pitches = chord_to_pitches(chord_str)
        start = cursor
        end = cursor + beats * seconds_per_beat
        for p in pitches:
            inst.notes.append(pretty_midi.Note(
                velocity=_CHORD_VELOCITY, pitch=p, start=start, end=end,
            ))
        cursor = end
    return inst


def render_chord_track(progression: ChordProgression, out_path: Path) -> None:
    """Writes chord-track MIDI file using build_chord_instrument.

    Тонкая обёртка вокруг build_chord_instrument для случаев, когда нужен
    отдельный MIDI файл (например, для прослушивания chord-track в изоляции).
    Pipeline-orchestrator (postprocess) использует build_chord_instrument напрямую,
    минуя файл.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    pm.instruments.append(build_chord_instrument(progression))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
