from __future__ import annotations

from pathlib import Path

import pretty_midi

from pipeline.chord_vocab import chord_to_pitches
from pipeline.progression import ChordProgression


_CHORD_VELOCITY = 70


def render_chord_track(progression: ChordProgression, out_path: Path) -> None:
    """ChordProgression → MIDI с одним piano-треком (Acoustic Grand Piano, program 0).

    На каждый аккорд — block voicing (root+3rd+5th+7th или triada) длительностью
    duration_in_beats. Tempo и BPM соответствуют progression.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    inst = pretty_midi.Instrument(program=0, name="ChordTrack")

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

    pm.instruments.append(inst)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
