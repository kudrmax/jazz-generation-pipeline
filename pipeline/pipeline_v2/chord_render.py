"""Построение chord-track из Progression.

Отдельная утилита потому что (а) ResultPacker её использует для
with_chords.mid, (б) её удобно использовать в тестах и отладке
независимо от pipeline.
"""
from __future__ import annotations

import pretty_midi

from pipeline_v2.chord_vocab import chord_to_pitches
from pipeline_v2.types import Progression


_CHORD_VELOCITY = 70
_CHORD_PROGRAM = 0  # Acoustic Grand Piano


def build_chord_instrument(progression: Progression) -> pretty_midi.Instrument:
    """Progression → pretty_midi.Instrument с piano block voicing.

    На каждый аккорд — все pitches играют от его start до start+duration
    одновременно (block voicing). Длительность считается из beats и
    progression.tempo.
    """
    inst = pretty_midi.Instrument(
        program=_CHORD_PROGRAM, name="ChordTrack", is_drum=False,
    )
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
