from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pretty_midi

from pipeline.chord_render import render_chord_track
from pipeline.progression import ChordProgression


def postprocess(
    melody: pretty_midi.Instrument,
    progression: ChordProgression,
    model_name: str,
    run_id: str,
    output_root: Path,
    melody_program: int,
) -> dict[str, Path]:
    """Принимает уже извлечённую мелодию (Instrument) и пишет два нормализованных MIDI.

    1. Клонирует melody, проставляет program ← melody_program, name ← 'Melody'.
    2. Сохраняет монофонную мелодию в `output_root/melody_only/<model>_<run_id>.mid`.
    3. Через chord_render строит наш piano chord track из progression.
    4. Склеивает melody + chord track в `output_root/with_chords/<model>_<run_id>.mid`.

    Не знает ни про MINGUS, ни про любую другую модель.
    """
    out = Path(output_root)
    melody_dir = out / "melody_only"
    chords_dir = out / "with_chords"
    melody_dir.mkdir(parents=True, exist_ok=True)
    chords_dir.mkdir(parents=True, exist_ok=True)

    melody_normalized = copy.deepcopy(melody)
    melody_normalized.program = melody_program
    melody_normalized.name = "Melody"
    melody_normalized.is_drum = False

    melody_path = melody_dir / f"{model_name}_{run_id}.mid"
    chords_path = chords_dir / f"{model_name}_{run_id}.mid"

    pm_melody = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    pm_melody.instruments.append(melody_normalized)
    pm_melody.write(str(melody_path))

    with tempfile.TemporaryDirectory() as td:
        chord_only_path = Path(td) / "chord_only.mid"
        render_chord_track(progression, chord_only_path)
        chord_pm = pretty_midi.PrettyMIDI(str(chord_only_path))

    pm_full = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    pm_full.instruments.append(copy.deepcopy(melody_normalized))
    for inst in chord_pm.instruments:
        pm_full.instruments.append(inst)
    pm_full.write(str(chords_path))

    return {"melody_only": melody_path, "with_chords": chords_path}
