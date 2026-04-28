"""Слой 5 — финальная сборка.

Принимает уже извлечённую Melody и исходный Progression. Накладывает
единый тембр, строит chord-track, пишет два MIDI в стандартные
директории. Имена файлов собираются из ctx.model_label и ctx.run_id —
packer не знает про enum ModelName.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pretty_midi

from pipeline_v2.chord_render import build_chord_instrument
from pipeline_v2.types import (
    FinalArtifacts,
    Melody,
    Progression,
    RunContext,
)


class ResultPacker(ABC):
    """Слой 5 — финальная сборка.

    Реализация на сегодня одна. Если какой-то модели понадобится
    писать дополнительные артефакты — появится per-model подкласс.

    output_root и параметры стандартизации (program единого тембра,
    имя инструмента, имена директорий) — в state экземпляра.
    """

    @abstractmethod
    def pack(
        self,
        melody: Melody,
        progression: Progression,
        ctx: RunContext,
    ) -> FinalArtifacts: ...


# 66 = Tenor Sax (GM, 0-indexed pretty_midi). Единый тембр для всех моделей —
# тогда на слух (и при расчёте features) тембр не вмешивается в сравнение.
_DEFAULT_MELODY_PROGRAM = 66
_DEFAULT_MELODY_NAME = "Melody"
_MELODY_DIR = "melody_only"
_WITH_CHORDS_DIR = "with_chords"


class DefaultResultPacker(ResultPacker):
    """Общая реализация ResultPacker.

    Пишет два MIDI:
      - <output_root>/melody_only/<model_label>_<run_id>.mid — только мелодия
        с единым тембром.
      - <output_root>/with_chords/<model_label>_<run_id>.mid — мелодия +
        chord-track из исходного Progression.

    Если какой-то модели понадобятся дополнительные артефакты — появится
    per-model подкласс, эта реализация не правится.
    """

    def __init__(
        self,
        output_root: Path | str,
        melody_program: int = _DEFAULT_MELODY_PROGRAM,
        melody_instrument_name: str = _DEFAULT_MELODY_NAME,
    ) -> None:
        self._output_root = Path(output_root)
        self._melody_program = melody_program
        self._melody_instrument_name = melody_instrument_name

    def pack(
        self,
        melody: Melody,
        progression: Progression,
        ctx: RunContext,
    ) -> FinalArtifacts:
        melody_dir = self._output_root / _MELODY_DIR
        chords_dir = self._output_root / _WITH_CHORDS_DIR
        melody_dir.mkdir(parents=True, exist_ok=True)
        chords_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{ctx.model_label}_{ctx.run_id}.mid"
        melody_path = melody_dir / filename
        chords_path = chords_dir / filename

        pm_melody = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
        pm_melody.instruments.append(self._melody_instrument(melody))
        pm_melody.write(str(melody_path))

        pm_full = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
        pm_full.instruments.append(self._melody_instrument(melody))
        pm_full.instruments.append(build_chord_instrument(progression))
        pm_full.write(str(chords_path))

        return FinalArtifacts(melody_only=melody_path, with_chords=chords_path)

    def _melody_instrument(self, melody: Melody) -> pretty_midi.Instrument:
        inst = pretty_midi.Instrument(
            program=self._melody_program,
            name=self._melody_instrument_name,
            is_drum=False,
        )
        for n in melody.notes:
            inst.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end,
            ))
        return inst
