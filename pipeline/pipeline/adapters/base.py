from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pretty_midi

from pipeline.progression import ChordProgression


class ModelAdapter(ABC):
    """Toolbox для одной модели — обе границы между pipeline и моделью."""

    @abstractmethod
    def prepare(
        self,
        progression: ChordProgression,
        tmp_dir: Path,
    ) -> dict:
        """Из progression собирает params для runner'а
        (включая физическую подготовку входных файлов в tmp_dir, если нужно).

        Конфигурация модели хранится в state экземпляра adapter'а (из __init__),
        а не передаётся в prepare. Это позволяет config быть immutable после
        инициализации.

        Возвращает словарь, который pipeline пробрасывает runner'у через JSON stdin.
        """

    @abstractmethod
    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        """Из сырого выхода модели возвращает монофонную мелодию как pretty_midi.Instrument.

        Контракт: возвращаемый Instrument МОЖЕТ быть сконструирован adapter'ом из чего
        угодно — track'а MIDI (MINGUS, BebopNet), highest-pitch проекции полифонии
        (Polyffusion), декодированного pianoroll (EC²-VAE), декодированных REMI-токенов
        (ComMU). Pipeline не предполагает «у модели есть готовый melody-track» —
        adapter сам отвечает за получение монофонной мелодии в нужном формате.
        """
