"""Слой 5 — финальная сборка.

Принимает уже извлечённую Melody и исходный Progression. Накладывает
единый тембр, строит chord-track, пишет два MIDI в стандартные
директории. Имена файлов собираются из ctx.model_label и ctx.run_id —
packer не знает про enum ModelName.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

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
