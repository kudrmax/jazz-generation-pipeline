"""Слой 4 — извлечение мелодии из сырого выхода модели.

Реализации знают, как из RawOutput модели выделить монофонную
мелодическую линию (по имени трека, по индексу, проекцией полифонии).
Возвращают Melody — наш единый внутренний тип.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline_v2.types import Melody, RawOutput


class MelodyExtractor(ABC):
    """Слой 4 — извлечение мелодии.

    Реализации: MingusMelodyExtractor, BebopNetMelodyExtractor,
    CMTMelodyExtractor. Параметры (имя трека, индекс) — в state
    экземпляра через __init__.

    Не накладывает финальный тембр и не строит chord-track — это слой 5.
    """

    @abstractmethod
    def extract(self, raw: RawOutput) -> Melody: ...
