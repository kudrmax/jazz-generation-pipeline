"""Слой 3 — генерация. Per-model монолит.

Реализации внутри себя:
    (а) готовят входной формат модели в tmp_dir (MusicXML, npz и т.п.);
    (б) запускают subprocess в venv модели;
    (в) возвращают RawOutput.

Эти три действия не вынесены в отдельные слои, потому что формат файла
и протокол subprocess — внутренние детали конкретной модели, не общий
контракт.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline_v2.types import PipelineInput, RawOutput, RunContext


class Generator(ABC):
    """Слой 3 — генерация.

    Реализации: MingusGenerator, BebopNetGenerator, CMTGenerator.
    ModelSpec и runtime-настройки (температура, beam_width, путь
    к чекпоинту) живут в state экземпляра — приходят через __init__.
    """

    @abstractmethod
    def generate(
        self,
        pipeline_input: PipelineInput,
        ctx: RunContext,
    ) -> RawOutput: ...
