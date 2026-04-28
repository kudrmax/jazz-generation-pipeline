"""Оркестратор — точка входа pipeline.

Принимает в __init__: InputSource, ModelName, ModelRegistry, ResultPacker.
В run() создаёт RunContext и гонит шаги 1→5 в последовательности.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline_v2.types import FinalArtifacts


class Orchestrator(ABC):
    """Точка входа pipeline. Гонит шаги 1→5.

    В run() создаёт RunContext (run_id, tmp_dir, model_label) и
    исполняет поток так, что PipelineInput жив до самого шага 5
    (его поля нужны и для генерации, и для финальной сборки):

        inp     = source.load()                          # шаг 1
        bundle  = registry.get(model_name)               # выбор реализаций
        result  = bundle.validator.validate(inp)         # шаг 2
        if not result.ok:
            raise ValidationFailedError(result.errors)
        raw     = bundle.generator.generate(inp, ctx)    # шаг 3
        melody  = bundle.extractor.extract(raw)          # шаг 4
        return packer.pack(
            melody, inp.progression, ctx,                # шаг 5
        )

    PipelineInput намеренно не пересоздаётся между шагами и не
    мутируется: один объект на весь запуск, читается слоями 2, 3, 5.
    """

    @abstractmethod
    def run(self) -> FinalArtifacts: ...
