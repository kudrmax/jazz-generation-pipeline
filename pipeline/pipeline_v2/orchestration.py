"""Оркестратор — точка входа pipeline.

Принимает в __init__: InputSource, ModelName, ModelRegistry, ResultPacker.
В run() создаёт RunContext и гонит шаги 1→5 в последовательности.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from pipeline_v2.input_source import InputSource
from pipeline_v2.packing import ResultPacker
from pipeline_v2.registry import ModelRegistry
from pipeline_v2.types import FinalArtifacts, ModelName, Progression, RunContext
from pipeline_v2.validation import CommonInputValidator, ValidationFailedError


class Orchestrator(ABC):
    """Точка входа pipeline. Гонит шаги 1→5.

    В __init__ принимает: InputSource, CommonInputValidator, ModelName,
    ModelRegistry, ResultPacker. В run() создаёт RunContext
    (run_id, tmp_dir, model_label) и исполняет поток:

        inp     = source.load()                            # шаг 1
        common  = common_validator.validate(inp)           # шаг 2.A
        if not common.ok:
            raise ValidationFailedError(common.errors)
        bundle  = registry.get(model_name)
        per_m   = bundle.validator.validate(inp)           # шаг 2.Б
        if not per_m.ok:
            raise ValidationFailedError(per_m.errors)
        raw     = bundle.generator.generate(inp, ctx)      # шаг 3
        melody  = bundle.extractor.extract(raw)            # шаг 4
        return packer.pack(melody, inp.progression, ctx)   # шаг 5

    Шаги 2.A и 2.Б НЕ объединяются: если общий валидатор отказал,
    per-model даже не зовётся. tmp_dir создаётся только после успешной
    валидации, чтобы не оставлять мусор при отказе.

    PipelineInput не мутируется: один объект на весь запуск.
    """

    @abstractmethod
    def run(self) -> FinalArtifacts: ...


def _progression_hash(progression: Progression) -> str:
    """Детерминированный 8-символьный hash от композиционного замысла.

    Используется как часть run_id: одна и та же прогрессия даёт один
    и тот же суффикс, что упрощает отслеживание выходов разных моделей
    на одном замысле.
    """
    payload = repr((
        progression.chords,
        progression.tempo,
        progression.time_signature,
    )).encode()
    return hashlib.sha256(payload).hexdigest()[:8]


def _make_run_id(progression: Progression) -> str:
    """Run-id формат: YYYYMMDD-HHMMSS-<8charhash>."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_progression_hash(progression)}"


class DefaultOrchestrator(Orchestrator):
    """Стандартная реализация оркестратора.

    Двухэтапная валидация (общая + per-model) выполняется ДО создания
    tmp_dir, чтобы при отказе валидатора не оставалось мусора на диске.
    """

    def __init__(
        self,
        input_source: InputSource,
        common_validator: CommonInputValidator,
        model_name: ModelName,
        registry: ModelRegistry,
        packer: ResultPacker,
        tmp_root: Path | str,
    ) -> None:
        self._input_source = input_source
        self._common_validator = common_validator
        self._model_name = model_name
        self._registry = registry
        self._packer = packer
        self._tmp_root = Path(tmp_root)

    def run(self) -> FinalArtifacts:
        inp = self._input_source.load()                       # шаг 1

        common_result = self._common_validator.validate(inp)  # шаг 2.A
        if not common_result.ok:
            raise ValidationFailedError(common_result.errors)

        bundle = self._registry.get(self._model_name)

        per_model_result = bundle.validator.validate(inp)     # шаг 2.Б
        if not per_model_result.ok:
            raise ValidationFailedError(per_model_result.errors)

        run_id = _make_run_id(inp.progression)
        tmp_dir = self._tmp_root / run_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(
            run_id=run_id,
            tmp_dir=tmp_dir,
            model_label=self._model_name.value,
        )

        raw = bundle.generator.generate(inp, ctx)             # шаг 3
        melody = bundle.extractor.extract(raw)                # шаг 4
        return self._packer.pack(melody, inp.progression, ctx)  # шаг 5
