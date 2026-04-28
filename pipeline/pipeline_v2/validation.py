"""Слой 2 — валидация замысла под модель.

Возвращает ValidationResult со всеми найденными проблемами разом, не
первую попавшуюся. Не правит вход, не строит файлов, не обращается к
чекпоинту и subprocess.

Здесь же живёт CommonInputValidator — pipeline-уровневая валидация,
не зависит от модели. Зовётся оркестратором ПЕРЕД per-model валидатором.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pipeline_v2.chord_vocab import parse_chord
from pipeline_v2.types import PipelineInput


@dataclass(frozen=True)
class ValidationError:
    """Одна претензия к замыслу: что не подошло и почему."""

    code: str       # короткий идентификатор причины (e.g. "unsupported_chord")
    message: str    # человекочитаемое объяснение


@dataclass(frozen=True)
class ValidationResult:
    errors: list[ValidationError]

    @property
    def ok(self) -> bool:
        return not self.errors


class ValidationFailedError(Exception):
    """Поднимается оркестратором, если валидатор вернул непустой список
    ошибок. Несёт все претензии.
    """

    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"[{e.code}] {e.message}" for e in errors))


class InputValidator(ABC):
    """Слой 2 — валидация замысла под конкретную модель.

    Реализации: MingusInputValidator, BebopNetInputValidator,
    CMTInputValidator. Каждая опирается на свой ModelSpec (получает
    его через __init__).
    """

    @abstractmethod
    def validate(self, pipeline_input: PipelineInput) -> ValidationResult: ...


class CommonInputValidator(ABC):
    """Pipeline-уровневая валидация, не зависит от модели.

    Зовётся оркестратором ПЕРЕД per-model валидатором (InputValidator).
    Если возвращает непустой errors — pipeline обрывается, per-model
    валидатор не зовётся.

    На сегодня одна реализация (общая для всех моделей). ABC введён
    для единообразия и тестируемости — мокать в тестах удобнее.
    """

    @abstractmethod
    def validate(self, pipeline_input: PipelineInput) -> ValidationResult: ...


_DEFAULT_MIN_TEMPO = 40.0
_DEFAULT_MAX_TEMPO = 300.0


class DefaultCommonInputValidator(CommonInputValidator):
    """Стандартная pipeline-уровневая валидация.

    Что проверяется:
    - все аккорды парсятся chord_vocab.parse_chord (root + одно из
      7 поддерживаемых качеств);
    - прогрессия не пуста;
    - длительности всех аккордов > 0;
    - сумма долей делится на beats_per_bar (укладывается в целое
      число баров);
    - темп в [min_tempo, max_tempo];
    - размер такта в формате "N/M" с положительными N и M.

    Что НЕ проверяется (это per-model):
    - конкретный размер такта (4/4 only — это для MingusValidator);
    - кратность длительности аккорда бару (Mingus);
    - длина прогрессии под конкретный чекпоинт (CMT).
    """

    def __init__(
        self,
        min_tempo: float = _DEFAULT_MIN_TEMPO,
        max_tempo: float = _DEFAULT_MAX_TEMPO,
    ) -> None:
        if min_tempo <= 0 or max_tempo < min_tempo:
            raise ValueError(
                f"invalid tempo range: min={min_tempo}, max={max_tempo}"
            )
        self._min_tempo = min_tempo
        self._max_tempo = max_tempo

    def validate(self, pipeline_input: PipelineInput) -> ValidationResult:
        prog = pipeline_input.progression
        errors: list[ValidationError] = []

        if not prog.chords:
            errors.append(ValidationError(
                code="empty_progression",
                message="progression has no chords",
            ))

        for i, (chord_str, beats) in enumerate(prog.chords):
            try:
                parse_chord(chord_str)
            except ValueError as e:
                errors.append(ValidationError(
                    code="unsupported_chord",
                    message=f"chords[{i}]={chord_str!r}: {e}",
                ))
            if beats <= 0:
                errors.append(ValidationError(
                    code="non_positive_duration",
                    message=(
                        f"chords[{i}] {chord_str!r}: beats={beats} must be > 0"
                    ),
                ))

        beats_per_bar = self._parse_time_signature(prog.time_signature, errors)
        if beats_per_bar is not None:
            total_beats = sum(b for _, b in prog.chords)
            if total_beats > 0 and total_beats % beats_per_bar != 0:
                errors.append(ValidationError(
                    code="not_full_bars",
                    message=(
                        f"total beats {total_beats} not divisible by "
                        f"beats_per_bar {beats_per_bar} "
                        f"(time_signature={prog.time_signature!r})"
                    ),
                ))

        if not (self._min_tempo <= prog.tempo <= self._max_tempo):
            errors.append(ValidationError(
                code="tempo_out_of_range",
                message=(
                    f"tempo={prog.tempo} not in "
                    f"[{self._min_tempo}, {self._max_tempo}]"
                ),
            ))

        return ValidationResult(errors=errors)

    @staticmethod
    def _parse_time_signature(
        ts: str, errors: list[ValidationError],
    ) -> int | None:
        parts = ts.split("/")
        if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
            errors.append(ValidationError(
                code="bad_time_signature",
                message=(
                    f"time_signature={ts!r} must be 'N/M' with positive ints"
                ),
            ))
            return None
        return int(parts[0])
