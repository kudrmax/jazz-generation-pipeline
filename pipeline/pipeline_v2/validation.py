"""Слой 2 — валидация замысла под модель.

Возвращает ValidationResult со всеми найденными проблемами разом, не
первую попавшуюся. Не правит вход, не строит файлов, не обращается к
чекпоинту и subprocess.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
