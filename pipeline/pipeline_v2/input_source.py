"""Слой 1 — приём входа.

Реализации (JSON-файл, CLI-аргументы, dict) превращают сырой вход
в типизированный PipelineInput. Источник в state экземпляра, поэтому
load() без аргументов.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline_v2.types import PipelineInput


class InputSource(ABC):
    """Слой 1 — приём входа.

    Не валидирует семантику (это слой 2). Не строит файлов для моделей
    (это слой 3). Не знает, какую модель пользователь выбрал — model_name
    не часть данных, это директива оркестратора.
    """

    @abstractmethod
    def load(self) -> PipelineInput: ...
