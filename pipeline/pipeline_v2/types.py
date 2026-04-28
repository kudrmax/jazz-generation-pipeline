"""Типы данных pipeline_v2.

Все frozen dataclass-ы и Enum-ы, на которые опираются абстрактные слои.
Поля у Progression / Melody / ModelSpec пока пусты — наполним когда
будем реализовывать конкретную модель.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ModelName(str, Enum):
    """Директива оркестратору: какую модель запустить.

    Не часть PipelineInput — это управляющий параметр, а не данные
    замысла. Используется только в оркестраторе. В слой 5 не пробрасывается:
    packer работает с непрозрачной строкой ctx.model_label.
    """

    MINGUS = "mingus"
    BEBOPNET = "bebopnet"
    CMT = "cmt"


@dataclass(frozen=True)
class Progression:
    """Замысел: аккорды + темп + размер. Поля наполним позже."""


@dataclass(frozen=True)
class Melody:
    """Монофонная мелодия. Один тип на две роли:
    опциональная затравка на входе и итоговая мелодия на выходе.
    Поля наполним позже.
    """


@dataclass(frozen=True)
class PipelineInput:
    """То что пользователь принёс. Только данные замысла, без управляющих
    параметров (model_name живёт в оркестраторе, не здесь).
    """

    progression: Progression
    theme: Melody | None


class ModelSpec(ABC):
    """Описание возможностей модели — "вшитые" свойства чекпоинта и
    архитектуры (словарь аккордов, допустимый размер такта, ограничения
    по длине, требования к теме). Не runtime-настройки.

    Подклассы (MingusSpec, BebopNetSpec, CMTSpec) — frozen dataclass-ы
    со своими полями. Базовый класс пуст: общая база только запутает.

    На ModelSpec одной модели смотрят оба слоя — валидатор (2) и
    генератор (3). Реестр обязан передать им один и тот же инстанс.
    """


@dataclass(frozen=True)
class RunContext:
    """Состояние одного запуска. Создаётся оркестратором, пробрасывается
    в слои, работающие с диском (3, 4, 5).

    model_label — короткая строка для имён файлов и логов (типа "mingus",
    "bebopnet"). Это непрозрачный для слоя 5 префикс: packer не должен
    знать enum ModelName, ему достаточно строки.
    """

    run_id: str
    tmp_dir: Path
    model_label: str


@dataclass(frozen=True)
class RawOutput:
    """Сырой выход модели после subprocess. Контракт между слоем 3 и слоем 4.

    Все три модели сейчас пишут MIDI на диск, поэтому минимально храним
    Path. Если завтра потребуется хранить дополнительные артефакты —
    расширяем dataclass.
    """

    path: Path


@dataclass(frozen=True)
class FinalArtifacts:
    """Итоговые артефакты одного запуска: пара MIDI-файлов."""

    melody_only: Path  # чистая мелодия с единым тембром
    with_chords: Path  # мелодия + chord-track из исходной прогрессии
