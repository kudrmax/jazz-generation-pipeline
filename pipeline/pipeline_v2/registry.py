"""DI-реестр и bundle.

ModelRegistry по ModelName отдаёт уже сконструированный ModelBundle
(тройка per-model реализаций). InputSource и ResultPacker сюда не
входят — они общие, живут в оркестраторе.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pipeline_v2.extraction import MelodyExtractor
from pipeline_v2.generation import Generator
from pipeline_v2.types import ModelName
from pipeline_v2.validation import InputValidator


@dataclass(frozen=True)
class ModelBundle:
    """Тройка per-model реализаций под одну модель.

    Условие сборки bundle: validator и generator получают **один и тот
    же** инстанс ModelSpec в свои __init__. Это контракт реестра, а не
    стилевая договорённость: модельное знание (словарь аккордов,
    размеры) должно быть согласованным между проверкой замысла и его
    последующим преобразованием.
    """

    validator: InputValidator
    generator: Generator
    extractor: MelodyExtractor


class ModelRegistry(ABC):
    """DI-реестр. По ModelName отдаёт ModelBundle.

    Как реестр собран (явный dict, фабрики, конфиг) — детали реализации.
    Контракт минимален: имя → bundle.
    """

    @abstractmethod
    def get(self, model_name: ModelName) -> ModelBundle: ...


class UnknownModelError(KeyError):
    """Поднимается реестром, если для запрошенного ModelName нет bundle."""


class DictModelRegistry(ModelRegistry):
    """Простейший реестр — словарь ModelName → ModelBundle.

    Всё знание о связи имени и реализаций задаётся снаружи при сборке
    (в фабрике или композиционном корне приложения). Сам реестр —
    тонкая обёртка над dict, без логики.
    """

    def __init__(self, bundles: dict[ModelName, ModelBundle]) -> None:
        self._bundles = dict(bundles)

    def get(self, model_name: ModelName) -> ModelBundle:
        try:
            return self._bundles[model_name]
        except KeyError as e:
            known = sorted(m.value for m in self._bundles)
            raise UnknownModelError(
                f"no bundle registered for {model_name.value!r}; "
                f"known: {known}"
            ) from e
