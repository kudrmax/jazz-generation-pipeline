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
