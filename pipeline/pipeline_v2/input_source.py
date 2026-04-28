"""Слой 1 — приём входа.

Реализации (JSON-файл, CLI-аргументы, dict) превращают сырой вход
в типизированный PipelineInput. Источник в state экземпляра, поэтому
load() без аргументов.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from pipeline_v2.types import PipelineInput, Progression


class InputSource(ABC):
    """Слой 1 — приём входа.

    Не валидирует семантику (это слой 2). Не строит файлов для моделей
    (это слой 3). Не знает, какую модель пользователь выбрал — model_name
    не часть данных, это директива оркестратора.
    """

    @abstractmethod
    def load(self) -> PipelineInput: ...


_REQUIRED_KEYS = {"chords", "tempo", "time_signature"}
_ALLOWED_KEYS = _REQUIRED_KEYS | {"theme"}


class JsonInputSource(InputSource):
    """JSON-файл → PipelineInput.

    Формат:
        {
          "tempo": 120.0,
          "time_signature": "4/4",
          "chords": [["Cmaj7", 4], ["Am7", 4], ...],
          "theme": null
        }

    "theme" — опциональное поле под опциональную затравку. Сейчас
    парсится только null; конкретный формат добавим, когда будем
    реализовывать первую модель, которой тема нужна.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def load(self) -> PipelineInput:
        if not self._path.exists():
            raise FileNotFoundError(f"input file not found: {self._path}")

        try:
            data = json.loads(self._path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON in {self._path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(
                f"expected JSON object at top level in {self._path}, "
                f"got {type(data).__name__}"
            )

        unknown = set(data.keys()) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"unknown fields in {self._path}: {sorted(unknown)}; "
                f"allowed: {sorted(_ALLOWED_KEYS)}"
            )
        missing = _REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"missing required fields in {self._path}: {sorted(missing)}"
            )

        chords = self._parse_chords(data["chords"])
        tempo = self._parse_tempo(data["tempo"])
        time_signature = self._parse_time_signature(data["time_signature"])

        progression = Progression(
            chords=chords, tempo=tempo, time_signature=time_signature,
        )

        if data.get("theme") is not None:
            raise NotImplementedError(
                "theme parsing not implemented yet; pass null or omit"
            )

        return PipelineInput(progression=progression, theme=None)

    @staticmethod
    def _parse_chords(raw: object) -> tuple[tuple[str, int], ...]:
        if not isinstance(raw, list):
            raise ValueError(f"'chords' must be a list, got {type(raw).__name__}")
        out: list[tuple[str, int]] = []
        for i, item in enumerate(raw):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    f"chords[{i}] must be a [name, beats] pair, got {item!r}"
                )
            name, beats = item
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"chords[{i}][0] must be a non-empty string, got {name!r}"
                )
            if not isinstance(beats, int) or beats <= 0:
                raise ValueError(
                    f"chords[{i}][1] must be a positive int, got {beats!r}"
                )
            out.append((name, beats))
        if not out:
            raise ValueError("'chords' must contain at least one chord")
        return tuple(out)

    @staticmethod
    def _parse_tempo(raw: object) -> float:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"'tempo' must be a number, got {raw!r}")
        tempo = float(raw)
        if tempo <= 0:
            raise ValueError(f"'tempo' must be > 0, got {tempo}")
        return tempo

    @staticmethod
    def _parse_time_signature(raw: object) -> str:
        if not isinstance(raw, str):
            raise ValueError(
                f"'time_signature' must be a string like '4/4', got {raw!r}"
            )
        parts = raw.split("/")
        if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
            raise ValueError(
                f"'time_signature' must be 'N/M' with positive ints, got {raw!r}"
            )
        return raw
