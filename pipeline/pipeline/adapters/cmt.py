from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pretty_midi

from pipeline.adapters.base import ModelAdapter
from pipeline.progression import ChordProgression


@dataclass
class CMTPipelineConfig:
    """Все настройки CMT на уровне pipeline. immutable после init.

    Размеры модели (frame_per_bar, num_bars, num_pitch) НЕ хранятся здесь —
    они читаются из hparams.yaml в момент prepare(). Замена весов на
    другие гипер-параметры = подмена пары (checkpoint_path, hparams_path)
    без правок этого dataclass.
    """

    checkpoint_path: Path
    hparams_path: Path
    repo_path: Path
    seed_strategy: Literal["tonic_held", "tonic_quarters", "custom_pkl"] = "tonic_held"
    custom_pkl_path: Path | None = None
    prime_bars: int = 1
    topk: int = 5
    device: str = "cpu"


class CMTAdapter(ModelAdapter):
    def __init__(self, config: CMTPipelineConfig) -> None:
        self._config = config

    def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
        raise NotImplementedError("model cmt: prepare not implemented yet")

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        raise NotImplementedError("model cmt: extract_melody not implemented yet")
