from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pretty_midi

from pipeline.adapters.base import ModelAdapter
from pipeline.progression import ChordProgression


@dataclass
class BebopNetPipelineConfig:
    """Все настройки BebopNet на уровне pipeline. immutable после init.

    Замена весов на другую папку с (model.pt, converter_and_duration.pkl,
    args.json, train_model.yml) = подмена model_dir / checkpoint_filename
    без правок этого dataclass.
    """

    model_dir: Path
    repo_path: Path
    checkpoint_filename: str = "model.pt"
    seed_strategy: Literal["tonic_whole", "tonic_quarters", "custom_xml"] = "tonic_whole"
    custom_xml_path: Path | None = None
    melody_instrument_name: str = "Tenor Sax"
    temperature: float = 1.0
    top_p: bool = True
    beam_search: Literal["", "note", "measure"] = "measure"
    beam_width: int = 2
    device: str = "cpu"


class BebopNetAdapter(ModelAdapter):
    def __init__(self, config: BebopNetPipelineConfig) -> None:
        self._config = config

    def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
        raise NotImplementedError("model bebopnet: prepare not implemented yet")

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        raise NotImplementedError("model bebopnet: extract_melody not implemented yet")
