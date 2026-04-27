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
        from pipeline._xml_builders.jazz_xml import build_xml

        self._validate(progression)
        cfg = self._config
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        xml_path = tmp_dir / "input.xml"
        midi_path = tmp_dir / "raw.mid"

        build_xml(
            progression,
            seed_strategy=cfg.seed_strategy,
            custom_xml_path=cfg.custom_xml_path,
            out_path=xml_path,
            melody_instrument_name=cfg.melody_instrument_name,
        )

        return {
            "input_xml_path":      str(xml_path),
            "output_midi_path":    str(midi_path),
            "model_dir":           str(cfg.model_dir),
            "checkpoint_filename": cfg.checkpoint_filename,
            "model_repo_path":     str(cfg.repo_path),
            "num_measures":        progression.num_bars(),
            "temperature":         cfg.temperature,
            "top_p":               cfg.top_p,
            "beam_search":         cfg.beam_search,
            "beam_width":          cfg.beam_width,
            "device":              cfg.device,
        }

    def _validate(self, progression: ChordProgression) -> None:
        cfg = self._config
        valid_strategies = {"tonic_whole", "tonic_quarters", "custom_xml"}
        if cfg.seed_strategy not in valid_strategies:
            raise ValueError(
                f"unknown seed_strategy={cfg.seed_strategy!r}; "
                f"expected one of {sorted(valid_strategies)}"
            )
        if cfg.seed_strategy == "custom_xml" and cfg.custom_xml_path is None:
            raise ValueError("seed_strategy=custom_xml requires custom_xml_path")
        if not progression.chords:
            raise ValueError("progression has no chords")

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        pm = pretty_midi.PrettyMIDI(str(raw_midi_path))
        if not pm.instruments:
            raise ValueError(
                f"no instruments in {raw_midi_path}; cannot extract melody"
            )
        return pm.instruments[0]
