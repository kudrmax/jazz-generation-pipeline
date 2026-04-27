from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pretty_midi

from pipeline.adapters.base import ModelAdapter
from pipeline.progression import ChordProgression


@dataclass
class MingusPipelineConfig:
    """Все настройки MINGUS на уровне нашего пайплайна.

    Часть из них — стратегии подготовки входа (живут только здесь),
    часть — параметры MINGUS-API, которые мы пробрасываем в runner.
    """

    seed_strategy: Literal["tonic_whole", "tonic_quarters", "custom_xml"] = "tonic_whole"
    custom_xml_path: Path | None = None
    temperature: float = 1.0
    device: str = "cpu"
    # У MINGUS веса лежат как `MINGUS COND I-C-NC-B-BE-O Epochs <N>.pt`. Мы не передаём
    # path напрямую — runner сам собирает имена из этой числовой настройки.
    checkpoint_epochs: int = 100
    melody_instrument_name: str = "Tenor Sax"


class MingusAdapter(ModelAdapter):
    def prepare(
        self,
        progression: ChordProgression,
        config: MingusPipelineConfig,
        tmp_dir: Path,
    ) -> dict:
        from pipeline.config import MINGUS_REPO_PATH  # ленивый импорт чтобы избежать circular
        from pipeline._xml_builders.mingus_xml import build_mingus_xml  # circular: mingus_xml импортирует MingusPipelineConfig

        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        xml_path = tmp_dir / "input.xml"
        midi_path = tmp_dir / "raw.mid"
        build_mingus_xml(progression, config, xml_path)
        return {
            "input_xml_path": str(xml_path),
            "output_midi_path": str(midi_path),
            "checkpoint_epochs": config.checkpoint_epochs,
            "temperature": config.temperature,
            "device": config.device,
            "model_repo_path": str(MINGUS_REPO_PATH),
        }

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        # реализуется в Task 8
        raise NotImplementedError("MingusAdapter.extract_melody not yet implemented")
