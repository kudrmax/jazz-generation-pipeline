from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


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
