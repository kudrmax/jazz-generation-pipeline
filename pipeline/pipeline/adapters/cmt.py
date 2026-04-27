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
        import numpy as np
        import yaml

        from pipeline.adapters._cmt_input import build_seed, progression_to_chroma

        cfg = self._config

        # 1. Прочитать гипер-параметры модели из hparams.yaml.
        with open(cfg.hparams_path, "r") as f:
            hparams = yaml.safe_load(f)
        model_cfg = hparams["model"]
        frame_per_bar: int = int(model_cfg["frame_per_bar"])
        num_bars: int = int(model_cfg["num_bars"])

        # 2. Pipeline-уровневая валидация (то что зависит от нашего конфига).
        valid_strategies = {"tonic_held", "tonic_quarters", "custom_pkl"}
        if cfg.seed_strategy not in valid_strategies:
            raise ValueError(
                f"unknown seed_strategy={cfg.seed_strategy!r}; "
                f"expected one of {sorted(valid_strategies)}"
            )
        if cfg.seed_strategy == "custom_pkl" and cfg.custom_pkl_path is None:
            raise ValueError("seed_strategy=custom_pkl requires custom_pkl_path")
        if cfg.prime_bars < 1 or cfg.prime_bars > num_bars:
            raise ValueError(
                f"prime_bars must be in [1, num_bars={num_bars}], got {cfg.prime_bars}"
            )

        # 3. Конвертация и затравка (внутри валит ValueError если progression
        #    несовместима с frame_per_bar / num_bars).
        chroma = progression_to_chroma(progression, frame_per_bar, num_bars)
        prime_len = cfg.prime_bars * frame_per_bar
        prime_pitch, prime_rhythm = build_seed(progression, cfg, frame_per_bar, prime_len)

        # 4. Сохранение seed.npz и возврат params.
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        seed_npz = tmp_dir / "seed.npz"
        np.savez(seed_npz, chord_chroma=chroma, prime_pitch=prime_pitch, prime_rhythm=prime_rhythm)

        return {
            "checkpoint_path":   str(cfg.checkpoint_path),
            "hparams_path":      str(cfg.hparams_path),
            "model_repo_path":   str(cfg.repo_path),
            "seed_npz_path":     str(seed_npz),
            "output_midi_path":  str(tmp_dir / "raw.mid"),
            "topk":              cfg.topk,
            "device":            cfg.device,
        }

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        raise NotImplementedError("model cmt: extract_melody not implemented yet")
