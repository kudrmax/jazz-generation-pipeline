from __future__ import annotations

from pathlib import Path

from pipeline.adapters.base import ModelAdapter
from pipeline.adapters.bebopnet import BebopNetAdapter
from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig
from pipeline.adapters.commu import ComMUAdapter
from pipeline.adapters.ec2vae import EC2VaeAdapter
from pipeline.adapters.mingus import MingusAdapter, MingusPipelineConfig
from pipeline.adapters.polyffusion import PolyffusionAdapter


DIPLOMA_ROOT: Path = Path(__file__).resolve().parents[2]
PIPELINE_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_ROOT: Path = PIPELINE_ROOT / "output"
RUNNERS_ROOT: Path = PIPELINE_ROOT / "runners"

MINGUS_REPO_PATH: Path = DIPLOMA_ROOT / "models" / "MINGUS"
CMT_REPO_PATH: Path = DIPLOMA_ROOT / "models" / "CMT-pytorch"

MODEL_NAMES: list[str] = ["mingus", "bebopnet", "ec2vae", "cmt", "commu", "polyffusion"]

MODEL_VENV_PYTHON: dict[str, Path] = {
    "mingus":      DIPLOMA_ROOT / "models/MINGUS/.venv/bin/python",
    "bebopnet":    DIPLOMA_ROOT / "models/bebopnet-code/.venv/bin/python",
    "ec2vae":      DIPLOMA_ROOT / "models/EC2-VAE/.venv/bin/python",
    "cmt":         DIPLOMA_ROOT / "models/CMT-pytorch/.venv/bin/python",
    "commu":       DIPLOMA_ROOT / "models/ComMU-code/.venv/bin/python",
    "polyffusion": DIPLOMA_ROOT / "models/polyffusion/.venv/bin/python",
}

MODEL_RUNNER_SCRIPT: dict[str, Path] = {
    "mingus":      RUNNERS_ROOT / "mingus_runner.py",
    # остальные runner-скрипты появляются вместе с реализацией модели
}

ADAPTERS: dict[str, ModelAdapter] = {
    "mingus":      MingusAdapter(MingusPipelineConfig(
        seed_strategy="tonic_whole",
        temperature=1.0,
        device="cpu",
        checkpoint_epochs=100,
    )),
    "bebopnet":    BebopNetAdapter(),
    "ec2vae":      EC2VaeAdapter(),
    "cmt":         CMTAdapter(CMTPipelineConfig(
        checkpoint_path=CMT_REPO_PATH / "result" / "smoke_wjazzd_5epochs" / "model.pth.tar",
        hparams_path=CMT_REPO_PATH / "hparams.yaml",
        repo_path=CMT_REPO_PATH,
    )),
    "commu":       ComMUAdapter(),
    "polyffusion": PolyffusionAdapter(),
}

# Общее pipeline-решение: монофонная мелодия в melody_only.mid пишется единым
# тембром у всех моделей — тогда на слух (и в feature-расчёте метрик)
# тембр не вмешивается в сравнение. 66 = Tenor Sax (GM, 0-indexed pretty_midi),
# исторически бэйслайны (BebopNet/MINGUS) как раз и пишут саксофоном.
MELODY_PROGRAM: int = 66

RUNNER_TIMEOUT_SEC: int = 600
