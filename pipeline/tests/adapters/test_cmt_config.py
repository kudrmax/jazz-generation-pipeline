from pathlib import Path
import pytest
from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig


def _make_config(tmp_path: Path) -> CMTPipelineConfig:
    return CMTPipelineConfig(
        checkpoint_path=tmp_path / "ckpt.pth.tar",
        hparams_path=tmp_path / "hparams.yaml",
        repo_path=tmp_path / "repo",
    )


def test_cmt_adapter_requires_config():
    with pytest.raises(TypeError):
        CMTAdapter()  # type: ignore[call-arg]


def test_cmt_adapter_stores_config(tmp_path: Path):
    cfg = _make_config(tmp_path)
    adapter = CMTAdapter(cfg)
    assert adapter._config is cfg


def test_cmt_pipeline_config_defaults(tmp_path: Path):
    cfg = _make_config(tmp_path)
    assert cfg.seed_strategy == "tonic_held"
    assert cfg.custom_seed_path is None
    assert cfg.prime_bars == 1
    assert cfg.topk == 5
    assert cfg.device == "cpu"
