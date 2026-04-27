from pathlib import Path

import pytest

from pipeline.adapters.bebopnet import BebopNetAdapter, BebopNetPipelineConfig


def _make_config(tmp_path: Path) -> BebopNetPipelineConfig:
    return BebopNetPipelineConfig(
        model_dir=tmp_path / "model",
        repo_path=tmp_path / "repo",
    )


def test_bebopnet_adapter_requires_config():
    with pytest.raises(TypeError):
        BebopNetAdapter()  # type: ignore[call-arg]


def test_bebopnet_adapter_stores_config(tmp_path: Path):
    cfg = _make_config(tmp_path)
    adapter = BebopNetAdapter(cfg)
    assert adapter._config is cfg


def test_bebopnet_pipeline_config_defaults(tmp_path: Path):
    cfg = _make_config(tmp_path)
    assert cfg.checkpoint_filename == "model.pt"
    assert cfg.seed_strategy == "tonic_whole"
    assert cfg.custom_xml_path is None
    assert cfg.melody_instrument_name == "Tenor Sax"
    assert cfg.temperature == 1.0
    assert cfg.top_p is True
    assert cfg.beam_search == "measure"
    assert cfg.beam_width == 2
    assert cfg.device == "cpu"
