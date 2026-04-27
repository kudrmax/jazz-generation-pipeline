from pathlib import Path

import pytest

from pipeline.adapters.bebopnet import BebopNetAdapter, BebopNetPipelineConfig
from pipeline.progression import ChordProgression


def _cfg(
    tmp_path: Path,
    *,
    seed_strategy: str = "tonic_whole",
    custom_xml_path: Path | None = None,
) -> BebopNetPipelineConfig:
    return BebopNetPipelineConfig(
        model_dir=tmp_path / "model",
        repo_path=tmp_path / "repo",
        seed_strategy=seed_strategy,  # type: ignore[arg-type]
        custom_xml_path=custom_xml_path,
    )


def _4bars_4_4() -> ChordProgression:
    return ChordProgression(chords=[("Cmaj7", 4)] * 4, tempo=120.0, time_signature="4/4")


def test_validation_unknown_seed_strategy(tmp_path: Path):
    cfg = _cfg(tmp_path, seed_strategy="random_walk")
    with pytest.raises(ValueError, match="seed_strategy"):
        BebopNetAdapter(cfg).prepare(_4bars_4_4(), tmp_path / "work")


def test_validation_custom_xml_without_path(tmp_path: Path):
    cfg = _cfg(tmp_path, seed_strategy="custom_xml", custom_xml_path=None)
    with pytest.raises(ValueError, match="custom_xml_path"):
        BebopNetAdapter(cfg).prepare(_4bars_4_4(), tmp_path / "work")


def test_validation_empty_progression(tmp_path: Path):
    cfg = _cfg(tmp_path)
    empty = ChordProgression(chords=[], tempo=120.0, time_signature="4/4")
    with pytest.raises(ValueError, match="chords|empty|no chord"):
        BebopNetAdapter(cfg).prepare(empty, tmp_path / "work")
