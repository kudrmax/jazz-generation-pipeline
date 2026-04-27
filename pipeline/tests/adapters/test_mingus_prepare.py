from pathlib import Path

import pytest

from pipeline.progression import ChordProgression
from pipeline.adapters.mingus import MingusAdapter, MingusPipelineConfig


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_prepare_returns_required_keys(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    for key in ["input_xml_path", "output_midi_path", "checkpoint_epochs", "temperature", "device", "model_repo_path"]:
        assert key in params, f"missing key: {key}"


def test_prepare_writes_input_xml(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    assert Path(params["input_xml_path"]).exists()
    assert Path(params["input_xml_path"]).suffix == ".xml"


def test_prepare_output_midi_path_in_tmp_dir(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    assert str(params["output_midi_path"]).startswith(str(tmp_path))
    assert params["output_midi_path"].endswith(".mid")


def test_prepare_passes_through_temperature_and_device(tmp_path: Path):
    cfg = MingusPipelineConfig(
        seed_strategy="tonic_whole", checkpoint_epochs=100,
        temperature=0.7, device="cpu",
    )
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    assert params["temperature"] == 0.7
    assert params["device"] == "cpu"
    assert params["checkpoint_epochs"] == 100


def test_prepare_does_not_leak_pipeline_concepts(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_quarters", checkpoint_epochs=100)
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    forbidden = {"seed_strategy", "run_id", "model_name", "progression"}
    leaked = forbidden & params.keys()
    assert not leaked, f"adapter leaked pipeline concepts to runner params: {leaked}"


def test_mingus_adapter_requires_config():
    """После рефакторинга MingusAdapter() без аргументов должен падать TypeError."""
    with pytest.raises(TypeError):
        MingusAdapter()  # type: ignore[call-arg]
