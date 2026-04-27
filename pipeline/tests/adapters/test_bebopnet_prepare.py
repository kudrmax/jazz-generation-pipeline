from pathlib import Path

from pipeline.adapters.bebopnet import BebopNetAdapter, BebopNetPipelineConfig
from pipeline.progression import ChordProgression


def _cfg(tmp_path: Path) -> BebopNetPipelineConfig:
    return BebopNetPipelineConfig(
        model_dir=tmp_path / "model",
        repo_path=tmp_path / "repo",
    )


def _prog(first_chord: str = "Cmaj7", n_bars: int = 4) -> ChordProgression:
    chords = [(first_chord, 4)] + [("G7", 4)] * (n_bars - 1)
    return ChordProgression(chords=chords, tempo=120.0, time_signature="4/4")


def test_prepare_returns_required_keys(tmp_path: Path):
    work = tmp_path / "work"
    params = BebopNetAdapter(_cfg(tmp_path)).prepare(_prog(), work)
    for key in [
        "input_xml_path", "output_midi_path", "model_dir", "checkpoint_filename",
        "model_repo_path", "num_measures", "temperature", "top_p",
        "beam_search", "beam_width", "device",
    ]:
        assert key in params, f"missing: {key}"


def test_prepare_creates_tmp_dir(tmp_path: Path):
    work = tmp_path / "deep" / "work"
    assert not work.exists()
    BebopNetAdapter(_cfg(tmp_path)).prepare(_prog(), work)
    assert work.is_dir()


def test_prepare_writes_input_xml(tmp_path: Path):
    work = tmp_path / "work"
    params = BebopNetAdapter(_cfg(tmp_path)).prepare(_prog(), work)
    xml_path = Path(params["input_xml_path"])
    assert xml_path.exists()
    assert xml_path.suffix == ".xml"
    content = xml_path.read_text()
    assert "<harmony" in content or "<root" in content


def test_prepare_num_measures_equals_num_bars(tmp_path: Path):
    cfg = _cfg(tmp_path)
    p1 = BebopNetAdapter(cfg).prepare(_prog(n_bars=8), tmp_path / "a")
    assert p1["num_measures"] == 8
    p2 = BebopNetAdapter(cfg).prepare(_prog(n_bars=4), tmp_path / "b")
    assert p2["num_measures"] == 4


def test_prepare_different_progressions_yield_different_xml(tmp_path: Path):
    cfg = _cfg(tmp_path)
    p_a = BebopNetAdapter(cfg).prepare(_prog("Cmaj7"), tmp_path / "a")
    p_b = BebopNetAdapter(cfg).prepare(_prog("Am7"),   tmp_path / "b")
    xml_a = Path(p_a["input_xml_path"]).read_text()
    xml_b = Path(p_b["input_xml_path"]).read_text()
    assert xml_a != xml_b


def test_prepare_paths_are_strings(tmp_path: Path):
    params = BebopNetAdapter(_cfg(tmp_path)).prepare(_prog(), tmp_path / "work")
    for key in ["input_xml_path", "output_midi_path", "model_dir", "model_repo_path"]:
        assert isinstance(params[key], str)


def test_prepare_does_not_leak_pipeline_concepts(tmp_path: Path):
    params = BebopNetAdapter(_cfg(tmp_path)).prepare(_prog(), tmp_path / "work")
    forbidden = {"seed_strategy", "run_id", "model_name", "progression", "custom_xml_path"}
    leaked = forbidden & params.keys()
    assert not leaked, f"adapter leaked pipeline concepts: {leaked}"
