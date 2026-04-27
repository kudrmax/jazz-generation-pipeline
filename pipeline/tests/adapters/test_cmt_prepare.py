from pathlib import Path

import numpy as np
import yaml

from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig
from pipeline.progression import ChordProgression


def _hparams(tmp_path: Path, fpb: int = 16, num_bars: int = 8) -> Path:
    p = tmp_path / "hparams.yaml"
    p.write_text(yaml.safe_dump({"model": {
        "frame_per_bar": fpb, "num_bars": num_bars, "num_pitch": 50,
        "chord_emb_size": 128, "pitch_emb_size": 256, "hidden_dim": 512,
        "num_layers": 8, "num_heads": 16, "key_dim": 512, "value_dim": 512,
        "input_dropout": 0.2, "layer_dropout": 0.2, "attention_dropout": 0.2,
    }}))
    return p


def _cfg(tmp_path: Path, fpb: int = 16, num_bars: int = 8) -> CMTPipelineConfig:
    return CMTPipelineConfig(
        checkpoint_path=tmp_path / "ckpt.pth.tar",
        hparams_path=_hparams(tmp_path, fpb, num_bars),
        repo_path=tmp_path / "repo",
    )


def _prog(first_chord: str = "Cmaj7") -> ChordProgression:
    return ChordProgression(
        chords=[(first_chord, 4)] + [("G7", 4)] * 7, tempo=120.0, time_signature="4/4",
    )


def test_prepare_returns_required_keys(tmp_path: Path):
    work = tmp_path / "work"
    params = CMTAdapter(_cfg(tmp_path)).prepare(_prog(), work)
    for key in ["checkpoint_path", "hparams_path", "model_repo_path",
                "seed_npz_path", "output_midi_path", "topk", "device"]:
        assert key in params, f"missing: {key}"


def test_prepare_creates_tmp_dir(tmp_path: Path):
    work = tmp_path / "deep" / "work"
    assert not work.exists()
    CMTAdapter(_cfg(tmp_path)).prepare(_prog(), work)
    assert work.is_dir()


def test_prepare_writes_seed_npz_default_size(tmp_path: Path):
    work = tmp_path / "work"
    params = CMTAdapter(_cfg(tmp_path)).prepare(_prog(), work)
    npz = np.load(params["seed_npz_path"])
    assert set(npz.files) == {"chord_chroma", "prime_pitch", "prime_rhythm"}
    assert npz["chord_chroma"].shape == (129, 12)  # 16*8 + 1
    assert npz["chord_chroma"].dtype == np.float32
    assert npz["prime_pitch"].shape == (16,)
    assert npz["prime_pitch"].dtype == np.int64
    assert npz["prime_rhythm"].shape == (16,)


def test_prepare_writes_seed_npz_alternative_size(tmp_path: Path):
    """Доказывает что размеры берутся из hparams, а не хардкодятся."""
    cfg = _cfg(tmp_path, fpb=8, num_bars=4)
    prog = ChordProgression(chords=[("Cmaj7", 4)] * 4, tempo=120.0, time_signature="4/4")
    params = CMTAdapter(cfg).prepare(prog, tmp_path / "work")
    npz = np.load(params["seed_npz_path"])
    assert npz["chord_chroma"].shape == (33, 12)  # 8*4 + 1
    assert npz["prime_pitch"].shape == (8,)       # prime_bars=1 * fpb=8


def test_prepare_different_progressions_yield_different_seeds(tmp_path: Path):
    cfg = _cfg(tmp_path)
    p_a = CMTAdapter(cfg).prepare(_prog("Cmaj7"), tmp_path / "a")
    p_b = CMTAdapter(cfg).prepare(_prog("Am7"),   tmp_path / "b")
    npz_a = np.load(p_a["seed_npz_path"])
    npz_b = np.load(p_b["seed_npz_path"])
    assert not np.array_equal(npz_a["chord_chroma"], npz_b["chord_chroma"])
    assert not np.array_equal(npz_a["prime_pitch"], npz_b["prime_pitch"])


def test_prepare_paths_are_strings(tmp_path: Path):
    params = CMTAdapter(_cfg(tmp_path)).prepare(_prog(), tmp_path / "work")
    for key in ["checkpoint_path", "hparams_path", "model_repo_path",
                "seed_npz_path", "output_midi_path"]:
        assert isinstance(params[key], str)


def test_prepare_does_not_leak_pipeline_concepts(tmp_path: Path):
    params = CMTAdapter(_cfg(tmp_path)).prepare(_prog(), tmp_path / "work")
    forbidden = {"seed_strategy", "run_id", "model_name", "progression", "prime_bars"}
    leaked = forbidden & params.keys()
    assert not leaked, f"adapter leaked pipeline concepts: {leaked}"
