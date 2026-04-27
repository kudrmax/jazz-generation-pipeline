from pathlib import Path

import pytest
import yaml

from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig
from pipeline.progression import ChordProgression


def _write_hparams(path: Path, frame_per_bar: int, num_bars: int, num_pitch: int = 50) -> None:
    path.write_text(yaml.safe_dump({
        "model": {
            "frame_per_bar": frame_per_bar,
            "num_bars": num_bars,
            "num_pitch": num_pitch,
            "chord_emb_size": 128, "pitch_emb_size": 256, "hidden_dim": 512,
            "num_layers": 8, "num_heads": 16,
            "key_dim": 512, "value_dim": 512,
            "input_dropout": 0.2, "layer_dropout": 0.2, "attention_dropout": 0.2,
        }
    }))


def _cfg(
    tmp_path: Path, *,
    seed_strategy: str = "tonic_held",
    custom_seed_path: Path | None = None,
    prime_bars: int = 1,
    fpb: int = 16, num_bars: int = 8,
) -> CMTPipelineConfig:
    hparams_path = tmp_path / "hparams.yaml"
    _write_hparams(hparams_path, fpb, num_bars)
    return CMTPipelineConfig(
        checkpoint_path=tmp_path / "ckpt.pth.tar",
        hparams_path=hparams_path,
        repo_path=tmp_path / "repo",
        seed_strategy=seed_strategy,  # type: ignore[arg-type]
        custom_seed_path=custom_seed_path,
        prime_bars=prime_bars,
    )


def _8bars_4_4() -> ChordProgression:
    return ChordProgression(chords=[("Cmaj7", 4)] * 8, tempo=120.0, time_signature="4/4")


def test_validation_unknown_seed_strategy(tmp_path: Path):
    cfg = _cfg(tmp_path, seed_strategy="random_walk")
    with pytest.raises(ValueError, match="seed_strategy"):
        CMTAdapter(cfg).prepare(_8bars_4_4(), tmp_path / "work")


def test_validation_custom_seed_without_path(tmp_path: Path):
    cfg = _cfg(tmp_path, seed_strategy="custom_seed", custom_seed_path=None)
    with pytest.raises(ValueError, match="custom_seed_path"):
        CMTAdapter(cfg).prepare(_8bars_4_4(), tmp_path / "work")


def test_validation_prime_bars_too_small(tmp_path: Path):
    cfg = _cfg(tmp_path, prime_bars=0)
    with pytest.raises(ValueError, match="prime_bars"):
        CMTAdapter(cfg).prepare(_8bars_4_4(), tmp_path / "work")


def test_validation_prime_bars_exceeds_num_bars(tmp_path: Path):
    cfg = _cfg(tmp_path, prime_bars=10, fpb=16, num_bars=8)
    with pytest.raises(ValueError, match="prime_bars"):
        CMTAdapter(cfg).prepare(_8bars_4_4(), tmp_path / "work")


def test_validation_progression_wrong_length(tmp_path: Path):
    """fpb=16, num_bars=8 → max_len=128. 4 такта = 64 frames."""
    cfg = _cfg(tmp_path, fpb=16, num_bars=8)
    short_prog = ChordProgression(chords=[("Cmaj7", 4)] * 4, tempo=120.0, time_signature="4/4")
    with pytest.raises(ValueError, match="frame"):
        CMTAdapter(cfg).prepare(short_prog, tmp_path / "work")


def test_validation_indivisible_time_signature(tmp_path: Path):
    """fpb=16, time_sig=3/4 → 16 % 3 != 0."""
    cfg = _cfg(tmp_path, fpb=16, num_bars=8)
    prog_3_4 = ChordProgression(chords=[("Cmaj7", 6)] * 4, tempo=120.0, time_signature="3/4")
    with pytest.raises(ValueError, match="not divisible"):
        CMTAdapter(cfg).prepare(prog_3_4, tmp_path / "work")


def test_validation_alternative_size_passes(tmp_path: Path):
    """fpb=8, num_bars=4 → max_len=32. 4 такта × 4 beats × 2 fpb/beat = 32."""
    cfg = _cfg(tmp_path, fpb=8, num_bars=4)
    prog = ChordProgression(chords=[("Cmaj7", 4)] * 4, tempo=120.0, time_signature="4/4")
    params = CMTAdapter(cfg).prepare(prog, tmp_path / "work")
    assert "seed_npz_path" in params
