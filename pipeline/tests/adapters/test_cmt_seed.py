import pickle
from pathlib import Path

import numpy as np
import pytest

from pipeline.adapters._cmt_input import (
    ONSET_RHYTHM_IDX, SUSTAIN_PITCH_IDX, SUSTAIN_RHYTHM_IDX, build_seed,
)
from pipeline.adapters.cmt import CMTPipelineConfig
from pipeline.progression import ChordProgression


def _prog(first_chord: str = "Cmaj7", n_bars: int = 8) -> ChordProgression:
    chords = [(first_chord, 4)] + [("G7", 4)] * (n_bars - 1)
    return ChordProgression(chords=chords, tempo=120.0, time_signature="4/4")


def _cfg(strategy: str, custom_pkl_path: Path | None = None, prime_bars: int = 1) -> CMTPipelineConfig:
    return CMTPipelineConfig(
        checkpoint_path=Path("/dev/null"),
        hparams_path=Path("/dev/null"),
        repo_path=Path("/dev/null"),
        seed_strategy=strategy,  # type: ignore[arg-type]
        custom_pkl_path=custom_pkl_path,
        prime_bars=prime_bars,
    )


def test_seed_tonic_held_shapes_and_dtype():
    pitch, rhythm = build_seed(_prog(), _cfg("tonic_held"), frame_per_bar=16, prime_len=16)
    assert pitch.shape == (16,)
    assert rhythm.shape == (16,)
    assert pitch.dtype == np.int64
    assert rhythm.dtype == np.int64


def test_seed_tonic_held_cmaj7():
    """Cmaj7 → root=0 → pitch_idx=0 на фрейме 0, sustain дальше."""
    pitch, rhythm = build_seed(_prog("Cmaj7"), _cfg("tonic_held"), frame_per_bar=16, prime_len=16)
    assert pitch[0] == 0
    assert rhythm[0] == ONSET_RHYTHM_IDX
    for i in range(1, 16):
        assert pitch[i] == SUSTAIN_PITCH_IDX
        assert rhythm[i] == SUSTAIN_RHYTHM_IDX


def test_seed_tonic_held_am7():
    pitch, _ = build_seed(_prog("Am7"), _cfg("tonic_held"), frame_per_bar=16, prime_len=16)
    assert pitch[0] == 9


def test_seed_tonic_held_two_bars():
    """prime_bars=2 → prime_len = 2 * 16 = 32. Один онсет в начале, sustain до конца."""
    pitch, rhythm = build_seed(
        _prog("Cmaj7"), _cfg("tonic_held", prime_bars=2), frame_per_bar=16, prime_len=32,
    )
    assert pitch.shape == (32,)
    assert pitch[0] == 0
    assert rhythm[0] == ONSET_RHYTHM_IDX
    for i in range(1, 32):
        assert pitch[i] == SUSTAIN_PITCH_IDX
        assert rhythm[i] == SUSTAIN_RHYTHM_IDX


def test_seed_tonic_held_alternative_fpb():
    """fpb=8, prime_bars=1 → prime_len=8. Должно работать без хардкода."""
    pitch, rhythm = build_seed(_prog("Cmaj7"), _cfg("tonic_held"), frame_per_bar=8, prime_len=8)
    assert pitch.shape == (8,)
    assert pitch[0] == 0


def test_seed_tonic_quarters_4_4():
    """4/4, fpb=16 → frames_per_beat=4 → онсеты на фреймах 0,4,8,12."""
    pitch, rhythm = build_seed(_prog("Cmaj7"), _cfg("tonic_quarters"), frame_per_bar=16, prime_len=16)
    onset_frames = {0, 4, 8, 12}
    for i in range(16):
        if i in onset_frames:
            assert pitch[i] == 0, f"frame {i}: expected root pitch_idx=0"
            assert rhythm[i] == ONSET_RHYTHM_IDX, f"frame {i}: expected onset"
        else:
            assert pitch[i] == SUSTAIN_PITCH_IDX
            assert rhythm[i] == SUSTAIN_RHYTHM_IDX


def test_seed_tonic_quarters_two_bars():
    """prime_bars=2, fpb=16 → онсеты на 0,4,8,12, 16,20,24,28 (8 онсетов)."""
    pitch, rhythm = build_seed(
        _prog("Cmaj7"), _cfg("tonic_quarters", prime_bars=2), frame_per_bar=16, prime_len=32,
    )
    onset_frames = {0, 4, 8, 12, 16, 20, 24, 28}
    for f in onset_frames:
        assert rhythm[f] == ONSET_RHYTHM_IDX, f"frame {f}"
        assert pitch[f] == 0


def test_seed_custom_pkl(tmp_path: Path):
    pkl_path = tmp_path / "seed.pkl"
    custom_pitch = np.arange(129, dtype=np.int64) % 50
    custom_rhythm = np.full(129, ONSET_RHYTHM_IDX, dtype=np.int64)
    instance = {
        "pitch": custom_pitch,
        "rhythm": custom_rhythm,
        "chord": np.zeros((129, 12), dtype=np.float32),
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(instance, f)

    pitch, rhythm = build_seed(
        _prog(), _cfg("custom_pkl", custom_pkl_path=pkl_path), frame_per_bar=16, prime_len=16,
    )
    np.testing.assert_array_equal(pitch, custom_pitch[:16])
    np.testing.assert_array_equal(rhythm, custom_rhythm[:16])


def test_seed_unknown_strategy_raises():
    with pytest.raises(ValueError, match="seed_strategy"):
        build_seed(_prog(), _cfg("nonsense"), frame_per_bar=16, prime_len=16)


def test_seed_custom_pkl_without_path_raises():
    with pytest.raises(ValueError, match="custom_pkl_path"):
        build_seed(_prog(), _cfg("custom_pkl", custom_pkl_path=None), frame_per_bar=16, prime_len=16)


def test_seed_custom_pkl_too_short_raises(tmp_path: Path):
    pkl_path = tmp_path / "seed.pkl"
    instance = {
        "pitch": np.zeros(8, dtype=np.int64),  # < prime_len=16
        "rhythm": np.zeros(8, dtype=np.int64),
        "chord": np.zeros((8, 12), dtype=np.float32),
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(instance, f)
    with pytest.raises(ValueError, match="too short|prime_len"):
        build_seed(
            _prog(), _cfg("custom_pkl", custom_pkl_path=pkl_path), frame_per_bar=16, prime_len=16,
        )
