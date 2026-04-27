import numpy as np
import pytest
from pipeline.adapters._cmt_input import progression_to_chroma
from pipeline.progression import ChordProgression


def _prog(chords: list[tuple[str, int]], time_signature: str = "4/4") -> ChordProgression:
    return ChordProgression(chords=chords, tempo=120.0, time_signature=time_signature)


def test_chroma_shape_and_dtype_default_size():
    """fpb=16, num_bars=8 → max_len=128, shape (129, 12)."""
    prog = _prog([("Cmaj7", 4)] * 8)
    chroma = progression_to_chroma(prog, frame_per_bar=16, num_bars=8)
    assert chroma.shape == (129, 12)
    assert chroma.dtype == np.float32


def test_chroma_shape_alternative_size():
    """fpb=8, num_bars=4 → max_len=32, shape (33, 12). Доказывает: размеры не хардкод."""
    prog = _prog([("Cmaj7", 4)] * 4)
    chroma = progression_to_chroma(prog, frame_per_bar=8, num_bars=4)
    assert chroma.shape == (33, 12)


def test_chroma_cmaj7_pitch_classes():
    """Cmaj7 → {C=0, E=4, G=7, B=11}."""
    prog = _prog([("Cmaj7", 4)] * 8)
    chroma = progression_to_chroma(prog, frame_per_bar=16, num_bars=8)
    expected = np.zeros(12, dtype=np.float32)
    expected[[0, 4, 7, 11]] = 1.0
    for t in range(16):
        np.testing.assert_array_equal(chroma[t], expected, err_msg=f"frame {t}")


def test_chroma_chord_change_at_correct_frame():
    """[Cmaj7, Am7] × 4: смена аккорда на 16-м фрейме (1 бар = 16 frames при 4/4 fpb=16)."""
    prog = _prog([("Cmaj7", 4), ("Am7", 4)] * 4)
    chroma = progression_to_chroma(prog, frame_per_bar=16, num_bars=8)
    cmaj7 = np.zeros(12, dtype=np.float32); cmaj7[[0, 4, 7, 11]] = 1.0
    am7   = np.zeros(12, dtype=np.float32); am7[[9, 0, 4, 7]] = 1.0
    np.testing.assert_array_equal(chroma[0],  cmaj7)
    np.testing.assert_array_equal(chroma[15], cmaj7)
    np.testing.assert_array_equal(chroma[16], am7)
    np.testing.assert_array_equal(chroma[31], am7)


def test_chroma_last_frame_equals_second_to_last():
    """Последний фрейм (index max_len) — копия предыдущего (tail-продолжение)."""
    prog = _prog([("Cmaj7", 4)] * 7 + [("F7", 4)])  # last bar = F7
    chroma = progression_to_chroma(prog, frame_per_bar=16, num_bars=8)
    np.testing.assert_array_equal(chroma[127], chroma[128])
    f7 = np.zeros(12, dtype=np.float32); f7[[5, 9, 0, 3]] = 1.0  # F=5, A=9, C=0, Eb=3
    np.testing.assert_array_equal(chroma[128], f7)


def test_chroma_different_progressions_differ():
    a = progression_to_chroma(_prog([("Cmaj7", 4)] * 8), frame_per_bar=16, num_bars=8)
    b = progression_to_chroma(_prog([("F7", 4)] * 8), frame_per_bar=16, num_bars=8)
    assert not np.array_equal(a, b)


def test_chroma_raises_when_fpb_not_divisible_by_bpb():
    """fpb=16, time_sig=3/4 → 16 % 3 != 0 → ValueError."""
    prog = _prog([("Cmaj7", 6)] * 4, time_signature="3/4")  # 24 beats / 3 = 8 bars
    with pytest.raises(ValueError, match="not divisible"):
        progression_to_chroma(prog, frame_per_bar=16, num_bars=8)


def test_chroma_raises_when_total_frames_mismatch():
    """fpb=16, num_bars=8 → expected max_len=128. Прогрессия на 4 такта = 64 frames."""
    prog = _prog([("Cmaj7", 4)] * 4)  # 4 bars
    with pytest.raises(ValueError, match="frame"):
        progression_to_chroma(prog, frame_per_bar=16, num_bars=8)
