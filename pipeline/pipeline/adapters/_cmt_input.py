"""CMT-специфичные преобразования. Не общий модуль — формат CMT.

progression → chord_chroma
seed_strategy → prime_pitch + prime_rhythm  (добавится в Task 7)

Все размеры (frame_per_bar, num_bars, prime_len) приходят параметрами —
никаких магических чисел. Подмена весов с другими параметрами модели =
другие значения параметров, без правок этого файла.

Семантика последнего фрейма chord_chroma подтверждена на seed_instance.pkl:
chord имеет shape [max_len + 1, 12]; все max_len + 1 фреймов несут реальные
данные; последний (index max_len) повторяет предпоследний — это «хвост»
последнего аккорда, не нулевой padding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.chord_vocab import chord_to_pitches
from pipeline.progression import ChordProgression


def progression_to_chroma(
    progression: ChordProgression,
    frame_per_bar: int,
    num_bars: int,
) -> np.ndarray:
    """Развернуть progression в chord_chroma `[max_len + 1, 12]` float32.

    max_len = frame_per_bar * num_bars.

    Raises:
        ValueError: если frame_per_bar не делится на beats_per_bar нацело
                    (физически нельзя развернуть в целое число фреймов).
        ValueError: если total_frames(progression) != max_len.
    """
    bpb = progression.beats_per_bar()
    if frame_per_bar % bpb != 0:
        raise ValueError(
            f"frame_per_bar={frame_per_bar} not divisible by beats_per_bar={bpb}; "
            f"cannot expand chord progression to integer frame count"
        )
    frames_per_beat = frame_per_bar // bpb

    frames: list[np.ndarray] = []
    for chord_str, beats in progression.chords:
        pitches = chord_to_pitches(chord_str)
        chroma_vec = np.zeros(12, dtype=np.float32)
        for p in pitches:
            chroma_vec[p % 12] = 1.0
        n_frames = beats * frames_per_beat
        frames.append(np.tile(chroma_vec, (n_frames, 1)))

    chroma = np.concatenate(frames, axis=0)
    expected_max_len = frame_per_bar * num_bars
    if chroma.shape[0] != expected_max_len:
        raise ValueError(
            f"progression yields {chroma.shape[0]} frames, "
            f"but model expects {expected_max_len} (= {num_bars} bars × {frame_per_bar} fpb)"
        )

    # Tail-фрейм: дублируем последний реальный (как в реальных pkl-файлах CMT-датасета).
    chroma = np.vstack([chroma, chroma[-1:]])
    return chroma
