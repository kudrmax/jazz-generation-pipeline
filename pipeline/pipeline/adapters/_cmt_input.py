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

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pipeline.chord_vocab import chord_to_pitches, parse_chord
from pipeline.progression import ChordProgression

if TYPE_CHECKING:
    from pipeline.adapters.cmt import CMTPipelineConfig


# Pitch / rhythm vocab — фундаментальные коды CMT, подтверждены diagnostic'ом
# на seed_instance.pkl (см. controller's task brief Task 7):
#   - pitch_idx ∈ [0..47]: onset с MIDI = 60 + pitch_idx
#   - pitch_idx == 48: rest marker (используется вместе с rhythm=1 на note-off)
#   - pitch_idx == 49: sustain (нота продолжает звучать)
#   - rhythm == 0: sustain frame
#   - rhythm == 1: rest start / note-off
#   - rhythm == 2: onset (новая нота)
ONSET_RHYTHM_IDX:    int = 2
SUSTAIN_RHYTHM_IDX:  int = 0
SUSTAIN_PITCH_IDX:   int = 49


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


def build_seed(
    progression: ChordProgression,
    config: "CMTPipelineConfig",
    frame_per_bar: int,
    prime_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (prime_pitch, prime_rhythm), оба shape `[prime_len]` int64."""
    if config.seed_strategy == "tonic_held":
        return _seed_tonic_held(progression, prime_len)
    if config.seed_strategy == "tonic_quarters":
        return _seed_tonic_quarters(progression, frame_per_bar, prime_len)
    if config.seed_strategy == "custom_pkl":
        if config.custom_pkl_path is None:
            raise ValueError("seed_strategy=custom_pkl requires custom_pkl_path")
        return _seed_custom_pkl(config.custom_pkl_path, prime_len)
    raise ValueError(
        f"unknown seed_strategy={config.seed_strategy!r}; "
        f"expected 'tonic_held' | 'tonic_quarters' | 'custom_pkl'"
    )


def _root_pitch_idx(progression: ChordProgression) -> int:
    """pitch_idx = root_idx (т.к. MIDI=60+root_idx и pitch_idx=MIDI-60)."""
    first_chord = progression.chords[0][0]
    root_idx, _quality = parse_chord(first_chord)
    return root_idx


def _seed_tonic_held(progression: ChordProgression, prime_len: int) -> tuple[np.ndarray, np.ndarray]:
    pitch = np.full(prime_len, SUSTAIN_PITCH_IDX, dtype=np.int64)
    rhythm = np.full(prime_len, SUSTAIN_RHYTHM_IDX, dtype=np.int64)
    pitch[0] = _root_pitch_idx(progression)
    rhythm[0] = ONSET_RHYTHM_IDX
    return pitch, rhythm


def _seed_tonic_quarters(
    progression: ChordProgression,
    frame_per_bar: int,
    prime_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    pitch = np.full(prime_len, SUSTAIN_PITCH_IDX, dtype=np.int64)
    rhythm = np.full(prime_len, SUSTAIN_RHYTHM_IDX, dtype=np.int64)
    bpb = progression.beats_per_bar()
    frames_per_beat = frame_per_bar // bpb
    root_idx = _root_pitch_idx(progression)
    for frame in range(0, prime_len, frames_per_beat):
        pitch[frame] = root_idx
        rhythm[frame] = ONSET_RHYTHM_IDX
    return pitch, rhythm


def _seed_custom_pkl(pkl_path: Path, prime_len: int) -> tuple[np.ndarray, np.ndarray]:
    with open(pkl_path, "rb") as f:
        instance = pickle.load(f)
    pitch_full = np.asarray(instance["pitch"], dtype=np.int64)
    rhythm_full = np.asarray(instance["rhythm"], dtype=np.int64)
    if pitch_full.shape[0] < prime_len or rhythm_full.shape[0] < prime_len:
        raise ValueError(
            f"custom_pkl too short for prime_len={prime_len}: "
            f"pitch.shape={pitch_full.shape}, rhythm.shape={rhythm_full.shape}"
        )
    return pitch_full[:prime_len], rhythm_full[:prime_len]
