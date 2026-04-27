# CMT Integration — Implementation Plan (production)

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (fresh subagent per task, TDD discipline). Каждый таск — failing test → implementation → passing test → commit.

**Goal.** Production-интеграция CMT в pipeline. Все размеры модели (`num_bars`, `frame_per_bar`, `num_pitch`) читаются из `hparams.yaml`, никаких магических чисел в коде. Замена весов = подмена пары `(checkpoint_path, hparams_path)` в `pipeline/pipeline/config.py`.

**Architecture.** `CMTAdapter` (immutable config) + `_cmt_input` (chroma + seed builders, принимают размеры параметрами) + `cmt_runner.py` (subprocess в CMT-venv, ничего не импортирует из `pipeline.*`).

**Spec:** `docs/superpowers/specs/2026-04-27-pipeline-cmt-design.md`.

---

## Контекст для исполнителя

- **Ветка:** `feat/pipeline-cmt` (существует, на ней 1 старый smoke-коммит — Task 0 переинициализирует ветку с master).
- **Pipeline-venv:** `pipeline/.venv` (для тестов и adapter-уровня).
- **CMT-venv:** `models/CMT-pytorch/.venv` (создаётся в Task 3).
- **Чекпоинт уже на машине:** `models/CMT-pytorch/result/smoke_wjazzd_5epochs/{smoke_5epochs.pth.tar,hparams.yaml,seed_instance.pkl}`. Не должны потеряться при превращении CMT в submodule.
- **Форк:** `https://github.com/kudrmax/CMT-pytorch` (clean upstream, нужны 3 коммита-патча — Task 1).

CMT-API:
- `from model import ChordConditionedMelodyTransformer` — конструктор берёт `**model_config` из `hparams.yaml` секции `model:`.
- Чекпоинт: `torch.load(...)` возвращает dict, грузим `state['model']`.
- `model.sampling(prime_rhythm, prime_pitch, chord, topk)` → `{'rhythm': [B, max_len], 'pitch': [B, max_len]}`.
- `from utils.utils import pitch_to_midi` — пишет MIDI с двумя треками `name='melody'` и `name='chord'`.

Все коммиты — Conventional Commits (`feat(cmt):`, `chore(cmt):`, `test(cmt):`). Push в `feat/pipeline-cmt` разрешён. Push в форк CMT в Task 1 явно разрешён пользователем. Никаких других git-операций без явного указания.

---

## Task 0: Подготовка ветки

- [ ] **Step 1: Перейти на feat/pipeline-cmt и сбросить на master**

```bash
cd /Users/maxos/PythonProjects/diploma
git checkout feat/pipeline-cmt
git log --oneline -3
git reset --hard master
git log --oneline -3
```

Старый smoke-коммит исчезает из ветки.

- [ ] **Step 2: Закоммитить spec + plan**

```bash
git add docs/superpowers/specs/2026-04-27-pipeline-cmt-design.md \
        docs/superpowers/plans/2026-04-27-pipeline-cmt.md
git commit -m "docs(cmt): production integration spec and plan"
```

---

## Task 1: Подготовить форк kudrmax/CMT-pytorch

Цель: запушить 3 коммита-патча в форк, чтобы `git submodule add` подтянул нужные правки.

- [ ] **Step 1: Клонировать форк во временную директорию**

```bash
mkdir -p /tmp/cmt_fork_work
cd /tmp/cmt_fork_work
git clone https://github.com/kudrmax/CMT-pytorch.git
cd CMT-pytorch
git log --oneline -5
```

- [ ] **Step 2: Патч 1 — `preprocess.py` py3.11+ random.sample**

В `preprocess.py` строки 42-43 заменить:

```python
eval_set = random.sample(eval_test_cand, num_eval)
test_set = random.sample(eval_test_cand - set(eval_set), num_test)
```

на:

```python
eval_set = random.sample(sorted(eval_test_cand), num_eval)
test_set = random.sample(sorted(eval_test_cand - set(eval_set)), num_test)
```

```bash
git add preprocess.py
git commit -m "fix: Python 3.11+ compat (random.sample requires sequence, not set)"
```

- [ ] **Step 3: Патч 2 — `utils/hparams.py` yaml.safe_load**

В `utils/hparams.py:28` заменить `yaml.load(f)` → `yaml.safe_load(f)`.

```bash
git add utils/hparams.py
git commit -m "fix: use yaml.safe_load (yaml.load(...) without Loader is deprecated)"
```

- [ ] **Step 4: Создать `requirements-py312.txt`**

```
# CMT — runtime requirements for Python 3.12 на arm64 macOS.
# Используется pipeline'ом jazz-generation-pipeline для inference.
#
# Install:
#   python3.12 -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements-py312.txt

torch
numpy
scipy
pretty_midi
pyyaml
matplotlib
tensorboardX
tqdm
```

```bash
git add requirements-py312.txt
git commit -m "add: requirements-py312.txt for Python 3.12 inference runtime"
```

- [ ] **Step 5: Push**

```bash
git push origin master
```

- [ ] **Step 6: Cleanup**

```bash
cd /Users/maxos/PythonProjects/diploma
trash /tmp/cmt_fork_work
```

---

## Task 2: Превратить локальный CMT-pytorch в git submodule

- [ ] **Step 1: Сохранить `result/`**

```bash
cd /Users/maxos/PythonProjects/diploma
mv models/CMT-pytorch/result /tmp/cmt_result_backup
ls /tmp/cmt_result_backup/smoke_wjazzd_5epochs/
```

Проверить: 3 файла на месте.

- [ ] **Step 2: Удалить локальную директорию**

```bash
trash models/CMT-pytorch
```

- [ ] **Step 3: Добавить как submodule**

```bash
git submodule add https://github.com/kudrmax/CMT-pytorch.git models/CMT-pytorch
git status
```

- [ ] **Step 4: Вернуть `result/`**

```bash
mv /tmp/cmt_result_backup models/CMT-pytorch/result
ls models/CMT-pytorch/result/smoke_wjazzd_5epochs/
```

- [ ] **Step 5: Обновить корневой `.gitignore`**

В блоке `models/*` исключений добавить `!models/CMT-pytorch`:

```
# Submodule-исключения — иначе models/* спрятал бы их.
models/*
!models/MINGUS
!models/CMT-pytorch
```

- [ ] **Step 6: Verify gitignore**

```bash
git check-ignore models/CMT-pytorch        # пусто (виден как submodule)
git check-ignore models/CMT-pytorch/result # путь возвращается (gitignored)
git check-ignore models/CMT-pytorch/.venv  # путь возвращается
```

- [ ] **Step 7: Commit**

```bash
git add .gitmodules .gitignore models/CMT-pytorch
git commit -m "chore(cmt): add CMT-pytorch as git submodule on kudrmax fork"
```

---

## Task 3: CMT-venv

- [ ] **Step 1: Создать venv и установить зависимости**

```bash
cd /Users/maxos/PythonProjects/diploma/models/CMT-pytorch
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-py312.txt
deactivate
```

- [ ] **Step 2: Smoke-проверить импорты**

```bash
cd /Users/maxos/PythonProjects/diploma/models/CMT-pytorch
.venv/bin/python -c "
import sys, os
sys.path.insert(0, os.getcwd())
import torch, yaml, pickle, numpy as np
from scipy.sparse import csc_matrix
from model import ChordConditionedMelodyTransformer
from utils.utils import pitch_to_midi
print('OK', torch.__version__, np.__version__)
"
```

Если что-то не импортируется — поправить `requirements-py312.txt` в форке (новый коммит → push), пересобрать venv.

Не коммитим — venv gitignored.

---

## Task 4: Убрать CMTAdapter из stub-параметризации

- [ ] **Step 1: Baseline зелёный**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/adapters/test_base.py -v
```

- [ ] **Step 2: Удалить CMTAdapter из импортов и parametrize**

В `pipeline/tests/adapters/test_base.py`:

Удалить:
```python
from pipeline.adapters.cmt import CMTAdapter
```

В обоих `@pytest.mark.parametrize` убрать `CMTAdapter`:
```python
@pytest.mark.parametrize("AdapterCls", [
    BebopNetAdapter, EC2VaeAdapter, ComMUAdapter, PolyffusionAdapter,
])
```

- [ ] **Step 3: Тесты зелёные**

```bash
.venv/bin/python -m pytest tests/adapters/test_base.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/tests/adapters/test_base.py
git commit -m "test(cmt): remove CMTAdapter from stub parametrize before refactor"
```

---

## Task 5: CMTPipelineConfig + CMTAdapter constructor

**Files:**
- Modify: `pipeline/pipeline/adapters/cmt.py`
- Create: `pipeline/tests/adapters/test_cmt_config.py`

- [ ] **Step 1: Failing tests**

`pipeline/tests/adapters/test_cmt_config.py`:

```python
from pathlib import Path
import pytest
from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig


def _make_config(tmp_path: Path) -> CMTPipelineConfig:
    return CMTPipelineConfig(
        checkpoint_path=tmp_path / "ckpt.pth.tar",
        hparams_path=tmp_path / "hparams.yaml",
        repo_path=tmp_path / "repo",
    )


def test_cmt_adapter_requires_config():
    with pytest.raises(TypeError):
        CMTAdapter()  # type: ignore[call-arg]


def test_cmt_adapter_stores_config(tmp_path: Path):
    cfg = _make_config(tmp_path)
    adapter = CMTAdapter(cfg)
    assert adapter._config is cfg


def test_cmt_pipeline_config_defaults(tmp_path: Path):
    cfg = _make_config(tmp_path)
    assert cfg.seed_strategy == "tonic_held"
    assert cfg.custom_pkl_path is None
    assert cfg.prime_bars == 1
    assert cfg.topk == 5
    assert cfg.device == "cpu"
```

- [ ] **Step 2: Реализовать skeleton**

Переписать `pipeline/pipeline/adapters/cmt.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pretty_midi

from pipeline.adapters.base import ModelAdapter
from pipeline.progression import ChordProgression


@dataclass
class CMTPipelineConfig:
    """Все настройки CMT на уровне pipeline. immutable после init."""

    checkpoint_path: Path
    hparams_path: Path
    repo_path: Path
    seed_strategy: Literal["tonic_held", "tonic_quarters", "custom_pkl"] = "tonic_held"
    custom_pkl_path: Path | None = None
    prime_bars: int = 1
    topk: int = 5
    device: str = "cpu"


class CMTAdapter(ModelAdapter):
    def __init__(self, config: CMTPipelineConfig) -> None:
        self._config = config

    def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
        raise NotImplementedError("model cmt: prepare not implemented yet")

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        raise NotImplementedError("model cmt: extract_melody not implemented yet")
```

- [ ] **Step 3: Тесты зелёные**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/adapters/test_cmt_config.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/pipeline/adapters/cmt.py pipeline/tests/adapters/test_cmt_config.py
git commit -m "feat(cmt): add CMTPipelineConfig and CMTAdapter constructor"
```

---

## Task 6: Diagnostic + конвертер `progression → chord_chroma`

**Files:**
- Create: `pipeline/pipeline/adapters/_cmt_input.py`
- Create: `pipeline/tests/adapters/test_cmt_chroma.py`

**Diagnostic шаг (ВНУТРИ Task 6, не отдельный таск):**

- [ ] **Step 0 (diagnostic): Подтвердить семантику chord[0] vs chord[max_len]**

```bash
cd /Users/maxos/PythonProjects/diploma
pipeline/.venv/bin/python -c "
import pickle, numpy as np
with open('models/CMT-pytorch/result/smoke_wjazzd_5epochs/seed_instance.pkl', 'rb') as f:
    inst = pickle.load(f)
chord = inst['chord'].toarray()
print('chord shape:', chord.shape)
print('chord[0] (sum activations):', int(chord[0].sum()), '— first frame')
print('chord[1] (sum activations):', int(chord[1].sum()), '— second frame')
print('chord[-2] (sum activations):', int(chord[-2].sum()), '— second-to-last')
print('chord[-1] (sum activations):', int(chord[-1].sum()), '— last frame')
print()
print('chord[0] active pitches:', chord[0].nonzero()[0].tolist())
print('chord[1] active pitches:', chord[1].nonzero()[0].tolist())
print('chord[-1] active pitches:', chord[-1].nonzero()[0].tolist())
"
```

Ожидаемый вывод покажет какой из конечных фреймов — нулевой/служебный, какой — реальный. Один из вариантов:
- (A) Если `chord[0]` имеет активные pitches и `chord[-1]` нули → реальные данные `[0..127]`, target padding `[128]`. Конвертер заполняет `[0..127]` и нулит `[128]`.
- (B) Если `chord[0]` нули и `chord[-1]` имеет активные pitches → стартовый padding `[0]`, реальные `[1..128]`. Конвертер заполняет `[1..128]` и нулит `[0]`.

Зафиксировать вывод в комментарии в `_cmt_input.py`. Реализация Step 3 ниже использует найденную семантику.

- [ ] **Step 1: Failing tests**

`pipeline/tests/adapters/test_cmt_chroma.py`:

```python
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
    # Проверяем достаточное количество фреймов в начале
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


def test_chroma_different_progressions_differ():
    a = progression_to_chroma(_prog([("Cmaj7", 4)] * 8), frame_per_bar=16, num_bars=8)
    b = progression_to_chroma(_prog([("F7", 4)] * 8), frame_per_bar=16, num_bars=8)
    assert not np.array_equal(a, b)


def test_chroma_raises_when_fpb_not_divisible_by_bpb():
    """fpb=16, time_sig=3/4 → 16%3 != 0 → ValueError."""
    prog = _prog([("Cmaj7", 6)] * 4, time_signature="3/4")  # 24 beats / 3 = 8 bars
    with pytest.raises(ValueError, match="not divisible"):
        progression_to_chroma(prog, frame_per_bar=16, num_bars=8)


def test_chroma_raises_when_total_frames_mismatch():
    """fpb=16, num_bars=8, max_len=128. Прогрессия на 4 такта = 64 frames ≠ 128."""
    prog = _prog([("Cmaj7", 4)] * 4)  # 4 bars
    with pytest.raises(ValueError, match="frame"):
        progression_to_chroma(prog, frame_per_bar=16, num_bars=8)
```

- [ ] **Step 2: Тесты падают**

```bash
.venv/bin/python -m pytest tests/adapters/test_cmt_chroma.py -v
```

- [ ] **Step 3: Реализовать `progression_to_chroma`**

Создать `pipeline/pipeline/adapters/_cmt_input.py`:

```python
"""CMT-специфичные преобразования. Не общий модуль — формат CMT.

progression → chord_chroma
seed_strategy → prime_pitch + prime_rhythm

Все размеры (frame_per_bar, num_bars, prime_len) приходят параметрами —
никаких магических чисел. Подмена весов с другими параметрами модели =
другие значения параметров, без правок этого файла.
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


# Семантика последнего/первого фрейма chord — определяется тем как тренировали CMT.
# Подтверждено на seed_instance.pkl в Task 6 Step 0:
#   <вписать сюда что показал diagnostic — например:
#    chord[0] = реальные данные первого аккорда; chord[-1] = нули (target padding)>
# При обновлении кода CMT (что редко) — пересмотреть.


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

    # Target padding zero-frame в конце — подтверждено diagnostic-шагом на seed_instance.pkl.
    chroma = np.vstack([chroma, np.zeros((1, 12), dtype=np.float32)])
    return chroma
```

(Если diagnostic Step 0 покажет что padding в начале, а не в конце — поменять последние 2 строки на `np.vstack([np.zeros((1, 12), ...), chroma])`.)

- [ ] **Step 4: Тесты зелёные**

```bash
.venv/bin/python -m pytest tests/adapters/test_cmt_chroma.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/adapters/_cmt_input.py pipeline/tests/adapters/test_cmt_chroma.py
git commit -m "feat(cmt): progression → chord_chroma converter (size-agnostic)"
```

---

## Task 7: Diagnostic + seed builder (3 стратегии)

**Files:**
- Modify: `pipeline/pipeline/adapters/_cmt_input.py`
- Create: `pipeline/tests/adapters/test_cmt_seed.py`

- [ ] **Step 0 (diagnostic): Подтвердить семантику pitch-vocab и rhythm-vocab**

```bash
cd /Users/maxos/PythonProjects/diploma
pipeline/.venv/bin/python -c "
import pickle
with open('models/CMT-pytorch/result/smoke_wjazzd_5epochs/seed_instance.pkl', 'rb') as f:
    inst = pickle.load(f)
pitch = inst['pitch']
rhythm = inst['rhythm']
print('pitch first 32:', pitch[:32].tolist())
print('rhythm first 32:', rhythm[:32].tolist())
print()
print('unique pitch:', sorted(set(pitch.tolist())))
print('unique rhythm:', sorted(set(rhythm.tolist())))
print()
# Найти первую онсет-границу: где rhythm меняется
print('rhythm transitions:')
for i in range(min(64, len(rhythm))):
    print(f'  t={i}: pitch={pitch[i]}, rhythm={rhythm[i]}')
"
```

Из вывода понять:
- `ONSET_RHYTHM_IDX` — какое значение rhythm на старте новой ноты (обычно 2).
- `SUSTAIN_RHYTHM_IDX` — какое значение rhythm пока нота держится (обычно 1).
- `SUSTAIN_PITCH_IDX` — какое значение pitch на удерживаемых фреймах (вариант 48 или 49 — выбрать из реального pkl).

Также свериться с `models/CMT-pytorch/utils/utils.py:58-96` (функция `pitch_to_midi`) и `:200-210` (`rhythm_to_symbol_list`).

Зафиксировать найденные числа в `_cmt_input.py` как константы.

- [ ] **Step 1: Failing tests**

`pipeline/tests/adapters/test_cmt_seed.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from pipeline.adapters._cmt_input import (
    ONSET_RHYTHM_IDX, SUSTAIN_RHYTHM_IDX, SUSTAIN_PITCH_IDX, build_seed,
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
        seed_strategy=strategy,
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


def test_seed_tonic_held_alternative_size():
    """fpb=8, prime_bars=2 → prime_len=16. Затравка должна корректно построиться."""
    pitch, rhythm = build_seed(_prog("Cmaj7"), _cfg("tonic_held"), frame_per_bar=8, prime_len=16)
    assert pitch[0] == 0
    assert rhythm[0] == ONSET_RHYTHM_IDX


def test_seed_tonic_held_two_bars():
    """prime_bars=2 → prime_len = 2 * 16 = 32. Только один онсет в начале (held)."""
    pitch, rhythm = build_seed(
        _prog("Cmaj7"), _cfg("tonic_held", prime_bars=2), frame_per_bar=16, prime_len=32,
    )
    assert pitch.shape == (32,)
    assert pitch[0] == 0
    assert rhythm[0] == ONSET_RHYTHM_IDX
    # фреймы 1..31 должны быть sustain
    for i in range(1, 32):
        assert rhythm[i] == SUSTAIN_RHYTHM_IDX


def test_seed_tonic_quarters_4_4():
    """4/4, fpb=16, prime_bars=1 → онсеты на фреймах 0,4,8,12."""
    pitch, rhythm = build_seed(_prog("Cmaj7"), _cfg("tonic_quarters"), frame_per_bar=16, prime_len=16)
    onset_frames = {0, 4, 8, 12}
    for i in range(16):
        if i in onset_frames:
            assert pitch[i] == 0, f"frame {i}: expected root"
            assert rhythm[i] == ONSET_RHYTHM_IDX, f"frame {i}: expected onset"
        else:
            assert pitch[i] == SUSTAIN_PITCH_IDX
            assert rhythm[i] == SUSTAIN_RHYTHM_IDX


def test_seed_tonic_quarters_two_bars():
    """prime_bars=2, fpb=16 → онсеты на 0,4,8,12, 16,20,24,28 (8 онсетов всего)."""
    pitch, rhythm = build_seed(
        _prog("Cmaj7"), _cfg("tonic_quarters", prime_bars=2), frame_per_bar=16, prime_len=32,
    )
    onset_frames = {0, 4, 8, 12, 16, 20, 24, 28}
    for f in onset_frames:
        assert rhythm[f] == ONSET_RHYTHM_IDX, f"frame {f}"
        assert pitch[f] == 0


def test_seed_custom_pkl(tmp_path: Path):
    import pickle
    from scipy.sparse import csc_matrix
    pkl_path = tmp_path / "seed.pkl"
    custom_pitch = np.arange(129, dtype=np.int64) % 50
    custom_rhythm = np.full(129, ONSET_RHYTHM_IDX, dtype=np.int64)
    instance = {
        "pitch": custom_pitch,
        "rhythm": custom_rhythm,
        "chord": csc_matrix(np.zeros((129, 12), dtype=np.float32)),
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
    import pickle
    from scipy.sparse import csc_matrix
    pkl_path = tmp_path / "seed.pkl"
    instance = {
        "pitch": np.zeros(8, dtype=np.int64),  # < prime_len=16
        "rhythm": np.zeros(8, dtype=np.int64),
        "chord": csc_matrix(np.zeros((8, 12))),
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(instance, f)
    with pytest.raises(ValueError, match="too short|prime_len"):
        build_seed(
            _prog(), _cfg("custom_pkl", custom_pkl_path=pkl_path), frame_per_bar=16, prime_len=16,
        )
```

- [ ] **Step 2: Тесты падают**

```bash
.venv/bin/python -m pytest tests/adapters/test_cmt_seed.py -v
```

- [ ] **Step 3: Реализовать `build_seed`**

Дописать в `pipeline/pipeline/adapters/_cmt_input.py`:

```python
# Pitch / rhythm vocab — фундаментальные коды CMT.
# Источник: utils/utils.py:58-96 (pitch_to_midi), :200-210 (rhythm_to_symbol_list).
# Подтверждены diagnostic-шагом Task 7 Step 0 на seed_instance.pkl.
ONSET_RHYTHM_IDX:    int = 2  # rhythm code новой ноты
SUSTAIN_RHYTHM_IDX:  int = 1  # rhythm code продолжения
SUSTAIN_PITCH_IDX:   int = 48  # <или 49 — фиксируется по diagnostic вывода>


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
    """pitch_idx = root_idx (т.к. C4 = MIDI 60 = pitch_idx 0)."""
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
```

- [ ] **Step 4: Тесты зелёные**

```bash
.venv/bin/python -m pytest tests/adapters/test_cmt_seed.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/adapters/_cmt_input.py pipeline/tests/adapters/test_cmt_seed.py
git commit -m "feat(cmt): seed builder with 3 strategies (tonic_held, tonic_quarters, custom_pkl)"
```

---

## Task 8: Валидация + чтение hparams в `CMTAdapter.prepare`

**Files:**
- Modify: `pipeline/pipeline/adapters/cmt.py`
- Create: `pipeline/tests/adapters/test_cmt_validation.py`

- [ ] **Step 1: Failing tests**

`pipeline/tests/adapters/test_cmt_validation.py`:

```python
from pathlib import Path
import yaml

import pytest

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
    custom_pkl_path: Path | None = None,
    prime_bars: int = 1,
    fpb: int = 16, num_bars: int = 8,
) -> CMTPipelineConfig:
    hparams_path = tmp_path / "hparams.yaml"
    _write_hparams(hparams_path, fpb, num_bars)
    return CMTPipelineConfig(
        checkpoint_path=tmp_path / "ckpt.pth.tar",
        hparams_path=hparams_path,
        repo_path=tmp_path / "repo",
        seed_strategy=seed_strategy,
        custom_pkl_path=custom_pkl_path,
        prime_bars=prime_bars,
    )


def _8bars_4_4() -> ChordProgression:
    return ChordProgression(chords=[("Cmaj7", 4)] * 8, tempo=120.0, time_signature="4/4")


def test_validation_unknown_seed_strategy(tmp_path: Path):
    cfg = _cfg(tmp_path, seed_strategy="random_walk")
    with pytest.raises(ValueError, match="seed_strategy"):
        CMTAdapter(cfg).prepare(_8bars_4_4(), tmp_path / "work")


def test_validation_custom_pkl_without_path(tmp_path: Path):
    cfg = _cfg(tmp_path, seed_strategy="custom_pkl", custom_pkl_path=None)
    with pytest.raises(ValueError, match="custom_pkl_path"):
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
    """fpb=16, num_bars=8 → max_len=128. Прогрессия на 4 такта = 64 frames."""
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
    """fpb=8, num_bars=4 → max_len=32. 4 такта × 4 beats × 2 fpb/beat = 32. Должно пройти валидацию."""
    cfg = _cfg(tmp_path, fpb=8, num_bars=4)
    prog = ChordProgression(chords=[("Cmaj7", 4)] * 4, tempo=120.0, time_signature="4/4")
    # Не падает на валидации (но дойдёт до записи seed.npz и вернёт params).
    params = CMTAdapter(cfg).prepare(prog, tmp_path / "work")
    assert "seed_npz_path" in params
```

- [ ] **Step 2: Реализовать валидацию + чтение hparams**

В `pipeline/pipeline/adapters/cmt.py` заменить тело `prepare`:

```python
def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
    import yaml
    from pipeline.adapters._cmt_input import build_seed, progression_to_chroma

    cfg = self._config

    # 1. Прочитать гипер-параметры модели из hparams.yaml.
    with open(cfg.hparams_path, "r") as f:
        hparams = yaml.safe_load(f)
    model_cfg = hparams["model"]
    frame_per_bar: int = int(model_cfg["frame_per_bar"])
    num_bars: int = int(model_cfg["num_bars"])

    # 2. Pipeline-валидация (то что зависит от нашего конфига).
    valid_strategies = {"tonic_held", "tonic_quarters", "custom_pkl"}
    if cfg.seed_strategy not in valid_strategies:
        raise ValueError(
            f"unknown seed_strategy={cfg.seed_strategy!r}; "
            f"expected one of {sorted(valid_strategies)}"
        )
    if cfg.seed_strategy == "custom_pkl" and cfg.custom_pkl_path is None:
        raise ValueError("seed_strategy=custom_pkl requires custom_pkl_path")
    if cfg.prime_bars < 1 or cfg.prime_bars > num_bars:
        raise ValueError(
            f"prime_bars must be in [1, num_bars={num_bars}], got {cfg.prime_bars}"
        )

    # 3. Конвертация и затравка (внутри валит ValueError если progression
    #    несовместима с frame_per_bar / num_bars).
    chroma = progression_to_chroma(progression, frame_per_bar, num_bars)
    prime_len = cfg.prime_bars * frame_per_bar
    prime_pitch, prime_rhythm = build_seed(progression, cfg, frame_per_bar, prime_len)

    # 4. Сохранение и возврат params для runner'а.
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seed_npz = tmp_dir / "seed.npz"
    import numpy as np
    np.savez(seed_npz, chord_chroma=chroma, prime_pitch=prime_pitch, prime_rhythm=prime_rhythm)

    return {
        "checkpoint_path":   str(cfg.checkpoint_path),
        "hparams_path":      str(cfg.hparams_path),
        "model_repo_path":   str(cfg.repo_path),
        "seed_npz_path":     str(seed_npz),
        "output_midi_path":  str(tmp_dir / "raw.mid"),
        "topk":              cfg.topk,
        "device":            cfg.device,
    }
```

- [ ] **Step 3: Тесты зелёные**

```bash
.venv/bin/python -m pytest tests/adapters/test_cmt_validation.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/pipeline/adapters/cmt.py pipeline/tests/adapters/test_cmt_validation.py
git commit -m "feat(cmt): read hparams.yaml in prepare, validate against model size"
```

---

## Task 9: `CMTAdapter.prepare` — проверка возвращаемого dict

Вся реализация уже сделана в Task 8. Этот таск — отдельные тесты на возвращаемый dict.

**Files:**
- Create: `pipeline/tests/adapters/test_cmt_prepare.py`

- [ ] **Step 1: Failing tests**

```python
from pathlib import Path
import yaml

import numpy as np
import pytest

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
    assert not (forbidden & params.keys())
```

- [ ] **Step 2: Тесты зелёные** (Task 8 уже реализовал prepare, тесты должны пройти сразу)

```bash
.venv/bin/python -m pytest tests/adapters/test_cmt_prepare.py -v
.venv/bin/python -m pytest -v
```

Если что-то падает — поправить реализацию `prepare` в `cmt.py`.

- [ ] **Step 3: Commit**

```bash
git add pipeline/tests/adapters/test_cmt_prepare.py
git commit -m "test(cmt): verify CMTAdapter.prepare contract (seed.npz, params dict)"
```

---

## Task 10: `CMTAdapter.extract_melody`

**Files:**
- Modify: `pipeline/pipeline/adapters/cmt.py`
- Create: `pipeline/tests/adapters/test_cmt_extract_melody.py`

- [ ] **Step 1: Failing tests**

```python
from pathlib import Path

import pretty_midi
import pytest

from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig


def _cfg(tmp_path: Path) -> CMTPipelineConfig:
    return CMTPipelineConfig(
        checkpoint_path=tmp_path / "x", hparams_path=tmp_path / "x", repo_path=tmp_path / "x",
    )


def _two_track_midi(out: Path, melody_name: str = "melody") -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    melody = pretty_midi.Instrument(program=0, name=melody_name)
    for i, p in enumerate([60, 64, 67, 71]):
        melody.notes.append(pretty_midi.Note(80, p, i * 0.5, (i + 1) * 0.5))
    pm.instruments.append(melody)
    chord = pretty_midi.Instrument(program=0, name="chord")
    for p in [60, 64, 67]:
        chord.notes.append(pretty_midi.Note(60, p, 0.0, 2.0))
    pm.instruments.append(chord)
    pm.write(str(out))


def test_extract_melody_picks_melody_track(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _two_track_midi(midi)
    inst = CMTAdapter(_cfg(tmp_path)).extract_melody(midi)
    assert inst.name == "melody"
    assert len(inst.notes) == 4
    assert isinstance(inst, pretty_midi.Instrument)


def test_extract_melody_raises_when_track_missing(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _two_track_midi(midi, melody_name="not_melody")
    with pytest.raises(ValueError, match="melody"):
        CMTAdapter(_cfg(tmp_path)).extract_melody(midi)
```

- [ ] **Step 2: Реализовать `extract_melody`**

```python
def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
    pm = pretty_midi.PrettyMIDI(str(raw_midi_path))
    for inst in pm.instruments:
        if inst.name == "melody":
            return inst
    names = [i.name for i in pm.instruments]
    raise ValueError(
        f"melody track 'melody' not found in {raw_midi_path} (have: {names})"
    )
```

- [ ] **Step 3: Тесты зелёные**

```bash
.venv/bin/python -m pytest tests/adapters/test_cmt_extract_melody.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/pipeline/adapters/cmt.py pipeline/tests/adapters/test_cmt_extract_melody.py
git commit -m "feat(cmt): implement CMTAdapter.extract_melody"
```

---

## Task 11: Регистрация в `pipeline/config.py`

**Files:**
- Modify: `pipeline/pipeline/config.py`

- [ ] **Step 1: Дополнить config**

После строки `MINGUS_REPO_PATH: ...` добавить:

```python
CMT_REPO_PATH:        Path = DIPLOMA_ROOT / "models" / "CMT-pytorch"
CMT_RESULT_DIR:       Path = CMT_REPO_PATH / "result" / "smoke_wjazzd_5epochs"
CMT_CHECKPOINT_PATH:  Path = CMT_RESULT_DIR / "smoke_5epochs.pth.tar"  # ← подмена весов
CMT_HPARAMS_PATH:     Path = CMT_RESULT_DIR / "hparams.yaml"           # ← гипер-параметры в паре с весами
```

Заменить `from pipeline.adapters.cmt import CMTAdapter` на:

```python
from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig
```

В `MODEL_RUNNER_SCRIPT`:

```python
"cmt": RUNNERS_ROOT / "cmt_runner.py",
```

В `ADAPTERS` заменить `"cmt": CMTAdapter()` на:

```python
"cmt": CMTAdapter(CMTPipelineConfig(
    checkpoint_path=CMT_CHECKPOINT_PATH,
    hparams_path=CMT_HPARAMS_PATH,
    repo_path=CMT_REPO_PATH,
    seed_strategy="tonic_held",
    prime_bars=1,
    topk=5,
    device="cpu",
)),
```

- [ ] **Step 2: Импорт работает**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -c "from pipeline.config import ADAPTERS, MODEL_RUNNER_SCRIPT, CMT_CHECKPOINT_PATH; print(ADAPTERS['cmt']); print(MODEL_RUNNER_SCRIPT['cmt']); print(CMT_CHECKPOINT_PATH)"
```

- [ ] **Step 3: Полный pytest**

```bash
.venv/bin/python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/pipeline/config.py
git commit -m "feat(cmt): register CMT adapter and runner script in pipeline config"
```

---

## Task 12: `cmt_runner.py`

**Files:**
- Create: `pipeline/runners/cmt_runner.py`

Этот таск без юнит-тестов в pipeline-venv (runner импортирует torch). Smoke-тест через subprocess.

- [ ] **Step 1: Создать runner**

```python
#!/usr/bin/env python3
"""CMT runner: запускается интерпретатором models/CMT-pytorch/.venv/bin/python.

Контракт:
- читает JSON payload со stdin (см. pipeline.runner_protocol)
- params: checkpoint_path, hparams_path, model_repo_path, seed_npz_path,
          output_midi_path, topk, device
- импортирует CMT-API напрямую, вызывает model.sampling и pitch_to_midi
- пишет MIDI в output_midi_path
- exit 0 при успехе, exit 1 при ошибке (traceback в stderr)

НЕ импортирует ничего из pipeline.* — живёт в CMT-venv.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    payload = json.loads(sys.stdin.read())
    params = payload["params"]
    checkpoint_path = Path(params["checkpoint_path"])
    hparams_path    = Path(params["hparams_path"])
    repo            = Path(params["model_repo_path"])
    seed_npz_path   = Path(params["seed_npz_path"])
    output_midi     = Path(params["output_midi_path"])
    topk: int       = int(params["topk"])
    device_name: str = params["device"]

    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    import yaml
    from model import ChordConditionedMelodyTransformer
    from utils.utils import pitch_to_midi

    device = torch.device(device_name)

    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)
    model_config = hparams["model"]

    model = ChordConditionedMelodyTransformer(**model_config).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    npz = np.load(seed_npz_path)
    chord       = torch.tensor(npz["chord_chroma"]).float().unsqueeze(0).to(device)
    prime_pitch = torch.tensor(npz["prime_pitch"]).long().unsqueeze(0).to(device)
    prime_rhythm = torch.tensor(npz["prime_rhythm"]).long().unsqueeze(0).to(device)

    with torch.no_grad():
        result = model.sampling(prime_rhythm, prime_pitch, chord, topk=topk)

    pitch_idx = result["pitch"][0].cpu().numpy()
    chord_arr = chord[0].cpu().numpy()

    output_midi.parent.mkdir(parents=True, exist_ok=True)
    pitch_to_midi(
        pitch_idx, chord_arr,
        frame_per_bar=model.frame_per_bar,
        save_path=str(output_midi),
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
```

- [ ] **Step 2: Smoke-test runner изолированно**

```bash
cd /Users/maxos/PythonProjects/diploma
mkdir -p /tmp/cmt_smoke

pipeline/.venv/bin/python -c "
from pathlib import Path
from pipeline.adapters.cmt import CMTAdapter, CMTPipelineConfig
from pipeline.progression import ChordProgression
cfg = CMTPipelineConfig(
    checkpoint_path=Path('/Users/maxos/PythonProjects/diploma/models/CMT-pytorch/result/smoke_wjazzd_5epochs/smoke_5epochs.pth.tar'),
    hparams_path=Path('/Users/maxos/PythonProjects/diploma/models/CMT-pytorch/result/smoke_wjazzd_5epochs/hparams.yaml'),
    repo_path=Path('/Users/maxos/PythonProjects/diploma/models/CMT-pytorch'),
)
prog = ChordProgression(
    chords=[('Cmaj7', 4),('Am7', 4),('Dm7', 4),('G7', 4)] * 2,
    tempo=120.0, time_signature='4/4',
)
params = CMTAdapter(cfg).prepare(prog, Path('/tmp/cmt_smoke'))
import json
print(json.dumps({'model':'cmt','run_id':'smoke','params':params}))
" > /tmp/cmt_smoke/payload.json

cat /tmp/cmt_smoke/payload.json | models/CMT-pytorch/.venv/bin/python pipeline/runners/cmt_runner.py
echo "exit: $?"
ls -la /tmp/cmt_smoke/raw.mid
```

Ожидается: exit 0, raw.mid создан.

- [ ] **Step 3: Проверить MIDI**

```bash
pipeline/.venv/bin/python -c "
import pretty_midi
pm = pretty_midi.PrettyMIDI('/tmp/cmt_smoke/raw.mid')
print('instruments:', [i.name for i in pm.instruments])
melody = [i for i in pm.instruments if i.name == 'melody'][0]
print('melody notes:', len(melody.notes))
"
```

- [ ] **Step 4: Cleanup + commit**

```bash
trash /tmp/cmt_smoke
git add pipeline/runners/cmt_runner.py
git commit -m "feat(cmt): add cmt_runner.py for CMT-venv subprocess execution"
```

---

## Task 13: e2e через `pipeline.cli`

- [ ] **Step 1: e2e на sample.json**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
```

Ожидается: `cmt = ok`, файлы в `output/melody_only/cmt_*.mid` и `output/with_chords/cmt_*.mid`.

Если error — открыть `pipeline/output/_tmp/<run_id>/cmt/stderr.log`, починить.

- [ ] **Step 2: Создать альтернативную progression**

`pipeline/test_progressions/alt.json`:

```json
{
    "tempo": 120.0,
    "time_signature": "4/4",
    "chords": [
        ["F7", 4], ["Bm7", 4], ["E7", 4], ["A7", 4],
        ["F7", 4], ["Bm7", 4], ["E7", 4], ["A7", 4]
    ]
}
```

- [ ] **Step 3: Прод-инвариант — разные progression → разные мелодии**

```bash
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
.venv/bin/python -m pipeline.cli generate test_progressions/alt.json

.venv/bin/python -c "
import pretty_midi, glob
midis = sorted(glob.glob('output/melody_only/cmt_*.mid'))
pm_a = pretty_midi.PrettyMIDI(midis[-2])
pm_b = pretty_midi.PrettyMIDI(midis[-1])
notes_a = [(n.pitch, round(n.start, 2)) for n in pm_a.instruments[0].notes]
notes_b = [(n.pitch, round(n.start, 2)) for n in pm_b.instruments[0].notes]
print('a (first 5 notes):', notes_a[:5])
print('b (first 5 notes):', notes_b[:5])
assert notes_a != notes_b, 'CMT returned same melody for different progressions'
print('OK: different progressions yield different melodies')
"
```

- [ ] **Step 4: Полный pytest финально**

```bash
.venv/bin/python -m pytest -v
```

Все зелёные, никаких xfail.

- [ ] **Step 5: Commit alt.json**

```bash
git add pipeline/test_progressions/alt.json
git commit -m "test(cmt): add alternative progression for chord-conditioning verification"
```

- [ ] **Step 6: Сводка пользователю**

В чат: количество коммитов в ветке, подтверждение `cmt = ok` на двух progression, мелодии не совпадают, pitch range и notes count. **Не предлагать merge в master без явного указания.**

---

## Self-Review

**Spec coverage:**
- §1 DoD → Task 13 проверяет cmt=ok, разные progression → разные мелодии, pitch range.
- §2 архитектурный инвариант → Tasks 5-13 не лезут в общие модули.
- §3 CMT-вход → Tasks 6, 7 (диагностика реальных pkl).
- §4 lazy чтение hparams → Task 8 (hparams читается в `prepare`).
- §5 3 стратегии затравки → Task 7.
- §6 chord_chroma converter → Task 6.
- §7 CMTAdapter API → Tasks 5, 8, 9, 10.
- §8 runner контракт → Task 12.
- §9 форк патчи → Task 1.
- §10 pipeline config → Task 11.
- §11 артефакты на хосте → Task 2 (сохранение result/).
- §12 структура файлов → Tasks 5-12 создают именно её.
- §13 тесты → Tasks 5-10 покрывают.
- §14 риски — diagnostic-шаги в Tasks 6 и 7 закрывают риски 1 и 6.

**Type/key consistency:**
- `prepare` returns dict с ключами `{checkpoint_path, hparams_path, model_repo_path, seed_npz_path, output_midi_path, topk, device}` (Tasks 8, 11, 12 — совпадают).
- `seed.npz` ключи `{chord_chroma, prime_pitch, prime_rhythm}` (Task 8, Task 12 — совпадают).
- `melody` track name (Task 10, Task 12, `utils/utils.py:60`) — совпадают.
- `frame_per_bar` / `num_bars` берутся из hparams в адаптере (Task 8) и в runner'е (Task 12) — каждый читает свою копию.

**Hardcode scan:** проверено — нет магических чисел `8`, `16`, `128`, `50`, `"4/4"` в адаптере. Все размеры приходят параметрами или из hparams.yaml. Константы `ONSET_RHYTHM_IDX` / `SUSTAIN_RHYTHM_IDX` / `SUSTAIN_PITCH_IDX` — свойства pitch/rhythm vocab CMT-кода (utils/utils.py), не зависят от чекпоинта; зафиксированы по результатам diagnostic в Task 7.

**Scope check:** план фокусирован на single-feature (CMT integration). Не размывается.
