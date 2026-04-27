# BebopNet Integration — Implementation Plan (production)

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (fresh subagent per task, TDD discipline). Каждый таск — failing test → implementation → passing test → commit.

**Goal.** Production-интеграция BebopNet в pipeline. Любой `ChordProgression` → реальная chord-conditioned монофоническая мелодия Tenor Sax длиной ровно `progression.num_bars()` тактов. Замена весов = подмена `BEBOPNET_MODEL_DIR` или `BEBOPNET_CHECKPOINT_FILENAME` в `pipeline/pipeline/config.py`.

**Architecture.** `BebopNetAdapter` (immutable config) + общий `_xml_builders/jazz_xml.py` (рефакторинг из `mingus_xml.py`, переиспользуется MINGUS и BebopNet) + `bebopnet_runner.py` (subprocess в bebopnet-venv, ничего не импортирует из `pipeline.*`).

**Spec:** `docs/superpowers/specs/2026-04-27-pipeline-bebopnet-design.md`.

---

## Контекст для исполнителя

- **Ветка:** `feat/pipeline-bebopnet` (создаётся в Task 0 от master).
- **Pipeline-venv:** `pipeline/.venv`.
- **BebopNet-venv:** `models/bebopnet-code/.venv` (создаётся в Task 4).
- **Чекпоинт уже на машине:** `models/bebopnet-code/training_results/transformer/model/{model.pt, converter_and_duration.pkl, args.json, train_model.yml}`. Не должны потеряться при превращении локальной директории в submodule.
- **Форк:** `https://github.com/kudrmax/bebopnet-code` (создан пользователем, clean upstream). Туда уйдут 5 коммитов-патчей в Task 2.

BebopNet API (см. `models/bebopnet-code/jazz_rnn/B_next_note_prediction/`):
- Главный entry point: `MusicGenerator.create_song(...)` (внутри `music_generator.py`). Точная сигнатура читается на стадии Task 11 — runner должен вызывать его минуя CLI `generate_from_xml.py`.
- Чекпоинт: `torch.load(model_path, map_location=device, weights_only=False)['state_dict']` (формат после патчей).
- Vocabulary: pickled `converter_and_duration.pkl` (bidict). Bidict-monkey-patch уже в форке (Task 2 patch 2).
- Вход: MusicXML (тот же формат что MINGUS).
- Выход: один монофоничный MIDI-track (Tenor Sax program 65), ticks_per_beat=10080.

Все коммиты — Conventional Commits (`feat(bebopnet):`, `chore(bebopnet):`, `test(bebopnet):`, `refactor(jazz_xml):`). Push в `feat/pipeline-bebopnet` разрешён. Push в форк CMT в Task 2 явно разрешён пользователем. Никаких других git-операций без явного указания.

---

## Task 0: Подготовка ветки

- [ ] **Step 1: Создать ветку от master**

```bash
cd /Users/maxos/PythonProjects/diploma
git status
git checkout master
git checkout -b feat/pipeline-bebopnet
```

Ожидается: ветка `feat/pipeline-bebopnet` создана от свежего master.

- [ ] **Step 2: Закоммитить spec + plan**

```bash
git add docs/superpowers/specs/2026-04-27-pipeline-bebopnet-design.md \
        docs/superpowers/plans/2026-04-27-pipeline-bebopnet.md
git commit -m "docs(bebopnet): production integration spec and plan"
```

---

## Task 1: Refactor MINGUS xml builder в общий jazz_xml.py

**Files:**
- Create: `pipeline/pipeline/_xml_builders/jazz_xml.py`
- Delete: `pipeline/pipeline/_xml_builders/mingus_xml.py` (через `trash` после миграции)
- Modify: `pipeline/pipeline/adapters/mingus.py` (использует новый импорт)
- Rename: `pipeline/tests/_xml_builders/test_mingus_xml.py` → `pipeline/tests/_xml_builders/test_jazz_xml.py`

**Цель:** общий XML builder без зависимости от `MingusPipelineConfig`. Принимает `seed_strategy` и `custom_xml_path` параметрами напрямую. MINGUS продолжает работать 1-в-1.

- [ ] **Step 1: Прочитать текущую реализацию**

```bash
cd /Users/maxos/PythonProjects/diploma
cat pipeline/pipeline/_xml_builders/mingus_xml.py
ls pipeline/tests/_xml_builders/
```

Зафиксировать сигнатуру `build_mingus_xml(progression, config, out_path)` и три стратегии `tonic_whole` / `tonic_quarters` / `custom_xml`.

- [ ] **Step 2: Создать новый `jazz_xml.py` параметризованным**

Создать `pipeline/pipeline/_xml_builders/jazz_xml.py`:

```python
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from music21 import (
    chord as m21_chord, harmony, instrument, key, meter, note, stream, tempo,
)

from pipeline.chord_vocab import parse_chord
from pipeline.progression import ChordProgression


_TONIC_OCTAVE = 5  # C5 = MIDI 72; средне-высокий регистр сакса


SeedStrategy = Literal["tonic_whole", "tonic_quarters", "custom_xml"]


def _instrument_for(name: str):
    """Возвращает music21-инструмент по имени.

    Поддерживаются основные имена использующиеся в pipeline. Для остальных —
    fallback на TenorSaxophone (это и так используется обоими моделями).
    """
    name_norm = name.lower().replace(" ", "")
    mapping = {
        "tenorsax":         instrument.TenorSaxophone,
        "tenorsaxophone":   instrument.TenorSaxophone,
        "altosax":          instrument.AltoSaxophone,
        "sopranoSax":       instrument.SopranoSaxophone,
    }
    cls = mapping.get(name_norm, instrument.TenorSaxophone)
    return cls()


def build_xml(
    progression: ChordProgression,
    seed_strategy: SeedStrategy,
    custom_xml_path: Path | None,
    out_path: Path,
    melody_instrument_name: str = "Tenor Sax",
) -> None:
    """Пишет MusicXML, готовый для music21-парсера (использует MINGUS и BebopNet).

    seed_strategy:
      - tonic_whole    — 1 whole-нота тоники в каждом баре
      - tonic_quarters — 4 quarter-ноты тоники в каждом баре
      - custom_xml     — копирует custom_xml_path в out_path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if seed_strategy == "custom_xml":
        if custom_xml_path is None:
            raise ValueError("seed_strategy=custom_xml requires custom_xml_path")
        shutil.copy(custom_xml_path, out_path)
        return

    bpb = progression.beats_per_bar()
    if progression.total_beats() % bpb != 0:
        raise ValueError(
            f"progression total_beats={progression.total_beats()} not divisible by "
            f"beats_per_bar={bpb} (time_signature={progression.time_signature})"
        )

    for chord_str, beats in progression.chords:
        if beats % bpb != 0:
            raise ValueError(
                f"chord {chord_str} duration {beats} not multiple of {bpb}; "
                f"sub-bar chord placement not supported in MVP"
            )

    if not progression.chords:
        raise ValueError("progression has no chords")

    score = stream.Score()
    score.metadata = score.metadata or None
    part = stream.Part()
    part.id = "P1"
    part.partName = "Melody"
    part.insert(0, _instrument_for(melody_instrument_name))

    measure_idx = 1
    chord_iter = iter(progression.chords)
    cur_chord, cur_remaining = next(chord_iter)
    for bar in range(progression.num_bars()):
        m = stream.Measure(number=measure_idx)
        if measure_idx == 1:
            m.append(meter.TimeSignature(progression.time_signature))
            m.append(tempo.MetronomeMark(number=progression.tempo))
            m.append(key.KeySignature(0))  # C-мажор
        cs = harmony.ChordSymbol(cur_chord)
        m.insert(0, cs)
        root_idx, _quality = parse_chord(cur_chord)
        tonic_pitch = note.Pitch()
        tonic_pitch.midi = root_idx + 12 * (_TONIC_OCTAVE + 1)
        if seed_strategy == "tonic_whole":
            n = note.Note(tonic_pitch)
            n.quarterLength = bpb
            m.append(n)
        elif seed_strategy == "tonic_quarters":
            for _ in range(bpb):
                n = note.Note(tonic_pitch)
                n.quarterLength = 1
                m.append(n)
        else:
            raise ValueError(f"unsupported seed_strategy: {seed_strategy}")
        part.append(m)
        cur_remaining -= bpb
        if cur_remaining <= 0 and bar < progression.num_bars() - 1:
            cur_chord, cur_remaining = next(chord_iter)
        measure_idx += 1

    remaining = list(chord_iter)
    assert not remaining, (
        f"chord iterator has {len(remaining)} unused chords after "
        f"{progression.num_bars()} bars; total_beats validation should have caught this"
    )

    score.insert(0, part)
    score.write("musicxml", fp=str(out_path))
```

- [ ] **Step 3: Обновить `MingusAdapter.prepare` на использование нового builder'а**

В `pipeline/pipeline/adapters/mingus.py` найти строки:

```python
from pipeline._xml_builders.mingus_xml import build_mingus_xml
...
build_mingus_xml(progression, cfg, xml_path)
```

Заменить на:

```python
from pipeline._xml_builders.jazz_xml import build_xml
...
build_xml(
    progression,
    seed_strategy=cfg.seed_strategy,
    custom_xml_path=cfg.custom_xml_path,
    out_path=xml_path,
    melody_instrument_name=cfg.melody_instrument_name,
)
```

(Ничего другого в `mingus.py` менять не надо — все три стратегии MINGUS совпадают с jazz_xml.)

- [ ] **Step 4: Перенести и переименовать тесты**

```bash
cd /Users/maxos/PythonProjects/diploma
ls pipeline/tests/_xml_builders/  # проверить структуру
```

Если файл `test_mingus_xml.py` существует:
- Прочитать его содержимое.
- Создать новый `test_jazz_xml.py` с тем же содержимым, но:
  - Импорт `from pipeline._xml_builders.jazz_xml import build_xml`.
  - Вызовы `build_mingus_xml(prog, cfg, path)` → `build_xml(prog, seed_strategy=cfg.seed_strategy, custom_xml_path=cfg.custom_xml_path, out_path=path)`.
  - Если тесты делали fake `MingusPipelineConfig` чтобы передать seed_strategy — заменить на прямой параметр.
- Через `trash` удалить старый `test_mingus_xml.py`.

Затем добавить в `test_jazz_xml.py` минимум один новый тест на `melody_instrument_name`:

```python
def test_build_xml_uses_specified_instrument(tmp_path):
    """Параметр melody_instrument_name влияет на XML."""
    prog = ChordProgression(
        chords=[("Cmaj7", 4)] * 2, tempo=120.0, time_signature="4/4",
    )
    xml_path = tmp_path / "out.xml"
    build_xml(
        prog,
        seed_strategy="tonic_whole",
        custom_xml_path=None,
        out_path=xml_path,
        melody_instrument_name="Tenor Sax",
    )
    content = xml_path.read_text()
    assert "Tenor Saxophone" in content or "tenor" in content.lower()
```

- [ ] **Step 5: Удалить старый `mingus_xml.py`**

```bash
trash pipeline/pipeline/_xml_builders/mingus_xml.py
```

- [ ] **Step 6: Прогнать pytest**

```bash
cd pipeline
.venv/bin/python -m pytest -v 2>&1 | tail -10
```

Ожидается: все тесты зелёные, MINGUS продолжает работать (test_mingus_*, test_pipeline e2e). Если что-то падает — исправить в `jazz_xml.py` или `mingus.py`. Не трогать другие файлы.

- [ ] **Step 7: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/pipeline/_xml_builders/jazz_xml.py \
        pipeline/pipeline/adapters/mingus.py \
        pipeline/tests/_xml_builders/
git rm pipeline/pipeline/_xml_builders/mingus_xml.py 2>/dev/null || true
git status --short
git commit -m "refactor(jazz_xml): generalize MINGUS xml builder for reuse with BebopNet"
```

---

## Task 2: Push patches to kudrmax/bebopnet-code fork

**Authorization:** push в форк `kudrmax/bebopnet-code` явно разрешён пользователем для этого таска.

**Files:** работаем во временной директории `/tmp/bebopnet_fork_work/`, не трогаем `/Users/maxos/PythonProjects/diploma`.

- [ ] **Step 1: Клонировать форк**

```bash
mkdir -p /tmp/bebopnet_fork_work
cd /tmp/bebopnet_fork_work
git clone https://github.com/kudrmax/bebopnet-code.git
cd bebopnet-code
git log --oneline -3
```

Ожидается: история upstream `shunithaviv/bebopnet-code` без наших патчей.

- [ ] **Step 2: Patch 1 — закомментировать SongLabels import**

В `jazz_rnn/A_data_prep/gather_data_from_xml.py` строка 14:

```python
from jazz_rnn.utils.music.vectorXmlConverter import *
from jazz_rnn.C_reward_induction.online_tagger_gauge import SongLabels
```

Заменить вторую строку на:

```python
from jazz_rnn.utils.music.vectorXmlConverter import *
# from jazz_rnn.C_reward_induction.online_tagger_gauge import SongLabels  # disabled: tkinter; SongLabels not used here
```

```bash
git add jazz_rnn/A_data_prep/gather_data_from_xml.py
git commit -m "fix: comment out tkinter SongLabels import (not used in inference)"
```

- [ ] **Step 3: Patch 2 — bidict monkey-patch + torch.load weights_only=False**

В `jazz_rnn/B_next_note_prediction/generate_from_xml.py` после строки `import torch` (примерно строка 15) вставить:

```python

# bidict 0.14 -> 0.20+ rename: _fwdm/_invm became fwdm/invm. Pickled converter relies on the old names.
import bidict as _bd
if not hasattr(_bd.BidictBase, '_fwdm'):
    _bd.BidictBase._fwdm = property(
        lambda self: self.fwdm,
        lambda self, v: object.__setattr__(self, 'fwdm', v),
    )
if not hasattr(_bd.BidictBase, '_invm'):
    _bd.BidictBase._invm = property(
        lambda self: self.invm,
        lambda self, v: object.__setattr__(self, 'invm', v),
    )
```

И в том же файле (около строки 124) найти строку:

```python
            model.load_state_dict(torch.load(model_path))
```

Заменить на:

```python
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
```

```bash
git add jazz_rnn/B_next_note_prediction/generate_from_xml.py
git commit -m "fix: bidict 0.20+ compat (monkey-patch _fwdm/_invm) + torch.load(weights_only=False)"
```

- [ ] **Step 4: Patch 3 — int(c) для numpy 2.x bool**

В `jazz_rnn/B_next_note_prediction/music_generator.py` строка 372:

```python
        next_chord = [self.chords[measure_idx % self.head_len][c] if last_note_in_measure_mask[ind] == 0 else
```

Заменить `[c]` на `[int(c)]`:

```python
        next_chord = [self.chords[measure_idx % self.head_len][int(c)] if last_note_in_measure_mask[ind] == 0 else
```

```bash
git add jazz_rnn/B_next_note_prediction/music_generator.py
git commit -m "fix: numpy 2.x bool indexing in music_generator (int(c))"
```

- [ ] **Step 5: Patch 4 — PyTorch 2.x bool tensor API**

В `jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py`:

(a) около строки 326-337 — заменить блок:

```python
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 4:
                attn_score = attn_score.float().masked_fill(
                    attn_mask, -float('inf')).type_as(attn_score)
```

на:

```python
        if attn_mask is not None and attn_mask.any().item():
            attn_mask_b = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask
            if attn_mask_b.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask_b[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask_b.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask_b[:, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask_b.dim() == 4:
                attn_score = attn_score.float().masked_fill(
                    attn_mask_b, -float('inf')).type_as(attn_score)
```

(b) около строки 619 заменить:

```python
        eos_mask = 1 - eos_mask
```

на:

```python
        eos_mask = (1 - eos_mask.long()).bool() if eos_mask.dtype == torch.bool else 1 - eos_mask
```

```bash
git add jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py
git commit -m "fix: PyTorch 2.x bool tensor API (masked_fill, 1-bool subtract)"
```

- [ ] **Step 6: Создать requirements-py312.txt + .gitignore**

Создать `requirements-py312.txt`:

```
# BebopNet — runtime requirements for Python 3.12 на arm64 macOS.
# Используется pipeline'ом jazz-generation-pipeline для inference.
#
# Install:
#   python3.12 -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements-py312.txt

torch
numpy
music21
lxml
pretty_midi
pyyaml
bidict
ConfigArgParse
imbalanced-learn
scipy
tqdm
matplotlib
pandas
```

Создать `.gitignore`:

```
# Virtualenv
.venv/

# Python
__pycache__/
*.py[cod]
*.egg-info/

# Training artefacts (могут быть сотнями MB)
training_results/
output/

# OS
.DS_Store
.idea/
```

```bash
git add requirements-py312.txt .gitignore
git commit -m "add: requirements-py312.txt and .gitignore for Python 3.12 inference runtime"
```

- [ ] **Step 7: Push в форк**

```bash
git push origin master
git log --oneline -7
```

Ожидается: 5 коммитов в форке. Если push падает — STOP, report BLOCKED со stderr.

- [ ] **Step 8: Cleanup**

```bash
cd /Users/maxos/PythonProjects/diploma
trash /tmp/bebopnet_fork_work
```

---

## Task 3: Превратить локальный bebopnet-code в submodule

- [ ] **Step 1: Сохранить training_results во временную папку**

```bash
cd /Users/maxos/PythonProjects/diploma
mv models/bebopnet-code/training_results /tmp/bebopnet_training_results_backup
ls /tmp/bebopnet_training_results_backup/transformer/model/
```

Ожидается: 4 файла (model.pt, converter_and_duration.pkl, args.json, train_model.yml). Если есть .venv внутри bebopnet-code — она потеряется при удалении, это ОК (пересоздадим в Task 4).

- [ ] **Step 2: Удалить локальную директорию**

```bash
trash models/bebopnet-code
```

- [ ] **Step 3: Обновить корневой `.gitignore`**

В `/Users/maxos/PythonProjects/diploma/.gitignore` найти блок:

```
# Submodule-исключения — иначе models/* спрятал бы их.
models/*
!models/MINGUS
!models/CMT-pytorch
```

Добавить ещё одну строку:

```
# Submodule-исключения — иначе models/* спрятал бы их.
models/*
!models/MINGUS
!models/CMT-pytorch
!models/bebopnet-code
```

- [ ] **Step 4: Добавить как submodule**

```bash
git submodule add https://github.com/kudrmax/bebopnet-code.git models/bebopnet-code
git status
```

Ожидается: в staged — `.gitmodules` и `models/bebopnet-code` (как submodule pointer 160000).

- [ ] **Step 5: Добавить branch=master в .gitmodules для consistency**

В `/Users/maxos/PythonProjects/diploma/.gitmodules` найти блок:

```
[submodule "models/bebopnet-code"]
	path = models/bebopnet-code
	url = https://github.com/kudrmax/bebopnet-code.git
```

Добавить строку `branch = master`:

```
[submodule "models/bebopnet-code"]
	path = models/bebopnet-code
	url = https://github.com/kudrmax/bebopnet-code.git
	branch = master
```

- [ ] **Step 6: Вернуть training_results**

```bash
mv /tmp/bebopnet_training_results_backup /Users/maxos/PythonProjects/diploma/models/bebopnet-code/training_results
ls /Users/maxos/PythonProjects/diploma/models/bebopnet-code/training_results/transformer/model/
```

- [ ] **Step 7: Verify submodule с патчами**

```bash
cd /Users/maxos/PythonProjects/diploma/models/bebopnet-code
git log --oneline -7
ls requirements-py312.txt
grep "weights_only=False" jazz_rnn/B_next_note_prediction/generate_from_xml.py | head -1
grep "int(c)" jazz_rnn/B_next_note_prediction/music_generator.py | head -1
```

Ожидается: 5 наших коммитов + upstream history; файлы requirements-py312.txt и патчи на месте.

- [ ] **Step 8: Verify gitignore работает (внутри submodule training_results gitignored)**

```bash
cd /Users/maxos/PythonProjects/diploma/models/bebopnet-code
git status --short
```

Ожидается: пусто (training_results/ ignored через submodule .gitignore).

- [ ] **Step 9: Commit submodule в основном репо**

```bash
cd /Users/maxos/PythonProjects/diploma
git add .gitignore .gitmodules models/bebopnet-code
git status --short
git commit -m "chore(bebopnet): add bebopnet-code as git submodule on kudrmax fork"
```

---

## Task 4: Создать BebopNet-venv и проверить импорты

- [ ] **Step 1: Создать venv и установить зависимости**

```bash
cd /Users/maxos/PythonProjects/diploma/models/bebopnet-code
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-py312.txt
```

Ожидается: установка проходит без критических ошибок. Возможны warnings о deprecated пакетах — это OK.

- [ ] **Step 2: Smoke-проверить импорты**

```bash
cd /Users/maxos/PythonProjects/diploma/models/bebopnet-code
.venv/bin/python -c "
import sys, os
sys.path.insert(0, os.getcwd())
import torch, music21, numpy as np
import bidict
from jazz_rnn.B_next_note_prediction.transformer.mem_transformer import MemTransformerLM
from jazz_rnn.B_next_note_prediction.music_generator import MusicGenerator
print('OK', 'torch', torch.__version__, 'music21', music21.__version__, 'numpy', np.__version__)
"
```

Ожидается: `OK torch <ver> music21 <ver> numpy <ver>` без ошибок.

Если что-то не импортируется — поправить `requirements-py312.txt` в форке (новый коммит → push), пересобрать venv. Не коммитить локальный venv (gitignored).

---

## Task 5: Убрать BebopNetAdapter из stub-параметризации

**Files:**
- Modify: `pipeline/tests/adapters/test_base.py`

После Task 6 `BebopNetAdapter()` без аргументов будет падать `TypeError`.

- [ ] **Step 1: Baseline**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/adapters/test_base.py -v
```

Ожидается: все тесты зелёные, BebopNetAdapter в parametrize.

- [ ] **Step 2: Удалить BebopNetAdapter из импортов и parametrize**

В `pipeline/tests/adapters/test_base.py`:

Удалить строку:

```python
from pipeline.adapters.bebopnet import BebopNetAdapter
```

В обоих `@pytest.mark.parametrize("AdapterCls", [...])` убрать `BebopNetAdapter`. Останется:

```python
@pytest.mark.parametrize("AdapterCls", [
    EC2VaeAdapter, ComMUAdapter, PolyffusionAdapter,
])
```

- [ ] **Step 3: Тесты зелёные**

```bash
.venv/bin/python -m pytest tests/adapters/test_base.py -v
.venv/bin/python -m pytest -v
```

Ожидается: всё зелёное (минус 2 stub-проверки BebopNet).

- [ ] **Step 4: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/tests/adapters/test_base.py
git commit -m "test(bebopnet): remove BebopNetAdapter from stub parametrize before refactor"
```

---

## Task 6: BebopNetPipelineConfig + BebopNetAdapter constructor

**Files:**
- Modify: `pipeline/pipeline/adapters/bebopnet.py` (полностью переписать)
- Create: `pipeline/tests/adapters/test_bebopnet_config.py`

- [ ] **Step 1: Failing tests**

Создать `pipeline/tests/adapters/test_bebopnet_config.py`:

```python
from pathlib import Path

import pytest

from pipeline.adapters.bebopnet import BebopNetAdapter, BebopNetPipelineConfig


def _make_config(tmp_path: Path) -> BebopNetPipelineConfig:
    return BebopNetPipelineConfig(
        model_dir=tmp_path / "model",
        repo_path=tmp_path / "repo",
    )


def test_bebopnet_adapter_requires_config():
    with pytest.raises(TypeError):
        BebopNetAdapter()  # type: ignore[call-arg]


def test_bebopnet_adapter_stores_config(tmp_path: Path):
    cfg = _make_config(tmp_path)
    adapter = BebopNetAdapter(cfg)
    assert adapter._config is cfg


def test_bebopnet_pipeline_config_defaults(tmp_path: Path):
    cfg = _make_config(tmp_path)
    assert cfg.checkpoint_filename == "model.pt"
    assert cfg.seed_strategy == "tonic_whole"
    assert cfg.custom_xml_path is None
    assert cfg.melody_instrument_name == "Tenor Sax"
    assert cfg.temperature == 1.0
    assert cfg.top_p is True
    assert cfg.beam_search == "measure"
    assert cfg.beam_width == 2
    assert cfg.device == "cpu"
```

- [ ] **Step 2: Реализовать skeleton**

Полностью переписать `pipeline/pipeline/adapters/bebopnet.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pretty_midi

from pipeline.adapters.base import ModelAdapter
from pipeline.progression import ChordProgression


@dataclass
class BebopNetPipelineConfig:
    """Все настройки BebopNet на уровне pipeline. immutable после init.

    Размеры/sampling-параметры BebopNet — здесь. Замена весов на другую папку
    с (model.pt, converter_and_duration.pkl, args.json, train_model.yml) =
    подмена model_dir / checkpoint_filename без правок этого dataclass.
    """

    model_dir: Path
    repo_path: Path
    checkpoint_filename: str = "model.pt"
    seed_strategy: Literal["tonic_whole", "tonic_quarters", "custom_xml"] = "tonic_whole"
    custom_xml_path: Path | None = None
    melody_instrument_name: str = "Tenor Sax"
    temperature: float = 1.0
    top_p: bool = True
    beam_search: Literal["", "note", "measure"] = "measure"
    beam_width: int = 2
    device: str = "cpu"


class BebopNetAdapter(ModelAdapter):
    def __init__(self, config: BebopNetPipelineConfig) -> None:
        self._config = config

    def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
        raise NotImplementedError("model bebopnet: prepare not implemented yet")

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        raise NotImplementedError("model bebopnet: extract_melody not implemented yet")
```

**Внимание:** после этой правки `pipeline/pipeline/config.py` где-то делает `BebopNetAdapter()` без аргументов — это сломает import. Нужно временно заменить на placeholder с реальными путями (по аналогии с тем что делал Task 5 для CMT). В `pipeline/pipeline/config.py`:

Заменить:

```python
from pipeline.adapters.bebopnet import BebopNetAdapter
```

на:

```python
from pipeline.adapters.bebopnet import BebopNetAdapter, BebopNetPipelineConfig
```

И в `ADAPTERS["bebopnet"]` заменить `BebopNetAdapter()` на:

```python
"bebopnet":    BebopNetAdapter(BebopNetPipelineConfig(
    model_dir=DIPLOMA_ROOT / "models" / "bebopnet-code" / "training_results" / "transformer" / "model",
    repo_path=DIPLOMA_ROOT / "models" / "bebopnet-code",
)),
```

(Task 10 потом перепишет это на правильные именованные константы.)

- [ ] **Step 3: Зелёные тесты**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/adapters/test_bebopnet_config.py -v
.venv/bin/python -m pytest -v
```

Ожидается: 3 новых теста PASS, полный pytest зелёный.

- [ ] **Step 4: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/pipeline/adapters/bebopnet.py \
        pipeline/pipeline/config.py \
        pipeline/tests/adapters/test_bebopnet_config.py
git commit -m "feat(bebopnet): add BebopNetPipelineConfig and BebopNetAdapter constructor"
```

---

## Task 7: Валидация в `prepare`

**Files:**
- Modify: `pipeline/pipeline/adapters/bebopnet.py`
- Create: `pipeline/tests/adapters/test_bebopnet_validation.py`

- [ ] **Step 1: Failing tests**

Создать `pipeline/tests/adapters/test_bebopnet_validation.py`:

```python
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
```

- [ ] **Step 2: Реализовать валидацию**

В `pipeline/pipeline/adapters/bebopnet.py` заменить тело `prepare`:

```python
def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
    self._validate(progression)
    raise NotImplementedError("model bebopnet: prepare not fully implemented yet")

def _validate(self, progression: ChordProgression) -> None:
    cfg = self._config
    valid_strategies = {"tonic_whole", "tonic_quarters", "custom_xml"}
    if cfg.seed_strategy not in valid_strategies:
        raise ValueError(
            f"unknown seed_strategy={cfg.seed_strategy!r}; "
            f"expected one of {sorted(valid_strategies)}"
        )
    if cfg.seed_strategy == "custom_xml" and cfg.custom_xml_path is None:
        raise ValueError("seed_strategy=custom_xml requires custom_xml_path")
    if not progression.chords:
        raise ValueError("progression has no chords")
```

- [ ] **Step 3: Тесты зелёные**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/adapters/test_bebopnet_validation.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/pipeline/adapters/bebopnet.py \
        pipeline/tests/adapters/test_bebopnet_validation.py
git commit -m "feat(bebopnet): validate seed_strategy and progression in prepare"
```

---

## Task 8: `BebopNetAdapter.prepare` — полная реализация

**Files:**
- Modify: `pipeline/pipeline/adapters/bebopnet.py`
- Create: `pipeline/tests/adapters/test_bebopnet_prepare.py`

- [ ] **Step 1: Failing tests**

Создать `pipeline/tests/adapters/test_bebopnet_prepare.py`:

```python
from pathlib import Path

import pytest

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
    assert "<harmony" in content or "<root" in content  # MusicXML chord-symbol
    assert "Cmaj7" in content or "C-maj7" in content or "C major seventh" in content.lower() or "kind text=\"major-seventh\"" in content.lower()


def test_prepare_num_measures_equals_num_bars(tmp_path: Path):
    """num_measures должен быть = progression.num_bars()."""
    cfg = _cfg(tmp_path)
    work = tmp_path / "work"
    params = BebopNetAdapter(cfg).prepare(_prog(n_bars=8), work)
    assert params["num_measures"] == 8

    work2 = tmp_path / "work2"
    params2 = BebopNetAdapter(cfg).prepare(_prog(n_bars=4), work2)
    assert params2["num_measures"] == 4


def test_prepare_different_progressions_yield_different_xml(tmp_path: Path):
    """Прод-инвариант: разные progression → разные input.xml."""
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
```

- [ ] **Step 2: Тесты падают на NotImplementedError**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/adapters/test_bebopnet_prepare.py -v
```

- [ ] **Step 3: Реализовать `prepare` полностью**

В `pipeline/pipeline/adapters/bebopnet.py` заменить тело `prepare` на полную реализацию:

```python
def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
    from pipeline._xml_builders.jazz_xml import build_xml

    self._validate(progression)
    cfg = self._config
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    xml_path = tmp_dir / "input.xml"
    midi_path = tmp_dir / "raw.mid"

    build_xml(
        progression,
        seed_strategy=cfg.seed_strategy,
        custom_xml_path=cfg.custom_xml_path,
        out_path=xml_path,
        melody_instrument_name=cfg.melody_instrument_name,
    )

    return {
        "input_xml_path":      str(xml_path),
        "output_midi_path":    str(midi_path),
        "model_dir":           str(cfg.model_dir),
        "checkpoint_filename": cfg.checkpoint_filename,
        "model_repo_path":     str(cfg.repo_path),
        "num_measures":        progression.num_bars(),
        "temperature":         cfg.temperature,
        "top_p":               cfg.top_p,
        "beam_search":         cfg.beam_search,
        "beam_width":          cfg.beam_width,
        "device":              cfg.device,
    }
```

- [ ] **Step 4: Тесты зелёные**

```bash
.venv/bin/python -m pytest tests/adapters/test_bebopnet_prepare.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 5: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/pipeline/adapters/bebopnet.py pipeline/tests/adapters/test_bebopnet_prepare.py
git commit -m "feat(bebopnet): implement prepare (build XML via shared builder)"
```

---

## Task 9: `BebopNetAdapter.extract_melody`

**Files:**
- Modify: `pipeline/pipeline/adapters/bebopnet.py`
- Create: `pipeline/tests/adapters/test_bebopnet_extract_melody.py`

BebopNet пишет один монофоничный трек. По умолчанию берём `pm.instruments[0]`. Если 0 инструментов — `ValueError`.

- [ ] **Step 1: Failing tests**

Создать `pipeline/tests/adapters/test_bebopnet_extract_melody.py`:

```python
from pathlib import Path

import pretty_midi
import pytest

from pipeline.adapters.bebopnet import BebopNetAdapter, BebopNetPipelineConfig


def _cfg(tmp_path: Path) -> BebopNetPipelineConfig:
    return BebopNetPipelineConfig(
        model_dir=tmp_path / "model", repo_path=tmp_path / "repo",
    )


def _single_track_midi(out: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    melody = pretty_midi.Instrument(program=65, name="Tenor Sax")
    for i, p in enumerate([60, 64, 67, 71]):
        melody.notes.append(pretty_midi.Note(80, p, i * 0.5, (i + 1) * 0.5))
    pm.instruments.append(melody)
    pm.write(str(out))


def _empty_midi(out: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.write(str(out))


def test_extract_melody_picks_first_track(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _single_track_midi(midi)
    inst = BebopNetAdapter(_cfg(tmp_path)).extract_melody(midi)
    assert isinstance(inst, pretty_midi.Instrument)
    assert len(inst.notes) == 4


def test_extract_melody_raises_when_no_instruments(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _empty_midi(midi)
    with pytest.raises(ValueError, match="no instruments|empty|melody"):
        BebopNetAdapter(_cfg(tmp_path)).extract_melody(midi)
```

- [ ] **Step 2: Реализовать**

В `pipeline/pipeline/adapters/bebopnet.py`:

```python
def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
    pm = pretty_midi.PrettyMIDI(str(raw_midi_path))
    if not pm.instruments:
        raise ValueError(
            f"no instruments in {raw_midi_path}; cannot extract melody"
        )
    return pm.instruments[0]
```

- [ ] **Step 3: Тесты зелёные**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/adapters/test_bebopnet_extract_melody.py -v
.venv/bin/python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/pipeline/adapters/bebopnet.py pipeline/tests/adapters/test_bebopnet_extract_melody.py
git commit -m "feat(bebopnet): implement extract_melody (first instrument from monophonic MIDI)"
```

---

## Task 10: Регистрация в `pipeline/config.py`

**Files:**
- Modify: `pipeline/pipeline/config.py`

- [ ] **Step 1: Дополнить config**

В `pipeline/pipeline/config.py` после CMT-блока добавить:

```python
BEBOPNET_REPO_PATH:           Path = DIPLOMA_ROOT / "models" / "bebopnet-code"
BEBOPNET_MODEL_DIR:           Path = BEBOPNET_REPO_PATH / "training_results" / "transformer" / "model"
BEBOPNET_CHECKPOINT_FILENAME: str  = "model.pt"  # ← подмена весов
```

В `MODEL_RUNNER_SCRIPT` добавить:

```python
"bebopnet": RUNNERS_ROOT / "bebopnet_runner.py",
```

В `ADAPTERS` заменить provisional `"bebopnet": BebopNetAdapter(BebopNetPipelineConfig(...))` (из Task 6) на:

```python
"bebopnet":    BebopNetAdapter(BebopNetPipelineConfig(
    model_dir=BEBOPNET_MODEL_DIR,
    repo_path=BEBOPNET_REPO_PATH,
    checkpoint_filename=BEBOPNET_CHECKPOINT_FILENAME,
    seed_strategy="tonic_whole",
    temperature=1.0,
    top_p=True,
    beam_search="measure",
    beam_width=2,
    device="cpu",
)),
```

- [ ] **Step 2: Импорт работает**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -c "
from pipeline.config import (
    ADAPTERS, MODEL_RUNNER_SCRIPT, BEBOPNET_MODEL_DIR, BEBOPNET_CHECKPOINT_FILENAME,
)
print('bebopnet adapter:', ADAPTERS['bebopnet'])
print('runner:', MODEL_RUNNER_SCRIPT['bebopnet'])
print('model_dir:', BEBOPNET_MODEL_DIR)
print('checkpoint:', BEBOPNET_CHECKPOINT_FILENAME)
print('model_dir exists:', BEBOPNET_MODEL_DIR.exists())
print('checkpoint file exists:', (BEBOPNET_MODEL_DIR / BEBOPNET_CHECKPOINT_FILENAME).exists())
"
```

Ожидается: 6 строк без exception. Оба `exists` = True.

- [ ] **Step 3: Полный pytest**

```bash
.venv/bin/python -m pytest -v
```

Все зелёные.

- [ ] **Step 4: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/pipeline/config.py
git commit -m "feat(bebopnet): register adapter and runner script in pipeline config"
```

---

## Task 11: `bebopnet_runner.py` + smoke-тест

**Files:**
- Create: `pipeline/runners/bebopnet_runner.py`

Этот таск без юнит-тестов в pipeline-venv (runner импортирует bebopnet модули из subprocess). Проверка — реальный smoke-test через subprocess (Step 3).

- [ ] **Step 1: Изучить точную сигнатуру `MusicGenerator`**

```bash
cd /Users/maxos/PythonProjects/diploma
grep -n "class MusicGenerator\|def __init__\|def create_song\|def generate" models/bebopnet-code/jazz_rnn/B_next_note_prediction/music_generator.py | head -30
```

Изучить какие именно параметры принимает `MusicGenerator.__init__` и `create_song` (или эквивалент). Это нужно для правильного вызова в runner.

Если основной entry point — `generate_from_xml.generate_from_xml(args)` (с argparse-стилем args), можно либо:
(a) использовать `argparse.Namespace(...)` чтобы передать args напрямую,
(b) вызвать `MusicGenerator` напрямую без CLI-обёртки.

Выбираем (b) если возможно — меньше зависимостей. Если (a) — runner собирает Namespace из params.

- [ ] **Step 2: Создать runner**

Создать `pipeline/runners/bebopnet_runner.py`:

```python
#!/usr/bin/env python3
"""BebopNet runner: запускается интерпретатором models/bebopnet-code/.venv/bin/python.

Контракт:
- читает JSON payload со stdin (см. pipeline.runner_protocol)
- params: input_xml_path, output_midi_path, model_dir, checkpoint_filename,
          model_repo_path, num_measures, temperature, top_p, beam_search,
          beam_width, device
- импортирует BebopNet API, парсит XML, вызывает generation, пишет MIDI
- exit 0 при успехе, exit 1 при ошибке (traceback в stderr)

НЕ импортирует ничего из pipeline.* — живёт в bebopnet-venv.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
from argparse import Namespace
from pathlib import Path


def main() -> int:
    payload = json.loads(sys.stdin.read())
    params = payload["params"]
    input_xml      = Path(params["input_xml_path"])
    output_midi    = Path(params["output_midi_path"])
    model_dir      = Path(params["model_dir"])
    checkpoint     = params["checkpoint_filename"]
    repo           = Path(params["model_repo_path"])
    num_measures   = int(params["num_measures"])
    temperature    = float(params["temperature"])
    top_p          = bool(params["top_p"])
    beam_search    = params["beam_search"]
    beam_width     = int(params["beam_width"])
    device_name    = params["device"]

    os.chdir(repo)
    sys.path.insert(0, str(repo))

    # Импорты — после chdir/path setup
    import torch
    import music21 as m21

    # bidict-monkey-patch уже применён в форке через
    # generate_from_xml.py (см. fork commit "fix: bidict 0.20+ compat").
    # Здесь импортируем тот файл, чтобы patch применился до загрузки converter.
    from jazz_rnn.B_next_note_prediction import generate_from_xml as _gfx_module  # noqa: F401
    from jazz_rnn.B_next_note_prediction.transformer.mem_transformer import MemTransformerLM
    from jazz_rnn.B_next_note_prediction.music_generator import MusicGenerator

    device = torch.device(device_name)
    torch.manual_seed(0)

    # 1. Загрузить args.json и converter
    with open(model_dir / "args.json", "r") as f:
        args_json = json.load(f)
    with open(model_dir / "converter_and_duration.pkl", "rb") as f:
        converter_and_duration = pickle.load(f)

    # 2. Построить модель
    # Список kwargs совпадает с тем что MemTransformerLM принимает в __init__.
    # Берём только те поля args_json, которые модель действительно ожидает —
    # точные имена смотрим в jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py:MemTransformerLM
    # __init__ при имплементации Step 1 (см. plan).
    # Здесь оставляем сырой передачей, как делает generate_from_xml.py:
    model = MemTransformerLM(**args_json)
    model.load_state_dict(torch.load(model_dir / checkpoint, map_location=device, weights_only=False))
    model.converter = converter_and_duration  # generate_from_xml.py делает то же самое
    model = model.to(device)
    model.eval()

    # 3. Парсим XML
    score = m21.converter.parse(str(input_xml))

    # 4. Создаём генератор
    # Точный конструктор подгоняется в Step 3 smoke-тесте если что-то упадёт.
    args = Namespace(
        no_cuda=(device_name == "cpu"),
        temperature=temperature,
        top_p=top_p,
        non_stochstic_search=False,
        beam_search=beam_search,
        beam_width=beam_width,
        beam_depth=1,
        batch_size=1,
        threshold=0.0,
        score_model="",
        verbose=False,
        seed=None,
        num_measures=num_measures,
        num_heads=0,
    )

    generator = MusicGenerator(
        model=model,
        converter=converter_and_duration,
        score=score,
        args=args,
        device=device,
    )

    # 5. Генерация — точное имя метода уточняем по реальному коду.
    # Стандартное имя в BebopNet — generator.create_song(...) или .generate(...).
    pm = generator.create_song(num_measures=num_measures)

    # 6. Сохраняем MIDI
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(pm, "write"):
        pm.write(str(output_midi))
    elif hasattr(pm, "midi_filename") and pm.midi_filename:
        # generator уже записал файл по своему пути — копируем
        from shutil import copy as _copy
        _copy(pm.midi_filename, output_midi)
    else:
        # Fallback: возможно generator вернул music21 stream
        pm.write("midi", fp=str(output_midi))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
```

**Замечание для исполнителя.** Точные имена `MusicGenerator` параметров и метода генерации могут отличаться от ожидаемых выше. На стадии smoke-теста Step 3 ниже:
- Если падает на `MemTransformerLM(**args_json)` — посмотреть какие kwargs реально принимает `__init__`, отфильтровать.
- Если падает на `MusicGenerator(...)` — посмотреть `__init__` сигнатуру и привести аргументы в соответствие.
- Если падает на `generator.create_song(...)` — найти реальное имя метода (`generate_song`, `improvise`, и т.п.).

Главный принцип: **runner — чистая обёртка**. Если для правильного вызова нужны model-specific преобразования (например build kwargs из args.json), они живут в runner'е, не в адаптере.

- [ ] **Step 3: Smoke-test runner изолированно**

```bash
cd /Users/maxos/PythonProjects/diploma
mkdir -p /tmp/bebopnet_smoke

pipeline/.venv/bin/python -c "
from pathlib import Path
from pipeline.adapters.bebopnet import BebopNetAdapter, BebopNetPipelineConfig
from pipeline.progression import ChordProgression
cfg = BebopNetPipelineConfig(
    model_dir=Path('/Users/maxos/PythonProjects/diploma/models/bebopnet-code/training_results/transformer/model'),
    repo_path=Path('/Users/maxos/PythonProjects/diploma/models/bebopnet-code'),
)
prog = ChordProgression(
    chords=[('Cmaj7', 4),('Am7', 4),('Dm7', 4),('G7', 4)] * 2,
    tempo=120.0, time_signature='4/4',
)
params = BebopNetAdapter(cfg).prepare(prog, Path('/tmp/bebopnet_smoke'))
import json
print(json.dumps({'model':'bebopnet','run_id':'smoke','params':params}))
" > /tmp/bebopnet_smoke/payload.json

cat /tmp/bebopnet_smoke/payload.json | models/bebopnet-code/.venv/bin/python pipeline/runners/bebopnet_runner.py
echo "exit code: $?"
ls -la /tmp/bebopnet_smoke/raw.mid 2>&1
```

Ожидается: exit 0, `raw.mid` существует, размер > 0.

Если упало — читать traceback из stderr. Типичные проблемы и где править:
- `MemTransformerLM.__init__() got an unexpected keyword 'X'` — отфильтровать `X` из args.json перед передачей.
- `MusicGenerator.__init__() missing argument 'Y'` — посмотреть default значения в исходнике и добавить в Namespace или kwargs.
- `MusicGenerator has no attribute 'create_song'` — найти реальное имя метода, заменить.

Все правки — **в runner'е**. Адаптер не трогать.

- [ ] **Step 4: Проверить MIDI**

```bash
pipeline/.venv/bin/python -c "
import pretty_midi
pm = pretty_midi.PrettyMIDI('/tmp/bebopnet_smoke/raw.mid')
print('instruments:', [(i.name, i.program) for i in pm.instruments])
print('total notes:', sum(len(i.notes) for i in pm.instruments))
melody = pm.instruments[0]
if melody.notes:
    print('pitch range:', min(n.pitch for n in melody.notes), '..', max(n.pitch for n in melody.notes))
print('duration:', pm.get_end_time(), 'sec')
"
```

Ожидается: ≥ 1 инструмент, total notes > 0, pitch range разумный (примерно 50..90 для tenor sax).

- [ ] **Step 5: Cleanup + commit**

```bash
trash /tmp/bebopnet_smoke
cd /Users/maxos/PythonProjects/diploma
git add pipeline/runners/bebopnet_runner.py
git commit -m "feat(bebopnet): add bebopnet_runner.py for bebopnet-venv subprocess execution"
```

---

## Task 12: e2e через `pipeline.cli` + chord-conditioning verification

- [ ] **Step 1: e2e на sample.json**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
```

Ожидается: в выводе таблицы `bebopnet = ok`. MINGUS и CMT тоже остаются `ok`.

Если `bebopnet = error` — открыть `pipeline/output/_tmp/<run_id>/bebopnet/stderr.log`, прочитать traceback, починить в runner'е (не в адаптере).

- [ ] **Step 2: Проверить выходы**

```bash
.venv/bin/python -c "
import pretty_midi, glob
midi = sorted(glob.glob('output/melody_only/bebopnet_*.mid'))[-1]
pm = pretty_midi.PrettyMIDI(midi)
print('path:', midi)
print('instruments:', [(i.name, i.program) for i in pm.instruments])
print('total notes:', sum(len(i.notes) for i in pm.instruments))
melody = pm.instruments[0]
pitches = [n.pitch for n in melody.notes]
print('pitch range:', min(pitches), '..', max(pitches), 'over', len(pitches), 'notes')
"
```

Ожидается: 1 трек в melody_only, notes > 0, диапазон в районе 50..90.

- [ ] **Step 3: Прод-инвариант на двух прогрессиях**

```bash
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
.venv/bin/python -m pipeline.cli generate test_progressions/alt.json

.venv/bin/python -c "
import pretty_midi, glob
midis = sorted(glob.glob('output/melody_only/bebopnet_*.mid'))
print('found:', midis[-2:])
pm_a = pretty_midi.PrettyMIDI(midis[-2])
pm_b = pretty_midi.PrettyMIDI(midis[-1])
notes_a = [(n.pitch, round(n.start, 3)) for n in pm_a.instruments[0].notes]
notes_b = [(n.pitch, round(n.start, 3)) for n in pm_b.instruments[0].notes]
print('a (first 5):', notes_a[:5])
print('b (first 5):', notes_b[:5])
assert notes_a != notes_b, 'BebopNet returned identical melody for different progressions'
print('OK: different progressions yield different melodies')
"
```

- [ ] **Step 4: Полный pytest финально**

```bash
.venv/bin/python -m pytest -v
```

Все зелёные.

- [ ] **Step 5: Финальный коммит (если есть что коммитить)**

```bash
cd /Users/maxos/PythonProjects/diploma
git status --short
```

`output/` файлы — runtime, gitignored, не коммитятся. Если `git status` пустой — пропускаем.

- [ ] **Step 6: Сводка пользователю**

В чат:
- Сколько коммитов в ветке `feat/pipeline-bebopnet`.
- Подтверждение `bebopnet = ok` на двух progression, мелодии не совпадают.
- pitch range, notes count.
- Не предлагать merge в master без явного указания пользователя.

---

## Self-Review

**Spec coverage:**
- §1 DoD → Task 12 проверяет bebopnet=ok, разные progression → разные мелодии, длина = num_bars, MIDI валидность.
- §2 архитектурный инвариант → Tasks 6-12 не лезут в общие модули (за исключением Task 1 рефакторинга — он явно scope этой задачи и сохраняет MINGUS поведение).
- §3 BebopNet-вход → Tasks 8 (XML build), 11 (runner usage).
- §4 3 стратегии затравки → Task 1 (jazz_xml.build_xml поддерживает все 3) и Task 8 (использует через cfg.seed_strategy).
- §5 CMT-style adapter contract → Tasks 6, 7, 8, 9.
- §6 валидация → Task 7.
- §7 runner контракт → Task 11.
- §8 venv → Tasks 2, 4.
- §9 pipeline config → Task 10.
- §10 рефакторинг xml builder → Task 1.
- §11 артефакты → Task 3 (сохранение training_results/).
- §12 структура файлов → создаётся Tasks 1, 6, 8, 11, 10.
- §13 5 patches в форк → Task 2.
- §14 тесты → Tasks 1 (jazz_xml), 6 (config), 7 (validation), 8 (prepare), 9 (extract_melody), 12 (e2e).
- §15 риски — все митигации описаны прямо в Task 11 step 3.

**Placeholder scan:** проверено — все шаги имеют точный код или точные команды. Нет "TBD", нет "implement appropriate ..."

**Type/key consistency:**
- `prepare` returns dict с ключами `{input_xml_path, output_midi_path, model_dir, checkpoint_filename, model_repo_path, num_measures, temperature, top_p, beam_search, beam_width, device}` (Tasks 8, 10, 11 — совпадают).
- `melody_instrument_name` (Task 6 default = "Tenor Sax") передаётся в `build_xml` (Task 8) и в `_instrument_for(...)` в Task 1 — все согласованы.
- `seed_strategy` Literal значения `tonic_whole | tonic_quarters | custom_xml` совпадают между Task 1 (jazz_xml), Task 6 (config), Task 7 (validation).
