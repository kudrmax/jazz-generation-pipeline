# Pipeline + MINGUS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Построить общий каркас пайплайна `generate_all(progression) → {model: paths_or_error}` с реализованной MINGUS-моделью; остальные пять моделей (BebopNet, EC²-VAE, CMT, ComMU, Polyffusion) присутствуют как stub-adapter'ы возвращающие `not implemented`.

**Architecture:** Слоистая. Pipeline-venv (свой набор зависимостей) оркестрирует subprocess'ы model-venv'ов. Каждая модель = `ModelAdapter` (с `prepare(...)` и `extract_melody(...)`) в pipeline-venv + `<model>_runner.py` исполняемый в model-venv. Postprocess + chord_render — общие, ничего не знают про конкретные модели.

**Tech Stack:** Python 3.12, pretty_midi, music21, numpy, pytest. Тесты — unit + один end-to-end smoke. Никаких HTTP/Docker — только subprocess + JSON через stdin + файлы на диске.

**Источник дизайна:** `docs/superpowers/specs/2026-04-27-pipeline-mingus-design.md`. Все архитектурные решения — оттуда. План — про реализацию.

## Ветка и контекст работы

**ВАЖНО для исполнителей плана — git-флоу:**

- `master` = «итоговое repo». На момент старта работы — пуст (один initial-commit). Никаких прямых коммитов на `master`. Master обновляется только PR-мёрджем `feat/pipeline-mingus → master` после завершения и одобрения.
- `develop` — рабочая ветка с историческим контекстом ВКР (5 md в корне: README, THESIS_PLAN, PIPELINE_SPEC, WORK_PLAN, CONTEXT; archive/, articles/, finetune/, старый pipeline/). **Не используется для текущей работы.** Эти 5 md из корня develop **НЕ переезжают** на master — они остаются в develop как контекст разработки. План самодостаточен; для выполнения они не нужны.
- `feat/pipeline-mingus` — текущая рабочая ветка, отделённая от **пустого** master. Все коммиты по этому плану — сюда. Старого `pipeline/` в этой ветке нет, начинаем с нуля.

**Что есть в feat/pipeline-mingus на старте (после bootstrap-коммита):**
- `.gitignore` (перенесён из develop одной строкой `git checkout develop -- .gitignore`)
- `docs/superpowers/specs/2026-04-27-pipeline-mingus-design.md`
- `docs/superpowers/plans/2026-04-27-pipeline-mingus.md` (этот файл)

Всё остальное — `pipeline/`, тесты, runner — создаётся задачами T0–T14 ниже.

**Что лежит на физическом диске, но не в git** (gitignored, не пропадает при checkout):
- `models/MINGUS/` со своим `.venv`, pretrained весами и DATA.json
- `models/<other>/` для остальных моделей

**Pre-requisites (проверяются в Task 0):**
- `models/MINGUS/.venv` существует и работает (memory `project_mingus_setup.md`)
- `models/MINGUS/A_preprocessData/data/DATA.json` существует (115 MB, иначе один раз `python A_preprocessData/data_preprocessing.py --format xml` в MINGUS-venv ~3 минуты)
- `models/MINGUS/B_train/models/{pitchModel,durationModel}/MINGUS COND I-C-NC-B-BE-O Epochs 100.pt` существуют

---

## File Structure

Все пути относительно корня репо (`/Users/maxos/PythonProjects/diploma/`).

```
pipeline/                                  # новая папка для всего пайплайна
├── .venv/                                 # изолированный venv (создаётся в Task 0)
├── pyproject.toml                         # минимальная конфигурация пакета
├── requirements.txt                       # pretty_midi, music21, numpy, pytest
├── pipeline/                              # python-пакет «pipeline»
│   ├── __init__.py                        # пусто
│   ├── progression.py                     # dataclass ChordProgression + JSON I/O
│   ├── chord_vocab.py                     # parse_chord, chord_to_pitches, fallback'и
│   ├── chord_render.py                    # render_chord_track(progression, out_path)
│   ├── postprocess.py                     # postprocess(melody, progression, ...) → 2 MIDI
│   ├── runner_protocol.py                 # RunnerError, run_runner_subprocess
│   ├── pipeline.py                        # MODEL_NAMES, generate_all(progression)
│   ├── cli.py                             # python -m pipeline.cli generate <path>
│   ├── config.py                          # пути, MELODY_PROGRAM, MODEL_VENV_PYTHON, MODEL_CONFIGS, ADAPTERS
│   ├── adapters/
│   │   ├── __init__.py                    # пусто
│   │   ├── base.py                        # ABC ModelAdapter
│   │   ├── mingus.py                      # MingusPipelineConfig + MingusAdapter
│   │   ├── bebopnet.py                    # stub
│   │   ├── ec2vae.py                      # stub
│   │   ├── cmt.py                         # stub
│   │   ├── commu.py                       # stub
│   │   └── polyffusion.py                 # stub
│   └── _xml_builders/
│       ├── __init__.py                    # пусто
│       └── mingus_xml.py                  # 3 стратегии XML-затравки для MINGUS
├── runners/                               # запускаются в model-venv'ах
│   └── mingus_runner.py                   # читает stdin JSON, вызывает MINGUS gen-функции
├── tests/                                 # pytest
│   ├── __init__.py
│   ├── conftest.py                        # фикстуры
│   ├── test_progression.py
│   ├── test_chord_vocab.py
│   ├── test_chord_render.py
│   ├── test_postprocess.py
│   ├── test_runner_protocol.py
│   ├── test_pipeline.py
│   ├── test_cli.py
│   └── adapters/
│       ├── __init__.py
│       ├── test_base.py
│       ├── test_mingus_xml.py
│       ├── test_mingus_prepare.py
│       └── test_mingus_extract_melody.py
├── test_progressions/
│   └── sample.json                        # Cmaj7-Am7-Dm7-G7 ×2
└── output/                                # создаётся в Task 0; gitignore
    ├── _tmp/<run_id>/                     # input.xml, raw.mid, stdout.log, stderr.log
    ├── melody_only/<model>_<run_id>.mid
    ├── with_chords/<model>_<run_id>.mid
    └── mp3/                               # сюда складывается результат convert_to_mp3.sh (если будет)
```

**На старте `pipeline/` не существует** — feat-ветка отделена от пустого master. Task 0 создаёт всё с нуля. Старая версия `pipeline/` (упомянутая в README ветки develop) остаётся в develop и сюда не переезжает.

---

## Task 0: Setup — venv, deps, структура папок, pre-requisites

**Files:**
- Create: `pipeline/requirements.txt`
- Create: `pipeline/pyproject.toml`
- Create: `pipeline/.venv/` (через `python3.12 -m venv`)
- Create: `pipeline/pipeline/__init__.py` (пустой)
- Create: `pipeline/pipeline/adapters/__init__.py` (пустой)
- Create: `pipeline/pipeline/_xml_builders/__init__.py` (пустой)
- Create: `pipeline/runners/__init__.py` (пустой; чтоб тесты могли импортировать при необходимости)
- Create: `pipeline/tests/__init__.py` (пустой)
- Create: `pipeline/tests/adapters/__init__.py` (пустой)

- [ ] **Step 1: Проверить pre-requisites MINGUS**

```bash
ls -la /Users/maxos/PythonProjects/diploma/models/MINGUS/.venv/bin/python
ls -la /Users/maxos/PythonProjects/diploma/models/MINGUS/A_preprocessData/data/DATA.json
ls -la "/Users/maxos/PythonProjects/diploma/models/MINGUS/B_train/models/pitchModel/MINGUS COND I-C-NC-B-BE-O Epochs 100.pt"
ls -la "/Users/maxos/PythonProjects/diploma/models/MINGUS/B_train/models/durationModel/MINGUS COND I-C-NC-B-BE-O Epochs 100.pt"
```

Expected: все 4 файла существуют. Если DATA.json отсутствует — остановиться и сказать пользователю что нужно один раз сделать `cd models/MINGUS && source .venv/bin/activate && export PYTHONPATH=$PWD && python A_preprocessData/data_preprocessing.py --format xml` (~3 минуты), потом продолжить план.

- [ ] **Step 2: Проверить что `pipeline/` НЕ существует**

```bash
ls /Users/maxos/PythonProjects/diploma/pipeline/ 2>&1 | head -1
```

Expected: `ls: ... No such file or directory` (мы на feat-ветке от пустого master, старого pipeline/ здесь нет — он остался в develop). Если папка вдруг существует — остановиться: значит ветка не та или произошёл случайный merge. Не продолжать.

- [ ] **Step 3: Создать venv и базовые файлы**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
/opt/homebrew/bin/python3.12 -m venv .venv
mkdir -p pipeline/adapters pipeline/_xml_builders runners tests/adapters test_progressions output/_tmp output/melody_only output/with_chords output/mp3
touch pipeline/__init__.py pipeline/adapters/__init__.py pipeline/_xml_builders/__init__.py runners/__init__.py tests/__init__.py tests/adapters/__init__.py
```

- [ ] **Step 4: Создать `requirements.txt`**

```
pretty_midi==0.2.10
music21==9.1.0
numpy>=1.24,<3.0
pytest>=8.0
```

- [ ] **Step 5: Создать `pyproject.toml`**

```toml
[project]
name = "pipeline"
version = "0.1.0"
description = "Chord-conditioned jazz solo generation pipeline"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 6: Установить зависимости**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
source .venv/bin/activate
http_proxy="" https_proxy="" pip install --upgrade pip
http_proxy="" https_proxy="" pip install -r requirements.txt
```

Expected: успешная установка без ошибок. Если есть прокси-блок — `http_proxy="" https_proxy=""` уже учтён.

- [ ] **Step 7: Smoke-проверка установки**

```bash
.venv/bin/python -c "import pretty_midi, music21, numpy, pytest; print('ok')"
```

Expected: `ok`.

- [ ] **Step 8: Дополнить корневой `.gitignore`**

Добавить строки в `/Users/maxos/PythonProjects/diploma/.gitignore`:

```
pipeline/.venv/
pipeline/output/
pipeline/**/__pycache__/
pipeline/.pytest_cache/
```

- [ ] **Step 9: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/requirements.txt pipeline/pyproject.toml pipeline/pipeline pipeline/runners pipeline/tests pipeline/test_progressions .gitignore
git commit -m "chore(pipeline): bootstrap venv, package skeleton and pytest config"
```

---

## Task 1: ChordProgression dataclass

**Files:**
- Create: `pipeline/pipeline/progression.py`
- Create: `pipeline/tests/test_progression.py`

- [ ] **Step 1: Написать failing tests**

Файл `pipeline/tests/test_progression.py`:

```python
import json
from pathlib import Path

import pytest

from pipeline.progression import ChordProgression


def test_progression_total_beats():
    p = ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4), ("Dm7", 2), ("G7", 2)],
        tempo=120.0,
        time_signature="4/4",
    )
    assert p.total_beats() == 12


def test_progression_num_bars_4_4():
    p = ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )
    assert p.num_bars() == 2


def test_progression_num_bars_3_4():
    p = ChordProgression(
        chords=[("Cmaj7", 3), ("Am7", 3)],
        tempo=120.0,
        time_signature="3/4",
    )
    assert p.num_bars() == 2


def test_from_json_round_trip(tmp_path: Path):
    src = tmp_path / "p.json"
    src.write_text(json.dumps({
        "tempo": 100.0,
        "time_signature": "4/4",
        "chords": [["Cmaj7", 4], ["G7", 4]],
    }))
    p = ChordProgression.from_json(src)
    assert p.tempo == 100.0
    assert p.chords == [("Cmaj7", 4), ("G7", 4)]

    dst = tmp_path / "out.json"
    p.to_json(dst)
    reloaded = ChordProgression.from_json(dst)
    assert reloaded == p


def test_from_json_rejects_unknown_field(tmp_path: Path):
    src = tmp_path / "p.json"
    src.write_text(json.dumps({
        "tempo": 100.0,
        "time_signature": "4/4",
        "chords": [["Cmaj7", 4]],
        "seed": [60, 64],
    }))
    with pytest.raises(ValueError, match="seed"):
        ChordProgression.from_json(src)
```

- [ ] **Step 2: Запустить тесты — должны упасть**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/test_progression.py -v
```

Expected: `ImportError` или `ModuleNotFoundError: pipeline.progression`.

- [ ] **Step 3: Написать минимальную реализацию**

Файл `pipeline/pipeline/progression.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


_ALLOWED_KEYS = {"chords", "tempo", "time_signature"}


@dataclass
class ChordProgression:
    """Композиционный замысел: только аккорды + tempo + размер.

    Никакой модель-специфики (затравки, температуры, чекпоинтов) здесь нет."""

    chords: list[tuple[str, int]] = field(default_factory=list)
    tempo: float = 120.0
    time_signature: str = "4/4"

    @classmethod
    def from_json(cls, path: Path) -> "ChordProgression":
        data = json.loads(Path(path).read_text())
        unknown = set(data.keys()) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"unknown ChordProgression fields: {sorted(unknown)}. "
                f"allowed: {sorted(_ALLOWED_KEYS)}"
            )
        chords = [tuple(c) for c in data["chords"]]
        return cls(
            chords=chords,
            tempo=float(data.get("tempo", 120.0)),
            time_signature=data.get("time_signature", "4/4"),
        )

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "tempo": self.tempo,
            "time_signature": self.time_signature,
            "chords": [list(c) for c in self.chords],
        }, indent=2))

    def total_beats(self) -> int:
        return sum(d for _, d in self.chords)

    def beats_per_bar(self) -> int:
        num, _denom = self.time_signature.split("/")
        return int(num)

    def num_bars(self) -> int:
        bpb = self.beats_per_bar()
        if self.total_beats() % bpb != 0:
            raise ValueError(
                f"total beats {self.total_beats()} not divisible by {bpb} (time signature {self.time_signature})"
            )
        return self.total_beats() // bpb
```

- [ ] **Step 4: Запустить тесты — должны пройти**

```bash
.venv/bin/python -m pytest tests/test_progression.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/progression.py pipeline/tests/test_progression.py
git commit -m "feat(pipeline): add ChordProgression dataclass with JSON I/O"
```

---

## Task 2: chord_vocab — парсинг аккордов и pitches

**Files:**
- Create: `pipeline/pipeline/chord_vocab.py`
- Create: `pipeline/tests/test_chord_vocab.py`

- [ ] **Step 1: Написать failing tests**

Файл `pipeline/tests/test_chord_vocab.py`:

```python
import pytest

from pipeline.chord_vocab import (
    ROOTS, QUALITIES,
    parse_chord, chord_to_pitches,
    QUALITY_FALLBACK_TO_TRIADS, EXTENDED_FALLBACK_TO_VOCAB,
)


def test_roots_count():
    assert len(ROOTS) == 12
    assert ROOTS[0] == "C"
    assert ROOTS[11] == "B"


def test_qualities_count():
    assert len(QUALITIES) == 7
    assert "maj7" in QUALITIES
    assert "dim7" in QUALITIES


@pytest.mark.parametrize("chord_str, expected", [
    ("C",      (0,  "maj")),
    ("Cmaj",   (0,  "maj")),
    ("C#",     (1,  "maj")),
    ("Db",     (1,  "maj")),
    ("Cmaj7",  (0,  "maj7")),
    ("Cm7",    (0,  "min7")),
    ("Cmin7",  (0,  "min7")),
    ("C7",     (0,  "7")),
    ("Bbm7",   (10, "min7")),
    ("F#dim7", (6,  "dim7")),
    ("Adim",   (9,  "dim")),
])
def test_parse_chord(chord_str, expected):
    assert parse_chord(chord_str) == expected


def test_parse_chord_unknown_root():
    with pytest.raises(ValueError, match="root"):
        parse_chord("Hmaj")


def test_parse_chord_unknown_quality():
    with pytest.raises(ValueError, match="quality"):
        parse_chord("Csus2")


def test_chord_to_pitches_cmaj7():
    # C major 7: C E G B in pretty_midi pitch numbers (4-я октава, C4 = 60)
    assert chord_to_pitches("Cmaj7") == [60, 64, 67, 71]


def test_chord_to_pitches_am7():
    # A min 7: A C E G — корень A3 = 57
    assert chord_to_pitches("Am7") == [57, 60, 64, 67]


def test_chord_to_pitches_dim7():
    # C dim 7: C Eb Gb Bbb (=A) — все интервалы по минор-3
    assert chord_to_pitches("Cdim7") == [60, 63, 66, 69]


def test_quality_fallback_to_triads():
    assert QUALITY_FALLBACK_TO_TRIADS["7"] == "maj"
    assert QUALITY_FALLBACK_TO_TRIADS["min7"] == "min"
    assert QUALITY_FALLBACK_TO_TRIADS["dim7"] == "dim"


def test_extended_fallback():
    assert EXTENDED_FALLBACK_TO_VOCAB["m7b5"] == "dim"
    assert EXTENDED_FALLBACK_TO_VOCAB["13"] == "7"
    assert EXTENDED_FALLBACK_TO_VOCAB["6"] == "maj"
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/test_chord_vocab.py -v
```

Expected: ImportError.

- [ ] **Step 3: Реализовать**

Файл `pipeline/pipeline/chord_vocab.py`:

```python
from __future__ import annotations

ROOTS: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES: list[str] = ["maj", "min", "7", "maj7", "min7", "dim", "dim7"]

_ROOT_ALIASES = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "C#": "C#", "D#": "D#", "F#": "F#", "G#": "G#", "A#": "A#",
}

_QUALITY_ALIASES = {
    "":     "maj",
    "m":    "min",
    "min":  "min",
    "maj":  "maj",
    "M":    "maj",
    "7":    "7",
    "M7":   "maj7",
    "maj7": "maj7",
    "m7":   "min7",
    "min7": "min7",
    "dim":  "dim",
    "dim7": "dim7",
}

_QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "dim7": [0, 3, 6, 9],
}

QUALITY_FALLBACK_TO_TRIADS: dict[str, str] = {
    "7": "maj", "maj7": "maj", "min7": "min", "dim7": "dim",
}

EXTENDED_FALLBACK_TO_VOCAB: dict[str, str] = {
    "m7b5": "dim",
    "alt": "7",
    "6": "maj",
    "9": "7",
    "13": "7",
}


def parse_chord(chord_str: str) -> tuple[int, str]:
    """`"Cmaj7"` → `(0, "maj7")`. Pitch class of root (0..11), normalized quality."""
    s = chord_str.strip()
    # выделяем root: либо одна буква + опциональный # или b, либо одна буква
    if len(s) >= 2 and s[1] in "#b":
        root_raw, rest = s[:2], s[2:]
    else:
        root_raw, rest = s[:1], s[1:]
    root_norm = _ROOT_ALIASES.get(root_raw, root_raw)
    if root_norm not in ROOTS:
        raise ValueError(f"unknown root in chord {chord_str!r}: {root_raw!r}")
    root_idx = ROOTS.index(root_norm)
    quality = _QUALITY_ALIASES.get(rest)
    if quality is None:
        raise ValueError(f"unknown quality in chord {chord_str!r}: {rest!r}")
    return root_idx, quality


def chord_to_pitches(chord_str: str, octave: int = 4) -> list[int]:
    """Возвращает MIDI-pitches аккорда в указанной октаве (по умолчанию 4-я: C4=60).

    Корень кладётся как pitch_class + 12*(octave+1) (pretty_midi convention).
    """
    root_idx, quality = parse_chord(chord_str)
    intervals = _QUALITY_INTERVALS[quality]
    base = root_idx + 12 * (octave + 1)
    return [base + i for i in intervals]
```

- [ ] **Step 4: Запустить тесты**

```bash
.venv/bin/python -m pytest tests/test_chord_vocab.py -v
```

Expected: все проходят. Если `chord_to_pitches("Am7")` возвращает `[69, 72, 76, 79]` вместо `[57, 60, 64, 67]` — значит надо учесть, что корень A в 4-й октаве уйдёт выше середины: правильное MIDI для A4 в pretty_midi — это 57+12=69, но мы хотим A3 (57) чтобы аккорд лежал в районе среднего регистра пианино. Поправить дефолт `octave=3` если нужно — пересчитать ожидания тестов и кода. (Решение: перетестируйте после реализации; если `Am7 → 69, 72, 76, 79` — обновите тест на эти значения и опишите выбор в docstring; критерий — все pitches должны лежать в `[40, 84]`.)

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/chord_vocab.py pipeline/tests/test_chord_vocab.py
git commit -m "feat(pipeline): add chord vocab parser with quality fallbacks"
```

---

## Task 3: chord_render — рендер chord-track piano MIDI

**Files:**
- Create: `pipeline/pipeline/chord_render.py`
- Create: `pipeline/tests/test_chord_render.py`

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/test_chord_render.py`:

```python
from pathlib import Path

import pretty_midi
import pytest

from pipeline.progression import ChordProgression
from pipeline.chord_render import render_chord_track


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4), ("Dm7", 4), ("G7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_render_chord_track_writes_file(tmp_path: Path):
    out = tmp_path / "chords.mid"
    render_chord_track(_basic_progression(), out)
    assert out.exists()
    pm = pretty_midi.PrettyMIDI(str(out))
    assert len(pm.instruments) == 1


def test_render_chord_track_has_one_piano_instrument(tmp_path: Path):
    out = tmp_path / "chords.mid"
    render_chord_track(_basic_progression(), out)
    pm = pretty_midi.PrettyMIDI(str(out))
    inst = pm.instruments[0]
    assert inst.program == 0  # Acoustic Grand Piano (GM 0-indexed)
    assert inst.is_drum is False


def test_render_chord_track_correct_number_of_chord_blocks(tmp_path: Path):
    """4 аккорда × 4 ноты voicing = 16 нот."""
    out = tmp_path / "chords.mid"
    render_chord_track(_basic_progression(), out)
    pm = pretty_midi.PrettyMIDI(str(out))
    notes = pm.instruments[0].notes
    # Cmaj7, Am7, Dm7, G7 — все четырёхголосные аккорды (3 интервала + корень)
    assert len(notes) == 16


def test_render_chord_track_durations_match_progression(tmp_path: Path):
    p = ChordProgression(
        chords=[("Cmaj7", 2), ("Am7", 6)],  # 2 beats + 6 beats
        tempo=120.0,
        time_signature="4/4",
    )
    out = tmp_path / "chords.mid"
    render_chord_track(p, out)
    pm = pretty_midi.PrettyMIDI(str(out))
    notes = pm.instruments[0].notes
    # каждый аккорд — 4 ноты с одинаковым start/end
    cmaj_notes = [n for n in notes if n.start == 0.0]
    am_notes   = [n for n in notes if n.start > 0.0]
    assert len(cmaj_notes) == 4
    assert len(am_notes)   == 4
    # 120 BPM → 1 beat = 0.5 sec
    # Cmaj7: 0.0 — 1.0; Am7: 1.0 — 4.0
    assert all(abs(n.end - 1.0) < 1e-6 for n in cmaj_notes)
    assert all(abs(n.start - 1.0) < 1e-6 for n in am_notes)
    assert all(abs(n.end - 4.0) < 1e-6 for n in am_notes)


def test_render_chord_track_tempo_in_midi(tmp_path: Path):
    out = tmp_path / "chords.mid"
    p = ChordProgression(chords=[("Cmaj7", 4)], tempo=100.0, time_signature="4/4")
    render_chord_track(p, out)
    pm = pretty_midi.PrettyMIDI(str(out))
    # initial_tempo читается из header
    assert abs(pm.estimate_tempo() - 100.0) < 5.0 or abs(pm.get_tempo_changes()[1][0] - 100.0) < 1e-6
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/test_chord_render.py -v
```

- [ ] **Step 3: Реализовать**

Файл `pipeline/pipeline/chord_render.py`:

```python
from __future__ import annotations

from pathlib import Path

import pretty_midi

from pipeline.chord_vocab import chord_to_pitches
from pipeline.progression import ChordProgression


_CHORD_VELOCITY = 70


def render_chord_track(progression: ChordProgression, out_path: Path) -> None:
    """ChordProgression → MIDI с одним piano-треком (Acoustic Grand Piano, program 0).

    На каждый аккорд — block voicing (root+3rd+5th+7th или triada) длительностью
    duration_in_beats. Tempo и BPM соответствуют progression.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    inst = pretty_midi.Instrument(program=0, name="ChordTrack")

    seconds_per_beat = 60.0 / progression.tempo
    cursor = 0.0
    for chord_str, beats in progression.chords:
        pitches = chord_to_pitches(chord_str)
        start = cursor
        end = cursor + beats * seconds_per_beat
        for p in pitches:
            inst.notes.append(pretty_midi.Note(
                velocity=_CHORD_VELOCITY, pitch=p, start=start, end=end,
            ))
        cursor = end

    pm.instruments.append(inst)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
```

- [ ] **Step 4: Запустить тесты**

```bash
.venv/bin/python -m pytest tests/test_chord_render.py -v
```

Expected: все 5 проходят. Если `test_render_chord_track_correct_number_of_chord_blocks` падает с 12 нотами вместо 16 — значит `chord_to_pitches` для какого-то качества вернул триаду; проверить `chord_to_pitches("Am7")` (должен быть 4 ноты).

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/chord_render.py pipeline/tests/test_chord_render.py
git commit -m "feat(pipeline): add chord_render for piano accompaniment MIDI"
```

---

## Task 4: postprocess — нормализация мелодии и склейка

**Files:**
- Create: `pipeline/pipeline/postprocess.py`
- Create: `pipeline/tests/test_postprocess.py`

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/test_postprocess.py`:

```python
from pathlib import Path

import pretty_midi
import pytest

from pipeline.progression import ChordProgression
from pipeline.postprocess import postprocess


def _fake_melody() -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=99, name="OriginalName")  # any non-66 program
    for i, pitch in enumerate([60, 62, 64, 65]):
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=pitch, start=i * 0.5, end=(i + 1) * 0.5
        ))
    return inst


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_postprocess_creates_two_files(tmp_path: Path):
    melody = _fake_melody()
    paths = postprocess(
        melody, _basic_progression(),
        model_name="mingus", run_id="20260427-test-aaaa",
        output_root=tmp_path, melody_program=66,
    )
    assert "melody_only" in paths and "with_chords" in paths
    assert Path(paths["melody_only"]).exists()
    assert Path(paths["with_chords"]).exists()


def test_postprocess_melody_only_has_one_instrument_with_target_program(tmp_path: Path):
    melody = _fake_melody()
    paths = postprocess(
        melody, _basic_progression(),
        model_name="mingus", run_id="20260427-test-bbbb",
        output_root=tmp_path, melody_program=66,
    )
    pm = pretty_midi.PrettyMIDI(str(paths["melody_only"]))
    assert len(pm.instruments) == 1
    assert pm.instruments[0].program == 66
    assert pm.instruments[0].name == "Melody"
    assert len(pm.instruments[0].notes) == 4


def test_postprocess_with_chords_has_two_instruments(tmp_path: Path):
    melody = _fake_melody()
    paths = postprocess(
        melody, _basic_progression(),
        model_name="mingus", run_id="20260427-test-cccc",
        output_root=tmp_path, melody_program=66,
    )
    pm = pretty_midi.PrettyMIDI(str(paths["with_chords"]))
    assert len(pm.instruments) == 2
    progs = sorted(i.program for i in pm.instruments)
    assert progs == [0, 66]  # piano chord track + melody


def test_postprocess_filename_pattern(tmp_path: Path):
    melody = _fake_melody()
    paths = postprocess(
        melody, _basic_progression(),
        model_name="bebopnet", run_id="20260427-153012-abcd1234",
        output_root=tmp_path, melody_program=66,
    )
    assert Path(paths["melody_only"]).name == "bebopnet_20260427-153012-abcd1234.mid"
    assert Path(paths["with_chords"]).name == "bebopnet_20260427-153012-abcd1234.mid"
    assert "melody_only" in str(paths["melody_only"])
    assert "with_chords" in str(paths["with_chords"])


def test_postprocess_does_not_mutate_input_melody(tmp_path: Path):
    melody = _fake_melody()
    original_program = melody.program
    original_name = melody.name
    postprocess(
        melody, _basic_progression(),
        model_name="mingus", run_id="20260427-test-dddd",
        output_root=tmp_path, melody_program=66,
    )
    assert melody.program == original_program
    assert melody.name == original_name
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/test_postprocess.py -v
```

- [ ] **Step 3: Реализовать**

Файл `pipeline/pipeline/postprocess.py`:

```python
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pretty_midi

from pipeline.chord_render import render_chord_track
from pipeline.progression import ChordProgression


def postprocess(
    melody: pretty_midi.Instrument,
    progression: ChordProgression,
    model_name: str,
    run_id: str,
    output_root: Path,
    melody_program: int,
) -> dict[str, Path]:
    """Принимает уже извлечённую мелодию (Instrument) и пишет два нормализованных MIDI.

    1. Клонирует melody, проставляет program ← melody_program, name ← 'Melody'.
    2. Сохраняет монофонную мелодию в `output_root/melody_only/<model>_<run_id>.mid`.
    3. Через chord_render строит наш piano chord track из progression.
    4. Склеивает melody + chord track в `output_root/with_chords/<model>_<run_id>.mid`.

    Не знает ни про MINGUS, ни про любую другую модель.
    """
    out = Path(output_root)
    melody_dir = out / "melody_only"
    chords_dir = out / "with_chords"
    melody_dir.mkdir(parents=True, exist_ok=True)
    chords_dir.mkdir(parents=True, exist_ok=True)

    melody_normalized = copy.deepcopy(melody)
    melody_normalized.program = melody_program
    melody_normalized.name = "Melody"
    melody_normalized.is_drum = False

    melody_path = melody_dir / f"{model_name}_{run_id}.mid"
    chords_path = chords_dir / f"{model_name}_{run_id}.mid"

    pm_melody = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    pm_melody.instruments.append(melody_normalized)
    pm_melody.write(str(melody_path))

    with tempfile.TemporaryDirectory() as td:
        chord_only_path = Path(td) / "chord_only.mid"
        render_chord_track(progression, chord_only_path)
        chord_pm = pretty_midi.PrettyMIDI(str(chord_only_path))

    pm_full = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    pm_full.instruments.append(copy.deepcopy(melody_normalized))
    for inst in chord_pm.instruments:
        pm_full.instruments.append(inst)
    pm_full.write(str(chords_path))

    return {"melody_only": melody_path, "with_chords": chords_path}
```

- [ ] **Step 4: Запустить — должны пройти**

```bash
.venv/bin/python -m pytest tests/test_postprocess.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/postprocess.py pipeline/tests/test_postprocess.py
git commit -m "feat(pipeline): add common postprocess for melody normalization"
```

---

## Task 5: ModelAdapter ABC + 5 stub-адаптеров

**Files:**
- Create: `pipeline/pipeline/adapters/base.py`
- Create: `pipeline/pipeline/adapters/bebopnet.py`
- Create: `pipeline/pipeline/adapters/ec2vae.py`
- Create: `pipeline/pipeline/adapters/cmt.py`
- Create: `pipeline/pipeline/adapters/commu.py`
- Create: `pipeline/pipeline/adapters/polyffusion.py`
- Create: `pipeline/tests/adapters/test_base.py`

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/adapters/test_base.py`:

```python
from pathlib import Path

import pytest

from pipeline.progression import ChordProgression
from pipeline.adapters.base import ModelAdapter
from pipeline.adapters.bebopnet import BebopNetAdapter
from pipeline.adapters.ec2vae import EC2VaeAdapter
from pipeline.adapters.cmt import CMTAdapter
from pipeline.adapters.commu import ComMUAdapter
from pipeline.adapters.polyffusion import PolyffusionAdapter


def test_model_adapter_is_abstract():
    with pytest.raises(TypeError):
        ModelAdapter()  # type: ignore[abstract]


@pytest.mark.parametrize("AdapterCls", [
    BebopNetAdapter, EC2VaeAdapter, CMTAdapter, ComMUAdapter, PolyffusionAdapter,
])
def test_stub_adapter_prepare_raises_not_implemented(tmp_path: Path, AdapterCls):
    adapter = AdapterCls()
    progression = ChordProgression(chords=[("Cmaj7", 4)], tempo=120.0, time_signature="4/4")
    with pytest.raises(NotImplementedError, match="adapter not implemented"):
        adapter.prepare(progression, config=None, tmp_dir=tmp_path)


@pytest.mark.parametrize("AdapterCls", [
    BebopNetAdapter, EC2VaeAdapter, CMTAdapter, ComMUAdapter, PolyffusionAdapter,
])
def test_stub_adapter_extract_melody_raises_not_implemented(tmp_path: Path, AdapterCls):
    adapter = AdapterCls()
    fake_midi = tmp_path / "x.mid"
    with pytest.raises(NotImplementedError, match="adapter not implemented"):
        adapter.extract_melody(fake_midi)
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/adapters/test_base.py -v
```

- [ ] **Step 3: Реализовать ABC**

Файл `pipeline/pipeline/adapters/base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pretty_midi

from pipeline.progression import ChordProgression


class ModelAdapter(ABC):
    """Toolbox для одной модели — обе границы между pipeline и моделью."""

    @abstractmethod
    def prepare(
        self,
        progression: ChordProgression,
        config: Any,
        tmp_dir: Path,
    ) -> dict:
        """Из progression и per-model config собирает params для runner'а
        (включая физическую подготовку входных файлов в tmp_dir, если нужно).

        Возвращает словарь, который pipeline пробрасывает runner'у через JSON stdin.
        """

    @abstractmethod
    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        """Из сырого выхода модели возвращает монофонную мелодию как pretty_midi.Instrument.

        Контракт: возвращаемый Instrument МОЖЕТ быть сконструирован adapter'ом из чего
        угодно — track'а MIDI (MINGUS, BebopNet), highest-pitch проекции полифонии
        (Polyffusion), декодированного pianoroll (EC²-VAE), декодированных REMI-токенов
        (ComMU). Pipeline не предполагает «у модели есть готовый melody-track» —
        adapter сам отвечает за получение монофонной мелодии в нужном формате.
        """
```

- [ ] **Step 4: Реализовать 5 stub-адаптеров**

Каждый файл `pipeline/pipeline/adapters/<model>.py` (где `<model>` = bebopnet, ec2vae, cmt, commu, polyffusion). Все одинаковые с заменой имени класса.

Пример для `bebopnet.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pretty_midi

from pipeline.adapters.base import ModelAdapter
from pipeline.progression import ChordProgression


class BebopNetAdapter(ModelAdapter):
    def prepare(self, progression: ChordProgression, config: Any, tmp_dir: Path) -> dict:
        raise NotImplementedError("model bebopnet: adapter not implemented")

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        raise NotImplementedError("model bebopnet: adapter not implemented")
```

Аналогично:
- `ec2vae.py` → `class EC2VaeAdapter(ModelAdapter)`, сообщение `"model ec2vae: adapter not implemented"`
- `cmt.py` → `class CMTAdapter(ModelAdapter)`, сообщение `"model cmt: adapter not implemented"`
- `commu.py` → `class ComMUAdapter(ModelAdapter)`, сообщение `"model commu: adapter not implemented"`
- `polyffusion.py` → `class PolyffusionAdapter(ModelAdapter)`, сообщение `"model polyffusion: adapter not implemented"`

- [ ] **Step 5: Запустить тесты**

```bash
.venv/bin/python -m pytest tests/adapters/test_base.py -v
```

Expected: 11 passed (1 abstract + 5×prepare + 5×extract_melody).

- [ ] **Step 6: Commit**

```bash
git add pipeline/pipeline/adapters
git commit -m "feat(pipeline): add ModelAdapter ABC and 5 stub adapters"
```

---

## Task 6: MingusPipelineConfig + XML builders для 3 стратегий

**Files:**
- Create: `pipeline/pipeline/_xml_builders/mingus_xml.py`
- Create: `pipeline/tests/adapters/test_mingus_xml.py`

(Сам `MingusAdapter` появится в Task 7. Этот таск только про XML-генерацию, чтобы её можно было тестировать изолированно.)

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/adapters/test_mingus_xml.py`:

```python
from pathlib import Path

import pytest
from music21 import converter

from pipeline.progression import ChordProgression
from pipeline.adapters.mingus import MingusPipelineConfig
from pipeline._xml_builders.mingus_xml import build_mingus_xml


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_tonic_whole_writes_valid_musicxml(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)  # music21 raises if XML невалиден
    # 2 такта
    measures = s.parts[0].getElementsByClass("Measure")
    assert len(measures) == 2


def test_tonic_whole_has_two_chord_symbols(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)
    chord_syms = list(s.parts[0].recurse().getElementsByClass("ChordSymbol"))
    assert len(chord_syms) == 2


def test_tonic_whole_one_note_per_measure(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)
    notes_per_measure = [
        len(list(m.getElementsByClass("Note")))
        for m in s.parts[0].getElementsByClass("Measure")
    ]
    assert notes_per_measure == [1, 1]


def test_tonic_quarters_four_notes_per_measure(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_quarters", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)
    notes_per_measure = [
        len(list(m.getElementsByClass("Note")))
        for m in s.parts[0].getElementsByClass("Measure")
    ]
    assert notes_per_measure == [4, 4]


def test_tonic_pitch_is_root_of_chord(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)
    measures = list(s.parts[0].getElementsByClass("Measure"))
    # Cmaj7 → C тоника; Am7 → A тоника
    note0 = list(measures[0].getElementsByClass("Note"))[0]
    note1 = list(measures[1].getElementsByClass("Note"))[0]
    assert note0.pitch.name in ("C",)
    assert note1.pitch.name in ("A",)


def test_custom_xml_copies_file(tmp_path: Path):
    src = tmp_path / "src.xml"
    src.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="3.1"><part-list>'
        '<score-part id="P1"><part-name>Melody</part-name></score-part>'
        '</part-list><part id="P1"></part></score-partwise>'
    )
    cfg = MingusPipelineConfig(
        seed_strategy="custom_xml", checkpoint_epochs=100, custom_xml_path=src,
    )
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    assert out.read_text() == src.read_text()


def test_custom_xml_requires_path(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="custom_xml", checkpoint_epochs=100, custom_xml_path=None)
    with pytest.raises(ValueError, match="custom_xml_path"):
        build_mingus_xml(_basic_progression(), cfg, tmp_path / "input.xml")
```

- [ ] **Step 2: Запустить — упадут (нет ни MingusPipelineConfig ни build_mingus_xml)**

```bash
.venv/bin/python -m pytest tests/adapters/test_mingus_xml.py -v
```

- [ ] **Step 3: Создать `MingusPipelineConfig`**

Файл `pipeline/pipeline/adapters/mingus.py` — пока заготовка только с конфигом:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class MingusPipelineConfig:
    """Все настройки MINGUS на уровне нашего пайплайна.

    Часть из них — стратегии подготовки входа (живут только здесь),
    часть — параметры MINGUS-API, которые мы пробрасываем в runner.
    """

    seed_strategy: Literal["tonic_whole", "tonic_quarters", "custom_xml"] = "tonic_whole"
    custom_xml_path: Path | None = None
    temperature: float = 1.0
    device: str = "cpu"
    # У MINGUS веса лежат как `MINGUS COND I-C-NC-B-BE-O Epochs <N>.pt`. Мы не передаём
    # path напрямую — runner сам собирает имена из этой числовой настройки.
    checkpoint_epochs: int = 100
    melody_instrument_name: str = "Tenor Sax"
```

- [ ] **Step 4: Реализовать XML-builder**

Файл `pipeline/pipeline/_xml_builders/mingus_xml.py`:

```python
from __future__ import annotations

import shutil
from pathlib import Path

from music21 import chord as m21_chord, harmony, instrument, key, meter, note, stream, tempo

from pipeline.adapters.mingus import MingusPipelineConfig
from pipeline.chord_vocab import parse_chord, ROOTS
from pipeline.progression import ChordProgression


_TONIC_OCTAVE = 5  # С5 = MIDI 72; для саксофона средне-высокий регистр


def build_mingus_xml(
    progression: ChordProgression,
    config: MingusPipelineConfig,
    out_path: Path,
) -> None:
    """Пишет MusicXML, готовый к скармливанию MINGUS gen.xmlToStructuredSong.

    seed_strategy:
      - tonic_whole    — 1 whole-нота тоники в каждом баре
      - tonic_quarters — 4 quarter-ноты тоники в каждом баре
      - custom_xml     — копирует config.custom_xml_path в out_path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if config.seed_strategy == "custom_xml":
        if config.custom_xml_path is None:
            raise ValueError("seed_strategy=custom_xml requires custom_xml_path")
        shutil.copy(config.custom_xml_path, out_path)
        return

    bpb = progression.beats_per_bar()
    if progression.total_beats() % bpb != 0:
        raise ValueError(
            f"progression total_beats={progression.total_beats()} not divisible by "
            f"beats_per_bar={bpb} (time_signature={progression.time_signature})"
        )

    score = stream.Score()
    score.metadata = score.metadata or None
    part = stream.Part()
    part.id = "P1"
    part.partName = "Melody"
    part.insert(0, instrument.TenorSaxophone())

    # построим список (chord_symbol, start_beat, beats) — равномерно по bar'ам
    cursor_beats = 0
    for chord_str, beats in progression.chords:
        cursor_beats += beats
    # cursor_beats == progression.total_beats(); используется только для проверки

    measure_idx = 1
    chord_iter = iter(progression.chords)
    cur_chord, cur_remaining = next(chord_iter)
    for bar in range(progression.num_bars()):
        m = stream.Measure(number=measure_idx)
        if measure_idx == 1:
            m.append(meter.TimeSignature(progression.time_signature))
            m.append(tempo.MetronomeMark(number=progression.tempo))
            m.append(key.KeySignature(0))  # C-мажор
        # положим chord-symbol в начало бара
        cs = harmony.ChordSymbol(cur_chord)
        m.insert(0, cs)
        # положим затравочные ноты
        root_idx, _quality = parse_chord(cur_chord)
        tonic_pitch = note.Pitch()
        tonic_pitch.midi = root_idx + 12 * (_TONIC_OCTAVE + 1)
        if config.seed_strategy == "tonic_whole":
            n = note.Note(tonic_pitch)
            n.quarterLength = bpb
            m.append(n)
        elif config.seed_strategy == "tonic_quarters":
            for _ in range(bpb):
                n = note.Note(tonic_pitch)
                n.quarterLength = 1
                m.append(n)
        else:
            raise ValueError(f"unsupported seed_strategy: {config.seed_strategy}")
        part.append(m)
        # забираем beats из текущего аккорда
        cur_remaining -= bpb
        if cur_remaining <= 0:
            try:
                cur_chord, cur_remaining = next(chord_iter)
            except StopIteration:
                cur_chord, cur_remaining = (cur_chord, 0)  # последний аккорд — больше итераций нет
        measure_idx += 1

    score.insert(0, part)
    score.write("musicxml", fp=str(out_path))
```

> **Note for executor:** Этот builder упрощён — он предполагает, что каждый аккорд занимает целое число баров (`beats >= bpb`, и `beats % bpb == 0`). Тестовая прогрессия `Cmaj7-Am7-Dm7-G7-Cmaj7-Am7-Dm7-G7` (по 4 beat'а) этому удовлетворяет. Если когда-то понадобится поддержать аккорды короче бара (например `Dm7-G7` каждый по 2 beat'а в одном баре) — потребуется переработка цикла. Пока ставим `assert beats % bpb == 0` как предохранитель.

Добавить в начало `build_mingus_xml`, после `progression.total_beats()` проверки:

```python
    for chord_str, beats in progression.chords:
        if beats % bpb != 0:
            raise ValueError(
                f"chord {chord_str} duration {beats} not multiple of {bpb}; "
                f"sub-bar chord placement not supported in MVP"
            )
```

- [ ] **Step 5: Запустить тесты**

```bash
.venv/bin/python -m pytest tests/adapters/test_mingus_xml.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add pipeline/pipeline/adapters/mingus.py pipeline/pipeline/_xml_builders pipeline/tests/adapters/test_mingus_xml.py
git commit -m "feat(pipeline): add MingusPipelineConfig and XML seed builders"
```

---

## Task 7: MingusAdapter.prepare

**Files:**
- Modify: `pipeline/pipeline/adapters/mingus.py` (добавить класс `MingusAdapter`)
- Create: `pipeline/tests/adapters/test_mingus_prepare.py`

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/adapters/test_mingus_prepare.py`:

```python
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
    params = MingusAdapter().prepare(_basic_progression(), cfg, tmp_path)
    for key in ["input_xml_path", "output_midi_path", "checkpoint_epochs", "temperature", "device", "model_repo_path"]:
        assert key in params, f"missing key: {key}"


def test_prepare_writes_input_xml(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    params = MingusAdapter().prepare(_basic_progression(), cfg, tmp_path)
    assert Path(params["input_xml_path"]).exists()
    assert Path(params["input_xml_path"]).suffix == ".xml"


def test_prepare_output_midi_path_in_tmp_dir(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    params = MingusAdapter().prepare(_basic_progression(), cfg, tmp_path)
    assert str(params["output_midi_path"]).startswith(str(tmp_path))
    assert params["output_midi_path"].endswith(".mid")


def test_prepare_passes_through_temperature_and_device(tmp_path: Path):
    cfg = MingusPipelineConfig(
        seed_strategy="tonic_whole", checkpoint_epochs=100,
        temperature=0.7, device="cpu",
    )
    params = MingusAdapter().prepare(_basic_progression(), cfg, tmp_path)
    assert params["temperature"] == 0.7
    assert params["device"] == "cpu"
    assert params["checkpoint_epochs"] == 100


def test_prepare_does_not_leak_pipeline_concepts(tmp_path: Path):
    """params не должен содержать seed_strategy, run_id, model_name и т.п. — это наши слои."""
    cfg = MingusPipelineConfig(seed_strategy="tonic_quarters", checkpoint_epochs=100)
    params = MingusAdapter().prepare(_basic_progression(), cfg, tmp_path)
    forbidden = {"seed_strategy", "run_id", "model_name", "progression"}
    leaked = forbidden & params.keys()
    assert not leaked, f"adapter leaked pipeline concepts to runner params: {leaked}"
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/adapters/test_mingus_prepare.py -v
```

- [ ] **Step 3: Реализовать `MingusAdapter`**

В файл `pipeline/pipeline/adapters/mingus.py` добавить:

```python
import pretty_midi

from pipeline.adapters.base import ModelAdapter
from pipeline.progression import ChordProgression
from pipeline._xml_builders.mingus_xml import build_mingus_xml


class MingusAdapter(ModelAdapter):
    def prepare(
        self,
        progression: ChordProgression,
        config: MingusPipelineConfig,
        tmp_dir: Path,
    ) -> dict:
        from pipeline.config import MINGUS_REPO_PATH  # ленивый импорт чтобы избежать circular

        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        xml_path = tmp_dir / "input.xml"
        midi_path = tmp_dir / "raw.mid"
        build_mingus_xml(progression, config, xml_path)
        return {
            "input_xml_path": str(xml_path),
            "output_midi_path": str(midi_path),
            "checkpoint_epochs": config.checkpoint_epochs,
            "temperature": config.temperature,
            "device": config.device,
            "model_repo_path": str(MINGUS_REPO_PATH),
        }

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        # реализуется в Task 8
        raise NotImplementedError("MingusAdapter.extract_melody not yet implemented")
```

- [ ] **Step 4: Заглушка для `MINGUS_REPO_PATH` в `config.py` (полный config будет в Task 11)**

Создать `pipeline/pipeline/config.py`:

```python
from __future__ import annotations

from pathlib import Path

DIPLOMA_ROOT: Path = Path(__file__).resolve().parents[2]
MINGUS_REPO_PATH: Path = DIPLOMA_ROOT / "models" / "MINGUS"
```

- [ ] **Step 5: Запустить тесты**

```bash
.venv/bin/python -m pytest tests/adapters/test_mingus_prepare.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add pipeline/pipeline/adapters/mingus.py pipeline/pipeline/config.py pipeline/tests/adapters/test_mingus_prepare.py
git commit -m "feat(pipeline): add MingusAdapter.prepare with config skeleton"
```

---

## Task 8: MingusAdapter.extract_melody

**Files:**
- Modify: `pipeline/pipeline/adapters/mingus.py` (заменить `raise NotImplementedError` в `extract_melody`)
- Create: `pipeline/tests/adapters/test_mingus_extract_melody.py`

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/adapters/test_mingus_extract_melody.py`:

```python
from pathlib import Path

import pretty_midi
import pytest

from pipeline.adapters.mingus import MingusAdapter


def _make_two_track_midi(out_path: Path, melody_name: str = "Tenor Sax") -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    melody_inst = pretty_midi.Instrument(program=66, name=melody_name)
    for i, p in enumerate([60, 64, 67, 71]):
        melody_inst.notes.append(pretty_midi.Note(80, p, i * 0.5, (i + 1) * 0.5))
    pm.instruments.append(melody_inst)

    chord_inst = pretty_midi.Instrument(program=0, name="piano")
    for i, p in enumerate([60, 64, 67]):
        chord_inst.notes.append(pretty_midi.Note(60, p, 0.0, 2.0))
    pm.instruments.append(chord_inst)

    pm.write(str(out_path))


def test_extract_melody_returns_tenor_sax_track(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi)
    melody = MingusAdapter().extract_melody(midi)
    assert melody.name == "Tenor Sax"
    assert len(melody.notes) == 4


def test_extract_melody_returns_instrument_type(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi)
    melody = MingusAdapter().extract_melody(midi)
    assert isinstance(melody, pretty_midi.Instrument)


def test_extract_melody_raises_when_track_missing(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi, melody_name="Wrong Name")
    with pytest.raises(ValueError, match="Tenor Sax"):
        MingusAdapter().extract_melody(midi)
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/adapters/test_mingus_extract_melody.py -v
```

- [ ] **Step 3: Реализовать**

Заменить тело `extract_melody` в `pipeline/pipeline/adapters/mingus.py`:

```python
    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        # имя трека настраивается в MingusPipelineConfig.melody_instrument_name,
        # но adapter держит дефолт здесь — он часть знания о MINGUS, а не общий конфиг
        target = "Tenor Sax"  # см. MingusPipelineConfig.melody_instrument_name
        pm = pretty_midi.PrettyMIDI(str(raw_midi_path))
        for inst in pm.instruments:
            if inst.name == target:
                return inst
        names = [i.name for i in pm.instruments]
        raise ValueError(
            f"melody track {target!r} not found in {raw_midi_path} (have: {names})"
        )
```

> **Note:** хотим брать `target` из `MingusPipelineConfig.melody_instrument_name`, но `extract_melody` сейчас не получает config. Чтобы не ломать ABC и не плодить методов — в `MingusAdapter.__init__` принимаем config:

Перед заменой `extract_melody` отредактировать класс:

```python
class MingusAdapter(ModelAdapter):
    def __init__(self, config: MingusPipelineConfig | None = None) -> None:
        self._config = config or MingusPipelineConfig()

    def prepare(
        self,
        progression: ChordProgression,
        config: MingusPipelineConfig,
        tmp_dir: Path,
    ) -> dict:
        # ... как в Task 7 ...
        # но ВНУТРИ запоминаем config, чтобы extract_melody мог использовать
        self._config = config
        # ... остальное как раньше ...
```

И `extract_melody` использует `self._config.melody_instrument_name`:

```python
    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        target = self._config.melody_instrument_name
        pm = pretty_midi.PrettyMIDI(str(raw_midi_path))
        for inst in pm.instruments:
            if inst.name == target:
                return inst
        names = [i.name for i in pm.instruments]
        raise ValueError(
            f"melody track {target!r} not found in {raw_midi_path} (have: {names})"
        )
```

Также обновить тест Task 5 (`test_base.py`) — там создаётся `BebopNetAdapter()` без аргументов; `MingusAdapter()` тоже должен поддерживаться (без аргументов = дефолтный конфиг). Уже учтено в `__init__`.

- [ ] **Step 4: Запустить все тесты**

```bash
.venv/bin/python -m pytest -v
```

Expected: всё проходит. Если `test_mingus_prepare.py` упал — значит метод `prepare` не сохраняет config; убедиться что `self._config = config` присутствует.

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/adapters/mingus.py pipeline/tests/adapters/test_mingus_extract_melody.py
git commit -m "feat(pipeline): add MingusAdapter.extract_melody"
```

---

## Task 9: runner_protocol — RunnerError + run_runner_subprocess

**Files:**
- Create: `pipeline/pipeline/runner_protocol.py`
- Create: `pipeline/tests/test_runner_protocol.py`

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/test_runner_protocol.py`:

```python
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from pipeline.runner_protocol import RunnerError, run_runner_subprocess


def _write_runner(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_run_runner_writes_logs_on_success(tmp_path: Path):
    runner = tmp_path / "runner.py"
    out_midi = tmp_path / "raw.mid"
    out_midi_str = str(out_midi).replace('"', '\\"')
    _write_runner(runner, (
        f'import json, sys\n'
        f'data = json.loads(sys.stdin.read())\n'
        f'open("{out_midi_str}", "wb").write(b"FAKE_MIDI")\n'
        f'print("hello stdout")\n'
        f'sys.exit(0)\n'
    ))
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = run_runner_subprocess(
        venv_python=sys.executable,
        runner_script=runner,
        payload={"params": {"output_midi_path": str(out_midi)}},
        tmp_dir=log_dir,
        timeout_sec=10,
    )
    assert result == out_midi
    assert (log_dir / "stdout.log").exists()
    assert (log_dir / "stdout.log").read_text().strip() == "hello stdout"


def test_run_runner_raises_when_exit_nonzero(tmp_path: Path):
    runner = tmp_path / "runner.py"
    _write_runner(runner, (
        'import sys\n'
        'sys.stderr.write("boom\\n")\n'
        'sys.exit(1)\n'
    ))
    with pytest.raises(RunnerError, match="exited with 1"):
        run_runner_subprocess(
            venv_python=sys.executable,
            runner_script=runner,
            payload={"params": {"output_midi_path": str(tmp_path / "noop.mid")}},
            tmp_dir=tmp_path,
            timeout_sec=10,
        )


def test_run_runner_raises_when_midi_not_written(tmp_path: Path):
    runner = tmp_path / "runner.py"
    _write_runner(runner, (
        'import sys\n'
        'sys.exit(0)\n'  # exit 0 но output не создан
    ))
    with pytest.raises(RunnerError, match="output MIDI not found"):
        run_runner_subprocess(
            venv_python=sys.executable,
            runner_script=runner,
            payload={"params": {"output_midi_path": str(tmp_path / "missing.mid")}},
            tmp_dir=tmp_path,
            timeout_sec=10,
        )


def test_run_runner_passes_payload_via_stdin(tmp_path: Path):
    runner = tmp_path / "runner.py"
    out_midi = tmp_path / "raw.mid"
    out_midi_str = str(out_midi).replace('"', '\\"')
    _write_runner(runner, (
        'import json, sys\n'
        'data = json.loads(sys.stdin.read())\n'
        'assert data["model"] == "test"\n'
        'assert data["params"]["foo"] == 42\n'
        f'open("{out_midi_str}", "wb").write(b"OK")\n'
        'sys.exit(0)\n'
    ))
    result = run_runner_subprocess(
        venv_python=sys.executable,
        runner_script=runner,
        payload={"model": "test", "run_id": "rid", "params": {"foo": 42, "output_midi_path": str(out_midi)}},
        tmp_dir=tmp_path,
        timeout_sec=10,
    )
    assert result == out_midi
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/test_runner_protocol.py -v
```

- [ ] **Step 3: Реализовать**

Файл `pipeline/pipeline/runner_protocol.py`:

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path


class RunnerError(RuntimeError):
    """Subprocess модели завершился с ошибкой или не положил output MIDI."""


def run_runner_subprocess(
    venv_python: str | Path,
    runner_script: str | Path,
    payload: dict,
    tmp_dir: Path,
    timeout_sec: int,
) -> Path:
    """Запускает runner-скрипт интерпретатором model-venv'а.

    1. Пишет JSON payload в stdin.
    2. Перехватывает stdout/stderr → tmp_dir/{stdout,stderr}.log.
    3. Если exit ≠ 0 → RunnerError со stderr-tail.
    4. Если exit == 0 но params.output_midi_path не существует → RunnerError.
    5. Возвращает Path к output MIDI.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    expected_midi = Path(payload["params"]["output_midi_path"])

    result = subprocess.run(
        [str(venv_python), str(runner_script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    (tmp_dir / "stdout.log").write_text(result.stdout)
    (tmp_dir / "stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-20:])
        raise RunnerError(
            f"runner {runner_script} exited with {result.returncode}:\n{tail}"
        )
    if not expected_midi.exists():
        raise RunnerError(
            f"runner {runner_script} exited 0 but output MIDI not found at {expected_midi}"
        )
    return expected_midi
```

- [ ] **Step 4: Запустить тесты**

```bash
.venv/bin/python -m pytest tests/test_runner_protocol.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/runner_protocol.py pipeline/tests/test_runner_protocol.py
git commit -m "feat(pipeline): add subprocess runner protocol with logging"
```

---

## Task 10: mingus_runner.py — обёртка вокруг MINGUS gen-функций

**Files:**
- Create: `pipeline/runners/mingus_runner.py`

(Тестируется end-to-end в Task 14. Юнит-тестировать в pipeline-venv нельзя — здесь нет torch; запуск возможен только в MINGUS-venv.)

- [ ] **Step 1: Создать runner**

Файл `pipeline/runners/mingus_runner.py`:

```python
#!/usr/bin/env python3
"""MINGUS runner: запускается интерпретатором models/MINGUS/.venv/bin/python.

Контракт:
- читает JSON payload со stdin (см. pipeline.runner_protocol)
- params: input_xml_path, output_midi_path, checkpoint_epochs, temperature, device, model_repo_path
- импортирует MINGUS gen-функции напрямую (без CLI), вызывает их с нашими параметрами
- пишет MIDI в output_midi_path
- exit 0 при успехе, exit 1 при ошибке (traceback в stderr)

Этот файл НЕ должен импортировать ничего из pipeline.* — он живёт в MINGUS-venv.
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
    input_xml = Path(params["input_xml_path"])
    output_midi = Path(params["output_midi_path"])
    epochs: int = int(params["checkpoint_epochs"])
    temperature: float = float(params["temperature"])
    device_name: str = params["device"]
    repo: Path = Path(params["model_repo_path"])

    # Импорты MINGUS требуют cwd = MINGUS_REPO и PYTHONPATH = MINGUS_REPO,
    # потому что они делают `import B_train.loadDB` и т.п.
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    import B_train.loadDB as dataset
    import B_train.MINGUS_model as mod
    import C_generate.gen_funct as gen

    device = torch.device(device_name)
    torch.manual_seed(1)

    # Аргументы как в C_generate/generate.py — нужны те же чтобы veca совпали с обученными.
    COND = "I-C-NC-B-BE-O"
    TRAIN_BATCH_SIZE = 20
    EVAL_BATCH_SIZE = 10
    BPTT = 35
    AUGMENTATION = False
    SEGMENTATION = True
    AUGMENTATION_CONST = 3
    NUM_CHORUS = 3

    music_db = dataset.MusicDB(
        device, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
        BPTT, AUGMENTATION, SEGMENTATION, AUGMENTATION_CONST,
    )
    structured_songs = music_db.getStructuredSongs()
    vocab_pitch, vocab_duration, vocab_beat, vocab_offset = music_db.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = music_db.getInverseVocabs()
    db_chords, db_to_music21, db_to_chord_composition, db_to_midi_chords = music_db.getChordDicts()

    def _build_model(is_pitch: bool):
        if is_pitch:
            pitch_embed_dim = 512
            duration_embed_dim = 512
            chord_encod_dim = 64
            next_chord_encod_dim = 32
            beat_embed_dim = 64
            bass_embed_dim = 64
            offset_embed_dim = 32
        else:
            pitch_embed_dim = 64
            duration_embed_dim = 64
            chord_encod_dim = 64
            next_chord_encod_dim = 32
            beat_embed_dim = 32
            bass_embed_dim = 32
            offset_embed_dim = 32
        emsize = 200
        nhid = 200
        nlayers = 4
        nhead = 4
        dropout = 0.2
        m = mod.TransformerModel(
            len(vocab_pitch), pitch_embed_dim,
            len(vocab_duration), duration_embed_dim,
            bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
            len(vocab_beat), beat_embed_dim,
            len(vocab_offset), offset_embed_dim,
            emsize, nhead, nhid, nlayers,
            pitch_to_ix["<pad>"], duration_to_ix["<pad>"], beat_to_ix["<pad>"], offset_to_ix["<pad>"],
            device, dropout, is_pitch, COND,
        ).to(device)
        return m

    model_pitch = _build_model(is_pitch=True)
    model_duration = _build_model(is_pitch=False)

    pitch_ckpt = repo / "B_train" / "models" / "pitchModel" / f"MINGUS COND {COND} Epochs {epochs}.pt"
    duration_ckpt = repo / "B_train" / "models" / "durationModel" / f"MINGUS COND {COND} Epochs {epochs}.pt"
    model_pitch.load_state_dict(torch.load(str(pitch_ckpt), map_location=device))
    model_duration.load_state_dict(torch.load(str(duration_ckpt), map_location=device))

    tune, _wjazz_to_music21, _wjazz_to_midi_chords, _wjazz_to_chord_composition, _wjazz_chords = (
        gen.xmlToStructuredSong(
            str(input_xml),
            db_to_music21, db_to_midi_chords, db_to_chord_composition, db_chords,
        )
    )
    is_jazz = False
    new_song = gen.generateOverStandard(
        tune, NUM_CHORUS, temperature,
        model_pitch, model_duration, db_to_midi_chords,
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
        vocab_pitch, vocab_duration,
        BPTT, device, is_jazz,
    )

    pm = gen.structuredSongsToPM(new_song, db_to_midi_chords, is_jazz)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_midi))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
```

- [ ] **Step 2: Smoke-проверка что MINGUS-venv может запустить файл (без реального inference, только парсинг JSON)**

```bash
echo '{"model":"mingus","run_id":"x","params":{"input_xml_path":"/nonexistent","output_midi_path":"/tmp/x.mid","checkpoint_epochs":100,"temperature":1.0,"device":"cpu","model_repo_path":"/Users/maxos/PythonProjects/diploma/models/MINGUS"}}' \
  | /Users/maxos/PythonProjects/diploma/models/MINGUS/.venv/bin/python /Users/maxos/PythonProjects/diploma/pipeline/runners/mingus_runner.py 2>&1 | head -30
```

Expected: traceback внутри `xmlToStructuredSong` (потому что `/nonexistent` файла нет), exit 1. Это ОК — мы убедились что runner парсит JSON, импортирует MINGUS, доходит до xml-парсинга.

Если упало раньше (например на импорте `B_train.loadDB`) — скорее всего `DATA.json` отсутствует. Сослаться на pre-requisite Task 0.

- [ ] **Step 3: Commit**

```bash
git add pipeline/runners/mingus_runner.py
git commit -m "feat(pipeline): add mingus_runner wrapping MINGUS inference"
```

---

## Task 11: config.py — registry, paths, константы

**Files:**
- Modify: `pipeline/pipeline/config.py` (расширить)

- [ ] **Step 1: Расширить config.py**

Файл `pipeline/pipeline/config.py` (заменить целиком):

```python
from __future__ import annotations

from pathlib import Path

from pipeline.adapters.base import ModelAdapter
from pipeline.adapters.bebopnet import BebopNetAdapter
from pipeline.adapters.cmt import CMTAdapter
from pipeline.adapters.commu import ComMUAdapter
from pipeline.adapters.ec2vae import EC2VaeAdapter
from pipeline.adapters.mingus import MingusAdapter, MingusPipelineConfig
from pipeline.adapters.polyffusion import PolyffusionAdapter


DIPLOMA_ROOT: Path = Path(__file__).resolve().parents[2]
PIPELINE_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_ROOT: Path = PIPELINE_ROOT / "output"
RUNNERS_ROOT: Path = PIPELINE_ROOT / "runners"

MINGUS_REPO_PATH: Path = DIPLOMA_ROOT / "models" / "MINGUS"

MODEL_NAMES: list[str] = ["mingus", "bebopnet", "ec2vae", "cmt", "commu", "polyffusion"]

MODEL_VENV_PYTHON: dict[str, Path] = {
    "mingus":      DIPLOMA_ROOT / "models/MINGUS/.venv/bin/python",
    "bebopnet":    DIPLOMA_ROOT / "models/bebopnet-code/.venv/bin/python",
    "ec2vae":      DIPLOMA_ROOT / "models/EC2-VAE/.venv/bin/python",
    "cmt":         DIPLOMA_ROOT / "models/CMT-pytorch/.venv/bin/python",
    "commu":       DIPLOMA_ROOT / "models/ComMU-code/.venv/bin/python",
    "polyffusion": DIPLOMA_ROOT / "models/polyffusion/.venv/bin/python",
}

MODEL_RUNNER_SCRIPT: dict[str, Path] = {
    "mingus":      RUNNERS_ROOT / "mingus_runner.py",
    # остальные runner-скрипты появляются вместе с реализацией модели
}

ADAPTERS: dict[str, ModelAdapter] = {
    "mingus":      MingusAdapter(MingusPipelineConfig()),
    "bebopnet":    BebopNetAdapter(),
    "ec2vae":      EC2VaeAdapter(),
    "cmt":         CMTAdapter(),
    "commu":       ComMUAdapter(),
    "polyffusion": PolyffusionAdapter(),
}

MODEL_CONFIGS: dict[str, object | None] = {
    "mingus": MingusPipelineConfig(
        seed_strategy="tonic_whole",
        temperature=1.0,
        device="cpu",
        checkpoint_epochs=100,
    ),
    "bebopnet":    None,
    "ec2vae":      None,
    "cmt":         None,
    "commu":       None,
    "polyffusion": None,
}

# Общее pipeline-решение: монофонная мелодия в melody_only.mid пишется единым
# тембром у всех моделей — тогда на слух (и в feature-расчёте метрик)
# тембр не вмешивается в сравнение. 66 = Tenor Sax (GM, 0-indexed pretty_midi),
# исторически бэйслайны (BebopNet/MINGUS) как раз и пишут саксофоном.
MELODY_PROGRAM: int = 66

RUNNER_TIMEOUT_SEC: int = 600
```

- [ ] **Step 2: Smoke-импорт**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -c "from pipeline.config import MODEL_NAMES, ADAPTERS, MODEL_CONFIGS, MELODY_PROGRAM; assert len(ADAPTERS) == 6; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Запустить весь test suite — убедиться что ничего не сломалось**

```bash
.venv/bin/python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/pipeline/config.py
git commit -m "feat(pipeline): add full config registry and constants"
```

---

## Task 12: pipeline.py — `generate_all`

**Files:**
- Create: `pipeline/pipeline/pipeline.py`
- Create: `pipeline/tests/test_pipeline.py`
- Create: `pipeline/tests/conftest.py`

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/conftest.py`:

```python
import pretty_midi
import pytest


@pytest.fixture
def fake_melody_instrument():
    inst = pretty_midi.Instrument(program=99, name="Tenor Sax")
    for i, p in enumerate([60, 64, 67, 71]):
        inst.notes.append(pretty_midi.Note(80, p, i * 0.5, (i + 1) * 0.5))
    return inst
```

Файл `pipeline/tests/test_pipeline.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pretty_midi
import pytest

from pipeline.progression import ChordProgression
from pipeline.pipeline import generate_all, make_run_id


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_make_run_id_format():
    rid = make_run_id(_basic_progression())
    parts = rid.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 8 and parts[0].isdigit()  # YYYYMMDD
    assert len(parts[1]) == 6 and parts[1].isdigit()  # HHMMSS
    assert len(parts[2]) == 8                         # 8charhash


def test_make_run_id_hash_deterministic_for_same_progression():
    p = _basic_progression()
    a = make_run_id(p).split("-")[2]
    b = make_run_id(p).split("-")[2]
    assert a == b


def test_make_run_id_hash_different_for_different_progressions():
    p1 = ChordProgression(chords=[("Cmaj7", 4)], tempo=120.0, time_signature="4/4")
    p2 = ChordProgression(chords=[("Dm7", 4)],   tempo=120.0, time_signature="4/4")
    a = make_run_id(p1).split("-")[2]
    b = make_run_id(p2).split("-")[2]
    assert a != b


def test_generate_all_returns_dict_with_all_models(tmp_path: Path, fake_melody_instrument, monkeypatch):
    """С моком adapter+runner: generate_all возвращает 1 ok + 5 stub-errors."""
    monkeypatch.setattr("pipeline.pipeline.OUTPUT_ROOT", tmp_path)

    fake_raw_midi = tmp_path / "fake_raw.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(fake_melody_instrument)
    pm.write(str(fake_raw_midi))

    def _fake_run(model, params, run_id, model_tmp):
        # Имитируем что MINGUS-runner написал raw MIDI
        return fake_raw_midi

    monkeypatch.setattr("pipeline.pipeline._run_model_subprocess", _fake_run)

    results = generate_all(_basic_progression(), run_id="20260427-000000-deadbeef")
    assert set(results.keys()) == {"mingus", "bebopnet", "ec2vae", "cmt", "commu", "polyffusion"}
    assert "melody_only" in results["mingus"]
    assert "with_chords" in results["mingus"]
    for model in ["bebopnet", "ec2vae", "cmt", "commu", "polyffusion"]:
        assert "error" in results[model]
        assert "not implemented" in results[model]["error"]


def test_generate_all_creates_tmp_dir(tmp_path: Path, fake_melody_instrument, monkeypatch):
    monkeypatch.setattr("pipeline.pipeline.OUTPUT_ROOT", tmp_path)
    fake_raw_midi = tmp_path / "fake_raw.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(fake_melody_instrument)
    pm.write(str(fake_raw_midi))
    monkeypatch.setattr("pipeline.pipeline._run_model_subprocess", lambda *a, **kw: fake_raw_midi)

    rid = "20260427-000000-cafebabe"
    generate_all(_basic_progression(), run_id=rid)
    assert (tmp_path / "_tmp" / rid).exists()
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Реализовать**

Файл `pipeline/pipeline/pipeline.py`:

```python
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from pipeline.config import (
    ADAPTERS, MELODY_PROGRAM, MODEL_CONFIGS, MODEL_NAMES,
    MODEL_RUNNER_SCRIPT, MODEL_VENV_PYTHON, OUTPUT_ROOT, RUNNER_TIMEOUT_SEC,
)
from pipeline.postprocess import postprocess
from pipeline.progression import ChordProgression
from pipeline.runner_protocol import RunnerError, run_runner_subprocess


def make_run_id(progression: ChordProgression) -> str:
    """Run-id формат: YYYYMMDD-HHMMSS-<8charhash>. Хвост — детерминирован от progression."""
    payload = repr((progression.chords, progression.tempo, progression.time_signature)).encode()
    h = hashlib.sha256(payload).hexdigest()[:8]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{h}"


def _run_model_subprocess(
    model: str, params: dict, run_id: str, model_tmp: Path,
) -> Path:
    payload = {"model": model, "run_id": run_id, "params": params}
    return run_runner_subprocess(
        venv_python=MODEL_VENV_PYTHON[model],
        runner_script=MODEL_RUNNER_SCRIPT[model],
        payload=payload,
        tmp_dir=model_tmp,
        timeout_sec=RUNNER_TIMEOUT_SEC,
    )


def generate_all(
    progression: ChordProgression,
    run_id: str | None = None,
) -> dict[str, dict]:
    """Один progression → набор MIDI от всех моделей.

    Для каждой модели либо `{"melody_only": Path, "with_chords": Path}`,
    либо `{"error": str}` если adapter — stub или runner упал.
    """
    run_id = run_id or make_run_id(progression)
    tmp_root = OUTPUT_ROOT / "_tmp" / run_id
    tmp_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for model in MODEL_NAMES:
        adapter = ADAPTERS[model]
        cfg = MODEL_CONFIGS[model]
        model_tmp = tmp_root / model
        model_tmp.mkdir(exist_ok=True)
        try:
            params = adapter.prepare(progression, cfg, model_tmp)
            raw_midi = _run_model_subprocess(model, params, run_id, model_tmp)
            melody = adapter.extract_melody(raw_midi)
            results[model] = postprocess(
                melody, progression, model, run_id, OUTPUT_ROOT,
                melody_program=MELODY_PROGRAM,
            )
        except NotImplementedError:
            results[model] = {"error": "not implemented (stub)"}
        except RunnerError as e:
            results[model] = {"error": str(e)}
    return results
```

- [ ] **Step 4: Запустить тесты**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/pipeline.py pipeline/tests/test_pipeline.py pipeline/tests/conftest.py
git commit -m "feat(pipeline): add generate_all orchestrator"
```

---

## Task 13: cli.py

**Files:**
- Create: `pipeline/pipeline/cli.py`
- Create: `pipeline/tests/test_cli.py`

- [ ] **Step 1: Failing tests**

Файл `pipeline/tests/test_cli.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch

import pretty_midi

from pipeline.cli import main


def _write_sample_progression(p: Path) -> None:
    p.write_text(json.dumps({
        "tempo": 120.0,
        "time_signature": "4/4",
        "chords": [["Cmaj7", 4], ["Am7", 4]],
    }))


def test_cli_generate_prints_table(tmp_path: Path, capsys, monkeypatch, fake_melody_instrument):
    src = tmp_path / "p.json"
    _write_sample_progression(src)

    monkeypatch.setattr("pipeline.pipeline.OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr("pipeline.cli.OUTPUT_ROOT", tmp_path)  # на случай если CLI читает напрямую

    fake_raw_midi = tmp_path / "fake_raw.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(fake_melody_instrument)
    pm.write(str(fake_raw_midi))
    monkeypatch.setattr("pipeline.pipeline._run_model_subprocess", lambda *a, **kw: fake_raw_midi)

    rc = main(["generate", str(src)])
    assert rc == 0
    out = capsys.readouterr().out
    for model in ["mingus", "bebopnet", "ec2vae", "cmt", "commu", "polyffusion"]:
        assert model in out
    assert "ok" in out
    assert "not implemented" in out


def test_cli_generate_invalid_path_exits_nonzero(tmp_path: Path, capsys):
    rc = main(["generate", str(tmp_path / "missing.json")])
    assert rc != 0
```

- [ ] **Step 2: Запустить — упадут**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

- [ ] **Step 3: Реализовать**

Файл `pipeline/pipeline/cli.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import OUTPUT_ROOT
from pipeline.pipeline import generate_all
from pipeline.progression import ChordProgression


def _format_table(results: dict[str, dict]) -> str:
    rows = [("model", "status", "melody_only", "with_chords")]
    for model, r in results.items():
        if "error" in r:
            rows.append((model, "error", r["error"], ""))
        else:
            rows.append((
                model, "ok",
                str(r["melody_only"].relative_to(OUTPUT_ROOT.parent.parent) if OUTPUT_ROOT.parent.parent in r["melody_only"].parents else r["melody_only"]),
                str(r["with_chords"].relative_to(OUTPUT_ROOT.parent.parent) if OUTPUT_ROOT.parent.parent in r["with_chords"].parents else r["with_chords"]),
            ))
    widths = [max(len(str(row[i])) for row in rows) for i in range(4)]
    out_lines = []
    for row in rows:
        out_lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out_lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen_p = sub.add_parser("generate", help="Generate MIDI for one progression")
    gen_p.add_argument("progression_path", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        if not args.progression_path.exists():
            print(f"error: {args.progression_path} not found", file=sys.stderr)
            return 2
        progression = ChordProgression.from_json(args.progression_path)
        results = generate_all(progression)
        print(_format_table(results))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Запустить тесты**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/cli.py pipeline/tests/test_cli.py
git commit -m "feat(pipeline): add CLI command 'pipeline generate <progression>'"
```

---

## Task 14: sample.json + end-to-end smoke test

**Files:**
- Create: `pipeline/test_progressions/sample.json`

- [ ] **Step 1: Создать sample.json**

Файл `pipeline/test_progressions/sample.json`:

```json
{
    "tempo": 120.0,
    "time_signature": "4/4",
    "chords": [
        ["Cmaj7", 4], ["Am7", 4], ["Dm7", 4], ["G7", 4],
        ["Cmaj7", 4], ["Am7", 4], ["Dm7", 4], ["G7", 4]
    ]
}
```

- [ ] **Step 2: Запустить весь pytest — все тесты должны быть зелёными**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest -v
```

Expected: всё проходит. Если что-то падает — фиксить до зелёного.

- [ ] **Step 3: Запустить end-to-end smoke (РЕАЛЬНЫЙ MINGUS)**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
```

Expected:
- Команда работает без необработанных исключений (~30-60 секунд из-за загрузки MINGUS).
- Выводит таблицу: mingus=ok, остальные 5 = error "not implemented (stub)".
- Файлы существуют:
  - `output/melody_only/mingus_<run_id>.mid`
  - `output/with_chords/mingus_<run_id>.mid`
  - `output/_tmp/<run_id>/mingus/{input.xml, raw.mid, stdout.log, stderr.log}`

Если падает на MINGUS-runner'е — посмотреть `output/_tmp/<run_id>/mingus/stderr.log`. Самые вероятные причины:
- DATA.json отсутствует → запустить preprocessing один раз (см. Task 0)
- Чекпоинт `Epochs 100` отсутствует → попробовать `MingusPipelineConfig(checkpoint_epochs=10)` если есть только Epochs 10
- Наш input.xml не парсится `gen.xmlToStructuredSong` → посмотреть что именно ожидает (возможно нужен `<work-title>`, `<sound tempo>`, и т.п.; сравнить с любым файлом из `models/MINGUS/C_generate/xml4gen/*.xml`); поправить `_xml_builders/mingus_xml.py`.

- [ ] **Step 4: Проверить полученные MIDI**

```bash
.venv/bin/python -c "
import pretty_midi
mel = pretty_midi.PrettyMIDI('output/melody_only/mingus_$(ls output/melody_only/ | head -1 | sed s/mingus_// | sed s/.mid//).mid' if False else None)
" 2>/dev/null

ls -la output/melody_only/ output/with_chords/
# открыть в любом MIDI-плеере или через ffplay/timidity для проверки на слух
```

Expected: оба файла открываются.

- [ ] **Step 5: Сгенерировать MP3 (если есть существующий convert_to_mp3.sh)**

```bash
ls /Users/maxos/PythonProjects/diploma/pipeline/convert_to_mp3.sh 2>/dev/null && \
  bash /Users/maxos/PythonProjects/diploma/pipeline/convert_to_mp3.sh
```

Expected: MP3 в `output/mp3/`. Если скрипта нет — пропустить.

- [ ] **Step 6: Commit**

```bash
git add pipeline/test_progressions/sample.json
git commit -m "feat(pipeline): add sample progression and verify end-to-end with MINGUS"
```

---

## Self-Review (заполняется автором плана)

**1. Spec coverage:**

| Раздел спеки | Покрытие |
|---|---|
| §1 Цель | Task 0–14 в совокупности |
| §2.1–2.8 Базовые решения | venv (T0), subprocess+JSON (T9), `_tmp` (T0,T9), run_id (T12), chord_render (T3,T4), 8 баров (sample.json T14) |
| §3 Слоистая архитектура + границы | base ABC (T5), inv. runner (T10), затравка per-model (T6) |
| §3 Stub-модели | T5 |
| §4 Структура файлов | T0 + создаются по мере задач |
| §5 Контракт runner'а + run_runner_subprocess | T9 + T10 |
| §6 MingusPipelineConfig + MingusAdapter | T6 + T7 + T8 |
| §7 ChordProgression / chord_vocab / chord_render / postprocess | T1, T2, T3, T4 |
| §8 generate_all | T12 |
| §9 CLI | T13 |
| §10 config.py + MELODY_PROGRAM | T11 |
| §11 sample.json | T14 |
| §12 Definition of Done | T14 step 2 + step 3 |

Все требования спеки покрыты задачами.

**2. Placeholder scan:** в плане нет "TODO/TBD/implement later/handle edge cases" без конкретики. Места где сказано «если упадёт — посмотреть в stderr и исправить» — это диагностические подсказки для исполнителя, не placeholder'ы.

**3. Type consistency:** `MingusPipelineConfig`, `MingusAdapter`, `ModelAdapter`, `RunnerError`, `run_runner_subprocess`, `generate_all`, `make_run_id`, `postprocess`, `render_chord_track`, `build_mingus_xml`, `parse_chord`, `chord_to_pitches` — имена совпадают между задачами. `MELODY_PROGRAM` параметризовано через `melody_program=` в `postprocess` (T4 фикстура и сигнатура — `melody_program: int`), в `generate_all` берётся из config (T12).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-27-pipeline-mingus.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session via executing-plans skill, batch execution with checkpoints for review.

Which approach?
