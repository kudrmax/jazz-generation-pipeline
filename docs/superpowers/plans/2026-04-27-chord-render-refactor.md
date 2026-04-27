# Chord Render Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task.

**Goal:** Убрать `tempfile` round-trip из `postprocess`. Сейчас `chord_render` пишет MIDI в файл, `postprocess` тут же читает обратно — лишний disk I/O и грязный паттерн. После рефакторинга у `chord_render` будет два уровня: `build_chord_instrument(progression) -> pretty_midi.Instrument` (чистая логика) + `render_chord_track(progression, out_path) -> None` (тонкая обёртка-writer). `postprocess` использует первую напрямую.

**Architecture:** Не меняется на уровне публичного API. Внутри `chord_render` появляется приватный helper, `postprocess` упрощается.

**Tech Stack:** Python 3.12, pytest. Никаких новых зависимостей.

**Источник:** Финальный code review M2.

---

## Pre-conditions

- Branch `feat/pipeline-mingus`
- pipeline-robustness-fixes plan может быть применён или нет — этот рефакторинг от него не зависит.

---

## Task 1: split chord_render into builder + writer

**Files:**
- Modify: `pipeline/pipeline/chord_render.py` — добавить `build_chord_instrument`, `render_chord_track` становится тонкой обёрткой
- Modify: `pipeline/pipeline/postprocess.py` — убрать tempfile, вызвать `build_chord_instrument` напрямую
- Modify: `pipeline/tests/test_chord_render.py` — добавить тесты `build_chord_instrument`, существующие тесты `render_chord_track` остаются как «integration»

- [ ] **Step 1: Failing test for build_chord_instrument**

В `pipeline/tests/test_chord_render.py`, добавить новый блок тестов перед существующими:

```python
def test_build_chord_instrument_returns_instrument():
    inst = build_chord_instrument(_basic_progression())
    assert isinstance(inst, pretty_midi.Instrument)


def test_build_chord_instrument_program_zero():
    inst = build_chord_instrument(_basic_progression())
    assert inst.program == 0
    assert inst.is_drum is False


def test_build_chord_instrument_correct_note_count():
    """4 chord × 4-voice voicing = 16 notes."""
    inst = build_chord_instrument(_basic_progression())
    assert len(inst.notes) == 16


def test_build_chord_instrument_durations_match_progression():
    p = ChordProgression(
        chords=[("Cmaj7", 2), ("Am7", 6)],
        tempo=120.0,
        time_signature="4/4",
    )
    inst = build_chord_instrument(p)
    cmaj_notes = [n for n in inst.notes if n.start == 0.0]
    am_notes = [n for n in inst.notes if n.start > 0.0]
    assert len(cmaj_notes) == 4
    assert len(am_notes) == 4
    assert all(abs(n.end - 1.0) < 1e-6 for n in cmaj_notes)
    assert all(abs(n.start - 1.0) < 1e-6 for n in am_notes)
    assert all(abs(n.end - 4.0) < 1e-6 for n in am_notes)


def test_build_chord_instrument_does_not_write_file(tmp_path: Path):
    """build_chord_instrument should NOT touch disk — only return Instrument."""
    snapshot_before = set(tmp_path.iterdir())
    build_chord_instrument(_basic_progression())
    snapshot_after = set(tmp_path.iterdir())
    assert snapshot_before == snapshot_after
```

И обновить import:

```python
from pipeline.chord_render import build_chord_instrument, render_chord_track
```

- [ ] **Step 2: Run — fail (build_chord_instrument doesn't exist)**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/test_chord_render.py -v
```

Expected: ImportError.

- [ ] **Step 3: Refactor chord_render.py**

Заменить содержимое `pipeline/pipeline/chord_render.py`:

```python
from __future__ import annotations

from pathlib import Path

import pretty_midi

from pipeline.chord_vocab import chord_to_pitches
from pipeline.progression import ChordProgression


_CHORD_VELOCITY = 70


def build_chord_instrument(progression: ChordProgression) -> pretty_midi.Instrument:
    """ChordProgression → pretty_midi.Instrument с piano block voicing.

    Возвращает Instrument (program=0 Acoustic Grand Piano, name='ChordTrack',
    is_drum=False), без записи файла на диск.

    На каждый аккорд — block voicing (root+3rd+5th+7th или triada) длительностью
    `duration_in_beats * (60 / tempo)` секунд.
    """
    inst = pretty_midi.Instrument(program=0, name="ChordTrack", is_drum=False)

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
    return inst


def render_chord_track(progression: ChordProgression, out_path: Path) -> None:
    """Writes chord-track MIDI file using build_chord_instrument.

    Тонкая обёртка вокруг build_chord_instrument для случаев, когда нужен
    отдельный MIDI файл (например, для прослушивания chord-track в изоляции).
    Pipeline-orchestrator (postprocess) использует build_chord_instrument напрямую,
    минуя файл.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    pm.instruments.append(build_chord_instrument(progression))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
```

- [ ] **Step 4: Update postprocess.py — drop tempfile**

Заменить блок в `pipeline/pipeline/postprocess.py`:

```python
# было
    with tempfile.TemporaryDirectory() as td:
        chord_only_path = Path(td) / "chord_only.mid"
        render_chord_track(progression, chord_only_path)
        chord_pm = pretty_midi.PrettyMIDI(str(chord_only_path))

    pm_full = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    pm_full.instruments.append(copy.deepcopy(melody_normalized))
    for inst in chord_pm.instruments:
        pm_full.instruments.append(inst)
    pm_full.write(str(chords_path))

# стало
    pm_full = pretty_midi.PrettyMIDI(initial_tempo=progression.tempo)
    pm_full.instruments.append(copy.deepcopy(melody_normalized))
    pm_full.instruments.append(build_chord_instrument(progression))
    pm_full.write(str(chords_path))
```

И обновить импорт в начале файла:

```python
# было
from pipeline.chord_render import render_chord_track

# стало
from pipeline.chord_render import build_chord_instrument
```

Также удалить уже неиспользуемый `import tempfile`.

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: на 5 больше тестов чем было (5 новых из Step 1). Все проходят.

В частности `tests/test_postprocess.py` — 5 тестов — должны остаться зелёными (поведение не меняется, только реализация).

- [ ] **Step 6: Run end-to-end smoke**

```bash
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
```

Expected: same output as before — mingus=ok, others=stub. Размер `with_chords/<...>.mid` файла может отличаться на байт-два (другой порядок MIDI events), но это нормально.

- [ ] **Step 7: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/pipeline/chord_render.py pipeline/pipeline/postprocess.py pipeline/tests/test_chord_render.py
git commit -m "refactor(pipeline): split chord_render into build_chord_instrument + writer"
```

---

## Self-Review

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest -v
# expect: previous count + 5 new tests in test_chord_render.py
```

```bash
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
# expect: mingus=ok, 5 stubs error
```

Спецификация поведения постпроцесса не меняется — добавился только новый builder API. Если test_postprocess сломался — значит изменилось поведение постпроцесса (не должно было).

## Definition of Done

- ✅ `chord_render.py` экспортирует и `build_chord_instrument`, и `render_chord_track`
- ✅ `postprocess.py` не использует `tempfile`
- ✅ pytest +5 тестов
- ✅ end-to-end smoke даёт тот же output как до рефакторинга
