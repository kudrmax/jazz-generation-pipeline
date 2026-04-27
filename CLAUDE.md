# CLAUDE.md — гайд для агентов в этом проекте

> Глобальные правила (trash, git, conda, типизация, OOP) — `~/.claude/CLAUDE.md`.
> Этот файл дополняет их **проектной спецификой**.

## Контекст

ВКР НИУ ВШЭ МИЭМ: сравнительный анализ chord-conditioned моделей джазовой генерации мелодий. Pipeline принимает chord progression (JSON) → 6 моделей → MIDI/MP3.

Научный план — `THESIS_PLAN.md`. Технический дизайн — `docs/superpowers/specs/2026-04-27-pipeline-mingus-design.md`.

Сейчас полноценно реализована **MINGUS**. Остальные 5 моделей (BebopNet, EC²-VAE, CMT, ComMU, Polyffusion) — stub-адаптеры в `pipeline/pipeline/adapters/`. См. `README.md` → раздел Models.

## Архитектурный инвариант

Pipeline спроектирован слоисто. **Не нарушай это правило** — финальный code review за этим следит:

- **`pipeline-venv` ≠ `model-venv`.** Pipeline в своём venv не импортирует код моделей. Общается через subprocess + JSON через stdin (см. `pipeline/pipeline/runner_protocol.py`).
- **Адаптер — toolbox для модели** (`pipeline/pipeline/adapters/<model>.py`). Реализует `prepare(progression, tmp_dir) → params` и `extract_melody(raw_midi) → Instrument`. Всё model-специфичное знание живёт здесь.
- **Runner — чистая обёртка** (`pipeline/runners/<model>_runner.py`). Не импортирует ничего из `pipeline.*`. Принимает только то что **API модели реально требует** (не наши pipeline-решения вроде `seed_strategy`).
- **Общие модули** (`progression`, `chord_vocab`, `chord_render`, `postprocess`, `pipeline.py`, `runner_protocol.py`) — **без model-специфики**. Никаких `if model_name == "mingus"`, никаких `"Tenor Sax"` в постпроцессе, никаких `seed_strategy` в `ChordProgression`.

Граница «factor API модели или наш pipeline-фактор» — главный критерий: если факт принадлежит API модели (temperature, checkpoint_path) — он идёт в runner. Если это наше pipeline-решение (затравка, выбор тембра для melody_only.mid) — живёт в adapter или общем config.

`ChordProgression` несёт только **композиционный замысел** (chords + tempo + time_signature). Не добавляй туда `seed`, `temperature`, или per-model поля — у каждой модели свой формат затравки.

## Workflow

1. **Brainstorming** через `superpowers:brainstorming` — для любого нового фичи/решения. Не пиши код до согласования.
2. **Plan** через `superpowers:writing-plans` — bite-sized TDD-задачи в `docs/superpowers/plans/YYYY-MM-DD-<feature>.md`.
3. **Implement** через `superpowers:subagent-driven-development` — fresh subagent per task, spec/code review между задачами.
4. **TDD на каждый таск:** failing test → minimal implementation → passing test → commit. Не пропускай шаг "запустить failing test чтобы убедиться что он реально падает".

## Models / MINGUS

Единственная реализованная модель.

- **venv:** `models/MINGUS/.venv` (свой, не тот же что pipeline-venv)
- **Pre-requisites** (одноразовая подготовка машины):
  - `models/MINGUS/A_preprocessData/data/DATA.json` (~115 MB, генерируется через `python A_preprocessData/data_preprocessing.py --format xml`, ~3 минуты)
  - Pretrained чекпоинты `B_train/models/{pitchModel,durationModel}/MINGUS COND I-C-NC-B-BE-O Epochs 100.pt`
- **Затравка обязательна.** MINGUS — авторегрессионный, требует мелодию-seed в input.xml (не пустой XML с одними `<harmony>` тегами). 3 стратегии в `MingusPipelineConfig.seed_strategy`. Подробнее — `README.md` → раздел Models / MINGUS.
- **Runner:** `pipeline/runners/mingus_runner.py`. Запускается интерпретатором MINGUS-venv через subprocess.

## Запуск тестов и e2e

```bash
cd pipeline
.venv/bin/python -m pytest -v                                       # 81 passed ожидается
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json  # e2e с реальной MINGUS, ~30-60 сек
```

После любых правок в pipeline — оба должны быть зелёными.

## Чего НЕ делать

- **Не добавлять** model-specific код в общие модули. Если возникает соблазн положить «Tenor Sax» в `postprocess.py` — это бaг архитектуры, перенеси в adapter.
- **Не править** реализацию `MingusAdapter` так, чтобы конфиг приходил снаружи `prepare()`. Конфиг живёт в `__init__`. См. финальный code review для контекста.
- **Не удалять** debug-артефакты в `output/_tmp/<run_id>/` после успешного запуска — они полезны при дебаге следующих моделей.
- **Не переключать** ветки без `git stash` если в worktree есть изменения. Pipeline-venv в `.venv/` gitignored, переживает checkout. Но models/ и docs/ могут пострадать.
