# Pipeline для chord-conditioned джазовой генерации мелодий

> ВКР НИУ ВШЭ МИЭМ им. А. Н. Тихонова. Сравнительный анализ нейросетевых моделей
> для генерации монофонных джазовых соло по аккордовой прогрессии.
> Автор: Кудряшов М. Д. Защита: 2026.

Принимаешь chord progression в JSON → пайплайн прогоняет её через 6 ML-моделей разных архитектур (RNN, Transformer, VAE, Diffusion) → получаешь MIDI и MP3 с мелодией.

Сейчас полноценно реализована **MINGUS**. Остальные 5 моделей (BebopNet, EC²-VAE, CMT, ComMU, Polyffusion) присутствуют как stub-адаптеры, которые возвращают `error: not implemented (stub)`. Каркас спроектирован так, что добавление каждой следующей модели сводится к написанию её adapter'а и runner'а — без правки общих модулей.

Научный план — `THESIS_PLAN.md`.

---

## Quickstart

**Шаг 1.** Поставить pipeline-venv (один раз):

```bash
cd pipeline
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Шаг 2.** Подготовить MINGUS (один раз — подробнее см. раздел «Models / MINGUS» ниже):

```bash
# из корня репо
git submodule update --init --recursive    # подтянет models/MINGUS из нашего fork
cd models/MINGUS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-py312.txt      # пакеты под Python 3.12 + numpy 2.x
export PYTHONPATH=$PWD
python A_preprocessData/data_preprocessing.py --format xml   # ~3 мин, генерирует DATA.json
```

Pretrained чекпоинты `Epochs 100` уже идут вместе с submodule.

**Шаг 3.** Сгенерировать:

```bash
cd pipeline
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
```

Output:
- `pipeline/output/melody_only/mingus_<run_id>.mid` — монофонная мелодия (Tenor Sax)
- `pipeline/output/with_chords/mingus_<run_id>.mid` — мелодия + наш piano chord-track
- `pipeline/output/_tmp/<run_id>/mingus/` — debug-артефакты (input.xml, raw.mid, stdout/stderr.log)

5 stub-моделей в выводе — `error: not implemented (stub)`.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│ pipeline-venv  (свой набор зависимостей)                    │
│                                                             │
│   ChordProgression (JSON)                                   │
│           │                                                 │
│           ▼                                                 │
│   generate_all(progression)                                 │
│       │                                                     │
│       ├── for each model:                                   │
│       │     ┌──────────────────┐                            │
│       │     │ adapter.prepare  │  pipeline-side:            │
│       │     │  (in pipeline-   │  знает про конкретную      │
│       │     │   venv)          │  модель, готовит вход      │
│       │     └────────┬─────────┘                            │
│       │              │                                      │
│       │   ┌──────────▼──────────┐                           │
│       │   │ subprocess + JSON    │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─┐
│       │   │  via stdin           │                          │ │
│       │   └──────────────────────┘                          │ │
│       │              ▲                                      │ │
│       │   ┌──────────┴──────────┐                           │ │
│       │   │ adapter.extract_     │  pipeline-side:          │ │
│       │   │  melody              │  достаёт монофонную      │ │
│       │   │  → Instrument        │  мелодию из raw output   │ │
│       │   └──────────┬───────────┘                          │ │
│       │              ▼                                      │ │
│       │   ┌──────────────────────┐                          │ │
│       │   │ postprocess          │  generic:                │ │
│       │   │  + chord_render      │  пишет 2 нормализованных │ │
│       │   │                      │  MIDI (melody_only +     │ │
│       │   │                      │  with_chords)            │ │
│       │   └──────────────────────┘                          │ │
└─────────────────────────────────────────────────────────────┘ │
                                                                │
┌─────────────────────────────────────────────────────────────┐ │
│ model-venv  (например models/MINGUS/.venv)                  │ │
│                                                             │ │
│  pipeline/runners/<model>_runner.py                         │◄┘
│   • импортирует только torch + код самой модели             │
│   • не импортирует ничего из pipeline.*                     │
│   • читает JSON со stdin → вызывает inference → пишет MIDI  │
└─────────────────────────────────────────────────────────────┘
```

**Ключевые принципы:**

1. **pipeline-venv ≠ model-venv.** Каждая ML-модель тащит несовместимые версии torch/numpy/etc, поэтому каждая живёт в собственном venv. Pipeline в своём venv даже не пытается импортировать model-код — общается через subprocess.

2. **Адаптер — toolbox для модели.** На каждую модель есть один adapter с двумя методами: `prepare(progression, tmp_dir)` строит вход, `extract_melody(raw_midi)` парсит выход. Всё model-специфичное знание живёт здесь.

3. **Runner — чистая обёртка.** Скрипт в model-venv, который ничего не знает про pipeline. Принимает JSON со stdin (только то что MINGUS API реально ждёт), пишет MIDI в указанный путь, exit 0/≠0.

4. **Общие модули — без model-специфики.** `chord_render`, `postprocess`, `progression`, `chord_vocab`, `pipeline.py` не знают ни про MINGUS, ни про любую другую модель — только про абстрактный `ChordProgression` и `Instrument`.

Подробный архитектурный спек — `docs/superpowers/specs/2026-04-27-pipeline-mingus-design.md`.

---

## Структура

```
diploma/
├── README.md                  ← этот файл
├── CLAUDE.md                  ← инструкции для AI-агентов в этом репо
├── THESIS_PLAN.md             ← научный план ВКР, состояние моделей
│
├── pipeline/                  ← Python-package с пайплайном
│   ├── .venv/                 ← изолированный venv (gitignored)
│   ├── pyproject.toml         ← конфигурация проекта + pytest
│   ├── requirements.txt       ← зависимости pipeline-venv
│   ├── pipeline/              ← Python-package
│   │   ├── progression.py     ← ChordProgression dataclass + JSON I/O
│   │   ├── chord_vocab.py     ← парсинг аккордов, MIDI-pitches
│   │   ├── chord_render.py    ← ChordProgression → piano chord-track
│   │   ├── postprocess.py     ← мелодия + наш chord-track → 2 MIDI
│   │   ├── runner_protocol.py ← subprocess + JSON + RunnerError
│   │   ├── pipeline.py        ← generate_all(progression) — оркестратор
│   │   ├── cli.py             ← python -m pipeline.cli generate <path>
│   │   ├── config.py          ← пути venv'ов, ADAPTERS, константы
│   │   ├── adapters/          ← по 1 adapter на модель
│   │   │   ├── base.py        ← ABC ModelAdapter
│   │   │   ├── mingus.py      ← MingusAdapter + MingusPipelineConfig
│   │   │   └── {bebopnet,ec2vae,cmt,commu,polyffusion}.py  (stubs)
│   │   └── _xml_builders/
│   │       └── mingus_xml.py  ← генерация MusicXML для MINGUS (3 стратегии)
│   ├── runners/               ← запускаются в model-venv'ах
│   │   └── mingus_runner.py
│   ├── tests/                 ← pytest unit-тесты (81 passing)
│   ├── test_progressions/
│   │   └── sample.json        ← тестовая прогрессия Cmaj7-Am7-Dm7-G7 ×2
│   └── output/                ← результаты генерации (gitignored)
│       ├── melody_only/<model>_<run_id>.mid
│       ├── with_chords/<model>_<run_id>.mid
│       └── _tmp/<run_id>/<model>/{input.xml,raw.mid,stdout.log,stderr.log}
│
├── models/                    ← склонированные репозитории моделей (gitignored)
│   └── MINGUS/                ← со своим .venv, DATA.json, чекпоинтами
│
└── docs/superpowers/
    ├── specs/                 ← архитектурные спеки
    └── plans/                 ← планы реализации
```

---

## Models

### MINGUS

**Статус:** ✅ полностью реализована.

**Источник:** [MINGUS — Cosenza & Diviltio, 2021](https://github.com/EmanueleCosenza/MINGUS). Seq2Seq Transformer с двумя декодерами (pitch + duration), conditioning на C+NC+B+BE+O. Pretrained на Weimar Jazz Database.

**Что в нашем pipeline:**
- Adapter — `pipeline/pipeline/adapters/mingus.py` (`MingusAdapter`, `MingusPipelineConfig`)
- XML-билдеры — `pipeline/pipeline/_xml_builders/mingus_xml.py` (3 стратегии затравочной мелодии: `tonic_whole`, `tonic_quarters`, `custom_xml`)
- Runner — `pipeline/runners/mingus_runner.py` (запускается в `models/MINGUS/.venv`)

**Pre-requisites** (один раз для машины):

`models/MINGUS` — git submodule на наш fork [kudrmax/MINGUS](https://github.com/kudrmax/MINGUS) с применёнными патчами под Python 3.12 и numpy 2.x. Upstream — [vincenzomadaghiele/MINGUS](https://github.com/vincenzomadaghiele/MINGUS).

```bash
# из корня нашего репо
git submodule update --init --recursive       # склонирует MINGUS-fork в models/MINGUS
                                              # вместе с pretrained чекпоинтами Epochs 100

cd models/MINGUS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-py312.txt    # см. в форке: torch 2.11, numpy 2.4, music21 6.7, и т.д.

export PYTHONPATH=$PWD
python A_preprocessData/data_preprocessing.py --format xml   # ~3 минуты, генерирует DATA.json
```

После этого `pipeline.cli generate` будет вызывать MINGUS через subprocess в `models/MINGUS/.venv`.

**Затравочная мелодия.** MINGUS требует не только аккорды, но и мелодию-затравку в XML — он её использует как prefix для авторегрессии. Качество выхода зависит от затравки. Стратегии в нашем `MingusPipelineConfig`:

- `tonic_whole` (default) — 1 whole-нота тоники в каждом баре. Простейшая, MINGUS играет редко.
- `tonic_quarters` — 4 четверти тоники в каждом баре. Плотнее.
- `custom_xml` — путь к готовому MusicXML с темой (например `models/MINGUS/C_generate/xml4gen/Donna_Lee_short.xml` — bebop-тема даёт самый плотный output).

### BebopNet

**Статус:** 🔴 stub-адаптер. См. `THESIS_PLAN.md` — раздел про BebopNet.

### EC²-VAE

**Статус:** 🔴 stub-адаптер. См. `THESIS_PLAN.md` — раздел про EC²-VAE.

### CMT

**Статус:** 🔴 stub-адаптер. См. `THESIS_PLAN.md` — раздел про CMT.

### ComMU

**Статус:** 🔴 stub-адаптер. См. `THESIS_PLAN.md` — раздел про ComMU.

### Polyffusion

**Статус:** 🔴 stub-адаптер. См. `THESIS_PLAN.md` — раздел про Polyffusion.

---

## Тесты

```bash
cd pipeline
.venv/bin/python -m pytest -v
```

Ожидается: **81 passed**. Юнит-тесты покрывают каждый модуль; runner MINGUS тестируется через end-to-end smoke (запуск `cli generate` на `sample.json`).

---

## Окружение

- macOS arm64 (Apple Silicon), но pipeline переносится на любой Python 3.12 с pretty_midi/music21
- Python 3.12 (`/opt/homebrew/bin/python3.12` по умолчанию)
- Без conda — везде venv
- MPS работает; для тяжёлых train MINGUS-checkpoint'а нужен CUDA (Colab T4 / RunPod), но inference и так быстрый на CPU
