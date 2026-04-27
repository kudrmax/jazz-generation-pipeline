# Дизайн интеграции BebopNet в pipeline — production

**Дата:** 2026-04-27
**Контекст:** ВКР НИУ ВШЭ МИЭМ. Базовые дизайны — `2026-04-27-pipeline-mingus-design.md` и `2026-04-27-pipeline-cmt-design.md`. MINGUS и CMT уже в `master`. Этот документ — дельта для третьей модели (BebopNet).

Production-интеграция. Любая `ChordProgression` превращается в реальную мелодию-соло BebopNet. Замена весов на финальные — подмена `BEBOPNET_MODEL_DIR` (или `BEBOPNET_CHECKPOINT_FILENAME`) в `pipeline/pipeline/config.py`, без правок кода адаптера/runner'а/общих модулей.

---

## 1. Цель и Definition of Done

**Цель.** Подключить BebopNet (Transformer-XL chord-conditioned LSTM из статьи Haviv et al. ISMIR'20) как третью модель pipeline. На вход — `ChordProgression`. На выход — MIDI с реальной chord-conditioned монофонической мелодией Tenor Sax длиной ровно `progression.num_bars()` тактов.

**DoD:**

- ✅ `models/bebopnet-code` — git submodule на `kudrmax/bebopnet-code.git`. Форк содержит 4 патча совместимости (Python 3.12 / numpy 2.x / PyTorch 2.x / bidict 0.20+) и `requirements-py312.txt`.
- ✅ `models/bebopnet-code/.venv` создан, импорт `MemTransformerLM` и `MusicGenerator` работает.
- ✅ `python -m pipeline.cli generate test_progressions/sample.json` показывает `bebopnet = ok`.
- ✅ Две разные `ChordProgression` дают разные мелодии (chord-conditioning подтверждено).
- ✅ MIDI-выход валиден: `pretty_midi.PrettyMIDI(...)` открывает; `melody_only.mid` ровно 1 трек, len(notes) > 0, монофоничность (нет наложений нот по времени).
- ✅ Длина мелодии в тактах = `progression.num_bars()`. Семантика «что подал, то и получил».
- ✅ Адаптер не содержит магических чисел длины/sampling — все параметры через `BebopNetPipelineConfig` или из progression.
- ✅ Все pytest-тесты зелёные. `BebopNetAdapter` удалён из stub-параметризации в `tests/adapters/test_base.py`.
- ✅ MINGUS продолжает работать: рефактор xml builder сохранил поведение MINGUS-XML 1-в-1, все его e2e-тесты + `test_mingus_*` зелёные.
- ✅ Замена весов — переписать `BEBOPNET_MODEL_DIR` (или `BEBOPNET_CHECKPOINT_FILENAME`) в `pipeline/pipeline/config.py`. Никаких других правок.

**За пределами этого раунда:**
- ❌ Reward-induction / score-modeling (BebopNet поддерживает RL поверх соло — наш pipeline это не использует, оставляем default sampling).
- ❌ Backing-track MP3 mixing (BebopNet может сливать соло поверх audio backing track — pipeline для MP3 использует свой `convert_to_mp3.sh`).
- ❌ Метрики качества (FMD, scale match).

---

## 2. Архитектурный инвариант (без отступлений)

Полное соответствие MINGUS/CMT-дизайну (см. `pipeline/CLAUDE.md`):

- **`pipeline-venv` ≠ `models/bebopnet-code/.venv`.** Pipeline в своём venv не импортирует код BebopNet. Общается через `subprocess + JSON через stdin`.
- **Adapter (`pipeline/pipeline/adapters/bebopnet.py`)** — toolbox. Реализует `prepare(progression, tmp_dir) → params: dict` и `extract_melody(raw_midi) → Instrument`. Знает что BebopNet принимает MusicXML.
- **Runner (`pipeline/runners/bebopnet_runner.py`)** — чистая обёртка. **Не импортирует** ничего из `pipeline.*`. Принимает только то что API BebopNet реально требует.
- **Общие модули** (`progression`, `chord_vocab`, `chord_render`, `postprocess`, `pipeline.py`, `runner_protocol.py`) — **не трогаются**.

Граница «factor API модели или наш pipeline-фактор»:
- API модели → runner: путь к model_dir / checkpoint, `temperature`, `top_p`, `beam_search`, `beam_width`, `device`, `num_measures`.
- Наш pipeline-фактор → adapter (через `BebopNetPipelineConfig`): `seed_strategy`, `custom_xml_path`.

`ChordProgression` несёт только композиционный замысел. Никаких BebopNet-полей в ней.

---

## 3. BebopNet-вход: что требуется от модели

### 3.1. Вход

BebopNet принимает **MusicXML** (тот же формат что MINGUS — music21 score):
- Часть `<part>` с инструментом (Tenor Sax по умолчанию).
- Поверх такта — `<harmony>` теги с chord-symbols.
- Внутри такта — последовательность `<note>` (head melody / тоник-затравка).
- `<time>`, `<tempo>` метаданные.

XML парсится через `m21.converter.parse(xml_path)` внутри `MusicGenerator.start_from_xml(...)`.

### 3.2. Параметры sampling (BebopNet API)

| параметр | смысл |
|---|---|
| `--num_measures` | сколько тактов соло сгенерировать поверх progression |
| `--temperature` | temperature softmax (default 1.0) |
| `--top-p` | nucleus sampling (default true) |
| `--beam_search` | `""` / `"note"` / `"measure"` (default `"measure"`) |
| `--beam_width` | ширина beam search (default 2) |
| `--no-cuda` | CPU only (нужен на arm64) |
| `--batch_size` | параллельная генерация (нам достаточно 1) |
| `--create_mp3 0` | НЕ сливать MP3 backing track (это делает наш `convert_to_mp3.sh`) |

### 3.3. Артефакты модели

`model_dir` содержит:
- `model.pt` — чекпоинт.
- `converter_and_duration.pkl` — pitch+duration vocabulary (pickled bidict).
- `args.json` — аргументы trains-time (модель использует часть из них в `MemTransformerLM(**kwargs)`).
- `train_model.yml` — обучающий конфиг.

Все эти файлы загружаются runner'ом. Адаптер их не трогает.

### 3.4. Выход

`MusicGenerator.create_song(...)` пишет MIDI с одним монофоничным треком (Tenor Sax, program 65 по дефолту BebopNet) + опциональный MusicXML.

---

## 4. Pipeline-фактор: затравка (3 стратегии)

`BebopNetPipelineConfig.seed_strategy`. Полная аналогия с MINGUS — XML-формат тот же.

| `seed_strategy` | поведение |
|---|---|
| `tonic_whole` (default) | Целая нота тоники текущего аккорда в каждом баре. |
| `tonic_quarters` | 4 четверти тоники в каждом баре (по биту). |
| `custom_xml` | Пользовательский MusicXML (`config.custom_xml_path`). Перекрывает progression. |

Стратегии **идентичны** MINGUS. После рефакторинга XML-builder'а (раздел 10) и MINGUS, и BebopNet используют **один общий код** для построения XML.

---

## 5. Контракт adapter'а

```python
@dataclass
class BebopNetPipelineConfig:
    """Все настройки BebopNet на уровне pipeline. immutable после init."""
    model_dir: Path                       # содержит model.pt + converter.pkl + args.json + train_model.yml
    repo_path: Path                       # models/bebopnet-code/
    checkpoint_filename: str = "model.pt" # ← подмена весов: эта строка или model_dir
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
        # 1. Validation (раздел 6).
        # 2. Build XML через общий jazz_xml.build_xml(...) — раздел 10.
        # 3. Возвращает params для runner'а.
        ...

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        # BebopNet пишет один монофоничный трек. Возвращаем pm.instruments[0].
        # Если 0 инструментов — ValueError с понятным сообщением.
        ...
```

`prepare` возвращает:

```python
{
    "input_xml_path":      str(xml_path),
    "output_midi_path":    str(tmp_dir / "raw.mid"),
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

---

## 6. Валидация в `prepare`

Адаптер валидирует только **наш pipeline-фактор**. Существование model_dir / checkpoint валидирует runner.

1. `seed_strategy not in {"tonic_whole", "tonic_quarters", "custom_xml"}` → ValueError.
2. `seed_strategy == "custom_xml"` && `custom_xml_path is None` → ValueError.
3. `not progression.chords` → ValueError("progression has no chords").
4. `progression.beats_per_bar()` divides `progression.total_beats()` evenly — это уже проверено в `progression.num_bars()`, не дублируем.

Никакой валидации по time_signature: BebopNet через music21 принимает любые тактовые размеры, которые music21 умеет читать.

---

## 7. Контракт runner'а

`pipeline/runners/bebopnet_runner.py`. Запускается `models/bebopnet-code/.venv/bin/python`. **Не импортирует ничего из `pipeline.*`.**

Шаги:

1. `os.chdir(repo_path); sys.path.insert(0, repo_path)`.
2. Загрузить args (`args.json`) и pickle-конвертер (`converter_and_duration.pkl`) из `model_dir`. Bidict-monkey-patch уже в форке (см. раздел 13), импортируется автоматически.
3. Построить `MemTransformerLM(**kwargs)` из args, загрузить state из `model.pt` через `torch.load(..., map_location=device, weights_only=False)`.
4. Парсить `input_xml_path` через `m21.converter.parse(...)` и сконструировать `MusicGenerator`.
5. Вызвать `MusicGenerator.create_song(num_measures=N, temperature=..., top_p=..., beam_search=..., beam_width=..., batch_size=1, no_cuda=(device=='cpu'), create_mp3=0)`.
6. Записать результирующий MIDI в `output_midi_path`.
7. `exit 0`. Ошибка → `traceback.print_exc(); exit 1`.

Точные имена методов уточняем при реализации (BebopNet `MusicGenerator` имеет более сложный API чем MINGUS — там есть head_extension / improvisation модули). Главное соблюдать контракт «JSON params → MIDI на диске».

---

## 8. BebopNet-venv

`models/bebopnet-code/.venv` (Python 3.12). Зависимости (минимум для inference, в порядке памяти проекта):

```
torch>=2.10
numpy>=2.0
music21
lxml
pretty_midi
pyyaml
bidict
ConfigArgParse
imbalanced-learn  # импортируется в data_prep, может потребоваться
scipy             # impl deps в torch utils
tqdm
matplotlib
pandas
```

Закрепить как `models/bebopnet-code/requirements-py312.txt` в форке (по аналогии с MINGUS и CMT).

---

## 9. Изменения в pipeline-конфигурации

`pipeline/pipeline/config.py`:

```python
BEBOPNET_REPO_PATH:           Path = DIPLOMA_ROOT / "models" / "bebopnet-code"
BEBOPNET_MODEL_DIR:           Path = BEBOPNET_REPO_PATH / "training_results" / "transformer" / "model"
BEBOPNET_CHECKPOINT_FILENAME: str  = "model.pt"  # ← подмена весов

ADAPTERS["bebopnet"] = BebopNetAdapter(BebopNetPipelineConfig(
    model_dir=BEBOPNET_MODEL_DIR,
    repo_path=BEBOPNET_REPO_PATH,
    checkpoint_filename=BEBOPNET_CHECKPOINT_FILENAME,
    seed_strategy="tonic_whole",
    temperature=1.0,
    top_p=True,
    beam_search="measure",
    beam_width=2,
    device="cpu",
))
MODEL_RUNNER_SCRIPT["bebopnet"] = RUNNERS_ROOT / "bebopnet_runner.py"
```

**Замена весов на финальные:**
- Положить новые `(model.pt, converter_and_duration.pkl, args.json)` в новую папку, например `BEBOPNET_REPO_PATH / "training_results" / "final_v1" / "model"`.
- Поменять `BEBOPNET_MODEL_DIR` (или `BEBOPNET_CHECKPOINT_FILENAME`).

Ровно одна-две строки. Адаптер и runner возьмут новые без изменений.

---

## 10. Архитектурный рефактор: общий XML builder

**Мотивация.** XML-вход у MINGUS и BebopNet идентичен (music21 score с chord-symbols + тоник-затравка). Существующий `pipeline/pipeline/_xml_builders/mingus_xml.py` уже почти что "общий" — model-specific только то, что он принимает `MingusPipelineConfig` как источник полей `seed_strategy` / `custom_xml_path`.

**Решение.** Параметризовать функцию по этим полям напрямую, без зависимости от конкретного config-типа.

**Новый файл:** `pipeline/pipeline/_xml_builders/jazz_xml.py`:

```python
def build_xml(
    progression: ChordProgression,
    seed_strategy: Literal["tonic_whole", "tonic_quarters", "custom_xml"],
    custom_xml_path: Path | None,
    out_path: Path,
    melody_instrument_name: str = "Tenor Sax",
) -> None:
    """Пишет MusicXML, готовый к скармливанию music21-парсеру.

    Аргументы — тот же набор что MINGUS использует, но без зависимости
    от MingusPipelineConfig. Используется обоими адаптерами (MINGUS, BebopNet).
    """
```

`MingusAdapter.prepare`: вызов `build_xml(progression, cfg.seed_strategy, cfg.custom_xml_path, xml_path, "Tenor Sax")`.
`BebopNetAdapter.prepare`: вызов `build_xml(progression, cfg.seed_strategy, cfg.custom_xml_path, xml_path, cfg.melody_instrument_name)`.

**Старый файл `mingus_xml.py` удаляется.** Тесты `test_mingus_xml.py` переезжают как `test_jazz_xml.py`, расширяются параметризацией по `melody_instrument_name`.

**Это часть данной интеграции, не отдельная задача.** Без рефакторинга появилось бы дублирование `_xml_builders/bebopnet_xml.py` ≈ 99% копии MINGUS — это и есть тот случай когда CLAUDE.md требует «таргетированных улучшений по ходу работы».

---

## 11. Артефакты на хосте (pre-requisites)

По аналогии с `DATA.json` в MINGUS и `result/` в CMT:

- `models/bebopnet-code/training_results/transformer/model/model.pt` — чекпоинт.
- `models/bebopnet-code/training_results/transformer/model/converter_and_duration.pkl` — pickled vocabulary.
- `models/bebopnet-code/training_results/transformer/model/args.json` — train-time hyperparameters.
- `models/bebopnet-code/training_results/transformer/model/train_model.yml` — train config.

Папка `training_results/` — gitignored на стороне submodule (см. раздел 13).

---

## 12. Структура файлов после изменений

```
diploma/
├── docs/superpowers/
│   ├── specs/2026-04-27-pipeline-bebopnet-design.md
│   └── plans/2026-04-27-pipeline-bebopnet.md
├── models/bebopnet-code/                        # submodule на kudrmax/bebopnet-code
│   ├── .venv/                                   # gitignored
│   ├── requirements-py312.txt                   # в форке
│   ├── jazz_rnn/A_data_prep/gather_data_from_xml.py     # пропатчен в форке
│   ├── jazz_rnn/B_next_note_prediction/generate_from_xml.py  # пропатчен
│   ├── jazz_rnn/B_next_note_prediction/music_generator.py   # пропатчен
│   ├── jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py  # пропатчен
│   └── training_results/                        # gitignored локально
│       └── transformer/model/
│           ├── model.pt
│           ├── converter_and_duration.pkl
│           ├── args.json
│           └── train_model.yml
├── pipeline/
│   ├── pipeline/
│   │   ├── adapters/
│   │   │   ├── bebopnet.py                      # переписан (был stub)
│   │   │   └── mingus.py                        # обновлён: использует jazz_xml.build_xml
│   │   ├── _xml_builders/
│   │   │   ├── jazz_xml.py                      # NEW: общий builder
│   │   │   └── (mingus_xml.py удалён)
│   │   └── config.py                            # +BEBOPNET_* константы
│   ├── runners/
│   │   └── bebopnet_runner.py                   # NEW
│   └── tests/
│       ├── adapters/
│       │   ├── test_base.py                     # BebopNetAdapter убран из stub
│       │   ├── test_bebopnet_config.py          # NEW
│       │   ├── test_bebopnet_validation.py      # NEW
│       │   ├── test_bebopnet_prepare.py         # NEW
│       │   └── test_bebopnet_extract_melody.py  # NEW
│       └── _xml_builders/
│           └── test_jazz_xml.py                 # rename из test_mingus_xml.py + расширение
```

---

## 13. Изменения в форке `kudrmax/bebopnet-code`

Форк сейчас — clean upstream `shunithaviv/bebopnet-code`. Push 5 коммитов:

1. **`fix: comment out tkinter SongLabels import (not used in inference)`** — `jazz_rnn/A_data_prep/gather_data_from_xml.py:14`. SongLabels тянет tkinter, при inference не используется.

2. **`fix: bidict 0.20+ compat (monkey-patch _fwdm/_invm) + torch.load(weights_only=False)`** — `jazz_rnn/B_next_note_prediction/generate_from_xml.py`. bidict переименовал атрибуты, наш pickled converter ждёт старые. Плюс новый PyTorch требует `weights_only=False` для load старых state-dict.

3. **`fix: numpy 2.x bool indexing in music_generator`** — `jazz_rnn/B_next_note_prediction/music_generator.py:372`. `int(c)` для индексации (`numpy.bool` больше не auto-converts).

4. **`fix: PyTorch 2.x bool tensor API (masked_fill, 1-bool subtract)`** — `jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py`. Два места: attn_mask требует bool через `.bool()`, `1 - eos_mask` (bool) → ошибка, использовать `.long()`.

5. **`add: requirements-py312.txt + .gitignore`** — runtime requirements для Python 3.12 / arm64. `.gitignore` (новый файл) с `.venv/`, `__pycache__/`, `output/`, `training_results/`.

Каждый патч точечный (1-15 строк изменений), хорошо документирован в commit message.

---

## 14. Тесты

| тест | поведение |
|---|---|
| `test_jazz_xml.py` | (rename из `test_mingus_xml.py`) builder создаёт корректный MusicXML для всех 3 стратегий; разные progression → разные XML; параметризация по `melody_instrument_name`; `custom_xml` копирует файл |
| `test_bebopnet_config.py` | `BebopNetAdapter()` без аргументов TypeError; defaults в `BebopNetPipelineConfig` соответствуют разделу 5 |
| `test_bebopnet_validation.py` | `prepare` валит ValueError для: пустой progression, неверного seed_strategy, custom_xml без path |
| `test_bebopnet_prepare.py` | `prepare` создаёт tmp_dir, пишет input.xml, возвращает dict с правильными ключами; разные progression → разные `input.xml` (chord-conditioning); `num_measures = progression.num_bars()` |
| `test_bebopnet_extract_melody.py` | вытаскивает первый трек из MIDI с одним инструментом (BebopNet монофония); валит ValueError если MIDI пустой |
| MINGUS тесты (`test_mingus_*`) | продолжают проходить после рефакторинга xml builder — проверяется что MINGUS XML 1-в-1 совпадает с тем что было до рефакторинга |
| e2e (Task 12) | `pipeline.cli generate sample.json` → `bebopnet = ok`, разные progressions → разные мелодии |

---

## 15. Открытые вопросы / риски

| # | Риск | Митигация |
|---|---|---|
| 1 | `MusicGenerator.create_song()` точные имена параметров и сигнатура — нужно проверить на стадии реализации runner'а (Task 11). Возможны мелкие отличия от того что описано в §7 | Smoke-тест runner'а в Task 11 step 2 запускает с реальным чекпоинтом и progression — все API-несоответствия проявятся там, фикс на месте |
| 2 | `--song` параметр в BebopNet CLI содержит фиксированный `song_params_dict` (Fly Me to the Moon, Giant Steps и т.п.). Наш runner обходит CLI и передаёт XML напрямую через `--xml_` или прямой вызов `MusicGenerator(xml_path=...)` | Используем прямой вызов внутри Python, минуя CLI — никакой зависимости от song_params_dict |
| 3 | Постройка `MemTransformerLM(**kwargs)` использует часть параметров из `args.json` — точный список kwargs определяется в коде BebopNet. Возможна несовместимость с новым PyTorch (некоторые параметры устарели) | На этапе Task 11 smoke-теста — увидим. Если есть deprecated параметры, фильтруем их в runner'е перед передачей в model |
| 4 | BebopNet's `MusicGenerator` пишет MIDI с ticks_per_beat=10080 (см. memory) — нестандартное значение. `pretty_midi` это переварит, но pipeline-postprocess может показать странные таймиги | Принимаем как есть. В тесте `test_bebopnet_extract_melody` проверяем что `pm.get_end_time() > 0` и notes есть; точные таймиги — задача downstream обработки |
| 5 | На чужой машине `training_results/transformer/model/*` отсутствует | `pipeline.cli generate` упадёт с понятной ошибкой `FileNotFoundError` от runner'а. Pre-requisite задокументирован |
| 6 | Multi-track выход: BebopNet может писать backing-track инструменты (если `--create_mp3` включён). С `create_mp3=0` остаётся только melody | `extract_melody` берёт `pm.instruments[0]`. Если выход неожиданно multi-track — расширим логику отбора по `program=65` или `name~="Sax"`. Smoke-тест Task 11 покажет |
