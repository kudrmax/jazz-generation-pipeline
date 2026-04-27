# Дизайн пайплайна chord-conditioned генерации — первая модель MINGUS

**Дата:** 2026-04-27
**Статус:** одобрен пользователем (брейншторм закрыт)
**Контекст:** ВКР НИУ ВШЭ МИЭМ, тема — сравнение моделей chord-conditioned джазовой генерации. Подробности — `THESIS_PLAN.md`. Контракт пайплайна на верхнем уровне — `PIPELINE_SPEC.md`. Этот документ — детализация архитектуры **финального проекта**; в этом раунде реализуем общий каркас и первую модель.

---

## 1. Цель

Построить общий каркас пайплайна `generate_all(progression) → {model_name: paths_or_error}`, в который встраиваются 6 моделей. В этом раунде полностью реализована только **MINGUS**; остальные пять моделей (BebopNet, EC²-VAE, CMT, ComMU, Polyffusion) присутствуют как заглушки, возвращающие `{"error": "not implemented"}`.

После этого раунда добавление каждой следующей модели сводится к написанию её adapter'а (в pipeline-venv) и runner'а (в model-venv) — без правки общего кода.

---

## 2. Базовые архитектурные решения

| # | Решение | Значение |
|---|---|---|
| 1 | Scope текущего раунда | каркас + MINGUS, 5 stub-моделей |
| 2 | Окружение pipeline | свой `pipeline/.venv/` (минимум: pretty_midi, music21, numpy, pydantic) |
| 3 | Запуск моделей | subprocess в их собственных venv через `<model>/.venv/bin/python` |
| 4 | Обмен между pipeline и runner | JSON через stdin (короткая полезная нагрузка) + файлы для MIDI/XML на диске (`pipeline/output/_tmp/<run_id>/`) |
| 5 | Хранение промежуточных файлов | `pipeline/output/_tmp/<run_id>/` (видимо, не чистится автоматически) |
| 6 | Формат `run_id` | `YYYYMMDD-HHMMSS-<8charhash>`, hash детерминирован от прогрессии |
| 7 | Chord track в `with_chords.mid` | pipeline всегда рендерит **свой** chord track из `ChordProgression` (унификация для всех 6 моделей); chord track модели игнорируется |
| 8 | Длина прогрессии | 8 баров в 4/4 (32 beat'а) |

---

## 3. Слоистая архитектура

Главный принцип: **runner — чистая обёртка над моделью, не знает про наш пайплайн**. Pipeline-специфика (стратегии затравки, run_id, постпроцесс) живёт на стороне adapter'а в pipeline-venv.

```
┌─────────────────────────────────────────────────────────┐
│ Слой 1 — Pipeline (pipeline-venv)                       │
│                                                         │
│  pipeline/pipeline.py  generate_all(progression)        │
│  pipeline/adapters/<model>.py  prepare(progression, …)  │
│  pipeline/postprocess.py  raw → 2 нормализованных MIDI  │
│  pipeline/chord_render.py  ChordProgression → chord MIDI│
└────────────────────────────┬────────────────────────────┘
                             │ subprocess + JSON via stdin
                             │ + files on disk in _tmp/<run_id>/
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Слой 2 — Runner (model-venv)                            │
│                                                         │
│  pipeline/runners/<model>_runner.py                     │
│   - принимает только то что модель API реально требует  │
│   - вызывает inference                                  │
│   - пишет MIDI в указанный output_midi_path             │
│   - exit 0 при успехе, ≠0 при ошибке                    │
└─────────────────────────────────────────────────────────┘
```

### Граница «runner или adapter»

Правило: **«это факт API модели или наше pipeline-решение?»**

- **Факт API модели** (`temperature`, `checkpoint_path`, `device`, формат входного XML) → runner.
- **Наше pipeline-решение** (`seed_strategy`, выбор chord_render для `with_chords`, схема `run_id`) → adapter / общая часть pipeline.
- **Деривативы от ChordProgression** (`tempo`, `num_bars`, harmony tags) → закодировать в `input.xml`. Через JSON-границу не передавать — модель прочтёт из XML сама.

### Затравка (seed / prime) — per-model, не общая

`ChordProgression` содержит только **композиционный замысел**: аккорды + tempo + размер. Никакого `seed` поля.

Затравочный prefix («сыграй эти ноты, дальше продолжай сам») каждая авторегрессионная модель ждёт в своём формате:
- BebopNet/MINGUS — мелодия в XML;
- CMT — `prime_pitch[16]` + `prime_rhythm[16]` индексы из vocab;
- ComMU — последовательность REMI-токенов;
- EC²-VAE/Polyffusion — затравка не нужна, латент сэмплируется из распределения.

Это значит: затравка — **implementation detail модели**, она не имеет универсального представления. Поэтому затравка живёт в per-model `<Model>PipelineConfig` (например, `MingusPipelineConfig.seed_strategy`), не в `ChordProgression`. Каждый adapter сам знает как из аккордов и своего конфига собрать вход.

### Инвариант runner'а

Runner-скрипты запускаются интерпретатором model-venv (`models/<name>/.venv/bin/python`). В этом venv пакет `pipeline.pipeline.*` **не установлен**.

Поэтому runner-скрипт:
- импортирует **только** stdlib (`json`, `sys`, `pathlib`, `argparse`, `subprocess`) и пакеты model-venv'а (`torch`, `music21`, …, что уже стоит в этом venv);
- **не импортирует** ничего из `pipeline.*` (`pipeline.adapters`, `pipeline.progression`, `pipeline.chord_vocab`).

Если runner'у нужны вспомогательные функции (например, `parse_chord`) — он либо получает уже разобранные данные через JSON, либо имеет собственную копию утилиты в своём файле. Дублирование 5 строк парсинга лучше зависимости через границу venv.

### Adapter — toolbox для модели

Каждый adapter (`pipeline/adapters/<model>.py`) реализует общий интерфейс `ModelAdapter` с двумя методами:

- `prepare(progression, config, tmp_dir) → params` — граница «вниз»: из ChordProgression и per-model конфига собрать вход для модели (XML-файл / numpy / токены / …) и вернуть `params` для runner'а.
- `extract_melody(raw_midi_path) → pretty_midi.Instrument` — граница «вверх»: из сырого MIDI который вернула модель достать монофонную мелодию-инструмент.

Postprocess принимает уже извлечённый `Instrument` и не знает про конкретную модель. Adapter — единственное место где живёт модель-специфичное знание про формат входа и формат выхода.

### Stub-модели

5 моделей кроме MINGUS присутствуют только как stub-adapter'ы. Оба метода (`prepare`, `extract_melody`) немедленно бросают `NotImplementedError("model <name>: adapter not implemented")`.

`pipeline.generate_all` ловит `NotImplementedError` **до** запуска subprocess'а (т.е. при вызове `prepare`) и кладёт `{"error": "not implemented (stub)"}`. Runner-скриптов для stub-моделей нет — они появляются только когда реализуется соответствующая модель.

---

## 4. Структура файлов

```
pipeline/
├── .venv/                        # изолированный venv пайплайна
├── requirements.txt              # pretty_midi, music21, numpy, pydantic
├── pipeline/                     # python-пакет
│   ├── __init__.py
│   ├── progression.py            # dataclass ChordProgression + JSON load/save
│   ├── chord_vocab.py            # 12 roots × 7 qualities + lossy fallback'и
│   ├── chord_render.py           # ChordProgression → MIDI piano chord track
│   ├── postprocess.py            # raw_midi + progression → (melody_only, with_chords)
│   ├── runner_protocol.py        # JSON-схема входа/выхода runner'а (общая)
│   ├── pipeline.py               # generate_all(progression) → dict
│   ├── cli.py                    # python -m pipeline.cli generate <progression.json>
│   ├── config.py                 # пути venv'ов, чекпоинтов, дефолты на модель
│   └── adapters/
│       ├── __init__.py           # ADAPTERS = {"mingus": MingusAdapter, ...}
│       ├── base.py               # ABC ModelAdapter (интерфейс prepare + extract_melody)
│       ├── mingus.py             # MingusAdapter: prepare() + extract_melody()
│       ├── bebopnet.py           # stub: ABC методы raises NotImplementedError
│       ├── ec2vae.py             # stub
│       ├── cmt.py                # stub
│       ├── commu.py              # stub
│       └── polyffusion.py        # stub
├── runners/                      # запускаются в model-venv'ах
│   └── mingus_runner.py          # для остальных моделей runner появляется
│                                 # вместе с реализацией adapter'а (не сейчас)
├── test_progressions/
│   └── sample.json               # Cmaj7-Am7-Dm7-G7 ×2 (8 bars)
├── convert_to_mp3.sh             # уже существует; не трогаем
└── output/
    ├── _tmp/<run_id>/            # input.xml, raw.mid, stderr.log
    ├── melody_only/<model>_<run_id>.mid
    ├── with_chords/<model>_<run_id>.mid
    └── mp3/<model>_<run_id>.mp3
```

---

## 5. Контракт runner'а

Pipeline вызывает runner так:

```bash
<model>/.venv/bin/python pipeline/runners/<model>_runner.py
# JSON-payload подаётся в stdin
```

### Вход (stdin) — единый шаблон, специфика в `params`

```json
{
    "model": "mingus",
    "run_id": "20260427-153012-a1b2c3d4",
    "params": { ...model-specific... }
}
```

### Выход

| Случай | Поведение |
|---|---|
| Успех | runner пишет MIDI в `params.output_midi_path` и завершается с `exit 0`. stdout/stderr могут содержать что угодно (логи модели) — pipeline их сохраняет, но не парсит |
| Ошибка | runner завершается с `exit ≠ 0`. Текст ошибки ожидается в stderr (traceback или человеческое сообщение) |

### Как pipeline обрабатывает результат subprocess

```python
result = subprocess.run(
    [model_venv_python, runner_path],
    input=json.dumps(payload),
    capture_output=True,
    text=True,
    timeout=RUNNER_TIMEOUT_SEC,  # из config.py, дефолт 600
)
(tmp_dir / "stdout.log").write_text(result.stdout)
(tmp_dir / "stderr.log").write_text(result.stderr)

if result.returncode != 0:
    tail = "\n".join(result.stderr.strip().splitlines()[-20:])
    raise RunnerError(f"{model} runner exited with {result.returncode}:\n{tail}")
if not Path(params["output_midi_path"]).exists():
    raise RunnerError(f"{model} runner exited 0 but output MIDI not found")
return Path(params["output_midi_path"])
```

`stdout.log` и `stderr.log` всегда сохраняются (для дебага), даже при успехе.

### `params` для MINGUS

```json
{
    "input_xml_path":  "/abs/.../output/_tmp/20260427-…/input.xml",
    "output_midi_path":"/abs/.../output/_tmp/20260427-…/raw.mid",
    "checkpoint_path": "/abs/.../models/MINGUS/checkpoints/mingus.pt",
    "temperature":     1.0,
    "device":          "cpu"
}
```

`input_xml_path` и `output_midi_path` — служебные поля pipeline-протокола (мы решаем где лежат файлы, MINGUS просто читает/пишет туда). Остальные три — реально MINGUS API: `temperature` управляет sampling, `device` — где считать, `checkpoint_path` — какие веса грузить.

`mingus_runner.py` делает:
1. Парсит JSON со stdin.
2. Добавляет `models/MINGUS` в `sys.path`.
3. Вызывает существующий MINGUS inference — либо через import (`from C_generate.generate import generate_xml_to_midi`), либо через `subprocess.run` MINGUS CLI (`generate.py --xmlSTANDARD ...`). Конкретный путь — implementation detail; runner отвечает только за корректное преобразование `params → MINGUS API`.
4. Пишет MIDI в `output_midi_path`.

---

## 6. Adapter MINGUS

Живёт в `pipeline/adapters/mingus.py` (в pipeline-venv).

### Базовый интерфейс

```python
# pipeline/adapters/base.py
class ModelAdapter(ABC):
    @abstractmethod
    def prepare(self, progression: ChordProgression,
                config: Any, tmp_dir: Path) -> dict:
        """Из progression и конфига собирает params для runner'а
        (включая физическую подготовку файлов в tmp_dir, если нужно)."""

    @abstractmethod
    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        """Из сырого выхода модели возвращает монофонную мелодию как pretty_midi.Instrument.

        Контракт: возвращаемый Instrument МОЖЕТ быть сконструирован adapter'ом из чего
        угодно — track'а MIDI (MINGUS, BebopNet), highest-pitch проекции полифонии
        (Polyffusion), декодированного pianoroll (EC²-VAE), декодированных REMI-токенов
        (ComMU). Pipeline не предполагает, что у модели «есть готовый melody-track»;
        adapter сам отвечает за получение монофонной мелодии в нужном формате."""
```

### Конфигурация MINGUS

```python
@dataclass
class MingusPipelineConfig:
    """Все настройки MINGUS на уровне нашего пайплайна.
    Часть из них — стратегии подготовки входа (живут только здесь),
    часть — параметры MINGUS-API, которые мы пробрасываем в runner."""

    seed_strategy: Literal["tonic_whole", "tonic_quarters", "custom_xml"]
    # tonic_whole       — 8 whole-нот тоники каждого аккорда (дефолт)
    # tonic_quarters    — 32 четверти, тоника каждого бара
    # custom_xml        — использовать готовый XML из custom_xml_path

    custom_xml_path: Path | None = None
    temperature: float = 1.0
    device: str = "cpu"
    checkpoint_path: Path = ...  # из pipeline/config.py

    # Какой инструмент в raw MIDI считать мелодией:
    melody_instrument_name: str = "Tenor Sax"
```

### MingusAdapter

```python
class MingusAdapter(ModelAdapter):
    def prepare(self, progression, config: MingusPipelineConfig, tmp_dir):
        """
        1. Строит input.xml согласно config.seed_strategy:
           - tonic_whole / tonic_quarters — генерим XML
             с harmony tags из progression и затравочной мелодией
           - custom_xml — копируем config.custom_xml_path в tmp_dir/input.xml
        2. Возвращает params для runner'а:
           {
             input_xml_path: tmp_dir/input.xml,
             output_midi_path: tmp_dir/raw.mid,
             checkpoint_path: config.checkpoint_path,
             temperature: config.temperature,
             device: config.device,
           }
        """

    def extract_melody(self, raw_midi_path):
        """
        MINGUS возвращает MIDI с двумя треками: Tenor Sax (мелодия) + piano (chord
        accompaniment, нам не нужен — мы строим свой через chord_render).

        1. Открывает raw_midi_path через pretty_midi.
        2. Находит инструмент с name == config.melody_instrument_name.
        3. Возвращает его как pretty_midi.Instrument (не пишет файл — это работа postprocess).
        """
```

### Что adapter НЕ делает

- Не вызывает MINGUS (не запускает subprocess) — это работа `pipeline.py`.
- Не пишет финальные MIDI файлы (`melody_only`, `with_chords`) — это работа `postprocess.py`.
- Не знает про `run_id` иначе чем как часть `tmp_dir`.

---

## 7. ChordProgression и связанные модули

### `progression.py`

```python
@dataclass
class ChordProgression:
    """Композиционный замысел: только аккорды + tempo + размер.
    Никакой модель-специфики (затравки, температуры, чекпоинтов) здесь нет."""

    chords: list[tuple[str, int]]    # [("Cmaj7", 4), ("Am7", 4), ...]
    tempo: float = 120.0
    time_signature: str = "4/4"

    @classmethod
    def from_json(cls, path: Path) -> "ChordProgression": ...
    def to_json(self, path: Path) -> None: ...
    def total_beats(self) -> int: return sum(d for _, d in self.chords)
    def num_bars(self) -> int: return self.total_beats() // beats_per_bar(self.time_signature)
```

### `chord_vocab.py`

```python
ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["maj", "min", "7", "maj7", "min7", "dim", "dim7"]

def parse_chord(chord_str: str) -> tuple[int, str]:  # ("Cmaj7" → (0, "maj7"))
def chord_to_pitches(chord_str: str) -> list[int]:    # ("Cmaj7" → [60, 64, 67, 71])
QUALITY_FALLBACK_TO_TRIADS = {"7": "maj", "maj7": "maj", "min7": "min", "dim7": "dim"}
EXTENDED_FALLBACK_TO_VOCAB = {"m7b5": "dim", "alt": "7", "6": "maj", "9": "7", "13": "7"}
```

### `chord_render.py`

```python
def render_chord_track(progression: ChordProgression, out_path: Path) -> None:
    """Сохраняет MIDI с одним piano-треком: на каждый аккорд — 4-голосный voicing
    длительностью равной duration_in_beats. Используется для with_chords.mid у всех моделей."""
```

### `postprocess.py`

```python
def postprocess(melody: pretty_midi.Instrument,
                progression: ChordProgression,
                model_name: str,
                run_id: str,
                output_root: Path) -> dict[str, Path]:
    """
    Принимает УЖЕ извлечённую мелодию (Instrument) — извлечение из raw MIDI
    делает adapter.extract_melody, postprocess про модели не знает.

    1. Нормализует мелодию: program ← config.MELODY_PROGRAM, name ← 'Melody'.
    2. Сохраняет монофонную мелодию → melody_only/<model>_<run_id>.mid.
    3. Через chord_render.render_chord_track строит наш piano chord track из progression.
    4. Склеивает melody + наш chord track → with_chords/<model>_<run_id>.mid.
    5. Возвращает {"melody_only": ..., "with_chords": ...}.
    """
```

`postprocess.py` не импортирует ничего из `pipeline.adapters` и не содержит ни одного `if model_name == ...`. Логика на этом уровне действительно общая.

---

## 8. Главная функция `generate_all`

```python
# pipeline/pipeline.py

MODEL_NAMES = ["mingus", "bebopnet", "ec2vae", "cmt", "commu", "polyffusion"]

def generate_all(progression: ChordProgression,
                 run_id: str | None = None) -> dict[str, dict]:
    run_id = run_id or make_run_id(progression)
    tmp_root = OUTPUT_ROOT / "_tmp" / run_id
    tmp_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for model in MODEL_NAMES:
        adapter = ADAPTERS[model]
        cfg = MODEL_CONFIGS[model]
        model_tmp = tmp_root / model
        model_tmp.mkdir(exist_ok=True)
        try:
            params = adapter.prepare(progression, cfg, model_tmp)
            raw_midi = run_model_subprocess(model, params, run_id, model_tmp)
            melody = adapter.extract_melody(raw_midi)
            results[model] = postprocess(melody, progression, model, run_id, OUTPUT_ROOT)
        except NotImplementedError:
            results[model] = {"error": "not implemented (stub)"}
        except RunnerError as e:
            results[model] = {"error": str(e)}
    return results


def run_model_subprocess(model: str, params: dict, run_id: str, tmp: Path) -> Path:
    """Запускает <model>_runner.py в model-venv через subprocess.
    Передаёт {"model": model, "run_id": run_id, "params": params} через stdin.
    Ловит exit code; при ошибке поднимает RunnerError со stderr.
    Возвращает params["output_midi_path"] при успехе.
    """
```

---

## 9. CLI

```
python -m pipeline.cli generate test_progressions/sample.json
```

Печатает таблицу:

```
model         status   melody_only                            with_chords
mingus        ok       output/melody_only/mingus_<rid>.mid    output/with_chords/mingus_<rid>.mid
bebopnet      error    not implemented (stub)
ec2vae        error    not implemented (stub)
cmt           error    not implemented (stub)
commu         error    not implemented (stub)
polyffusion   error    not implemented (stub)
```

---

## 10. Конфигурация

```python
# pipeline/config.py

DIPLOMA_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT  = Path(__file__).resolve().parent.parent / "output"

MODEL_VENV_PYTHON = {
    "mingus":      DIPLOMA_ROOT / "models/MINGUS/.venv/bin/python",
    "bebopnet":    DIPLOMA_ROOT / "models/bebopnet-code/.venv/bin/python",
    "ec2vae":      DIPLOMA_ROOT / "models/EC2-VAE/.venv/bin/python",
    "cmt":         DIPLOMA_ROOT / "models/CMT-pytorch/.venv/bin/python",
    "commu":       DIPLOMA_ROOT / "models/ComMU-code/.venv/bin/python",
    "polyffusion": DIPLOMA_ROOT / "models/polyffusion/.venv/bin/python",
}

MODEL_REPO = { ...пути к коду каждой модели... }

MODEL_CONFIGS = {
    "mingus": MingusPipelineConfig(
        seed_strategy="tonic_whole",
        temperature=1.0,
        device="cpu",
        checkpoint_path=DIPLOMA_ROOT / "models/MINGUS/checkpoints/mingus.pt",
    ),
    # остальные — None или stub-конфиги
}

# Общее pipeline-решение: монофонная мелодия в melody_only.mid пишется единым
# тембром у всех моделей — тогда на слух (и в feature-расчёте метрик)
# тембр не вмешивается в сравнение. 66 = Tenor Sax (GM), исторически бэйслайны
# (BebopNet/MINGUS) как раз и пишут саксофоном.
MELODY_PROGRAM: int = 66

```

(Селектор «какой трек считать мелодией» больше не живёт в общем `config.py` — он переехал
в `MingusPipelineConfig.melody_instrument_name` и аналогичные поля per-model конфигов
других моделей, потому что это часть знания adapter'а о модели.)

---

## 11. Тестовая прогрессия (`test_progressions/sample.json`)

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

---

## 12. Definition of Done текущего раунда

- ✅ `pipeline/.venv` создан, `requirements.txt` зафиксирован.
- ✅ Все модули из раздела 4 существуют (stub-adapter'ы для 5 моделей; runner только для MINGUS).
- ✅ `python -m pipeline.cli generate test_progressions/sample.json` отрабатывает без необработанных исключений.
- ✅ MINGUS возвращает `{"melody_only": …, "with_chords": …}`; оба файла существуют, открываются `pretty_midi.PrettyMIDI(...)` без ошибок, `melody_only.mid` монофонный, `with_chords.mid` содержит melody + аккомпанемент из `chord_render`.
- ✅ 5 stub-моделей возвращают `{"error": "not implemented (stub)"}` — без падения пайплайна.
- ✅ `bash pipeline/convert_to_mp3.sh` рендерит MP3 для MINGUS (существующий скрипт, не правим).

---

## 13. Что отложено за пределы текущего раунда

- ❌ Реализация adapter'ов и runner'ов для BebopNet, EC²-VAE, CMT, ComMU, Polyffusion — только stubs.
- ❌ Метрики (FMD, Chord-Tone Ratio, Scale Match, …).
- ❌ Множественные samples per progression.
- ❌ `generate_long(progression, bars=32)` со склейкой 8-баровых сегментов.
- ❌ Расширение chord vocabulary (m7b5, alt, 6/9/13, sus4) — только 7 базовых qualities.
- ❌ HTTP/gRPC обёртка вокруг runner'ов — пока только subprocess.
- ❌ Параллельный запуск моделей — последовательно в цикле.

Каждый пункт превращается в отдельный design + plan, когда до него дойдём.

---

## 14. Открытые риски

| # | Риск | Митигация |
|---|---|---|
| 1 | MINGUS требует на вход не только аккорды, но и затравочную мелодию; неизвестно как поведёт себя на минимальной (whole-note тоника) затравке | Реализуем 3 стратегии (`tonic_whole`/`tonic_quarters`/`custom_xml`); если дефолт даёт мусор — переключаемся параметром в `MODEL_CONFIGS` без правки кода |
| 2 | MINGUS inference может не уметь принять произвольный XML (внутренний код мог быть заточен под `Donna_Lee_short.xml`) | Runner вначале проверяется на дефолтном MINGUS XML (smoke); только потом — на нашем сгенерированном |
| 3 | `pretty_midi`/`music21` могут не открыть MIDI корректно если MINGUS пишет нестандартный формат | Postprocess обёрнут в try/except, в случае фейла — `{"error": ...}` с детализацией |
| 4 | Subprocess overhead на каждый запуск (~1-2 с импорт torch) | Допустимо для текущего scope — пайплайн не запускается в hot loop. Если станет проблемой — ввести long-running runner с pipe (отдельный design) |

---

## 15. Что делать дальше — следующие модели

1. Добавить **BebopNet** — adapter (`prepare`: XML с минимальным seed без головы; `extract_melody`: единственный инструмент raw MIDI) + runner (вызов их `generate_from_xml.py`). Generic-инфраструктура должна позволить обойтись без правки `pipeline.py` / `postprocess.py` / `chord_render.py`.
2. То же для **CMT** (после Colab-train), **EC²-VAE** (после Colab-train), **ComMU** (после fine-tune), **Polyffusion** (после fine-tune).
3. Параллельный отдельный design для **пайплайна метрик** (FMD, OA, KL, и т.д.).
