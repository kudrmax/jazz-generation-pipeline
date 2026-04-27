# Pipeline Robustness Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Закрыть Important + Minor issues, найденные финальным code review ветки `feat/pipeline-mingus`. Все правки сохраняют 73 теста зелёными и не меняют поведение для пользователя CLI.

**Architecture:** Не меняется — это серия точечных правок поверх существующего пайплайна. Все коммиты ложатся на текущую ветку `feat/pipeline-mingus`.

**Tech Stack:** Python 3.12, pytest. Без новых зависимостей.

**Источник:** Финальный code review (см. transcript `feat/pipeline-mingus`). 

---

## Pre-conditions

- Branch `feat/pipeline-mingus`, HEAD = `c859b42`.
- `cd /Users/maxos/PythonProjects/diploma/pipeline && .venv/bin/python -m pytest -v` → 73 passed.

После каждой задачи `pytest -v` должен оставаться 73+ passed (часть задач добавляет тесты).

---

## Task 1: MingusAdapter owns config via __init__ (Important — I1+I2+I3)

**Что:** Убрать двойное хранение конфига в `MingusAdapter`. Сейчас config принимается и в `__init__`, и в `prepare()` (где переписывает первый). После правки — config живёт только в `__init__`, метод `prepare()` его не принимает. Поле `MODEL_CONFIGS` в `config.py` становится не нужным и удаляется.

**Files:**
- Modify: `pipeline/pipeline/adapters/base.py` — ABC `prepare` signature
- Modify: `pipeline/pipeline/adapters/mingus.py` — `__init__` обязателен, `prepare` без config
- Modify: `pipeline/pipeline/adapters/{bebopnet,ec2vae,cmt,commu,polyffusion}.py` — 5 stubs, новая сигнатура prepare
- Modify: `pipeline/pipeline/config.py` — `ADAPTERS["mingus"]` создаётся с реальным конфигом, `MODEL_CONFIGS` удаляется
- Modify: `pipeline/pipeline/pipeline.py` — `_run_model_subprocess` и `generate_all` без `cfg`
- Modify: `pipeline/tests/adapters/test_base.py` — обновить вызовы prepare без config
- Modify: `pipeline/tests/adapters/test_mingus_prepare.py` — `MingusAdapter(cfg).prepare(prog, tmp)`
- Modify: `pipeline/tests/adapters/test_mingus_extract_melody.py` — конструктор требует config

- [ ] **Step 1: Failing test — fix test_mingus_prepare to new signature**

В `pipeline/tests/adapters/test_mingus_prepare.py` — все 5 тестов сейчас вызывают `MingusAdapter().prepare(progression, cfg, tmp_path)`. Перепиши на новую форму:

```python
def test_prepare_returns_required_keys(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    for key in ["input_xml_path", "output_midi_path", "checkpoint_epochs", "temperature", "device", "model_repo_path"]:
        assert key in params, f"missing key: {key}"


def test_prepare_writes_input_xml(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    assert Path(params["input_xml_path"]).exists()
    assert Path(params["input_xml_path"]).suffix == ".xml"


def test_prepare_output_midi_path_in_tmp_dir(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    assert str(params["output_midi_path"]).startswith(str(tmp_path))
    assert params["output_midi_path"].endswith(".mid")


def test_prepare_passes_through_temperature_and_device(tmp_path: Path):
    cfg = MingusPipelineConfig(
        seed_strategy="tonic_whole", checkpoint_epochs=100,
        temperature=0.7, device="cpu",
    )
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    assert params["temperature"] == 0.7
    assert params["device"] == "cpu"
    assert params["checkpoint_epochs"] == 100


def test_prepare_does_not_leak_pipeline_concepts(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_quarters", checkpoint_epochs=100)
    params = MingusAdapter(cfg).prepare(_basic_progression(), tmp_path)
    forbidden = {"seed_strategy", "run_id", "model_name", "progression"}
    leaked = forbidden & params.keys()
    assert not leaked, f"adapter leaked pipeline concepts to runner params: {leaked}"


def test_mingus_adapter_requires_config():
    """После рефакторинга MingusAdapter() без аргументов должен падать."""
    with pytest.raises(TypeError):
        MingusAdapter()  # type: ignore[call-arg]
```

- [ ] **Step 2: Run tests — should fail (signature mismatch)**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest tests/adapters/test_mingus_prepare.py -v
```

Expected: failures because `MingusAdapter()` doesn't yet require config and `prepare` still takes 3 args.

- [ ] **Step 3: Update ABC base.py**

В `pipeline/pipeline/adapters/base.py`:

```python
class ModelAdapter(ABC):
    """Toolbox для одной модели — обе границы между pipeline и моделью."""

    @abstractmethod
    def prepare(
        self,
        progression: ChordProgression,
        tmp_dir: Path,
    ) -> dict:
        """Из progression собирает params для runner'а
        (включая физическую подготовку входных файлов в tmp_dir, если нужно).

        Конфигурация модели хранится в state экземпляра adapter'а (из __init__),
        а не передаётся в prepare. Это позволяет config быть immutable после
        инициализации.

        Возвращает словарь, который pipeline пробрасывает runner'у через JSON stdin.
        """

    @abstractmethod
    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        """[unchanged docstring]"""
```

- [ ] **Step 4: Update 5 stubs**

Каждый stub (`bebopnet.py`, `ec2vae.py`, `cmt.py`, `commu.py`, `polyffusion.py`) — заменить `prepare(self, progression, config, tmp_dir)` на `prepare(self, progression, tmp_dir)`. Шаблон для bebopnet.py:

```python
class BebopNetAdapter(ModelAdapter):
    def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
        raise NotImplementedError("model bebopnet: adapter not implemented")

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        raise NotImplementedError("model bebopnet: adapter not implemented")
```

Аналогично для всех 5. Импорт `Any` больше не нужен — удалить из stubs.

- [ ] **Step 5: Update MingusAdapter (mingus.py)**

```python
class MingusAdapter(ModelAdapter):
    def __init__(self, config: MingusPipelineConfig) -> None:
        # config теперь обязателен — нет default fallback
        self._config = config

    def prepare(
        self,
        progression: ChordProgression,
        tmp_dir: Path,
    ) -> dict:
        from pipeline.config import MINGUS_REPO_PATH
        from pipeline._xml_builders.mingus_xml import build_mingus_xml

        cfg = self._config
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        xml_path = tmp_dir / "input.xml"
        midi_path = tmp_dir / "raw.mid"
        build_mingus_xml(progression, cfg, xml_path)
        return {
            "input_xml_path": str(xml_path),
            "output_midi_path": str(midi_path),
            "checkpoint_epochs": cfg.checkpoint_epochs,
            "temperature": cfg.temperature,
            "device": cfg.device,
            "model_repo_path": str(MINGUS_REPO_PATH),
        }

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

- [ ] **Step 6: Update config.py**

В `pipeline/pipeline/config.py`:
- Удалить весь блок `MODEL_CONFIGS = {...}`
- В `ADAPTERS["mingus"]` передать конфиг напрямую:

```python
ADAPTERS: dict[str, ModelAdapter] = {
    "mingus":      MingusAdapter(MingusPipelineConfig(
        seed_strategy="tonic_whole",
        temperature=1.0,
        device="cpu",
        checkpoint_epochs=100,
    )),
    "bebopnet":    BebopNetAdapter(),
    "ec2vae":      EC2VaeAdapter(),
    "cmt":         CMTAdapter(),
    "commu":       ComMUAdapter(),
    "polyffusion": PolyffusionAdapter(),
}
```

- [ ] **Step 7: Update pipeline.py**

В `pipeline/pipeline/pipeline.py`:
- Удалить импорт `MODEL_CONFIGS` из `from pipeline.config import ...`
- В `generate_all` убрать строку `cfg = MODEL_CONFIGS[model]`
- Заменить `params = adapter.prepare(progression, cfg, model_tmp)` на `params = adapter.prepare(progression, model_tmp)`

Полный body цикла:

```python
    for model in MODEL_NAMES:
        adapter = ADAPTERS[model]
        model_tmp = tmp_root / model
        model_tmp.mkdir(exist_ok=True)
        try:
            params = adapter.prepare(progression, model_tmp)
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

- [ ] **Step 8: Update test_base.py**

В `pipeline/tests/adapters/test_base.py`, оба parametrized теста: убрать `config=None` аргумент:

```python
@pytest.mark.parametrize("AdapterCls", [
    BebopNetAdapter, EC2VaeAdapter, CMTAdapter, ComMUAdapter, PolyffusionAdapter,
])
def test_stub_adapter_prepare_raises_not_implemented(tmp_path: Path, AdapterCls):
    adapter = AdapterCls()
    progression = ChordProgression(chords=[("Cmaj7", 4)], tempo=120.0, time_signature="4/4")
    with pytest.raises(NotImplementedError, match="adapter not implemented"):
        adapter.prepare(progression, tmp_path)


@pytest.mark.parametrize("AdapterCls", [
    BebopNetAdapter, EC2VaeAdapter, CMTAdapter, ComMUAdapter, PolyffusionAdapter,
])
def test_stub_adapter_extract_melody_raises_not_implemented(tmp_path: Path, AdapterCls):
    adapter = AdapterCls()
    fake_midi = tmp_path / "x.mid"
    with pytest.raises(NotImplementedError, match="adapter not implemented"):
        adapter.extract_melody(fake_midi)
```

`test_model_adapter_is_abstract` остаётся как был.

- [ ] **Step 9: Update test_mingus_extract_melody.py**

Сейчас тесты используют `MingusAdapter()` (без аргументов). После рефакторинга это TypeError. Перепиши на:

```python
def test_extract_melody_returns_tenor_sax_track(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi)
    cfg = MingusPipelineConfig()
    melody = MingusAdapter(cfg).extract_melody(midi)
    assert melody.name == "Tenor Sax"
    assert len(melody.notes) == 4


def test_extract_melody_returns_instrument_type(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi)
    cfg = MingusPipelineConfig()
    melody = MingusAdapter(cfg).extract_melody(midi)
    assert isinstance(melody, pretty_midi.Instrument)


def test_extract_melody_raises_when_track_missing(tmp_path: Path):
    midi = tmp_path / "raw.mid"
    _make_two_track_midi(midi, melody_name="Wrong Name")
    cfg = MingusPipelineConfig()
    with pytest.raises(ValueError, match="Tenor Sax"):
        MingusAdapter(cfg).extract_melody(midi)
```

Импорт `MingusPipelineConfig` уже есть в файле (или добавить).

- [ ] **Step 10: Run all tests**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest -v
```

Expected: 74 passed (73 prior + 1 new `test_mingus_adapter_requires_config`).

- [ ] **Step 11: Run end-to-end smoke**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
```

Expected: same output as before refactor — mingus=ok, others=stub error. Refactor не меняет поведение пользователя.

- [ ] **Step 12: Commit**

```bash
cd /Users/maxos/PythonProjects/diploma
git add pipeline/pipeline/adapters pipeline/pipeline/config.py pipeline/pipeline/pipeline.py pipeline/tests/adapters
git commit -m "refactor(pipeline): MingusAdapter owns config via __init__, drop MODEL_CONFIGS"
```

---

## Task 2: Validate tempo > 0 in ChordProgression (M1)

**Files:**
- Modify: `pipeline/pipeline/progression.py` — add `__post_init__` with validation
- Modify: `pipeline/tests/test_progression.py` — add 2 tests

- [ ] **Step 1: Failing tests**

В конец `pipeline/tests/test_progression.py`:

```python
def test_progression_rejects_zero_tempo():
    with pytest.raises(ValueError, match="tempo"):
        ChordProgression(chords=[("Cmaj7", 4)], tempo=0.0, time_signature="4/4")


def test_progression_rejects_negative_tempo():
    with pytest.raises(ValueError, match="tempo"):
        ChordProgression(chords=[("Cmaj7", 4)], tempo=-120.0, time_signature="4/4")
```

- [ ] **Step 2: Run — both fail**

```bash
.venv/bin/python -m pytest tests/test_progression.py::test_progression_rejects_zero_tempo tests/test_progression.py::test_progression_rejects_negative_tempo -v
```

- [ ] **Step 3: Add `__post_init__`**

В `pipeline/pipeline/progression.py`, добавить метод после `time_signature`:

```python
    def __post_init__(self) -> None:
        if self.tempo <= 0:
            raise ValueError(f"tempo must be > 0, got {self.tempo}")
```

- [ ] **Step 4: Tests pass**

```bash
.venv/bin/python -m pytest tests/test_progression.py -v
```

Expected: 7 passed (5 original + 2 new).

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/progression.py pipeline/tests/test_progression.py
git commit -m "fix(pipeline): validate tempo > 0 in ChordProgression"
```

---

## Task 3: Wrap subprocess.TimeoutExpired in RunnerError (M7)

**Files:**
- Modify: `pipeline/pipeline/runner_protocol.py`
- Modify: `pipeline/tests/test_runner_protocol.py`

- [ ] **Step 1: Failing test**

В `pipeline/tests/test_runner_protocol.py`, добавить:

```python
def test_run_runner_raises_runner_error_on_timeout(tmp_path: Path):
    runner = tmp_path / "runner.py"
    _write_runner(runner, (
        'import sys, time\n'
        'time.sleep(5)\n'
        'sys.exit(0)\n'
    ))
    with pytest.raises(RunnerError, match="timed out"):
        run_runner_subprocess(
            venv_python=sys.executable,
            runner_script=runner,
            payload={"params": {"output_midi_path": str(tmp_path / "x.mid")}},
            tmp_dir=tmp_path,
            timeout_sec=1,  # короткий таймаут — sleep 5 не успеет
        )
```

- [ ] **Step 2: Test fails (TimeoutExpired не оборачивается)**

```bash
.venv/bin/python -m pytest tests/test_runner_protocol.py::test_run_runner_raises_runner_error_on_timeout -v
```

- [ ] **Step 3: Wrap TimeoutExpired**

В `pipeline/pipeline/runner_protocol.py`, в `run_runner_subprocess`, обернуть `subprocess.run`:

```python
    try:
        result = subprocess.run(
            [str(venv_python), str(runner_script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        # сохраняем что успело прийти
        (tmp_dir / "stdout.log").write_text(e.stdout or "")
        (tmp_dir / "stderr.log").write_text(e.stderr or "")
        raise RunnerError(
            f"runner {runner_script} timed out after {timeout_sec}s"
        ) from e
```

- [ ] **Step 4: Test passes**

```bash
.venv/bin/python -m pytest tests/test_runner_protocol.py -v
```

Expected: 5 passed (4 original + 1 new).

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/runner_protocol.py pipeline/tests/test_runner_protocol.py
git commit -m "fix(pipeline): wrap subprocess.TimeoutExpired in RunnerError"
```

---

## Task 4: Guard MODEL_RUNNER_SCRIPT lookup (M5)

**Files:**
- Modify: `pipeline/pipeline/pipeline.py` — `_run_model_subprocess` проверяет наличие ключа
- Modify: `pipeline/tests/test_pipeline.py` — добавить 1 тест

- [ ] **Step 1: Failing test**

В `pipeline/tests/test_pipeline.py`, добавить:

```python
def test_run_model_subprocess_raises_runner_error_when_runner_script_missing(tmp_path: Path, monkeypatch):
    """Если в MODEL_RUNNER_SCRIPT нет ключа — должна быть понятная RunnerError, не KeyError."""
    from pipeline.pipeline import _run_model_subprocess
    from pipeline.runner_protocol import RunnerError

    monkeypatch.setattr("pipeline.pipeline.MODEL_RUNNER_SCRIPT", {})  # пустой dict
    with pytest.raises(RunnerError, match="runner script not registered"):
        _run_model_subprocess(
            "bebopnet", {"output_midi_path": str(tmp_path / "x.mid")}, "rid", tmp_path,
        )
```

- [ ] **Step 2: Test fails (KeyError, not RunnerError)**

- [ ] **Step 3: Add guard**

В `pipeline/pipeline/pipeline.py`, в `_run_model_subprocess`, добавить guard в начале:

```python
def _run_model_subprocess(
    model: str, params: dict, run_id: str, model_tmp: Path,
) -> Path:
    if model not in MODEL_RUNNER_SCRIPT:
        raise RunnerError(
            f"runner script not registered for model {model!r}; "
            f"add MODEL_RUNNER_SCRIPT[{model!r}] in pipeline/config.py"
        )
    payload = {"model": model, "run_id": run_id, "params": params}
    return run_runner_subprocess(...)  # как было
```

- [ ] **Step 4: Tests pass**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/pipeline.py pipeline/tests/test_pipeline.py
git commit -m "fix(pipeline): raise RunnerError instead of KeyError when runner script missing"
```

---

## Task 5: Drop dead OUTPUT_ROOT import in cli.py (M3)

**Files:**
- Modify: `pipeline/pipeline/cli.py`

- [ ] **Step 1: Find and verify import is unused**

```bash
.venv/bin/python -c "
import ast
src = open('/Users/maxos/PythonProjects/diploma/pipeline/pipeline/cli.py').read()
tree = ast.parse(src)
# OUTPUT_ROOT used anywhere outside imports?
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id == 'OUTPUT_ROOT' and not isinstance(node.ctx, ast.Store):
        # only references in imports/definitions count as 'used'
        pass
print('OUTPUT_ROOT references:')
import re
for i, line in enumerate(src.splitlines(), 1):
    if 'OUTPUT_ROOT' in line:
        print(f'  line {i}: {line}')
"
```

Expected: only the import line. Если в коде используется — НЕ удалять.

- [ ] **Step 2: Remove line**

В `pipeline/pipeline/cli.py` удалить строку:
```python
from pipeline.config import OUTPUT_ROOT
```

- [ ] **Step 3: Run all tests + smoke import**

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -c "from pipeline.cli import main; print('ok')"
```

Expected: тесты зелёные, import работает.

- [ ] **Step 4: Commit**

```bash
git add pipeline/pipeline/cli.py
git commit -m "chore(pipeline): drop dead OUTPUT_ROOT import in cli.py"
```

---

## Task 6: Replace silent StopIteration with assert in build_mingus_xml (M6)

**Files:**
- Modify: `pipeline/pipeline/_xml_builders/mingus_xml.py`

**Контекст:** В `build_mingus_xml` цикл по барам исчерпывает chord_iter синхронно с количеством баров (это гарантируется валидаторами выше). Текущий код «глотает» возможную рассинхронизацию через try/except StopIteration. Меняем на явный assert — если рассинхрон случится, увидим где.

- [ ] **Step 1: Read current code, locate the try/except**

```bash
grep -n "StopIteration" /Users/maxos/PythonProjects/diploma/pipeline/pipeline/_xml_builders/mingus_xml.py
```

Найти блок:
```python
        cur_remaining -= bpb
        if cur_remaining <= 0:
            try:
                cur_chord, cur_remaining = next(chord_iter)
            except StopIteration:
                cur_chord, cur_remaining = (cur_chord, 0)
```

- [ ] **Step 2: Refactor cycle to handle exhaustion explicitly**

Заменить блок выше на:

```python
        cur_remaining -= bpb
        if cur_remaining <= 0 and bar < progression.num_bars() - 1:
            # ещё есть бары, должен быть следующий аккорд
            cur_chord, cur_remaining = next(chord_iter)
```

То есть `next(chord_iter)` вызывается только если есть смысл (не последний бар). Если синхронизация нарушена — `StopIteration` пробьётся естественно (это и есть assert «по сути»).

Альтернативно можно использовать явный `assert`:

```python
        cur_remaining -= bpb
        if cur_remaining <= 0:
            try:
                cur_chord, cur_remaining = next(chord_iter)
            except StopIteration:
                assert bar == progression.num_bars() - 1, (
                    f"chord iterator exhausted before last bar: bar={bar}, "
                    f"num_bars={progression.num_bars()}"
                )
```

Выбрать первый вариант (он чище). После цикла проверить что итератор тоже исчерпан:

```python
    # после for bar in range(...):
    remaining_chords = list(chord_iter)
    assert not remaining_chords, (
        f"chord iterator has {len(remaining_chords)} unused chords after "
        f"{progression.num_bars()} bars; total_beats validation should have caught this"
    )
```

- [ ] **Step 3: Run all tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: 74+ passed. Особое внимание на `test_mingus_xml.py` — должны все пройти (8 tests).

- [ ] **Step 4: Commit**

```bash
git add pipeline/pipeline/_xml_builders/mingus_xml.py
git commit -m "refactor(pipeline): replace silent StopIteration fallback with explicit assert"
```

---

## Task 7: Drop unused chord_vocab fallback dicts (M8)

**Что:** Сейчас `QUALITY_FALLBACK_TO_TRIADS` и `EXTENDED_FALLBACK_TO_VOCAB` экспортируются, но в `parse_chord` не используются. Если кто-то даст `Cm7b5`, `parse_chord` упадёт с `unknown quality` — словарь не помогает. Это «висящий API». Удаляем до момента, когда реально понадобится (вместе с использованием в `parse_chord`).

**Files:**
- Modify: `pipeline/pipeline/chord_vocab.py` — удалить две константы
- Modify: `pipeline/tests/test_chord_vocab.py` — удалить 2 теста (`test_quality_fallback_to_triads`, `test_extended_fallback`)

- [ ] **Step 1: Remove constants from chord_vocab.py**

В `pipeline/pipeline/chord_vocab.py` удалить блоки:

```python
QUALITY_FALLBACK_TO_TRIADS: dict[str, str] = {...}

EXTENDED_FALLBACK_TO_VOCAB: dict[str, str] = {...}
```

- [ ] **Step 2: Remove tests in test_chord_vocab.py**

Удалить функции `test_quality_fallback_to_triads` и `test_extended_fallback`. Также удалить эти имена из import statement:

```python
# было
from pipeline.chord_vocab import (
    ROOTS, QUALITIES,
    parse_chord, chord_to_pitches,
    QUALITY_FALLBACK_TO_TRIADS, EXTENDED_FALLBACK_TO_VOCAB,
)

# стало
from pipeline.chord_vocab import (
    ROOTS, QUALITIES,
    parse_chord, chord_to_pitches,
)
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/test_chord_vocab.py -v
```

Expected: было 20, стало 18 (удалили 2). Все проходят.

- [ ] **Step 4: Run full suite — никто не импортирует удалённые константы**

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -c "from pipeline.chord_vocab import ROOTS, QUALITIES, parse_chord, chord_to_pitches; print('ok')"
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline/chord_vocab.py pipeline/tests/test_chord_vocab.py
git commit -m "chore(pipeline): drop unused chord_vocab fallback dicts"
```

---

## Task 8: Document why setuptools<81 is pinned

**Files:**
- Modify: `pipeline/requirements.txt`

- [ ] **Step 1: Add inline comment**

В `pipeline/requirements.txt` заменить:

```
setuptools<81
```

на

```
# pretty_midi 0.2.10 imports pkg_resources (removed in setuptools>=81).
# Drop this pin when pretty_midi switches to importlib.resources.
setuptools<81
```

- [ ] **Step 2: Smoke check that pip still parses it**

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/pip install --dry-run -r requirements.txt 2>&1 | tail -5
```

Expected: no parse errors.

- [ ] **Step 3: Commit**

```bash
git add pipeline/requirements.txt
git commit -m "docs(pipeline): document why setuptools<81 is pinned"
```

---

## Self-Review

После выполнения всех 8 задач проверить:

```bash
cd /Users/maxos/PythonProjects/diploma/pipeline
.venv/bin/python -m pytest -v
# expect: ~78 passed (73 + tests added in T1, T2, T3, T4 - tests removed in T7)
# T1: +1 test (requires_config) → +1
# T2: +2 tests (zero/negative tempo) → +2
# T3: +1 test (timeout) → +1
# T4: +1 test (runner_script_missing) → +1
# T7: -2 tests (fallback dicts) → -2
# total delta = +3, expect 76 passed

.venv/bin/python -m pipeline.cli generate test_progressions/sample.json
# expect same output as before — pipeline still works end-to-end
```

Если e2e сломалось — есть регрессия в Task 1 (наиболее вероятно). Откатить Task 1 commits, дебажить.

---

## Definition of Done

- ✅ 8 коммитов на ветке `feat/pipeline-mingus`
- ✅ pytest 76 passed
- ✅ end-to-end smoke (`cli generate sample.json`) даёт тот же результат что и до правок
- ✅ 8 пунктов из финального code review закрыты
