# Abstract pipeline

Скелет контрактов для нового pipeline: один запуск = одна модель.

## Порядок слоёв

1. **Приём входа** — общая реализация.
2. **Валидация под модель** — per-model, опирается на `ModelSpec`.
3. **Генерация** — per-model. Монолит: подготовка входа + запуск subprocess + сырой выход.
4. **Извлечение мелодии** — per-model. Сырой выход → `Melody`.
5. **Финальная сборка** — общая.

Сбоку (не шаги, а инжектируемые сущности):
- `ModelSpec` — что модель поддерживает (словарь аккордов, ограничения по длине, размер такта и т.п.).
- `CommonInputValidator` — pipeline-уровневая валидация (что наш pipeline в принципе принимает). Не зависит от модели. Зовётся оркестратором **до** per-model валидатора.
- `ModelRegistry` — DI: `ModelName → (validator, generator, extractor)`.
- `Orchestrator` — гонит шаги 1→5 в последовательности.
- `RunContext` — состояние одного запуска (`run_id`, `tmp_dir`).

---

## Слой 1 — Приём входа

**Что делает.** Берёт сырой вход (JSON-файл, CLI-аргументы, dict) и превращает его в типизированный `PipelineInput`. Дальше pipeline работает только с этим объектом.

**Чего не делает.**
- Не проверяет, подходит ли замысел конкретной модели — это слой 2.
- Не строит для моделей никаких файлов — это слой 3.
- Не знает, какую модель пользователь выбрал — `model_name` не часть данных, это директива оркестратора.

**Граница со следующим слоем.** Слой 1 отвечает на вопрос «это вообще наш формат». Слой 2 — «этот формат подходит выбранной модели».

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Progression:
    """Замысел: аккорды + темп + размер. Поля наполним позже."""


@dataclass(frozen=True)
class Melody:
    """Монофонная мелодия. Один тип для двух ролей:
    опциональная затравка на входе и итоговая мелодия на выходе."""


@dataclass(frozen=True)
class PipelineInput:
    """Только данные замысла. Без управляющих параметров типа model_name."""
    progression: Progression
    theme: Melody | None  # затравка опциональна


class InputSource(ABC):
    """Слой 1 — приём входа.

    Реализации (JSON-файл, CLI-аргументы, dict) превращают сырой вход
    в PipelineInput. Источник в state самого экземпляра, поэтому load()
    без аргументов.
    """

    @abstractmethod
    def load(self) -> PipelineInput: ...
```

---

## Слой 2 — Валидация под модель

**Что делает.** Получает `PipelineInput`, читает `ModelSpec` своей модели и проверяет: подходит ли замысел и тема под выбранную модель. Возвращает либо «ок», либо список всех претензий разом.

**Чего не делает.**
- Не правит вход.
- Не строит файлов.
- Не обращается к чекпоинту и subprocess.

**Граница со следующим слоем.** Слой 2 отвечает «можно ли запускать модель на этом входе». Слой 3 — «как мы переводим вход в формат модели и запускаем её».

**`ModelSpec`** — описание возможностей модели (словарь аккордов, допустимая длина, размер такта). Не путать с runtime-настройками генерации (температура и т.п.) — те живут внутри генератора. На `ModelSpec` одной модели смотрят оба слоя — валидатор и генератор — чтобы модельное знание хранилось в одном месте.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ModelSpec(ABC):
    """Описание того, что модель умеет принимать на вход.

    "Вшитые" свойства чекпоинта и архитектуры: словарь аккордов,
    допустимый размер такта, ограничения по длине, требования к теме.
    Не runtime-настройки.

    На ModelSpec одной модели смотрят и валидатор (слой 2), и
    генератор (слой 3) — так модельное знание не дублируется.

    Конкретные поля у моделей радикально разные, поэтому базовый
    класс пуст. Подклассы (MingusSpec, BebopNetSpec, CMTSpec) —
    frozen dataclass-ы со своими полями.
    """


@dataclass(frozen=True)
class ValidationError:
    """Одна претензия к замыслу: что не подошло и почему."""
    code: str       # короткий идентификатор причины (e.g. "unsupported_chord")
    message: str    # человекочитаемое объяснение


@dataclass(frozen=True)
class ValidationResult:
    errors: list[ValidationError]

    @property
    def ok(self) -> bool:
        return not self.errors


class InputValidator(ABC):
    """Слой 2 — валидация замысла под конкретную модель.

    Реализации: MingusInputValidator, BebopNetInputValidator,
    CMTInputValidator. Каждая опирается на свой ModelSpec
    (получает его через __init__).

    Контракт: возвращает ValidationResult со всеми найденными
    проблемами разом, не первую попавшуюся. Не правит вход,
    не строит файлов, не обращается к чекпоинту.
    """

    @abstractmethod
    def validate(self, pipeline_input: PipelineInput) -> ValidationResult: ...
```

### CommonInputValidator (pipeline-уровневая валидация)

Помимо per-model валидатора есть **общая** валидация — то что наш pipeline в принципе принимает, независимо от модели. Зовётся оркестратором **перед** per-model валидатором; если возвращает непустой `errors` — pipeline обрывается, per-model валидатор даже не зовётся.

Что валидируется на pipeline-уровне:
- Все аккорды парсятся нашим словарём (root + одно из 7 поддерживаемых качеств).
- Прогрессия не пуста, длительности > 0.
- Сумма долей делится на `beats_per_bar` (укладывается в целое число баров).
- Темп в разумном диапазоне.
- Размер такта в формате `"N/M"`.

Что **не** валидируется на pipeline-уровне (это per-model):
- Конкретный размер такта (например, MINGUS требует 4/4).
- Кратность длительности аккорда бару (MINGUS).
- Длина прогрессии под чекпоинт (CMT).

Контракт-результат тот же что у `InputValidator` (`ValidationResult`), но это **отдельный класс** с собственным ABC. Per-model валидаторы на него не наследуются — это совсем другая ответственность.

```python
from abc import ABC, abstractmethod


class CommonInputValidator(ABC):
    """Pipeline-уровневая валидация, не зависит от модели.

    Реализация на сегодня одна (общая для всех моделей).
    Зовётся оркестратором ПЕРЕД per-model валидатором.
    """

    @abstractmethod
    def validate(self, pipeline_input: PipelineInput) -> ValidationResult: ...
```

---

## Слой 3 — Генерация

**Что делает.** Per-model монолит. Получает `PipelineInput` и `RunContext`. Внутри последовательно: (а) готовит модели её родной входной формат в `tmp_dir` (MusicXML, npz и т.п.); (б) запускает subprocess в её venv; (в) возвращает `RawOutput`. Эти три действия — внутренние детали конкретной модели, не общий контракт, поэтому не разнесены по разным слоям.

**Чего не делает.**
- Не извлекает мелодию (слой 4).
- Не пишет финальные артефакты (слой 5).
- Не валидирует — слой 2 уже отработал.

**Опирается на.** `ModelSpec` и runtime-настройки (температура, beam_width, путь к чекпоинту). Все они приходят в `__init__` и живут в state экземпляра.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    """Состояние одного запуска pipeline. Создаётся оркестратором,
    пробрасывается в слои, работающие с диском (3, 4, 5).

    model_label — короткая строка для имён файлов и логов (типа
    "mingus", "bebopnet"). Это непрозрачный для слоя 5 префикс:
    packer не должен знать enum ModelName, ему достаточно строки.
    """
    run_id: str
    tmp_dir: Path       # уникальная директория этого запуска
    model_label: str    # префикс для имён артефактов


@dataclass(frozen=True)
class RawOutput:
    """Сырой выход модели после subprocess.

    Контракт между слоем 3 (генерация) и слоем 4 (извлечение).
    Все три модели сейчас пишут MIDI-файл на диск, поэтому
    минимально храним Path. При необходимости можно расширить
    (метаданные, дополнительные артефакты).
    """
    path: Path


class Generator(ABC):
    """Слой 3 — генерация. Per-model монолит.

    Реализации: MingusGenerator, BebopNetGenerator, CMTGenerator.
    Каждая внутри себя:
        (а) готовит входной формат модели в tmp_dir (MusicXML, npz);
        (б) запускает subprocess в venv модели;
        (в) возвращает RawOutput.

    Эти три действия не вынесены в отдельные слои, потому что
    формат файла и протокол subprocess — внутренние детали
    конкретной модели, а не общий контракт.

    ModelSpec и runtime-настройки (температура, beam_width, путь
    к чекпоинту) живут в state экземпляра — приходят через __init__.
    """

    @abstractmethod
    def generate(
        self,
        pipeline_input: PipelineInput,
        ctx: RunContext,
    ) -> RawOutput: ...
```

---

## Слой 4 — Извлечение мелодии

**Что делает.** Получает `RawOutput`, достаёт из него монофонную мелодию и приводит её к нашему единому типу `Melody`. Per-model потому что модели возвращают MIDI по-разному: одна кладёт мелодию в трек с конкретным именем, другая в трек с другим именем, третья — первым по индексу.

**Чего не делает.**
- Не накладывает финальный тембр (слой 5).
- Не строит chord-track (слой 5).
- Не пишет файлов наружу.

**Граница со слоем 5.** Слой 4 отвечает «вот мелодия в нашем внутреннем виде, чистая». Слой 5 упаковывает её для пользователя.

```python
from abc import ABC, abstractmethod


class MelodyExtractor(ABC):
    """Слой 4 — извлечение мелодии из сырого выхода модели.

    Реализации: MingusMelodyExtractor, BebopNetMelodyExtractor,
    CMTMelodyExtractor. Каждая знает, как из RawOutput выделить
    монофонную мелодическую линию (по имени трека, по индексу,
    проекцией полифонии и т.п.).

    Контракт: возвращает Melody — наш единый внутренний тип
    (тот же, что используется для опциональной темы на входе).
    Не накладывает финальный тембр и не строит chord-track.

    Параметры (имя трека, индекс) — в state экземпляра через __init__.
    """

    @abstractmethod
    def extract(self, raw: RawOutput) -> Melody: ...
```

---

## Слой 5 — Финальная сборка

**Что делает.** Берёт уже извлечённую `Melody`, исходную `Progression` и `ModelName`. Накладывает единый тембр, строит chord-track из прогрессии, пишет два MIDI-файла: «чистая мелодия» и «мелодия + аккорды». Имена файлов включают модель и `run_id`.

**Чего не делает.** Не извлекает мелодию. Не запускает модели. Не валидирует.

**Per-model или общий.** На сегодня одна общая реализация — моделям не нужны дополнительные артефакты. Если кому-то понадобится — появится per-model подкласс.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ModelName(str, Enum):
    """Директива оркестратору: какую модель запустить.

    Не часть PipelineInput — это управляющий параметр, а не данные
    замысла. Используется только в оркестраторе (выбор реализаций
    через реестр). В слой 5 не пробрасывается: packer работает с
    непрозрачной строкой ctx.model_label.
    """
    MINGUS = "mingus"
    BEBOPNET = "bebopnet"
    CMT = "cmt"


@dataclass(frozen=True)
class FinalArtifacts:
    """Итоговые артефакты одного запуска."""
    melody_only: Path  # чистая мелодия с нашим единым тембром
    with_chords: Path  # мелодия + chord-track из исходной прогрессии


class ResultPacker(ABC):
    """Слой 5 — финальная сборка.

    Реализация на сегодня одна. Если какой-то модели понадобится
    писать дополнительные артефакты — появится per-model подкласс.

    Принимает уже извлечённую Melody и исходный Progression.
    Накладывает единый тембр, строит chord-track, пишет два MIDI
    в стандартные директории внутри output_root. Имена файлов
    собираются из ctx.model_label и ctx.run_id — packer не знает
    про enum ModelName, ему достаточно строки.

    output_root и параметры стандартизации (program единого тембра,
    имя инструмента, имена директорий) — в state экземпляра.
    """

    @abstractmethod
    def pack(
        self,
        melody: Melody,
        progression: Progression,
        ctx: RunContext,
    ) -> FinalArtifacts: ...
```

---

## Сбоку — `ModelRegistry` и `Orchestrator`

Это не шаги pipeline'а, а инжектируемые сущности и точка входа.

**`ModelRegistry`** — DI-реестр. По `ModelName` отдаёт уже сконструированный набор per-model реализаций: `(InputValidator, Generator, MelodyExtractor)`. Каждая реализация в наборе уже несёт свои настройки (ModelSpec и runtime-параметры пришли в её `__init__` при сборке реестра).

`InputSource` и `ResultPacker` — общие, в реестре их нет.

**`Orchestrator`** — точка входа. В `__init__` принимает: `InputSource`, `CommonInputValidator`, `ModelName`, `ModelRegistry`, `ResultPacker`. В `run()` создаёт `RunContext` (новый `run_id`, новый `tmp_dir`) и гонит пять шагов. На шаге 2 валидация двухэтапная: сначала `CommonInputValidator`, затем per-model `InputValidator` из bundle. Если хоть один вернул непустой `errors` — поднимает `ValidationFailedError` со всеми претензиями этого этапа (per-model не зовётся, если общий уже отказал).

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelBundle:
    """Тройка per-model реализаций под одну модель.
    InputSource и ResultPacker сюда не входят — они общие.

    Условие сборки bundle: validator и generator получают **один
    и тот же** инстанс ModelSpec в свои __init__. Это не просто
    стилевая договорённость, а контракт: модельное знание
    (словарь аккордов, размеры) должно быть согласованным между
    проверкой замысла и его последующим преобразованием.
    Реестр обязан это обеспечить при создании bundle.
    """
    validator: InputValidator
    generator: Generator
    extractor: MelodyExtractor


class ModelRegistry(ABC):
    """DI-реестр. По ModelName отдаёт ModelBundle.

    Как реестр собран (явный dict, фабрики, конфиг) — детали реализации.
    Контракт минимален: имя → bundle.
    """

    @abstractmethod
    def get(self, model_name: ModelName) -> ModelBundle: ...


class ValidationFailedError(Exception):
    """Поднимается оркестратором, если валидатор вернул непустой
    список ошибок. Несёт все претензии, не первую попавшуюся.
    """
    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"[{e.code}] {e.message}" for e in errors))


class Orchestrator(ABC):
    """Точка входа pipeline. Гонит шаги 1→5 в последовательности.

    В __init__: InputSource, CommonInputValidator, ModelName,
    ModelRegistry, ResultPacker. В run() создаёт RunContext
    (run_id, tmp_dir, model_label) и исполняет поток так, что
    PipelineInput жив до самого шага 5 (его поля нужны и для
    генерации, и для финальной сборки):

        inp     = source.load()                          # шаг 1
        common  = common_validator.validate(inp)         # шаг 2.A (pipeline)
        if not common.ok:
            raise ValidationFailedError(common.errors)
        bundle  = registry.get(model_name)               # выбор реализаций
        per_m   = bundle.validator.validate(inp)         # шаг 2.Б (модель)
        if not per_m.ok:
            raise ValidationFailedError(per_m.errors)
        raw     = bundle.generator.generate(inp, ctx)    # шаг 3
        melody  = bundle.extractor.extract(raw)          # шаг 4
        return packer.pack(
            melody, inp.progression, ctx,                # шаг 5
        )

    Ошибки этапов 2.A и 2.Б НЕ объединяются: если общий валидатор
    отказал, per-model даже не зовётся (модельные ограничения
    обсуждать смысла нет, если pipeline вообще не примет вход).

    PipelineInput намеренно не пересоздаётся между шагами и не
    мутируется: один объект на весь запуск, читается слоями 2, 3, 5.
    """

    @abstractmethod
    def run(self) -> FinalArtifacts: ...
```
