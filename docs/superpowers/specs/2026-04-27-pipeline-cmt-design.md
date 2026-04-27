# Дизайн интеграции CMT в pipeline — production

**Дата:** 2026-04-27
**Контекст:** ВКР НИУ ВШЭ МИЭМ. Базовый дизайн — `2026-04-27-pipeline-mingus-design.md`. MINGUS закоммичена. Этот документ — дельта для второй модели (CMT).

Production-интеграция. Любая `ChordProgression`, для которой текущие веса физически могут произвести вывод, превращается в реальную chord-conditioned мелодию. Замена весов на финальные — подмена пары `(checkpoint, hparams)` в `pipeline/pipeline/config.py`, без правок кода адаптера/runner'а/общих модулей.

---

## 1. Цель и Definition of Done

**Цель.** Подключить CMT как полноправную модель pipeline. На вход — `ChordProgression`. На выход — MIDI с реальной мелодией под эту гармонию. Никаких ограничений в коде, которые завязаны на конкретный чекпоинт.

**DoD:**

- ✅ `models/CMT-pytorch` — git submodule на `kudrmax/CMT-pytorch.git`. Форк содержит `requirements-py312.txt` и патчи под Python 3.12 (`preprocess.py` + `utils/hparams.py`).
- ✅ `models/CMT-pytorch/.venv` создан, импорт `model.ChordConditionedMelodyTransformer` и `utils.utils.pitch_to_midi` работает.
- ✅ `python -m pipeline.cli generate test_progressions/sample.json` показывает `cmt = ok`.
- ✅ Две разные `ChordProgression` дают разные мелодии — модель действительно conditioned на гармонии.
- ✅ MIDI-выход валиден: `pretty_midi.PrettyMIDI(...)` открывает; в `melody_only.mid` ровно 1 трек, len(notes) > 0.
- ✅ Адаптер не содержит магических чисел `8`/`16`/`128`/`50`/`"4/4"` — все размеры читаются из `hparams.yaml`.
- ✅ Адаптер валидирует только то что **физически невозможно** для текущих весов: длина прогрессии в фреймах ≠ `max_len`, или `frame_per_bar` не делится на `beats_per_bar`.
- ✅ Все pytest-тесты зелёные. `CMTAdapter` удалён из stub-параметризации в `tests/adapters/test_base.py`. Новые тесты на каждый блок (config, валидация, chroma, seed, prepare, extract_melody) — независимые, не требуют локальных весов.
- ✅ Замена весов — переписать `CMT_CHECKPOINT_PATH` и `CMT_HPARAMS_PATH` в `pipeline/pipeline/config.py`. Никаких других правок.

**За пределами этого раунда:**
- ❌ Поддержка прогрессий длиннее чем `max_len` через chain-инференс — отдельный design.
- ❌ Метрики качества (FMD, scale match).

---

## 2. Архитектурный инвариант (без отступлений)

Полное соответствие MINGUS-дизайну (см. `pipeline/CLAUDE.md`):

- **`pipeline-venv` ≠ `models/CMT-pytorch/.venv`.** Pipeline в своём venv не импортирует код CMT. Общается через `subprocess + JSON через stdin`.
- **Adapter (`pipeline/pipeline/adapters/cmt.py`)** — toolbox. Реализует `prepare(progression, tmp_dir) → params: dict` и `extract_melody(raw_midi) → Instrument`. Знает формат CMT.
- **Runner (`pipeline/runners/cmt_runner.py`)** — чистая обёртка. **Не импортирует** ничего из `pipeline.*`. Принимает только то что API CMT реально требует.
- **Общие модули** (`progression`, `chord_vocab`, `chord_render`, `postprocess`, `pipeline.py`, `runner_protocol.py`) — **не трогаются**.

Граница «factor API модели или наш pipeline-фактор»:
- API модели → runner: путь к чекпоинту/hparams, `topk`, `device`.
- Наш pipeline-фактор → adapter (через `CMTPipelineConfig`): `seed_strategy`, `prime_bars`, `custom_seed_path`.

`ChordProgression` несёт только композиционный замысел (chords + tempo + time_signature). Никаких CMT-полей в ней.

---

## 3. CMT-вход: что требуется от модели

Из `models/CMT-pytorch/model.py:8-15`:

```python
class ChordConditionedMelodyTransformer(nn.Module):
    def __init__(self, num_pitch=89, frame_per_bar=16, num_bars=8, ...):
        self.max_len = frame_per_bar * num_bars
        ...

    def sampling(self, prime_rhythm, prime_pitch, chord, topk):
        ...
```

**Что определяется текущим чекпоинтом (читается из `hparams.yaml`, секция `model`):**

| параметр | смысл |
|---|---|
| `frame_per_bar` | сколько фреймов помещается в один такт (минимальная длительность) |
| `num_bars` | сколько тактов модель генерирует за один прогон |
| `num_pitch` | размер pitch-словаря (включая спец-токены) |
| `chord_emb_size` / `pitch_emb_size` / `hidden_dim` / `num_layers` / `num_heads` / `*_dropout` / `key_dim` / `value_dim` | архитектурные параметры — нужны runner'у при инстанцировании модели, адаптеру не нужны |

`max_len = frame_per_bar × num_bars` — длина временной развёртки.

**Что определяется кодом CMT (`utils/utils.py`, не зависит от чекпоинта):**

- `pitch_to_midi` (строки 58-96): из последовательности pitch-индексов и chord-тензора пишет MIDI с двумя треками `name='melody'` и `name='chord'`.
- Pitch-vocab семантика: `idx ∈ [0, num_pitch-2-?]` — onset notes; есть два спец-токена для sustain/note-off. **Точные индексы определяются на стадии реализации Task 7** (открываем локальный `seed_instance.pkl`, смотрим какие коды на удерживаемых нотах).
- Rhythm-vocab: 3 класса (`onset` / `sustain` / прочее).

Эти константы фиксируются в коде адаптера (`pipeline/pipeline/adapters/_cmt_input.py`) с комментарием-ссылкой на `utils/utils.py` строки. При смене кода CMT (что происходит крайне редко) — обновляются вместе.

**Chord-chroma:** `[B, max_len + 1, 12]`. Каждый фрейм — 12-bool вектор активных pitch-классов аккорда. Размерность 12 — фундаментальное свойство (12 pitch classes), не зависит от чекпоинта.

**`sampling(prime_rhythm, prime_pitch, chord, topk)`:**
- `prime_rhythm`, `prime_pitch`: `[B, prime_len]`, dtype long.
- `chord`: `[B, max_len + 1, 12]`, dtype float.
- Модель генерирует фреймы от `prime_len` до `max_len` авторегрессивно под `chord`.

---

## 4. Адаптер: чтение hparams

`CMTAdapter.prepare` в момент вызова:

1. Читает `hparams.yaml` через `yaml.safe_load`, берёт секцию `model`.
2. Извлекает `frame_per_bar` и `num_bars`. Считает `max_len = frame_per_bar × num_bars`.
3. Использует эти числа для построения `chord_chroma`, проверки длины прогрессии и расчёта `prime_len`.

**Lazy чтение:** hparams читается **внутри `prepare(...)`**, не в `__init__`. Это значит:
- `from pipeline.config import ADAPTERS` работает без чекпоинта/hparams на машине (важно для разработки и юнит-тестов pipeline).
- Ошибка отсутствия hparams проявляется только при попытке генерации — с явным `FileNotFoundError`.

---

## 5. Pipeline-фактор: затравка (3 стратегии)

`CMTPipelineConfig.seed_strategy` + `CMTPipelineConfig.prime_bars: int = 1` (длина затравки в тактах).

Реальная длина затравки в фреймах: `prime_len = prime_bars × frame_per_bar` (frame_per_bar читается из hparams).

| `seed_strategy` | поведение |
|---|---|
| `tonic_held` (default) | На фрейме 0 — корень первого аккорда (MIDI = 60 + root_idx → pitch_idx; rhythm = onset). На фреймах 1..prime_len-1 — sustain (pitch_idx = SUSTAIN_PITCH_IDX, rhythm_idx = SUSTAIN_RHYTHM_IDX). Одна целая нота на всю длину затравки. |
| `tonic_quarters` | На каждом бите внутри prime_bars — onset тоники первого аккорда (pitch_idx = root_idx, rhythm_idx = onset). Между битами — sustain. Шаг между онсетами = `frame_per_bar / beats_per_bar` (frame_per_beat). |
| `custom_seed` | Читает `config.custom_seed_path` (формат `.npz` с массивами `pitch` и `rhythm`, оба int64 длины ≥ `prime_len`). Создаётся через `np.savez(path, pitch=..., rhythm=...)`. Берёт `pitch[:prime_len]`, `rhythm[:prime_len]`. Без зависимости от scipy/pickle. |

Имена `tonic_held` и `tonic_quarters` отличаются от MINGUS-овских `tonic_whole`/`tonic_quarters` намеренно: у CMT затравка живёт только в первых `prime_bars` тактах (модель досочиняет остальное), у MINGUS — на всех. Семантика разная, имена честные.

---

## 6. Конвертер `ChordProgression → chord_chroma`

Чистая функция в `pipeline/pipeline/adapters/_cmt_input.py`:

```python
def progression_to_chroma(
    progression: ChordProgression,
    frame_per_bar: int,
    num_bars: int,
) -> np.ndarray:
    """Возвращает chord_chroma shape `[max_len + 1, 12]` dtype float32.

    max_len = frame_per_bar * num_bars.

    Raises:
        ValueError: если frame_per_bar не делится на beats_per_bar нацело
                    (физически нельзя развернуть в целое число фреймов).
        ValueError: если total_frames(progression) != max_len.
    """
```

Алгоритм:
1. `bpb = progression.beats_per_bar()`. Если `frame_per_bar % bpb != 0` → ValueError.
2. `frames_per_beat = frame_per_bar / bpb`.
3. Для каждого `(chord_str, beats)` в progression: разворачиваем chroma-вектор аккорда (через существующий `chord_vocab.chord_to_pitches` и `% 12`) на `beats × frames_per_beat` фреймов.
4. Конкатенируем → `[total_frames, 12]`. Если `total_frames != frame_per_bar × num_bars` → ValueError.
5. Добавляем zero-frame в конце (для target padding по контракту CMT) → `[max_len + 1, 12]`.

Использует только `pipeline.chord_vocab` и `pipeline.progression`. Не лезет в общие модули.

---

## 7. Контракт adapter'а

```python
@dataclass
class CMTPipelineConfig:
    """Все настройки CMT на уровне pipeline. immutable после init."""
    checkpoint_path: Path
    hparams_path: Path
    repo_path: Path
    seed_strategy: Literal["tonic_held", "tonic_quarters", "custom_seed"] = "tonic_held"
    custom_seed_path: Path | None = None
    prime_bars: int = 1
    topk: int = 5
    device: str = "cpu"


class CMTAdapter(ModelAdapter):
    def __init__(self, config: CMTPipelineConfig) -> None:
        self._config = config

    def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
        # 1. Читаем hparams.yaml → frame_per_bar, num_bars.
        # 2. Валидация:
        #    - seed_strategy ∈ {tonic_held, tonic_quarters, custom_seed}
        #    - custom_seed ⇒ custom_seed_path задан
        #    - prime_bars ≥ 1 и prime_bars ≤ num_bars
        # 3. chroma = progression_to_chroma(progression, frame_per_bar, num_bars)
        #    (внутри валит ValueError если frame_per_bar не делится на bpb,
        #    или если total_frames != max_len)
        # 4. prime_len = prime_bars * frame_per_bar
        #    prime_pitch, prime_rhythm = build_seed(progression, cfg, frame_per_bar, prime_len)
        # 5. Сохраняем seed.npz, возвращаем dict с params.
        ...

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        # CMT pitch_to_midi пишет 2 трека: 'melody' и 'chord'. Берём 'melody'.
        ...
```

`prepare` возвращает:

```python
{
    "checkpoint_path":   str(cfg.checkpoint_path),
    "hparams_path":      str(cfg.hparams_path),
    "model_repo_path":   str(cfg.repo_path),
    "seed_npz_path":     str(tmp_dir / "seed.npz"),
    "output_midi_path":  str(tmp_dir / "raw.mid"),
    "topk":              cfg.topk,
    "device":            cfg.device,
}
```

`seed.npz` хранит: `chord_chroma` (float32, `[max_len+1, 12]`), `prime_pitch` (int64, `[prime_len]`), `prime_rhythm` (int64, `[prime_len]`).

Адаптер не передаёт `frame_per_bar`/`num_bars` в runner — runner сам прочитает их из hparams.yaml для конструктора модели.

---

## 8. Контракт runner'а

`pipeline/runners/cmt_runner.py`. Запускается `models/CMT-pytorch/.venv/bin/python`. **Не импортирует ничего из `pipeline.*`.**

Шаги:

1. `os.chdir(repo_path); sys.path.insert(0, repo_path)`.
2. Загрузить `hparams.yaml` через `yaml.safe_load`; взять секцию `model` как `**kwargs`.
3. `model = ChordConditionedMelodyTransformer(**model_config); model.load_state_dict(torch.load(checkpoint)['model']); model.eval()`.
4. `npz = np.load(seed_npz_path)`. Конвертировать в torch:
   - `chord = torch.tensor(npz['chord_chroma']).float().unsqueeze(0)`
   - `prime_pitch = torch.tensor(npz['prime_pitch']).long().unsqueeze(0)`
   - `prime_rhythm = torch.tensor(npz['prime_rhythm']).long().unsqueeze(0)`
5. `with torch.no_grad(): result = model.sampling(prime_rhythm, prime_pitch, chord, topk=topk)`.
6. `pitch_idx = result['pitch'][0].cpu().numpy()`; `chord_arr = chord[0].cpu().numpy()`.
7. `pitch_to_midi(pitch_idx, chord_arr, frame_per_bar=model.frame_per_bar, save_path=output_midi_path)`.
8. `exit 0`. Ошибка → `traceback.print_exc(); exit 1`.

---

## 9. Изменения в форке `kudrmax/CMT-pytorch`

Форк сейчас — clean upstream. Push 3 коммита:

1. **`fix: Python 3.11+ compat (random.sample on set)`** — `preprocess.py` строки 42-43: добавить `sorted(...)`.
2. **`fix: yaml.safe_load instead of yaml.load`** — `utils/hparams.py:28`.
3. **`add: requirements-py312.txt for runtime deps`** — `torch`, `numpy`, `scipy`, `pretty_midi`, `pyyaml`, `matplotlib`, `tensorboardX`, `tqdm`.

Артефакты smoke train (`hparams.yaml` корня, `trainer.py`, `result/`) — **не пушим**.

---

## 10. Изменения в pipeline-конфигурации

`pipeline/pipeline/config.py`:

```python
CMT_REPO_PATH:        Path = DIPLOMA_ROOT / "models" / "CMT-pytorch"
CMT_RESULT_DIR:       Path = CMT_REPO_PATH / "result" / "smoke_wjazzd_5epochs"
CMT_CHECKPOINT_PATH:  Path = CMT_RESULT_DIR / "smoke_5epochs.pth.tar"  # ← подмена весов
CMT_HPARAMS_PATH:     Path = CMT_RESULT_DIR / "hparams.yaml"           # ← гипер-параметры в паре с весами

ADAPTERS["cmt"] = CMTAdapter(CMTPipelineConfig(
    checkpoint_path=CMT_CHECKPOINT_PATH,
    hparams_path=CMT_HPARAMS_PATH,
    repo_path=CMT_REPO_PATH,
    seed_strategy="tonic_held",
    prime_bars=1,
    topk=5,
    device="cpu",
))
MODEL_RUNNER_SCRIPT["cmt"] = RUNNERS_ROOT / "cmt_runner.py"
```

**Замена весов на финальные:**
- Положить `final_v1/checkpoint.pth.tar` + `final_v1/hparams.yaml` в `models/CMT-pytorch/result/`.
- Поменять `CMT_RESULT_DIR = CMT_REPO_PATH / "result" / "final_v1"` (или прямо обе константы).

Адаптер автоматически прочтёт новые `frame_per_bar`/`num_bars` из нового `hparams.yaml`. Никаких других правок.

---

## 11. Артефакты на хосте (pre-requisites)

По аналогии с `DATA.json` в MINGUS:

- `models/CMT-pytorch/result/smoke_wjazzd_5epochs/smoke_5epochs.pth.tar` — 147 MB. **Не коммитится**.
- `models/CMT-pytorch/result/smoke_wjazzd_5epochs/hparams.yaml` — 1 KB. По смыслу пара к чекпоинту, лежит рядом.
- `models/CMT-pytorch/result/` — gitignored на стороне submodule (паттерн `**/result/` в корневом `.gitignore`).

---

## 12. Структура файлов после изменений

```
diploma/
├── docs/superpowers/
│   ├── specs/2026-04-27-pipeline-cmt-design.md
│   └── plans/2026-04-27-pipeline-cmt.md
├── models/CMT-pytorch/                       # submodule на kudrmax/CMT-pytorch
│   ├── .venv/                                # gitignored
│   ├── requirements-py312.txt                # в форке
│   ├── preprocess.py                         # пропатчен в форке
│   ├── utils/hparams.py                      # пропатчен в форке
│   └── result/                               # gitignored локально
│       └── smoke_wjazzd_5epochs/
│           ├── smoke_5epochs.pth.tar
│           └── hparams.yaml
├── pipeline/
│   ├── pipeline/
│   │   ├── adapters/
│   │   │   ├── cmt.py                        # переписан
│   │   │   └── _cmt_input.py                 # NEW: chroma + seed builders
│   │   └── config.py                         # +CMT_* константы
│   ├── runners/
│   │   └── cmt_runner.py                     # NEW
│   └── tests/adapters/
│       ├── test_base.py                      # CMTAdapter убран из stub
│       ├── test_cmt_config.py                # NEW
│       ├── test_cmt_chroma.py                # NEW: разные num_bars/fpb
│       ├── test_cmt_seed.py                  # NEW: разные prime_bars
│       ├── test_cmt_validation.py            # NEW
│       ├── test_cmt_prepare.py               # NEW
│       └── test_cmt_extract_melody.py        # NEW
```

---

## 13. Тесты

| тест | поведение |
|---|---|
| `test_cmt_config` | `CMTAdapter()` без аргументов падает TypeError; defaults в `CMTPipelineConfig` соответствуют разделу 7 |
| `test_cmt_chroma` | `progression_to_chroma` с явно переданными `frame_per_bar`/`num_bars`: проверка shape, dtype, правильных pitch-классов для Cmaj7/Am7, разных параметров (fpb=8/num_bars=4, fpb=16/num_bars=8, fpb=24/num_bars=2 — параметризовано); разные progression → разные chroma; ValueError при frame_per_bar не делящемся на beats_per_bar |
| `test_cmt_seed` | `build_seed` с явным `frame_per_bar` и `prime_len`: все 3 стратегии; `tonic_held` для разных первых аккордов даёт разный prime_pitch[0]; `tonic_quarters` ставит онсеты на правильных позициях для разных fpb/bpb; `custom_seed` берёт из pkl, валит ValueError если короче prime_len |
| `test_cmt_validation` | `prepare` валит ValueError для: неверного seed_strategy; custom_seed без path; prime_bars < 1 или > num_bars; total_frames в progression ≠ max_len; frame_per_bar % beats_per_bar ≠ 0. Использует **fake hparams.yaml** в tmp_path — реальные веса для тестов не нужны |
| `test_cmt_prepare` | `prepare` создаёт tmp_dir, записывает seed.npz с правильными ключами и shapes (читая параметры из fake hparams), возвращает dict с правильными ключами; разные progression → разные seed.npz |
| `test_cmt_extract_melody` | вытаскивает 'melody' track из 2-трекового MIDI, валит ValueError если трека нет |
| e2e (Task 13 в plane) | `pipeline.cli generate sample.json` → cmt = ok, плюс две разные progression → разные мелодии |

Тесты не зависят от наличия реального чекпоинта на машине — везде используются fake hparams.yaml в tmp_path с заданными параметрами. Это позволяет CI прогонять тесты без 147 МБ артефакта.

---

## 14. Открытые вопросы / риски

| # | Риск | Митигация |
|---|---|---|
| 1 | Точные значения `SUSTAIN_PITCH_IDX` / `ONSET_RHYTHM_IDX` / `SUSTAIN_RHYTHM_IDX` подтверждены diagnostic'ом на `seed_instance.pkl`: **`ONSET_RHYTHM_IDX = 2`**, **`SUSTAIN_RHYTHM_IDX = 0`**, **`SUSTAIN_PITCH_IDX = 49`**. Зафиксировать константы в `_cmt_input.py` с комментарием-ссылкой на pkl-данные | Учтено в реализации |
| 2 | `model.sampling` использует `torch.multinomial` → недетерминирован | Принимаем. e2e-проверка «разные progression → разные мелодии» работает независимо от детерминизма (сравнивается только seed.npz, а не выход модели) |
| 3 | Smoke-чекпоинт 5 эпох даёт хаотичную мелодию | Принимаем. DoD проверяет валидность MIDI и зависимость от условий, не музыкальное качество |
| 4 | На чужой машине `result/smoke_wjazzd_5epochs/*` отсутствует | `pipeline.cli generate` упадёт с понятной ошибкой `FileNotFoundError` от runner'а или адаптера. Документируется как pre-requisite |
| 5 | Если будущий чекпоинт обучен с `num_pitch ≠ 50` (другое количество спец-токенов) — наши `SUSTAIN_PITCH_IDX` могут стать невалидны | Это редкий риск (изменение CMT-кода). Если случится — обновить константы в `_cmt_input.py` вручную. Не пытаемся это автоматизировать |
| 6 | Семантика `chord[0]` vs `chord[max_len]` в CMT — подтверждена diagnostic'ом: **все 129 фреймов содержат реальные данные**, нет zero-padding. Последний фрейм (`chord[max_len]`) — продолжение последнего реального аккорда. Конвертер разворачивает progression на `max_len = num_bars × frame_per_bar` фреймов и дублирует последний фрейм для index `max_len` | Учтено в реализации |
