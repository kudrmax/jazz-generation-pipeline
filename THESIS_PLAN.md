# План ВКР — финальный список моделей и методология

> Этот файл фиксирует **финальный научный план ВКР** после серии исследований.
> Технический дизайн пайплайна — `docs/superpowers/specs/2026-04-27-pipeline-mingus-design.md`.

---

## Полная таблица моделей

| # | Модель | 1. Аккорды на вход | 2. Жанр сейчас (наш доступ) | 3. Что делать для джаза |
|---|---|:-:|---|---|
| 1 | **BebopNet** | ✅ через chord track в XML | **Джаз** (bebop sax solos: Charlie Parker, Sonny Stitt и др.) | **ничего** |
| 2 | **MINGUS** | ✅ через `<harmony>` в MusicXML | **Джаз** (WJazzD) | **ничего** |
| 3 | **Jazz Transformer** | ❌ **Нет** — unconditional по архитектуре | Джаз (WJazzD) | **Не применимо** (архитектура не поддерживает chord-input). Только как unconditional baseline |
| 4 | **EC²-VAE** | ✅ через chord chroma 12-D per beat | Никакой осмысленной (pretrained Nottingham/POP909 публично нет) | **Обучить с нуля** на WJazzD |
| 5 | **CMT** | ✅ через chord vector 12-D per frame | Никакой осмысленной (pretrained K-POP в репо нет) | **Обучить с нуля** на WJazzD |
| 6 | **ComMU** | ✅ через `--chord_progression "C-C-E-E"` token sequence | **New-age / cinematic** (POZAlabs датасет, pretrained есть) | **И дообучить, и с нуля — сравнить** |
| 7 | **Polyffusion** | ✅ через chord matrix 8-bar | **Pop** (POP909 pretrained, работает) | **И дообучить, и с нуля — сравнить** |
| 8 | ~~Genchel~~ | ✅ | ~~LSTM на FolkDB+BebopDB~~ | **Выкидываем** — датасет недоступен (Dropbox 404) |

---

## Группировка по стратегии для ВКР

### 🟢 In-domain pretrained (ничего не обучаем)
- **BebopNet, MINGUS** — джазовые baselines, pretrained веса публично доступны.

### 🔴 Не chord-conditioned
- **Jazz Transformer** — unconditional по архитектуре. В основном сравнении не участвует.
  Использовать как unconditional baseline («что генерируется без chord-conditioning»).

### 🟡 Только train с нуля (pretrained на не-джазе не выложен публично)
- **EC²-VAE** — train с нуля на WJazzD
- **CMT** — train с нуля на WJazzD

### 🔵 И дообучить, и с нуля — сравнить (есть готовый pretrained на не-джазе)
- **ComMU** (pretrained на new-age/cinematic) → fine-tune на WJazzD + train с нуля на WJazzD
- **Polyffusion** (pretrained на POP909) → fine-tune на WJazzD + train с нуля на WJazzD

---

## Финальная таблица сравнения для ВКР

| Модель | Архитектура | Источник весов |
|---|---|---|
| BebopNet | LSTM/Transformer-XL | in-domain pretrained на bebop |
| MINGUS | Seq2Seq Transformer | in-domain pretrained на WJazzD |
| EC²-VAE (from-scratch) | Conditional VAE | train с нуля на WJazzD |
| CMT (from-scratch) | Transformer | train с нуля на WJazzD |
| ComMU (from-scratch) | Transformer-XL+REMI | train с нуля на WJazzD |
| ComMU (fine-tune) | Transformer-XL+REMI | new-age pretrained → fine-tune на WJazzD |
| Polyffusion (from-scratch) | Latent Diffusion | train с нуля на WJazzD |
| Polyffusion (fine-tune) | Latent Diffusion | POP909 pretrained → fine-tune на WJazzD |

**Итого: 8 экспериментальных конфигураций (6 уникальных моделей).**

Опционально-вспомогательные: Jazz Transformer (unconditional baseline).

---

## Научный вклад ВКР — два ортогональных угла

### 1. Архитектурный sweep
Сравнение разных архитектурных классов на одной chord-conditioned задаче генерации джазовых соло:
- **RNN / LSTM** (BebopNet)
- **Transformer-XL** (MINGUS, ComMU)
- **Transformer encoder-decoder** (CMT)
- **VAE** (EC²-VAE)
- **Diffusion** (Polyffusion)

### 2. Transfer learning vs from-scratch
Сравнение двух режимов обучения для двух не-джазовых моделей:
- **From-scratch** на WJazzD vs **Fine-tune** не-джазового pretrained на WJazzD
- Применимо к ComMU (new-age) и Polyffusion (pop)

Ответ на вопрос: **полезно ли pretraining на не-джазовом домене для джазовой задачи?**

---

## Главный нарратив

Это **не просто «сравнили 4 модели как у других»** — это **методологический вклад**:

1. Унифицированный экспериментальный протокол по 6 разным архитектурным классам генерации монофонных джазовых соло
2. Первое исследование влияния pretraining-домена (не-джаз → джаз) на качество chord-conditioned джазовой генерации
3. Прозрачная фиксация ограничений воспроизводимости: какие модели из литературы заявлены как chord-conditioned, но кода/данных не имеют

---

## Что было исключено из ВКР и почему

| Модель | Причина исключения |
|---|---|
| **MelodyDiffusion** (Li, Sung 2023) | Публичного кода не существует. Подтверждено deep research |
| **JazzGAN** (Trieu, Keller 2018) | Публичный репозиторий — форк SeqGAN без работающего chord-conditioned API |
| **Jazz Transformer** (Wu, Yang 2020) | Архитектурно unconditional — не chord-conditioned. Подтверждено цитатами из статьи и анализом кода |
| **ImprovNet** (Bhandari et al. 2025) | Conditioning на genre token + corruption, не на аккорды |
| **Genchel et al.** (2019) | Датасет (Dropbox) недоступен (404). Альтернатив не нашлось |

Сильный научный нарратив: **«проверили 12+ моделей из литературы, 6 реально работают по нашему протоколу, 5+ оказались либо неприменимыми, либо невоспроизводимыми»**.

---

## Текущий статус реализации в pipeline

**`pipeline/`** в master — единая точка генерации MIDI по аккордовой прогрессии. Подробности — `README.md`.

✅ **Реализовано полностью:**
- **MINGUS** — adapter (`pipeline/pipeline/adapters/mingus.py`) + runner (`pipeline/runners/mingus_runner.py`). Работает с pretrained WJazzD-весами. Pre-requisites: `models/MINGUS/.venv` + `DATA.json` + чекпоинты Epochs 100.

🔴 **Заглушки в pipeline** (adapter возвращает `NotImplementedError`):
- BebopNet — нужен adapter+runner поверх их `generate_from_xml.py` + получить чекпоинт публично
- EC²-VAE — нужен train с нуля на WJazzD, потом adapter+runner
- CMT — то же
- ComMU — нужен train (from-scratch + fine-tune), потом adapter+runner
- Polyffusion — нужен train (from-scratch + fine-tune), потом adapter+runner

🚫 **Окончательно исключено:** Genchel, MelodyDiffusion, JazzGAN, ImprovNet, Jazz Transformer (см. таблицу выше).

---

## Подробный разбор каждой модели

### 1. MINGUS ✅ — реализована

**Архитектура.** Seq2Seq Transformer с двумя декодерами (pitch + duration). Conditioning на C+NC+B+BE+O (Chord+NextChord+Bass+BassEvolution+Offset).

**Вход.** MusicXML с прописанными `<harmony>` тегами (chord progression) **и** мелодической партией (тема, используется как seed для авторегрессии).

**Выход.** MIDI с двумя треками: `Tenor Sax` (program 67, генерированная мелодия) + `piano` (program 1, chord track из standard).

**Jazz training.** ✅ in-domain. Pretrained на WJazzD (Weimar Jazz Database).

**Pipeline-интеграция.** Полная: adapter генерирует input.xml через music21 (3 стратегии затравочной мелодии), runner вызывает MINGUS gen-функции и возвращает raw MIDI, postprocess извлекает Tenor Sax трек и склеивает с нашим chord_render.

---

### 2. BebopNet 🔴 — заглушка

**Архитектура.** Hybrid: Transformer-XL (autoregressive language model на последовательности нот) + LSTM tail.

**Вход.** XML с темой и chord track. Тема используется как seed для авторегрессии. Их CLI: `--song fly` → предустановленный XML standard (Fly Me To The Moon).

**Выход.** Single-track MIDI (Tenor Sax program 65).

**Jazz training.** ✅ in-domain. Pretrained на 284 bebop sax solos.

**Что нужно для подключения:**
1. Реализовать `BebopNetAdapter` — генерация XML с минимальной темой, парсинг output MIDI
2. Реализовать `bebopnet_runner.py` — обёртка над их `generate_from_xml.py` в их venv
3. Проверить совместимость pretrained весов (репо есть, веса в составе)

---

### 3. EC²-VAE 🔴 — заглушка

**Архитектура.** Conditional VAE: GRU encoder + два GRU decoder (pitch decoder + rhythm decoder).

**Вход.** 12-D chord chroma per beat (32 timesteps × 12) + случайный латент `z ~ N(0, I)`. **Тема не нужна** — латент стохастический.

**Выход.** Pianoroll (32 × 130: 128 pitch + hold + rest) → MIDI (1 трек).

**Jazz training.** ⚠️ pretrained Nottingham/POP909 публично нет. Нужен train с нуля.

**Что нужно для подключения:**
1. Train с нуля на полном WJazzD (21K сегментов, 30+ эпох) — Colab T4 / RunPod
2. Реализовать `EC2VaeAdapter` — построение chord chroma из ChordProgression, postprocess pianoroll → Instrument
3. Реализовать `ec2vae_runner.py`

---

### 4. CMT 🔴 — заглушка

**Архитектура.** Transformer: BLSTM chord encoder + два self-attention декодера (rhythm + pitch).

**Вход.** Для каждого 8-bar сегмента: chord vector (96 frames × 12) + prime rhythm (16 frames) + prime pitch (16 frames). **Требует короткую затравку** (16 frames pitch+rhythm).

**Выход.** Pitch sequence (50 vocab) + rhythm sequence (3 vocab) → MIDI с двумя треками (generated melody + ground truth chord accompaniment).

**Jazz training.** ⚠️ pretrained K-POP в репо нет. Нужен train с нуля на WJazzD.

**Что нужно для подключения:**
1. Train с нуля на WJazzD (1499+ instances, 50+ эпох) — Colab T4
2. Реализовать `CMTAdapter` — построение chord vector + prime tokens, postprocess
3. Реализовать `cmt_runner.py`

---

### 5. ComMU 🔴 — заглушка

**Архитектура.** Transformer-XL поверх REMI tokens с metadata.

**Вход.** REMI-токены + 12 metadata токенов (BPM, key, instrument, chord_progression в формате `Am-Am-F-F`, и т.д.). Поддерживает prime sequence для авторегрессии.

**Выход.** REMI tokens → MIDI через их декодер. С `track_role=main_melody` — monophonic мелодия.

**Jazz training.** Pretrained на new-age/cinematic (POZAlabs). На джазе ничего не делали.

**Что нужно для подключения:**
1. Конвертер WJazzD → их REMI формат + metadata CSV
2. Train: from-scratch + fine-tune от их pretrained — RunPod
3. Реализовать `ComMUAdapter` + `commu_runner.py`

---

### 6. Polyffusion 🔴 — заглушка

**Архитектура.** Latent Diffusion + cross-attention на chord encoder (POLYDIS). **Полифонический** на выходе.

**Вход.** Chord matrix 8-bar через chord encoder. **Тема не нужна** — латент шум диффузии стохастический.

**Выход.** 32×128 piano-roll → MIDI (полифоничен; для нашего pipeline извлекаем мелодию через highest-pitch projection).

**Jazz training.** Pretrained на POP909. На джазе ничего не делали.

**Что нужно для подключения:**
1. Конвертер WJazzD → prmat2c формат
2. Train: fine-tune от POP909 + from-scratch — RunPod (диффузия дорогая)
3. Реализовать `PolyffusionAdapter` (включая highest-pitch projection в `extract_melody`) + `polyffusion_runner.py`
