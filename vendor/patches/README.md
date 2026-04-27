# Patches

Архив патчей к внешним репозиториям. Долгосрочно эти патчи живут в наших fork'ах (через git submodule); файлы здесь — историческая копия для случая если submodule станет недоступен.

## MINGUS-py312-numpy2.patch

**Текущий статус:** ✅ применён в наш fork `https://github.com/kudrmax/MINGUS` (коммит `22cf61b0`). Подключается через git submodule в `models/MINGUS`.

**Upstream:** `https://github.com/vincenzomadaghiele/MINGUS.git`
**Base commit (поверх которого патч):** `3c4ac1210c6b09cc9ed8f904a0bf49336a4fd5af`

**Что патчит:**

1. `A_preprocessData/data_preprocessing.py` — try/except вокруг `xmlToStructuredSong`. Без этого preprocessing падает на отдельных XML с пустыми chord tracks.
2. `B_train/loadDB.py` — `np.array(..., dtype=object)` для всех ragged массивов. Без этого numpy 2.x не позволяет implicit ragged → требует явный dtype.

**Окружение (под которое патчилось):**

```
Python 3.12
torch==2.11
numpy==2.4.4
music21==6.7.1
pretty_midi==0.2.11
note-seq
```
