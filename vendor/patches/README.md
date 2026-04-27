# Patches

Патчи к внешним репозиториям, которые нужно применить чтобы код работал на нашей машине (Python 3.12, numpy 2.x).

Эта папка — **временная страховка**. Долгосрочное решение — git submodule на наш fork соответствующего репо, в котором эти патчи уже закоммичены.

## MINGUS-py312-numpy2.patch

**Upstream:** `https://github.com/vincenzomadaghiele/MINGUS.git`
**Base commit:** `3c4ac1210c6b09cc9ed8f904a0bf49336a4fd5af`

**Что патчит:**

1. `A_preprocessData/data_preprocessing.py` — try/except вокруг `xmlToStructuredSong`. Без этого preprocessing падает на отдельных XML с пустыми chord tracks.
2. `B_train/loadDB.py` — `np.array(..., dtype=object)` для всех ragged массивов. Без этого numpy 2.x не позволяет implicit ragged → требует явный dtype.

**Как применять (если нет нашего fork'а с уже зашитыми патчами):**

```bash
cd models/MINGUS
git checkout 3c4ac1210c6b09cc9ed8f904a0bf49336a4fd5af
git apply /path/to/diploma/vendor/patches/MINGUS-py312-numpy2.patch
```

**Окружение (которое мы тестировали):**

```
Python 3.12
torch==2.11
numpy==2.4.4
music21==6.7.1
pretty_midi==0.2.11
note-seq
```
