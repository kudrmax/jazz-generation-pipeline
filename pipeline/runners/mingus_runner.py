#!/usr/bin/env python3
"""MINGUS runner: запускается интерпретатором models/MINGUS/.venv/bin/python.

Контракт:
- читает JSON payload со stdin (см. pipeline.runner_protocol)
- params: input_xml_path, output_midi_path, checkpoint_epochs, temperature, device, model_repo_path
- импортирует MINGUS gen-функции напрямую (без CLI), вызывает их с нашими параметрами
- пишет MIDI в output_midi_path
- exit 0 при успехе, exit 1 при ошибке (traceback в stderr)

Этот файл НЕ должен импортировать ничего из pipeline.* — он живёт в MINGUS-venv.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    payload = json.loads(sys.stdin.read())
    params = payload["params"]
    input_xml = Path(params["input_xml_path"])
    output_midi = Path(params["output_midi_path"])
    epochs: int = int(params["checkpoint_epochs"])
    temperature: float = float(params["temperature"])
    device_name: str = params["device"]
    repo: Path = Path(params["model_repo_path"])

    # Импорты MINGUS требуют cwd = MINGUS_REPO и PYTHONPATH = MINGUS_REPO,
    # потому что они делают `import B_train.loadDB` и т.п.
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    import B_train.loadDB as dataset
    import B_train.MINGUS_model as mod
    import C_generate.gen_funct as gen

    device = torch.device(device_name)
    torch.manual_seed(1)

    # Аргументы как в C_generate/generate.py — нужны те же чтобы veca совпали с обученными.
    COND = "I-C-NC-B-BE-O"
    TRAIN_BATCH_SIZE = 20
    EVAL_BATCH_SIZE = 10
    BPTT = 35
    AUGMENTATION = False
    SEGMENTATION = True
    AUGMENTATION_CONST = 3
    NUM_CHORUS = 3

    music_db = dataset.MusicDB(
        device, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
        BPTT, AUGMENTATION, SEGMENTATION, AUGMENTATION_CONST,
    )
    structured_songs = music_db.getStructuredSongs()
    vocab_pitch, vocab_duration, vocab_beat, vocab_offset = music_db.getVocabs()
    pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = music_db.getInverseVocabs()
    db_chords, db_to_music21, db_to_chord_composition, db_to_midi_chords = music_db.getChordDicts()

    def _build_model(is_pitch: bool):
        if is_pitch:
            pitch_embed_dim = 512
            duration_embed_dim = 512
            chord_encod_dim = 64
            next_chord_encod_dim = 32
            beat_embed_dim = 64
            bass_embed_dim = 64
            offset_embed_dim = 32
        else:
            pitch_embed_dim = 64
            duration_embed_dim = 64
            chord_encod_dim = 64
            next_chord_encod_dim = 32
            beat_embed_dim = 32
            bass_embed_dim = 32
            offset_embed_dim = 32
        emsize = 200
        nhid = 200
        nlayers = 4
        nhead = 4
        dropout = 0.2
        m = mod.TransformerModel(
            len(vocab_pitch), pitch_embed_dim,
            len(vocab_duration), duration_embed_dim,
            bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
            len(vocab_beat), beat_embed_dim,
            len(vocab_offset), offset_embed_dim,
            emsize, nhead, nhid, nlayers,
            pitch_to_ix["<pad>"], duration_to_ix["<pad>"], beat_to_ix["<pad>"], offset_to_ix["<pad>"],
            device, dropout, is_pitch, COND,
        ).to(device)
        return m

    model_pitch = _build_model(is_pitch=True)
    model_duration = _build_model(is_pitch=False)

    pitch_ckpt = repo / "B_train" / "models" / "pitchModel" / f"MINGUS COND {COND} Epochs {epochs}.pt"
    duration_ckpt = repo / "B_train" / "models" / "durationModel" / f"MINGUS COND {COND} Epochs {epochs}.pt"
    model_pitch.load_state_dict(torch.load(str(pitch_ckpt), map_location=device))
    model_duration.load_state_dict(torch.load(str(duration_ckpt), map_location=device))

    tune, _wjazz_to_music21, _wjazz_to_midi_chords, _wjazz_to_chord_composition, _wjazz_chords = (
        gen.xmlToStructuredSong(
            str(input_xml),
            db_to_music21, db_to_midi_chords, db_to_chord_composition, db_chords,
        )
    )
    is_jazz = False
    new_song = gen.generateOverStandard(
        tune, NUM_CHORUS, temperature,
        model_pitch, model_duration, db_to_midi_chords,
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix,
        vocab_pitch, vocab_duration,
        BPTT, device, is_jazz,
    )

    pm = gen.structuredSongsToPM(new_song, db_to_midi_chords, is_jazz)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_midi))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
