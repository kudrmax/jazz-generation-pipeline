#!/usr/bin/env python3
"""CMT runner: запускается интерпретатором models/CMT-pytorch/.venv/bin/python.

Контракт:
- читает JSON payload со stdin (см. pipeline.runner_protocol)
- params: checkpoint_path, hparams_path, model_repo_path, seed_npz_path,
          output_midi_path, topk, device
- импортирует CMT-API напрямую (без CLI), вызывает model.sampling и pitch_to_midi
- пишет MIDI в output_midi_path
- exit 0 при успехе, exit 1 при ошибке (traceback в stderr)

НЕ импортирует ничего из pipeline.* — живёт в CMT-venv.
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
    checkpoint_path = Path(params["checkpoint_path"])
    hparams_path    = Path(params["hparams_path"])
    repo            = Path(params["model_repo_path"])
    seed_npz_path   = Path(params["seed_npz_path"])
    output_midi     = Path(params["output_midi_path"])
    topk: int       = int(params["topk"])
    device_name: str = params["device"]

    # Импорты CMT требуют cwd = CMT_REPO и PYTHONPATH = CMT_REPO,
    # потому что они делают `from model import ...`, `from layers import ...` etc.
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    import yaml
    from model import ChordConditionedMelodyTransformer
    from utils.utils import pitch_to_midi

    device = torch.device(device_name)

    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)
    model_config = hparams["model"]

    model = ChordConditionedMelodyTransformer(**model_config).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    npz = np.load(seed_npz_path)
    chord       = torch.tensor(npz["chord_chroma"]).float().unsqueeze(0).to(device)
    prime_pitch = torch.tensor(npz["prime_pitch"]).long().unsqueeze(0).to(device)
    prime_rhythm = torch.tensor(npz["prime_rhythm"]).long().unsqueeze(0).to(device)

    with torch.no_grad():
        result = model.sampling(prime_rhythm, prime_pitch, chord, topk=topk)

    pitch_idx = result["pitch"][0].cpu().numpy()
    chord_arr = chord[0].cpu().numpy()

    output_midi.parent.mkdir(parents=True, exist_ok=True)
    pitch_to_midi(
        pitch_idx, chord_arr,
        frame_per_bar=model.frame_per_bar,
        save_path=str(output_midi),
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
