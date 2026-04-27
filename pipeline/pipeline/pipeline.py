from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from pipeline.config import (
    ADAPTERS, MELODY_PROGRAM, MODEL_NAMES,
    MODEL_RUNNER_SCRIPT, MODEL_VENV_PYTHON, OUTPUT_ROOT, RUNNER_TIMEOUT_SEC,
)
from pipeline.postprocess import postprocess
from pipeline.progression import ChordProgression
from pipeline.runner_protocol import RunnerError, run_runner_subprocess


def make_run_id(progression: ChordProgression) -> str:
    """Run-id формат: YYYYMMDD-HHMMSS-<8charhash>. Хвост — детерминирован от progression."""
    payload = repr((progression.chords, progression.tempo, progression.time_signature)).encode()
    h = hashlib.sha256(payload).hexdigest()[:8]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{h}"


def _run_model_subprocess(
    model: str, params: dict, run_id: str, model_tmp: Path,
) -> Path:
    payload = {"model": model, "run_id": run_id, "params": params}
    return run_runner_subprocess(
        venv_python=MODEL_VENV_PYTHON[model],
        runner_script=MODEL_RUNNER_SCRIPT[model],
        payload=payload,
        tmp_dir=model_tmp,
        timeout_sec=RUNNER_TIMEOUT_SEC,
    )


def generate_all(
    progression: ChordProgression,
    run_id: str | None = None,
) -> dict[str, dict]:
    """Один progression → набор MIDI от всех моделей.

    Для каждой модели либо `{"melody_only": Path, "with_chords": Path}`,
    либо `{"error": str}` если adapter — stub или runner упал.
    """
    run_id = run_id or make_run_id(progression)
    tmp_root = OUTPUT_ROOT / "_tmp" / run_id
    tmp_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
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
