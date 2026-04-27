from __future__ import annotations

import json
import subprocess
from pathlib import Path


class RunnerError(RuntimeError):
    """Subprocess модели завершился с ошибкой или не положил output MIDI."""


def run_runner_subprocess(
    venv_python: str | Path,
    runner_script: str | Path,
    payload: dict,
    tmp_dir: Path,
    timeout_sec: int,
) -> Path:
    """Запускает runner-скрипт интерпретатором model-venv'а.

    1. Пишет JSON payload в stdin.
    2. Перехватывает stdout/stderr → tmp_dir/{stdout,stderr}.log.
    3. Если exit ≠ 0 → RunnerError со stderr-tail.
    4. Если exit == 0 но params.output_midi_path не существует → RunnerError.
    5. Возвращает Path к output MIDI.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    expected_midi = Path(payload["params"]["output_midi_path"])

    result = subprocess.run(
        [str(venv_python), str(runner_script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    (tmp_dir / "stdout.log").write_text(result.stdout)
    (tmp_dir / "stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-20:])
        raise RunnerError(
            f"runner {runner_script} exited with {result.returncode}:\n{tail}"
        )
    if not expected_midi.exists():
        raise RunnerError(
            f"runner {runner_script} exited 0 but output MIDI not found at {expected_midi}"
        )
    return expected_midi
