import json
import os
import stat
import sys
from pathlib import Path

import pytest

from pipeline.runner_protocol import RunnerError, run_runner_subprocess


def _write_runner(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_run_runner_writes_logs_on_success(tmp_path: Path):
    runner = tmp_path / "runner.py"
    out_midi = tmp_path / "raw.mid"
    out_midi_str = str(out_midi).replace('"', '\\"')
    _write_runner(runner, (
        f'import json, sys\n'
        f'data = json.loads(sys.stdin.read())\n'
        f'open("{out_midi_str}", "wb").write(b"FAKE_MIDI")\n'
        f'print("hello stdout")\n'
        f'sys.exit(0)\n'
    ))
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = run_runner_subprocess(
        venv_python=sys.executable,
        runner_script=runner,
        payload={"params": {"output_midi_path": str(out_midi)}},
        tmp_dir=log_dir,
        timeout_sec=10,
    )
    assert result == out_midi
    assert (log_dir / "stdout.log").exists()
    assert (log_dir / "stdout.log").read_text().strip() == "hello stdout"


def test_run_runner_raises_when_exit_nonzero(tmp_path: Path):
    runner = tmp_path / "runner.py"
    _write_runner(runner, (
        'import sys\n'
        'sys.stderr.write("boom\\n")\n'
        'sys.exit(1)\n'
    ))
    with pytest.raises(RunnerError, match="exited with 1"):
        run_runner_subprocess(
            venv_python=sys.executable,
            runner_script=runner,
            payload={"params": {"output_midi_path": str(tmp_path / "noop.mid")}},
            tmp_dir=tmp_path,
            timeout_sec=10,
        )


def test_run_runner_raises_when_midi_not_written(tmp_path: Path):
    runner = tmp_path / "runner.py"
    _write_runner(runner, (
        'import sys\n'
        'sys.exit(0)\n'  # exit 0 но output не создан
    ))
    with pytest.raises(RunnerError, match="output MIDI not found"):
        run_runner_subprocess(
            venv_python=sys.executable,
            runner_script=runner,
            payload={"params": {"output_midi_path": str(tmp_path / "missing.mid")}},
            tmp_dir=tmp_path,
            timeout_sec=10,
        )


def test_run_runner_passes_payload_via_stdin(tmp_path: Path):
    runner = tmp_path / "runner.py"
    out_midi = tmp_path / "raw.mid"
    out_midi_str = str(out_midi).replace('"', '\\"')
    _write_runner(runner, (
        'import json, sys\n'
        'data = json.loads(sys.stdin.read())\n'
        'assert data["model"] == "test"\n'
        'assert data["params"]["foo"] == 42\n'
        f'open("{out_midi_str}", "wb").write(b"OK")\n'
        'sys.exit(0)\n'
    ))
    result = run_runner_subprocess(
        venv_python=sys.executable,
        runner_script=runner,
        payload={"model": "test", "run_id": "rid", "params": {"foo": 42, "output_midi_path": str(out_midi)}},
        tmp_dir=tmp_path,
        timeout_sec=10,
    )
    assert result == out_midi
