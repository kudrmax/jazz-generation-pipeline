import json
from pathlib import Path
from unittest.mock import patch

import pretty_midi

from pipeline.cli import main


def _write_sample_progression(p: Path) -> None:
    p.write_text(json.dumps({
        "tempo": 120.0,
        "time_signature": "4/4",
        "chords": [["Cmaj7", 4], ["Am7", 4]],
    }))


def test_cli_generate_prints_table(tmp_path: Path, capsys, monkeypatch, fake_melody_instrument):
    src = tmp_path / "p.json"
    _write_sample_progression(src)

    monkeypatch.setattr("pipeline.pipeline.OUTPUT_ROOT", tmp_path)

    fake_raw_midi = tmp_path / "fake_raw.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(fake_melody_instrument)
    pm.write(str(fake_raw_midi))
    monkeypatch.setattr("pipeline.pipeline._run_model_subprocess", lambda *a, **kw: fake_raw_midi)

    rc = main(["generate", str(src)])
    assert rc == 0
    out = capsys.readouterr().out
    for model in ["mingus", "bebopnet", "ec2vae", "cmt", "commu", "polyffusion"]:
        assert model in out
    assert "ok" in out
    assert "not implemented" in out


def test_cli_generate_invalid_path_exits_nonzero(tmp_path: Path, capsys):
    rc = main(["generate", str(tmp_path / "missing.json")])
    assert rc != 0
