from pathlib import Path
from unittest.mock import MagicMock, patch

import pretty_midi
import pytest

from pipeline.progression import ChordProgression
from pipeline.pipeline import generate_all, make_run_id


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_make_run_id_format():
    rid = make_run_id(_basic_progression())
    parts = rid.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 8 and parts[0].isdigit()  # YYYYMMDD
    assert len(parts[1]) == 6 and parts[1].isdigit()  # HHMMSS
    assert len(parts[2]) == 8                         # 8charhash


def test_make_run_id_hash_deterministic_for_same_progression():
    p = _basic_progression()
    a = make_run_id(p).split("-")[2]
    b = make_run_id(p).split("-")[2]
    assert a == b


def test_make_run_id_hash_different_for_different_progressions():
    p1 = ChordProgression(chords=[("Cmaj7", 4)], tempo=120.0, time_signature="4/4")
    p2 = ChordProgression(chords=[("Dm7", 4)],   tempo=120.0, time_signature="4/4")
    a = make_run_id(p1).split("-")[2]
    b = make_run_id(p2).split("-")[2]
    assert a != b


def test_generate_all_returns_dict_with_all_models(tmp_path: Path, fake_melody_instrument, monkeypatch):
    """С моком adapter+runner: generate_all возвращает 1 ok + 5 stub-errors."""
    monkeypatch.setattr("pipeline.pipeline.OUTPUT_ROOT", tmp_path)

    fake_raw_midi = tmp_path / "fake_raw.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(fake_melody_instrument)
    pm.write(str(fake_raw_midi))

    def _fake_run(model, params, run_id, model_tmp):
        # Имитируем что MINGUS-runner написал raw MIDI
        return fake_raw_midi

    monkeypatch.setattr("pipeline.pipeline._run_model_subprocess", _fake_run)

    results = generate_all(_basic_progression(), run_id="20260427-000000-deadbeef")
    assert set(results.keys()) == {"mingus", "bebopnet", "ec2vae", "cmt", "commu", "polyffusion"}
    for model in ["mingus", "bebopnet"]:
        assert "melody_only" in results[model]
        assert "with_chords" in results[model]
    # CMT после Task 8 — prepare уже работает, но 2-такта прогрессия не подходит
    # для модели с 8 bars × 16 fpb → adapter validation error. Стаб-модели
    # по-прежнему не реализованы.
    for model in ["ec2vae", "commu", "polyffusion"]:
        assert "error" in results[model]
        assert "not implemented" in results[model]["error"]
    assert "error" in results["cmt"]


def test_generate_all_creates_tmp_dir(tmp_path: Path, fake_melody_instrument, monkeypatch):
    monkeypatch.setattr("pipeline.pipeline.OUTPUT_ROOT", tmp_path)
    fake_raw_midi = tmp_path / "fake_raw.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(fake_melody_instrument)
    pm.write(str(fake_raw_midi))
    monkeypatch.setattr("pipeline.pipeline._run_model_subprocess", lambda *a, **kw: fake_raw_midi)

    rid = "20260427-000000-cafebabe"
    generate_all(_basic_progression(), run_id=rid)
    assert (tmp_path / "_tmp" / rid).exists()


def test_run_model_subprocess_raises_runner_error_when_runner_script_missing(tmp_path: Path, monkeypatch):
    """Если в MODEL_RUNNER_SCRIPT нет ключа — должна быть понятная RunnerError, не KeyError."""
    from pipeline.pipeline import _run_model_subprocess
    from pipeline.runner_protocol import RunnerError

    monkeypatch.setattr("pipeline.pipeline.MODEL_RUNNER_SCRIPT", {})
    with pytest.raises(RunnerError, match="runner script not registered"):
        _run_model_subprocess(
            "bebopnet", {"output_midi_path": str(tmp_path / "x.mid")}, "rid", tmp_path,
        )
