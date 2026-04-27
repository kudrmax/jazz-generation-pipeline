from pathlib import Path

import pytest

from pipeline.progression import ChordProgression
from pipeline.adapters.base import ModelAdapter
from pipeline.adapters.bebopnet import BebopNetAdapter
from pipeline.adapters.ec2vae import EC2VaeAdapter
from pipeline.adapters.cmt import CMTAdapter
from pipeline.adapters.commu import ComMUAdapter
from pipeline.adapters.polyffusion import PolyffusionAdapter


def test_model_adapter_is_abstract():
    with pytest.raises(TypeError):
        ModelAdapter()  # type: ignore[abstract]


@pytest.mark.parametrize("AdapterCls", [
    BebopNetAdapter, EC2VaeAdapter, CMTAdapter, ComMUAdapter, PolyffusionAdapter,
])
def test_stub_adapter_prepare_raises_not_implemented(tmp_path: Path, AdapterCls):
    adapter = AdapterCls()
    progression = ChordProgression(chords=[("Cmaj7", 4)], tempo=120.0, time_signature="4/4")
    with pytest.raises(NotImplementedError, match="adapter not implemented"):
        adapter.prepare(progression, config=None, tmp_dir=tmp_path)


@pytest.mark.parametrize("AdapterCls", [
    BebopNetAdapter, EC2VaeAdapter, CMTAdapter, ComMUAdapter, PolyffusionAdapter,
])
def test_stub_adapter_extract_melody_raises_not_implemented(tmp_path: Path, AdapterCls):
    adapter = AdapterCls()
    fake_midi = tmp_path / "x.mid"
    with pytest.raises(NotImplementedError, match="adapter not implemented"):
        adapter.extract_melody(fake_midi)
