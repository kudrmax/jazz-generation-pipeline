import pretty_midi
import pytest


@pytest.fixture
def fake_melody_instrument():
    inst = pretty_midi.Instrument(program=99, name="Tenor Sax")
    for i, p in enumerate([60, 64, 67, 71]):
        inst.notes.append(pretty_midi.Note(80, p, i * 0.5, (i + 1) * 0.5))
    return inst
