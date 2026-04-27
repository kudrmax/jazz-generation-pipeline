from __future__ import annotations

from pathlib import Path

import pretty_midi

from pipeline.adapters.base import ModelAdapter
from pipeline.progression import ChordProgression


class CMTAdapter(ModelAdapter):
    def prepare(self, progression: ChordProgression, tmp_dir: Path) -> dict:
        raise NotImplementedError("model cmt: adapter not implemented")

    def extract_melody(self, raw_midi_path: Path) -> pretty_midi.Instrument:
        raise NotImplementedError("model cmt: adapter not implemented")
