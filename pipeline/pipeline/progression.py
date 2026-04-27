from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


_ALLOWED_KEYS = {"chords", "tempo", "time_signature"}


@dataclass
class ChordProgression:
    """Композиционный замысел: только аккорды + tempo + размер.

    Никакой модель-специфики (затравки, температуры, чекпоинтов) здесь нет."""

    chords: list[tuple[str, int]] = field(default_factory=list)
    tempo: float = 120.0
    time_signature: str = "4/4"

    def __post_init__(self) -> None:
        if self.tempo <= 0:
            raise ValueError(f"tempo must be > 0, got {self.tempo}")

    @classmethod
    def from_json(cls, path: Path) -> "ChordProgression":
        data = json.loads(Path(path).read_text())
        unknown = set(data.keys()) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"unknown ChordProgression fields: {sorted(unknown)}. "
                f"allowed: {sorted(_ALLOWED_KEYS)}"
            )
        chords = [tuple(c) for c in data["chords"]]
        return cls(
            chords=chords,
            tempo=float(data.get("tempo", 120.0)),
            time_signature=data.get("time_signature", "4/4"),
        )

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "tempo": self.tempo,
            "time_signature": self.time_signature,
            "chords": [list(c) for c in self.chords],
        }, indent=2))

    def total_beats(self) -> int:
        return sum(d for _, d in self.chords)

    def beats_per_bar(self) -> int:
        num, _denom = self.time_signature.split("/")
        return int(num)

    def num_bars(self) -> int:
        bpb = self.beats_per_bar()
        if self.total_beats() % bpb != 0:
            raise ValueError(
                f"total beats {self.total_beats()} not divisible by {bpb} (time signature {self.time_signature})"
            )
        return self.total_beats() // bpb
