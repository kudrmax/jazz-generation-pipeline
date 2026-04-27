from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from music21 import (
    chord as m21_chord, harmony, instrument, key, meter, note, stream, tempo,
)

from pipeline.chord_vocab import parse_chord
from pipeline.progression import ChordProgression


_TONIC_OCTAVE = 5  # C5 = MIDI 72; средне-высокий регистр сакса


SeedStrategy = Literal["tonic_whole", "tonic_quarters", "custom_xml"]


def _instrument_for(name: str):
    """Возвращает music21-инструмент по имени.

    Поддерживаются основные имена использующиеся в pipeline. Для остальных —
    fallback на TenorSaxophone (это и так используется обоими моделями).
    """
    name_norm = name.lower().replace(" ", "")
    mapping = {
        "tenorsax":         instrument.TenorSaxophone,
        "tenorsaxophone":   instrument.TenorSaxophone,
        "altosax":          instrument.AltoSaxophone,
        "sopranosax":       instrument.SopranoSaxophone,
    }
    cls = mapping.get(name_norm, instrument.TenorSaxophone)
    return cls()


def build_xml(
    progression: ChordProgression,
    seed_strategy: SeedStrategy,
    custom_xml_path: Path | None,
    out_path: Path,
    melody_instrument_name: str = "Tenor Sax",
) -> None:
    """Пишет MusicXML, готовый для music21-парсера (используется MINGUS и BebopNet).

    seed_strategy:
      - tonic_whole    — 1 whole-нота тоники в каждом баре
      - tonic_quarters — 4 quarter-ноты тоники в каждом баре
      - custom_xml     — копирует custom_xml_path в out_path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if seed_strategy == "custom_xml":
        if custom_xml_path is None:
            raise ValueError("seed_strategy=custom_xml requires custom_xml_path")
        shutil.copy(custom_xml_path, out_path)
        return

    if not progression.chords:
        raise ValueError("progression has no chords")

    bpb = progression.beats_per_bar()
    if progression.total_beats() % bpb != 0:
        raise ValueError(
            f"progression total_beats={progression.total_beats()} not divisible by "
            f"beats_per_bar={bpb} (time_signature={progression.time_signature})"
        )

    for chord_str, beats in progression.chords:
        if beats % bpb != 0:
            raise ValueError(
                f"chord {chord_str} duration {beats} not multiple of {bpb}; "
                f"sub-bar chord placement not supported in MVP"
            )

    score = stream.Score()
    score.metadata = score.metadata or None
    part = stream.Part()
    part.id = "P1"
    part.partName = "Melody"
    part.insert(0, _instrument_for(melody_instrument_name))

    measure_idx = 1
    chord_iter = iter(progression.chords)
    cur_chord, cur_remaining = next(chord_iter)
    for bar in range(progression.num_bars()):
        m = stream.Measure(number=measure_idx)
        if measure_idx == 1:
            m.append(meter.TimeSignature(progression.time_signature))
            m.append(tempo.MetronomeMark(number=progression.tempo))
            m.append(key.KeySignature(0))  # C-мажор
        cs = harmony.ChordSymbol(cur_chord)
        m.insert(0, cs)
        root_idx, _quality = parse_chord(cur_chord)
        tonic_pitch = note.Pitch()
        tonic_pitch.midi = root_idx + 12 * (_TONIC_OCTAVE + 1)
        if seed_strategy == "tonic_whole":
            n = note.Note(tonic_pitch)
            n.quarterLength = bpb
            m.append(n)
        elif seed_strategy == "tonic_quarters":
            for _ in range(bpb):
                n = note.Note(tonic_pitch)
                n.quarterLength = 1
                m.append(n)
        else:
            raise ValueError(f"unsupported seed_strategy: {seed_strategy}")
        part.append(m)
        cur_remaining -= bpb
        if cur_remaining <= 0 and bar < progression.num_bars() - 1:
            cur_chord, cur_remaining = next(chord_iter)
        measure_idx += 1

    remaining = list(chord_iter)
    assert not remaining, (
        f"chord iterator has {len(remaining)} unused chords after "
        f"{progression.num_bars()} bars; total_beats validation should have caught this"
    )

    score.insert(0, part)
    score.write("musicxml", fp=str(out_path))
