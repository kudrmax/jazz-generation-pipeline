from __future__ import annotations

import shutil
from pathlib import Path

from music21 import chord as m21_chord, harmony, instrument, key, meter, note, stream, tempo

from pipeline.adapters.mingus import MingusPipelineConfig
from pipeline.chord_vocab import parse_chord, ROOTS
from pipeline.progression import ChordProgression


_TONIC_OCTAVE = 5  # С5 = MIDI 72; для саксофона средне-высокий регистр


def build_mingus_xml(
    progression: ChordProgression,
    config: MingusPipelineConfig,
    out_path: Path,
) -> None:
    """Пишет MusicXML, готовый к скармливанию MINGUS gen.xmlToStructuredSong.

    seed_strategy:
      - tonic_whole    — 1 whole-нота тоники в каждом баре
      - tonic_quarters — 4 quarter-ноты тоники в каждом баре
      - custom_xml     — копирует config.custom_xml_path в out_path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if config.seed_strategy == "custom_xml":
        if config.custom_xml_path is None:
            raise ValueError("seed_strategy=custom_xml requires custom_xml_path")
        shutil.copy(config.custom_xml_path, out_path)
        return

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
    part.insert(0, instrument.TenorSaxophone())

    measure_idx = 1
    if not progression.chords:
        raise ValueError("progression has no chords")
    chord_iter = iter(progression.chords)
    cur_chord, cur_remaining = next(chord_iter)
    for bar in range(progression.num_bars()):
        m = stream.Measure(number=measure_idx)
        if measure_idx == 1:
            m.append(meter.TimeSignature(progression.time_signature))
            m.append(tempo.MetronomeMark(number=progression.tempo))
            m.append(key.KeySignature(0))  # C-мажор
        # положим chord-symbol в начало бара
        cs = harmony.ChordSymbol(cur_chord)
        m.insert(0, cs)
        # положим затравочные ноты
        root_idx, _quality = parse_chord(cur_chord)
        tonic_pitch = note.Pitch()
        tonic_pitch.midi = root_idx + 12 * (_TONIC_OCTAVE + 1)
        if config.seed_strategy == "tonic_whole":
            n = note.Note(tonic_pitch)
            n.quarterLength = bpb
            m.append(n)
        elif config.seed_strategy == "tonic_quarters":
            for _ in range(bpb):
                n = note.Note(tonic_pitch)
                n.quarterLength = 1
                m.append(n)
        else:
            raise ValueError(f"unsupported seed_strategy: {config.seed_strategy}")
        part.append(m)
        # забираем beats из текущего аккорда
        cur_remaining -= bpb
        if cur_remaining <= 0 and bar < progression.num_bars() - 1:
            cur_chord, cur_remaining = next(chord_iter)
        measure_idx += 1

    # Sanity check: chord iterator должен быть полностью исчерпан
    remaining = list(chord_iter)
    assert not remaining, (
        f"chord iterator has {len(remaining)} unused chords after "
        f"{progression.num_bars()} bars; total_beats validation should have caught this"
    )

    score.insert(0, part)
    score.write("musicxml", fp=str(out_path))
