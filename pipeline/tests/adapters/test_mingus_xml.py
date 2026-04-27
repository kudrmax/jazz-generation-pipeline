from pathlib import Path

import pytest
from music21 import converter

from pipeline.progression import ChordProgression
from pipeline.adapters.mingus import MingusPipelineConfig
from pipeline._xml_builders.mingus_xml import build_mingus_xml


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_tonic_whole_writes_valid_musicxml(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)  # music21 raises if XML невалиден
    # 2 такта
    measures = s.parts[0].getElementsByClass("Measure")
    assert len(measures) == 2


def test_tonic_whole_has_two_chord_symbols(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)
    chord_syms = list(s.parts[0].recurse().getElementsByClass("ChordSymbol"))
    assert len(chord_syms) == 2


def test_tonic_whole_one_note_per_measure(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)
    notes_per_measure = [
        len(list(m.getElementsByClass("Note")))
        for m in s.parts[0].getElementsByClass("Measure")
    ]
    assert notes_per_measure == [1, 1]


def test_tonic_quarters_four_notes_per_measure(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_quarters", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)
    notes_per_measure = [
        len(list(m.getElementsByClass("Note")))
        for m in s.parts[0].getElementsByClass("Measure")
    ]
    assert notes_per_measure == [4, 4]


def test_tonic_pitch_is_root_of_chord(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="tonic_whole", checkpoint_epochs=100)
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    s = converter.parse(out)
    measures = list(s.parts[0].getElementsByClass("Measure"))
    # Cmaj7 → C тоника; Am7 → A тоника
    note0 = list(measures[0].getElementsByClass("Note"))[0]
    note1 = list(measures[1].getElementsByClass("Note"))[0]
    assert note0.pitch.name in ("C",)
    assert note1.pitch.name in ("A",)


def test_custom_xml_copies_file(tmp_path: Path):
    src = tmp_path / "src.xml"
    src.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="3.1"><part-list>'
        '<score-part id="P1"><part-name>Melody</part-name></score-part>'
        '</part-list><part id="P1"></part></score-partwise>'
    )
    cfg = MingusPipelineConfig(
        seed_strategy="custom_xml", checkpoint_epochs=100, custom_xml_path=src,
    )
    out = tmp_path / "input.xml"
    build_mingus_xml(_basic_progression(), cfg, out)
    assert out.read_text() == src.read_text()


def test_custom_xml_requires_path(tmp_path: Path):
    cfg = MingusPipelineConfig(seed_strategy="custom_xml", checkpoint_epochs=100, custom_xml_path=None)
    with pytest.raises(ValueError, match="custom_xml_path"):
        build_mingus_xml(_basic_progression(), cfg, tmp_path / "input.xml")
