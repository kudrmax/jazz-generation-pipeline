from pathlib import Path

import pytest
from music21 import converter

from pipeline.progression import ChordProgression
from pipeline._xml_builders.jazz_xml import build_xml


def _basic_progression() -> ChordProgression:
    return ChordProgression(
        chords=[("Cmaj7", 4), ("Am7", 4)],
        tempo=120.0,
        time_signature="4/4",
    )


def test_tonic_whole_writes_valid_musicxml(tmp_path: Path):
    out = tmp_path / "input.xml"
    build_xml(
        _basic_progression(),
        seed_strategy="tonic_whole",
        custom_xml_path=None,
        out_path=out,
    )
    s = converter.parse(out)  # music21 raises if XML невалиден
    # 2 такта
    measures = s.parts[0].getElementsByClass("Measure")
    assert len(measures) == 2


def test_tonic_whole_has_two_chord_symbols(tmp_path: Path):
    out = tmp_path / "input.xml"
    build_xml(
        _basic_progression(),
        seed_strategy="tonic_whole",
        custom_xml_path=None,
        out_path=out,
    )
    s = converter.parse(out)
    chord_syms = list(s.parts[0].recurse().getElementsByClass("ChordSymbol"))
    assert len(chord_syms) == 2


def test_tonic_whole_one_note_per_measure(tmp_path: Path):
    out = tmp_path / "input.xml"
    build_xml(
        _basic_progression(),
        seed_strategy="tonic_whole",
        custom_xml_path=None,
        out_path=out,
    )
    s = converter.parse(out)
    notes_per_measure = [
        len(list(m.getElementsByClass("Note")))
        for m in s.parts[0].getElementsByClass("Measure")
    ]
    assert notes_per_measure == [1, 1]


def test_tonic_quarters_four_notes_per_measure(tmp_path: Path):
    out = tmp_path / "input.xml"
    build_xml(
        _basic_progression(),
        seed_strategy="tonic_quarters",
        custom_xml_path=None,
        out_path=out,
    )
    s = converter.parse(out)
    notes_per_measure = [
        len(list(m.getElementsByClass("Note")))
        for m in s.parts[0].getElementsByClass("Measure")
    ]
    assert notes_per_measure == [4, 4]


def test_tonic_pitch_is_root_of_chord(tmp_path: Path):
    out = tmp_path / "input.xml"
    build_xml(
        _basic_progression(),
        seed_strategy="tonic_whole",
        custom_xml_path=None,
        out_path=out,
    )
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
    out = tmp_path / "input.xml"
    build_xml(
        _basic_progression(),
        seed_strategy="custom_xml",
        custom_xml_path=src,
        out_path=out,
    )
    assert out.read_text() == src.read_text()


def test_custom_xml_requires_path(tmp_path: Path):
    with pytest.raises(ValueError, match="custom_xml_path"):
        build_xml(
            _basic_progression(),
            seed_strategy="custom_xml",
            custom_xml_path=None,
            out_path=tmp_path / "input.xml",
        )


def test_empty_progression_raises(tmp_path: Path):
    prog = ChordProgression(chords=[], tempo=120.0, time_signature="4/4")
    with pytest.raises(ValueError, match="no chords"):
        build_xml(
            prog,
            seed_strategy="tonic_whole",
            custom_xml_path=None,
            out_path=tmp_path / "input.xml",
        )


def test_build_xml_uses_specified_instrument(tmp_path: Path):
    """Параметр melody_instrument_name влияет на XML."""
    prog = ChordProgression(
        chords=[("Cmaj7", 4)] * 2, tempo=120.0, time_signature="4/4",
    )
    xml_path = tmp_path / "out.xml"
    build_xml(
        prog,
        seed_strategy="tonic_whole",
        custom_xml_path=None,
        out_path=xml_path,
        melody_instrument_name="Tenor Sax",
    )
    content = xml_path.read_text()
    # music21 writes "Tenor Saxophone" or similar identifier
    assert "tenor" in content.lower() or "saxophone" in content.lower()
