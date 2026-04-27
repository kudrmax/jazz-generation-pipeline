#!/usr/bin/env python3
"""BebopNet runner: запускается интерпретатором models/bebopnet-code/.venv/bin/python.

Контракт:
- читает JSON payload со stdin (см. pipeline.runner_protocol)
- params: input_xml_path, output_midi_path, model_dir, checkpoint_filename,
          model_repo_path, num_measures, temperature, top_p, beam_search,
          beam_width, device
- импортирует BebopNet API напрямую (без CLI), парсит XML, генерирует, пишет MIDI
- exit 0 при успехе, exit 1 при ошибке (traceback в stderr)

НЕ импортирует ничего из pipeline.* — живёт в bebopnet-venv.

Mirrors the loading + generation sequence in
``jazz_rnn/B_next_note_prediction/generate_from_xml.py`` but driven by JSON
params instead of argparse.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
from pathlib import Path


def main() -> int:
    payload = json.loads(sys.stdin.read())
    params = payload["params"]
    input_xml      = Path(params["input_xml_path"])
    output_midi    = Path(params["output_midi_path"])
    model_dir      = Path(params["model_dir"])
    checkpoint     = params["checkpoint_filename"]
    repo           = Path(params["model_repo_path"])
    num_measures   = int(params["num_measures"])
    temperature    = float(params["temperature"])
    top_p          = bool(params["top_p"])
    beam_search    = params["beam_search"]
    beam_width     = int(params["beam_width"])
    device_name    = params["device"]

    # cwd + sys.path — иначе `from jazz_rnn...` не подхватит локальный пакет.
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    # bidict 0.14 -> 0.20+ rename: _fwdm/_invm -> fwdm/invm.
    # Pickled converter_and_duration.pkl полагается на старые имена; патчим до unpickle.
    import bidict as _bd
    if not hasattr(_bd.BidictBase, "_fwdm"):
        _bd.BidictBase._fwdm = property(  # type: ignore[attr-defined]
            lambda self: self.fwdm,
            lambda self, v: object.__setattr__(self, "fwdm", v),
        )
    if not hasattr(_bd.BidictBase, "_invm"):
        _bd.BidictBase._invm = property(  # type: ignore[attr-defined]
            lambda self: self.invm,
            lambda self, v: object.__setattr__(self, "invm", v),
        )

    import lxml.etree as le
    import music21 as m21
    import torch

    from jazz_rnn.B_next_note_prediction.music_generator import MusicGenerator
    from jazz_rnn.B_next_note_prediction.transformer.mem_transformer import (
        MemTransformerLM,
    )
    from jazz_rnn.utils.music_utils import notes_to_stream

    device = torch.device(device_name)

    # === Load converter ===
    converter_path = model_dir / "converter_and_duration.pkl"
    with open(converter_path, "rb") as f:
        converter = pickle.load(f)

    # === Load model ===
    args_json_path = model_dir / "args.json"
    with open(args_json_path, "r") as f:
        kwargs = json.load(f)

    # Адаптируем mem_len под длину затравки (head). MemTransformerLM выбрасывает
    # ValueError("number of notes in head N is smaller than memory size M"),
    # если nm_notes_in_head <= mem_len. Pipeline-затравки (tonic_whole etc.)
    # короткие — обычно 8..32 нот; checkpoint обучен с mem_len=64.
    # Считаем <note> элементов в input.xml.
    head_xml = le.parse(str(input_xml))
    head_notes_count = len(list(head_xml.getroot().iter("note")))
    if head_notes_count <= kwargs.get("mem_len", 0):
        # Запас 1: реально нужно strict-less-than; берём -2 для безопасности
        # (extract_vectors может срезать <EOS> ноту).
        kwargs["mem_len"] = max(1, head_notes_count - 2)

    model = MemTransformerLM(**kwargs)
    state_dict = torch.load(
        str(model_dir / checkpoint),
        map_location=str(device),
        weights_only=False,
    )
    model.load_state_dict(state_dict)
    model.converter = converter
    model.to(device)
    model.eval()

    # === Construct generator ===
    # batch_size должен быть кратен beam_width; берём batch_size = beam_width
    # для минимального footprint (одна "линия" — без diversity-fan-out).
    batch_size = beam_width
    generator = MusicGenerator(
        model,
        converter,
        batch_size=batch_size,
        beam_width=beam_width,
        beam_depth=1,
        beam_search=beam_search,
        non_stochastic_search=False,
        top_p=top_p,
        temperature=temperature,
        score_model="",
        threshold=0.0,
        ensemble=True,
        song="",
        no_head=False,
    )

    # === Init stream from input XML ===
    generator.init_stream(str(input_xml))

    # === Generate ===
    notes, _top_likelihood = generator.generate_measures(num_measures)

    # === Build music21 stream from generated notes ===
    stream = notes_to_stream(
        notes[:, 0, :],
        generator.stream,
        generator.chords,
        generator.head_len,
        remove_head=False,
        head_early_start=generator.early_start,
    )

    # === Strip <harmony> via XML round-trip (mirrors generate_from_xml.py) ===
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    tmp_xml_with_chords = output_midi.with_suffix(".with_chords.xml")
    tmp_xml_no_chords = output_midi.with_suffix(".xml")

    xml_converter = m21.converter.subConverters.ConverterMusicXML()
    xml_converter.write(stream, "musicxml", str(tmp_xml_with_chords))

    with open(tmp_xml_with_chords, "rb") as f:
        doc = le.parse(f)
    root = doc.getroot()
    for elem in root.iter("harmony"):
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)
    doc.write(str(tmp_xml_no_chords))

    music_stream_out = m21.converter.parse(str(tmp_xml_no_chords)).parts[0]
    music_stream_out.autoSort = False

    mf = m21.midi.translate.streamToMidiFile(music_stream_out)
    mf.open(str(output_midi), "wb")
    mf.write()
    mf.close()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
