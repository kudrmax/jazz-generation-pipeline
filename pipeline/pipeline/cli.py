from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import OUTPUT_ROOT
from pipeline.pipeline import generate_all
from pipeline.progression import ChordProgression


def _format_table(results: dict[str, dict]) -> str:
    rows = [("model", "status", "melody_only", "with_chords")]
    for model, r in results.items():
        if "error" in r:
            rows.append((model, "error", r["error"], ""))
        else:
            rows.append((
                model, "ok",
                str(r["melody_only"]),
                str(r["with_chords"]),
            ))
    widths = [max(len(str(row[i])) for row in rows) for i in range(4)]
    out_lines = []
    for row in rows:
        out_lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out_lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen_p = sub.add_parser("generate", help="Generate MIDI for one progression")
    gen_p.add_argument("progression_path", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        if not args.progression_path.exists():
            print(f"error: {args.progression_path} not found", file=sys.stderr)
            return 2
        progression = ChordProgression.from_json(args.progression_path)
        results = generate_all(progression)
        print(_format_table(results))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
