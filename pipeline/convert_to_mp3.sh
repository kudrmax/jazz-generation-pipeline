#!/bin/bash
# MIDI -> MP3 конвертер для всех файлов в pipeline/output/<model>/
# Требует: fluidsynth (brew install fluid-synth), ffmpeg, soundfont в pipeline/soundfonts/
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
SF2="${SF2:-$ROOT/soundfonts/TimGM6mb.sf2}"

if [ ! -f "$SF2" ]; then
  echo "[!] Soundfont not found: $SF2"
  echo "    Download FluidR3_GM.sf2 (or any GM .sf2) into $ROOT/soundfonts/"
  exit 1
fi

if ! command -v fluidsynth >/dev/null; then
  echo "[!] fluidsynth not installed. Run: brew install fluid-synth"
  exit 1
fi

if ! command -v ffmpeg >/dev/null; then
  echo "[!] ffmpeg not installed. Run: brew install ffmpeg"
  exit 1
fi

# Принимает либо конкретный файл, либо обходит все output/<model>/*.{mid,midi}
TARGETS=()
if [ -n "$1" ]; then
  TARGETS+=("$1")
else
  while IFS= read -r f; do TARGETS+=("$f"); done < <(find "$ROOT/output" -type f \( -name '*.mid' -o -name '*.midi' \))
fi

for MIDI in "${TARGETS[@]}"; do
  if [ ! -f "$MIDI" ]; then
    echo "[!] not a file, skip: $MIDI"
    continue
  fi
  BASE="${MIDI%.*}"
  WAV="$BASE.wav"
  MP3="$BASE.mp3"

  echo "==> $MIDI"
  fluidsynth -ni -F "$WAV" -r 44100 -g 0.7 "$SF2" "$MIDI" >/dev/null 2>&1
  ffmpeg -y -loglevel error -i "$WAV" -codec:a libmp3lame -qscale:a 2 "$MP3"
  rm -f "$WAV"
  echo "    -> $MP3"
done

echo
echo "[OK] done. MP3 файлы лежат рядом с .mid"
