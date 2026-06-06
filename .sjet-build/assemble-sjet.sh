#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/.sjet-build"
OUT="$ROOT/sjet.tex"
inline() { echo "% --- $1 ---"; cat "$BUILD/$1"; echo "% --- end $1 ---"; }
{
  cat "$BUILD/preamble.tex"
  for u in 01-introduction.tex 02-foundations.tex 03-ladder.tex 04-dynamics.tex \
           05-spacetime.tex 06-gauge-gravity.tex 07-standard-model.tex \
           08-verification.tex 09-conclusions.tex; do
    inline "$u"
  done
  cat "$BUILD/bibliography.tex"
  echo '\end{document}'
} > "$OUT"
echo "Wrote $OUT ($(wc -l < "$OUT") lines)"