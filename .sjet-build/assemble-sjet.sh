#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/.sjet-build"
OUT="$ROOT/sjet.tex"
inline() { echo "% --- $1 ---"; cat "$BUILD/$1"; echo "% --- end $1 ---"; }
{
  cat "$BUILD/preamble.tex"
  for u in 01-introduction.tex 02-mirror.tex 03-climb.tex 04-universe.tex \
           05-observer.tex 06-physics.tex 10-yukawa.tex 11-flavon-rg.tex \
           12-scales.tex 13-g2.tex 09-predictions.tex 07-conclusions.tex \
           08-gap-ledger.tex 14-derivation-complete.tex 15-tier-c-precision.tex appendix.tex; do
    inline "$u"
  done
  cat "$BUILD/bibliography.tex"
  echo '\end{document}'
} > "$OUT"
echo "Wrote $OUT ($(wc -l < "$OUT") lines)"