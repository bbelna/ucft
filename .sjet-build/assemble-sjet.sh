#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/.sjet-build"
OUT="$ROOT/sjet.tex"
inline() { echo "% --- $1 ---"; cat "$BUILD/$1"; echo "% --- end $1 ---"; }
{
  cat "$BUILD/preamble.tex"
  for u in \
    01-intro-mathphys.tex \
    02-mirror.tex 03-climb.tex 04-universe.tex \
    05-observer.tex 06-physics.tex \
    14-core-summary.tex \
    23-sm-lagrangian.tex \
    10-yukawa.tex 11-flavon-rg.tex 12-scales.tex 13-g2.tex \
    19-quark-precision.tex 24-qcd-hadrons.tex 25-ew-precision.tex \
    16-baryogenesis-cp.tex 17-dark-sector.tex 18-cosmology-hubble.tex \
    26-nuclear-cosmo.tex 20-proton-decay.tex \
    21-quantum-origin.tex \
    08-standard-theories-bridge.tex \
    09-predictions.tex 15-tier-c-precision.tex \
    07-conclusions-mathphys.tex \
    appendix.tex; do
    inline "$u"
  done
  cat "$BUILD/bibliography.tex"
  echo '\end{document}'
} > "$OUT"
echo "Wrote $OUT ($(wc -l < "$OUT") lines)"