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
           12-scales.tex 16-baryogenesis-cp.tex 17-dark-sector.tex \
           18-cosmology-hubble.tex 19-quark-precision.tex 20-proton-decay.tex \
           13-g2.tex 09-predictions.tex 07-conclusions.tex \
           08-gap-ledger.tex 14-derivation-complete.tex 15-tier-c-precision.tex \
           21-quantum-origin.tex 22-universe-ledger.tex \
           23-sm-lagrangian.tex 24-qcd-hadrons.tex 25-ew-precision.tex \
           26-nuclear-cosmo.tex 27-emergent-physics.tex 28-physics-map.tex \
           29-open-threads-intro.tex 29-threads-1-2.tex \
           29-threads-3-4.tex 29-threads-5-6.tex \
           29-threads-7-9-header.tex 29-threads-7-9.tex \
           30-wave1-thread3.tex 30-wave1-thread4.tex \
           30-wave2-thread4-hadrons.tex 30-wave2-thread5.tex \
           30-wave2-thread1.tex \
           31-wave3-thread5.tex 31-wave3-thread6.tex \
           32-wave4-thread1.tex 32-wave4-thread7.tex \
           33-wave5-thread2.tex 33-wave5-thread8.tex \
           34-wave6-thread9.tex 34-wave6-ledger-closure.tex \
           35-wave7-thread3-gauge-gap.tex 35-wave7-thread4-vub.tex \
           35-wave7-thread5-cmb.tex 35-wave7-tierc-precision.tex \
           35-wave7-residual-closure.tex \
           appendix.tex; do
    inline "$u"
  done
  cat "$BUILD/bibliography.tex"
  echo '\end{document}'
} > "$OUT"
echo "Wrote $OUT ($(wc -l < "$OUT") lines)"