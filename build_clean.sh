#!/bin/bash
# Build script for the cleaned SJET_true manuscript
set -euo pipefail

echo "Building SJET_true (cleaned theorem/conjecture ledger version)..."

if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode sjet_true.tex
elif command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode sjet_true.tex
    pdflatex -interaction=nonstopmode sjet_true.tex  # second pass for TOC/refs
else
    echo "No LaTeX engine found (pdflatex or latexmk). Install TeX Live or use toolbox/container."
    exit 1
fi

echo "Done. Output: sjet_true.pdf"
ls -lh sjet_true.pdf 2>/dev/null || true
