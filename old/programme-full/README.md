# Full SJET Programme Archive

This folder archives the **complete programme manuscript** (Waves 1--7, nine-thread
ledger, Tier~C residuals, formal foundations §36). As of June 2026 it was moved
here when the repository root was refocused on the **mathematical physics**
deliverable.

## Contents

- `.sjet-build/` — modular sources and `assemble-sjet.sh` (56 fragments)
- `sjet.tex` — assembled master (~205 pages)
- `sjet.pdf` — built PDF
- LaTeX auxiliary files (`sjet.aux`, `sjet.log`, etc.)

## Rebuild the archived programme

```bash
cd old/programme-full
bash .sjet-build/assemble-sjet.sh
toolbox run --container latex latexmk -pdf sjet.tex
```

## Last commit before archive

Branch `main`, commit `a886ee9` (formal foundations layer).