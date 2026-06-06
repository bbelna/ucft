# SJET — Stone–Jordan Exceptional Theory

**Building the universe from the mirror.**

This repository contains a single mathematical-physics manuscript, `sjet.tex`, that develops SJET from two axioms (void and mirror) through the exceptional ladder to Standard Model and gravity invariants.

## Build

The manuscript is self-contained in `sjet.tex`. Modular sources live in `.sjet-build/`:

```bash
bash .sjet-build/assemble-sjet.sh   # optional: regenerate sjet.tex
latexmk -pdf sjet.tex
```

Or with the LaTeX toolbox container:

```bash
toolbox run --container latex latexmk -pdf sjet.tex
```

## Archive

The previous modular manuscript (verification ledger, unit files, assemble script) lives in `old/`:

- `old/sjet.tex` — prior master document
- `old/.sjet-build/` — modular unit sources and `assemble-sjet.sh`

## Citation

If you use this work, cite the manuscript title and repository URL.