# Universal Coset Field Theory (UCFT)

Manuscript and modular LaTeX build for UCFT: from axioms on $J_3(\mathbb{O})$ and compact $E_6$ to induced gravitation and Standard Model embedding.

## Build

```bash
.ucft-build/assemble-ucft-final.sh
latexmk -pdf -f ucft-final.tex
```

With Fedora toolbox:

```bash
toolbox run -c latex bash -lc 'cd /path/to/ucft && latexmk -pdf -f ucft-final.tex'
```

## Layout

- `ucft-final.tex` — standalone master manuscript (generated; do not edit by hand)
- `ucft.tex` — condensed modular wrapper
- `.ucft-build/unit-*.tex` — source units
- `.ucft-build/assemble-ucft-final.sh` — regeneration script

Edit the unit files, then rerun the assembly script before compiling.