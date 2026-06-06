# Universal Coset Field Theory (UCFT)

Complete manuscript: graph--Stone--BAMLP foundation $\to$ $E_6$/Albert coset $\to$
induced gravitation and Standard Model embedding. The universe is the unique fixed
point $\mathfrak{U}^\star$ of the master operator $\mathbb{U}$.

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