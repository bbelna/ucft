# SJET — Mathematical Physics Deliverable

**Stone–Jordan Exceptional Theory:** mathematical construction from void and
mirror to four-dimensional gauge theory, gravity, and cosmology, with contact
to SM, GR, QFT, and PDG/Planck data.

## Build

```bash
bash .sjet-build/assemble-sjet.sh
toolbox run --container latex latexmk -pdf sjet.tex
```

Output: `sjet.pdf` (mathematical-physics manuscript).

## Repository layout

| Path | Contents |
|------|----------|
| `.sjet-build/` | Current math-physics sources |
| `sjet.tex`, `sjet.pdf` | Assembled deliverable |
| `old/programme-full/` | Full programme archive (Waves 1–7, threads, formalization, ~205 pp) |
| `old/sjet-stack/`, `old/.ucft-build/` | Earlier UCFT modular manuscript |

## Citation

If you use this work, cite the manuscript title and repository URL.