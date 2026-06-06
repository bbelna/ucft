# Stone–Jordan Exceptional Theory (SJET)

A verifiable framework for fundamental physics, by Brandon Belna.

SJET is a mathematically closed theory of everything built from three axioms:
a profinite graph arena with Stone spectrum, an order parameter in the exceptional
Jordan algebra \(J_3(\mathbb{O})\), and dynamics as the fixed point of a concave
capacity functional. From these, the framework derives the \(E_6\)-invariant
potential, vacuum selection on EIII, composite \(\mathrm{Spin}(10)\times U(1)\)
gauge theory, induced gravitation, and Standard Model descent with three chiral
families.

## Repository layout

| Path | Description |
|------|-------------|
| `sjet.tex` / `sjet.pdf` | Main manuscript (assembled from `.sjet-build/`) |
| `.sjet-build/` | Modular LaTeX units and `assemble-sjet.sh` |
| `old/` | Prior UCFT/BAMLP work (archived) |

## Build

```bash
bash .sjet-build/assemble-sjet.sh
toolbox run -c latex bash -lc 'cd /var/home/bbelna/ucft && latexmk -pdf -f sjet.tex'
```

## Epistemic status

| Category | Examples |
|----------|----------|
| **Theorem** | Stone lift \(\cong J_3(\mathbb{O})\), EIII vacuum, anomaly cancellation, \(b_0(3\,\mathrm{gen})=12\) |
| **Derived** | Potential from BAMLP capacity dual, universe fixed point \(\mathbb{U}(\mathfrak{U}^\star)=\mathfrak{U}^\star\) |
| **Postulate** | Soldering (P1), three families (P3), GUT–Higgs/Yukawa (P4) |
| **Conjecture** | Gravitational UV fixed point \(\kappa_\star=96\pi^2/5\), heterotic correspondence |

See Section 8 (Verification Ledger) in the manuscript for the full table.

## Remote

```text
git@github.com:bbelna/ucft.git
```