# Stone–Jordan Exceptional Theory (SJET)

**From the void by mirror climb** — Brandon Belna.

## Goal: axiomatic physics

**Two axioms** (void + mirror) → all observed physics, **zero postulates**.

| Postulate | Status |
|-----------|--------|
| P3 (three families) | **Eliminated** → `thm:sm:three-families` |
| P2 (chiral matter) | **Eliminated** → `def:sm:physical-sector` + `thm:sm:chirality-derived` |
| P1 (soldering) | **Elimination target** → `thm:spacetime:soldering-derived` (§11) |
| P4 (GUT–Higgs) | Open → `conj:sm:breaking-derived` |
| P5 (nonpert. RP) | Partial → `thm:spacetime:rp-semiclassical`; `conj:spacetime:rp-full` |
| P3′ (flavons) | Open |
| CC | Open → `conj:gauge-gravity:cc-sequester` |

Full program: **§11 Axiomatic Physics** in `sjet.pdf`.

## True axioms (only two)

| Axiom | Content |
|-------|---------|
| **Void** \(\mathcal{V}\) | No observable distinction |
| **Mirror** \(\mathsf{M}\) | Involutive doubling; climb \(0\to\mathbb{R}\to\cdots\to E_6\to 0\) |

Universe: \(\mathfrak{U}^\star=\mathsf{C}^\infty(\mathcal{V})\), \(\mathsf{C}=\mathrm{Bal}_{\mathcal{C}}\circ\mathsf{M}\).

## Key derived results

- Ladder through \(E_6\); chart sync packaging (rungs 0–8)
- \(n_F=3\) from mirror-quotient **physicality** (not postulated chirality)
- \(h^1(\hat Q,V_{16})=3\); semiclassical reflection positivity on EIII
- \(b_0=12\), GS \(-1\), \(\kappa_\star=96\pi^2/5\)

## Build

```bash
bash .sjet-build/assemble-sjet.sh
toolbox run -c latex bash -lc 'cd /var/home/bbelna/ucft && latexmk -pdf -f sjet.tex'
```

## Remote

`git@github.com:bbelna/ucft.git`