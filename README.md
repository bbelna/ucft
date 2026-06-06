# Stone–Jordan Exceptional Theory (SJET)

**From the void by mirror climb** — Brandon Belna.

## True axioms (only two)

| Axiom | Content |
|-------|---------|
| **Void** \(\mathcal{V}\) | No observable distinction; trivial Boolean algebra / one-point Stone spectrum |
| **Mirror** \(\mathsf{M}\) | Involutive doubling operator; repeated application climbs \(0\to\mathbb{R}\to\mathbb{C}\to\mathbb{H}\to\mathbb{O}\to J_3(\mathbb{O})\to E_8\to E_8\times E_8\to E_6\to\cdots\to 0\) |

The climb operator is \(\mathsf{C}=\mathrm{Bal}_{\mathcal{C}}\circ\mathsf{M}\) (mirror then capacity balance). The universe is \(\mathfrak{U}^\star=\mathsf{C}^\infty(\mathcal{V})\).

## Proved rungs (Theorem in §2–3)

| \(n\) | \(\mathsf{M}^n(\mathcal{V})\) | \(\dim\) |
|------|-------------------------------|----------|
| 0 | void | 0 |
| 1 | \(\mathbb{R}\) | 1 |
| 2 | \(\mathbb{C}\) | 2 |
| 3 | \(\mathbb{H}\) | 4 |
| 4 | \(\mathbb{O}\) | 8 (Hurwitz terminal) |
| 5 | \(J_3(\mathbb{O})\) | 27 (Jordan exit; sedenions fail) |
| 6 | \(E_8\) | 248 (automorphism mirror) |
| 7 | \(E_8\times E_8\) | 496 (product mirror) |
| 8 | \(E_6\) | 78 (Cartan mirror) |

**Derived from axioms:** Albert graph, \(E_6\) potential, gauge sector, capacity dual, **three families** (\(n_F=3\), conditional on chiral-matter postulate P2 only).

**Still open:** profinite Stone refinement at \(n\geq6\), mirror-quotient \(h^1=3\), reflection positivity (P5), \(\kappa_\star\) beyond one-loop.

## Physical postulates (not axioms)

| Postulate | Content |
|-----------|---------|
| P1 | Lorentzian soldering, \(d=4\) |
| P2 | Chiral matter (one \(\mathbf{16}\) per family) |
| P4 | GUT–Higgs and Yukawa |

Former P3 (three families) is **retired** — replaced by Theorem `thm:sm:three-families`.

## Build

```bash
bash .sjet-build/assemble-sjet.sh
toolbox run -c latex bash -lc 'cd /var/home/bbelna/ucft && latexmk -pdf -f sjet.tex'
```

## Remote

`git@github.com:bbelna/ucft.git`