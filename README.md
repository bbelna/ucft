# Stone–Jordan Exceptional Theory (SJET)

**From the void by mirror climb** — Brandon Belna.

## True axioms (only two)

| Axiom | Content |
|-------|---------|
| **Void** \(\mathcal{V}\) | No observable distinction; trivial Boolean algebra / one-point Stone spectrum |
| **Mirror** \(\mathsf{M}\) | Involutive doubling operator; repeated application climbs \(0\to\mathbb{R}\to\mathbb{C}\to\mathbb{H}\to\mathbb{O}\to J_3(\mathbb{O})\to E_8\to E_8\times E_8\to E_6\to\cdots\to 0\) |

The climb operator is \(\mathsf{C}=\mathrm{Bal}_{\mathcal{C}}\circ\mathsf{M}\) (mirror then capacity balance). The universe is \(\mathfrak{U}^\star=\mathsf{C}^\infty(\mathcal{V})\).

**Proved rungs** (Theorem in §2):
| \(n\) | \(\mathsf{M}^n(\mathcal{V})\) | \(\dim\) |
|------|-------------------------------|----------|
| 0 | void | 0 |
| 1 | \(\mathbb{R}\) | 1 |
| 2 | \(\mathbb{C}\) | 2 |
| 3 | \(\mathbb{H}\) | 4 |
| 4 | \(\mathbb{O}\) | 8 (Hurwitz terminal) |
| 5 | \(J_3(\mathbb{O})\) | 27 (Jordan exit; sedenions fail) |

Everything else — graph, potential, gauge — is **derived**.

## Build

```bash
bash .sjet-build/assemble-sjet.sh
toolbox run -c latex bash -lc 'cd /var/home/bbelna/ucft && latexmk -pdf -f sjet.tex'
```

## Remote

`git@github.com:bbelna/ucft.git`