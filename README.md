# Stone–Jordan Exceptional Theory (SJET)

A conditional unified framework in mathematical physics (Brandon Belna).

SJET is an \(E_6\)/Albert-algebra–motivated **derivation-grade embedding** of gravitation and the Standard Model. Three axioms fix the profinite graph arena, the order parameter \(\Phi\in J_3(\mathbb{O})\), and capacity dynamics; postulates P1–P4 supply soldering, chiral matter, three families, and the GUT–Higgs sector. Major claims are labeled theorem / postulate / conjecture and checked in Section 8.

## Layout

| Path | Description |
|------|-------------|
| `sjet.tex` / `sjet.pdf` | Main manuscript |
| `.sjet-build/` | Modular units + `assemble-sjet.sh` |
| `old/` | Archived UCFT/BAMLP work |

## Build

```bash
bash .sjet-build/assemble-sjet.sh
toolbox run -c latex bash -lc 'cd /var/home/bbelna/ucft && latexmk -pdf -f sjet.tex'
```

## Epistemic summary

| Status | Content |
|--------|---------|
| **Theorem** | Potential identification, EIII vacuum, anomaly cancellation, \(b_0(3)=12\), \(\mathfrak{m}\) irreducibility |
| **Postulate** | P1 soldering; P2–P4 matter/Higgs; \(\Lambda_{\mathrm{cc}}\) |
| **Conjecture** | \(\kappa_\star=96\pi^2/5\); OS positivity; heterotic correspondence |

## Remote

`git@github.com:bbelna/ucft.git`