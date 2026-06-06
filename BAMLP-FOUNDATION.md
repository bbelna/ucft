# BAMLP Foundation Layer for UCFT

This document records the pre-axiomatic substrate that UCFT is meant to descend
from. It extends `UCFT-SPINE.md` without replacing the proved $E_6$ core.

## The three tiers

1. **Graph tier (primitive).** Configuration space is a weighted graph
   $(\mathcal{G},\mu)$ with barrier-aware geodesic costs $c_i$. This is the
   *graph description of the universe*.

2. **Stone--division tier (combinatorial lift).** The Stone spectrum
   $\mathrm{Spec}(\mathcal{B}(\mathcal{G}))$ of the graph's cut Boolean algebra
   carries a sequential lift through the Hurwitz tower
   $$0 \to \mathbb{R} \to \mathbb{C} \to \mathbb{H} \to \mathbb{O} \to J_3(\mathbb{O}).$$

3. **Exceptional tier (BAMLP-selected).** Capacity balancing via the BAMLP dual
   $$x \mapsto \arg\min_i(c_i(x)-\omega_i)$$
   partitions the graph into symmetry-labelled cells along
   $$0 \to E_8 \to E_8\times E_8 \to E_6 \to \mathrm{Spin}(10)\times U(1)
   \to \cdots \to 0.$$

UCFT (axioms I--II, postulates P1--P6) begins at the $E_6$ rung.

## Epistemic status

| Claim | Status |
|-------|--------|
| Stone duality, Hurwitz tower | Theorem (standard math) |
| BAMLP capacity dual (finite) | Theorem (BAMLP manuscript) |
| Graph arena, Stone--division lift | Postulate |
| BAMLP selects $E_6$ vacuum / full ladder | Conjecture |
| UCFT vacuum selection, SM descent | Theorem (under P1--P6) |

## Open problems

- Explicit cosmic graph reproducing EIII ($\dim 32$).
- Derive P1 potential from BAMLP capacity functional.
- Controlled $E_8 \to E_6$ breaking compatible with heterotic $E_8\times E_8$.

## Source files

- `.ucft-build/unit-bamlp.tex` — manuscript section
- `/var/mnt/data/Git/bamlp` — reference BAMLP implementation and territorial manuscript