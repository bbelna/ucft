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

**Derived from axioms:** Albert graph, \(E_6\) potential, gauge sector, capacity dual, global Stone equivalence (rungs 0–8), Stone chart dimensions \(\chi(S_6),\chi(S_7),\chi(S_8)=248,496,78\) (not point cardinality), mirror-quotient \(h^1=3\), Sakharov induced gravity, anomaly cancellation, one-loop \(\kappa_\star=96\pi^2/5\), extended FRG Jacobian \(\{+2,+2,+2,-2,-4\}\), semiclassical reflection positivity on EIII, **three families** (\(n_F=3\), conditional on P2 via `prop:sm:capacity7` sketch).

## Conditional ToE closure

| Status | Content |
|--------|---------|
| **Unconditional** | Mirror climb, Albert/\(E_6\) dynamics, gauge, induced \(\bar M_{\mathrm{Pl}}\), FRG Jacobian table |
| **Conditional on P2** | Three chiral families |
| **Postulates (not derivable)** | P1 soldering; P2 chirality; P3′ flavon hierarchy; P4 GUT–Higgs; P5 nonperturbative RP; tuned \(\Lambda_{\mathrm{cc}}\) (`post:gauge-gravity:cc`) |
| **Derived (semiclassical)** | Reflection positivity on EIII (`thm:spacetime:rp-semiclassical`) |
| **No-go** | P1 from mirror (`prop:spacetime:p1-nogo`); P2 from mirror (`prop:sm:p2-nogo`); CC from axioms (`prop:gauge-gravity:cc-nogo`) |

Under P1, P2, P4, P5: SJET is a **conditional Theory of Everything** — mathematics from the void is largely unconditional; contact with observed physics requires the named postulates.

## Physical postulates (not axioms)

| Postulate | Content |
|-----------|---------|
| P1 | Lorentzian soldering, \(d=4\) |
| P2 | Chiral matter (one \(\mathbf{16}\) per family) |
| P3′ | Flavon hierarchy, CKM/PMNS mixing (`post:sm:flavons`) |
| P4 | GUT–Higgs and Yukawa |
| P5 | Nonperturbative reflection positivity (semiclassical RP derived) |
| CC | Tuned cosmological constant |

Former P3 (three families) is **retired** — replaced by Theorem `thm:sm:three-families` (via `prop:sm:capacity7` sketch). P3′ (`post:sm:flavons`) is a separate flavon-hierarchy postulate.

## Open problems (minimal)

1. **Gravitational fixed point:** \(\kappa_\star\) persistence with \(R^2\)/graviton FRG (Conjecture `conj:gauge-gravity:kappa`)
2. **Reflection positivity (nonperturbative):** prove or refute narrowed P5 (`post:spacetime:P5`); semiclassical RP derived (`thm:spacetime:rp-semiclassical`)
3. **Mirror geometry (computational):** explicit Chevalley cell labeling at \(n\geq6\) (mirror-quotient \(h^1=3\) proved: `thm:sm:h1`)
4. **Cosmological constant (external):** no mechanism within SJET core (`prop:gauge-gravity:cc-nogo`); tuned coupling is `post:gauge-gravity:cc`

## Build

```bash
bash .sjet-build/assemble-sjet.sh
toolbox run -c latex bash -lc 'cd /var/home/bbelna/ucft && latexmk -pdf -f sjet.tex'
```

## Remote

`git@github.com:bbelna/ucft.git`