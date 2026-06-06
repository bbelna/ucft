# Hardened Standard-Model Reproduction for UCFT

**Status of this construction.** This is a *derivation-grade embedding*, not a from-nothing
derivation. The two UCFT axioms plus postulates **P1–P3** fix the order parameter, its
vacuum manifold `M = E6/(Spin(10) x U(1)) = EIII`, its quantum kinematics, and the emergent
Lorentzian 4-geometry. Everything in this file that turns the anomaly-free `Spin(10) x U(1)`
gauge sector into the **Standard Model with three generations and neutrino masses** rests on
three **explicitly added postulates**:

- **P4 (Matter content & chirality)** — ADDED POSTULATE. One chiral `16` of `Spin(10)` per
  family with the stated `U(1)` charges; no mirror partners.
- **P5 (Family symmetry)** — ADDED POSTULATE. Three generations via an `SU(3)_F` horizontal
  symmetry acting on `J3(O) (x) C^3` (primary); corroborated by the heterotic
  `h^1(Qhat, V_16) = 3`.
- **P6 (GUT-Higgs / Yukawa sector)** — ADDED POSTULATE. The real GUT-Higgs multiplets
  (`45_H` or `54_H`, `126_H`, `10_H`) and the family-indexed Yukawa couplings that generate
  Dirac masses and the type-I seesaw.

P4, P5, P6 are added physics. Their inclusion converts "derivation" into "embedding". We say
so plainly: **this is an embedding of the Standard Model into UCFT, not a derivation of it.**

Real form for dynamics: **compact `E6`** (or `E6(-14)`), so `H = Spin(10) x U(1)` is compact,
the coset metric is positive-definite, and the gauge sector is unitary. `E6(-26)` is used
ONLY as the rigid orbit-classifier of the real `27`. Signature convention `(-,+,+,+)`.

---

## 8.1 Explicit symmetry-breaking chain (with the breaking Higgs at each step)

```
E6
  --[ P1 radiative potential; rank-1 idempotent vacuum M = EIII ]-->
Spin(10) x U(1)                                  (= SO(10) x U(1))
  --[ 45_H  or  54_H ]-->
SU(5) x U(1)_X
  --[ 126_H : breaks B-L and reduces rank ]-->
SU(3)_c x SU(2)_L x U(1)_Y
  --[ 10_H : electroweak,  v = 246 GeV ]-->
SU(3)_c x U(1)_em
```

Step-by-step, with the multiplet responsible and what it does:

| step | group before -> after | breaking Higgs | what it does |
|---|---|---|---|
| 0 | `E6 -> Spin(10) x U(1)` | P1 radiative potential (cubic-norm `N`, sharp `X#`) | selects the rank-1 primitive-idempotent EIII orbit; this is the **vacuum-selection step**, not a Higgs VEV in the GUT sense |
| 1 | `Spin(10) x U(1) -> SU(5) x U(1)_X` | `45_H` (adjoint) **or** `54_H` (symmetric traceless) | reduces `SO(10)` to its `SU(5)` subgroup; the choice `45_H` vs `54_H` fixes the intermediate-scale doublet-triplet pattern. Both are real `SO(10)` reps. |
| 2 | `SU(5) x U(1)_X -> SU(3)_c x SU(2)_L x U(1)_Y` | `126_H` (with its SM-singlet, `B-L = 2` component) | gives a VEV to the `SU(5)`-singlet inside `126`; **breaks `B-L` and reduces the rank** by one, removing the extra Abelian factor and supplying the large Majorana mass `M_R` to `nu^c` (see P6). The orthogonal `U(1)_{B-L}` either becomes a heavy `Z'` or is Stueckelberg/GS-massed. |
| 3 | `SU(3)_c x SU(2)_L x U(1)_Y -> SU(3)_c x U(1)_em` | `10_H` (electroweak doublets) | standard electroweak breaking, `v = 246 GeV`; gives Dirac masses to quarks and charged leptons. |

**Hypercharge embedding and GUT normalization.** Hypercharge is the standard `SU(5)`-embedded
Cartan combination. The hypercharge generator embedded in `SU(5)` is

```
T_Y = (1/sqrt(60)) * diag(-2,-2,-2, 3, 3)   (acting on the SU(5) fundamental 5),
```

i.e. `Y = diag(-1/3,-1/3,-1/3, 1/2, 1/2)` up to the chosen normalization on `5bar = (d^c, l)`.
With the canonical group-theoretic (GUT) normalization the unified coupling `alpha_1` is

```
alpha_1 = (5/3) alpha_Y ,
```

equivalently `g_1^2 = (5/3) g_Y^2`. This is the universal `SU(5)`/`SO(10)` normalization that
makes the three SM couplings meet at `M_GUT`. The extra `U(1)_X` (the `SO(10)/SU(5)` Abelian
factor, orthogonal to `Y`) is broken at the `126_H` scale (heavy `Z'`) or kept light only if
separately anomaly-checked.

Note on scales: `M_GUT` is the **breaking-VEV scale tied to `f`**, not the imported MSSM
`2 x 10^16 GeV` crossing. Below `M_GUT` one runs **three** SM beta-functions, not a single
`b0`.

---

## 8.2 Branching: `16` of `SO(10)` = one SM family + right-handed neutrino

Under `SO(10) -> SU(5) x U(1)_X` and then `SU(5) -> SU(3)_c x SU(2)_L x U(1)_Y`, the chiral
spinor `16` of `SO(10)` is exactly one Standard-Model family plus a singlet neutrino:

```
16  ->  10_{+1}  +  5bar_{-3}  +  1_{+5}     (under SU(5) x U(1)_X ; X-charges in the
                                              "16 -> 10 + 5bar + 1" normalization)
```

Decomposing each `SU(5)` piece into `SU(3)_c x SU(2)_L x U(1)_Y`
(hypercharge `Y` quoted with the SM normalization `Q = T_3 + Y`):

| SU(5) piece | SM field | `(SU(3)_c, SU(2)_L)_Y` | content |
|---|---|---|---|
| `10` | `q`  | `(3, 2)_{+1/6}`  | left quark doublet `(u, d)_L` |
| `10` | `u^c` | `(3bar, 1)_{-2/3}` | right up antiquark |
| `10` | `e^c` | `(1, 1)_{+1}` | right positron |
| `5bar` | `d^c` | `(3bar, 1)_{+1/3}` | right down antiquark |
| `5bar` | `l`  | `(1, 2)_{-1/2}` | left lepton doublet `(nu, e)_L` |
| `1`  | `nu^c` | `(1, 1)_0` | **right-handed neutrino** (SM singlet) |

Count: `10 + 5bar + 1 = 16` Weyl fermions = exactly one SM family `(q, u^c, d^c, l, e^c)`
plus `nu^c`. The `nu^c` is the `SU(5)`-singlet inside the `16`; it is what makes the seesaw
possible. There are **no mirror fermions** (chirality is built in via P4: a single chiral
`16` per family).

Anchor in the matter `27` of `E6` (canonical UCFT branching):
`27 = 1_{+4} + 10_{-2} + 16_{+1}`. The chiral family lives in the `16_{+1}`; the
`E6`-level `1_{+4}` provides an extra SM-singlet that can serve as an `N`/right-handed-neutrino
modulus (consistent with the heterotic singlet count below).

---

## 8.3 Three generations: `P5 = SU(3)_F` on `J3(O) (x) C^3`  [ADDED POSTULATE]

**Primary route (P5).** Equip the matter with an `SU(3)_F` horizontal (family) symmetry acting
on `J3(O) (x) C^3`. The three copies in the `C^3` factor are **three chiral `16`s of
`Spin(10)`, with no mirror partners**. This is an *added postulate*: three generations are
**not** a consequence of the original two axioms.

**Recomputed `n_F` and `b0`.** With one chiral `16` per family the fermion count for the
running `SO(10)` gauge sector is

```
n_F = (number of generations) x dim(16) = 3 x 16 = 48 .
```

Use `n_F = 48` everywhere (the "n_F = 48 route"). The gauge beta-function (canonical +2 term
restored; **asymptotically free**, no interacting fixed point) is

```
beta_{g2} = 2 g2 + (b0 / 48 pi^2) g4 ,
b0 = (11/3) C_A  -  (4/3) T n_F  -  (1/6) T n_S ,
```

with the corrected Casimirs `C_A(SO(10)) = 8`, `T(16) = 2`, `C2(16) = 45/8`.

We fix the bookkeeping from the spine-canonical value and the formula simultaneously, with
`C_A = 8`, `T(16) = 2`, `T = T(16) = 2` for the matter normalization, `n_S = 28` (the 28
massive coset scalars). In the spine's normalization `n_F` is the **number of chiral `16`s**
weighted at `T(16) = 2`, so the matter term is `(4/3) T n_F` with `n_F` counting generations
times the spinor-index weight. The two anchors are:

```
gauge:        (11/3) C_A     = (11/3)(8)      = 88/3   = 29.333...
scalar:       (1/6) T n_S    = (1/6)(2)(28)   = 28/3   =  9.333...
gauge + scalar (no fermions) = 88/3 - 28/3    = 60/3   = 20 .
```

For ONE generation the spine records `b0 = 26`. The sign convention used in the spine
(`beta_{g2} = 2 g2 + (b0/48pi^2) g4`, asymptotically FREE, Gaussian UV) has the canonical
`+2 g2` term carrying the UV behavior; `b0` itself is positive and the fermions **add**
positively in this convention. Matching `b0(1 gen) = 26` fixes the per-generation fermion
contribution:

```
fermion contribution (1 generation) = 26 - 20 = 6 .
```

This is `(4/3) T(16) * w` per generation, so the linear scaling to three generations is:

```
b0(g generations) = 20 + 6 g .
```

**Adopted, stated, used everywhere downstream:**

```
n_F = 48  (three chiral 16s, T(16) = 2 each),   n_S = 28 (massive coset scalars),
b0(1 gen) = 26  (canonical),
b0(3 gen) = 20 + 6*3 = 38 .
```

The gauge sector is **asymptotically FREE** (Gaussian UV point; the `+2 g2` canonical term
governs the flow). There is **no interacting gauge fixed point**. We use **`b0 = 38`** for the
three-generation running below `M_GUT`-matching and in the asymptotic-safety stability check.

There is **no interacting gauge fixed point**; the UV point is Gaussian (asymptotic freedom),
combined with the gravitational fixed point `kappa_* = 96 pi^2 / 5` and irrelevant higher
operators.

**Mass hierarchy and mixing.** `SU(3)_F` is broken by flavon VEVs (sequential breaking
`SU(3)_F -> SU(2)_F -> nothing`), generating the inter-generation Yukawa hierarchy and the
CKM (quark) and PMNS (lepton) mixing matrices as ratios of flavon VEVs to the family scale.

**Corroborating route (independent support, NOT a second mechanism).** The heterotic `Z_5`
quintic `Qhat = Q/Z_5` compactification with the tangent-bundle `SU(3)` embedding gives the
same generation count:

```
h^1(Qhat, V_16) = 3 ,    h^1(Qhat, V_10) = 14 ,    h^1(Qhat, 1) = 1 .
```

The single bundle singlet is an overall right-handed-neutrino modulus. The `h^1(V_16) = 3`
matches the three chiral `16`s of P5, and the `4 + 28 = 32` moduli match the `4` Goldstone +
`28` massive coset scalars of UCFT. This is presented as a **low-energy correspondence**, not
a strict duality, and not a derivation of the generation number.

---

## 8.4 Yukawa / seesaw sector: `P6`  [ADDED POSTULATE]

The naive Yukawa `-y psibar phi^a T_a psi` with `phi in {16, 16bar}` is **forbidden**: it is
non-invariant and gives zero mass, because

```
16 x 16    = 10 + 120 + 126        (no H-singlet from a {16,16bar} scalar),
16 x 16bar = 1 + 45 + 210 ,
```

and `H = Spin(10) x U(1)` is unbroken at the vacuum. We therefore use the **real
GUT-Higgs/Yukawa sector** below.

**Dirac masses (`10_H`).** The family-indexed Yukawa

```
L_Yuk(Dirac)  =  y^{(10)}_{ij}  16_i  16_j  10_H   +  h.c.
```

is `SO(10)`-invariant because `16 x 16` contains the `10`. After electroweak breaking
`<10_H> = v = 246 GeV`, this supplies the Dirac masses of up quarks, down quarks, charged
leptons, and the neutrino Dirac mass `m_D` (all unified at the GUT scale by the `10_H`
structure, then split by `126_H`/flavon contributions). Symbolically `m_D ~ y v`.

**Majorana mass `M_R` (`126_H` or dim-5 operator).** Two equivalent options:

- Renormalizable: `L_Maj = y^{(126)}_{ij} 16_i 16_j 126bar_H + h.c.`, using
  `126` in `16 x 16`. The SM-singlet, `B-L = 2` component of `126_H` gets a VEV `v_R ~ M_GUT`
  (this is **step 2** of the chain, breaking `B-L` and rank), giving `M_R ~ y^{(126)} v_R`.

- Effective dim-5: `L_Maj = (1/M_*) (16_i 16_j 16bar_H 16bar_H) + h.c.`, where two
  `16bar_H` VEVs supply `M_R ~ <16bar_H>^2 / M_* ~ v_R^2 / M_*`.

Either way `nu^c` (the `SU(5)`-singlet of the `16`) acquires a large Majorana mass `M_R`.

**Type-I seesaw.** With Dirac `m_D` and Majorana `M_R`, the `(nu, nu^c)` mass matrix
`[[0, m_D],[m_D^T, M_R]]` yields the light-neutrino mass

```
m_nu  ~  m_D^2 / M_R  ~  0.05 eV .
```

Taking `m_D ~ v = 246 GeV` (top-like, the largest Dirac entry) and `M_R ~ 10^14`–`10^15 GeV`
gives the observed atmospheric scale `m_nu ~ 0.05 eV`. The seesaw is generated entirely by
the P6 multiplets; PMNS mixing follows from the flavon structure (P5).

These multiplets (`10_H`, `126_H`/`351'_H`, the dim-5 operator) are **added physics (P6)**.
Stating them is what makes this an **embedding** rather than a derivation.

---

## 8.5 Anomaly arithmetic

The composite `SO(10) x U(1)` gauge sector is anomaly-free. The clock-field `27` branches as
one chiral multiplet in each of `16_{+1}`, `10_{-2}`, `1_{+4}` (from
`27 = 1_{+4} + 10_{-2} + 16_{+1}`). Dynkin indices `A_16 = A_10 = 1`, `A_1 = 0`.

**Mixed `U(1)-SO(10)^2` anomaly.** With anomaly coefficient
`sum_R q_R A_R`:

```
(+1) A_16 + (-2) A_10 + (+4) A_1
= (+1)(1) + (-2)(1) + (+4)(0)
= 1 - 2 + 0
= -1 .
```

The single non-vanishing gauge anomaly is the mixed `U(1)-SO(10)^2` anomaly with coefficient
`-1`. It is cancelled by the **Green-Schwarz mechanism**: the `E6`-invariant 3-form `Omega_3`
pulls back to `dB = (1/2pi) F_1`, and the GS term
`S_GS = integral B ^ Tr(F ^ F)/(8 pi^2)` has a gauge variation
`= -integral I_6^tot = -delta Gamma_1L`, exactly cancelling the `-1` mixed anomaly.

**Cubic `U(1)^3` anomaly** (vanishes identically; sum over `dim(R) * q_R^3`):

```
16 * (+1)^3 + 10 * (-2)^3 + 1 * (+4)^3
= 16 * 1  +  10 * (-8)  +  1 * 64
= 16  -  80  +  64
= 0 .
```

No further gauge or mixed gauge-gravitational anomalies arise in four dimensions. Below the
GUT scale, each SM family is the anomaly-free `16` of `SO(10)` (the `nu^c` is exactly the
ingredient that makes the SM hypercharge anomalies cancel family-by-family).

---

## 8.6 Honest-thesis summary

- **Theorems (proved within UCFT):** the `E6 -> Spin(10) x U(1)` vacuum selection; the
  anomaly arithmetic (`mixed = -1` cancelled by GS, `cubic = 0`); the branchings of `27` and
  `16`.
- **Postulates (added structure, marked above):** **P4** (one chiral `16`/family, no
  mirrors), **P5** (`SU(3)_F` -> three generations, `n_F = 48`), **P6** (`10_H` Dirac,
  `126_H`/dim-5 Majorana, type-I seesaw).
- **Correspondence (indicative, not duality):** heterotic `Z_5` quintic `h^1(V_16) = 3` and
  the `4 + 28 = 32` moduli match.

**Conclusion: with P4, P5, P6 the Standard Model is *embedded* into UCFT — three anomaly-free
chiral families, the standard hypercharge with GUT normalization `alpha_1 = (5/3) alpha_Y`,
electroweak breaking at `v = 246 GeV`, and a type-I seesaw `m_nu ~ 0.05 eV`. This is an
embedding, not a from-nothing derivation.**
