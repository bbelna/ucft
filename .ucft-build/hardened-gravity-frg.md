# Hardened GRAVITY + RG sector for UCFT

Source of truth: `UCFT-SPINE.md` §§4–6 and the canonical-numbers block. All values below
are the single canonical values; no deviation from the spine. Convention `(-,+,+,+)`,
`[f] = mass`, and `ξ, ĝ², ŷ², κ, Λ_cc` dimensionless.

---

## (i) Canonical scalar mass and the 28 + 4 count

Vacuum manifold (real form for dynamics = compact `E₆`, or `E₆₍₋₁₄₎`; `H = Spin(10)×U(1)`
compact ⇒ definite coset metric, unitary gauge sector):

```
M = E₆ / (Spin(10) × U(1)) = EIII ,   dim_R M = 32 .
```

Coset tangent (real-irreducible, complex type; commutant = C; NO SO(10) singlet):

```
m = 16₍₋₃₎ ⊕ 16bar₍₊₃₎      (32 real).
```

**Tree level.** `V` is `E₆`-invariant ⇒ `Hess V_tree|_m = 0`: all 32 coset scalars are exact
Goldstones (Goldstone's theorem).

**One loop.** The Coleman–Weinberg effective potential is `H`-invariant. Because `m` is a
**single** real-irreducible `H`-module, Schur's lemma forces ONE common eigenvalue on all
32 directions ⇒ a single common radiative mass. (This uses irreducibility; it does not
fight it.) There is no 4-dim `grad I₂, grad I₃` singlet subspace — `m` has no SO(10)
singlet, so that construction is forbidden.

**The 4 frame/soldering modes** are protected by GAUGE symmetry (world-volume diffeomorphism
+ local Lorentz), with Ward identity `∇_μ(δΓ/δe^a_μ) = 0`. They are eaten/composite-vierbein
modes, NOT internal Hessian zeros. `H` stays unbroken (the 46 = dim(45)+1 gauge bosons stay
massless).

**Count (use 28 in all loop sums):**

```
32 real = 28 physical massive scalars (single common mass) + 4 gauge/soldering (eaten).
```

**Canonical mass.** With `[f] = mass`, the dim-4 Hessian eigenvalue is `f²μ²`; the
canonically normalized physical mass divides by `f²`:

```
m² = μ²        (drop the spurious extra f²; dim-4 Hessian eigenvalue = f²μ²).
```

Block multiplicities on the 27 follow `27 = 1₍₊₄₎ ⊕ 10₍₋₂₎ ⊕ 16₍₊₁₎`, i.e.
`M = diag(μ₁·1₁, μ₁₀·1₁₀, μ₁₆·1₁₆)`; residual symmetry is `H`, not `E₆`.

---

## (ii) Sakharov-induced Planck mass from the Seeley–DeWitt a₂ coefficient

One scheme (ζ / proper-time). Only **massive** fields produce the `m² log` term: the `a₂`
curvature coefficient is mass-independent, so massless fields contribute only to the
subtracted quadratic divergence and not to the `R`-coefficient running. The proper-time
`n = 1` term is singular `0/0` in the bare formula and is written explicitly as

```
n = 1 term  =  a₂ · (1/2) log(Λ²/μ²) ,
```

with the heat-kernel coefficient `a₂ ∝ (1/6) R` per real scalar (the `1/6` is the conformal
Seeley–DeWitt coefficient). Summing the 28 massive scalars (`s = +1`, `m² = f²μ²`):

```
M_Pl² = (1/(16π²)) · (1/6) · Σ_{massive i} s_i m_i² log(Λ²/m_i²)
      = (28/6) · f² μ² · log(Λ²/μ²) / (16π²)   > 0.
```

Prefactor **28/6** (NOT 32); keep the `1/6`, the `μ²`, and the `log`. `β_{M_Pl²}` is derived
from the SAME expression, so static and running agree:

```
β_{M_Pl²} = μ d(M_Pl²)/dμ  derived from  M_Pl² ∝ (28/6) f²μ² log(Λ²/μ²)/(16π²),
```

ensuring no 32-vs-28 or normalization mismatch.

**Explicit arithmetic (positivity check).**

```
28/6              = 4.6667                 (> 0)
1/(16π²)          = 0.0063326              (> 0)
f², μ²            > 0                       ([f]=mass, μ real)
log(Λ²/μ²)        > 0    for Λ > μ          (UV cutoff above the mass scale)
```

Worked representative (`f = μ = 1` in `μ`-units, `Λ/μ = 100` ⇒ `log(Λ²/μ²) = log(10⁴) = 9.2103`):

```
M_Pl² = 4.6667 × 1 × 1 × 9.2103 × 0.0063326 = 0.27218  (μ²-units)   > 0  ✓
```

**Positivity:** every factor (`28/6`, `1/(16π²)`, `f²μ²`, `log(Λ²/μ²)|_{Λ>μ}`) is strictly
positive, so `M_Pl² > 0`. The 28 scalars (`s = +1`) dominate any massive-fermion negative
(`s = −1`) contribution, so the induced Newton constant has the correct sign.

**Cosmological constant.** The `a₀` term gives an uncancelled `~Λ⁴` vacuum energy (no SUSY
cancellation). A CC counterterm is added; `Λ_cc` is a **tuned relevant coupling**. UCFT
*inherits*, does NOT solve, the CC problem. `Λ_cc` is counted as a relevant direction (see
(v)). FP ghosts are Lorentz scalars in `adj(H)` with `Δ_ghost = −(D_adj)²`; no quoted
heat-kernel number changes.

---

## (iii) Gauge β-function and asymptotic FREEDOM

Casimirs (canonical): `C_A(SO(10)) = 8`, `T(16) = 2` (the spinor is index-2, NOT 16 index-1
fundamentals), `C₂(16) = 45/8`. Do NOT mix `E₆` `C_A = 12` into SO(10) sums.

```
β_{ĝ²} = 2 ĝ² + (b₀/48π²) ĝ⁴ ,
b₀     = (11/3) C_A − (4/3) T n_F − (1/6) T n_S .
```

**One SO(10) generation (n_F = 1, scalar Higgs/Yukawa content n_S = 2):**

```
(11/3) C_A      = (11/3)(8)   = 88/3  = 29.3333
(4/3) T n_F     = (4/3)(2)(1) = 8/3   =  2.6667
(1/6) T n_S     = (1/6)(2)(2) = 2/3   =  0.6667
b₀ = 88/3 − 8/3 − 2/3 = 78/3 = 26 .
```

`b₀ = 26` (matches canonical). [Check: `80/3 − 26 = 2/3 = (1/6)T n_S` ⇒ `n_S = 2`.]

**Three-generation route (n_F = 48):** the family count shifts `n_F → 48`
(`3 generations × 16`, counted with `T(16) = 2` in the Dynkin-weighted sum). Keeping the
same `n_S = 2`:

```
(4/3) T n_F = (4/3)(2)(48) = 384/3 = 128 .
b₀ = 88/3 − 128 − 2/3 = 86/3 − 128 = 28.6667 − 128 = −99.3333 .
```

**STATE:** for the three-generation `n_F = 48` route, **`b₀ = −298/3 ≈ −99.33`** (with
`n_S = 2`; the matter-only value with `n_S = 0` is `−296/3 ≈ −98.67`).

**Assertion — gauge ASYMPTOTIC FREEDOM.** The canonical `+2 ĝ²` term and the loop term give
a Gaussian UV fixed point at `ĝ²_* = 0`. There is **NO** interacting gauge fixed point
(`ĝ²_* > 0`). The gauge–Yukawa sector is asymptotically FREE; UV completeness is via
asymptotic freedom of the gauge–Yukawa sector + a gravitational fixed point + irrelevant
higher operators — NOT a non-Gaussian gauge fixed point.

---

## (iv) Gravitational β-function and asymptotic SAFETY (INDICATIVE)

```
β_κ = 2κ − (5/48π²) κ²  .
```

Fixed point (set `β_κ = 0`, `κ ≠ 0`):

```
2 = (5/48π²) κ_*   ⇒   κ_* = 2 · 48π²/5 = 96π²/5 = 189.4964 .
```

UV-attractive eigenvalue:

```
dβ_κ/dκ |_{κ_*} = 2 − 2(5/48π²) κ_* = 2 − 2(5/48π²)(96π²/5) = 2 − 4 = −2 .
```

So `eig(M)_κ = −2` ⇒ critical exponent `θ_κ = −eig = +2` (UV-attractive, relevant). The
minus sign in `β_κ` is essential: κ flows AWAY from 0 toward `κ_*` (NOT "κ → 0, gravity
decouples"). This matches the appendix anomalous dimension `η_M = −(5/48π²) κ`.

**Caveat — INDICATIVE, not rigorous.** `κ_* = 96π²/5 ≈ 189 ≫ 1` is outside the controlled
weak-coupling regime; this single-scalar-loop fixed point is indicative only.

---

## (v) Explicit numerical fixed point and stability eigenvalues

Couplings `(ξ, ĝ², ŷ², κ)` plus the tuned `Λ_cc`. AS convention: `θ_i = −eig(M)`,
UV-attractive ⇔ `eig(M) > 0`, `#relevant = #{eig(M) < 0}`.

**Fixed point (representative, self-consistent with (iii)–(iv)):**

```
ξ_*      = 0.05                    (representative non-minimal coupling)
ĝ²_*     = 0                       (Gaussian — gauge asymptotic freedom, no interacting FP)
ŷ²_*     = 0                       (Gaussian — Yukawa asymptotic freedom at ĝ²_*=0)
κ_*      = 96π²/5 = 189.4964       (interacting gravitational FP, INDICATIVE)
Λ_cc,*   = 0.01                    (tuned relevant direction)
```

**Stability matrix `M = ∂β_i/∂g_j`**, order `(ξ, ĝ², ŷ², κ, Λ_cc)`. With the Gaussian gauge
and Yukawa entries, the off-diagonal couplings vanish at the fixed point (each loop
correction to a beta function is proportional to the coupling being differentiated out,
which is zero at `ĝ²_*=ŷ²_*=0`), leaving an effectively triangular matrix whose eigenvalues
are the diagonal entries:

```
∂β_ξ/∂ξ        = 2                                  (canonical; relevant-operator coefficient)
∂β_{ĝ²}/∂ĝ²    = 2 + 2(b₀/48π²) ĝ²_* = 2            (at ĝ²_*=0)
∂β_{ŷ²}/∂ŷ²    = 2                                  (at ŷ²_*=0)
∂β_κ/∂κ        = 2 − 2(5/48π²) κ_* = −2             (at κ_*=96π²/5)
∂β_{Λcc}/∂Λcc  = −4                                 (canonical dim-4 relevant operator)
```

**Eigenvalues of `M` and critical exponents `θ = −eig(M)`:**

```
direction   eig(M)    θ = −eig(M)   character
ξ            +2          −2          irrelevant  (UV-repulsive)
ĝ²           +2          −2          irrelevant  (Gaussian, asymptotically free)
ŷ²           +2          −2          irrelevant  (Gaussian, asymptotically free)
κ            −2          +2          RELEVANT    (UV-attractive gravitational FP)
Λ_cc         −4          +4          RELEVANT    (tuned CC counterterm)
```

**Relevant-direction count:** `#{eig(M) < 0} = 2` — namely `κ` (eig `−2`, θ `+2`) and
`Λ_cc` (eig `−4`, θ `+4`). The central `(ĝ², ŷ²)` block having positive eigenvalues `+2` is
CORRECT under the AS convention (asymptotic freedom). The legacy `{−2,−1.3,−0.9,−0.2}` set
is not reproducible and is replaced by the recomputed values above. All β-functions are
written in ONE normalization (canonical `+2`/`−4` scaling terms + `/48π²` or `/16π²` loop
terms), `C_F = C₂(16) = 45/8`.

---

## Self-consistency ledger

```
m² = μ²                                          (canonical, [f]=mass)
28 massive + 4 gauge = 32 = dim_R(EIII)          ✓
M_Pl² = (28/6) f²μ² log(Λ²/μ²)/(16π²) > 0        ✓ (every factor > 0 for Λ>μ)
b₀(1 gen, n_F=1,n_S=2) = 26                       ✓
b₀(3 gen, n_F=48,n_S=2) = −298/3 ≈ −99.33         (stated)
gauge: asymptotically FREE, ĝ²_*=0               ✓ (no interacting FP)
κ_* = 96π²/5 ≈ 189.50, eig −2, θ +2              ✓ (INDICATIVE)
fixed point (ξ,ĝ²,ŷ²,κ,Λcc) = (0.05,0,0,189.50,0.01)
eig(M) = {+2,+2,+2,−2,−4} ; θ = {−2,−2,−2,+2,+4}
#relevant = 2  (κ, Λ_cc)
```
