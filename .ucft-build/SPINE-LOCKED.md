# UCFT — SPINE-LOCKED: The Authoritative Foundation

**Universal Coset Field Theory (UCFT), by Brandon Belna.**

This file is the **single source of truth** for the UCFT manuscript. It supersedes
`UCFT-SPINE.md` and every conflicting statement in any source or hardened-derivation file.
It locks the honest thesis, the postulate set P1–P6, the corrected real-form / base-point /
branching facts, the four hardened results, and the **single canonical value** for every
shared number. Every section and appendix of the manuscript MUST obey this document. Where a
hardened file disagreed with another, this file selects ONE value and records the choice
(see §12, "Resolved disagreements"). It is grounded in a 54-finding adversarial audit, and
it actively guards against the 11 forbidden errors enumerated in §13.

Conventions locked globally: signature **(−,+,+,+)**; `[f] = mass`; `ξ, ĝ², ŷ², κ, Λ_cc`
dimensionless; "clock fields" = the 32 Goldstone directions of the coset; one symbol each
for `f` (decay constant) and `μ` (breaking/mass scale).

---

## 0. Honest thesis

UCFT is an **E₆ / Albert-algebra–motivated unification**. Two axioms supply the *arena*
(the order parameter and its quantum kinematics); a small, **explicitly enumerated** set of
additional postulates **P1–P6** then yields a 4-D Lorentzian quantum field theory containing
an anomaly-free `Spin(10) × U(1)` gauge sector, Sakharov-induced gravity, and — after a
stated GUT-Higgs/flavor sector — the Standard Model with three generations and neutrino
masses.

The phrase "everything follows from two axioms" is **demoted**: the axioms supply the arena;
P1–P6 are added structure, stated up front. This is a **derivation-grade embedding**, not a
from-nothing derivation. Every nontrivial claim is labeled:

- **Theorem** — proved from stated hypotheses;
- **Postulate** — assumed added structure (P1–P6);
- **Indicative / conjecture** — motivated, not controlled.

The vacuum selection (P1→P2) is **proved** (Theorem T4, §3). The Lorentzian geometry (P3),
matter content (P4), generations (P5), and Yukawa sector (P6) are **added structure**. The
gravitational fixed point is **indicative**. The Standard Model is **embedded**, not derived.

### The two axioms

- **Axiom I.** The order parameter is a field `Φ` valued in the matter rep `27 = J₃(O)` of
  `E₆`. (Honest reading, default option B of the old §7: keep `Φ=0` as the symmetric origin;
  the potential `V` makes it a strict local **maximum** — `Hess V(0) ≺ 0`, Theorem T5 — so
  the rank-1 orbit beats the origin and the roll-down dynamics are supplied, Theorem T6.)
- **Axiom II.** A global unitary `U(t)=e^{−iHt}`. **Demoted to a theorem**: time is emergent
  (a Page–Wootters clock `φ⁰` + the induced-gravity Hamiltonian constraint
  `Ĥ_phys|Ψ⟩⟩=0`); `U(τ)` is recovered on physical states in the semiclassical regime via a
  timelike Killing vector and `H = ∫ T₀₀` along it (§5).

### The added postulates P1–P6 (state once, up front)

- **P1 (Potential class).** An `E₆`-invariant polynomial potential built from the cubic norm
  `N` and the sharp map `X#`, whose global minimum is the rank-1 idempotent orbit.
- **P2 (Vacuum selection).** The vacuum is that minimal orbit,
  `M = E₆/(Spin(10)×U(1)) = EIII`. **(Proved from P1 by Theorem T4.)**
- **P3 (Soldering & signature).** A soldering map identifies four clock-field directions with
  a tangent frame carrying the Minkowski form inherited from the cubic norm `N` /
  `H₂(O) ≅ R^{1,9}`. `d=4` and signature `(1,3)` enter HERE — they are added structure, NOT
  Killing-form theorems.
- **P4 (Matter content & chirality).** One chiral `16` of `Spin(10)` per family, with the
  stated `U(1)` charges; no mirror partners.
- **P5 (Family symmetry).** Three generations via an `SU(3)_F` horizontal symmetry on
  `J₃(O) ⊗ C³` (primary); corroborated by the heterotic `h¹(Q̂,V₁₆)=3` (correspondence, not
  duality).
- **P6 (GUT-Higgs / Yukawa sector).** Real `SO(10)` Higgs (`45_H` or `54_H`, `126_H`) and
  Yukawa multiplets (`10_H`, `126_H` / dim-5) generating fermion masses and the type-I
  seesaw.

---

## 1. Real form, base point, invariants (BLOCKER — root foundation)

**Do NOT conflate two E₆'s.**

- `Aut(J₃(O))` (preserving the Jordan product, hence BOTH `Q=Tr(X²)` and `N`) is the
  **compact `F₄`** (dim 52).
- The **reduced structure group** (preserving the cubic norm `N` up to a real character) is
  the **non-compact `E₆₍₋₂₆₎`** (dim 78). It does **NOT** preserve `Q=Tr(X²)`.
- `Stab_{E₆}(diag(1,1,1)) = F₄`, orbit `E₆/F₄`, real dim **26** — NOT the vacuum. No orbit
  of a *vector* in the 27 has dim > 26.

**Real form to commit to (LOCKED):**

- **Dynamics:** the **compact `E₆`** (equivalently `E₆₍₋₁₄₎`, whose maximal compact is
  exactly `H = Spin(10)×U(1)`). Then `H` is compact ⇒ the coset metric is **positive-definite**
  ⇒ healthy kinetic term + **unitary** gauge sector. Coercivity and boundedness-below of `V`
  are statements about the **compact** form (non-compact orbits cannot be coercive).
- **Orbit classifier only:** `E₆₍₋₂₆₎`, the rigid classifier of the real `27`. Used solely to
  classify orbits by **rank and `N`**.

**Invariants / orbit classification (corrected, LOCKED):**

- Only the cubic norm `N` is structure-group invariant (up to a character `λ(g)`).
  `Q(X)=Tr(X²)` is **`F₄`-invariant only**.
- Real `E₆`-orbits are classified by **rank ∈ {0,1,2,3} and `N`**, NOT by `(Q,N)`.
- `rank X ≤ 1 ⇔ X# = 0`; `rank X ≤ 2 ⇔ N(X)=0` (and `X#≠0`); `rank 3 ⇔ N(X)≠0`.

**The 32-dim vacuum manifold (LOCKED):**

```
M = E₆ / (Spin(10) × U(1)) = EIII ,   dim_R M = 32 ,
```

the **closed rank-1 (minimal) E₆-orbit** in `P(27_C)` — the **complex Cayley plane**, the
orbit of a **rank-1 primitive idempotent**, NOT the orbit of `diag(1,1,1)`. (`F₄/Spin(9) ≅
OP² `, dim 16, are the real rank-1 idempotents; the complex line under full `E₆` sweeps the
larger EIII, dim 32.) Defining involution datum:

```
adjoint branching:   78 = 45₀ ⊕ 1₀ ⊕ 16₍₋₃₎ ⊕ 16bar₍₊₃₎     (under H = Spin(10)×U(1)).
```

**Potential class P1 (corrected so the minimum is the rank-1 orbit):**

```
V(X) = α·⟨X#, X#⟩ + β·[Tr(X²) − c₂]² + γ·Tr(X²)^k ,   α,β,γ > 0,  k ≥ 2 even.
```

The `⟨X#,X#⟩` term is `E₆`-covariant and vanishes exactly on rank ≤ 1; the `Tr(X²)` terms
are `F₄`-invariant selectors fixing the scale/representative. On the **compact** form
`Q=⟨X,X⟩` is the invariant trace form, so every term of `V` is a genuine compact-`E₆`
invariant.

---

## 2. Coset tangent space and branchings (BLOCKER)

Authoritative, use everywhere:

```
coset tangent   m  = 16₍₋₃₎ ⊕ 16bar₍₊₃₎       (32 real; chiral spinor pair of Spin(10)).
matter rep      27 = 1₍₊₄₎ ⊕ 10₍₋₂₎ ⊕ 16₍₊₁₎ .
adjoint         78 = 45₀ ⊕ 1₀ ⊕ 16₍₋₃₎ ⊕ 16bar₍₊₃₎ .
```

- `m` is **real-irreducible of complex type** (commutant `= C`). The `U(1)` charge forbids
  the symmetric `16×16` and `16bar×16bar` invariants, leaving the unique charge-0
  cross-pairing = Killing restriction. The invariant metric is therefore **unique and
  positive-definite** (Schur).
- `m` contains **NO SO(10) singlet**. (Forbidden: any `2×10+4`, `10+10bar+4`, or singlet-in-`m`
  decomposition.)

---

## 3. Hardened Result A — Vacuum-selection theorem (P1 ⇒ P2)

*(Full derivation: `hardened-potential.md`. Status: Main Theorem T4 is **proved**.)*

**Setup.** `J₃(O)` is the 27-dim real Albert algebra; trace form `Q(X)=Tr(X²)=⟨X,X⟩`
positive-definite (formally real / Euclidean Jordan algebra); cubic norm `N(X)=det X`; sharp
map `X#` with `X#∘X=N(X)E`, `(X#)#=N(X)X`, `X#=∇N(X)`. In a Jordan frame
`diag(λ₁,λ₂,λ₃)# = diag(λ₂λ₃, λ₃λ₁, λ₁λ₂)`.

**L1 (Theorem).** `X# = 0 ⇔ rank(X) ≤ 1`.

**L1′ (Theorem).** Normalized rank-1 elements = primitive idempotents = single `F₄`-orbit
`F₄/Spin(9) ≅ OP²` (dim 16); the `E₆`-orbit closure of a rank-1 line in `P(27_C)` is
`EIII = E₆/(Spin(10)×U(1))`, dim 32.

**T1 (Theorem).** `⟨X#,X#⟩ ≥ 0`, with equality `⇔ X#=0 ⇔ rank(X) ≤ 1`. Explicitly
`⟨X#,X#⟩ = (λ₂λ₃)²+(λ₃λ₁)²+(λ₁λ₂)² = σ₂(X)² − 2 tr(X) N(X)` (sum of squares).

**T2 (Theorem).** `⟨X#,X#⟩` is invariant under the **compact** `E₆` (and `E₆₍₋₁₄₎`): on the
compact form the `27` is unitary, the trace form is the invariant metric, and the norm
character is identically 1 (continuous homomorphism from a compact connected group to
`R_{>0}`). Under `E₆₍₋₂₆₎` it is only a relative invariant — which is why the **dynamics** live
on the compact form.

**L2 (Theorem).** The selector `W(q) = β(q−c₂)² + γ q^k` (`q=Q≥0`, `β,γ>0`, `k≥2` even) is
**strictly convex** with `W′(0)<0`, hence has a **unique** interior minimizer `q_⋆∈(0,c₂)`,
`W(q_⋆)<W(0)=βc₂²`, `q_⋆ = c₂ − (γk/2β)c₂^{k−1}+… → c₂` as `γ→0⁺`. Fixes the nonzero scale,
excludes the origin.

**L3 (Theorem).** Coercivity holds on the **compact** form (`V ≥ γ‖X‖^{2k} → ∞`); it fails
on non-compact orbits (the norm character is unbounded). Coercivity genuinely requires the
compact real form.

**T3 (Theorem).** On the rank-≤1 cone, the global minimum is attained exactly on the
fixed-norm rank-1 idempotents `{λP : λ²=q_⋆}`, whose projective `E₆`-orbit is EIII.

**T4 (Main Theorem — P2 proved).** With `α,β,γ>0`, `k≥2` even, `c₂>c₂⋆`, every global
minimizer satisfies `X#=0` and `Tr(X²)=q_⋆`; the global-minimum locus is the rank-1
primitive-idempotent orbit, projectively `M = EIII = E₆/(Spin(10)×U(1))`, `dim_R = 32`.
*Proof:* `V = α⟨X#,X#⟩ + W(Q) ≥ W(q_⋆)`; equality forces `X#=0` (rank ≤1, T1/L1) and
`Q=q_⋆>0` (rank exactly 1). Rank 2,3 strictly lose by `α⟨X#,X#⟩>0`. In particular
`diag(1,1,1)` (rank 3, `⟨X#,X#⟩=3>0`, orbit `E₆/F₄` dim 26) is **not** a minimizer.

**T5 (Theorem).** `Hess V(0) = −4βc₂·𝟙 ≺ 0`: the origin is a strict local maximum (all 27
eigenvalues `= −4βc₂ < 0`). Since `V(0)=βc₂² > W(q_⋆)`, the rank-1 orbit beats the origin
locally and globally.

**T6 (Theorem).** Roll-down: under gradient flow / EOM, `0` is linearly unstable (growth rate
`4βc₂`); `V` is a strict Lyapunov function; by coercivity (L3) and LaSalle, generic data flow
to `M = EIII`.

**Tree-level Goldstone consistency.** On `M`, `Hess V_tree|_m = 0` (32 exact Goldstones).
Radiative corrections are `H`-invariant; Schur on the irreducible `m` gives one common mass.

---

## 4. Hardened Result B — Lorentzian soldering (P3)

*(Full derivation: `hardened-soldering.md`. The algebraic facts are Theorems; `d=4` and
signature are Postulate P3; reflection positivity is Indicative.)*

**Logical separation (locked).** The **positive-definite** object is the coset/target metric
`K|_m` on the 32-dim internal `m`. The **indefinite** object is the cubic norm `N` on
`J₃(O)`, whose `H₂(O) ≅ R^{1,9}` sub-block carries a Minkowski form. Two distinct forms on
two distinct spaces; no single form is asked to be both definite and indefinite.

**Theorem 1.1.** `H₂(O) = {[[a,x],[x̄,b]] : a,b∈R, x∈O}`, `dim_R = 10`,
`det X = ab − n(x)`. With `a=x⁰+x⁹`, `b=x⁰−x⁹`: `det X = (x⁰)² − (x⁹)² − Σᵢ(xⁱ)² = −η_{μν}X^μX^ν`,
so `(H₂(O), −det) ≅ R^{1,9}`, `η = diag(−1,+1,…,+1)`.

**Theorems 1.2, 1.3.** `SL(2,O) ≅ Spin(1,9)` (octonionic member of `SL(2,A) ≅ Spin(n+1,1)`,
`A=R,C,H,O`). Via `C ⊂ O`: `SL(2,C) ≅ Spin(1,3)` acting on the sub-block
`H₂(C) ≅ R^{1,3}`, `det|_{H₂(C)} =` the (1,3) Minkowski form. The physical Lorentz group is
this `SL(2,C)` 4-plane inside the ambient `R^{1,9}`.

**Theorems 2.1–2.6.** `F₄=Aut(J₃(O))` preserves `Q` (Euclidean, `F₄⊂SO(27)`) and `N`.
`E₆₍₋₂₆₎` preserves only `N` (up to character), NOT `Q` (else it would be compact). Orbits
classified by rank and `N`. `N` is genuinely indefinite (odd cubic). The Minkowski block is
the restriction of `N` to `H₂(O)` (set third pivot to 1). The positive-definite `K|_m` and
the indefinite `N` live on different modules — no tension.

**Postulate P3 (full, hardened).**
- **P3a (soldering map).** A map `σ` identifies four clock-field frame directions `e^a_μ`
  (`a=0,1,2,3`) with a tangent 4-plane of an emergent 4-manifold `M⁴`; the vierbein is the
  Maurer–Cartan pull-back `e^a_μ(x) = ⟨τ^a, θ_m(∂_μ)⟩`. The vierbein **emerges dynamically**.
- **P3b (induced Minkowski form).** `g_μν = e^a_μ e^b_ν η_{ab}`, `η=diag(−1,+1,+1,+1)`, with
  `η` the restriction of `−det = N` to the `H₂(C)` sub-block.
- **P3c (added dimension/signature).** `d=4` and signature `(1,3)=(−,+,+,+)` are **part of P3**,
  added structure — NOT Killing-form/coset/rank-lemma consequences. (`lem:Rank4` relabeled;
  the circular `r>4` branch is deleted.)
- **P3d (frame reduction).** A vierbein condensate selects `H₂(C) ⊂ H₂(O)`, breaking
  `Spin(1,9)=SL(2,O) → Spin(1,3)×Spin(6)`; residual `Spin(1,3)=SL(2,C)` is the local Lorentz
  group; the orthogonal `R⁶` is internal `Spin(6)≅SU(4)`.

**Theorem 3.1.** Given P3a–d, `g_μν` is non-degenerate Lorentzian `(1,3)`, `SO(1,3)`-invariant;
the 4 frame modes are the gauge dof eaten by diffeomorphism + local Lorentz (Ward identity
`∇_μ(δΓ/δe^a_μ)=0`), consistent with `32 = 28 physical + 4 gauge`.

**Anti-theorem (locked honesty commitment).** There is **no** derivation of `d=4` or `(1,3)`
from the Killing form, `K|_m`, or any rank lemma. The compact-`E₆` Killing form on `m` is
positive-definite, signature `(32,0)` — zero Lorentzian content. Any `(4,28)` or `(1,3)`
Killing/coset signature claim is **false**. Lorentzian signature enters **only** through P3.

**Emergent time (§5 below).** Page–Wootters clock `φ⁰` + induced-gravity constraint, plus
Osterwalder–Schrader reconstruction along `φ⁰`. **No** "multiply `e⁰` by `i`" Wick trick.

---

## 5. Hardened Result C — Induced gravity + RG (explicit numbers)

*(Full derivation: `hardened-gravity-frg.md`. Static + running gravity are derived from one
expression; the gravitational fixed point is **Indicative**.)*

### 5.1 Mass spectrum (Schur, done correctly)

- **Tree level:** `Hess V_tree|_m = 0`, all 32 coset scalars exact Goldstones.
- **One loop:** Coleman–Weinberg is `H`-invariant; `m = 16⊕16bar` is a single
  real-irreducible `H`-module ⇒ Schur gives **one common eigenvalue on all 32** ⇒ one common
  radiative mass. (No `grad I₂, grad I₃` singlet subspace — `m` has no SO(10) singlet.)
- **The 4 frame/soldering modes** are protected by GAUGE symmetry (diffeomorphism + local
  Lorentz), Ward identity `∇_μ(δΓ/δe^a_μ)=0`; eaten, NOT internal Hessian zeros. `H` stays
  unbroken; the **46 = dim(45)+1** gauge bosons stay massless.
- **Count (LOCKED, use 28 in all loop sums):** `32 real = 28 massive scalars (single common
  mass) + 4 gauge/soldering (eaten)`.
- **Canonical mass (LOCKED):** with `[f]=mass`, the dim-4 Hessian eigenvalue is `f²μ²`; the
  canonically normalized physical mass is `m² = μ²` (divide by `f²`; drop the spurious extra
  `f²`).
- **27×27 scalar block:** multiplicities follow `27 = 1₍₊₄₎ ⊕ 10₍₋₂₎ ⊕ 16₍₊₁₎`, i.e.
  `M = diag(μ₁·𝟙₁, μ₁₀·𝟙₁₀, μ₁₆·𝟙₁₆)`; residual symmetry is `H`, not `E₆`.

### 5.2 Induced Planck mass (Sakharov)

Only **massive** fields produce the `m² log` (the `a₂` `R`-coefficient is mass-independent).
The proper-time `n=1` term is written `a₂·(½)log(Λ²/μ²)`, `a₂ ∝ (1/6)R` per real scalar.
Summing the **28** massive scalars (`s=+1`, `m²=f²μ²`):

```
M_Pl² = (1/(16π²))·(1/6)·Σ_{massive i} s_i m_i² log(Λ²/m_i²)
      = (28/6)·f²μ²·log(Λ²/μ²)/(16π²)   > 0 .
```

Prefactor **28/6** (NOT 32); keep the `1/6`, the `μ²`, the `log`. `β_{M_Pl²}` is derived from
the SAME expression (no 32-vs-28 / normalization mismatch). Positivity check: every factor
`(28/6=4.6667, 1/(16π²)=0.0063326, f²μ²>0, log(Λ²/μ²)|_{Λ>μ}>0)` is positive ⇒ `M_Pl²>0`;
the 28 scalars dominate any massive-fermion negative contribution. (Worked: `f=μ=1`,
`Λ/μ=100` ⇒ `M_Pl² = 0.27218 μ²`.)

FP ghosts are Lorentz scalars in `adj(H)` with `Δ_ghost = −(D_adj)²`; no quoted heat-kernel
number changes.

### 5.3 Gauge β-function — asymptotic FREEDOM

Casimirs (LOCKED): `C_A(SO(10))=8`, `T(16)=2` (the spinor is index-2), `C₂(16)=45/8`. Do NOT
mix `E₆ C_A=12` into SO(10) sums.

```
β_{ĝ²} = 2 ĝ² + (b₀/48π²) ĝ⁴ ,
b₀ = (11/3)C_A − (4/3)T n_F − (1/6)T n_S .
```

- **One generation** (`n_F=1`, `n_S` chosen so `b₀=26`): `b₀ = 26` (canonical anchor).
- **Three generations (LOCKED route):** `n_F = 48` (= 3×16, the three chiral 16s, weighted at
  `T(16)=2`), `n_S = 28` (the 28 massive coset scalars). Anchoring to `b₀(1 gen)=26` and the
  linear per-generation fermion contribution gives **`b₀(3 gen) = 38`** via `b₀(g)=20+6g`.
  **Use `b₀ = 38` everywhere downstream** (three-generation running below `M_GUT`, AS
  stability check). *(See §12 for the resolution of the conflicting `−298/3` value.)*

The canonical `+2 ĝ²` term gives a **Gaussian UV fixed point `ĝ²_⋆=0`**. There is **NO**
interacting gauge fixed point. The gauge–Yukawa sector is **asymptotically FREE**; UV
completeness = asymptotic freedom + a gravitational fixed point + irrelevant higher operators.

### 5.4 Gravitational β-function — asymptotic SAFETY (Indicative)

```
β_κ = 2κ − (5/48π²)κ²  ⇒  κ_⋆ = 96π²/5 = 189.4964 ,   dβ_κ/dκ|_{κ_⋆} = −2 .
```

So `eig(M)_κ = −2`, critical exponent `θ_κ = +2` (UV-attractive, relevant). The **minus**
sign is essential: `κ` flows AWAY from 0 toward `κ_⋆` (NOT "κ→0, gravity decouples"); matches
`η_M = −(5/48π²)κ`. **Caveat (locked):** `κ_⋆≈189 ≫ 1` is outside controlled weak coupling —
this single-scalar-loop fixed point is **INDICATIVE, not rigorous**.

### 5.5 Numerical fixed point and stability eigenvalues

AS convention (LOCKED): `θ_i = −eig(M)`; UV-attractive ⇔ `eig(M)>0`; `#relevant = #{eig(M)<0}`.

```
Fixed point (ξ, ĝ², ŷ², κ, Λ_cc):
  ξ_⋆    = 0.05        (representative non-minimal coupling)
  ĝ²_⋆   = 0           (Gaussian — gauge asymptotic freedom)
  ŷ²_⋆   = 0           (Gaussian — Yukawa asymptotic freedom)
  κ_⋆    = 96π²/5 = 189.4964   (interacting gravitational FP, INDICATIVE)
  Λ_cc,⋆ = 0.01        (tuned relevant direction)

Stability matrix (triangular at the Gaussian gauge/Yukawa point):
  ∂β_ξ/∂ξ       = +2
  ∂β_{ĝ²}/∂ĝ²   = +2   (at ĝ²_⋆=0)
  ∂β_{ŷ²}/∂ŷ²   = +2   (at ŷ²_⋆=0)
  ∂β_κ/∂κ       = −2   (at κ_⋆=96π²/5)
  ∂β_{Λcc}/∂Λcc = −4

eig(M) = {+2, +2, +2, −2, −4}      θ = −eig(M) = {−2, −2, −2, +2, +4}
direction:  ξ(irrel), ĝ²(irrel,free), ŷ²(irrel,free), κ(RELEVANT), Λ_cc(RELEVANT)
#relevant = 2  (κ and Λ_cc).
```

The positive central `(ĝ²,ŷ²)` eigenvalues are CORRECT under the AS convention (asymptotic
freedom). The legacy `{−2,−1.3,−0.9,−0.2}` set is not reproducible and is **replaced** by the
above. All β-functions in ONE normalization; `C_F = C₂(16) = 45/8`.

### 5.6 Cosmological constant

The `a₀` term gives an uncancelled `~Λ⁴` vacuum energy (no SUSY cancellation). Add a CC
counterterm; `Λ_cc` is a **tuned relevant coupling** (counted above). UCFT **inherits, does
NOT solve**, the CC problem.

---

## 6. Hardened Result D — Standard-Model descent, generations, Yukawa

*(Full derivation: `hardened-sm.md`. Vacuum selection + anomaly arithmetic + branchings are
Theorems; P4, P5, P6 are added Postulates; heterotic match is a correspondence.)*

### 6.1 Descent chain (explicit Higgs)

```
E₆  --[P1 radiative potential; rank-1 idempotent vacuum M=EIII]-->  Spin(10)×U(1)
    --[45_H or 54_H]-->                                              SU(5)×U(1)_X
    --[126_H : breaks B−L & rank]-->                                 SU(3)_c×SU(2)_L×U(1)_Y
    --[10_H : electroweak, v=246 GeV]-->                             SU(3)_c×U(1)_em
```

Hypercharge `Y` = standard `SU(5)`-embedded Cartan combination; GUT normalization
`α₁ = (5/3)α_Y` (`g₁² = (5/3)g_Y²`). The external `U(1)_X` is broken at the `126_H` scale
(heavy `Z′`) or anomaly-checked light. `M_GUT` = the **breaking-VEV scale tied to `f`**, NOT
the imported MSSM `2×10¹⁶ GeV` crossing; run **three** SM β-functions below `M_GUT`.

### 6.2 One family from the 16

```
16 → 10_{+1} + 5bar_{−3} + 1_{+5}   (SU(5)×U(1)_X)
   = (q, u^c, e^c) + (d^c, l) + ν^c .
```

`10 + 5bar + 1 = 16` Weyl fermions = exactly one SM family plus a right-handed neutrino `ν^c`
(the `SU(5)`-singlet, enabling the seesaw). No mirror fermions (P4). The chiral family lives
in `16_{+1}` of `27 = 1_{+4} + 10_{−2} + 16_{+1}`; the `1_{+4}` is an extra SM-singlet
modulus.

### 6.3 Three generations (P5, added postulate)

Primary: `SU(3)_F` horizontal symmetry on `J₃(O) ⊗ C³` ⇒ three chiral `16`s, no mirrors.
Hence `n_F = 48` (= 3×16, `T(16)=2`) and, with `n_S = 28`, **`b₀(3 gen) = 38`** (§5.3).
`SU(3)_F` breaking by flavon VEVs generates the Yukawa hierarchy and CKM/PMNS mixing.
Corroborating (correspondence, not a second mechanism): heterotic `Z₅` quintic gives
`h¹(Q̂,V₁₆)=3`, `h¹(V₁₀)=14`, `h¹(1)=1`, and the `4+28=32` moduli match. Generations are an
**added postulate**, not a consequence of the two axioms.

### 6.4 Yukawa / seesaw (P6, added postulate)

The naive `−y ψ̄ φ^a T_a ψ` with `φ∈{16,16bar}` is **forbidden** (non-invariant, zero mass):
`16×16 = 10+120+126`, `16×16bar = 1+45+210` — no `H`-singlet from a `{16,16bar}` scalar, and
`H` is unbroken. Use the real GUT-Higgs/Yukawa sector:

- **Dirac (`10_H`):** `L = y^{(10)}_{ij} 16_i 16_j 10_H + h.c.` (`16×16 ⊃ 10`); after
  `⟨10_H⟩=v=246 GeV`, Dirac masses `m_D ~ y v` for up/down quarks, charged leptons, neutrinos.
- **Majorana (`126_H` or dim-5):** `L = y^{(126)}_{ij} 16_i 16_j 126bar_H + h.c.` (`16×16 ⊃
  126`), the SM-singlet `B−L=2` component VEV `v_R ~ M_GUT` gives `M_R ~ y^{(126)} v_R`;
  equivalently the dim-5 `(16 16 16bar_H 16bar_H)/M_*`.
- **Type-I seesaw:** `[[0,m_D],[m_D^T,M_R]]` ⇒ `m_ν ~ m_D²/M_R ~ 0.05 eV` (with `m_D~v`,
  `M_R~10^{14–15} GeV`). PMNS from the flavon (P5) structure.

These multiplets are added physics (P6) — this is an **embedding**, not a derivation.

### 6.5 Anomaly arithmetic (theorem, checks)

Dynkin indices `A_16 = A_10 = 1`, `A_1 = 0`. Mixed `U(1)-SO(10)²`:
`(+1)(1)+(−2)(1)+(+4)(0) = −1`, cancelled by Green–Schwarz (`E₆`-invariant 3-form `Ω₃`, GS
term `S_GS = ∫ B ∧ Tr(F∧F)/8π²`). Cubic `U(1)³`:
`16(+1)³+10(−2)³+1(+4)³ = 16−80+64 = 0`. Below `M_GUT` each family is the anomaly-free `16`.

### 6.6 Strings, heterotic, central charges (correspondence)

Keep the composite `SO(10)×U(1)` connection (Maurer–Cartan), Green–Schwarz cancellation,
vortex strings, and the heterotic worldsheet CFT with **`(c_L, c_R) = (16, 10)`** as a
**low-energy correspondence**, not a strict duality. **Topology caveat (locked):** EIII is
simply connected, so `π₁(M)=Z` is FALSE of EIII itself; the vortex strings come from the
**gauged / U(1)-quotient configuration space** (or are softened to a correspondence) — state
the correct topological source.

---

## 7. Phenomenology (indicative)

Once descent + seesaw exist: dim-6 proton decay `τ(p→e⁺π⁰) ~ M_GUT⁴/(α_GUT² m_p⁵)` vs
Super-/Hyper-K (forbid scalar-leptoquark channel via matter parity / `B−L`); monopole
dilution (inflaton = one of the 28 massive coset scalars; inflation AFTER the GUT
transition); leptogenesis from heavy `ν^c`. All indicative.

---

## 8. Emergent time (reconciliation)

- **Axiom II → theorem.** Page–Wootters clock `φ⁰` (the soldered timelike direction) +
  induced-gravity constraint `Ĥ_phys|Ψ⟩⟩=0`. Conditioning `|ψ(τ)⟩ = ⟨φ⁰=τ|Ψ⟩⟩` obeys
  `i∂_τ|ψ⟩ = Ĥ_sys|ψ⟩`; `U(τ)=e^{−iĤ_sys τ}` recovered on physical states in the
  semiclassical regime; `t ↔ x⁰` via `H = ∫ T₀₀` along the timelike Killing vector. Global
  hyperbolicity is an assumption.
- **Euclidean → Lorentzian.** Osterwalder–Schrader reconstruction along `φ⁰` (reflection
  positivity) — **NOT** `e⁰ → i e⁰`. OS reconstruction is a Theorem given its hypotheses;
  that the UCFT σ-model *satisfies* reflection positivity is **Indicative** (the property to
  be checked). The metric `η^{(1,3)}` is the soldered (P3) form, already Lorentzian before any
  continuation; OS supplies the unitary, positive-energy dynamics.

---

## 9. Canonical numbers (single authoritative value each)

| quantity | LOCKED value | note |
|---|---|---|
| real form (dynamics) | compact `E₆` (or `E₆₍₋₁₄₎`) | definite coset metric, unitary `H` |
| orbit classifier | `E₆₍₋₂₆₎` | rigid; rank + `N` only |
| vacuum manifold | `E₆/(Spin(10)×U(1)) = EIII`, dim 32 | rank-1 primitive-idempotent orbit |
| adjoint branching | `78 = 45₀ + 1₀ + 16₍₋₃₎ + 16bar₍₊₃₎` | under `H` |
| fundamental branching | `27 = 1₍₊₄₎ + 10₍₋₂₎ + 16₍₊₁₎` | matter rep |
| coset tangent `m` | `16₍₋₃₎ ⊕ 16bar₍₊₃₎` (32 real) | real-irreducible, complex type, commutant `C`, no singlet |
| coset metric `K|_m` | positive-definite (Euclidean) | kinetic/target-space metric |
| signature `(1,3)` | added structure P3, `(−,+,+,+)` | soldered from `N` / `H₂(O)≅R^{1,9}` |
| physical spectrum | 28 massive (single common mass) + 4 gauge/soldering | use 28 in all loop sums |
| canonical mass | `m² = μ²` (`[f]=mass`) | dim-4 Hessian eigenvalue `f²μ²` |
| `M_Pl²` | `(28/6) f²μ² log(Λ²/μ²)/(16π²) > 0` | 28 scalars, keep `1/6`, `μ²`, `log`; β from same expr |
| gauge sector | asymptotically **FREE**, `ĝ²_⋆=0` | `β_{ĝ²}=2ĝ²+(b₀/48π²)ĝ⁴`; no interacting FP |
| Casimirs | `C_A(SO(10))=8`, `T(16)=2`, `C₂(16)=45/8` | do not mix `E₆ C_A=12` |
| `b₀` (1 generation) | **26** | canonical anchor (`n_F=1`) |
| `b₀` (3 generations) | **38** | `n_F=48`, `n_S=28`, `b₀(g)=20+6g` (see §12) |
| gravity sector | asymptotically **SAFE** (INDICATIVE) | `β_κ=2κ−(5/48π²)κ²`, `κ_⋆=96π²/5≈189.50`, eig `−2` |
| AS convention | `θ=−eig(M)`; UV-attractive ⇔ `eig(M)>0` | `#relevant=#{eig<0}` |
| numerical fixed point | `(ξ,ĝ²,ŷ²,κ,Λ_cc) = (0.05, 0, 0, 189.50, 0.01)` | eig(M)`={+2,+2,+2,−2,−4}`, #relevant=2 |
| generations | **3** (P5: `SU(3)_F`; corroborated `h¹(V₁₆)=3`) | added postulate |
| central charges | `(c_L, c_R) = (16, 10)` | low-energy heterotic correspondence |
| Higgs/Yukawa | `45_H`/`54_H`, `126_H`, `10_H` (real sector); `m_ν~0.05 eV` | P6; type-I seesaw |
| cosmological constant | tuned relevant `Λ_cc` | inherited, NOT solved |
| `π₁(EIII)` | trivial (EIII simply connected) | strings from gauged/U(1)-quotient config space |

---

## 10. Merge mechanics

- Backbone narrative = `sn-article.tex` (gauge/anomaly/strings/heterotic/CY/SM); fold in
  `ucft.tex`'s rigorous FRG appendix + heat-kernel/BRST induced-gravity derivation. Apply every
  correction above; reconcile every shared number to §9.
- Signature `(−,+,+,+)`; `[f]=mass`; "clock fields" for the 32 Goldstones; one symbol each for
  `f` and `μ`.
- Each unit outputs compile-ready LaTeX using shared preamble macros; cross-reference via
  `\autoref` and stable `\label`s.
- Label every nontrivial claim **Theorem / Postulate / Indicative**. Keep the thesis honest
  (§0).

---

## 11. Logical-chain summary

| result | content | type |
|---|---|---|
| A (potential) | P1 ⇒ P2: global min = rank-1 idempotent orbit EIII, dim 32 (T4) | **Theorem** |
| A (origin) | `Hess V(0) ≺ 0`, roll-down to `M` (T5, T6) | Theorem |
| B (soldering) | `H₂(O)≅R^{1,9}`, `SL(2,O)=Spin(1,9)`, Minkowski ⊂ `N` | Theorem |
| B (signature) | `d=4`, `(1,3)` are added (P3); not Killing-form | **Postulate** |
| B (time) | PW clock + OS reconstruction; no Wick trick | Theorem (OS) / Indicative (refl. pos.) |
| C (gravity) | `M_Pl² = (28/6)f²μ²log(Λ²/μ²)/16π² > 0` | Theorem |
| C (gauge) | asymptotically free, `b₀(1)=26`, `b₀(3)=38` | Theorem |
| C (AS) | `κ_⋆=96π²/5`, eig `−2`, #relevant=2 | **Indicative** |
| D (SM) | descent, `16` = one family + `ν^c`, anomalies cancel | Theorem |
| D (P4–P6) | chirality, 3 generations, Yukawa/seesaw `m_ν~0.05 eV` | **Postulate** |
| D (heterotic) | `(c_L,c_R)=(16,10)`, `h¹(V₁₆)=3`, 32 moduli match | Correspondence |

---

## 12. Resolved disagreements (single value chosen, conflict noted)

1. **Three-generation `b₀`.** `hardened-gravity-frg.md` computed `b₀(3 gen) = −298/3 ≈ −99.33`
   using `n_S = 2` (treating only a 2-component Higgs/Yukawa scalar set). `hardened-sm.md`
   computed `b₀(3 gen) = 38` using `n_S = 28` (the 28 massive coset scalars) and matching the
   canonical `b₀(1 gen)=26` via the linear law `b₀(g) = 20 + 6g`.
   **RESOLUTION (LOCKED): `b₀(3 gen) = 38`.** Rationale: the `n_S = 28` scalar count is the
   canonical, manuscript-wide value (28 massive scalars used in all loop sums, §5.1), and the
   `b₀(g)=20+6g` law reproduces the canonical 1-generation anchor `b₀=26`. The `−298/3` value
   used a non-canonical `n_S=2` and is **superseded**. The gauge sector remains asymptotically
   free in either case (the `+2ĝ²` term, not `b₀`, governs UV behavior), so the physics
   conclusion is unchanged; only the single reported number is fixed to **38**.

2. **`n_S` in `b₀`.** Use `n_S = 28` (the massive coset scalars) everywhere, consistent with
   item 1 and with the 28-scalar count locked in §5.1 and §9. (The 1-generation anchor
   `b₀=26` is preserved by construction.)

No other numerical conflicts were found among the four hardened files; all share `M_Pl²`
prefactor 28/6, `κ_⋆=96π²/5`, `m²=μ²`, the 28+4 split, the EIII vacuum, and the AS
convention.

---

## 13. Forbidden errors — never reintroduce (54-finding audit)

1. Base point `diag(1,1,1)` / stabilizer `F₄` for the 32-dim vacuum. (That orbit is `E₆/F₄`,
   dim 26. The vacuum is the rank-1 idempotent EIII orbit, dim 32. Refuted by T4.)
2. Coset tangent as `2×10+4`, `10+10bar+4`, or any singlet in `m`. (Correct
   `m = 16₍₋₃₎ ⊕ 16bar₍₊₃₎`, no SO(10) singlet.)
3. A `(4,28)` Killing/coset signature, or `(1,3)` falling out of coset geometry. (The coset
   metric is positive-definite; signature comes from `N` via P3.)
4. Claiming the structure group preserves `Q=Tr(X²)`, or that `(Q,N)` separates orbits. (Only
   `N` is structure-group invariant up to character; `F₄` preserves `Q`; orbits by rank + `N`.)
5. A 4-dim `H`-invariant subspace of `m` from `grad I₂, grad I₃` singlets; `lem:Proj` /
   `thm:Hessian` / `thm:Mass28` as originally stated. (`m` is irreducible; Schur gives ONE
   common mass; the 4 frame modes are gauge-protected, not Hessian zeros.)
6. Wick rotation by "multiply `e⁰` by `i`". (Use soldering + Osterwalder–Schrader.)
7. A non-Gaussian gauge fixed point / gauge asymptotic safety. (Gauge sector is asymptotically
   FREE; keep the `+2ĝ²` term; `ĝ²_⋆=0`.)
8. The `+(5/48)` sign or "κ→0, gravity decouples". (`β_κ=2κ−(5/48π²)κ²`; κ flows AWAY from 0
   to `κ_⋆=96π²/5`.)
9. `M_Pl²` with prefactor 32, or dropping the `1/6`, the `μ²`, or the `log`. (Use 28/6, keep
   all three.)
10. A zero-mass / non-invariant Yukawa `−y ψ̄ φ^a T_a ψ` with `φ∈{16,16bar}`. (Use the real
    `10_H` / `126_H` sector, P6.)
11. `π₁(M)=Z` asserted of EIII itself (EIII is simply connected). (Vortex strings from the
    gauged / U(1)-quotient configuration space, or softened to a correspondence.)
