# UCFT — Authoritative Reconstruction Spine

This document is the **single source of truth** for merging `sn-article.tex` (Clock-Field
draft: composite gauge sector, Green–Schwarz anomaly cancellation, vortex/heterotic
correspondence, Z₅ quintic → 3 generations, SO(10) → SM descent) and `ucft.tex` (Coset
draft: functional RG, induced gravity, heat-kernel machinery) into one internally
consistent manuscript that reproduces the Standard Model and gravitation.

It supersedes any conflicting statement in either source. It is grounded in a 54-finding
adversarial audit (`/tmp/.../tasks/w6dgrrae0.output`). Every section/appendix of the merged
paper MUST obey the commitments below. Where a source claim contradicts this spine, the
spine wins.

UCFT = **Universal Coset Field Theory**, by Brandon Belna.

---

## 0. Framing (honest thesis)

UCFT is an **E₆ / Albert-algebra–motivated unification**: two axioms fix the order
parameter and its quantum kinematics; a small, **explicitly enumerated** set of additional
postulates (P1–P6 below) then yields a 4-D Lorentzian quantum field theory containing an
anomaly-free `SO(10)×U(1)` gauge sector, Sakharov-induced gravity, and — after a stated
GUT-Higgs/flavor sector — the Standard Model with three generations and neutrino masses.

The phrase "everything follows from two axioms" is **demoted**: the axioms supply the
arena; P1–P6 are added structure, stated up front. This is a *derivation-grade embedding*,
not a from-nothing derivation. Claims are labeled **theorem** (proved), **postulate**
(assumed), or **conjecture/indicative** (motivated, not controlled).

**Added postulates (state these explicitly, once, near the axioms):**
- **P1 (Potential class).** An `E₆`-invariant polynomial potential built from the cubic
  norm `N` and the sharp map `X#`, whose global minimum is the rank-1 idempotent orbit.
- **P2 (Vacuum selection).** The vacuum is that minimal orbit, `M = E₆/(Spin(10)×U(1))`.
- **P3 (Soldering & signature).** A soldering map identifies four clock-field directions
  with a tangent frame carrying the Minkowski form inherited from the cubic-norm /
  `H₂(O) ≅ R^{1,9}` structure; `d=4` and signature `(1,3)` enter here (see §3).
- **P4 (Matter content & chirality).** One chiral `16` of `Spin(10)` per family (the SO(10)
  generation), with the stated `U(1)` charges.
- **P5 (Family symmetry).** Three generations via an `SU(3)_F` horizontal symmetry acting
  on `J₃(O) ⊗ C³` (equivalently three `27`s); equivalently realized by the heterotic
  `h¹(Q̂,V₁₆)=3` count (§8). Pick ONE as primary, mention the other as corroboration.
- **P6 (GUT-Higgs / Yukawa sector).** `SO(10)→SM` Higgs multiplets (`45_H`/`54_H` + `126_H`)
  and Yukawa multiplets (`10_H`, `126_H`) generating fermion masses and the seesaw.

---

## 1. The real-form / base-point foundation  *(BLOCKER — root fix; everything depends on it)*

**Do NOT conflate two E₆'s.**

- The automorphism group of `J₃(O)` (preserving the Jordan product, hence BOTH `Tr(X²)`
  and `N`) is the **compact `F₄`** (dim 52).
- The **reduced structure group** (preserving the cubic norm `N` up to scalar) of the
  *division*-octonion Albert algebra is the **non-compact `E₆₍₋₂₆₎`** (dim 78). It does
  **not** preserve `Tr(X²)`.
- `Stab_{E₆}(diag(1,1,1)) = F₄`, orbit `E₆/F₄` of real dim **26** — NOT the 32-dim coset.
  No orbit of a *vector* in the 27 has dim > 26.

**The 32-dim vacuum manifold is the EIII Hermitian symmetric space:**
```
M = E₆ / (Spin(10) × U(1)) ,   dim_R M = 32 ,
```
realized as the **closed rank-1 (minimal) E₆-orbit in P(27_C)** — the complex Cayley plane,
i.e. the orbit of a **rank-1 primitive idempotent**, NOT the orbit of `diag(1,1,1)`. Define
it via the involution datum
```
78 = 45₀ ⊕ 1₀ ⊕ 16₍₋₃₎ ⊕ 16bar₍₊₃₎      (adjoint branching under H = Spin(10)×U(1)).
```

**Real form to commit to:** use the **compact `E₆`** (or its dual `E₆₍₋₁₄₎`, whose maximal
compact is exactly `Spin(10)×U(1)`) for the **dynamics**: then `H` is compact, the coset
metric (Killing-induced) is **positive-definite** → healthy kinetic term + unitary gauge
sector. Use `E₆₍₋₂₆₎` only as the rigid orbit-classifier of the real 27. Name the real form
explicitly wherever signature/positivity is invoked.

**Invariants / orbit classification (corrected):**
- Only the cubic norm `N` is structure-group invariant (up to a character `λ(g)`).
  `Q(X)=Tr(X²)` is **`F₄`-invariant only**. State this.
- Real `E₆`-orbits are classified by **rank (1,2,3) and `N`**, NOT by `(Q,N)`. Delete the
  false "(Q,N) separates orbits" and "structure group preserves Q" claims.
- `rank X ≤ 1 ⇔ X# = 0`, where `X#` is the `E₆`-covariant sharp (adjoint) map.

**Potential (P1, corrected so the minimum is the rank-1 orbit):**
```
V(X) = α · ⟨X#, X#⟩  +  β · [Tr(X²) − c₂]²  +  γ · Tr(X²)^k ,   α,β,γ>0, k≥2 even.
```
The `X#` term (genuinely `E₆`-covariant) vanishes exactly on rank ≤ 1; the `Tr(X²)` terms
are `F₄`-invariant **selectors** that fix the scale/representative within the `N`-level set.
State that coercivity uses the compact real form (non-compact orbits cannot be coercive).

---

## 2. Coset tangent space & branchings  *(BLOCKER)*

Authoritative, use everywhere (replace every "2×10+4" / "10+10bar+4" — they don't even sum
to 32/28 and use the wrong (fundamental, not adjoint) reps):
```
coset tangent  m  =  16₍₋₃₎ ⊕ 16bar₍₊₃₎      (32 real; chiral spinor pair of Spin(10)).
matter  27  =  1₍₊₄₎ ⊕ 10₍₋₂₎ ⊕ 16₍₊₁₎ .
```
- `m` is **real-irreducible of complex type** (commutant `= C`): the `U(1)` charge forbids
  the symmetric `16×16` and `16bar×16bar` invariants, leaving the unique charge-0
  cross-pairing = Killing restriction. So the invariant metric is **unique and definite**
  (Schur) — keep `thm:uniqueMetric`'s *result*, replace its proof with this U(1)-charge
  argument; keep `lem:Irrep`'s *conclusion*, replace "single weight orbit" with the
  Hermitian-symmetric-space argument.

---

## 3. Emergent spacetime & Lorentzian signature  *(BLOCKER — the honest Lorentzian mechanism)*

**The coset metric `K|_m` is positive-definite (Euclidean). Do NOT claim a `(4,28)` Killing
signature or that `(1,3)` falls out of the coset geometry — both are false.** Keep `K|_m`
positive-definite as the **target-space (kinetic) metric** the σ-model needs.

**Where Lorentzian signature genuinely comes from (honors the author's intent):** the
indefinite invariant of `J₃(O)` is the **cubic norm `N`** (the trace form is the Euclidean
`F₄`-invariant). Inside `J₃(O)` the `2×2` Hermitian-octonionic block is
`H₂(O) ≅ R^{1,9}` with `det = Minkowski norm` and `SL(2,O) ≅ Spin(1,9)`. The **soldering
map (P3)** identifies the four clock-field frame directions `e^a_μ` with a 4-plane carrying
the Minkowski form `η_{ab}=diag(-1,+1,+1,+1)` inherited from this cubic-norm / `R^{1,9}`
structure. Thus:

- The vierbein/frame `e^a_μ` **emerges dynamically** (pull-back of the Maurer–Cartan form);
- the **signature `(1,3)` is carried by the soldering form rooted in the cubic norm**, and
  is an **explicit added structure (P3)**, not a Killing-form theorem.
- Optionally strengthen to a *mechanism*: a Wick rotation tied to a selected timelike clock
  field `φ⁰` (Osterwalder–Schrader reconstruction along `φ⁰`), or a signature-selecting
  vierbein condensate breaking the frame symmetry to `SO(1,3)`. State which route is taken.

**`d=4`:** `lem:Rank4` is currently circular (`φ` is defined on `R⁴`, so the `r>4` branch is
vacuous). Either (a) honestly relabel it as *fixing the world-manifold dimension by the
soldering postulate* (delete the impossible `r>4` branch), or (b) keep `d` symbolic and
argue `d=4` is selected by induced-gravity finiteness / the asymptotic-safety window
(research option). Default: **(a)** — `d=4` is part of P3.

Convention: signature `(-,+,+,+)` everywhere (adopt sn-article's).

---

## 4. Mass spectrum  *(BLOCKER — Schur done correctly)*

- **Tree level:** `V` is `E₆`-invariant ⇒ all **32** coset scalars are exact Goldstones,
  massless (`Hess V_tree|_m = 0`), consistent with Goldstone's theorem.
- **One loop:** the Coleman–Weinberg effective potential is `H`-invariant; since
  `m = 16⊕16bar` is a **single real-irreducible `H`-module**, Schur ⇒ **one common
  eigenvalue on all 32** ⇒ common radiative mass. (This USES irreducibility; it does not
  fight it.)
- **Delete** the invalid "4-dim `H`-invariant subspace of `grad I₂, grad I₃` singlets"
  (`lem:Proj`, `thm:Hessian`, `thm:Mass28`): `m` has NO `SO(10)` singlets, and
  `grad I₂|_{X₀}=2X₀` is not in the coset.
- **The 4 frame directions are protected by GAUGE symmetry** (world-volume diffeomorphism +
  local Lorentz), not by an internal Hessian zero: they are the eaten/soldering modes
  (composite vierbein bilinear), with Ward identity `∇_μ(δΓ/δe^a_μ)=0`. This keeps `H`
  unbroken (the 46 gauge bosons stay massless) and removes the contradiction.
- **Count:** `32 real = 28 physical massive scalars (common mass) + 4 gauge/soldering`.
  Use **28** in all loop sums.
- Mass normalization: with `[f]=mass`, the canonically normalized physical mass is
  `m² = μ²` (drop the spurious extra `f²`; the dim-4 Hessian eigenvalue is `f²μ²` but the
  canonical mass divides by `f²`). State `[f]=mass` globally; `ĝ², ξ, κ` are dimensionless.

---

## 5. Induced gravity (Sakharov)  *(BLOCKER — recompute cleanly)*

One scheme (ζ / dim-reg). Only **massive** fields produce the `m² log` (the `a₂` R-coefficient
is mass-independent, so massless fields contribute only to the subtracted quadratic
divergence):
```
M_Pl² = (1/(16π²)) · (1/6) · Σ_{massive i} s_i m_i² log(Λ²/m_i²)
      ≈ (28/6) · f² μ² /(16π²) · log(Λ²/μ²)   > 0,
```
dominated by the 28 massive scalars (`s=+1`, `m²=f²μ²`). Prefactor **28/6**, NOT 32; keep the
`1/6` and the `μ²`. Derive `β_{M_Pl²}` from the SAME expression so static and running agree
(this removes the ~38× and 32-vs-28 discrepancies and the unexplained `5/48`). Write the
`n=1` proper-time term explicitly as `a₂·(1/2)log(Λ²/μ²)` (the bare formula is singular `0/0`
there). **Verify `M_Pl²>0`** (28 scalars dominate any massive-fermion negative contribution).

Cosmological constant: the `a₀` term gives an uncancelled `~Λ⁴` vacuum energy (no SUSY
cancellation). **Add a CC counterterm; declare `Λ_cc` a tuned relevant coupling.** State
plainly: UCFT *inherits*, does not solve, the CC problem. Add `Λ_cc` to the relevant-direction
count.

Fix the ghost operator: FP ghosts are Lorentz scalars in `adj(H)`; use the minimal adjoint
covariant Laplacian `Δ_ghost = -(D_adj)²` (the `γ^μγ^ν` form is ill-defined on scalars). No
quoted heat-kernel number changes.

---

## 6. Gauge coupling & FRG  *(BLOCKER — signs, Casimirs, framing)*

**Casimirs (fix once, regenerate all tables):** the relevant gauge group that runs is
`SO(10)` with `C_A(SO(10)) = 8`, `T(16) = 2` (the spinor is index-2, NOT 16 index-1
fundamentals), `C₂(16) = 45/8`. (Do not mix the `E₆` `C_A=12` into SO(10) matter sums.)

**Gauge β (asymptotic FREEDOM, not safety):** restore the dropped canonical term:
```
β_{ĝ²} = 2 ĝ² + (b₀/48π²) ĝ⁴ ,
```
with `b₀ = (11/3)C_A − (4/3)T n_F − (1/6)T n_S` evaluated with the corrected Casimirs and the
chosen `n_F` (one SO(10) generation gives `b₀ = 26`; with three generations `n_F` and `b₀`
shift — recompute and STATE the number used). The gauge coupling is **asymptotically free**
(Gaussian UV point) — there is no interacting `ĝ²_*>0`. Recast `thm:FixedPoint` /
`thm:UVcomplete` as: **asymptotic freedom of the gauge–Yukawa sector + a gravitational
fixed point + irrelevant higher operators**, NOT a non-Gaussian gauge fixed point.

**Gravitational κ (asymptotic SAFETY — minus sign):**
```
β_κ = 2κ − (5/48π²) κ² ,   ⇒  UV-attractive fixed point  κ_* = 96π²/5  (eigenvalue −2).
```
This matches the appendix `η_M=−(5/48π²)κ`. **Delete** the wrong `+` sign, the wrong
`lem:kappaIrrel` ("κ→0, gravity decouples" — it flows AWAY from 0), and the wrong closed
form; quote the logistic solution. Caveat honestly: `κ_*≫1` is outside controlled weak
coupling — this single-scalar-loop fixed point is **indicative**, not rigorous.

**Stability matrix:** adopt the standard AS convention `θ_i = −eig(M)` (UV-attractive ⇔
`eig(M)>0`; #relevant = #`eig(M)<0`). STATE the numerical fixed point
`(ξ_*, ĝ²_*, ŷ²_*, κ_*)` explicitly (currently never given), build the TRUE Jacobian
**including canonical terms**, recompute `σ(M)` honestly (the central `2×2` block has
positive eigenvalues — that is *correct* under the AS convention; the quoted
`{-2,-1.3,-0.9,-0.2}` set is not reproducible and must be replaced by the recomputed values).
Fix `C_F=45/8` and the bare-vs-divided-`b₀` normalization (write every β in ONE
normalization); ship or correct the ancillary `twoLoopCoeffs`.

**27×27 scalar block:** multiplicities must follow `27 → 1₄ + 10₋₂ + 16₁`, i.e.
`M = diag(μ₁·1₁, μ₁₀·1₁₀, μ₁₆·1₁₆)` with three computed (verified-negative) eigenvalues, not
the spurious `{6 singles + (×21)}` list. Residual symmetry is `H`, not `E₆`.

**Higher-curvature:** drop the dimensionally-wrong `α ~ a₄/M_Pl²` sketch. `R²`/`Weyl²` are
classically marginal (2 independent 4-D invariants after Gauss–Bonnet); either compute their
anomalous dimensions from a curved-background truncation or downgrade `prop:irrelevant` to an
assumption and make `thm:UVcomplete` conditional on the truncation.

---

## 7. Axioms → vacuum, and time  *(MAJOR)*

**Axiom I → broken vacuum.** The 27 is a non-trivial irreducible `E₆` rep ⇒ the only vector
fixed by the whole structure group is `Φ=0`. Two honest options; default **(B)**:
- (A) Reinterpret Axiom I as "the initial *state* is the unique `E₆`-invariant measure
  `∝ e^{−V}`"; breaking = clustering of the partition function on the `V`-minimizing orbit.
- (B) Keep `Φ=0` as the symmetric origin; note the existing `V` already makes it a strict
  local **maximum** (`Hess V(0)` negative-definite — the destabilizing term is present), so
  the rank-1 orbit beats the origin; supply the roll-down dynamics. **Demote the
  "all constructions follow from two axioms" line** and enumerate P1–P6.

**Axiom II ↔ emergent time.** Axiom II posits a global `U(t)=e^{−iHt}` with external time;
§3 makes time a Goldstone/clock. Reconcile (default): **demote Axiom II to a theorem** — keep
Axiom I + a quantization prescription; recognize the diffeomorphism/Hamiltonian constraint
`H_phys|ψ⟩=0` implied by induced gravity; designate the timelike clock `φ⁰` a **Page–Wootters
clock**; recover `U(τ)` on physical states in the semiclassical/background regime (timelike
Killing vector). Add a short subsection "Reconciliation of axiomatic and emergent time" with
the explicit `t ↔ x⁰` map (`H = ∫ T₀₀` along the timelike Killing direction) and the group
law on physical states; flag global hyperbolicity as an assumption.

---

## 8. Standard Model descent, generations, fermion masses  *(MAJOR — the "reproduces SM" core)*

Keep sn-article's composite gauge sector + Green–Schwarz anomaly cancellation (the GS
arithmetic checks: mixed `U(1)-SO(10)²` from `16·1+10·(−2)+1·4 = −6`... use Dynkin indices
`A_16=A_10=1, A_1=0` ⇒ `1·1+(−2)·1+4·0 = −1`, cancelled by GS; cubic `U(1)³`:
`16·1³+10·(−2)³+1·4³ = 16−80+64 = 0`). Then ADD (P5, P6):

**Descent chain (explicit Higgs):**
```
E₆ →(P1 potential)→ SO(10)×U(1)
    →(45_H or 54_H)→ SU(5)×U(1)_X  [or Pati–Salam SU(4)×SU(2)×SU(2)]
    →(126_H, breaks B−L & rank)→ SU(3)_c×SU(2)_L×U(1)_Y
    →(10_H, electroweak)→ SU(3)_c×U(1)_em .
```
Hypercharge `Y` = standard SU(5)-embedded Cartan combination, GUT normalization
`α₁ = (5/3)α_Y`. The external `U(1)_X` is broken (heavy `Z′`) or anomaly-checked light.
Redefine `M_GUT` as the **breaking-VEV scale** (tied to `f`), not the imported MSSM `2×10¹⁶`
crossing; run **three** SM β-functions below `M_GUT`, not a single `b₀`.

**Per family:** `16 = (q, uᶜ, dᶜ, l, eᶜ, νᶜ)` (one generation + right-handed neutrino);
`10_H` → electroweak Higgs doublets (`v=246 GeV`); the `16`'s SM-singlet `νᶜ` → seesaw.

**Three generations (P5):** primary route — `E₆ × SU(3)_F` horizontal symmetry on
`J₃(O) ⊗ C³` (three chiral `16`s, no mirrors); recompute `n_F` (=48) and hence `b₀`, the
fixed point, `M_GUT`. `SU(3)_F` breaking (flavon VEVs) generates the hierarchy and CKM/PMNS.
Corroborating route — the heterotic `Z₅` quintic already gives `h¹(Q̂,V₁₆)=3` (§ heterotic):
present as independent support, not a second mechanism. State generations are an **added
postulate**, not a consequence of the original two axioms.

**Yukawa / masses (P6):** the source Yukawa `−y ψ̄ φ^a T_a ψ` with `φ∈{16,16bar}` is
**non-invariant and gives zero mass** (`16×16 = 10+120+126`, `16×16bar = 1+45+210`: no
`H`-singlet from a `{16,16bar}` scalar) AND `H` is unbroken at the vacuum. Replace with a
genuine Higgs/Yukawa sector: `10_H` (Dirac masses, doublets, `v=246`), `126_H/351′_H` or the
dim-5 operator `(16·16·16bar_H·16bar_H)/M_*` for the Majorana `M_R` of `νᶜ` (type-I seesaw,
`m_ν = m_D²/M_R ~ 0.05 eV`), family-indexed `y_ij 16_i 16_j H`. State these multiplets are
added physics (P6) — this converts "derivation" into "embedding," say so.

**Cosmology/phenomenology (indicative):** once descent + seesaw exist, estimate dim-6 proton
decay `τ(p→e⁺π⁰) ~ M_GUT⁴/(α_GUT² m_p⁵)` vs Super-/Hyper-K (forbid scalar-leptoquark channel
via matter parity / B−L); monopole dilution (inflaton = one of the 28 massive coset scalars,
inflation AFTER the GUT transition); leptogenesis from heavy `νᶜ`.

---

## 9. Keep (good content from sn-article, made consistent)

- **Composite `SO(10)×U(1)` connection** from the Maurer–Cartan form; induced Yang–Mills
  term with the **corrected** `g⁻²` (use the right massive multiplet `16⊕16bar`, total Dynkin
  index 4 and dim 32 — NOT the wrong `2×10+4` with index 5/2).
- **Green–Schwarz anomaly cancellation** (cubic-norm 3-form `Ω₃`, GS term, the arithmetic
  above).
- **Vortex strings & heterotic worldsheet CFT:** `π₁(M)=Z`, BPS finite tension, left-moving
  `SO(10)₁ ⊕ U(1)₁` affine algebra, `(c_L,c_R)=(16,10)`. Frame as a **low-energy
  correspondence**, not a strict duality. (Note: `π₁(EIII)` — verify; EIII is simply
  connected as a symmetric space, so the `π₁(M)=Z` claim must be re-examined — likely the
  relevant `π₁` comes from the *gauged*/quotient configuration space or the `U(1)` factor;
  state the correct topological source of the strings or soften the claim.)
- **Z₅ quintic correspondence:** `32 = 4 + 28` moduli match, `h¹(V₁₆)=3` generations,
  moduli-space diffeomorphism, coupling/Newton-constant matching. Keep as corroborating
  low-energy correspondence.

---

## 10. Reconciled numbers (one authoritative value each)

| quantity | value | note |
|---|---|---|
| real form (dynamics) | compact `E₆` (or `E₆₍₋₁₄₎`) | definite coset metric, unitary `H` |
| vacuum manifold | `E₆/(Spin(10)×U(1))` = EIII, dim 32 | rank-1 idempotent orbit |
| coset tangent `m` | `16₍₋₃₎ ⊕ 16bar₍₊₃₎` | real-irreducible, complex type |
| coset metric `K|_m` | positive-definite | kinetic/target metric |
| signature `(1,3)` | added structure (P3) | soldering from cubic norm / `H₂(O)≅R^{1,9}` |
| physical scalars | 28 massive (common mass) + 4 gauge | from §4 |
| `m²` (canonical) | `μ²` (`[f]=mass`) | drop spurious `f²` |
| `M_Pl²` | `(28/6) f²μ² log(Λ²/μ²)/(16π²) > 0` | 28 scalars, keep `1/6`, `μ²` |
| gauge sector | asymptotically **free** | `β_{ĝ²}=2ĝ²+(b₀/48π²)ĝ⁴` |
| `b₀` | recompute with `C_A=8,T(16)=2`; state n_F used | `26` for 1 gen; shifts for 3 gen |
| gravity | asymptotically **safe** | `β_κ=2κ−(5/48π²)κ²`, `κ_*=96π²/5` (indicative) |
| AS convention | `θ=−eig(M)` | UV-attractive ⇔ `eig(M)>0` |
| generations | 3 (P5: `SU(3)_F`; corroborated by `h¹(V₁₆)=3`) | added postulate |
| central charges | `(c_L,c_R)=(16,10)` | low-energy heterotic correspondence |
| cosmological constant | tuned relevant `Λ_cc` | inherited, not solved |

---

## 11. Merge mechanics

- Backbone narrative = sn-article.tex (it has the gauge/anomaly/strings/heterotic/CY/SM
  content); fold in ucft.tex's rigorous FRG appendix + heat-kernel/BRST induced-gravity
  derivation. Apply EVERY correction above. Preserve the author's notation, theorem/proof
  style, and voice; reconcile every shared number to §10.
- Signature `(-,+,+,+)`; `[f]=mass`; "clock fields" for the 32 Goldstones; one symbol each
  for `f` (decay constant) and `μ` (breaking/mass scale).
- Each unit (section/appendix) outputs compile-ready LaTeX using the shared preamble macros;
  cross-reference via `\autoref` and stable `\label`s.
- Label every nontrivial claim **theorem / postulate / conjecture-indicative**. Keep the
  thesis honest (§0).
