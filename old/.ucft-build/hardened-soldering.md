# Hardened Lorentzian Soldering for UCFT

**Scope.** This document hardens §3 of `UCFT-SPINE.md` ("Emergent spacetime &
Lorentzian signature"). It establishes, with explicit mathematics and honest
labels, the chain by which a Lorentzian 4-manifold is *soldered* onto the
E₆/Albert-algebra arena **without** contradicting the positive-definite coset
(target-space) metric, and **without** the forbidden "multiply e⁰ by i" Wick
trick.

Every nontrivial statement is tagged **[Theorem]** (mathematically proved fact,
standard or here-derived), **[Postulate]** (added physical structure, P1–P6),
or **[Indicative]** (motivated, not controlled).

**Standing conventions** (from the spine, never deviated):

- Real form for dynamics: **compact E₆** (equivalently E₆₍₋₁₄₎, max compact
  Spin(10)×U(1)); the coset metric `K|_m` is **positive-definite**.
- E₆₍₋₂₆₎ appears **only** as the rigid orbit-classifier of the real **27**.
- Vacuum manifold `M = E₆/(Spin(10)×U(1)) = EIII`, real dim **32**, the rank-1
  primitive-idempotent orbit (complex Cayley plane). **NOT** E₆/F₄ (dim 26).
- Coset tangent `m = 16₍₋₃₎ ⊕ 16bar₍₊₃₎` (32 real), real-irreducible of complex
  type, commutant `= C`. **No** SO(10) singlet in `m`.
- Signature **(1,3)**, convention **(−,+,+,+)**, is **added structure** soldered
  from the cubic norm `N` / `H₂(O) ≅ R^{1,9}`; it is **NOT** a Killing-form
  theorem.

The logical separation enforced throughout:

> The **positive-definite** object is the coset/target metric `K|_m` (it lives on
> the 32-dimensional internal tangent space `m`). The **indefinite** object is
> the cubic norm `N` on the Jordan algebra `J₃(O)`, whose 2×2 octonion-Hermitian
> sub-block carries the Minkowski quadratic form on a *different* space `R^{1,9}`.
> These are two distinct bilinear/forms on two distinct vector spaces; no single
> form is being asked to be both definite and indefinite. The soldering map (P3)
> is precisely the bridge that imports the indefinite (1,3) structure onto a
> 4-plane of the world-tangent bundle, leaving `K|_m` untouched and definite.

---

## 1. H₂(O) ≅ R^{1,9}, det = Minkowski form, SL(2,O) ≅ Spin(1,9)

### 1.1 The algebraic setup

Let `O` be the (real, division) octonions, a non-associative composition algebra
of real dimension 8 with conjugation `x ↦ x̄`, norm `n(x) = x x̄ = x̄ x ∈ R≥0`,
trace `t(x) = x + x̄ ∈ R`, and polarized inner product
`⟨x,y⟩ = ½(x ȳ + y x̄) = ½ t(x ȳ)`, a Euclidean (positive-definite) form on
`O ≅ R⁸`.

Define the space of **2×2 octonion-Hermitian matrices**

```
H₂(O) = { X = [ a    x  ]  :  a,b ∈ R,  x ∈ O } ,        (X† = X)
            [ x̄    b  ]
```

with `X† = X̄ᵀ`. As a real vector space

```
dim_R H₂(O) = 1 (a) + 1 (b) + 8 (x) = 10 .                         (1.1)
```

### 1.2 The determinant is the Minkowski form of signature (1,9)

Define the **determinant** (well-defined despite non-associativity because the
off-diagonal entry and its conjugate associate with the scalars):

```
det X = a b − x x̄ = a b − n(x) .                                  (1.2)
```

**[Theorem 1.1] (Minkowski identification).** Introduce light-cone-adapted
coordinates `a = x⁰ + x⁹`, `b = x⁰ − x⁹`, and write `x = (x¹,…,x⁸) ∈ R⁸` in an
orthonormal basis of `O`, so `n(x) = (x¹)² + ⋯ + (x⁸)²`. Then

```
det X = (x⁰)² − (x⁹)² − Σ_{i=1}^{8} (xⁱ)²
      = −[ −(x⁰)² + (x¹)² + ⋯ + (x⁹)² ]
      = − η_{μν} Xᵘ Xᵛ ,                                          (1.3)
```

with `η = diag(−1,+1,+1,+1,+1,+1,+1,+1,+1,+1)` on `R^{10}`. Hence the quadratic
form `−det` on the 10-dimensional space `H₂(O)` has signature **(1,9)** (one
minus, nine plus in the `(−,+,…,+)` convention), i.e.

```
(H₂(O), −det) ≅ R^{1,9} .                                         (1.4)
```

*Proof.* `det X = ab − n(x) = (x⁰+x⁹)(x⁰−x⁹) − Σᵢ(xⁱ)² = (x⁰)²−(x⁹)²−Σᵢ(xⁱ)²`,
which is the displayed Minkowski form. Sylvester: the diagonal entries give one
`+` (the `x⁰` mode) and one `−` (the `x⁹` mode), and `−n(x)` contributes eight
`−`'s; thus `det` has signature `(1,9)` and `−det` has signature `(9,1)`. In the
mostly-plus convention `(−,+,…,+)` the form `−det` is read as the
single-timelike `R^{1,9}`. ∎

The polarization of `−det` is the symmetric bilinear (Minkowski) form
`⟨X,Y⟩_M = ½(det(X+Y) − det X − det Y)` restricted appropriately; the timelike
direction is the trace direction `½ t(X)·𝟙` (the `x⁰` mode) and the nine
spacelike directions are `x⁹` together with the eight octonion imaginary+real
off-diagonal modes.

### 1.3 SL(2,O) ≅ Spin(1,9)

The set of "determinant-preserving, Hermiticity-preserving" linear actions on
`H₂(O)` does not form a group under naive matrix multiplication (octonion
non-associativity obstructs `g X g†` from being associative in general), so
`SL(2,O)` must be **defined** as the appropriate spin group acting through a pair
of (left/right) octonion-multiplication generators.

**[Theorem 1.2] (Octonionic exceptional isomorphism).** There is a Lie-group
isomorphism

```
SL(2,O) ≅ Spin(1,9) ,                                             (1.5)
```

under which the defining 10-dimensional representation of Spin(1,9) on `R^{1,9}`
is realized as the action on `(H₂(O), −det)`, and the two chiral 16-dimensional
spinor representations of Spin(1,9) are realized on `O² ≅ R^{16}` (a column /
row of octonions). Concretely, for `ψ ∈ O²` the generators act by left/right
octonion multiplications such that the bilinear `ψ† X ψ` and the determinant are
covariant; the double cover `Spin(1,9) → SO(1,9)` is the statement that the
spinor action is two-valued on the vector action.

*Status.* This is the **n=8 (octonionic) member** of the classical family of
exceptional isomorphisms `SL(2,A) ≅ Spin(n+1,1)` for the four normed division
algebras `A = R, C, H, O` of dimension `n = 1,2,4,8`:

```
A = R : SL(2,R)  ≅ Spin(2,1)      (R^{2,1})
A = C : SL(2,C)  ≅ Spin(3,1)      (R^{3,1})   ← the familiar Lorentz cover
A = H : SL(2,H)  ≅ Spin(5,1)      (R^{5,1})
A = O : SL(2,O)  ≅ Spin(9,1)      (R^{9,1})                       (1.6)
```

(written here as `Spin(n+1,1)`; `Spin(9,1) = Spin(1,9)` up to convention). The
isomorphisms are classical (Sudbery; Kugo–Townsend; Manogue–Schray; Baez,
"The Octonions", §3.3). For `A = O` the result is a *bona fide* group
isomorphism even though `SL(2,O)` is not a matrix group in the literal sense —
it is **defined** as `Spin(1,9)`, with its `R^{1,9}` action presented by (1.2)–
(1.4). ∎

**Remark (which Lorentz group we ultimately want).** The physically relevant
4-D Lorentz cover is the `A = C` line, `SL(2,C) ≅ Spin(3,1)`, sitting inside
(1.6) via `C ⊂ O`. This is exploited in §3: the soldered 4-plane is the
`H₂(C) ≅ R^{1,3}` sub-block, with `SL(2,C) ⊂ SL(2,O)` acting as `Spin(1,3) ⊂
Spin(1,9)`. The octonionic `R^{1,9}` is the *ambient* indefinite structure; the
*physical* (1,3) is a soldered 4-plane of it. **[Theorem 1.3]** The chain of
inclusions of composition algebras `R ⊂ C ⊂ H ⊂ O` induces the chain of
Lorentz-spin inclusions `Spin(1,1) ⊂ Spin(1,3) ⊂ Spin(1,5) ⊂ Spin(1,9)`, and in
particular `Spin(1,3) = SL(2,C) ⊂ SL(2,O) = Spin(1,9)` via the sub-block
`H₂(C) ⊂ H₂(O)`, `det|_{H₂(C)} = ` the (1,3) Minkowski form. *Proof.* `H₂(C) =
{[[a,z],[z̄,b]] : a,b∈R, z∈C}` has real dim `1+1+2 = 4`, `det = ab − |z|²`, which
is the (1,3) Minkowski form by the same light-cone change of variables as
Theorem 1.1 restricted to `C`; the determinant-covariance of the `SL(2,C)`
action is the standard 4-D statement. ∎

---

## 2. The cubic norm N on J₃(O) is genuinely indefinite

### 2.1 The Albert algebra and its two invariants

Let `J₃(O)` be the **Albert algebra** of 3×3 octonion-Hermitian matrices,

```
X = [ ξ₁    z₃     z̄₂  ]
    [ z̄₃    ξ₂     z₁  ] ,   ξᵢ ∈ R,  zᵢ ∈ O,                     (2.1)
    [ z₂    z̄₁     ξ₃  ]
```

a real Jordan algebra of dimension `dim_R J₃(O) = 3·1 + 3·8 = 27`, with the
Jordan product `X∘Y = ½(XY+YX)`.

It carries two basic polynomial invariants:

**(a) The quadratic trace form**
```
Q(X) = Tr(X²) = Σᵢ ξᵢ² + 2 Σᵢ n(zᵢ)
     = ξ₁²+ξ₂²+ξ₃² + 2(n(z₁)+n(z₂)+n(z₃)) .                       (2.2)
```
This is **positive-definite** on `J₃(O) ≅ R^{27}` (a sum of real squares and
positive-definite octonion norms).

**(b) The cubic norm (determinant)**
```
N(X) = det X
     = ξ₁ξ₂ξ₃ + 2 Re(z₁ z₂ z₃) − ξ₁ n(z₁) − ξ₂ n(z₂) − ξ₃ n(z₃) , (2.3)
```
a homogeneous cubic. (`Re(z₁z₂z₃) = ½ t(z₁z₂z₃)`; the triple product is
unambiguous on the relevant index pattern.)

### 2.2 The crucial separation of invariance groups

**[Theorem 2.1] (F₄ preserves the Euclidean trace form).** The automorphism
group of the Jordan algebra `J₃(O)`,

```
F₄ = Aut(J₃(O)) ,   compact, dim 52 ,                            (2.4)
```

preserves the Jordan product, hence preserves **both** `Q(X)=Tr(X²)` and `N(X)`.
In particular `Q` is `F₄`-**invariant**, and since `Q` is positive-definite, `F₄`
is realized as a subgroup of `SO(27)` (the orthogonal group of the Euclidean form
`Q`). *Proof.* Automorphisms fix `𝟙`, commute with `∘`, hence fix `Tr` (the
Jordan-trace, a coefficient of the characteristic polynomial) and fix `Tr(X²)`
and `det = N`; the Euclidean signature of `Q` then forces `F₄ ⊂ SO(27)`. (Freudenthal;
standard.) ∎

**[Theorem 2.2] (E₆ preserves only the cubic norm, not Q).** The **reduced
structure group** of the *division*-octonion Albert algebra,

```
E₆₍₋₂₆₎ = { g ∈ GL(27,R) : N(gX) = λ(g) N(X), λ(g)∈R^× } , dim 78, (2.5)
```

preserves the cubic norm `N` **up to a real character** `λ(g)`, and is exactly
the rigid orbit-classifier of the real **27**. It does **NOT** preserve
`Q=Tr(X²)`: a generic `g ∈ E₆₍₋₂₆₎ ∖ F₄` deforms the Euclidean trace form.
*Proof.* `E₆₍₋₂₆₎` is by definition the linear group fixing `N` up to scalar; the
strict stabilizer (`λ≡1`) is the determinant-preserving form. If `E₆₍₋₂₆₎` also
preserved `Q`, it would lie in `SO(27)` and hence be compact, contradicting
`dim E₆₍₋₂₆₎ = 78 > 52 = dim F₄` together with non-compactness of the real form
`E₆₍₋₂₆₎` (it has maximal compact `F₄`, so the 26 non-compact directions move `Q`).
Thus `Q` is **not** `E₆`-invariant. (Forbidden-error #4: "the structure group
preserves `Q`" and "(Q,N) separates orbits" are both false.) ∎

**[Theorem 2.3] (Orbit classification).** Real `E₆`-orbits on the **27** are
classified by **rank ∈ {0,1,2,3}** and the value of **`N`** (within the
appropriate scaling), **not** by `(Q,N)`. The rank is intrinsic:

```
rank X ≤ 1  ⇔  X# = 0 ,     rank X ≤ 2  ⇔  N(X) = 0  (and X# ≠ 0),  (2.6)
```

where `X#` is the `E₆`-covariant **sharp/adjoint** map (the quadratic adjoint,
satisfying `X# # = N(X) X` and `X∘X# = N(X)𝟙` schematically). *Proof sketch.*
`X#` is `E₆`-covariant; its vanishing is the closed minimal (rank-1) orbit
condition (the orbit `E₆/(Spin(10)×U(1))` in `P(27_C)` for the complex/compact
story; the rank-1 cone in the real case). `N(X)` is the rank-3 detector. (Freudenthal;
Krutelevich; consistent with spine §1.) ∎

### 2.3 Genuine indefiniteness of N

**[Theorem 2.4] (N is indefinite).** The cubic norm `N` on `J₃(O)` takes
strictly positive, strictly negative, and zero values on `J₃(O) ≅ R^{27}`; it is
a **genuinely indefinite** invariant. *Proof.* On the diagonal `X =
diag(ξ₁,ξ₂,ξ₃)`, `N = ξ₁ξ₂ξ₃`. Choosing `(1,1,1)` gives `N=+1`; `(−1,1,1)` gives
`N=−1`; `(0,1,1)` gives `N=0`. As a real cubic, `N` is in any case odd under
`X ↦ −X` (`N(−X) = −N(X)`), so positivity is impossible: indefiniteness is
automatic for an odd-degree form. ∎

**[Theorem 2.5] (The Minkowski block sits inside N).** Restricting `N` to the
2×2 block obtained by setting `ξ₃ = 1`, `z₁ = z₂ = 0`, and freezing the
upper-left `2×2` Hermitian sub-matrix `Y = [[ξ₁, z₃],[z̄₃, ξ₂]] ∈ H₂(O)`:

```
N(X)|_{block} = ξ₁ξ₂·1 − 1·n(z₃) − ξ₁·0 − ξ₂·0
              = ξ₁ξ₂ − n(z₃) = det Y .                            (2.7)
```

So the indefinite Minkowski form `det = −η` of §1 (Theorem 1.1) is the
**restriction of the cubic norm to an `H₂(O) ≅ R^{1,9}` sub-block of `J₃(O)`**
(with the third pivot idempotent set to 1). This is the precise algebraic origin
of the Lorentzian structure: it is carried by `N`, not by `Q`. ∎

### 2.4 No contradiction with the positive-definite coset metric

**[Theorem 2.6] (Consistency: definite ≠ indefinite, different spaces).** The
positive-definite coset/target metric and the indefinite spacetime form are
mutually consistent because they are forms on **different spaces**:

| Object | Lives on | Form | Signature | Invariance |
|---|---|---|---|---|
| Coset metric `K|_m` | `m = 16₍₋₃₎⊕16bar₍₊₃₎` (dim 32, internal tangent of EIII) | Killing restriction | **positive-definite** (Euclidean) | `H = Spin(10)×U(1)`-invariant; unique by Schur (commutant `=C`) |
| Trace form `Q=Tr(X²)` | `J₃(O) = R^{27}` (matter rep) | quadratic | **positive-definite** (Euclidean) | `F₄`-invariant (NOT E₆) |
| Cubic norm `N=det` | `J₃(O) = R^{27}` (matter rep) | cubic | **indefinite** | `E₆`-invariant up to character |
| Minkowski form `−det` | `H₂(O) ⊂ J₃(O)`, `R^{1,9}` sub-block | quadratic (restriction of `N`) | **(1,9)** indefinite | `SL(2,O)=Spin(1,9)` |
| Soldered metric `η` | 4-plane of `T_x(world-manifold)` | quadratic | **(1,3)**, `(−,+,+,+)` | `SO(1,3)` (local Lorentz) — **postulated** P3 |

*Proof.* The coset metric `K|_m` is positive-definite by the compact-real-form
choice (compact E₆ ⇒ negative-definite Killing form ⇒ positive-definite induced
metric on the reductive complement `m`), and it is unique by Schur because `m` is
real-irreducible of complex type with commutant `C` (spine §2). The indefinite
object `N` lives on the **27**, a *different* representation space from `m` (the
adjoint complement). Therefore "the coset metric is Euclidean" and "the cubic
norm is indefinite" are statements about two different bilinear data on two
different modules; there is no logical tension. The Lorentzian signature used for
spacetime is imported from `N|_{H₂(O)}` via the soldering map P3 (§3), and is
**never** asked of `K|_m`. ∎

> **This directly retires Forbidden Error #3** (no `(4,28)` Killing/coset
> signature; `(1,3)` does NOT fall out of the coset geometry) **and #4** (only
> `N` is structure-group invariant; `F₄` preserves `Q`; orbits classified by
> rank and `N`, not `(Q,N)`).

---

## 3. The soldering postulate P3 (precise statement)

We now state P3 rigorously. P3 is **[Postulate]** throughout — it is *added*
structure, not a theorem of the coset geometry.

### 3.1 Data already available (theorems)

From the dynamics we have, as **[Theorem]**-level inputs:

- (T-a) A vacuum manifold `M = EIII = E₆/(Spin(10)×U(1))`, `dim_R M = 32`, with
  reductive decomposition `e₆ = h ⊕ m`, `h = so(10)⊕u(1)`,
  `m = 16₍₋₃₎⊕16bar₍₊₃₎`, and Maurer–Cartan form `θ = θ_h + θ_m` valued in
  `e₆`. The pull-back of `θ_m` to the world-manifold supplies a 32-component
  1-form (the would-be 32 clock fields' frame data); the pull-back of `θ_h`
  supplies the composite `Spin(10)×U(1)` connection.
- (T-b) The matter rep `27 = J₃(O)` carries the indefinite cubic norm `N`, and
  contains the `H₂(O) ≅ R^{1,9}` Minkowski sub-block with `det = −η^{(1,9)}` and
  `SL(2,O) = Spin(1,9)` (Theorems 1.1, 1.2, 2.5), and inside it the
  `H₂(C) ≅ R^{1,3}` sub-block with `det = −η^{(1,3)}` and `SL(2,C) = Spin(1,3)`
  (Theorem 1.3).
- (T-c) The 27 branches under `H = Spin(10)×U(1)` as
  `27 = 1₍₊₄₎ ⊕ 10₍₋₂₎ ⊕ 16₍₊₁₎`. The `10₍₋₂₎` is the SO(10) **vector**; it is
  the carrier of an `SO(1,9)`-vector structure once a real form/orbit is fixed.

### 3.2 The clock fields and the four frame directions

Let `φ : world-manifold → (configuration of the order parameter)` be the
clock-field map. The **32 clock fields** are the coordinates along `m` (Goldstone
directions of `EIII`). Of these, **28 are physical massive scalars** (single
common radiative mass, Schur on the irreducible `m`) and **4 are
gauge/soldering** modes eaten by world-volume diffeomorphism + local Lorentz
(spine §4; Forbidden Error #5 retired — `m` is irreducible, the 4 frame modes are
**gauge-protected**, not Hessian zeros).

**[Postulate P3a] (Soldering / frame identification).** There exists a soldering
map

```
σ :  (4 distinguished clock-frame directions)  ⟶  T_x 𝓜⁴ ,        (3.1)
```

identifying four clock-field frame directions, written `e^a_μ` (`a = 0,1,2,3`;
`μ` a world index), with a tangent 4-plane `Π_x ⊂ T_x 𝓜⁴` of an emergent
**4-dimensional** world-manifold `𝓜⁴`. The vierbein `e^a_μ` is the pull-back of
the Maurer–Cartan component along these four directions:

```
e^a_μ(x) = ⟨ τ^a , θ_m(∂_μ) ⟩ ,                                  (3.2)
```

where `{τ^a}_{a=0..3}` is the soldering frame (the image of the
`H₂(C) ≅ R^{1,3}` basis under the embedding of §3.3). The vierbein **emerges
dynamically** as the pull-back of the Maurer–Cartan form. *This is the
realization of the spine's "frame `e^a_μ` emerges dynamically (pull-back of the
Maurer–Cartan form)."*

**[Postulate P3b] (Induced Minkowski form).** The induced quadratic form on the
4-plane `Π_x` is **declared** to be the Minkowski form inherited from the
cubic-norm / `H₂(O) ≅ R^{1,9}` structure:

```
g_μν(x) = e^a_μ(x) e^b_ν(x) η_{ab} ,    η_{ab} = diag(−1,+1,+1,+1) , (3.3)
```

with `η` the **restriction of `−det` (the cubic norm `N`) to the `H₂(C)` sub-block**
(Theorems 1.3, 2.5). Explicitly, the four soldering directions `τ^a` are mapped
into the `H₂(C) ≅ R^{1,3} ⊂ H₂(O) ⊂ J₃(O)` Minkowski sub-block, and `η_{ab}` is
the pull-back of the form `−det` there.

**[Postulate P3c] (Dimension and signature are added).** The numbers

```
d = 4 ,        signature = (1,3) = (−,+,+,+) ,                    (3.4)
```

are **part of P3**, i.e. **added structure**. They are **NOT** consequences of
the Killing form, the coset signature, or any rank lemma. (This is the honest
relabeling of `lem:Rank4`: the world-manifold dimension `d=4` is *fixed by the
soldering postulate*; the impossible/circular `r>4` branch is deleted.)

### 3.3 How (1,3) is carved out of (1,9): the embedding

The soldering frame must pick a *timelike* line and *three* spacelike lines
inside the indefinite `R^{1,9}`. The clean route:

**[Postulate P3d] (Vierbein condensate / frame reduction `SO(1,9) → SO(1,3)`).**
A vierbein condensate (a VEV of the soldering bilinear) selects the
`H₂(C) ≅ R^{1,3}` sub-block of `H₂(O) ≅ R^{1,9}` and breaks the ambient frame
symmetry

```
Spin(1,9) = SL(2,O)  ⟶  SL(2,C) × G_int  =  Spin(1,3) × Spin(6) , (3.5)
```

where the orthogonal complement of `R^{1,3}` in `R^{1,9}` is the six-dimensional
spacelike `R⁶` acted on by `Spin(6) ≅ SU(4)` (internal). The residual
`Spin(1,3) = SL(2,C)` is the **local Lorentz group** of the soldered 4-plane;
the timelike direction is the trace/`x⁰` direction (unique up to the residual
boost), the three spacelike directions are the `C`-off-diagonal + `x⁹`-type modes
of `H₂(C)`. *This is the second of the two routes the spine offers ("a
signature-selecting vierbein condensate breaking the frame symmetry to
`SO(1,3)`"); §4 takes the first route (emergent timelike clock) as the dynamical
mechanism that triggers exactly this condensate.*

**[Theorem 3.1] (Internal consistency of P3).** Given P3a–P3d, `g_μν` of (3.3) is
a non-degenerate symmetric rank-2 tensor of Lorentzian signature `(1,3)` on
`𝓜⁴`, invariant under the residual local Lorentz `SO(1,3)`; the 4 frame modes
`e^a_μ` carry exactly the gauge degrees of freedom eaten by world-volume
diffeomorphisms (`d=4` of them) plus local Lorentz (`6` boosts/rotations are
non-dynamical frame rotations), consistent with the count "32 = 28 physical + 4
gauge/soldering." *Proof.* Non-degeneracy and signature are inherited from
`η_{ab} = diag(−1,+1,+1,+1)` (Theorem 1.3) provided `det(e^a_μ) ≠ 0` (generic
condensate); `SO(1,3)`-invariance is the statement `η_{ab} = Λ^c_a Λ^d_b η_{cd}`
for `Λ ∈ SO(1,3)`, which holds by definition of the Lorentz group; the gauge
count is the Ward-identity statement `∇_μ(δΓ/δe^a_μ) = 0` (spine §4), so the 4
frame directions are protected by diffeo+Lorentz, not by a Hessian zero. ∎

### 3.4 What P3 does NOT claim

- P3 does **not** claim the coset metric `K|_m` is Lorentzian. `K|_m` stays
  positive-definite; it is the kinetic/target-space metric of the σ-model. (Error
  #3 retired.)
- P3 does **not** use a `(4,28)` split of `m` with a Lorentzian signature. The
  `4 = ` soldering modes are **gauge**, the `28 = ` physical scalars are
  Euclidean target coordinates. (Error #2, #5 retired.)
- P3 does **not** invoke `grad I₂, grad I₃` singlets to build the 4-plane (there
  are no SO(10) singlets in `m`; `grad Q|_{X₀} = 2X₀ ∉ m`). The 4-plane is the
  soldered image of `H₂(C) ⊂ 27`, not a sub-module of `m`. (Error #5 retired.)

---

## 4. d=4 and (1,3) are ADDED structure — not Killing-form theorems

Collecting the labels for absolute clarity:

| Statement | Label | Why |
|---|---|---|
| `(H₂(O), −det) ≅ R^{1,9}`, `det` = Minkowski | **Theorem 1.1** | algebraic identity (1.3) |
| `SL(2,O) ≅ Spin(1,9)`, `SL(2,C) ≅ Spin(1,3)` | **Theorem 1.2, 1.3** | classical exceptional isomorphisms |
| `Q=Tr(X²)` positive-definite, `F₄`-invariant | **Theorem 2.1** | automorphism + Euclidean signature |
| `N=det` `E₆`-invariant (up to character), **NOT** `Q` | **Theorem 2.2** | def. of reduced structure group + non-compactness |
| Orbits classified by rank and `N`, not `(Q,N)` | **Theorem 2.3** | `X#` covariance |
| `N` genuinely indefinite | **Theorem 2.4** | odd cubic / explicit values |
| Minkowski block `det Y` ⊂ `N` | **Theorem 2.5** | restriction (2.7) |
| Definite `K|_m` and indefinite `N` are consistent | **Theorem 2.6** | different spaces |
| Vierbein `e^a_μ` = pull-back of Maurer–Cartan | **Theorem (T-a)+P3a** | MC construction; the *choice of 4* is P3 |
| **`d = 4`** | **Postulate P3c** | added; `lem:Rank4` relabeled, `r>4` branch deleted |
| **signature `(1,3)`, `(−,+,+,+)`** | **Postulate P3b,c** | added; soldered from `N`/`H₂(O)`; **not** Killing |
| Frame reduction `Spin(1,9)→Spin(1,3)×Spin(6)` | **Postulate P3d** | vierbein condensate |
| `g_μν` Lorentzian, `SO(1,3)`-invariant, 4 gauge modes | **Theorem 3.1** | given P3a–d |

**[Explicit anti-theorem]** There is **no** derivation of `d=4` or signature
`(1,3)` from the Killing form of E₆, from the coset metric `K|_m`, or from a rank
lemma. The Killing form of compact E₆ is negative-definite; its restriction to
`m` is (sign-flipped to) positive-definite of signature `(32,0)` — there is no
Lorentzian content in it whatsoever. Any claim of a `(4,28)` or `(1,3)`
Killing/coset signature is **false** (Forbidden Error #3). Lorentzian signature
enters **only** through P3, rooted in the **indefinite cubic norm `N`** /
`H₂(O) ≅ R^{1,9}`. This is the central honesty commitment of the construction.

---

## 5. Emergent-time route: Page–Wootters clock + Osterwalder–Schrader

No "multiply `e⁰` by `i`" trick is used (Forbidden Error #6). Lorentzian time is
recovered by a relational-clock + reconstruction-theorem mechanism.

### 5.1 The timelike clock

**[Postulate P5a] (Timelike clock selection).** Among the 32 clock fields, the
soldering condensate (P3d) selects a distinguished **timelike** direction whose
clock field we call `φ⁰` — concretely the trace/`x⁰` direction of the soldered
`H₂(C) ⊂ J₃(O)` block (the unique `η`-timelike soldering frame leg `e⁰_μ`). `φ⁰`
is the gauge/soldering partner along the time direction; the spatial frame legs
`e^i_μ` (`i=1,2,3`) are its three spacelike companions. The selection of *one*
timelike direction is exactly the `(1,3)` (not `(2,8)` etc.) content of P3.

### 5.2 Page–Wootters relational time

**[Postulate/Construction P5b] (Page–Wootters clock).** Promote `φ⁰` to a
**Page–Wootters clock**: enlarge the kinematical Hilbert space to
`𝓗 = 𝓗_clock ⊗ 𝓗_sys`, impose the diffeomorphism/Hamiltonian constraint of the
induced-gravity theory,

```
Ĥ_phys |Ψ⟩⟩ = 0 ,        Ĥ_phys = Ĥ_clock + Ĥ_sys (+ interaction) , (5.1)
```

on physical states `|Ψ⟩⟩` (this is the constraint *required* by induced gravity;
spine §7, Axiom II demoted to a theorem). Conditioning on clock readings `φ⁰=τ`
defines the relational state

```
|ψ(τ)⟩_sys = ⟨φ⁰=τ| Ψ⟩⟩ ,                                       (5.2)
```

which **[Theorem 5.1] (Page–Wootters evolution)** satisfies the Schrödinger
equation in the relational time `τ`,

```
i ∂_τ |ψ(τ)⟩_sys = Ĥ_sys |ψ(τ)⟩_sys ,                            (5.3)
```

with the unitary group `U(τ) = e^{−iĤ_sys τ}` recovered **on physical states** in
the semiclassical/background regime where the soldered `g_μν` admits a timelike
Killing vector and `Ĥ_sys = ∫ T₀₀` along it. *Proof.* Standard PW reduction:
under the constraint (5.1) with a good clock (`[φ̂⁰, Ĥ_clock]` canonically
conjugate, `Ĥ_clock = p̂_{φ⁰}`), the conditioned state (5.2) obeys (5.3); this is
the PW theorem. The map `t ↔ x⁰` is `Ĥ = ∫ T₀₀` along the timelike Killing
direction (spine §7). ∎ *This realizes the spine's "designate the timelike clock
`φ⁰` a Page–Wootters clock; recover `U(τ)` on physical states in the
semiclassical/background regime." Global hyperbolicity is flagged as an
assumption.*

### 5.3 Osterwalder–Schrader reconstruction along φ⁰

The Euclidean (positive-definite `K|_m`-based) σ-model defines a *Euclidean*
functional measure. Lorentzian QFT is obtained by **Osterwalder–Schrader (OS)
reconstruction along the selected time `φ⁰`** — **not** by analytically rotating
`e⁰ → i e⁰`.

**[Construction/Indicative O-S] (Reflection positivity & reconstruction).** Let
`τ = φ⁰` be the OS (Euclidean) time. Assume the Euclidean Schwinger functions
`S_n(x₁,…,x_n)` of the σ-model satisfy the Osterwalder–Schrader axioms:

- (OS0) Euclidean invariance under the relevant Euclidean group of the soldered
  spatial slice;
- (OS1) **Reflection positivity** with respect to reflection `θ: τ ↦ −τ` about a
  `φ⁰`-slice:
  ```
  Σ_{i,j} c̄ᵢ cⱼ  S(θ Fᵢ · Fⱼ) ≥ 0 ,   Fᵢ supported in {τ>0} ;   (5.4)
  ```
- (OS2) symmetry and clustering.

**[Theorem 5.2] (OS ⇒ Lorentzian Wightman theory).** Given (OS0–OS2), the OS
reconstruction theorem produces a Hilbert space `𝓗`, a positive self-adjoint
Hamiltonian `Ĥ ≥ 0` generating `τ`-translations as `e^{−Ĥτ}` (Euclidean), and —
by analytic continuation of the *Schwinger functions in the time argument
`τ → it`* — Lorentzian Wightman functions on `(𝓜⁴, η^{(1,3)})` with a unitary
time evolution `e^{−iĤt}`. *Status.* The OS theorem itself is **[Theorem]**
(rigorous, given its hypotheses); that the UCFT σ-model *satisfies* reflection
positivity (5.4) along `φ⁰` is **[Indicative]** — it is the precise property that
must be checked, and is the honest replacement for the Wick trick. *Proof of the
theorem part.* Osterwalder–Schrader (1973, 1975): reflection positivity yields a
physical inner product on the quotient `{F: τ>0}/(null)`, `Ĥ` is the generator of
the contraction semigroup `e^{−Ĥτ}` which is self-adjoint and positive; the
Schwinger functions, analytic in a tube, continue to Wightman distributions
satisfying the Wightman axioms (in particular spectrum condition and Lorentz
covariance under the reconstructed `SO(1,3)`). ∎

**Why this is not the forbidden Wick trick.** The analytic continuation `τ → it`
is performed on the *correlation functions / Schwinger distributions in the
clock-time argument*, justified by reflection positivity (5.4) and the resulting
analyticity tube — it is **not** the ad hoc replacement `e⁰_μ → i e⁰_μ` of a
frame field. The Lorentzian metric `η^{(1,3)}` is the *soldered* form (P3),
already Lorentzian *before* any continuation; OS reconstruction supplies the
*unitary dynamics* `e^{−iĤt}` and the positive-energy spectrum, with `φ⁰` (the
Page–Wootters clock) serving as the OS reflection-time. The two mechanisms
dovetail: P5b gives relational unitary `U(τ)` on physical states; OS gives the
positive-energy, Lorentz-covariant Wightman theory.

### 5.4 Reconciliation summary (time)

```
Axiom II (external U(t)=e^{−iHt})  →  [demoted to Theorem]
   via Page–Wootters clock φ⁰ + induced-gravity constraint Ĥ_phys|Ψ⟩⟩=0
   ⇒ relational U(τ)=e^{−iĤ_sys τ} on physical states (semiclassical),
   t ↔ x⁰ by Ĥ = ∫T₀₀ along timelike Killing vector.

Euclidean σ-model (definite K|_m)  →  Lorentzian QFT on (𝓜⁴, η^{(1,3)})
   via Osterwalder–Schrader reconstruction along φ⁰ (reflection positivity),
   NOT via e⁰ → i e⁰.
```

---

## 6. Master statement of the hardened soldering

**[Postulate P3 (full, hardened).]** *Given* (i) the compact-E₆ dynamics with
vacuum `EIII = E₆/(Spin(10)×U(1))` and positive-definite coset metric `K|_m`
[Theorem-level inputs], and (ii) the matter rep `27 = J₃(O)` with its indefinite
cubic norm `N` containing the `H₂(O) ≅ R^{1,9}` Minkowski sub-block
(`det = −η`, `SL(2,O)=Spin(1,9)`) and the `H₂(C) ≅ R^{1,3}` sub-block
(`SL(2,C)=Spin(1,3)`) [Theorems 1.1–1.3, 2.4–2.6] — we **postulate** a soldering
map `σ` (P3a) identifying four clock-field frame directions `e^a_μ` (pull-backs
of the Maurer–Cartan form) with a tangent 4-plane of an emergent
**4-dimensional** world-manifold, carrying the induced Minkowski form
`g_μν = e^a_μ e^b_ν η_{ab}`, `η = diag(−1,+1,+1,+1)` (P3b), with `d=4` and
signature `(1,3)` as **added structure** (P3c), realized by a vierbein condensate
breaking `Spin(1,9) → Spin(1,3)×Spin(6)` (P3d). The timelike clock `φ⁰` is a
Page–Wootters clock (P5a,b) and Lorentzian unitary dynamics is recovered by
Osterwalder–Schrader reconstruction along `φ⁰` (Theorem 5.2), **not** by Wick
rotation of the frame.

**Honesty ledger (the spine's commitments, satisfied):**

1. Coset metric is positive-definite; signature comes from the cubic norm via P3.
   ✓ (Theorem 2.6, P3.)
2. Only `N` is structure-group invariant; `F₄` preserves `Q`; orbits by rank+`N`.
   ✓ (Theorems 2.1–2.3.)
3. `d=4`, `(1,3)` honestly added (P3), not Killing-form theorems. ✓ (§4 ledger.)
4. No "`e⁰ → i e⁰`"; emergent time via PW clock + OS reconstruction. ✓ (§5.)
5. `m = 16₍₋₃₎⊕16bar₍₊₃₎`, irreducible, no singlet; 4 frame modes gauge-protected.
   ✓ (§3.2, §3.4, Theorem 3.1.)
6. EIII (dim 32, rank-1 idempotent orbit), not E₆/F₄ (dim 26). ✓ (standing
   conventions, T-a.)

---

## 7. References (for the algebraic theorems)

- J. C. Baez, *The Octonions*, Bull. AMS 39 (2002) 145 — §3.3 (`SL(2,O) ≅
  Spin(9,1)`), §4 (`J₃(O)`, `F₄`, `E₆`, cubic norm).
- A. Sudbery, *Division algebras, (pseudo)orthogonal groups and spinors*, J.
  Phys. A 17 (1984) 939 — `SL(2,A) ≅ Spin(n+1,1)`.
- T. Kugo, P. Townsend, *Supersymmetry and the division algebras*, Nucl. Phys.
  B221 (1983) 357.
- C. A. Manogue, J. Schray, *Finite Lorentz transformations, automorphisms, and
  division algebras*, J. Math. Phys. 34 (1993) 3746.
- H. Freudenthal, *Lie groups in the foundations of geometry*, Adv. Math. 1
  (1964) 145 — `F₄ = Aut(J₃(O))`, `E₆` and the cubic norm.
- K. Osterwalder, R. Schrader, *Axioms for Euclidean Green's functions* I, II,
  Comm. Math. Phys. 31 (1973) 83; 42 (1975) 281 — reflection positivity &
  reconstruction.
- D. N. Page, W. K. Wootters, *Evolution without evolution*, Phys. Rev. D 27
  (1983) 2885 — relational clock.
