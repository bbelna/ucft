# SJET — Validation & PRD-Readiness Report

**Manuscript:** *Stone–Jordan Exceptional Theory* (SJET), B. Belna — `sjet.tex`, 10,710 lines, ~108 pp, 236 numbered theorem-environments, 25 sections.
**Review:** 39-agent adversarial validation (13 structure mappers, 8 mathematics validators, 14 phenomenology validators with live checks against PDG 2024 / Planck 2018 / DESI DR2 2025 / Muon g-2 WP25 + final Fermilab / FLAG 2024 / Super-K, 3 referee syntheses, 1 editor). ~2.07M tokens, 427 tool calls.
**Date:** 2026-06-06.

---

## Verdict: **reject & restructure**

Two independent specialist reports (mathematics, phenomenology) plus a parameter-counting audit converged on the same conclusion, and the load-bearing items were verified directly against the source:

> **The static endpoints of SJET are correct, externally-verifiable classical mathematics. Every *novel dynamical link* that is supposed to generate them and produce physics is undefined, circular, dimensionally false, or reverse-engineered to a known answer. The "Standard Model + gravity + cosmology from two primitives (void + mirror)" thesis is not supported.**

This is not a stylistic judgment. Three of the five central theorems contain **elementary, checkable errors**:

| # | Where | Error |
|---|---|---|
| 1 | `def:spacetime:H2O` (l.1089) | The whole d=4 spacetime sector rests on calling **H₂(ℍ) "four-dimensional"**. It is **6-dimensional**, signature (1,5) → Spin(1,5). The Minkowski (1,3) object is H₂(ℂ). |
| 2 | `prop:triality` / l.598 | σ is defined as a sign-twisted **3-cycle (order 6)** *and* as an involution **σ²=id (order 2)**. The ±1 eigenspace split and projector ½(id−σ) every downstream object uses are therefore invalid. |
| 3 | `lem:cellular` (l.10119) | The cohomology giving "**3 families**" is incoherent: the claimed diagonal coboundary is not in the cocycle space; the asserted 32/16 quotient is stated to be 32-dimensional (it did nothing); "=3" is never computed. |
| 4 | `thm:mirror-index` (l.685) | The **holomorphic** Atiyah–Bott Lefschetz formula is applied to an explicitly **antiholomorphic** involution; with no fixed points (cor:observer-fp) the equivariant sum is 0, not −3. |
| 5 | l.2149 | SO(10): **16×16 = 10+120+126** (dim 256), written as "1+45+126" (dim 172). The 126→SU(5)×U(1) branching listed sums to **51 ≠ 126**. |
| 6 | `thm:partition` (l.468) | The 1\|10\|16 weights have **no proof**; they are dim(irrep)/27 of the E₆→Spin(10) branching, fed back in as the *targets* of the capacity functional (`lem:capacity-gradient-d`, l.768). Circular. |
| 7 | `thm:flavon-vev` | **Verified false by direct computation**: the claimed κ_F⋆=62/243 does not zero the reduced determinant; for κ_F>0 only the trivial vacuum exists; the null-eigenvector ratio is 2.17 or 0.074, not the target 0.10. |

What *is* correct — division-algebra/Jordan/exceptional dimensions (1,2,4,8,27,78,248,496), the 27 = 1⊕10⊕16 branching, the 32 = 28⊕4 EIII split, sin²θ_W = 3/8 at unification, the b₀ = 12 one-loop bookkeeping — is **standard textbook representation theory inherited from an *assumed* embedding**. It carries no information specific to the void/mirror construction; sin²θ_W = 3/8 holds in *any* SU(5) GUT.

## The phenomenology is a postdiction engine

The parameter audit is decisive: SJET uses an estimated **35–45 effectively free inputs** to reproduce **~20–25 quantitative observables** — a parameter-to-observable ratio of **≈ 1.5–2 : 1, worse than the Standard Model's** (19 parameters for a vastly larger, sub-per-mille-tested edifice). The "0 free parameters / Tier-T1 parameter-free" claim is false. The knobs are hidden as (1) recycled integers (28, 3, 12, κ⋆) reused across unrelated sectors; (2) ~11 per-channel hand-picked rational exponents — the master one being n_S·n_F/b₀ = 7, exactly the power needed to lift tree m_μ/m_e = 10 to 206.8, and the Cabibbo exponent 1/5 *admitted in the proof* to be "the unique rational that maps λ_tree ≈ 0.249 to PDG λ"; (3) ~7 uncomputed "two-loop/threshold" factors R⁽²⁾; (4) matching scales chosen to position the logs; (5) ~6 measured quantities laundered back in as inputs.

Several flagship numbers **fail to reproduce from their own equations** (M_Pl off by ~16 orders of magnitude; Λ_QCD off by 4 orders; η_B off by 5.6×; deuteron binding off by ~8.5×), and at least **five contacts are already excluded or in tension with current data**:

- **Muon g−2:** SJET targets Δa_μ ≈ 2.6×10⁻⁹ — the *obsolete 2020* anomaly. WP25 + final Fermilab give Δa_μ ≈ 3.8(7)×10⁻¹⁰ (~0.6σ, consistent with zero); a 2.6×10⁻⁹ contribution is now excluded at ~3–4σ.
- **Δρ / T parameter:** SJET 0.0034 vs measured 0.00038 — **~11σ**, disguised as "<0.3%" by quoting the percentage on ρ≈1.
- **m_c/m_s:** the cited "PDG 13.6" is **fabricated**; the real lattice value is 11.77 (excluded by ~30σ).
- **w(z):** SJET asserts constant w = −1; DESI DR2 (2025) prefers evolving dark energy (w₀>−1, w_a<0) at 3.1–4.2σ — directly against the claim, and past SJET's own falsifier |w+1|>0.05.
- **⁷Li:** SJET quotes the *observed* 1.6×10⁻¹⁰ and calls it "consistent with BBN+CMB fits"; standard BBN+CMB *predicts* ~5×10⁻¹⁰. This is the lithium problem stated backwards.

## What is genuinely salvageable (the honest, publishable residue)

There is a real, modest contribution here — but it is **not** a theory of everything. Per the editor's structural recommendation, the path to something that could *read for PRD* (or a math-physics journal) is to **split and drastically cut**:

1. **A short, honest mathematics note** on the exceptional scaffold, restricted to what is provably true: (a) the capacity/branching bookkeeping reproducing 27 = 1⊕10⊕16 and the EIII 32 = 28⊕4 split, presented as **representation theory** (not a dynamical "partition theorem"); (b) the GUT-trace result sin²θ_W = 3/8 with the b₀ = 12 accounting, as standard Spin(10)/SU(5) phenomenology of the assumed embedding. These carry no false claims.
2. **A separate, explicitly *conjectural* framework paper** (a foundations venue, not PRD) sketching the void/mirror philosophy — only after relabeling every "derived/closed/Tier-T1" claim as conjectural and **removing the numerology**.
3. **Cut, or move to a clearly-flagged conjectural appendix:** the mirror operator M (M²=id fails for Cayley–Dickson), the observer family count, the soldering/spacetime sector, the OS/reflection-positivity construction, and every quantitative flavor/cosmology fit with a reverse-engineered exponent.

**Before any resubmission:** fix the demonstrable group theory (16×16, the 126 branching, the H₂(ℍ) dimension/signature); recompute the cellular cohomology with eigenspace dimensions that sum to 32; discharge or openly label ledger assumptions A3/A6/A7; retire the data-excluded contacts (g−2, Δρ/T, m_c/m_s, ⁷Li, w=−1); and replace "parameter-free" with the honest ~1.5:1 input-to-observable count.

---

*The three full specialist reports follow: mathematics, phenomenology, parameter audit. Appendix A tabulates every quantitative contact already in tension with or excluded by current data.*

---

# Part I — Mathematics referee report

# Mathematics

## Summary Verdict

The manuscript presents itself as a parameter-free derivation of the Standard Model and cosmology from two axioms (the "void" and the "mirror"). After auditing the load-bearing definitions, theorems, and proofs across all sections, my conclusion is unambiguous: **the central derivational chain does not hold at any of its dynamical links.** What is mathematically correct in this paper is, without exception, *imported* classical representation theory and Lie theory (the exceptional series, the Albert algebra, the E₆ → SO(10) × U(1) branching, EIII coset geometry). What is *novel* — the mirror functor, the capacity variational principle, the observer fixed-point count, the soldering of spacetime, the reflection-positivity construction, and the gravitational fixed point — is either undefined, circular, dimensionally false, or reverse-engineered to known answers. The recurrent failure mode is structural: a known target (3 families, d = 4 Lorentzian, sin²θ_W = 3/8, the dark-matter fraction, the muon anomaly) is reached by feeding that target in as the input of a "variational" or "RG" template whose free data (costs c_i, measure μ, anomalous-dimension coefficients, threshold scales, residue exponents) are never independently fixed. This is numerology and postdiction dressed in the vocabulary of derivation.

I recommend **rejection**. Below, fatal issues are listed first, then major, then minor, each with location.

---

## (1) The foundational construction (void → mirror → climb): mathematics or numerology?

**Partly genuine, ultimately numerology where it matters.** The static endpoints of the ladder are correct, externally-verifiable Lie theory: the division-algebra dimensions 1, 2, 4, 8; dim J₃(𝕆) = 27; dim E₈ = 248; dim E₈ × E₈ = 496; dim E₆ = 78; dim EIII = E₆/(Spin(10) × U(1)) = 32 with isotropy 16 ⊕ 16̄. These are stated correctly (the J₃(𝕆) parametrization at l.9965–9974, the E₆ branching at l.163/407, the EIII data at thm:vacuum l.486–489 are all sound).

But the *generative mechanism* — the single mirror operator M that is supposed to *force* this ladder — is not a well-defined object:

- The mirror M is not one rule. It acts in four qualitatively distinct, hand-selected modes (Boolean / Cayley–Dickson / Jordan / exceptional), one per rung, so the claim "the climb is forced, not chosen" (rem:climb:forced) is undermined by the per-rung mode-selection freedom.
- **def:mirror's defining identity M² = id is false for the Cayley–Dickson construction** it is applied to (M3 verdict, def:mirror). An operator whose stated algebraic property does not hold cannot be the generator it is claimed to be.
- **thm:rung-table is numerology:** the 8 → 27 step "dodges sedenions" — the actual Cayley–Dickson successor of the octonions is the 16-dimensional sedenions, not the 27-dimensional Albert algebra. The map from the 16-dim sedenions to the 27-dim J₃(𝕆) is asserted, never constructed (M2 verdict, J₃(𝕆) finding). lem:sedenion and lem:freudenthal are individually sound; the *transition* between rungs 4 and 5 is heuristic.
- **thm:e8 = gap, thm:e8e8 = unjustified, thm:e6 = circular** (see §2). The "product mirror is the unique M⁴ doubling compatible with the involution" (thm:e8e8) has no proof, only a plausibility appeal.
- The chart-dimension device def:chart-dim sets the (generally infinite) profinite cardinality |Sₙ| equal to dim Aₙ "by chart synchronization, not cardinality matching." This is a relabeling, not a theorem.

**Verdict on §1–5 foundations:** the endpoints are real mathematics; the operator and the climb that allegedly produce them are not. Stating a correct list of exceptional-group dimensions and asserting that a single involution generates them is reconstruction, not derivation.

---

## (2) The central theorems

### Capacity / partition (def:capacity, thm:concave, thm:partition) — **FATAL, numerology + circular**

This is the first dynamical link and it breaks.

- **def:capacity (eq:capacity, l.457–459) is an empty template.** The data that determine the maximizer are never defined: the costs c_i(a) appear only as "with costs c_i" (l.456); the vertex measure μ(a) is *never specified anywhere* (grep-confirmed); the index set, the vertex set V, and the "mirror-refined Albert–Cayley graph" G have no cell/edge structure. A variational problem with undefined data has no determinate maximizer, so "Bal_C is the unique maximizer" is **vacuous (unjustified, fatal).**
- **thm:concave (l.463–466) is stated wrongly and unproved (gap, major).** The bare claim "C is strictly concave" is false: −log-sum-exp is concave but invariant under a common shift, hence not strictly concave on ℝⁿ. Strictness holds only on the slice {Σωᵢ = 0} and only if the softmax Hessian is non-degenerate there — which depends on the undefined c_i, μ. The gradient sums to Σₐμ(a), so matching targets with Σmᵢ = 1 silently requires Σₐμ(a) = 1, never stated. The targets mᵢ are **inputs**, not outputs.
- **thm:partition (l.468–471) is the load-bearing numerology (FATAL).** It has *no proof block*. Its numbers (1/27, 10/27, 16/27) are exactly dim(irrep)/27 of the E₆ → Spin(10) × U(1) branching 27 = 1 + 10 + 16 — read off representation theory. They are fed back in as the targets mᵢ of thm:concave: **lem:capacity-gradient-d (l.768–769) explicitly writes "targets (m₁,m₂,m₃) = (1/27,10/27,16/27) from Theorem thm:partition."** The gradient equals the target because the target was the input. This is a closed circle: thm:partition is "justified" by lem:capacity-gradient-d, which cites thm:partition for its targets. With c_i undefined, the maximizer can be tuned to any simplex point (ω* − c gives any softmax distribution), so hitting the irrep dimensions is postdiction.
- **Cross-cutting circularity (major):** thm:e6 then uses the 1|10|16 split to *select* E₆ (l.402–408), but the split is only canonically defined *after* the Spin(10) × U(1) subgroup of E₆ is chosen. Selecting E₆ "because it yields 1|10|16" presupposes the subgroup it is meant to explain. At the bare J₃(𝕆) rung, E₆ acts irreducibly on the 27, so the split is not even defined at rung 5 without first importing E₆.

### Observer fixed point (thm:observer-fp, lem:cellular, prop:triality) — **FATAL, broken arithmetic**

The entire "3 families" result rests on one computation, **dim H¹(T₇, F₁₆)^{σ=−1} = 3**, and that computation is false on three independent grounds:

1. **The cochain quotient eq:cell-quotient (l.10119–10131) is arithmetically incoherent (false, fatal).** 1-cochains live in (ℂ¹⁶)³ (dim 48); the cocycle constraint φ₁+φ₂+φ₃ = 0 cuts to 32. The claimed coboundaries are the diagonal {(ψ,ψ,ψ)}, but the diagonal is **not in the cocycle space** (it satisfies Σ = 0 only if 3ψ = 0, i.e. ψ = 0). Worse, the paper asserts {32-dim}/diag(16-dim) = ℂ¹⁶ ⊕ ℂ¹⁶, a **32-dim quotient** — the quotient dimension equals the numerator dimension, so the quotient did nothing. A genuine quotient of 32 by a 16-dim subspace is 16. The headline "= 3" is then extracted by the phrase "triality with sign excludes linear dependence among the three choices" (l.10153) with **no computation.**
2. **σ cannot be both an order-2 involution and a 3-cycle-with-sign (false, fatal).** prop:triality makes σ a 3-cycle (C₁,C₂,C₃) → (C₂,C₃,C₁) with sign −1 on each stalk, giving order lcm(3,2) = 6. But σ² = id is asserted (l.598) and is *required* for the ±1 eigenspace splitting and for Π = ½(id − σ) to be a projector (eq:collapse-observer). An order-6 operator has 6th-root-of-unity eigenvalues; the entire ±1 decomposition every downstream object uses is invalid. The factor-exchange involution (def:prod-mirror) is genuinely order 2 — but then lem:triality-perm's exclusion of transpositions collapses the count.
3. **The mirror-index −3 misapplies Atiyah–Bott (unjustified, fatal).** eq:lefschetz/eq:mirror-index invoke the *holomorphic* Lefschetz formula on an explicitly **antiholomorphic** lift (l.668–669, 10181) — the wrong tool (one needs real/KR-equivariant index). An equivariant index is Σ_q (−1)^q Tr_{H^q}(σ), not Tr on H¹ alone; H⁰ is never shown to vanish. Decisively, **cor:observer-fp (l.694) states there are no fixed points in the 16-sector — by Atiyah–Bott the fixed-point sum is then empty = 0, contradicting −3.**

Everything downstream inherits this: def:observer presupposes dim = 3; the "family index operator" î is never constructed as a self-adjoint operator with derived spectrum {1,2,3}; the quotient I = H¹^{σ=−1}/Stab(d) with |I| = 3 is asserted (quotienting a vector space by a group action does not yield a 3-element set). **thm:iota-derived is circular (major):** argmax_ι cap_ι(τ) re-reads the input weights mᵢ — and additionally requires the cells *distinct*, contradicting the equal-cell requirement that the permutation/cohomology count needs.

### Soldering / spacetime (def:spacetime:H2O, thm:spacetime:soldering-derived) — **FATAL, dimensionally false**

The foundational premise of the entire spacetime sector is a checkable, elementary error:

- **def:spacetime:H2O (l.1089–1097) is false (fatal).** 2×2 Hermitian *quaternionic* matrices H₂(ℍ) = [[a,x],[x̄,b]] with a,b ∈ ℝ, x ∈ ℍ have real dimension 1+1+4 = **6, not 4**, and det = ab − |x|² has signature **(1,5)**, giving SL(2,ℍ) = Spin(1,5). The 4-dimensional Minkowski (1,3) object is H₂(**ℂ**), the complex one. The paper correctly gives H₂(𝕆) = ℝ^{1,9} (l.1091) but then mis-assigns the quaternionic restriction. The number d = 4 is matched to dim_ℝ(ℍ) = 4 as a numerical coincidence, while the object that supposedly carries it is 6-dimensional. Compounding error: l.1223–1226 says "the four imaginary quaternion units supply the clock directions" — Im(ℍ) has only **three** units (i, j, k).
- **thm:spacetime:soldering-derived (l.1198–1241) is therefore circular (fatal):** d = 4 and signature (−,+,+,+) are just a restatement of the false def:spacetime:H2O. The paper's own prop:coset-metric (l.1183) concedes the coset metric K|_m is *positive-definite (Euclidean)* and that "Lorentzian signature is carried by σ_sol" — i.e. the signature is inserted by hand through the soldering map, confirming it is an input, not an output. The claims "P1 fully derived" / "O2 closed" (rem:soldering-full-gap l.1316) are unsupported.
- **lem:quat-F4 is half-valid (unjustified, major).** The classical facts — G₂ = Aut(𝕆) acts transitively on quaternionic subalgebras; F₄ = Aut(J₃(𝕆)) — are correct and do single out *which* subalgebra ℍ_ι one uses. But equivariance is powerless to fix dimension or signature, which are intrinsic to the (mis-stated) H₂(−) construction. The lemma cannot do the work the chain assigns it.

### OS / reflection positivity (thm:rp → thm:rp-complete) — **FATAL on three independent grounds**

- **lem:goldstone-sigma (l.1465–1471) is false (fatal).** It identifies the σ = −1 eigenspace with the 28-dim massive block. But σ is the 16-vs-16̄ exchange involution (prop:mirror:involution; restated as Π = ½(id − σ) in prop:breaking-chiral), whose eigenspaces on V ⊕ V are diagonal/antidiagonal, each dim **16**. So the eigenspace split is necessarily 16/16, **not 28/4.** The 28/4 split is the *mass* split, a different grading; equating them is impossible (16 ≠ 28).
- **thm:sigma-measure (eq:gamma-split, l.1389–1393) does not factorize (unjustified, fatal).** The tensor product drops the sector-coupling term S_even with no justification; a generic H-invariant quartic (e.g. |δΦ₋₁|²|δΦ₊₁|²) is σ-even and couples the sectors. The claim that cross-terms vanish "because the action is σ-even while one insertion is σ-odd" (l.1434) is a non-sequitur (σ-evenness kills terms odd in total parity, not even quartic cross-terms).
- **prop:os-observer / thm:rp-complete are circular (fatal).** prop:os-observer begins "With thm:rp-gamma-neg **and the standard OS axioms** … for dμ_{Γ₋₁}" — taking the OS axioms as *hypotheses*. The ledger item **A7 (asm:os, l.2550) lists exactly those OS axioms as an assumption pointing back to prop:os-observer.** A7 is defined by prop:os-observer, and prop:os-observer assumes A7's content: a closed loop. The OS axioms (Euclidean invariance, clustering) are assumed, used to invoke the OS theorem, then cited as "OS reconstruction established."

Additional defects: thm:rp itself (l.1338–1365) is a **gap** — no covariance kernel, no reflection operator θ, no positivity inequality ⟨θF, F⟩ ≥ 0; "mu² > 0 ⇒ reflection-positive Gaussian" is false (positive mass is necessary, not sufficient). The reflection is taken on the *internal target coordinate* Φ⁰ rather than a spacetime time coordinate — a **category error** for OS. "Nonperturbative" (thm:rp-complete) is unsupported: every ingredient is semiclassical/one-loop or assumed.

### Gravitational fixed point (thm:kappa, κ* = 96π²/5) — **numerology (major)**

The algebra is fine but the result is the image of an assumed input. **thm:kappa assumption (iv) asserts the anomalous dimension η_MPl = −(5/48π²)κ;** the fixed point κ* = 96π²/5 and slope −2 then follow algebraically from this asserted η. β = 2κ − cκ² gives FP 2/c for *any* c, so κ* is simply the image of the chosen coefficient. **thm:kappa-persistence is circular (major):** it proves κ* survives observer collapse by *assuming* the graviton and R² modes are σ = +1; but the 28 scalars that induce gravity are σ = −1 (thm:planck), so the graviton should inherit σ = −1. The Sakharov MPl² coefficient (28/6) is sound (Seeley–DeWitt), but its inputs (ξ = 0, sign, neglected R²/Weyl² ghost terms) are assumed (gap, major).

---

## (3) The breaking chain and Yukawa construction

### GUT breaking chain (thm:breaking-*) — **FATAL group theory errors + reverse-engineered ordering**

The narrative is a standard SO(10)/SU(5) breaking recast in capacity language, with three compounding failures.

- **16 × 16 = 1 + 45 + 126 is false (fatal).** Used to source the 126_H Higgs (l.2149–2150, 2215). The correct product is 16 × 16 = 10 + 120 + 126 (dim 256). The stated "1 + 45 + 126" has dim 172 ≠ 256; "1 + 45 + 210" (dim 256) is 16 × **16̄**. The paper conflates the two products.
- **126 → SU(5) × U(1)_X branching is false (fatal).** The stated pieces (l.2219–2224) sum to 10+10+15+15+1 = **51 ≠ 126**; the correct branching has six terms summing to 126. ~75 states are missing.
- **SU(5) → SM branchings of 15 and 10 are false (major).** Stated pieces sum to 7 and 6 instead of 15 and 10 (l.2284–2285) — the content listed is that of 5/5̄ where a 10 belongs.
- **lem:breaking-cw is the unjustified engine (fatal):** "exactly one negative Hessian eigenvalue per ι-cell" is the mechanism producing the three sequential breaks, but **no CW eigenvalue is ever computed**, and the paper's own ledger flags the CW gap as **assumption A3 (asm:cw, l.2542)** while the breaking theorems use it as established.
- **lem:breaking-capacity-order is numerology (major) and internally inconsistent:** "the quadratic coefficient is ∝ w_ι" (l.2076) then "tachyonic when τ > τ* ∼ 1/w_ι" (l.2078) — a *positive* coefficient with w > 0 is stable, not tachyonic. The triple (1/27,10/27,16/27) is reused for three distinct roles (sector mass, VEV scale, threshold time) with no proof they coincide. The ordering 16/27 > 10/27 > 1/27 is mapped onto the breaking chain precisely because that is the order needed.
- **thm:breaking-e6so10 is circular (major):** E₆ → Spin(10) × U(1) is the *definition* of the assumed vacuum coset (thm:vacuum, thm:universe), so "first breaking" restates an input; and assigning it to a 16_{−3} direction is group-theoretically self-contradictory — a 16-VEV breaks Spin(10), it does not preserve it.

**What is sound here is input-equivalent:** sin²θ_W = 3/8 (prop:weinberg-gut) is the correct standard SU(5) trace identity ½ ÷ 4/3, but it is the universal textbook GUT value inherited from the *assumed* canonical embedding — it would be 3/8 in any SU(5) GUT and carries no distinctive information from the capacity machinery. b₀ = 12 arithmetic checks (88−24−28)/3 but the scalar count 28 is entered as a bare numerator (coset dimensions, not a Dynkin index).

### Yukawa / flavon (def:overlap-form, thm:flavon-vev, thm:Hij-derived) — **FATAL, undefined domain + algebraically false**

- **The overlap integral has no domain (unjustified, fatal).** Y_ij = ∫_{Q̂} ω_i ∧ ω_j ∧ Φ_Higgs requires a complex manifold with a Kähler volume form, but Q̂ = Q/σ is a **profinite Stone space** (prop:stone-lift) — totally disconnected, with no smooth structure, no differential forms, no wedge product. T₇ is a finite cell graph, not a 7-torus. The integral is undefined; the claimed continuum-integral = cellular-sum equivalence (lem:yukawa-cellular) cannot hold while one side is undefined.
- **thm:flavon-vev is algebraically false (fatal), verified by direct computation.** After eliminating φ₃, the reduced potential is a homogeneous quadratic Av₁² + Bv₂² + Cv₁v₂ with det = 3k² + 4S₁k + 4S₂. (i) The claimed κ_F* = S₂/S₁ = 62/243 gives det = 44020/19683 ≠ 0, so it does **not** zero the determinant as the proof claims. (ii) det vanishes only at k ≈ −0.344, −0.990, both **negative**, but def:flavon-potential requires κ_F > 0, for which the form is positive-definite and the only stationary point is the origin (no nonzero VEV). (iii) At the true roots the null eigenvector gives v₁²/v₂² = 2.17 or 0.074, **not** the target 0.10. The relation vᵢ² ∼ wᵢ is asserted, not derived; a homogeneous quadratic cannot fix a VEV magnitude even if degenerate.
- **thm:Hij-derived violates its own definition (false, major).** def:Pij-offdiag defines H_ij as a normalized correlation coefficient, which by Cauchy–Schwarz lies in [0,1]. eq:Hij-derived gives **H₁₂ = 1.37, H₁₃ = 1.56 > 1.** The "unique symmetric degree-zero rational function consistent with H_ij = 1 on the diagonal" is uniqueness-by-fiat (infinitely many functions satisfy that). It is computed from the non-existent flavon VEVs, so it is void regardless.
- **lem:Pij-hub limit is wrong (false, major):** the stated harmonic-mean form gives P_ij → 1/2 as w_i/w_j → 0, not the → 0 the proof requires for its own uniqueness boundary condition.

**The downstream numbers are postdictions.** m₃:m₂:m₁ = 16:10:1 does not match the measured charged-lepton ratios m_τ:m_μ:m_e ≈ 3477:207:1. CKM/PMNS are admitted ~10% fits requiring unstated RG "compression" factors and per-channel residue exponents (1/5, 1/4, 1/8, 1/7, 1/24 …) chosen to land on PDG values — by the authors' own statement the Cabibbo exponent 1/5 is "the unique leading rational that maps λ_tree ≈ 0.249 to PDG λ," i.e. an explicit tuning to the target.

---

## Consolidated issue list

### FATAL
1. **def:capacity (l.457–459):** functional data (c_i, μ, graph G, index set) wholly undefined; "unique maximizer" vacuous.
2. **thm:partition (l.468–471):** no proof; numbers are dim/27 of known irreps fed back as targets via lem:capacity-gradient-d (l.768–769). Circular numerology.
3. **lem:cellular / eq:cell-quotient (l.10119–10153):** quotient arithmetically incoherent (diagonal not in cocycle space; 32/16 → 32); "= 3" asserted.
4. **prop:triality (eq:triality-perm/sign):** σ claimed order-2 and order-6 simultaneously; ±1 eigenspace decomposition invalid.
5. **thm:mirror-index / eq:lefschetz (l.685–689, 10192–10207):** holomorphic Atiyah–Bott misapplied to an antiholomorphic involution; empty fixed-point set forces 0, not −3.
6. **def:spacetime:H2O (l.1089–1097):** H₂(ℍ) has dim 6 and signature (1,5), not 4 and (1,3). Elementary, checkable, load-bearing.
7. **thm:spacetime:soldering-derived (l.1198–1241):** d = 4 and Lorentzian signature are inputs (conceded by prop:coset-metric l.1183), not derived; circular.
8. **lem:goldstone-sigma (l.1465–1471):** σ-eigenspace split must be 16/16, falsely equated with 28/4 mass split.
9. **thm:sigma-measure (eq:gamma-split l.1389–1393):** tensor factorization drops sector-coupling S_even; unjustified.
10. **prop:os-observer / thm:rp-complete (l.1595–1637):** OS axioms assumed as A7 (l.2550) then cited as established; closed loop.
11. **16 × 16 = 1 + 45 + 126 (l.2149–2150):** dim 172 ≠ 256; sources the 126_H from a false product.
12. **126 → SU(5) × U(1)_X branching (l.2219–2224):** sums to 51 ≠ 126.
13. **lem:breaking-cw (l.2034–2053):** "one tachyon per cell" asserted, no eigenvalue computed; ledger A3.
14. **def:overlap-form / eq:yukawa-overlap:** integral over a profinite (totally disconnected) Q̂ is undefined.
15. **thm:flavon-vev:** κ_F* does not zero the determinant; with κ_F > 0 only the trivial minimum exists; null eigenvector ≠ target ratio (all verified by computation).

### MAJOR
- **thm:concave (l.463–466):** "strictly concave" false on ℝⁿ; no proof; targets are inputs.
- **thm:e6 (l.402–408):** uses 1|10|16 to select E₆, but the split presupposes the chosen E₆ subgroup — circular.
- **thm:iota-derived (l.794–818):** argmax re-reads input weights; requires distinct cells contradicting the equal-cell cohomology count.
- **def:observer / def:observer-index (l.610–639):** family index operator î never constructed; |I| = 3 asserted; quotient by Stab(d) undefined.
- **thm:lifetime-unique (l.914–967):** conditional on assumption A6 (iota-sector convexity, asm:convex l.2548), in tension with thm:vacuum's 32 flat Goldstone directions, yet advertised as "closing Open Problem O1."
- **def:lifetime / eq:lifetime-admitted:** first-order gradient flow posited, not derived from the second-order action eq:action.
- **lem:quat-F4 (l.1106–1141):** equivariance fixes the subalgebra but cannot fix dimension/signature; rung-3 → rung-5 map never constructed.
- **prop:vierbein (l.1276–1281):** signature imported via σ_sol; coset metric K is positive-definite (prop:coset-metric); non-degeneracy of e^a_μ not shown.
- **thm:rp (l.1338–1365):** no covariance kernel, no θ, no positivity inequality.
- **thm:kappa (κ* = 96π²/5):** η coefficient is assumption (iv); FP is image of chosen constant.
- **thm:kappa-persistence:** assumes graviton/R² are σ = +1 while the gravity-inducing 28 scalars are σ = −1.
- **thm:cc-sequester / thm:cc-rg:** CC = 0 assumed (prop:cc-decouple "Suppose…"); Step 2 (l.1850) σ=−1 projection of the σ=−1 component ≠ 0; circular.
- **thm:beta (b₀ = 12):** scalar count 28 entered as coset dimension, not a Dynkin index.
- **lem:breaking-capacity-order:** positive quadratic coefficient cannot be tachyonic; one input reused for three hierarchies.
- **thm:breaking-e6so10:** restates the assumed vacuum little group; 16-VEV cannot preserve Spin(10).
- **thm:breaking-complete (eq:breaking-chain l.2376):** U(1)_X silently dropped; SU(5) → SM needs the 24-adjoint absent from listed reps.
- **SU(5) → SM branchings of 15, 10 (l.2284–2285):** dimensionally wrong (7, 6 vs 15, 10).
- **thm:Hij-derived:** correlation coefficients 1.37, 1.56 > 1, violating the definition.
- **lem:Pij-hub:** limit → 1/2, contradicting the lemma's own boundary condition; usage inconsistent across tables.
- **lem:yukawa-cellular:** ε_ijk antisymmetric rule forces diagonal pairings to vanish, contradicting the diagonal-dominant Yukawas it feeds.
- **prop:yukawa-factor:** ‖ω_ι‖² ∼ w_ι asserted; 16:10:1 restates the input and mismatches measured lepton ratios.

### MINOR
- **thm:duality (l.587–603):** "the only Z₂ structure at n = 7" asserted with no classification; named, not proved, as a duality pairing.
- **thm:vacuum (l.486–489):** "coercive on compact E₆" is a category error (V lives on J₃(𝕆), not the group; coercivity is meaningless on a compact set); dynamical claims (rank-one minima, 32 negative modes) asserted without a Hessian.
- **prop:soldering-clock (l.1243–1274):** "unique timelike direction" is correct only for (1,3); on the actual (1,5) block it is ill-defined; Spin(1,9) → Spin(1,3) skips Spin(1,5)/Spin(1,4).
- **prop:stone-lift (l.10157–10177):** Leray E₂ degeneration asserted, no spectral-sequence computation; transports an unestablished "3."
- **lem:goldstone-rp-unique (l.1510–1562):** "unique" is unique modulo Diff₊(ℝ) × SO(3); existence never proved.
- **prop:weinberg-gut (3/8):** correct but input-equivalent to the standard SU(5) embedding; no distinctive content.
- **Internal contradiction in the 16-cell capacity:** def:tower7 (l.9994) says 16/81; lem:triality-perm (l.10050) says 16/27 — direct numerical contradiction (48/27 > 1 is impossible for a normalized partition); load-bearing for lem:cellular.
- **def:universe / eq:universe:** inverse limit varprojlim has no bonding maps; "stabilizes at J₃(𝕆)" contradicts the E₈/E₈×E₈/E₆ terminus.

---

## Recommendation

**Reject.** Three of the five central theorems (observer fixed point, soldering/spacetime, OS reconstruction) contain elementary, checkable mathematical errors (a 6-dimensional space called 4-dimensional; an order-6 operator called an involution; a quotient that does nothing; a holomorphic theorem applied to an antiholomorphic map; an OS-axiom assumption cited as its own proof). The two "quantitative successes" that are correct (sin²θ_W = 3/8, the b₀ arithmetic) are standard textbook results inherited from an assumed embedding and carry no information specific to the construction. Every framework-specific number — the 1|10|16 partition, the family count 3, κ*, the Yukawa hierarchy, the CKM/PMNS angles, Ω_DM = 4/15, the muon g−2 — traces to inputs (irrep dimensions, an assumed anomalous dimension, tuned residue exponents, an assumed IR matching scale) re-extracted at the output and presented as predictions. The manuscript is a reconstruction of known representation theory plus a suite of postdictions, not a derivation. The honest minimum revision would be to retitle every "derived" claim as "consistent with / reconstructed," fix the false group theory (16×16, the 126 branching, H₂(ℍ)), discharge or openly label assumptions A3/A6/A7, and recompute the cellular cohomology honestly with eigenspace dimensions that sum to 32 — but this would dismantle the paper's central thesis that the Standard Model follows from two axioms.


# Part II — Phenomenology referee report

I'll write the phenomenology/physics section of the referee report based on the structured verdicts provided. Let me synthesize these into a rigorous PRD referee report section.

# Referee Report: SJET — Phenomenology and Physics Section

## Summary of Physics Verdict

The manuscript advertises a "parameter-free Tier-T1" theory in which Standard-Model and cosmological observables are *derived* from a void-plus-mirror ledger with no adjustable inputs. On audit of the quantitative content (Secs. 8–23), this central claim does not survive. The recurrent pattern across every numerical sector is the same: a small set of group-theory integers — the E6 partition weights $(w_1,w_2,w_3)=(1/27,10/27,16/27)$, the coset count $n_S=28$, the family count $n_F=3$, the one-loop coefficient $b_0=12$, and $\kappa_\star=96\pi^2/5$ — are recombined into ad hoc exponents and prefactors that are tuned to land on known data. Tree-level values are typically wrong by factors of 2–20 and are rescued by "RG/threshold" factors whose normalizations have no legitimate renormalization-group basis. Several headline numbers do not even reproduce from the manuscript's own equations, and at least three "predictions" are now in direct conflict with current data. I find essentially **no genuine, parameter-free quantitative prediction** in the manuscript. The genuine content is qualitative and low-information (no fourth generation, normal neutrino ordering, a generic GUT-scale proton-lifetime bound), and the SM-Lagrangian section is a correct transcription rather than a derivation.

Below I assess each domain, classifying every claim as **derived** (follows by stated math from stated premises), **postdiction/fit** (parameters tuned to known data and presented as prediction), or **asserted** (claimed without working steps), and state its status against data.

---

## 1. Standard-Model Lagrangian (Sec. 8, lines 2615–3234) — *salvageable*

**Genuinely correct (but copied, not derived):** The gauge group $SU(3)\times SU(2)\times U(1)$, the $-\tfrac14 F^2$ kinetic term, the covariant derivative, the fermion kinetic term, one Higgs doublet, and the Higgs potential are all standard textbook forms. They are **asserted/transcribed**; the claimed "projection" that yields them is stated without computation. The single textbook GUT fact $\sin^2\theta_W(\text{GUT})=3/8$ is correctly **derived** (it is a group-theory identity, not a measurement). The Yukawa structure ($126_H/10_H$) has the correct form but the portal labels do **not** constrain the free $3\times3$ matrices, so no flavor content is fixed here.

**Defects requiring correction (major):**
- **Ghost sector** is internally inconsistent: it exhibits a dimension-6 (three-derivative) operator and a spurious abelian ghost coupling, contradicting the standard $-(D_{\rm adj})^2$ ghost Lagrangian.
- **Gauge-fixing term absent:** ghosts are retained while the $-\tfrac{1}{2\xi}(\partial A)^2$ term is omitted — BRST invariance is broken; the quantization is ill-posed as written.
- **Right-handed neutrinos omitted** while a $126_H$ seesaw is simultaneously "built" — internally contradictory.
- **Strong-CP $\theta=0$ "by $\sigma$-odd fiat"** ignores $\bar\theta$ (the physical combination $\theta-\arg\det M_q$); the parity-selection argument is unphysical (see Sec. 8 below).
- **Anomaly cancellation** ($16-80+64=0$) gives the right answer, but the $\mathbf{16}$ of Spin(10) is *automatically* anomaly-free; there is no Green-Schwarz mechanism in play, so the arithmetic is decorative numerology.

**Status:** The SM forms agree with reality because they are the SM, copied. As a *derivation* this section is incomplete; as a transcription it is mostly correct but contains real quantization defects (ghosts, gauge fixing) that must be fixed.

---

## 2. Fermion Masses and Mixings (Sec. 9.6–9.7) — *broken*

The one genuinely non-tuned input is the partition $(1/27,10/27,16/27)=$ the SO(10) sub-rep dimensions $1+10+16$ inside the E6 **27**. This is a real group-theory fact. **The load-bearing illegitimate move** is that these are sub-reps *within a single generation's* **27**, yet the manuscript silently reuses them as the three fermion *generations* ($e/\mu/\tau$). This reinterpretation (Table tab:yukawa-leading, Pred. pred:sector-ratios) is **asserted**, and every mass ratio below inherits it.

- **Tree-level hierarchy $1:10:16$ — EXCLUDED (fatal).** This gives $m_\mu/m_e=10$ vs measured 206.77 (off $\sim$20×) and $m_\tau/m_\mu=1.6$ vs 16.82 (off 10×). The Yukawa is $\sqrt{w_i}$ (ratios $1:3.16:4$) but the mass is *declared* $\propto w_i$ — the squaring is asserted to manufacture the larger spread. Neither $1:10:16$ nor $1:3.16:4$ matches any charged-fermion sector.
- **$m_\mu/m_e$ at $M_Z$ — FIT (fatal).** The factor needed to lift 10 to 206.8 is $(\text{base})^{6.84}$; the paper assembles the exponent $n_S n_F/b_0 = 28\cdot3/12 = 7$ from independently named integers to land exactly there. One-loop RG of a Yukawa ratio gives a *single* power, not a 7th power; "7 threshold crossings exponentiate the residue ratio" (lines 4209–4220) is **not** an RG result. Worse, the two equational forms (eq:lepton-ratio vs eq:lepton-ratio-RG) are **mutually contradictory** — one gives 0.018 (muon lighter than electron!), the other 222 — because the sign of the RG log is flipped between them. The stated 201 is reproducible from neither (recomputation gives 222).
- **Internal consistency — EXCLUDED (fatal).** The same machinery gives $m_\tau/m_\mu=14.6$ (13% off) and $m_\tau/m_e=1589$ (>2× off), and **violates multiplicativity**: $(m_\mu/m_e)(m_\tau/m_\mu)=3242$ vs direct $m_\tau/m_e=1589$ — a factor-of-2 internal contradiction, because the exponent acts on the non-additive difference $(w_i-w_j)$. The paper exhibits only $m_\mu/m_e$ and never tests the consistency relation that exposes the single-point fit.
- **Neutrino $\Delta m^2_{31}/\Delta m^2_{21}$ — FIT (major), agrees with data by engineering.** Tree $w_3/w_1=16$ vs measured $\sim$33.5; the factor-of-2 enhancement requires $\ln(M_{\rm GUT}/\mu_{\rm osc})=32.4$, obtained by replacing the oscillation scale $\mu_{\rm osc}\sim0.01$ eV with $M_Z$ (line 4035) — a $\sim$30-orders-of-magnitude substitution justified only because it produces the right log. The paper's own sensitivity table (row 4) shows $\mu_{\rm eff}=M_{F3}$ gives 22.4 (excluded). The two-loop $\Xi_{\rm th}$ arithmetic is also wrong (middle term 18.3 should be 15.4). Final number agrees, but the agreement is a postdiction.
- **Cabibbo $\lambda$ — FIT (major), tension.** Eq. ckm-lambda asserts a **false equality** $\sqrt{w_1/w_2}\,P_{12}=(2/11)H_{12}$ (LHS $=0.182$, RHS $=0.249$); the paper reports the 0.249 branch (11% above PDG 0.2245). Table tab:Pij-ckm mislabels $P_{12}\cdot H_{12}$ as $P_{12}$.
- **PMNS $\sin\theta_{23},\ \sin\theta_{13}$ — FIT (minor), tension.** 0.78 vs 0.75 ($\sim$5%), 0.18 vs 0.149 ($\sim$20%); honestly hedged as "O(10%) tree-level contacts," so lower severity, but they inherit the $H_{ij}/P_{ij}$ ambiguities. Not genuine predictions.

**Status:** Broken. Every numerical contact is a single-point fit with a reverse-engineered exponent ($n_S n_F/b_0=7$) or is internally inconsistent.

---

## 3. Scale Identification (Sec. 10, lines 4365–4727) — *broken, two fatal errors*

- **EW VEV $v$ — POSTDICTION (major), agrees trivially.** The formula reproduces 246.4 GeV, but only because $M_Z$ (measured) is multiplied by a dimensionless prefactor $\sqrt{\kappa_\star w_1/w_3}\sqrt{\sin^2\theta_W/(3/8)}\approx2.70$ that simply *equals the measured ratio* $v/M_Z=2.700$; $\sin^2\theta_W$ is itself a measured/RG input. This is a relation among already-measured quantities, not a removal of $v$ as a free parameter.
- **Induced Planck mass $M_{\rm Pl}$ — EXCLUDED (FATAL).** Plugging $\mu=v=246$ GeV and $\ln(\Lambda^2/v^2)\approx65$ into eq:planck-explicit gives $M_{\rm Pl}\approx342$ GeV, **not** $1.2\times10^{19}$ GeV — wrong by **16 orders of magnitude**. To reach $10^{19}$ GeV the log would need to be $\sim8\times10^{34}$. $M_{\rm Pl}$ is in fact *inserted by hand* and then used to fix $M_{\rm GUT}$, contradicting the "T1 derived" label and the "D3 is closed" remark. This is the section's central claim and it is false.
- **$\mu=v$ matching — ASSERTED (major).** $\mu=v$ is an imposed matching condition, not a derived equality; the radiative-curvature integral is never evaluated. This imposition is what propagates the fatal $M_{\rm Pl}$ error.
- **UV cutoff $\Lambda$ — FIT (major), internally inconsistent.** Eq. Lambda-kappa silently drops a $(4\pi)^2\approx267$ factor (claimed $\sim10^{-3}$ vs actual $1.2\times10^{-5}$), hidden in an "O(1)." $\Lambda:=M_{\rm GUT}$ by fiat, and $M_{\rm GUT}$ depends on the externally inserted $M_{\rm Pl}$ — the triangle $M_{\rm Pl}\!\to\!M_{\rm GUT}\!\to\!\Lambda\!\to\!M_{\rm Pl}$ is **circular**.
- **$M_{\rm GUT}$ — PARTIAL (minor).** $4.18\times10^{16}$ GeV is reproduced and lands in the conventional band, but it is $M_{\rm Pl}$ (external input) times a dimensionless combination; not an absolute-scale prediction.
- **$\kappa_\star=96\pi^2/5$ — DERIVED (none).** Correct pure number ($189.5$), but dimensionless, so it fixes no absolute scale by itself.
- **Muon $g-2$ — POSTDICTION (FATAL), now excluded.** See Sec. 4.

**Status:** The dimensional logic meant to fix absolute units is circular, arithmetically wrong by 16 orders of magnitude ($M_{\rm Pl}$), or a relabeling of measured ratios ($v$). The "D3 closed" claim is unsupported.

---

## 4. Muon $g-2$ (Sec. 11, lines 4727–5046) — *broken, now in direct conflict with the data it cites*

**The one sound piece:** the one-loop scalar integral eq:g2-integral correctly reproduces Leveille (1978); its heavy-scalar limit $y^2/(48\pi^2)$ is verified ($M/m_\mu=2327$ saturates the $1/6$ limit). The integral is not where the problem lies.

**Everything else is reverse-engineering.** The natural result for 28 scalars is $n_S y_\mu^2/(48\pi^2)\sim1.1\times10^{-8}$, $\sim$4× too big. Two manufactured factors then conspire: (i) "democratic sharing" $y_{\mu a}=y_\mu/\sqrt{28}$ divides out the 28 (asserted via undefined "capacity balance"); (ii) the **sector factor** $S_\mu=n_S n_F/b_0=7$ multiplies it back up. Net $7/28=1/4$.

- **$S_\mu=7$ — ASSERTED (FATAL).** A fixed-scale one-loop amplitude *cannot* be multiplied by a beta-function ratio $n_S n_F/b_0$; this conflates RG running with loop multiplicity and is conceptually unjustified. With $n_S,n_F,b_0$ fixed elsewhere, 7 is the unique product that lands on the old anomaly.
- **$y_\mu=m_\mu/v$ — FIT (minor).** This is the muon Yukawa from the *measured* muon mass; mislabeling it "T1 derived" overstates — it is an input.
- **$\Delta a_\mu^{\rm SJET}=7/(48\pi^2)(m_\mu/v)^2$ — POSTDICTION (FATAL), now EXCLUDED.** Arithmetic verified ($2.73\times10^{-9}$ one-loop, $2.58\times10^{-9}$ "two-loop"), but the target $\sim2.6\times10^{-9}$ is the **superseded 2020 data-driven anomaly**. The 2025 Theory Initiative White Paper (arXiv:2505.21476), *which the paper itself cites*, gives $a_\mu^{\rm SM}=116592033(62)\times10^{-11}$; with the final Fermilab $a_\mu^{\rm exp}=116592070.5(14.6)\times10^{-11}$ the real anomaly is $\Delta a_\mu=3.8(7)\times10^{-10}$, $\sim0.6\sigma$, **consistent with zero**. A $2.6\times10^{-9}$ contribution is now excluded at $\sim$3–4$\sigma$. The paper even quotes an internally inconsistent $a_\mu^{\rm SM}=1.16591810\times10^{-3}$ (the *old* number) while claiming it is the 2025 lattice value.
- **Two-loop factor $R\approx0.96$ — ASSERTED (major).** An unjustified single-log insertion whose only function is to nudge the result to claim "<1%."

**Status:** A numerological postdiction tuned to an obsolete anomaly, built on an unjustified beta-function normalization, now in direct conflict with the latest data the paper purports to use.

---

## 5. Quark Masses and CKM (Sec. 12, lines 5046–5490) — *broken*

The workbook arithmetic is internally reproducible to $\sim$1%, but the structure is numerology with **more knobs than targets**: $\sim$18 effectively free quantities (five asserted $R^{(2)}$ factors, $\sim$10 hand-picked rational exponents, $\sim$3 stray multipliers) reproduce 9 observables, directly contradicting Cor. cor:quark-tier-c's "no free parameters" claim (**asserted FATAL**).

- **Open admission of fitting:** the proof of eq:lambda-refined states the exponent $1/5$ is "the unique leading rational that maps $\lambda_{\rm tree}\sim0.249$ to PDG $\lambda$" — the textbook definition of a fit. **$\lambda=0.2265$ — FIT (FATAL):** $\sim$2.2$\sigma$ above the true PDG2024 $\lambda=0.22497(70)$.
- **Fabricated experimental target — FATAL:** $m_c/m_s$ is asserted as "PDG2025 $13.6\pm0.5$," but the real lattice value (FLAG2024) is $11.77(3)$. The SJET 13.6 is declared a "<1% match" to an invented number; the true value is excluded by $\sim$30$\sigma$.
- **Cherry-picked scheme — major:** $m_t/m_b=41.3$ holds only in a mixed pole-top/MSbar-bottom scheme; at common $M_Z$ MSbar it is $\sim$57.
- **Sign contradiction — major:** Theorem eq:flavon-residue uses $\ln(\mu/M_{\rm GUT})$ (giving $R<1$) while the workbook uses $\ln(M_{\rm GUT}/\mu)$ ($R>1$); only the workbook sign reproduces the headline $m_s/m_d=20.0$. The published numbers do not follow from the stated theorem.
- **Two inconsistent $|V_{ub}|$ values — major:** 0.00369 (direct) vs 0.00350 ($A\lambda^3$); the direct form uses $R_{ub}^{(2)}=0.095$, a 10× suppression mislabeled a "two-loop threshold factor" (a real 2-loop correction is O(1+few%)).
- **$\eta$ (the genuinely nontrivial CP observable) — FIT (major):** four independent hand-chosen exponents/factors (3/4, 1/2, 1/7, 1.009); landing near $\bar\eta=0.355$ carries zero predictive weight.
- Against current PDG2024 ($A=0.839(11)$), SJET $A=0.790$ is $\sim$5$\sigma$ low; the paper hides this by quoting selective "PDG2025" centrals with implausibly small errors.
- **Only on-target quantity:** $m_s/m_d\sim19.8$, and even it requires two free knobs (exponent + $R_{sd}$).

**Status:** Reverse-engineered postdiction with free parameters disguised as fixed, partly fabricated/stale targets.

---

## 6. QCD and Hadrons (Sec. 13, lines 5490–5998) — *broken*

**One sound result:** $SU(3)_c$ as the unbroken color factor of the Spin(10)$\to$SU(5)$\to$SM cascade is correct GUT group theory (**derived**).

- **$\alpha_s(M_Z)$ — FIT (FATAL), excluded:** eq:alpha-s-residue is *not* the QCD RG equation (it is a multiplicative residue increasing $\alpha_{\rm GUT}$, not the subtractive inverse-log; uses $b_0=12$ not the SM slope 7). Arithmetic is internally wrong ($\ln=33.8$ not 36.8). Either 0.126 or 0.14 lies tens of $\sigma$ above PDG 0.1180(9).
- **$\Lambda_{\rm QCD}$ — ASSERTED (FATAL), excluded:** the cited formula evaluates to $\sim$29 keV, **four orders of magnitude** below the claimed 0.28 GeV, and uses the wrong $\beta$ coefficient. The 0.28 GeV is asserted and propagates into the string tension and every hadron mass.
- **String tension — ASSERTED (major), tension:** $(16/27)\Lambda^2$ yields $\le0.073$ GeV$^2$, contradicting the table's claimed 0.13 GeV$^2$ and the lattice $\sim$0.18.
- **Hadron masses ($m_p,m_\pi,m_\rho$) — FIT (major):** the advertised cellular-overlap formula eq:hadron-mass-def is **never evaluated**; $A_H$, $P_{ijk}$, $\chi_{\rm flux}$ are never turned into numbers. The quoted masses are unrelated dimensional-analysis ansätze with hand-picked prefactors ($b_0/4=3$, "2", "2") and slot pairs. The "exact" $m_\pi=0.140$ is doubly suspect: it needs $\sqrt{w_1/w_3}=1/4$ with a fitted prefactor, AND it **mis-models the pion**, a pseudo-Goldstone boson whose mass is set by chiral symmetry breaking (GMOR), not a generic colorless overlap. $m_\rho$ is 14% high.
- **Confinement — ASSERTED (minor):** symmetry hand-wave; area law postulated, not derived.

**Status:** One correct group-theory corollary surrounded by fatal arithmetic errors and postdicted hadron numbers.

---

## 7. Electroweak Precision (Sec. 14, lines 6001–6491) — *broken*

The agreeable entries are **SM recomputations with hidden SM inputs**; the genuinely new quantity is excluded.

- **$\Delta\rho$ / $\rho$ — POSTDICTION (FATAL), excluded:** SJET gives $\Delta\rho=0.0034$ vs measured $0.00038$ — factor $\sim$9, $\sim$11$\sigma$ on $\rho$. Table 1 (line 6398) disguises this as "<0.3%" by quoting the percentage on $\rho\approx1$ instead of the physical $\Delta\rho$ — a denominator trick hiding an excluded result. The dominant SM top-loop contribution ($\sim3G_F m_t^2/8\sqrt2\pi^2$) is entirely absent.
- **$T$ parameter — ASSERTED (FATAL):** set $T=\Delta\rho$, violating Peskin-Takeuchi $\alpha T=\Delta\rho$ — wrong by $\sim$128. "Agrees" only because the band ($\pm0.12$) is $\sim$35× the quoted value.
- **$\alpha_{\rm em}^{-1}(M_Z)$ — FIT (major), tension:** running the *electromagnetic* coupling with the *unified* SO(10) $b_0=12$ is a category error; the dominant term is fixed by the tuned anchor (24) and $b_0$; arithmetic does not close (recompute 128.2, not 127.4).
- **$M_W$, $\Gamma_Z$, $\sin^2\theta_W$ — FIT/postdiction (major/minor), agree trivially:** these reuse SM relations ($M_W=g_2 v/2$) and SM reference values ($\Gamma_Z^{(0)}=2.496$ GeV inserted by hand, line 6351; $\sin^2\theta_W=0.231$ imported). The $M_W$ proof (line 6274) contains a sin/cos confusion ($\sqrt{1-0.231}=0.4806$ is actually $\sqrt{0.231}=\sin$, not cos).
- **$S$ — FIT, internally inconsistent:** eq:6296 evaluates to 0.005, but 0.0015 is reported.
- **Oblique $S,T,U$ "consistency" is vacuous:** every quoted value is 30–300× smaller than its experimental error bar.

**Status:** Agreeable entries are SM recomputations; the genuinely new quantity ($\Delta\rho$, hence $T$) is excluded at $\sim$11$\sigma$ and concealed by a misleading percentage.

---

## 8. Strong CP and Baryogenesis (Sec. 15, lines 6494–6951) — *broken*

- **Strong CP $\theta_{\rm obs}=0$ — ASSERTED (major), agrees but not derived.** The vanishing rests entirely on Lemma lem:theta-parity *asserting* (no computation) that the mirror involution $\sigma$ acts as $-1$ on $\mathrm{Tr}(F\tilde F)$. This is a relabeling of the known mirror/parity solution (Barr; Babu-Mohapatra), with $\theta$ sequestered into an unobservable $\sigma=+1$ reservoir (**unfalsifiable**). The hard part of such models — radiative feedback of $\theta$ into the visible sector — is never addressed.
- **$\kappa_{\rm sph}=28/27>1$ — ASSERTED (FATAL), tension:** a sphaleron conversion fraction must be $<1$; the canonical SM value is $28/79\approx0.35$. The "28" is grabbed from coset scalars and "27" from a capacity normalization — numerology, a $\sim$3× overcounting.
- **$\eta_B$ — POSTDICTION (FATAL), with a load-bearing arithmetic error:** the paper's own eq:6841, $(3/16)(10^{12}/4\times10^{16})(7\times10^{-4})(28/27)$, evaluates to $3.4\times10^{-9}$, **5.6× the claimed $6.1\times10^{-10}$**; the Theorem's alternative (also $\times w_3$) gives $2.0\times10^{-9}$ (3.3× too high and inconsistent with the Proposition). *Neither* reading reproduces the quoted number. The mass ratio $M_{F3}/M_{\rm GUT}$ illegitimately stands in for the entire thermal efficiency×abundance×entropy chain (no Boltzmann equation, no washout $K$, no $g_\star$, no $n/s$, no $\eta_B=7.04\,Y_B$). The CP phase $\arctan(1/4)$ is reverse-mapped from the already-fixed flavon vev ratio via an invalid "arg of a positive real $=\arctan$" step; $\epsilon_{\rm CP}=7\times10^{-4}$ is plugged in (and is anomalously large for high-scale leptogenesis).

**Status:** Strong CP is *assumed* (relabeled mirror parity); $\eta_B$ is a postdiction whose own arithmetic fails by 5.6×.

---

## 9. Dark Matter (Sec. 16, lines 6951–7337) — *broken*

- **$\Omega_{\rm DM}=4/15\approx0.267$ — FIT (major), agrees by knob placement.** eq:dm-fraction is a *static* capacity ratio $w_3/[4(w_1+w_2)+w_3]$ with **zero relic-abundance physics** — no Boltzmann integral, no $\langle\sigma v\rangle$, no $x_f$, no $g_\star$, no entropy dilution. The factor 4 multiplying *only* the observable weights is the sole free knob: $n=3\to0.327$, $n=5\to0.225$, $n=6\to0.195$; only $n=4$ hits 0.267, with no dynamical justification.
- **$\Omega_{\rm DM}h^2\approx0.12$ — POSTDICTION (major):** rides on the tuned fraction times an externally imported $h\approx0.67$.
- **$\sigma_8=\sqrt{11/16}\approx0.829$ — ASSERTED (major), tension:** $\sim$3$\sigma$ above Planck 0.811(6); the table downplays it as "~2%." A higher $\sigma_8$ *worsens* the $S_8$ tension. All structure-growth physics absorbed into "an unspecified normalization."
- **Shadow freeze-out $k_{\rm FO}\approx3.4$ TeV — ASSERTED (FATAL), tension:** a mirror sector in equilibrium with the SM down to $\sim$TeV contributes a large $\Delta N_{\rm eff}$ at BBN/CMB. No $\Delta N_{\rm eff}$, temperature-ratio, or entropy-dilution calculation is provided. "Quotient-non-observability" does NOT exempt gravitating relativistic shadow modes from the Friedmann equation. Planck $N_{\rm eff}=2.99(17)$ and BBN $\Delta N_{\rm eff}\lesssim0.3$ are unaddressed — a potential hard exclusion.
- **Sound elements:** "no new species" (structural restatement) and gravity-only direct-detection evasion are self-consistent — but by construction unfalsifiable except gravitationally.

**Status:** Qualitative framing coherent; quantitative $\Omega_{\rm DM}$ and $\sigma_8$ are fit/postdiction with a free knob and imported $h$; the BBN $N_{\rm eff}$ constraint is left dangling and potentially fatal.

---

## 10. Hubble Tension / Dark Energy (Sec. 17, lines 7340–7772) — *broken*

- **$H_0$ split $1+1/b_0=13/12$ — ASSERTED (major), agrees numerically:** $8.33\%$ vs observed $8.31\%$ is near-exact, but **not derived** — Thm thm:hubble-local Step 2 (lines 7492–7502) merely *states* $\delta_{\rm ledger}=1/b_0$. A dimensionless gauge $\beta$-coefficient is equated by verbal analogy ("cosmological avatar") to a *ratio of expansion rates* — physically and dimensionally unrelated. Reverse-engineering signature: 12 chosen because $1/12$ hits the known 8.3%; back-solving the latest JWST local values needs $b_0=12.65$–12.96.
- **$H_0^{\rm CMB}=67.4$ — ASSERTED (FATAL):** Cor. cor:h0-numeric literally says "Insert $H_0^{\rm CMB}=67.4$ (Planck LCDM central, PDG)." One of the two "predicted" numbers is a measured input in disguise; $H_0^{\rm loc}=67.4\times13/12=73.0$ is then input × chosen ratio.
- **Broken derivation chain (FATAL):** the supposedly first-principles Lemma formula $H_{\rm loc}\propto(w_1/\sqrt{\kappa_\star})(v/M_{\rm GUT})H_{\rm GUT}$ is **never evaluated**; with stated inputs it misses 73 km/s/Mpc by $\sim$39 orders of magnitude and has no $z=0$ meaning. The theorem silently abandons it — two mutually inconsistent $H_{\rm loc}$ definitions.
- **$w(z)\approx-1$, no drift — ASSERTED (major), tension trending to exclusion:** DESI DR1 gave $w_0=-0.827(63)$ ($\sim$2$\sigma$ from $-1$); DESI DR2 (2025) prefers evolving DE ($w_0>-1$, $w_a<0$, phantom crossing) over $\Lambda$CDM at **3.1–4.2$\sigma$**, breaching the paper's own falsifier $|w+1|>0.05$ at $z=0$. The single-anchor explanation is qualitative and cannot generate the coherent $w_a<0$ trend.

**Status:** The tension is *re-labeled* as a bookkeeping effect and *fit* to one observed number, not resolved by a derived prediction; the $w(z)=-1$ claim is trending toward observational exclusion.

---

## 11. Nuclear / Atomic / BBN / CMB (Sec. 18, lines 7772–8403) — *broken*

Almost entirely standard low-energy physics (textbook QCD running, hydrogen/Rydberg, BBN scaling, blackbody $T_{\rm CMB}$, $\Omega_b$–$\eta_B$ identity) relabeled, with SJET supplying only imported inputs ($\alpha_s,\alpha_{\rm em},m_e$ from PDG; $\eta_B$ claimed elsewhere).

- **Deuteron binding — ASSERTED (FATAL):** eq:deuteron-binding evaluates to **0.26 MeV, not the claimed 2.2 MeV** (off $\sim$8.5×); the right answer is hand-inserted while the displayed equation contradicts it.
- **$^7$Li/H — POSTDICTION (FATAL), excluded:** SJET quotes $1.6\times10^{-10}$ — the *observed* anomalously low stellar value — and falsely calls it "consistent with CMB+BBN joint fits," which actually *predict* $\sim5\times10^{-10}$. The paper quotes the wrong side of the unsolved lithium problem with no resolving mechanism.
- **Binding peak — FIT (major), tension:** 11 MeV vs real 8.8 MeV ($\sim$30% off, outside the stated 20% tolerance); no nuclear many-body physics.
- **Magic numbers — ASSERTED (major), tension:** reproduces only $\{2,8,20,28\}$, misses 50, 82, 126 entirely; $8\to\alpha$ is wrong ($\alpha$ is $A=4$).
- **$n_s$ — POSTDICTION (major), tension:** one-loop gives 0.970, $\sim$1.3$\sigma$ high; "2-loop refinements lower $n_s$ to 0.965" is asserted with no calculation — a free knob invoked after the one-loop value missed.
- **Mis-applied formulae:** fine-structure $1/[\ell(\ell+1)]$ diverges at the $\ell=0$ level it is evaluated on; $T_{\rm CMB}$ and $\Omega_b$ derivations are circular.
- **Correct-but-trivial (none/minor):** $E_1=-13.6$ eV, 21 cm, $m_p$, $Y_p$, D/H, $\Omega_b h^2$ — require no SJET content.

**Status:** A handful of correct textbook numbers wrapped in numerological capacity-weight dressings, with at least two numbers demonstrably erroneous against their own equations and against data.

---

## 12. Proton Decay (Sec. 19, lines 8403–8706) — *broken*

- **$\tau(p\to e^+\pi^0)=3.5\times10^{35}$ yr — POSTDICTION (FATAL), not reproducible:** eq:proton-rate with the quoted $K_6=3.5\times10^{-35}$, $M_{\rm GUT}=4.18\times10^{16}$, $\alpha=1/24$, $S_\Pi=0.91$ gives $1.0\times10^{35}$ yr (off 3.4×). A genuine dim-6 SU(5) calculation at the paper's *own* $M_{\rm GUT}$ gives $\sim10^{36}$–$10^{39}$ yr (standard formula $\sim4.5\times10^{36}$), 1–4 decades *longer*. The value sits exactly in the Hyper-K discovery window — the signature of tuning to the discoverable band.
- **$S_\Pi=0.91$ — ASSERTED (FATAL), fabricated:** its own definition $(16/27)(5/6)=0.494$, not 0.91. The table lists the two factors whose product is 0.494.
- **Conservative-bound recovery — ASSERTED (major), internally contradictory:** $S_\Pi\to1$ should change $\tau$ by $1/0.91^2=1.21\times$ ($8.5\times10^{34}\to1.0\times10^{35}$), yet the proof claims $3.5\times10^{35}$ — a factor 4.1 no $1/S_\Pi^2$ can produce.
- **$K_6$ — FIT (major):** the proof states it is "fixed by demanding" eq:proton-rate reproduce a pre-chosen estimate — a fit dressed as a constant; $A_{\rm had}=$"O(1)" is a hidden knob.

**Status:** A reverse-engineered number whose own arithmetic does not close (multiple errors that partially cancel), tuned to the Hyper-K discovery window rather than derived. Above the current Super-K bound ($2.4\times10^{34}$ yr) but not robust.

---

## 13. Quantum Mechanics from Observer Collapse (Sec. 20, lines 8706–8989) — *salvageable*

A re-description of standard quantum theory in SJET vocabulary, adding essentially no testable content.

- **OS reconstruction — ASSERTED (major):** the Osterwalder-Schrader theorem is correctly cited, but its hypotheses (reflection positivity, Euclidean invariance, clustering) are *postulated* as ledger entry A7, not proved for the interacting measure. Valid as a *conditional* ("IF OS axioms, THEN QFT"), but it imports QFT's analytic backbone via A7 — it does not derive QM from void+mirror.
- **Page-Wootters time — ASSERTED (major):** a misappropriation of the name. The genuine PW mechanism (global constraint $\hat H|\Psi\rangle=0$, clock subsystem, conditional states) is never constructed; what exists is ordinary Euclidean OS time relabeled "relational."
- **Born rule — ASSERTED (FATAL), circular:** GNS/OS *automatically* yields a positive state functional (positivity is the assumed RP input); calling its diagonal elements "probabilities" is *asserting* Born, not deriving it. The substantive content (the squared-modulus measure, additivity, Gleason/non-contextuality) is never touched. This is a fatal gap for the central "no probability postulate" claim.
- **Measurement/collapse — ASSERTED (major):** conflates the fixed kinematic superselection projector $\Pi_{\sigma=-1}$ with the outcome-dependent Lüders projector; selecting the observable sector once does not explain run-by-run stochastic collapse, definite outcomes, or the preferred basis.
- **Mirror-partner null — ASSERTED (minor), agrees trivially:** the only falsifiable item, but it is true by construction ($\sigma$-anti-invariance) and indistinguishable from ordinary SM chirality — adds nothing new.

**Status:** Mathematically unobjectionable *as an honest conditional*, but then it derives and predicts nothing new; the headline "QM derived" is overstated, with one fatal circularity (Born rule) and two major gaps.

---

## Cross-Cutting Findings

**The "parameter-free" claim is false.** The same small integer set $\{w_i, n_S=28, n_F=3, b_0=12, \kappa_\star\}$ is recycled across *every* sector as an all-purpose fitting kit. The exponent $n_S n_F/b_0=7$ alone is reused to fit $m_\mu/m_e$, the neutrino ratio, the muon $g-2$ sector factor, and the EW sector factor — a flagrant look-elsewhere/over-flexibility problem. The honest free-parameter-to-observable ratio is of order **1:1**, not the advertised 0:many. Tuned/circular inputs include: the rep$\to$generation map, the $\sqrt{w}$-vs-$w$ choice, $M_{\rm GUT}$ scanned over $[10^{16},5\times10^{16}]$ GeV, three flavon thresholds, and the IR matching scale ($\mu=M_Z$ vs $\mu_{\rm osc}$).

**Several "exact" rows compare the theory to itself or to spurious "experiments":** $b_0=12$ is not the SM beta coefficient (the SM values are $(41/10,-19/6,-7)$); $n_S=28$ is not an EWSB scalar count (the SM has one doublet); $\sin^2\theta_W=3/8$ and $v=246$ GeV are textbook identities or are derived circularly from $M_Z$ and $\sin^2\theta_W$.

---

## Claims That Would FALSIFY the Theory (and current status)

| Claim | Falsifier | Current status |
|---|---|---|
| Muon $\Delta a_\mu \approx 2.6\times10^{-9}$ | SM-confirming $g-2$ result | **ALREADY EXCLUDED** ($\sim$3–4$\sigma$) by 2025 WP25 lattice + Fermilab final ($\Delta a_\mu=3.8(7)\times10^{-10}$, the paper's own cited reference) |
| Dark energy $w(z)=-1$, no drift ($\vert w+1\vert>0.05$ falsifies) | Evolving DE / phantom crossing | **IN CONFLICT** — DESI DR2 2025 prefers evolving DE at 3.1–4.2$\sigma$; $w_0=-0.827$ gives $\vert w+1\vert=0.17\gg0.05$ |
| $\Delta\rho=0.0034$ ($\Rightarrow T$) | Measured $\Delta\rho$ | **ALREADY EXCLUDED** — measured $\Delta\rho=0.00038$, $\sim$11$\sigma$ discrepancy (hidden by a percentage trick) |
| $m_c/m_s = 13.6$ | Lattice $m_c/m_s$ | **ALREADY EXCLUDED** — true value 11.77(3), $\sim$30$\sigma$ off; target was fabricated |
| Cabibbo $\lambda=0.249$–0.2265 | PDG $\lambda$ | **IN TENSION** — PDG2024 $\lambda=0.22497(70)$; SJET $\sim$2$\sigma$–10% high |
| Wolfenstein $A=0.790$ | PDG $A$ | **IN TENSION** — PDG2024 $A=0.839(11)$, $\sim$5$\sigma$ low |
| $\sigma_8=0.829$ | Planck $\sigma_8$ | **IN TENSION** — Planck 0.811(6), $\sim$3$\sigma$ high (worsens $S_8$) |
| $^7$Li/H $=1.6\times10^{-10}$ as a BBN+CMB fit | Standard BBN $\sim$5$\times10^{-10}$ | **CONTRADICTS standard BBN** — quotes the observed (anomalous) value as if it were the prediction |
| Shadow freeze-out at $\sim$3.4 TeV (equilibrated mirror sector) | $\Delta N_{\rm eff}$ at BBN/CMB | **POTENTIALLY EXCLUDED** — uncomputed $\Delta N_{\rm eff}$ vs Planck $N_{\rm eff}=2.99(17)$, BBN $\lesssim0.3$ |
| $H_0$ split $=1/b_0=8.33\%$ | Tighter local $H_0$ | **FRAGILE** — fits round 73.0 but JWST 72.6–72.73 needs $b_0\approx12.7$–13.0 |
| Normal neutrino ordering | JUNO 2027–28 / inverted ordering | Currently consistent ($\sim$2.5$\sigma$ weak preference); genuine but low-information |
| No 4th generation | LHC | Consistent (already known) |
| $\tau(p\to e^+\pi^0)\sim3.5\times10^{35}$ yr | Hyper-K null above $\sim$10$^{35}$ yr, or detection | Above Super-K bound, but number is internally inconsistent and tuned |
| Strong CP $\theta_{\rm obs}=0$ | Neutron EDM | Consistent ($\vert\theta\vert<10^{-10}$), but *assumed*, and unfalsifiable via the sequestered reservoir |

**Already in conflict with data:** muon $g-2$ ($\Delta a_\mu$), the $\rho$/$T$ parameter, $m_c/m_s$ (fabricated target), $^7$Li/H (wrong side of the lithium problem), and the dark-energy $w(z)=-1$ claim (DESI DR2). The dark-matter $\Delta N_{\rm eff}$ constraint is uncomputed and potentially fatal.

---

## Overall Phenomenology Recommendation

With the sole exceptions of (i) the qualitative outputs — three generations, normal ordering, no fourth generation, a generic GUT-scale proton-lifetime bound — and (ii) correctly transcribed textbook identities ($\sin^2\theta_W=3/8$, the SM Lagrangian forms, the Leveille $g-2$ integral, $SU(3)_c$ from the breaking chain), **none of the manuscript's quantitative "predictions" is a genuine parameter-free derivation.** They are postdictions, single-point fits with reverse-engineered exponents, or assertions — several with internal arithmetic that does not reproduce the quoted numbers (Planck mass off by $10^{16}$; deuteron binding off by 8.5×; $\eta_B$ off by 5.6×; $S_\Pi$ fabricated; proton lifetime off by 3.4×) and at least five contacts already excluded or in tension with current data. The "Tier-T1, zero-free-parameter" framing of Secs. 22–23 is not supported. I recommend **rejection in present form** on the phenomenology; any resubmission must (a) drop the parameter-free claim and honestly count the $\sim$1:1 parameter-to-observable ratio, (b) correct the demonstrable arithmetic errors, (c) compute (not assert) every RG factor, threshold, and overlap integral, (d) retire the muon $g-2$, $\rho/T$, $^7$Li, $m_c/m_s$, and $w(z)=-1$ contacts that conflict with current data, and (e) supply the missing $\Delta N_{\rm eff}$ calculation for the mirror sector.


# Part III — Parameter-counting audit

I have all the data I need in the two JSON blobs (the physics verdicts and the numbers map). This is an analysis/synthesis task — I'll do the parameter-counting audit directly from the provided material rather than re-reading the paper. Let me cross-tabulate the hidden knobs against the claimed observables.

# SJET Parameter-Counting Audit: "Two Primitives" vs. Reality

## Executive verdict

The "everything from two primitives (void + mirror)" claim is **not supported**. When you tally every quantity that is actually chosen — not derived from a stated computation — the construction uses on the order of **35–45 effectively free inputs** to reproduce roughly **20–25 quantitative observables**. The honest free-parameter-to-observable ratio is **≈ 1.5 : 1 to 2 : 1**, i.e. *worse* than the Standard Model (~19–26 parameters for a far larger and sharper body of predictions), and far worse than the advertised **0 : many**.

The knobs are disguised in three ways: (1) **recycled integers** (n_S=28, n_F=3, b₀=12, the 1:10:16 weights) reused in different algebraic combinations per observable; (2) **per-channel hand-picked exponents and "threshold/residue" factors R⁽²⁾** that are never computed from the functionals they are attributed to; and (3) **matching scales** (μ_osc → M_Z, Λ ≔ M_GUT, μ ≔ v) chosen to land the logs where the data is. Several "predictions" are also circular (M_Pl→M_GUT→Λ≔M_GUT→M_Pl; v derived from M_Z and sin²θ_W which themselves require v) or arithmetically broken (M_Pl off by 16 orders, deuteron off by ~8×, proton lifetime not reproducible from its own equation).

---

## 1. The genuinely non-tuned inputs (the honest core)

These are the only items that are real, non-reverse-engineered structure:

| Input | Value | Status |
|---|---|---|
| Division-algebra / Jordan / exceptional dims | 1,2,4,8,27,78,248,496 | genuine math |
| EIII coset dim, Goldstone split 32 = 28 + 4 | 28, 4 | genuine group theory |
| SO(10) sub-rep dims inside **27** = 1+10+16 | (1,10,16)/27 | genuine group theory |
| Anomaly cancellation 16−80+64 = 0 | 0 | genuine (but **16 is auto anomaly-free; no GS needed** — see §4) |
| sin²θ_W(M_GUT) = 3/8 | 3/8 | textbook SU(5)/SO(10) fact |
| SU(3)_c as unbroken color factor | — | textbook GUT cascade |
| One-loop b₀ bookkeeping (88−24−28)/3 = 12 | 12 | internally consistent arithmetic |

**Critical caveat:** the (1,10,16) partition is a real group-theory fact, but it indexes the **three SO(10) sub-reps inside ONE generation's 27**. The paper then **silently reuses these same three numbers as the three fermion generations (e/μ/τ)**. That reinterpretation (Table tab:yukawa-leading, Prediction pred:sector-ratios) is *asserted, not derived*, and it is the single load-bearing move behind essentially every mass-ratio number. So even the "honest core" is contaminated where it does predictive work.

---

## 2. The hidden knobs — concrete inventory

### A. Recycled "structural" integers used as multipliers across unrelated sectors (≈ 4 base integers, but each reused as if independent)

- **n_S = 28** — coset scalar count. Reused as: g-2 sector factor numerator, M_Pl induction prefactor (28/6), lepton-ratio exponent ingredient, EW sector factor, sphaleron "28/27", nuclear "magic number 28," BBN "two-body network." Compared to a spurious "EWSB scalar count" observable (SM uses 1 doublet).
- **n_F = 3** — generations. Legitimately an index claim, but reused as an arbitrary multiplier in S_μ = n_S·n_F/b₀, in oblique/EW prefactors, etc.
- **b₀ = 12** — gauge β-coefficient. **Illegitimately transplanted** into: lepton/neutrino RG exponents, g-2 (÷b₀), proton mass (b₀/4 = 3), CKM exponent 1/(2b₀)=1/24, **the Hubble split 1/b₀**, string tension, alpha_s residue. A dimensionless log-running coefficient is identified by verbal analogy with a *ratio of cosmic expansion rates* — physically/dimensionally unjustified.
- **κ_⋆ = 96π²/5 ≈ 189.6** — a truncation-dependent FRG number whose value follows *algebraically from an asserted* η_MPl = −(5/48π²)κ (the whole fixed point is an input dressed as a result). Dimensionless, so it fixes no absolute scale, yet appears in v, M_GUT, freeze-out scale.

### B. Per-channel hand-picked rational exponents (≈ 11, each chosen to hit a target)

The exponent **n_S·n_F/b₀ = 28·3/12 = 7** is the master knob — exactly the power (6.84) needed to lift tree m_μ/m_e = 10 to 206.8. "7 threshold crossings exponentiate the residue" is asserted, not an RG result (one-loop RG gives a *single* power). The same 7 is reused for g-2 (S_μ), EW (S_EW), neutrinos. Then a zoo of one-off exponents:

| Exponent | Where | Tell |
|---|---|---|
| **1/5** | Cabibbo λ | proof *admits* it is "the unique rational that maps λ_tree≈0.249 to PDG λ" |
| 2, 2, 3/2 | m_s/m_d, m_c/m_s, m_t/m_b | per-channel |
| 1/24 = 1/(2b₀) | Wolfenstein A | |
| 1/4, 1/8, 1/7, 3/4, 1/2 | ρ, η, V_ub | η stacks **four** |
| 1/6 | quark channel | |

The η (CP phase — the genuinely nontrivial CKM observable) formula carries **four** independent hand-chosen factors plus a stray 1.009 multiplier. With that many knobs, hitting 0.357 is trivial.

### C. Asserted "two-loop / threshold" factors R⁽²⁾ — never computed from their Ξ functionals (≈ 7)

R_sd=0.790, R_cs=0.906, R_tb=0.960, R_CKM=0.989, **R_ub=0.095** (a 10× suppression mislabeled "threshold"; real 2-loop is O(1±few%)), plus K_QED=1.017 (leptons) and R_EW⁽²⁾=0.962 (g-2 / EW). Each is exactly the multiplier needed to convert a tree ratio to the cited PDG value.

### D. Matching/IR scales chosen to position the logs (≈ 5)

- **μ_osc → M_Z** (neutrino ratio): a ~30-orders-of-magnitude substitution; the paper's own sensitivity table shows the honest oscillation scale gives the wrong answer (row 4: 22.4, excluded).
- **μ ≔ v** (Coleman-Weinberg matching) — imposed, not derived; propagates the fatal M_Pl error.
- **Λ ≔ M_GUT** (cutoff) — definition, not derivation.
- **M_GUT scanned ∈ [1×10¹⁶, 5×10¹⁶] GeV** — kept floating to hold the neutrino ratio on target ((5/2)⁴ ≈ 39× swing in proton lifetime alone).
- Three flavon thresholds **M_F,k = (10¹², 10⁹, M_Z)** inserted by hand into Ξ_th.

### E. Choices of structure presented as forced (≈ 4)

- **Which coset direction breaks** / "exactly one tachyon per ι-cell" — asserted (lem:breaking-cw) with no eigenvalue computation.
- **rep → generation map** (the 1:10:16 reinterpretation, §1).
- **y ∝ √w vs. m ∝ w** (Prop gen-ordering) — the *square* is asserted, conveniently giving the larger spread.
- The factor-**4** in Ω_DM = w₃/[4(w₁+w₂)+w₃]: n=3,5,6 give 0.327/0.225/0.195; only n=4 hits 0.267. Pure single-knob targeting.

### F. Externally inserted "predicted" quantities (laundered inputs, ≈ 6)

M_Pl = 1.22×10¹⁹ (inserted, then used to "derive" M_GUT), h ≈ 0.67 (for Ω_DM h²), H₀^CMB = 67.4 (multiplied by 13/12 to "predict" 73.0), Γ_Z⁽⁰⁾ = 2.496 (SM width hand-inserted), sin²θ_W(M_Z) = 0.231 (input to the v "derivation"), a_μ^SM baseline (sets the entire g-2 "match").

---

## 3. Knob ledger vs. observable ledger

**Effectively free inputs (counting recycling conservatively):**

| Category | Count |
|---|---|
| Recycled base integers (28, 3, 12, κ_⋆) used as independent multipliers | ~4 (but ~12 "uses") |
| Hand-picked exponents | ~11 |
| R⁽²⁾ / threshold / QED factors | ~7 |
| Matching/IR scales + scanned M_GUT + 3 flavon thresholds | ~5–6 |
| Structural "which-breaks" / map / √w-vs-w / factor-4 | ~4 |
| Laundered external inputs (M_Pl, h, H₀^CMB, Γ_Z⁰, sin²θ_W, a_μ^SM) | ~6 |
| **Total** | **≈ 35–45** |

**Quantitative observables claimed:** 3 charged-lepton ratios (only m_μ/m_e ever shown), 3 quark-mass ratios, 6 CKM/Wolfenstein, 3 PMNS angles, Δm²₃₁/Δm²₂₁, v, M_GUT, M_Pl, α_s, Λ_QCD, m_p/m_π/m_ρ, α_em⁻¹(M_Z), Δρ, M_W, S/T/U, Γ_Z, Δa_μ, Ω_DM, σ₈, H₀ split, η_B, n_s, Y_p, D/H, ⁷Li, τ_p — **≈ 20–25 numbers that are actually quantitative** (excluding qualitative items like "no 4th gen," "normal ordering," "θ=0," which are low-information binaries or asserted).

**Ratio: ≈ 35–45 knobs : ≈ 20–25 observables ≈ 1.5–2 : 1.**

---

## 4. Contrast with the Standard Model

| | Standard Model | SJET (as audited) |
|---|---|---|
| Free parameters | 19 (26 with ν masses + mixings + θ̄) | ≈ 35–45 effective |
| Observables predicted | Thousands of cross-sections, decay rates, precision EW to sub-per-mille, all consistent | ~20–25 numbers, **many fits/postdictions**, several excluded |
| Predictions per parameter | ≫ 1 (hugely overconstrained) | < 1 (underconstrained) |
| Free params disguised as fixed | None | Most of them |

The SM is *overconstrained*: 19 numbers lock down an enormous, sharply-tested edifice with no remaining freedom. SJET has **more adjustable inputs than outputs**, with key consistency tests (multiplicativity of mass ratios, τ/μ and τ/e, common-scale quark ratios) **omitted precisely where they would expose the reverse-engineering**.

---

## 5. Decisive failures that make even the inflated count generous

- **m_μ/m_e:** tree 10 vs 206.8; the two equational forms give 0.018 and 222 (sign of the RG log flipped between them); stated 201/203 reproducible from neither. **Violates multiplicativity:** (μ/e)×(τ/μ)=3242 vs direct τ/e=1589.
- **M_Pl (central claim of Sec 10):** the Sakharov formula with μ=v gives **~342 GeV, not 1.2×10¹⁹** — wrong by 16 orders of magnitude. M_Pl is inserted by hand; the M_Pl→M_GUT→Λ≔M_GUT→M_Pl triangle is **circular**.
- **g-2:** target Δa_μ ≈ 2.6×10⁻⁹ is the **obsolete 2020 anomaly**; 2025 lattice+Fermilab gives ~0.4×10⁻⁹ (consistent with zero). The S_μ=7 factor (β-coefficient normalizing a fixed-scale loop) is conceptually illegitimate. **Excluded.**
- **Δρ / T:** Δρ = 0.0034 vs measured 0.00038 (**~11σ**), hidden by quoting "%" on ρ≈1; T = Δρ violates the Peskin–Takeuchi α·T = Δρ (off by ~128×).
- **m_c/m_s:** cited "PDG 13.6" is **fabricated**; real lattice value 11.77, excluded by ~30σ.
- **Anomaly:** the **16 of SO(10) is automatically anomaly-free** — the 16−80+64=0 display and the invoked Green-Schwarz are numerology, not a needed cancellation.
- **η_B, deuteron, ⁷Li, Λ_QCD, proton lifetime:** each fails to reproduce its own stated equation (η_B off 5.6×; deuteron off ~8.5×; Λ_QCD off 4 orders; τ_p off 3.4× and S_Π fabricated as 0.91 vs its own definition 0.494).

---

## Conclusion

The construction does **not** derive physics from two primitives. It derives a *scaffold* of correct group theory (division algebras, the E6/SO(10)/SU(5) cascade, 3/8, b₀=12, the 1+10+16 decomposition), then **fits ~20–25 observables using ~35–45 tunable handles** — recycled integers in bespoke combinations, ~11 hand-picked exponents (one of which the proof openly admits is chosen to hit PDG λ), ~7 uncomputed R⁽²⁾ factors, ~5 matching scales chosen to place the logs, and ~6 measured quantities laundered as inputs. The effective parameter-to-observable ratio is **worse than the Standard Model's**, the "0 free parameters" / "Tier T1 parameter-free" claim is false, and several flagship numbers are either circular, arithmetically broken by many orders of magnitude, or matched to obsolete/fabricated experimental targets.

**The "theory of everything from void + mirror" claim is unsupported: it is a postdiction engine with more knobs than predictions, not a derivation.**


---

## Appendix A. Quantitative contacts already in tension with or excluded by current data

Generated from the 14 phenomenology validators (live checks vs PDG 2024, Planck 2018, DESI DR2 2025, Muon g-2 WP25/Fermilab-final, FLAG 2024, Super-K). Status: **fit**=tuned to data, **postdiction**=matched to known value, **asserted**=no derivation.

| Quantity | SJET value | Current data | Verdict | Sev |
|---|---|---|---|---|
| Ghost sector | dim-6 + spurious U(1) | — | asserted/tension | major |
| Gauge-fixing term | absent | — | asserted/tension | major |
| Right-handed neutrinos/nu mass | omitted | — | asserted/tension | major |
| Tree-level charged-fermion mass hierarchy m1:m2:m3 = w1:w2:w3 = 1:10:1 | 1 : 10 : 16 (so m_mu/m_e = 10, m_tau/m_mu  | m_mu/m_e = 206.77; m_tau/m_mu = 16.82; m_tau/m | asserted/excluded | fatal |
| Charged-lepton ratio m_mu/m_e at M_Z (one-loop flavon RG) | approx 201 (claimed); formula as written a | m_mu/m_e = 206.768 (pole); ~207 (MSbar at M_Z) | fit/tension | fatal |
| Charged-lepton ratio internal consistency (tau/mu and multiplicativity | m_tau/m_mu approx 14.6; (m_mu/m_e)x(m_tau/ | m_tau/m_mu = 16.82; m_tau/m_e = 3477 | asserted/excluded | fatal |
| Cabibbo angle lambda_CKM (and PMNS sin theta_12) | lambda = (2/11) H12 approx 0.249 | lambda = 0.2245 (PDG) | fit/tension | major |
| PMNS angles sin theta_23, sin theta_13 | sin theta_23 approx 0.78, sin theta_13 app | sin theta_23 approx 0.75 (theta23~49 deg); sin | fit/tension | minor |
| Induced Planck mass M_Pl (Eq. eq:planck-explicit / eq:planck-v, Prop p | M_Pl^2 = (28/6)*(mu^2/16pi^2)*ln(Lambda^2/ | M_Pl = 1.22e19 GeV (reduced 2.435e18 GeV), COD | asserted/excluded | fatal |
| Muon anomalous moment a_mu / Delta a_mu (Rmk rmk:g2-scales) | Delta a_mu^SJET ~= 2.7e-9 'matching' delta | Fermilab 2025 final world avg a_mu(exp) = 1165 | postdiction/excluded | fatal |
| Scalar mass m_phi = v = 246 GeV (Coleman-Weinberg/mu=v identification) | m_phi = mu = v ~ 246 GeV | No 28 new ~246 GeV scalars coupling to muons a | asserted/tension | major |
| Delta a_mu^SJET (1-loop) = 7/(48pi^2) (m_mu/v)^2 | 2.7e-9 (I reproduce 2.728e-9) | Old (WP2020) anomaly was ~2.5e-9 (4-5 sigma);  | postdiction/excluded | fatal |
| delta a_mu (experimental anomaly) used as target, Eq. (eq:delta-amu) | Paper uses 2.6e-9 with a_mu^SM=1.16591810e | 2025: a_mu^exp=1.165920705(15)e-3 (Fermilab fi | asserted/excluded | fatal |
| m_c/m_s (mu_c) | 13.6 (eq:quark-ratio-values; cited 'PDG202 | ~11.7-11.85 (FLAG2024 Nf=2+1+1: m_c/m_s=11.766 | fit/excluded | fatal |
| m_t/m_b (mu_b=M_Z) | 41.3 (eq:quark-ratio-values) | scheme-dependent: ~41 (m_t pole / m_b(m_b) MSb | fit/tension | major |
| lambda = /V_us/ (Cabibbo) | 0.2265 (eq:ckm-wolfenstein-values, eq:work | PDG2024: lambda = 0.22497(70); Vus~0.2243 | fit/tension | fatal |
| A (Wolfenstein) | 0.790 (eq:ckm-wolfenstein-values) | PDG2024: A = 0.839(11) (rhobar/etabar prescrip | fit/tension | major |
| rho (Wolfenstein, =rhobar) | 0.141 (eq:ckm-wolfenstein-values) | PDG2024: rhobar = 0.1581(92) | fit/tension | major |
| /V_ub/ | 0.00350 (via A*lambda^3*sqrt(rho^2+eta^2)) | PDG2024: /V_ub/ ~ 0.00382(20); incl/excl tensi | fit/tension | major |
| String tension sigma_str (eq:string-tension) | approx 0.13 GeV^2 (table); text claims 0.1 | lattice ~0.18-0.19 GeV^2 (sqrt(sigma)~440 MeV) | asserted/tension | major |
| alpha_s(M_Z) (Prop prop:alpha-s, eq:alpha-s-mz) | approx 0.14 (claimed); the cited formula a | 0.1180(9) (PDG 2024 world average) | fit/excluded | fatal |
| Lambda_QCD (eq:Lambda-QCD) | approx 0.28 GeV (claimed) | Lambda_MSbar^(3) ~ 0.33-0.34 GeV (PDG/FLAG 202 | asserted/excluded | fatal |
| Proton mass m_p (eq:mp-pred) | (b0/4)*Lambda = 3*0.28 = 0.84 GeV | 0.938 GeV | fit/tension | major |
| Rho meson mass m_rho (eq:mrho-pred) | 2*m_pi*sqrt(w2/w1) = 2*0.14*sqrt(10) = 0.8 | 0.775 GeV | fit/tension | major |
| alpha_em^{-1}(M_Z) | 127.4 (Eq. 6201/6464; claimed 24 + 63.0 +  | 127.951(9) on-shell / 128.946(11) MS-bar (PDG  | fit/tension | major |
| rho parameter / Delta_rho | rho = 1.0034, Delta_rho = 0.0034 (Eq. 6234 | rho = 1.00038(27); Delta_rho ~ 0.00038 (PDG gl | postdiction/excluded | fatal |
| kappa_sph (sphaleron conversion efficiency) | 28/27 ~ 1.037 (Thm thm:baryogenesis, 6795) | Standard sphaleron L->B factor = 28/79 ~ 0.354 | asserted/tension | fatal |
| eta_B = n_B/n_gamma (baryon-to-photon ratio) | claimed ~6.1e-10 (eq eta-b-numeric 6841, P | (6.10 +/- 0.04)e-10 (Planck 2018 / PDG 2024, v | postdiction/tension | fatal |
| sigma_8 (rms fluctuation amplitude) | sqrt(11/16) ≈ 0.829 (eq:sigma8) | 0.811 ± 0.006 (Planck 2018) | asserted/tension | major |
| Shadow freeze-out scale k_FO | kappa_star^{1/2} v ≈ 3.4 TeV (eq:dm-freeze | n/a directly; constrained via N_eff | asserted/tension | fatal |
| w(z) equation of state (sequestered CC) | approx -1 +/- 0.05 (constant; phantom-free | DESI DR1 (2024) w0=-0.827 +/- 0.063, wa=-0.75  | asserted/tension | major |
| E_B/A binding peak (iron) | ~11 MeV (Eq. eq:binding-peak) | ~8.8 MeV/nucleon (Fe-56/Ni-62 peak) | fit/tension | major |
| Deuteron binding energy E_B(d) | ~2.2 MeV (Eq. eq:deuteron-binding) | 2.224 MeV | asserted/excluded | fatal |
| Magic numbers / shell closures A*={2,8,20,28} | {2,8,20,28} from n_S=28 coset (Eq. eq:magi | Nuclear magic numbers: 2,8,20,28,50,82,126 | asserted/tension | major |
| Lithium-7 abundance Li-7/H | ~1.6e-10 (Eq. li7, bbn-numeric), called 'c | BBN+CMB PREDICTION ~5e-10 (4.94(72)e-10); OBSE | postdiction/excluded | fatal |
| Scalar spectral index n_s | ~0.974 (one-loop), 0.965 (claimed 2-loop)  | 0.9649(42) (Planck 2018) | postdiction/tension | major |
| Capacity weights 1:10:16 | 1:10:16 (T1 derived; drives generation ord | Observed third/first hierarchies are O(10^3-10 | fit/tension | major |
| Muon Delta a_mu (new-physics contribution) | 2.58-2.7 x 10^-9 (T1 derived; (n_S n_F/b_0 | WP25 2025: a_mu^SM=(116592033 +/- 62)e-11, exp | fit/excluded | fatal |
| Lambda_obs / dark-energy equation of state w | Lambda_obs ~ 0; w -> -1 with NO drift (CC  | DESI DR2 2025: 2.8-4.2 sigma preference for EV | asserted/tension | major |
| CKM lambda (Cabibbo) and PMNS sin theta_23 | lambda ~ 0.249, sin theta_23 ~ 0.78 (D1, T | lambda = 0.22500(67) (PDG); sin theta_23 ~ 0.7 | fit/tension | major |
| Overall free-parameter vs observable count | Claimed ~0 free parameters ('no longer a f | Actual adjustable inputs: capacity-weight assi | asserted/tension | fatal |

---

# Appendix B — Editor's structural recommendation

Do not attempt to submit this 108-page monograph to PRD; PRD does not publish 300-theorem treatises and the current draft would be desk-rejected on format alone, independent of the scientific problems. Split and drastically cut. (1) Strongest standalone candidate: a short, honest mathematics note (PRD-length or a math-physics journal) on the exceptional scaffold itself, restricted to what is provably true and externally verifiable. The two cleanest results are (a) the capacity/branching bookkeeping that reproduces 27 = 1 + 10 + 16 and the EIII 32 = 28 + 4 split presented as representation theory, NOT as a dynamical "partition theorem," and (b) the GUT-trace derivation sin^2(theta_W) = 3/8 with the b_0 = 12 one-loop accounting, presented as standard Spin(10)/SU(5) phenomenology of the assumed embedding. These carry no false claims. (2) A separate, clearly speculative "framework/conjecture" paper (better suited to a foundations venue than PRD) may sketch the void-mirror philosophy, but only after relabeling every "derived"/"closed"/"Tier-T1" claim as conjectural and removing the numerology. (3) Cut entirely, or move to an explicitly conjectural appendix flagged as unproven: the mirror operator M (the M^2=id identity fails for Cayley-Dickson), the observer fixed-point family count (the H^1 cohomology arithmetic is incoherent and sigma cannot be simultaneously order 2 and order 6), the soldering/spacetime sector (H_2(H) is 6-dimensional, not 4), the OS/reflection-positivity construction (assumes its own axioms via ledger entry A7), and every quantitative flavor/cosmology fit whose exponents are reverse-engineered. (4) Before any resubmission: fix the demonstrable group theory (16 x 16 = 10 + 120 + 126, the correct 126 branching, H_2(H) dimension/signature), recompute the cellular cohomology with eigenspace dimensions that sum to 32, discharge or openly label ledger assumptions A3/A6/A7, retire the contacts already excluded by data (muon g-2, Delta-rho/T, m_c/m_s, 7Li, w(z) = -1), and replace "parameter-free" with an honest ~1.5:1 parameter-to-observable count. The honest residue is a modest but real contribution; the monograph as a unified "theory of everything" is not.
