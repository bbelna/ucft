# JSET v1 — Open Problems (the v2 Roadmap)

These six problems are extracted verbatim from the canonical ledger in `sjet_true.tex` (Section 10). They are the precise mathematical tasks required before stronger physical claims can be made. Each is stated with sufficient rigor that progress or refutation is checkable.

Success on any item can be measured by whether the corresponding assumption or conjecture in the ledger can be promoted to a theorem (or explicitly retained as an assumption with justification).

## 1. Unified mirror category

**Problem.** Construct (or prove the non-existence of) a single category $\mathcal{C}$ and involutive functor $\Mirror$ that recovers the dimensions and algebraic structures of the typed operations (M_Bool, M_CD, M_Jordan, M_FT, M_prod, M_EIII) uniformly, with $\Mirror^2 \simeq \id$ holding on all sectors where the $\pm 1$ eigenspace decomposition (and thus $\Pi_{\sigma=-1}$) is later invoked.

**Why it matters.** The current typed list is an honest description; the original claim of a single primitive mirror "forcing the climb" does not hold. A uniform functor would restore some of the original generative elegance while remaining rigorous.

**Minimal viable first step.** Define a candidate category whose objects are at minimum the structures appearing in the typed operations (Boolean algebras/Stone spaces, normed division algebras with conjugation, the Albert algebra with its automorphism group, and the relevant Lie groups with product structure). Exhibit functors or natural transformations that compose to the observed steps and check the involution property where needed for projectors.

**Success metric.** A fully specified category and functor (or a proof that no such single functor exists with the required properties) together with a clear statement of which (if any) of the original "derivation" steps become theorems under it.

## 2. Independent capacity data

**Problem.** Specify a concrete graph $G_{\mathrm{Albert}}$, a measure $\mu$, and cost functions $c_i(a)$ whose only input is the exceptional algebra structure (or the typed mirror operations) and that are independent of the target weights $(1/27, 10/27, 16/27)$. Compute the resulting maximizer of $\capacity$ or the Legendre form $\mathcal{L}_m$ and determine whether the 1|10|16 partition emerges.

**Why it matters.** The current capacity functional (with its gradient/Hessian analysis) selects the uniform distribution on the hyperplane absent further structure. The specific weights are the irrep dimensions of the $E_6$ branching. Without independent data, capacity remains bookkeeping.

**Minimal viable first step.** Propose an explicit finite graph whose vertices and edges are derived from the Albert algebra (e.g., the 27 coordinates or the EIII coset directions) and the typed operations. Assign costs $c_i$ from representation-theoretic or norm data that do not presuppose the 1|10|16 split. Run the variational problem.

**Success metric.** Either (a) an explicit data set independent of the target that yields exactly (or approximately, with controlled error) the weights 1:10:16, or (b) a clear demonstration that no such natural data exist and the weights must be treated as external input.

## 3. Explicit family geometry

**Problem.** Construct an explicit triple $(Q, \sigma, V_{16})$ satisfying the data requirements of Definition (family count) in `sjet_true.tex` (compact complex or algebraic space or suitable stratified/profinite space with well-behaved cohomology; genuine involution $\sigma$ with $\sigma^2 = \id$; $\sigma$-equivariant holomorphic vector bundle $V_{16}$ of appropriate rank) and compute $\dim H^1(Q, V_{16})^{-}$ rigorously. Only after this computation may a claim of "three families" (or any specific integer) be reinstated as a theorem rather than the conjecture **FAM**.

**Why it matters.** The prior cellular count was arithmetically incoherent, the involution had inconsistent order, and the Lefschetz application was invalid (plus no fixed points implying index 0). A correct, explicit computation is required.

**Minimal viable first step.** Begin with a finite or algebraic model (e.g., a suitable curve, surface, or finite cell complex with a rank-16 bundle and an order-2 involution that exchanges or acts on stalks in a manner compatible with the 16/16bar splitting). Compute the anti-invariant cohomology by hand, with Macaulay2/Sage, or via spectral sequences before attempting the profinite Stone lift.

**Success metric.** A fully specified triple together with a published or checkable computation of the anti-invariant dimension (the integer may be 3, or it may be another value; either outcome is progress).

## 4. Complex soldering selection

**Problem.** Exhibit the mechanism (or explicit additional postulate) that selects a complex subalgebra $\mathbb{C} \subset \mathbb{O}$ (or an equivalent 4-plane inside an off-diagonal Jordan block) such that the induced determinant form has signature $(1,3)$, and prove that this selection is compatible with the rest of the typed scaffold, any capacity/observer structures, and the overall ledger.

**Why it matters.** The corrected $H_2(\mathbb{K})$ table shows that four-dimensional Lorentzian geometry comes from the complex case, not the quaternionic one. The selection is extra structure.

**Minimal viable first step.** Propose a natural (or capacity-selected) subalgebra embedding or 4-plane inside $J_3(\mathbb{O})$ that is preserved or selected by one or more of the typed mirror operations. Compute the restricted quadratic form and verify the signature and compatibility.

**Success metric.** An explicit selection rule (with proof of signature and compatibility) or a clear statement that the selection must remain an independent geometric assumption.

## 5. EFT action terms and RG coefficients

**Problem.** Starting from the exceptional core plus the geometric assumptions (FAM, soldering selection, etc.), derive (or justify as the unique capacity-selected choice) the specific operators that appear in $S_{\mathrm{GUT/SM}}$ and $S_{\mathrm{EH}}$ and the numerical values of the one-loop coefficients (including the $5/(48\pi^2)$ that enters the gravitational beta function). State all truncations and regulators explicitly.

**Why it matters.** The algebraic fixed-point results are correct once the beta-function coefficients are granted; the coefficients themselves are the dynamical content that must be justified from the scaffold.

**Minimal viable first step.** In a controlled truncation (e.g., the EIII coset scalars + gauge fields projected under the observer collapse), write the most general capacity-compatible effective action up to a given order and compare its RG flow (one-loop or FRG) against the required coefficients. Identify which terms are forced vs. selected.

**Success metric.** Either an explicit derivation of the target operators and coefficients from the core + assumptions, or a precise statement of the additional selection principles required.

## 6. Testable constraints beyond standard E6 GUTs

**Problem.** Determine whether the framework, once the above items are addressed, imposes any quantitative relation among Standard-Model or cosmological observables that is not already present in ordinary $E_6$ or $\Spin(10)$ GUT model building with the same number of free parameters. If no such relation exists, state this clearly.

**Why it matters.** The validation audit showed a parameter-to-observable ratio worse than the SM and multiple postdictions/excluded contacts. v2 must demonstrate added value or honesty about the lack thereof.

**Minimal viable first step.** After any of 1--5 are advanced, enumerate the remaining free inputs in the conditional EFT layer and compare the predicted relations (masses, mixings, couplings, cosmological anchors) against the standard GUT literature with equivalent assumptions.

**Success metric.** Either (a) at least one new, falsifiable quantitative relation with tolerance, or (b) an explicit statement that the framework organizes known GUT structure but does not yet yield predictions beyond what is already achievable with the same number of inputs.

## Usage

These problems are the living v2 roadmap. When one is solved (or refuted), update this file and the corresponding section of `sjet_true.tex`, promote the relevant assumption/conjecture in the ledger, and note the change in `CHANGELOG.md`.

Cross-reference with the named assumptions in `sjet_true.tex` (A-Cap-Book / A-Cap-Legendre, A-Solder-C, A-GUT-E6, A-OS, etc.) and the global summary theorems ("What SJET_true is" and "What SJET_true is not").

For the current (v1) state, see `sjet_true.tex` (especially Sections 2, 5--7, and 10) and `SJET-VALIDATION-REPORT.md`.
