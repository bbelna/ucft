# JSET Changelog

## v1.0 (2026-06)

**Major restructuring to honest theorem/conjecture ledger (implements validation recommendations).**

- Primary deliverable: `sjet_true.tex` — self-contained publication-style document containing:
  - Rigorous exceptional Jordan/E₆ core (Standard Imported Theorems only: dims, EIII geometry, 27 branching, tangent representation).
  - Typed mirror scaffold (no claim of a single uniform involutive functor M).
  - Capacity layer with full convex analysis (gradient, Hessian, maximizer argument) showing that the 1|10|16 weights are not derived by the functional unless data are supplied independently.
  - Family/observer geometry recast as the named conjecture **FAM** with precise data requirements (Q, σ of order exactly 2, equivariant V₁₆, explicit cohomology computation).
  - Corrected spacetime soldering: full H₂(𝕂) dimension + signature table; d=4 Lorentzian requires explicit selection of ℂ ⊂ 𝕆 (Assumption A-Solder-C).
  - Conditional SM/GUT, QFT (OS), and gravity/RG layers with all assumptions named and all "derives X" claims removed or qualified.
  - Six precisely stated open problems (the JSET v2 roadmap).
  - Global summary theorems: "What JSET_true is" and "What JSET_true is not."
- Repo layout: `historical/` created with pre-v1 tarball + README (Option A archiving per user decision). Legacy `.sjet-build/` labeled as such.
- Supporting artifacts: `JSET_v1_onepager.tex` (short extract for quick sharing / foundations-note cover), `build_clean.sh` (primary v1 build), updated `README.md` with JSET v1 branding and clear file table, `CHANGELOG.md`.
- Language hygiene: All non-historical public documents now use the explicit classification system (Theorem / SIT / Conjecture / Assumption / CEC) and avoid unsupported derivation claims.
- User decisions incorporated (see plan): keep `sjet_true.tex` filename + JSET v1 branding; keep + language-sync `sjet-prd.tex`; Option A historical archive; produce one-pager; foundations/math-physics note target; open problems strictly post-v1.

**Validation alignment**: Directly addresses the 2026-06-06 39-agent report's call for an honest mathematics note on the scaffold + a clearly conjectural framework with all numerology, circularities, and dimensional errors corrected or labeled.

**Not in v1**: Resolution of any of the six open problems; numerical phenomenology or "Tier C" contacts (those are deferred until assumptions are discharged and belong in a future conditional-EFT paper).

See `sjet_true.tex` (especially the Open Problems section and the global ledger at the end) and `SJET-VALIDATION-REPORT.md` for the full picture.

All earlier commits (pre-v1) are preserved in git history under the `prresearch-submission` branch.
