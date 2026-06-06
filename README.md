# SJET — Stone–Jordan Exceptional Theory (Cleaned / True Version)

**Status (June 2026):** This repository now hosts a mathematically defensible version of the theory.

The primary deliverable for **JSET v1** is the cleaned, honest theorem/conjecture ledger:

- `sjet_true.tex` / `sjet_true.pdf` (when built) — canonical mathematical physics document (rigorous exceptional Jordan/E₆ scaffold + named conjectures/assumptions + conditional EFT layers). See the "JSET v1" branding and status throughout.

This implements the "honest mathematics note + explicitly conjectural framework" restructuring recommended by the June 2026 validation.

The older `sjet.tex` / `sjet.pdf` and the full programme archive in `old/` are retained for historical reference only. They contain claims that have been shown to be circular, dimensionally false, or otherwise unsupported.

## Core Thesis of SJET$_{\mathrm{true}}$

\[
\begin{aligned}
\text{SJET}_{\mathrm{true}}
&=
\underbrace{
\bigl(J_3(\mathbb{O}),\, E_6,\, \Spin(10)\cdot U(1),\, \mathbf{27}=\mathbf{1}\oplus\mathbf{10}\oplus\mathbf{16}\bigr)
}_{\text{rigorous exceptional algebra core (standard imported theorems)}}
\\
&\quad +
\underbrace{
\bigl(\mathcal{C},\, G_{\mathrm{Albert}},\,\mu,\, c_i\bigr)
}_{\text{capacity model (bookkeeping or Legendre form; weights not derived unless data independent)}}
\\
&\quad +
\underbrace{
(Q,\,\sigma,\, V_{16}),\quad \dim H^1(Q,V_{16})^{-}=3
}_{\text{family/observer conjecture (named \textbf{FAM})}}
\\
&\quad +
\underbrace{
H_2(\mathbb{C})\hookrightarrow J_3(\mathbb{O})
}_{\text{corrected $d=4$ soldering (requires explicit selection of complex subalgebra)}}
\\
&\quad +
\underbrace{
S_{\mathrm{GUT/SM}} + S_{\mathrm{EH}}
}_{\text{conditional EFT layer}}.
\end{aligned}
\]

The false thesis (void + single mirror $\Longrightarrow$ SM + GR + QFT + cosmology) has been removed.

## What is rigorously established

- Dimension of the Albert algebra: $\dim_{\R} J_3(\mathbb{O}) = 27$.
- Dimension of the EIII space: $\dim_{\R} E_6/(\Spin(10)\cdot U(1)) = 32$.
- Tangent representation and the branching $\mathbf{27} \downarrow = \mathbf{1}_{+4} \oplus \mathbf{10}_{-2} \oplus \mathbf{16}_{+1}$.
- Correct dimensions and signatures of the Hermitian matrix models $H_2(\mathbb{K})$ for normed division algebras (in particular, $H_2(\mathbb{C})$ is the source of 4d Minkowski space; $H_2(\mathbb{H})$ is 6-dimensional of signature $(1,5)$).
- The typed nature of the operations that appear in the exceptional ladder (no single uniform involutive functor $\Mirror$ with $\Mirror^2 \simeq \id$ is currently known).

All other physical claims are conditional on explicitly named assumptions or are open conjectures.

## Build

```bash
# With a working LaTeX installation (TeX Live / MacTeX / toolbox container, etc.)
pdflatex sjet_true.tex
# or
latexmk -pdf sjet_true.tex
```

The old build script `.sjet-build/assemble-sjet.sh` assembles the historical (uncorrected) manuscript.

## Key files (JSET v1)

| Path                        | Content |
|-----------------------------|---------|
| `sjet_true.tex`             | **Canonical JSET v1 deliverable**: publication-style theorem/conjecture ledger (rigorous core + conditional layers). Build with `build_clean.sh`. |
| `build_clean.sh`            | Primary build script for the v1 ledger (recommended). |
| `README.md`                 | This file — current status, build, and how to use v1. |
| `SJET-VALIDATION-REPORT.md` | Full 39-agent adversarial validation that drove the v1 corrections (must-read for context). |
| `sjet-prd.tex`              | Product/design requirements context (language-synchronized for v1 honesty). |
| `historical/`               | Pre-v1 archive (tar ball of old full programme + explanation; see `historical/README.historical.md`). Preserved for auditability; not current claims. |
| `neuro-sjet-mapping.md`     | Separate speculative side-mapping (all v1 caveats apply; not part of the rigorous core). |
| `.sjet-build/`              | Legacy modular sources for the pre-audit (historical) manuscript. |
| `sjet.tex` / `sjet.pdf`     | Historical full manuscript (superseded; use only for reference against the validation report). |

## Citation and use

When citing the rigorous core, refer to the statements classified as Standard Imported Theorems or Theorems in `sjet_true.tex`. Do not cite the document for claims that the theory derives the Standard Model, general relativity, or quantum field theory from two primitives.

## Tone and standards enforced in this version

- Every physical statement is conditional and lists its assumptions.
- No "derivation from void and mirror."
- No "parameter-free."
- No "the universe is forced."
- Capacity is treated as bookkeeping or a Legendre problem with external targets unless independent data are supplied.
- The family count is a named conjecture (\textbf{FAM}) pending an explicit construction and computation.
- Soldering uses $H_2(\mathbb{C})$, with the selection of $\mathbb{C}\subset\mathbb{O}$ stated as an assumption.
- Open problems are listed with sufficient precision that progress is falsifiable.

See Section 9 (Open Problems) and the global summary ledger in `sjet_true.tex` for the exact current status.

## Relation to previous work

The validation report and external technical reviews identified fatal algebraic, dimensional, and circularity issues in the load-bearing derivations (incorrect $H_2(\mathbb{H})$ dimension/signature, $\sigma$ simultaneously treated as order-2 involution and order-6 triality, incoherent cellular cohomology count, Lefschetz misapplication, capacity circularity, false flavon VEV algebra, etc.). This cleaned version implements the required structural corrections while preserving the valuable exceptional-algebra scaffold.

The neuro-SJET mapping (`neuro-sjet-mapping.md`) is a separate speculative exercise that applies the same generative style to a different domain; it inherits the same caveats about what is derived versus conjectural.