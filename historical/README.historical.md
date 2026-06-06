# Historical / Pre-v1 Material for JSET

This directory contains material from the pre-2026-06 development of the project (then often called SJET or UCFT-related).

**Important**: The content here reflects an earlier stage of the programme that underwent 39-agent adversarial validation in June 2026. The validation report (`SJET-VALIDATION-REPORT.md` at the repository root) concluded that the central dynamical claims were not supported and recommended a "reject & restructure" into:

- A short, honest mathematics note on the exceptional-algebra scaffold (representation theory only).
- An explicitly conjectural framework paper, with all "derivation", "closure", and "parameter-free" claims relabeled or removed.

**JSET v1** (`sjet_true.tex` at the repository root, plus this README and the overall repo hygiene) implements that restructuring.

## What is in the tarball

`pre-v1-full-programme.tar.gz` is an archive of the `old/` tree from git history (the "full programme archive (Waves 1–7, threads, formalization, ~205 pp)" plus earlier stacks). It is provided for:

- Historical context and audit trail.
- Reference for anyone who wants to see the exact claims that were audited.
- Potential future archaeological or "what was attempted" interest.

**Do not treat anything in this archive as current JSET v1 claims.** All load-bearing derivations (mirror functor, capacity partition, observer fixed-point count yielding exactly 3 families, quaternionic soldering for d=4 Lorentzian geometry, etc.) were found to have fatal issues (detailed in the validation report and corrected in the v1 ledger).

## How to use

```bash
tar -tzf historical/pre-v1-full-programme.tar.gz | head
tar -xzf historical/pre-v1-full-programme.tar.gz -C /tmp/jset-historical
```

The main v1 mathematical deliverable is always the file `sjet_true.tex` (or its built PDF) at the repository root, together with the explicit theorem/conjecture ledger, named assumptions (FAM, A-Solder-C, capacity as bookkeeping or Legendre, etc.), and the 6 open problems.

## See also (at repo root)

- `sjet_true.tex` — the canonical JSET v1 document (rigorous core + conditional layers).
- `README.md` — current status and build instructions for v1.
- `SJET-VALIDATION-REPORT.md` — the full 39-agent audit that drove the v1 corrections.
- `sjet-prd.tex` — the (language-synchronized for v1) product/design requirements context.

All claims in JSET v1 are classified (Theorem, Standard Imported Theorem, Conjecture, Assumption, Conditional EFT Consequence, etc.). Stronger physical statements require discharging the named open problems and assumptions.

This archive exists so that the correction is transparent and the prior work is not lost.
