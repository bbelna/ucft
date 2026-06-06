# Archive

Prior manuscripts and build tooling preserved when SJET was rewritten as a focused mathematical-physics paper (June 2026).

## `sjet-stack/`

Full previous SJET manuscript (~39 pages): modular `.sjet-build/` units (including §11 axiomatic physics), assembled `sjet.tex`, PDF, and verification ledger.

Build (historical):

```bash
cd sjet-stack
bash .sjet-build/assemble-sjet.sh
latexmk -pdf sjet.tex
```

## UCFT era

- `ucft.tex`, `ucft-final.tex`, `ucft-final.pdf` — UCFT master documents
- `.ucft-build/` — modular UCFT units and assemble script
- `BAMLP-FOUNDATION.md`, `UCFT-SPINE.md` — design notes