# Manuscripts

| directory | document |
|---|---|
| `notes/` | study notes: the original paper explained from scratch |
| `reproduction/` | reproduction report: every table and figure re-run |
| `extension/` | extension paper: general optimiser, theory, Clifford ground truth, hardware results; `DEVLOG.md` records design decisions and corrections |
| `talk/` | presentation (beamer) |

Build with `make` (from this directory) or `make -C manuscripts` (from the root). Generated inputs
(`*_numbers.tex`, `gen_*.tex`, figures) come from `experiments/make_*.py`; do not edit them by hand.
