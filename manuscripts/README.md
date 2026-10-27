# Manuscripts

| directory | document |
|---|---|
| `notes/` | study notes: the original paper explained from scratch |
| `reproduction/` | reproduction report: every table and figure re-run |
| `paper/` | the paper (`paper.tex`): ground truth, theory, general optimiser, benchmarks, baselines; `DEVLOG.md` records design decisions and corrections |
| `talk/` | presentations (beamer): `talk.tex` (full) and `prof_talk.tex` (short) |

Build with `make` (from this directory) or `make -C manuscripts` (from the root). Generated inputs
(`*_numbers.tex`, `gen_*.tex`, figures) come from `experiments/make_*.py`; do not edit them by hand.
