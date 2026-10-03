# Experiments

One script per table, figure or benchmark. Scripts write raw data to `../results/` (JSON + logs), figures to
`../figures/`, and large intermediate tables to `../cache/` (recreated when missing). Numbers reach the manuscripts
only through the three `make_*` scripts. Run everything from this directory after
`pip install -e ".[experiments]"` and `../tools/fetch_external.sh`.

## Reproduction of the original paper

| script | produces | output |
|---|---|---|
| `exp_figures.py table1 examples demo convergence` | Table I, Figs. 1/4/5 traces, authors' demo circuit, Fig. 6 | `table1.json`, `example_*.json`, `demo_rerun.json`, `convergence.json` |
| `exp_table2_stats.py` | Table II / Fig. 7 (V1, V2, V3 timing, 100 circuits; ~3 h on 8 cores) | `table2_stats.json` |
| `exp_rf_eval.py` | held-out recall/precision of the V3 random forest | `rf_eval.json` |
| `exp_quality.py` | Tables III-V and the journal's 4-qubit statistics vs. Qiskit | `quality_all.json`, `quality_summary.json` |
| `exp_hardware.py` | Fig. 9 on noisy FakeBrisbane / FakeKyiv at Qiskit levels 0/1/3 | `hardware_sim.json`, `hardware_summary.json` |

## Exact Clifford ground truth

| script | produces | output |
|---|---|---|
| `ext_cayley3.py ion_pm` / `nisq` | full 3-qubit Clifford group, distance of every element (~25 min, ~6 GB each) | `cayley3_*.json`, `cache/cayley3_*.npz` |
| `run_ext_queue.sh` | runs `ext_gap.py`, `ext_parity.py`, `ext_twoq.py` in order | see below |
| `ext_gap.py` | optimality gap, exact-window sweep, window x radius grid, benchmark mixing | `ext_gap3_*.json`, `ext_grid_*.json`, `ext_mixing.json` |
| `ext_parity.py` | shortest odd relation (parity theorem) | `ext_parity_*.json` |
| `ext_twoq.py` | minimal entangling-gate count of every 3-qubit Clifford | `ext_twoq3.json` |
| `ext_four.py cw ion nisq` | 4-qubit journal benchmark: CW-24, EW-16, sphere lower bounds | `ext_four_*.json` |
| `ext_bench_cliff.py` | adaptive commutation-aware windows (CW-sem) on 4 and 15 qubits | `ext_bench_cliff_*.json` |

## General optimiser

| script | produces | output |
|---|---|---|
| `gen_build_dbs.py` | hashed databases for Heron, ion and the paper's alphabets (up to 38 min each) | `cache/gdb_*.npz`, `gdb_info.json` |
| `gen_paper_bench.py` | general engine on the paper's benchmarks (paper semantics) | `general_paper_bench_*.json` |
| `gen_bench.py --tag v2` | 60-circuit hardware suite on Heron and ion vs. Qiskit L1/L3 and TKET, noisy TVD | `general/bench_v2.json` |
| `verify_large.py nam/pf2_10` | verification of outputs routed over more than 16 qubits | `general/verify_large_*.json` |

## Tables, figures and numbers for the manuscripts

| script | writes |
|---|---|
| `make_assets.py` | `figures/repro_*.png`, `results/tex/*.tex`, `manuscripts/reproduction/numbers.tex` |
| `make_ext_assets.py` | `manuscripts/extension/figs/ext_*.png`, `manuscripts/extension/ext_numbers.tex` |
| `make_general_assets.py` | `manuscripts/extension/gen_*.tex`, `manuscripts/extension/figs/gen_*` |
| `fig_notes.py` | explanatory figures for the study notes |

`common.py` holds shared paths and cached databases.
