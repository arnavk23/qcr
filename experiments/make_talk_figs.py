# This code is part of qcr.
#
# (C) Copyright Arnav Kapoor 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Figures for the advisor talk (manuscripts/talk/figs/talk_*.png).

  example   the paper-setting example circuit, one large PNG per stage (input, original method, optimum),
            with the gate lists saved to results/talk_example.json
  hardware  per-circuit success probability on ibm_marrakesh relative to Qiskit L3 (mean and sd over repeats)
  errors    median calibrated 1q/2q/readout errors of FakeTorino, printed for the motivation slide
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from common import RESULTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parents[1] / "manuscripts" / "talk" / "figs"
INK, MUTED, GRID = "#1F1F1E", "#6B6A63", "#E2E1DC"
L3OURS, L1OURS = "#2A78D6", "#EB6834"


def _circuit_png(gates, n, name, fold):
    from qiskit.visualization import circuit_drawer

    from qcr.verify import to_qiskit

    from figstyle import CIRCUIT_STYLE  # same gate colours as the paper

    f = circuit_drawer(to_qiskit(gates, n), output="mpl", fold=fold, style=CIRCUIT_STYLE)
    f.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(f)


def example():
    from ext_gap import oracle

    from qcr.compute_graph import cached_build
    from qcr.gates import make_pool, random_circuit
    from qcr.reducer import DBEngine, reduce_circuit

    # same search as make_paper_figs.examples(), so the circuit matches the paper figure's setting
    if (RESULTS / "talk_example.json").exists():  # reuse the circuit already shown (the search is time-budgeted)
        from qcr.gates import Gate

        d = json.load(open(RESULTS / "talk_example.json"))
        circ, orig, opt = ([Gate(n, tuple(q), t) for n, q, t in d[k]] for k in ("input", "orig", "opt"))
        _circuit_png(circ, 3, "talk_example_input", fold=28)
        _circuit_png(orig, 3, "talk_example_orig", fold=-1)
        _circuit_png(opt, 3, "talk_example_opt", fold=-1)
        return
    pool = make_pool("ion_pm", 3)
    cg = cached_build(3, pool, 5, "ion_pm", cache_dir=RESULTS.parent / "cache")
    O = oracle("ion_pm")
    best = None
    for seed in range(8):
        circ = random_circuit(pool, 80, np.random.default_rng(100 + seed))
        orig, _ = reduce_circuit(
            circ, DBEngine(cg, "hash"), pool, np.random.default_rng(seed), time_budget=40, stall_attempts=20000
        )
        opt = O.optimal_word(O.code_of(circ))
        if best is None or len(orig) - len(opt) > len(best[1]) - len(best[2]):
            best = (circ, orig, opt)
        if len(orig) - len(opt) >= 8:
            break
    circ, orig, opt = best
    _circuit_png(circ, 3, "talk_example_input", fold=28)
    _circuit_png(orig, 3, "talk_example_orig", fold=-1)
    _circuit_png(opt, 3, "talk_example_opt", fold=-1)
    rows = {k: [(g.name, list(g.qubits), g.theta) for g in v] for k, v in (("input", circ), ("orig", orig), ("opt", opt))}
    json.dump(rows, open(RESULTS / "talk_example.json", "w"), indent=1)
    print({k: len(v) for k, v in rows.items()})


def hardware():
    a = json.load(open(RESULTS / "hardware" / "ibm_marrakesh_v1" / "analysis.json"))
    rows = sorted(a["rows"], key=lambda r: r["pipes"]["L3"]["total"])
    names = [r["name"].split("/")[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(rows))
    for off, p, col, lab in ((-0.14, "L3_ours", L3OURS, "Qiskit L3, then ours"), (0.14, "L1_ours", L1OURS, "Qiskit L1, then ours")):
        base = np.array([r["pipes"]["L3"]["mean"] for r in rows])
        m = np.array([r["pipes"][p]["mean"] for r in rows]) / base
        # sd of the ratio from the per-repeat values (repeats are paired: same job, same calibration)
        sd = np.array([np.std(np.array(r["pipes"][p]["values"]) / np.array(r["pipes"]["L3"]["values"]), ddof=1) for r in rows])
        ax.errorbar(x + off, m, yerr=sd, fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.4, capsize=0, label=lab, zorder=3)
    ax.axhline(1, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(-0.45, 0.993, "Qiskit L3 alone", color=MUTED, fontsize=7, ha="left", va="top")
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=7, color=INK)
    ax.set_ylabel("success probability / Qiskit L3", fontsize=8, color=INK)
    ax.tick_params(axis="y", labelsize=7, colors=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2, handletextpad=0.3)
    ax.set_title(f"ibm_marrakesh, {a['repeats']} repeats x 4000 shots, circuits sorted by size (whiskers: 1 sd)",
                 fontsize=8, color=MUTED, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "talk_hardware.png", dpi=220)
    print(a["summary"])


def errors():
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    t = FakeTorino().target
    e1 = [p.error for q, p in t["sx"].items() if p and p.error is not None]
    e2 = [p.error for q, p in t["cz"].items() if p and p.error is not None and p.error < 1]
    em = [p.error for q, p in t["measure"].items() if p and p.error is not None]
    print(f"FakeTorino median errors: sx {np.median(e1):.2e}  cz {np.median(e2):.2e}  readout {np.median(em):.2e}  "
          f"cz/sx {np.median(e2) / np.median(e1):.0f}x  (pairs {len(e2)}, qubits {len(e1)})")  # fmt: skip


if __name__ == "__main__":
    for job in sys.argv[1:] or ["errors", "hardware", "example"]:
        dict(example=example, hardware=hardware, errors=errors)[job]()
