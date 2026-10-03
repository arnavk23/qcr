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


"""Table I, Fig. 1 check, Fig. 4/5 style reduction traces, Fig. 6 convergence,
and a re-run of the authors' own demo circuit (longcode10.txt)."""

import json
from pathlib import Path

import matplotlib
from common import FIGS, RESULTS, database, trained_rf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qcr.compute_graph import build
from qcr.gates import circuit_unitary, make_pool, parse_qasm, random_circuit
from qcr.reducer import DBEngine, RandomSamplingEngine, reduce_circuit
from qcr.unitary import matlab_compare
from qcr.verify import equivalent

DEMO = Path(__file__).resolve().parents[1] / "external" / "QCOptimDemo"  # official zip, NJP ref. [38]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def table1():
    rows = []
    for n, maxd in ((2, 6), (3, 5)):
        pool = make_pool("ion_pm", n)
        cg = build(n, pool, maxd)
        for d in range(1, maxd + 1):
            rows.append(
                dict(
                    qubits=n,
                    ops=len(pool),
                    depth=d,
                    nodes=int((cg.dist <= d).sum()),
                    edges=int((cg.dist < d).sum()) * len(pool),
                )
            )
        rows.append(dict(qubits=n, build_seconds=cg.build_seconds))
    json.dump(rows, open(RESULTS / "table1.json", "w"), indent=1)
    print(rows)


def trace_example(n, length, seed, dbkind, depth, tag):
    """Fig. 4 / Fig. 5 analogue: log every replacement as a token chain."""
    pool = make_pool("ion", n)
    tok = {g: i + 1 for i, g in enumerate(pool)}
    circ = random_circuit(pool, length, np.random.default_rng(seed))
    db_pool, cg = database(dbkind, 3 if n > 2 else 2, depth)
    out, tr = reduce_circuit(
        circ, DBEngine(cg, "hash"), db_pool, np.random.default_rng(seed), time_budget=60, stall_attempts=30000
    )
    res = dict(
        tag=tag,
        n=n,
        input=[tok[g] for g in circ],
        output=[str(g) for g in out],
        lengths=tr.lengths,
        ok=bool(equivalent(out, circ, n)),
    )
    json.dump(res, open(RESULTS / f"example_{tag}.json", "w"), indent=1)
    print(tag, len(circ), "->", len(out), res["ok"])
    return res


def demo_rerun():
    L, n = parse_qasm(open(DEMO / "longcode10.txt").read())
    S, _ = parse_qasm(open(DEMO / "shortcode10.txt").read())
    out = dict(
        authors_in=len(L),
        authors_out=len(S),
        authors_equiv_err=matlab_compare(circuit_unitary(S, n), circuit_unitary(L, n)),
    )
    for dbkind, depth in (("ion", 4), ("ion_pm", 5)):
        db_pool, cg = database(dbkind, 3, depth)
        res = []
        for s in range(5):
            red, tr = reduce_circuit(
                L, DBEngine(cg, "hash"), db_pool, np.random.default_rng(s), time_budget=300, stall_attempts=100000
            )
            assert equivalent(red, L, n)
            res.append(len(red))
        out[f"ours_{dbkind}_d{depth}"] = res
    json.dump(out, open(RESULTS / "demo_rerun.json", "w"), indent=1)
    print(out)


def convergence(seed=3, n=5, length=100):
    pool3, cg = database()
    circ = random_circuit(make_pool("ion", n), length, np.random.default_rng(seed))
    curves = {}
    for name in ("V1-RS", "V2-DR", "V3-RF", "V3-RF(t=0.2)"):
        clf = None
        if name == "V1-RS":
            eng = RandomSamplingEngine(pool3, 3, rng=np.random.default_rng(1))
        else:
            eng = DBEngine(cg, "scan")
            if name.startswith("V3"):
                clf = trained_rf(threshold=0.5 if name == "V3-RF" else 0.2)
        out, tr = reduce_circuit(
            circ, eng, pool3, np.random.default_rng(seed), target_len=48, time_budget=400, classifier=clf
        )
        curves[name] = (tr.times, tr.lengths)
        print(name, len(out), tr.times[-1])
    json.dump(curves, open(RESULTS / "convergence.json", "w"))
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for (name, (t, l)), c in zip(curves.items(), ["#c0392b", "#d35fd3", "#1f4e79", "#2e86c1"]):
        ax.step(t, l, where="post", lw=2, color=c, label=name, ls="--" if "t=0.2" in name else "-")
    ax.set_xlabel("computation time (s)")
    ax.set_ylabel("circuit length")
    ax.legend(frameon=False)
    ax.set_title(f"Fig. 6 reproduction: one {n}-qubit circuit of length {length}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "repro_convergence.png", dpi=200)


if __name__ == "__main__":
    import sys

    what = sys.argv[1:] or ["table1", "examples", "demo", "convergence"]
    if "table1" in what:
        table1()
    if "examples" in what:
        trace_example(2, 40, 7, "ion_pm", 6, "fig4_2q_len40")
        trace_example(3, 200, 8, "ion_pm", 5, "fig5_3q_len200")
    if "demo" in what:
        demo_rerun()
    if "convergence" in what:
        convergence()
