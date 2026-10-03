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


"""Fig. 9 (hardware): long (40 gates) vs. reduced (short) 2-qubit circuit on
IBM Eagle r3 chips Brisbane and Kyiv.  We use the calibrated noise models
of exactly these devices (qiskit_ibm_runtime FakeBrisbane / FakeKyiv)
and, unlike the paper, (a) repeat over 30 random circuits, (b) report the
qiskit transpiler level (0, 1, 3), (c) give uncertainties.
"""

import json

import matplotlib
from common import FIGS, RESULTS, database

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv

from qcr.gates import make_pool, random_circuit
from qcr.reducer import DBEngine, reduce_circuit
from qcr.verify import equivalent, to_qiskit

N_CIRC, SHOTS, LEVELS = 30, 4000, (0, 1, 3)


def tvd(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def run():
    pool2 = make_pool("ion", 2)
    db_pool, cg = database("ion_pm", 2, 6)
    backends = {"Brisbane": FakeBrisbane(), "Kyiv": FakeKyiv()}
    sims = {k: AerSimulator.from_backend(b) for k, b in backends.items()}
    rows = []
    for s in range(N_CIRC):
        long = random_circuit(pool2, 40, np.random.default_rng(500 + s))
        short, _ = reduce_circuit(
            long, DBEngine(cg, "hash"), db_pool, np.random.default_rng(s), time_budget=30, stall_attempts=20000
        )
        assert equivalent(short, long, 2)
        ideal = Statevector(to_qiskit(long, 2)).probabilities_dict()
        ideal = {k[::-1]: v for k, v in ideal.items()}  # qiskit little-endian -> q0 first
        for name, b in backends.items():
            for lvl in LEVELS:
                row = {"seed": s, "backend": name, "level": lvl, "len_long": len(long), "len_short": len(short)}
                for tag, c in (("long", long), ("short", short)):
                    qc = to_qiskit(c, 2)
                    qc.measure_all()
                    tq = transpile(qc, b, optimization_level=lvl, seed_transpiler=s)
                    ops = tq.count_ops()
                    row[f"{tag}_ops"] = int(sum(v for k, v in ops.items() if k not in ("measure", "barrier")))
                    row[f"{tag}_2q"] = int(ops.get("ecr", 0) + ops.get("cx", 0) + ops.get("cz", 0))
                    cnt = sims[name].run(tq, shots=SHOTS, seed_simulator=1000 + s).result().get_counts()
                    p = {k.replace(" ", "")[::-1]: v / SHOTS for k, v in cnt.items()}
                    row[f"{tag}_tvd"] = tvd(p, ideal)
                    row[f"{tag}_p"] = p
                row["ideal"] = ideal
                rows.append(row)
        print(s, len(long), len(short), flush=True)
    json.dump(rows, open(RESULTS / "hardware_sim.json", "w"), indent=1)
    return rows


def plot(rows):
    # (a) Fig.-9 style bars for circuit 0 at level 1
    r0 = {(r["backend"]): r for r in rows if r["seed"] == 0 and r["level"] == 1}
    keys = ["00", "01", "10", "11"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    bars = [
        ("Simulator", r0["Brisbane"]["ideal"]),
        ("Long, Brisbane", r0["Brisbane"]["long_p"]),
        ("Long, Kyiv", r0["Kyiv"]["long_p"]),
        ("Short, Brisbane", r0["Brisbane"]["short_p"]),
        ("Short, Kyiv", r0["Kyiv"]["short_p"]),
    ]
    w = 0.16
    for i, (lab, p) in enumerate(bars):
        axs[0].bar(np.arange(4) + (i - 2) * w, [100 * p.get(k, 0) for k in keys], w, label=lab)
    axs[0].set_xticks(range(4), keys)
    axs[0].set_ylabel("probability (%)")
    axs[0].set_title(
        f"circuit 0: {r0['Brisbane']['len_long']} vs {r0['Brisbane']['len_short']} gates (noisy simulation, level 1)",
        fontsize=9,
    )
    axs[0].legend(fontsize=7, frameon=False)
    # (b) TVD statistics over all circuits
    lab, mean, err, col = [], [], [], []
    for name in ("Brisbane", "Kyiv"):
        for lvl in LEVELS:
            for tag in ("long", "short"):
                v = np.array([r[f"{tag}_tvd"] for r in rows if r["backend"] == name and r["level"] == lvl])
                lab.append(f"{name[:4]} L{lvl}\n{tag}")
                mean.append(v.mean())
                err.append(v.std(ddof=1) / np.sqrt(len(v)))
                col.append("#c0392b" if tag == "long" else "#1f4e79")
    axs[1].bar(range(len(lab)), mean, yerr=err, color=col, capsize=2)
    axs[1].set_xticks(range(len(lab)), lab, fontsize=6)
    axs[1].set_ylabel("TVD to ideal (mean ± s.e.)")
    axs[1].set_title(f"{N_CIRC} random 2-qubit circuits", fontsize=9)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "repro_hardware.png", dpi=200)


if __name__ == "__main__":
    plot(run())
