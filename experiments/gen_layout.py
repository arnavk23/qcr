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

"""Layout-aware rewriting on IBM Heron (FakeTorino), open problem 4 of the paper.

Pipelines per circuit (all exact, calibrated objective, dead couplers removed everywhere):
  L3            Qiskit level 3, basis + coupling map only (not noise-aware), seed 0
  L3+ours       our optimiser on L3                                   (run 2 of gen_bench.py)
  L3na          Qiskit level 3 against the calibrated target (noise-aware VF2 layout), seed 0
  L3na+ours     our optimiser on L3na
  L3na+ours+rel ... followed by noise-aware relocation (qcr.layout.relocate) and a final sweep
  joint         best of 4 seeds x {coupling, noise-aware} candidates, each optimised + relocated,
                ranked by calibrated cost after rewriting (qcr.layout.joint_layout)
Metrics: total, 2q, ESP, verification, noisy TVD with readout (<= 8 active qubits).
"""

import json
import multiprocessing as mp
import sys
import time

import gen_bench as gb
import numpy as np
from common import RESULTS
from qiskit import qasm2, transpile
from qiskit.transpiler import CouplingMap

from qcr.general import Config, Optimizer
from qcr.hardware import coupled_target, coupling_list, fake_backend, heron, qiskit_basis
from qcr.layout import joint_layout, relocate
from qcr.noise import ideal_probabilities, noisy_probabilities, tvd

OUT = RESULTS / "general"
EXCLUDE = {"nam/pf1_10", "nam/pf2_10"}  # Qiskit L3 drops their 1e-7 rad rotations; not comparable
DEV = DBS = BACKEND = None


def init():
    global DEV, DBS, BACKEND
    DEV, BACKEND = heron(), fake_backend("torino")
    DBS = gb.load_dbs("heron")


def optimize(gates):
    return Optimizer(DEV, DBS, Config(mode="exact", cost="calibrated", patience=4)).run(gates, seed=0).gates


def row_metrics(gates, ref, n_act_max=8):
    m = gb.metrics(gates, DEV)
    m["success_readout"] = DEV.success_probability(gates, readout=True)
    m["verify"] = gb.verify(ref, gates, "exact", False, 0.0) if ref is not None else None
    act = len({q for g in gates for q in g.qubits})
    if act <= n_act_max:
        m["tvd_readout"] = tvd(ideal_probabilities(gates), noisy_probabilities(gates, "torino", readout=True))
    return m


def job(args):
    name, qasm = args
    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_str(qasm)
    t0 = time.perf_counter()
    plain = dict(basis_gates=qiskit_basis(DEV), coupling_map=CouplingMap(coupling_list(DEV)))
    l3 = gb.qk_to_gates(transpile(qc, optimization_level=3, seed_transpiler=0, **plain))
    l3na = gb.qk_to_gates(transpile(qc, optimization_level=3, seed_transpiler=0, target=coupled_target(BACKEND)))
    ours = optimize(l3)
    na_ours = optimize(l3na)
    pl = relocate(na_ours, DEV)
    na_rel = optimize(pl.gates) if pl.cost_after < pl.cost_before else pl.gates
    best, cands = joint_layout(qc, DEV, optimize, backend=BACKEND, seeds=(0, 1, 2, 3))
    row = dict(name=name, n=qc.num_qubits)
    # every optimiser output is verified against its input; relocation is a relabelling, checked by comparing
    # gate sequences under the mapping
    row["L3"] = row_metrics(l3, None)
    row["L3+ours"] = row_metrics(ours, l3)
    row["L3na"] = row_metrics(l3na, None)
    row["L3na+ours"] = row_metrics(na_ours, l3na)
    row["L3na+ours+rel"] = row_metrics(na_rel, pl.gates)
    tr = best.trace
    row["joint"] = row_metrics(best.gates, tr["relocated"]) | dict(
        choice=best.label,
        verify_optimised=gb.verify(tr["routed"], tr["optimised"], "exact", False, 0.0),
        relocation_preserves_gates=_same_up_to_relabel(tr["optimised"], tr["relocated"], tr["mapping"]),
    )
    row["joint_candidates"] = [dict(label=c.label, cost=c.cost, **c.stats) for c in cands]
    row["relocation_preserves_gates"] = _same_up_to_relabel(na_ours, pl.gates, pl.mapping)
    row["seconds"] = time.perf_counter() - t0
    print(
        json.dumps(
            dict(
                name=name,
                esp_l3=round(row["L3"]["success"], 4),
                esp_joint=round(row["joint"]["success"], 4),
                choice=best.label,
                s=round(row["seconds"]),
            )
        ),
        flush=True,
    )
    return row


def _same_up_to_relabel(a, b, mapping):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.name != y.name or abs(x.theta - y.theta) > 1e-12 or {mapping.get(q, q) for q in x.qubits} != set(y.qubits):
            return False
    return True


def _np(o):
    """JSON encoder for numpy scalars."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "layout"
    circs = {k: v for k, v in gb.suite_circuits().items() if k not in EXCLUDE}
    part = OUT / f"bench_{tag}.jsonl"
    rows = [json.loads(line) for line in open(part)] if part.exists() else []
    done = {r["name"] for r in rows}
    jobs = [(k, qasm2.dumps(v)) for k, v in circs.items() if k not in done]
    print(f"{len(done)} done, {len(jobs)} to run", flush=True)
    with mp.get_context("fork").Pool(4, initializer=init) as p, open(part, "a") as f:
        for r in p.imap_unordered(job, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=_np) + "\n")
            f.flush()
    json.dump(rows, open(OUT / f"bench_{tag}.json", "w"), indent=1, default=_np)
    print(
        "mean ESP:",
        {
            k: float(np.exp(np.mean([np.log(r[k]["success"]) for r in rows])))
            for k in ("L3", "L3+ours", "L3na", "L3na+ours", "L3na+ours+rel", "joint")
        },
    )
