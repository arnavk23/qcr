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


"""Tables III-V (arXiv) and the 100-circuit statistics of the NJP version:
gate counts after qiskit optimisation levels 1-3 vs. the local reducer.

Circuits are uniformly random token strings over the native pool (this
matches the gate proportions in the paper's tables).  "ours" = V2 database
retrieval with hash lookup (identical replacements to the linear scan),
3-qubit database, stopped after STALL consecutive unsuccessful attempts.
"""

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from common import RESULTS, database
from qiskit import transpile

from qcr.gates import counts, make_pool, random_circuit
from qcr.reducer import DBEngine, reduce_circuit
from qcr.verify import equivalent, from_qiskit, to_qiskit

CONFIGS = {
    # name: (kind, qubits, length, n_circuits, db_depth)
    "T3_ion_3q": ("ion", 3, 300, 20, 5),
    "T4_nisq_3q": ("nisq", 3, 300, 20, 6),
    "T5_nisq_15q": ("nisq", 15, 500, 10, 6),
    "NJP_ion_4q": ("ion", 4, 300, 30, 5),
    "NJP_nisq_4q": ("nisq", 4, 300, 30, 6),
}
# Database pools: the ion-trap database is the 24-operator (rxx +-pi/2) depth-5
# graph of Table I -- with it the faithful loop reproduces Table III (42 vs 45
# gates); the 21-operator depth-4 demo database stalls near 76.
DB_KIND = {"ion": "ion_pm", "nisq": "nisq"}
BASIS = {"ion": ["rx", "ry", "rz", "rxx"], "nisq": ["rx", "rz", "cz"]}
STALL, CAP = 100000, 400


def job(args):
    cfg, seed = args
    kind, n, length, _, depth = CONFIGS[cfg]
    pool = make_pool(kind, n)
    circ = random_circuit(pool, length, np.random.default_rng(10_000 + seed))
    row = {"cfg": cfg, "seed": seed, "original": counts(circ)}
    qc = to_qiskit(circ, n)
    for lvl in (1, 2, 3):
        t = time.perf_counter()
        out = transpile(qc, basis_gates=BASIS[kind], optimization_level=lvl, seed_transpiler=seed)
        gl = from_qiskit(out)
        row[f"qiskit_L{lvl}"] = counts(gl)
        row[f"qiskit_L{lvl}_s"] = time.perf_counter() - t
        if n <= 15:
            row[f"qiskit_L{lvl}_ok"] = equivalent(gl, circ, n)
    db_pool, cg = database(DB_KIND[kind], 3, depth)
    t = time.perf_counter()
    red, tr = reduce_circuit(
        circ, DBEngine(cg, "hash"), db_pool, np.random.default_rng(seed), time_budget=CAP, stall_attempts=STALL
    )
    row["ours"] = counts(red)
    row["ours_s"] = time.perf_counter() - t
    row["ours_ok"] = equivalent(red, circ, n)
    row["ours_attempts"] = tr.attempts
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    for c in a.configs:
        kind, _, _, _, depth = CONFIGS[c]
        database(DB_KIND[kind], 3, depth)
    jobs = [(c, s) for c in a.configs for s in range(CONFIGS[c][3])]
    rows = []
    with Pool(a.procs) as p:
        for r in p.imap_unordered(job, jobs):
            rows.append(r)
            print(
                r["cfg"],
                r["seed"],
                sum(r["original"].values()),
                sum(r["qiskit_L3"].values()),
                sum(r["ours"].values()),
                r["ours_ok"],
                round(r["ours_s"], 1),
                flush=True,
            )
    tag = "_".join(a.configs) if len(a.configs) < len(CONFIGS) else "all"
    json.dump(rows, open(RESULTS / f"quality_{tag}.json", "w"), indent=1)
