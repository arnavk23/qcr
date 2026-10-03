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


"""Exact-oracle optimiser with adaptive, commutation-aware windows (CW-sem)
on the 15-qubit (Table V) and 4-qubit (journal) benchmarks.  Qiskit L3 is
reported both as returned (rotations by any multiple of pi/2 count as one
gate) and re-expressed in the paper's +-pi/2 alphabet."""

import json
import multiprocessing as mp
import sys
import time

import numpy as np
from common import RESULTS
from ext_gap import oracle
from qiskit import transpile

from qcr.exact import adaptive_reduce, oracle_score
from qcr.gates import counts, make_pool, random_circuit, to_quarter_turns
from qcr.verify import equivalent, from_qiskit, to_qiskit

CFG = {"T5": ("nisq", 15, 500, 10), "NJP_ion": ("ion", 4, 300, 100), "NJP_nisq": ("nisq", 4, 300, 100)}
BASIS = {"ion": ["rx", "ry", "rz", "rxx"], "nisq": ["rx", "rz", "cz"]}


def job(args):
    cfg, seed = args
    kind, n, L, _ = CFG[cfg]
    circ = random_circuit(make_pool(kind, n), L, np.random.default_rng(10_000 + seed))  # = exp_quality seeds
    O = oracle("ion_pm" if kind == "ion" else "nisq")
    sc, rw = oracle_score(O), (lambda code, W: O.optimal_word(code))
    q3raw = from_qiskit(
        transpile(to_qiskit(circ, n), basis_gates=BASIS[kind], optimization_level=3, seed_transpiler=seed)
    )
    try:
        q3 = to_quarter_turns(q3raw)
    except ValueError:
        q3 = None  # qiskit used an angle outside the +-pi/2 alphabet
    row = dict(
        cfg=cfg, seed=seed, qiskit_raw=counts(q3raw), qiskit=counts(q3) if q3 else None, qiskit_legal=q3 is not None
    )
    for tag, start in (("ours", circ), ("q3_ours", q3)):
        if start is None:
            continue
        t = time.perf_counter()
        r = adaptive_reduce(
            start, 24, 3, sc, rw, rng=np.random.default_rng(seed), restarts=4 if n > 4 else 8, max_jump=400
        )
        row[tag] = counts(r)
        row[tag + "_s"] = time.perf_counter() - t
        row[tag + "_ok"] = bool(equivalent(r, circ, n))
    print(json.dumps({k: (sum(v.values()) if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
    return row


if __name__ == "__main__":
    cfgs = sys.argv[1:] or list(CFG)
    for k in ("ion_pm", "nisq"):
        oracle(k)
    jobs = [(c, s) for c in cfgs for s in range(CFG[c][3])]
    with mp.get_context("fork").Pool(6) as p:
        rows = p.map(job, jobs)
    json.dump(rows, open(RESULTS / f"ext_bench_cliff_{'_'.join(cfgs)}.json", "w"), indent=1)
