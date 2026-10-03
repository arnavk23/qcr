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


"""Does the general engine (no Clifford knowledge) hold up on the paper's own
benchmarks?  Same circuits and seeds as exp_quality.py / ext_bench_cliff.py.
Paper semantics: equality up to global phase, cost = total gate count,
discrete +-pi/2 alphabets, all-to-all connectivity."""

import json
import multiprocessing as mp
import sys

import numpy as np
from common import CACHE, RESULTS

from qcr.gates import counts, make_pool, random_circuit
from qcr.general import GRIDS, Config, Database, Optimizer
from qcr.hardware import paper_device
from qcr.verify import equivalent

CFG = {
    "T3": ("ion", 3, 300, 20),
    "T4": ("nisq", 3, 300, 20),
    "NJP_ion": ("ion", 4, 300, 100),
    "NJP_nisq": ("nisq", 4, 300, 100),
    "T5": ("nisq", 15, 500, 10),
}
DBS = {}


def job(args):
    cfg, seed = args
    kind, n, L, _ = CFG[cfg]
    circ = random_circuit(make_pool(kind, n), L, np.random.default_rng(10_000 + seed))
    dk = "ion" if kind == "ion" else "ibm"
    dev = paper_device(dk, n)
    opt = Optimizer(dev, DBS[dk], Config(mode="phase", cost="count", w=16, max_jump=400, patience=8, restarts=4))
    r = opt.run(circ, seed=seed)
    row = dict(
        cfg=cfg,
        seed=seed,
        general=counts(r.gates),
        seconds=r.seconds,
        stats=r.stats,
        ok=bool(equivalent(r.gates, circ, n)),
        legal=all(dev.is_native(g) for g in r.gates),
    )
    print(
        json.dumps(
            dict(cfg=cfg, seed=seed, total=len(r.gates), ok=row["ok"], legal=row["legal"], s=round(r.seconds, 1))
        ),
        flush=True,
    )
    return row


if __name__ == "__main__":
    cfgs = sys.argv[1:] or list(CFG)
    for dk in ("ion", "ibm"):
        p = CACHE / f"gdb_{dk}_paper_tri_phase.npz"
        DBS[dk] = {("phase", "tri"): Database.load(p, GRIDS[f"{dk}_paper"])}
    jobs = [(c, s) for c in cfgs for s in range(CFG[c][3])]
    with mp.get_context("fork").Pool(6) as pool:
        rows = pool.map(job, jobs)
    json.dump(rows, open(RESULTS / f"general_paper_bench_{'_'.join(cfgs)}.json", "w"), indent=1)
