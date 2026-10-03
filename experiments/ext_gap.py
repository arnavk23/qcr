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


"""Extension experiments on exact optima (needs ext_cayley3.py output).

mixing   : E[opt(L)] of random length-L circuits vs. the uniform-measure
           limit (n=2 exact, n=3 exact), + spectral gap of the n=2 walk
gap3     : paper-faithful reducer vs. exact optimum on the Table III/IV
           circuits (same seeds as exp_quality.py) and gap vs. window w
"""

import json
import sys
import time

import numpy as np
from common import CACHE, RESULTS, database

from qcr.exact import Oracle, exact_window_reduce
from qcr.gates import make_pool, random_circuit
from qcr.reducer import DBEngine, reduce_circuit
from qcr.verify import equivalent

_OR = {}


def oracle(kind, n=3):
    key = (kind, n)
    if key not in _OR:
        if n == 3:
            z = np.load(CACHE / f"cayley3_{kind}.npz")
            _OR[key] = Oracle(kind, 3, z["codes"], z["dists"])
        else:
            _OR[key] = Oracle(kind, n)
    return _OR[key]


def growth_expectation(levels):
    lv = np.asarray(levels, float)
    d = np.arange(len(lv))
    even, odd = lv[::2].sum(), lv[1::2].sum()
    return dict(
        mean=float(d @ lv / lv.sum()),
        mean_even=float((d[::2] @ lv[::2]) / even),
        mean_odd=float((d[1::2] @ lv[1::2]) / odd),
    )


def mixing():
    out = {}
    for kind, n in (("ion_pm", 2), ("nisq", 2), ("ion_pm", 3), ("nisq", 3)):
        O = oracle(kind, n)
        lv = np.bincount(O.dists).tolist()
        pool = make_pool(kind, n)
        rng = np.random.default_rng(0)
        Ls = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 60, 100, 300]
        res = []
        for L in Ls:
            m = 2000 if n == 2 else 400
            d = [O.dist(O.code_of(random_circuit(pool, L, rng))) for _ in range(m)]
            res.append(dict(L=L, mean=float(np.mean(d)), std=float(np.std(d)), hist=np.bincount(d).tolist()))
            print(kind, n, L, res[-1]["mean"], flush=True)
        out[f"{kind}_{n}"] = dict(levels=lv, uniform=growth_expectation(lv), curve=res)
    # spectral gap of the n=2 (ion) random walk
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    O = oracle("ion_pm", 2)
    N = len(O.codes)
    rows, cols = [], []
    for i, c in enumerate(O.codes):
        for g in O.pool:
            rows.append(i)
            cols.append(int(np.searchsorted(O.codes, np.uint64(O.step(int(c), g)))))
    P = sp.csr_matrix((np.full(len(rows), 1 / len(O.pool)), (rows, cols)), shape=(N, N))
    sym = abs(P - P.T).max()
    vals = eigsh((P + P.T) / 2, k=6, which="LM", return_eigenvectors=False)
    out["spectral_ion2"] = dict(N=N, asym=float(sym), top_abs_eigs=sorted(map(float, vals), key=abs, reverse=True))
    json.dump(out, open(RESULTS / "ext_mixing.json", "w"), indent=1)


def gap_job(args):
    kind, seed, ws = args
    n, length = 3, 300
    circ = random_circuit(make_pool(kind, n), length, np.random.default_rng(10_000 + seed))  # = exp_quality seeds
    okind = "ion_pm" if kind == "ion" else "nisq"
    O = oracle(okind)
    opt = O.dist(O.code_of(circ))
    row = dict(kind=kind, seed=seed, opt=opt)
    db_pool, cg = database(okind, 3, 5 if kind == "ion" else 6)
    t = time.perf_counter()
    red, _ = reduce_circuit(
        circ, DBEngine(cg, "hash"), db_pool, np.random.default_rng(seed), time_budget=400, stall_attempts=100000
    )
    row["faithful"] = len(red)
    row["faithful_s"] = time.perf_counter() - t
    for w in ws:
        t = time.perf_counter()
        r = exact_window_reduce(circ, O, w=w, rng=np.random.default_rng(seed))
        assert equivalent(r, circ, n)
        row[f"ew{w}"] = len(r)
        row[f"ew{w}_s"] = time.perf_counter() - t
    print(json.dumps(row), flush=True)
    return row


def gap3(kinds, n_circ=20, procs=3):
    import multiprocessing as mp

    ws = [2, 3, 4, 6, 8, 12, 16, 24, 32]
    for k in kinds:  # load once in the parent; forked workers share it copy-on-write
        oracle("ion_pm" if k == "ion" else "nisq")
        database("ion_pm" if k == "ion" else "nisq", 3, 5 if k == "ion" else 6)
    jobs = [(k, s, ws) for k in kinds for s in range(n_circ)]
    with mp.get_context("fork").Pool(procs) as p:
        rows = p.map(gap_job, jobs)
    json.dump(rows, open(RESULTS / f"ext_gap3_{'_'.join(kinds)}.json", "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1:]
    if "mixing" in what:
        mixing()
    if "gap3" in what:
        gap3([k for k in ("ion", "nisq") if k in what])
