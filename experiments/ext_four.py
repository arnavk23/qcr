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


"""Extension E5: the 4-qubit (NJP) protocol.

spheres : exact sphere sizes |S(d)| of the 4-qubit Clifford Cayley graph
          (d <= 6) -> rigorous counting lower bounds on the optimal gate
          count of a uniformly random 4-qubit Clifford, valid for ANY
          reduction method.
ew      : exact-window reduction (3-qubit exact oracle, wire reduction) on
          the same 100 + 100 circuits as exp_quality.py NJP_* configs.
grid    : 3-qubit gap decomposition over (window w, database radius r).
"""

import json
import sys
import time

import numpy as np
from common import RESULTS
from ext_gap import oracle

from qcr.exact import convex_window_reduce, exact_window_reduce
from qcr.gates import make_pool, random_circuit
from qcr.tableau import wide_ball_sizes
from qcr.verify import equivalent

CLIFF4 = 2 ** (16 + 8) * 3 * 15 * 63 * 255  # |C_4 / U(1)|


def lower_bounds(S, G):
    """S: exact sphere sizes for d <= len(S)-1. Extend by submultiplicativity
    |S(a+b)| <= |S(a)| |S(b)| and return bounds for a uniform element."""
    Sb = [float(x) for x in S]
    while sum(Sb) < G:
        k = len(Sb)
        Sb.append(min(Sb[a] * Sb[k - a] for a in range(1, k)))
    B = np.cumsum(Sb)
    cdf_ub = np.minimum(B / G, 1.0)  # Pr(|g| <= d) <= B(d)/|G|
    mean_lb = float(np.sum(1 - cdf_ub))  # E|g| = sum_d Pr(|g| > d)
    median_lb = int(np.argmax(cdf_ub >= 0.5))
    p_le = {d: float(cdf_ub[d]) for d in range(len(cdf_ub))}
    return dict(sphere_bounds=Sb, mean_lb=mean_lb, median_lb=median_lb, cdf_ub=p_le)


def spheres(depth=6):
    out = {}
    for kind in ("ion_pm", "nisq"):
        S = wide_ball_sizes(4, make_pool(kind, 4), depth)
        out[kind] = dict(spheres=S, **lower_bounds(S, CLIFF4))
        print(kind, out[kind]["mean_lb"], out[kind]["median_lb"], flush=True)
    json.dump(out, open(RESULTS / "ext_four_spheres.json", "w"), indent=1)


def ew_job(args):
    kind, seed, w = args
    circ = random_circuit(make_pool(kind, 4), 300, np.random.default_rng(10_000 + seed))  # = NJP_* seeds
    O = oracle("ion_pm" if kind == "ion" else "nisq")
    t = time.perf_counter()
    r = exact_window_reduce(circ, O, w=w, rng=np.random.default_rng(seed), patience=40)
    sec = time.perf_counter() - t
    assert equivalent(r, circ, 4)
    from qcr.gates import counts

    row = dict(kind=kind, seed=seed, w=w, counts=counts(r), total=len(r), seconds=sec)
    print(json.dumps(row), flush=True)
    return row


def cw_job(args):
    """Convex-window exact optimiser, best of K random restarts."""
    kind, seed, w = args
    K = 8
    circ = random_circuit(make_pool(kind, 4), 300, np.random.default_rng(10_000 + seed))
    O = oracle("ion_pm" if kind == "ion" else "nisq")
    t = time.perf_counter()
    best = None
    for k in range(K):
        r = convex_window_reduce(circ, O, 4, w=w, patience=30, rng=np.random.default_rng(1000 * seed + k))
        if best is None or len(r) < len(best):
            best = r
    sec = time.perf_counter() - t
    assert equivalent(best, circ, 4)
    from qcr.gates import counts

    row = dict(kind=kind, seed=seed, w=w, restarts=K, counts=counts(best), total=len(best), seconds=sec)
    print(json.dumps(row), flush=True)
    return row


def cw(kinds, w=24, n=100, procs=4):
    import multiprocessing as mp

    for k in kinds:
        oracle("ion_pm" if k == "ion" else "nisq")
    with mp.get_context("fork").Pool(procs) as p:
        rows = p.map(cw_job, [(k, s, w) for k in kinds for s in range(n)])
    json.dump(rows, open(RESULTS / f"ext_four_cw_{'_'.join(kinds)}.json", "w"), indent=1)


def ew(kinds, w=16, n=100, procs=4):
    import multiprocessing as mp

    for k in kinds:
        oracle("ion_pm" if k == "ion" else "nisq")
    jobs = [(k, s, w) for k in kinds for s in range(n)]
    with mp.get_context("fork").Pool(procs) as p:
        rows = p.map(ew_job, jobs)
    json.dump(rows, open(RESULTS / f"ext_four_ew_{'_'.join(kinds)}.json", "w"), indent=1)


def grid_job(args):
    kind, seed = args
    circ = random_circuit(make_pool(kind, 3), 300, np.random.default_rng(10_000 + seed))
    O = oracle("ion_pm" if kind == "ion" else "nisq")
    row = dict(kind=kind, seed=seed, opt=O.dist(O.code_of(circ)))
    for w in (4, 6, 8, 10, 12, 16):
        for r in (3, 4, 5, 6, 8, 99):
            row[f"w{w}_r{r}"] = len(exact_window_reduce(circ, O, w=w, r=r, rng=np.random.default_rng(seed)))
    print(json.dumps(row), flush=True)
    return row


def grid(kinds, n=20, procs=4):
    import multiprocessing as mp

    for k in kinds:
        oracle("ion_pm" if k == "ion" else "nisq")
    with mp.get_context("fork").Pool(procs) as p:
        rows = p.map(grid_job, [(k, s) for k in kinds for s in range(n)])
    json.dump(rows, open(RESULTS / f"ext_grid_{'_'.join(kinds)}.json", "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1]
    kinds = [k for k in ("ion", "nisq") if k in sys.argv[2:]]
    if what == "spheres":
        spheres()
    elif what == "ew":
        ew(kinds)
    elif what == "cw":
        cw(kinds)
    elif what == "grid":
        grid(kinds)
