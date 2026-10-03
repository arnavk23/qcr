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


"""Extension E1: enumerate the full 3-qubit Clifford group (92,897,280 elements)
under the paper's gate pools, giving exact optimal gate counts for every
3-qubit circuit, the exact growth series and diameter, and a bipartiteness
(gate-count parity) test."""

import json
import sys
import time

import numpy as np
from common import CACHE, RESULTS

from qcr.gates import make_pool
from qcr.tableau import cayley_bfs


def parity_test(codes, dists, tb, tables, chunk=4_000_000):
    """Is there an edge between two elements whose distances have equal parity?"""
    same = 0
    for s in range(0, len(codes), chunk):
        c, d = codes[s : s + chunk], dists[s : s + chunk]
        for T in tables.values():
            nb = tb.apply(T, c)
            dn = dists[np.searchsorted(codes, nb)]
            same += int(np.sum((dn % 2) == (d % 2)))
        if same:
            break
    return same


if __name__ == "__main__":
    kind = sys.argv[1]
    pool = make_pool(kind, 3)
    t = time.time()
    codes, dists, levels, tb, tables = cayley_bfs(3, pool)
    secs = time.time() - t
    np.savez(CACHE / f"cayley3_{kind}.npz", codes=codes, dists=dists)
    same = parity_test(codes, dists, tb, tables)
    out = dict(
        kind=kind,
        pool_size=len(pool),
        order=int(len(codes)),
        levels=levels,
        diameter=len(levels) - 1,
        mean_dist=float(np.dot(np.arange(len(levels)), levels) / len(codes)),
        bfs_seconds=secs,
        same_parity_edges_found=same,
    )
    json.dump(out, open(RESULTS / f"cayley3_{kind}.json", "w"), indent=1)
    print(out)
