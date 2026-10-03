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


"""Shortest odd relation (odd cycle through the identity) in the 3-qubit
Cayley graphs.  An edge between two elements at the same BFS distance d
closes an odd cycle of length 2d+1 through the root; the smallest such d
gives the shortest odd relator.  Every rewrite u -> v with |u|+|v| below
that length preserves the parity of the gate count."""

import json

import numpy as np
from common import CACHE, RESULTS

from qcr.gates import gate_matrix, make_pool
from qcr.tableau import Tableau

out = {}
import sys

for kind in sys.argv[1:]:
    z = np.load(CACHE / f"cayley3_{kind}.npz")
    codes, dists = z["codes"], z["dists"]
    tb = Tableau(3)
    Ts = [tb.row_table(gate_matrix(g, 3)) for g in make_pool(kind, 3)]
    found = None
    for d in range(1, int(dists.max()) + 1):
        lvl = codes[dists == d]
        for s in range(0, len(lvl), 2_000_000):
            c = lvl[s : s + 2_000_000]
            for T in Ts:
                nd = dists[np.searchsorted(codes, tb.apply(T, c))]
                if np.any(nd == d):
                    found = d
                    break
            if found:
                break
        if found:
            break
    out[kind] = dict(first_same_level_edge_depth=found, shortest_odd_relator=2 * found + 1)
    print(kind, out[kind], flush=True)
json.dump(out, open(RESULTS / f"ext_parity_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
