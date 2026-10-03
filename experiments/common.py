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


import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
CACHE = ROOT / "cache"
for p in (RESULTS, FIGS, CACHE):
    p.mkdir(exist_ok=True)

from functools import lru_cache

import numpy as np  # noqa: E402

from qcr.compute_graph import cached_build  # noqa: E402
from qcr.gates import make_pool  # noqa: E402
from qcr.reducer import DBEngine  # noqa: E402
from qcr.rf import make_training_set  # noqa: E402


@lru_cache(maxsize=4)
def database(kind="ion", dimD=3, depth=4):
    pool = make_pool(kind, dimD)
    cg = cached_build(dimD, pool, depth, kind, cache_dir=CACHE)
    return pool, cg


def trained_rf(kind="ion", dimD=3, depth=4, circ_n=5, n_samples=20000, threshold=0.5, seed=5):
    path = CACHE / f"rf_{kind}_D{dimD}_d{depth}_c{circ_n}.pkl"
    pool, cg = database(kind, dimD, depth)
    if path.exists():
        rf = pickle.load(open(path, "rb"))
    else:
        X, y, rf = make_training_set(
            DBEngine(cg, "hash"), make_pool(kind, circ_n), pool, dimD, n_samples, np.random.default_rng(seed)
        )
        rf.fit(X, y)
        rf.train_pos_rate = float(y.mean())
        pickle.dump(rf, open(path, "wb"))
    rf.threshold = threshold
    return rf
