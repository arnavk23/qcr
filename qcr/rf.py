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


"""V3: random-forest gate in front of the database lookup (Sec. III.C).

The paper does not specify features; we use what is available without
computing the block unitary: the remapped token sequence (positional,
padded) and a bag-of-tokens histogram.  Labels come from the exact
database (reducible iff a strictly shorter entry exists), i.e. exactly the
"successfully reduced blocks from V2 as reducible examples" of the paper.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .gates import circuit_unitary, random_circuit
from .reducer import sample_block, shuffle_commuting

MAXLEN = 8


class RFGate:
    def __init__(self, db_pool, threshold=0.5, n_estimators=50, seed=0):
        self.tok = {g: i for i, g in enumerate(db_pool)}
        self.P = len(db_pool)
        self.threshold = threshold
        self.clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=1, random_state=seed)

    def features(self, block):
        f = np.zeros(MAXLEN + self.P + 1)
        for i, g in enumerate(block):
            t = self.tok[g]
            f[i] = t + 1
            f[MAXLEN + t] += 1
        f[-1] = len(block)
        return f

    def fit(self, X, y):
        t = time.perf_counter()
        self.clf.fit(X, y)
        self.train_seconds = time.perf_counter() - t
        # Flatten trees for low-overhead single-sample inference (sklearn's
        # predict_proba costs ~1 ms per call, hiding the gate's purpose).
        self.trees = []
        for est in self.clf.estimators_:
            t_ = est.tree_
            v = t_.value[:, 0, :]
            pos = v[:, -1] / v.sum(1) if v.shape[1] > 1 else np.full(len(v), float(self.clf.classes_[0]))
            self.trees.append(
                (
                    t_.children_left.tolist(),
                    t_.children_right.tolist(),
                    t_.feature.tolist(),
                    t_.threshold.tolist(),
                    pos.tolist(),
                )
            )
        return self

    def proba(self, f) -> float:
        s = 0.0
        for cl, cr, ft, th, pos in self.trees:
            i = 0
            while cl[i] != -1:
                i = cl[i] if f[ft[i]] <= th[i] else cr[i]
            s += pos[i]
        return s / len(self.trees)

    def predict_reducible(self, block) -> bool:
        return self.proba(self.features(block)) >= self.threshold


def make_training_set(engine, circ_pool, db_pool, dimD, n_samples, rng, circ_len=100, rf=None):
    rf = rf or RFGate(db_pool)
    X, y = [], []
    g = random_circuit(circ_pool, circ_len, rng)
    while len(X) < n_samples:
        if len(X) % 200 == 0:
            g = shuffle_commuting(random_circuit(circ_pool, circ_len, rng), rng)
        s, blen, W, m = sample_block(g, rng, dimD)
        block = [x.remap(m) for x in g[s : s + blen]]
        U = circuit_unitary(block, dimD)
        X.append(rf.features(block))
        y.append(engine.find(U, blen) is not None)
    return np.array(X), np.array(y), rf
