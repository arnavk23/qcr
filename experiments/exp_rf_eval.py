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


"""Held-out accuracy of the V3 random forest (not reported in the paper)."""

import json

import numpy as np
from common import RESULTS, database, trained_rf

from qcr.gates import make_pool
from qcr.reducer import DBEngine
from qcr.rf import make_training_set

pool3, cg = database()
rf = trained_rf()
X, y, _ = make_training_set(
    DBEngine(cg, "hash"), make_pool("ion", 5), pool3, 3, 10000, np.random.default_rng(99), rf=rf
)
p = np.array([rf.proba(x) for x in X])
out = dict(pos_rate=float(y.mean()), train_pos_rate=rf.train_pos_rate)
for t in (0.1, 0.2, 0.3, 0.5):
    pr = p >= t
    out[str(t)] = dict(
        recall=float((pr & y).sum() / y.sum()),
        precision=float((pr & y).sum() / max(pr.sum(), 1)),
        pass_rate=float(pr.mean()),
    )
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS / "rf_eval.json", "w"), indent=1)
