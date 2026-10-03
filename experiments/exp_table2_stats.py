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


"""Table II / Fig. 6: computation time of V1, V2, V3 over 100 random circuits.

Protocol (paper: "100 different starting circuits of length 100 ... reduced
to a target size"; qubit count unspecified -- we use the authors' demo
setting: 5-qubit ion-trap circuits, 3-qubit depth-4 database).  V2/V3 use
the faithful linear-scan lookup (MATLAB-like cost); V3 is run with the
default majority-vote threshold (0.5) and with a recall-tuned threshold
(0.2).  Every run is capped at 600 s.
"""

import json
import time
from multiprocessing import Pool

import numpy as np
from common import RESULTS, database, trained_rf

from qcr.gates import circuit_unitary, make_pool, random_circuit
from qcr.reducer import DBEngine, RandomSamplingEngine, reduce_circuit
from qcr.unitary import matlab_compare

N_CIRC, LENGTH, NQ, TARGET, CAP = 100, 100, 5, 60, 600


def job(args):
    seed, variant = args
    pool3, cg = database()
    circ = random_circuit(make_pool("ion", NQ), LENGTH, np.random.default_rng(1000 + seed))
    clf = None
    if variant == "V1-RS":
        eng = RandomSamplingEngine(pool3, 3, rng=np.random.default_rng(seed + 77))
    else:
        eng = DBEngine(cg, "scan")
        if variant.startswith("V3"):
            clf = trained_rf(threshold=0.5 if variant == "V3-RF" else 0.2)
    t = time.perf_counter()
    out, tr = reduce_circuit(
        circ, eng, pool3, np.random.default_rng(seed), target_len=TARGET, time_budget=CAP, classifier=clf
    )
    err = matlab_compare(circuit_unitary(out, NQ), circuit_unitary(circ, NQ))
    return dict(
        seed=seed,
        variant=variant,
        final_len=len(out),
        seconds=time.perf_counter() - t,
        reached=len(out) <= TARGET,
        attempts=tr.attempts,
        lookups=tr.lookups,
        unitary_err=err,
    )


if __name__ == "__main__":
    database()
    trained_rf()
    variants = ["V1-RS", "V2-DR", "V3-RF", "V3-RF(t=0.2)"]
    jobs = [(s, v) for s in range(N_CIRC) for v in variants]
    out = []
    with Pool(8) as p:
        for r in p.imap_unordered(job, jobs):
            out.append(r)
            print(json.dumps(r), flush=True)
    json.dump(out, open(RESULTS / "table2_stats.json", "w"), indent=1)
