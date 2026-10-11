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

"""Certified optima beyond Clifford (open problem 2 of the extension paper).

Two-qubit circuits over the Clifford+T ion alphabet {rx, ry, rz}(+-pi/4, +-pi/2, pi) and rxx(+-pi/4, +-pi/2)
(the angle grid of the journal version of the reference paper). Exact arithmetic over Z[1/sqrt2, w]
(qcr.cliffordt) gives a collision-certified ball of radius 5 and exact distances up to 9 by meet in the middle.

Benchmarks:  random words of length 4..12 (exact optimum or certified lower bound);
             planted circuits: a random core whose optimum is certified, hidden in 60 / 120 gates.
Methods:     general optimiser (paper semantics: up to phase, gate count, this alphabet, ion/tri database),
             original method (faithful V2, 2-qubit compute graph of depth 4 over the same alphabet),
             Qiskit L3 (continuous angles, so a different alphabet: reference only).
Every output of the two discrete methods is checked for exact equality with its input.
"""

import json
import multiprocessing as mp
import time

import numpy as np
from common import CACHE, RESULTS
from qiskit import transpile

from qcr.cliffordt import ANGLES, MS_ANGLES, CliffordTBall, alphabet, exact_equal_up_to_phase, planted_circuit
from qcr.compute_graph import build
from qcr.general import GRIDS, Config, Database, Optimizer
from qcr.hardware import Device
from qcr.reducer import DBEngine, reduce_circuit
from qcr.verify import from_qiskit, to_qiskit

RADIUS = 5
BALL = DEV = DBS = CG = None


def setup():
    global BALL, DEV, DBS, CG
    t = time.time()
    BALL = CliffordTBall(RADIUS).build(verbose=True)
    info = dict(
        radius=RADIUS,
        levels=BALL.levels,
        seconds=time.time() - t,
        collision_free=BALL.check_collisions(),
        certified_up_to=RADIUS + len(BALL.spheres) - 1,
    )
    BALL.forms = []  # free memory once certified
    print(info, flush=True)
    grid = tuple(sorted(set(ANGLES) | set(MS_ANGLES)))
    DEV = Device("ion_cliffordT", 2, None, ("rx", "ry", "rz"), "rxx", (), False, angle_grid=grid)
    DBS = {("phase", "tri"): Database.load(CACHE / "gdb_ion_tri_phase.npz", GRIDS["ion"])}
    CG = build(2, alphabet(), 4)
    return info


def job(args):
    kind, size, seed = args
    key = size if kind == "random" else 100 * size[0] + size[1]
    rng = np.random.default_rng(1000 * key + seed + (0 if kind == "random" else 500_000))
    pool = alphabet()
    if kind == "random":
        circ = [pool[i] for i in rng.integers(len(pool), size=size)]
        core_len = None
    else:
        core_len, L = size
        while True:  # a core whose optimum is certified and equals its length
            core = [pool[i] for i in rng.integers(len(pool), size=core_len)]
            d, w, ex = BALL.distance(core)
            if ex and d == core_len:
                break
        circ = planted_circuit(core, L, BALL, rng)
    opt_len, opt_word, exact = BALL.distance(circ)
    t = time.perf_counter()
    gen = Optimizer(DEV, DBS, Config(mode="phase", cost="count", w=16, patience=8, restarts=4)).run(circ, seed=seed)
    t_gen = time.perf_counter() - t
    t = time.perf_counter()
    orig, _ = reduce_circuit(circ, DBEngine(CG, "hash"), alphabet(), rng, time_budget=30, stall_attempts=3000)
    t_orig = time.perf_counter() - t
    q3 = from_qiskit(
        transpile(to_qiskit(circ, 2), basis_gates=["rx", "ry", "rz", "rxx"], optimization_level=3, seed_transpiler=seed)
    )
    row = dict(
        kind=kind,
        size=size if kind == "random" else list(size),
        seed=seed,
        input=len(circ),
        optimum=opt_len,
        optimum_exact=exact,
        general=len(gen.gates),
        original=len(orig),
        qiskit_L3=len(q3),
        general_ok=exact_equal_up_to_phase(circ, gen.gates),
        original_ok=exact_equal_up_to_phase(circ, orig),
        general_legal=all(DEV.is_native(g) for g in gen.gates),
        t_general=t_gen,
        t_original=t_orig,
        core=core_len,
    )
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    # rows are appended to results/ext_cliffordt.jsonl as they finish, so an interrupted run resumes
    part = RESULTS / "ext_cliffordt.jsonl"
    rows = [json.loads(line) for line in open(part)] if part.exists() else []

    def key(kind, size, seed):
        return (kind, size if kind == "random" else tuple(size), seed)

    done = {key(r["kind"], r["size"], r["seed"]) for r in rows}
    jobs = [("random", L, s) for L in (4, 6, 8, 10, 12) for s in range(20)]
    jobs += [("planted", (c, L), s) for c in (5, 7, 9) for L in (60, 120) for s in range(10)]
    jobs = [j for j in jobs if key(*j) not in done]
    print(f"{len(done)} done, {len(jobs)} to run", flush=True)
    info = setup()
    with mp.get_context("fork").Pool(3) as p, open(part, "a") as f:
        for r in p.imap_unordered(job, jobs):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    json.dump(dict(ball=info, rows=rows), open(RESULTS / "ext_cliffordt.json", "w"), indent=1)
