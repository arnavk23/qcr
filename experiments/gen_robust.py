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


"""Robustness of the 60-circuit hardware-suite results: random seeds and an ablation.

Modes:
  seeds     For transpiler seeds t = 0..4: Qiskit L1/L3 (seed t) and ours (exact, calibrated) from each, with
            optimiser seed t. On the t = 0 inputs also ours from L3 with optimiser seeds 1..4, which isolates the
            optimiser's own randomness (the commuting shuffles) from Qiskit's layout. -> robust_seeds_<tag>.jsonl
  ablation  Ours (exact, calibrated) from L1 and from L3 (seed 0) with one part of the optimiser changed at a time:
            window shape and size, shuffles, database, meet in the middle, resynthesis, database radius, cost.
            -> robust_ablation_<tag>.jsonl

Every output of ours is verified against its input (statevector on the active qubits) and checked to be native.
Same circuits, devices and metrics as gen_bench.py. Rows are appended as they finish, so a run resumes.
"""

import argparse
import json
import multiprocessing as mp
import re
import time
import traceback

import runstatus
from gen_bench import CONFIGS, OUT, load_dbs, metrics, qk_to_gates, suite_circuits, verify

from qcr.general import Config, Optimizer
from qcr.hardware import coupling_list, heron, ion, qiskit_basis

SEEDS = range(5)
VARIANTS = {  # name -> Config overrides on top of exact_cal, patience 4 (the setting of gen_bench)
    "full": {},
    "contiguous": dict(max_jump=1),  # windows stop at the first gate they cannot take (no jumping)
    "w8": dict(w=8),
    "w4": dict(w=4),
    "no_shuffle": dict(patience=0),
    "no_db": dict(use_db=False),
    "no_mitm": dict(use_mitm=False),
    "no_resynth": dict(use_resynth=False),
    "radius3": dict(db_radius=3),
    "cost_count": dict(cost="count"),
    "cost_w2": dict(cost="w2"),
}
DBS = {}


def transpiled(qc, devname, seed):
    from qiskit import transpile

    dev = heron() if devname == "heron" else ion(qc.num_qubits)
    kw = dict(basis_gates=qiskit_basis(dev), seed_transpiler=seed)
    if dev.coupling is not None:
        from qiskit.transpiler import CouplingMap

        kw["coupling_map"] = CouplingMap(coupling_list(dev))
    out = {}
    for lv in (1, 3):
        t = time.perf_counter()
        out[f"L{lv}"] = (qk_to_gates(transpile(qc, optimization_level=lv, **kw)), time.perf_counter() - t)
    return dev, out


def ours(dev, devname, start, seed, **over):
    cfg = Config(**(CONFIGS["exact_cal"] | dict(patience=4) | over))
    r = Optimizer(dev, DBS[devname], cfg).run(start, seed=seed)
    return metrics(r.gates, dev) | dict(
        seconds=r.seconds,
        stats=r.stats,
        legal=all(dev.is_native(g) for g in r.gates),
        verify=verify(start, r.gates, "exact", False, 0.0),
    )


def seeds_job(args):
    name, devname, qasm_str, t = args
    from qiskit import QuantumCircuit

    dev, ins = transpiled(QuantumCircuit.from_qasm_str(qasm_str), devname, t)
    row = dict(name=name, device=devname, transpile_seed=t)
    for lv, (g, sec) in ins.items():
        row[f"qiskit_{lv}"] = metrics(g, dev) | dict(seconds=sec)
        row[f"ours_from{lv}"] = ours(dev, devname, g, seed=t)
    if t == 0:  # optimiser randomness alone, on fixed input
        for s in SEEDS[1:]:
            row[f"ours_fromL3_seed{s}"] = ours(dev, devname, ins["L3"][0], seed=s)
    return row


def ablation_job(args):
    name, devname, qasm_str = args
    from qiskit import QuantumCircuit

    dev, ins = transpiled(QuantumCircuit.from_qasm_str(qasm_str), devname, 0)
    row = dict(name=name, device=devname)
    for lv, (g, sec) in ins.items():
        row[f"qiskit_{lv}"] = metrics(g, dev) | dict(seconds=sec)
        for v, over in VARIANTS.items():
            row[f"{v}_from{lv}"] = ours(dev, devname, g, seed=0, **over)
    return row


def safe(fn):
    def run(args):
        try:
            return fn(args)
        except Exception:  # noqa: BLE001  recorded, the pool goes on, a resume retries it
            return dict(_error=traceback.format_exc()[-3000:], name=args[0], device=args[1])

    return run


def seeds_safe(a):
    return safe(seeds_job)(a)


def ablation_safe(a):
    return safe(ablation_job)(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["seeds", "ablation"])
    ap.add_argument("--devices", nargs="*", default=["heron", "ion"])
    ap.add_argument("--filter", default="")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    runstatus.start(f"robust_{a.mode}_{a.tag}")
    from qiskit import qasm2

    circs = {k: v for k, v in suite_circuits().items() if re.search(a.filter, k)}
    for d in a.devices:
        DBS[d] = load_dbs(d)
    part = OUT / f"robust_{a.mode}_{a.tag}.jsonl"
    rows = [json.loads(line) for line in open(part)] if part.exists() else []
    if a.mode == "seeds":
        done = {(r["name"], r["device"], r["transpile_seed"]) for r in rows}
        jobs = [(k, d, qasm2.dumps(v), t) for d in a.devices for k, v in circs.items() for t in SEEDS]
        jobs = [j for j in jobs if (j[0], j[1], j[3]) not in done]
        fn = seeds_safe
    else:
        done = {(r["name"], r["device"]) for r in rows}
        jobs = [(k, d, qasm2.dumps(v)) for d in a.devices for k, v in circs.items() if (k, d) not in done]
        fn = ablation_safe
    print(f"{len(circs)} circuits, {len(done)} rows done, {len(jobs)} to run", flush=True)
    runstatus.progress(len(done), len(done) + len(jobs))
    runstatus.phase("pool", jobs=len(jobs), procs=a.procs)
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=4) as p, open(part, "a") as f:
        for r in p.imap_unordered(fn, jobs):
            if "_error" in r:
                print(f"FAILED {r['name']} {r['device']}\n{r['_error']}", flush=True)
                runstatus.problem(f"{r['name']} {r['device']} failed:\n{r['_error'][-800:]}")
                continue
            bad = [
                k
                for k, v in r.items()
                if isinstance(v, dict) and "verify" in v and ((v["verify"] or 0) > 1e-8 or not v["legal"])
            ]
            if bad:
                runstatus.problem(f"{r['name']} {r['device']}: verification failed for {bad}")
            f.write(json.dumps(r) + "\n")
            f.flush()
            runstatus.row_done()
            print(json.dumps(dict(name=r["name"], dev=r["device"], seed=r.get("transpile_seed"))), flush=True)
