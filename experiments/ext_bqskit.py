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


"""BQSKit baseline on the hardware suite (same circuits and devices as gen_bench).

Modes:
  suite  BQSKit L2/L3 on the qiskit_L1 circuits (bqskit_<tag>.jsonl)
  stack  BQSKit L3, then our optimiser (exact_cal) on its output (bqskit_stack_<tag>.jsonl)
  njp    BQSKit L4 on the 100 + 100 four-qubit circuits of the NJP protocol (bqskit_njp_<tag>.jsonl)

BQSKit compiles the qiskit_L1 circuit (the same input "ours" and TKET get),
restricted to its active qubits: on Heron the model is the coupling graph induced
on those physical qubits (dead couplers already removed), on ions all-to-all.
BQSKit is free to re-place and re-route within that subgraph; the output is
mapped back onto the physical qubits so ESP uses the same calibration.
Outputs are checked up to global phase and BQSKit's final qubit permutation.
"""

import argparse
import json
import re
import time

import numpy as np
from gen_bench import OUT, metrics, qk_to_gates, suite_circuits

from qcr.gates import Gate
from qcr.hardware import coupling_list, heron, ion, qiskit_basis
from qcr.verify import apply

TO_BQ = {"rz": "RZGate", "sx": "SXGate", "x": "XGate", "cz": "CZGate", "rx": "RXGate", "ry": "RYGate", "rxx": "RXXGate"}
FROM_BQ = {v: k for k, v in TO_BQ.items()} | {"SqrtXGate": "sx"}  # bqskit's SXGate is an alias


def bqskit_opt(gates, dev, level, seed=0, compiler=None, gate_set=None):
    """Return (gates on physical qubits, initial mapping, final mapping, active qubits)."""
    import bqskit.ir.gates as G
    from bqskit import Circuit, MachineModel, compile

    act = sorted({q for g in gates for q in g.qubits})
    m = {q: i for i, q in enumerate(act)}
    n = len(act)
    c = Circuit(n)
    for g in gates:
        gt = getattr(G, TO_BQ[g.name])()
        c.append_gate(gt, [m[q] for q in g.qubits], [g.theta] if gt.num_params else [])
    if gate_set is not None:  # all-to-all with an explicit gate set (NJP protocol)
        model = MachineModel(n, gate_set={getattr(G, TO_BQ[x])() for x in gate_set})
    elif dev.coupling is None:
        model = MachineModel(n, gate_set={G.RXXGate(), G.RXGate(), G.RYGate(), G.RZGate()})
    else:
        edges = sorted({(m[a], m[b]) for a, b in coupling_list(dev) if a in m and b in m and a < b})
        model = MachineModel(n, edges, gate_set={G.CZGate(), G.RZGate(), G.SXGate(), G.XGate()})
    out, pi, pf = compile(c, model, optimization_level=level, seed=seed, with_mapping=True, compiler=compiler)
    res = []
    for op in out:
        nm = FROM_BQ[type(op.gate).__name__]
        qs = tuple(act[q] for q in op.location)
        if len(qs) == 2:
            qs = tuple(sorted(qs))
        res.append(Gate(nm, qs, float(op.params[0]) if op.params else 0.0))
    return res, list(pi), list(pf), act


def verify_mapped(a, b, act, pi, pf, trials=3):
    """max ||phase*P_f A v - B P_i v|| over random v (A on logical, B on physical indices)."""
    n = len(act)
    if n > 16:
        return None
    m = {q: i for i, q in enumerate(act)}
    A, B = [g.remap(m) for g in a], [g.remap(m) for g in b]
    rng = np.random.default_rng(1)
    worst, ph = 0.0, None

    def place(v, perm):  # logical qubit i -> physical qubit perm[i]
        return np.moveaxis(v.reshape([2] * n), list(range(n)), list(perm)).reshape(-1)

    for _ in range(trials):
        v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        v /= np.linalg.norm(v)
        x, y = place(apply(v, A, n), pf), apply(place(v, pi), B, n)
        o = np.vdot(x, y)
        ph = o / abs(o) if ph is None else ph
        worst = max(worst, float(np.linalg.norm(x * ph - y)))
    return worst


def job(args, compiler=None, stack=False):
    name, devname, qasm_str, levels = args
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit.from_qasm_str(qasm_str)
    dev = heron() if devname == "heron" else ion(qc.num_qubits)
    kw = dict(basis_gates=qiskit_basis(dev), seed_transpiler=0)
    if dev.coupling is not None:
        from qiskit.transpiler import CouplingMap

        kw["coupling_map"] = CouplingMap(coupling_list(dev))
    l1 = qk_to_gates(transpile(qc, optimization_level=1, **kw))  # identical to gen_bench's qiskit_L1
    row = dict(name=name, device=devname, n=qc.num_qubits, qiskit_L1=metrics(l1, dev))
    for lv in levels:
        key = f"bqskit_L{lv}"
        try:
            t = time.perf_counter()
            g, pi, pf, act = bqskit_opt(l1, dev, lv, compiler=compiler)
            row[key] = metrics(g, dev) | dict(
                seconds=time.perf_counter() - t,
                legal=all(dev.is_native(x) for x in g),
                verify=verify_mapped(l1, g, act, pi, pf),
                permuted=pi != list(range(len(act))) or pf != list(range(len(act))),
            )
        except Exception as e:  # noqa: BLE001
            row[key] = dict(error=f"{type(e).__name__}: {e}"[:300])
            continue
        if stack:  # our optimiser on BQSKit's output (exact relative to it)
            row[key]["gates"] = [[x.name, list(x.qubits), x.theta] for x in g]
            ckw = CONFIGS["exact_cal"]
            r = Optimizer(dev, DBS[devname], Config(**ckw, patience=4)).run(g, seed=0)
            row[key + "+ours"] = metrics(r.gates, dev) | dict(
                seconds=r.seconds,
                legal=all(dev.is_native(x) for x in r.gates),
                verify=verify_exact(g, r.gates, "exact", False, r.error_bound),
            )
    print(json.dumps({"name": name, "dev": devname} | {k: row[k].get("twoq", row[k].get("error")) for k in row
                                                        if k.startswith(("qiskit", "bqskit"))}), flush=True)
    return row


def njp_job(kind, seed, compiler, level=4):
    """BQSKit on one NJP circuit: all-to-all, the paper's gate set, continuous angles allowed."""
    from qcr.gates import counts, make_pool, random_circuit

    circ = random_circuit(make_pool(kind, 4), 300, np.random.default_rng(10_000 + seed))  # = exp_quality NJP seeds
    gs = ("rx", "ry", "rz", "rxx") if kind == "ion" else ("rx", "rz", "cz")
    t = time.perf_counter()
    g, pi, pf, act = bqskit_opt(circ, None, level, seed=seed, compiler=compiler, gate_set=gs)
    sec = time.perf_counter() - t
    return dict(kind=kind, seed=seed, level=level, original=counts(circ), counts=counts(g), total=len(g),
                twoq=sum(len(x.qubits) == 2 for x in g), seconds=sec, verify=verify_mapped(circ, g, act, pi, pf),
                permuted=pi != list(range(len(act))) or pf != list(range(len(act))))


DBS = {}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*", default=["heron", "ion"])
    ap.add_argument("--filter", default="")
    ap.add_argument("--levels", nargs="*", type=int, default=[2, 3])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--mode", choices=["suite", "stack", "njp"], default="suite")
    ap.add_argument("--n", type=int, default=100, help="njp: circuits per kind")
    a = ap.parse_args()
    from bqskit.compiler import Compiler

    if a.mode == "njp":
        part = OUT / f"bqskit_njp_{a.tag}.jsonl"
        done = {(r["kind"], r["seed"]) for r in map(json.loads, open(part))} if part.exists() else set()
        jobs = [(k, s) for s in range(a.n) for k in ("ion", "nisq") if (k, s) not in done]
        print(f"njp: {len(done)} done, {len(jobs)} to run", flush=True)
        with Compiler(num_workers=a.workers) as comp, open(part, "a") as f:
            for k, s in jobs:
                r = njp_job(k, s, comp)
                print(json.dumps({x: r[x] for x in ("kind", "seed", "total", "twoq", "seconds", "verify")}), flush=True)
                f.write(json.dumps(r) + "\n")
                f.flush()
        raise SystemExit
    if a.mode == "stack":
        from gen_bench import CONFIGS, load_dbs
        from gen_bench import verify as verify_exact

        from qcr.general import Config, Optimizer

        globals().update(CONFIGS=CONFIGS, verify_exact=verify_exact, Config=Config, Optimizer=Optimizer)
        for d in a.devices:
            DBS[d] = load_dbs(d)
        a.levels = [3]
    circs = {k: v for k, v in suite_circuits().items() if re.search(a.filter, k)}
    from qiskit import qasm2

    part = OUT / (f"bqskit_stack_{a.tag}.jsonl" if a.mode == "stack" else f"bqskit_{a.tag}.jsonl")
    rows = [json.loads(l) for l in open(part)] if part.exists() else []
    done = {(r["name"], r["device"]) for r in rows}
    jobs = [(k, d, qasm2.dumps(v), a.levels) for d in a.devices for k, v in circs.items() if (k, d) not in done]
    print(f"{len(circs)} circuits, {len(done)} rows done, {len(jobs)} to run", flush=True)
    # circuits run one at a time; BQSKit parallelises each compile over its own worker runtime
    with Compiler(num_workers=a.workers) as comp, open(part, "a") as f:
        for jb in jobs:
            f.write(json.dumps(job(jb, comp, stack=a.mode == "stack")) + "\n")
            f.flush()
