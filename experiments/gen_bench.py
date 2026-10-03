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


"""General benchmark: real/standard/random circuits on IBM Heron (heavy-hex,
calibrated) and an all-to-all trapped-ion model.

Pipelines (same device-native gate set and connectivity for all):
  qiskit_L1   qiskit transpile level 1      (also the input to "ours")
  qiskit_L3   qiskit transpile level 3
  tket        TKET peephole optimisation of the qiskit_L1 circuit + rebase
  ours        general optimiser on qiskit_L1 output
  L3+ours     general optimiser on qiskit_L3 output

Metrics: total gates, 2q gates, depth, estimated success probability
(product of gate fidelities), noisy-simulation TVD (<= 10 active qubits),
verification of every 'ours' output.
"""

import argparse
import json
import multiprocessing as mp
import re
import time
from math import pi
from pathlib import Path

import numpy as np
from common import CACHE, RESULTS

from qcr.gates import Gate, parse_qasm, to_qasm
from qcr.general import GRIDS, Config, Database, Optimizer
from qcr.hardware import coupling_list, heron, ion, qiskit_basis
from qcr.verify import apply

ROOT = Path(__file__).resolve().parents[1] / "external" / "benchmarks"
OUT = RESULTS / "general"
OUT.mkdir(exist_ok=True)


# ------------------------------------------------------------------ circuits
def suite_circuits(max_qubits=12, max_gates=1500):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFT, EfficientSU2

    circs = {}

    def add(name, qc):
        qc = qc.remove_final_measurements(inplace=False) if qc.num_clbits else qc
        qc = transpile(qc, basis_gates=["u3", "cx"], optimization_level=0)
        qc.data = [d for d in qc.data if d.operation.name not in ("barrier", "measure", "reset")]
        if any(d.operation.name in ("if_else", "while_loop", "for_loop", "switch_case") for d in qc.data):
            print("skip", name, "classical control flow")  # not a unitary: out of scope
            return
        if qc.num_qubits <= max_qubits and qc.size() <= max_gates and qc.size() > 0:
            circs[name] = qc

    CCZ = "gate ccz a,b,c { h c; ccx a,b,c; h c; }\n"
    for f in sorted((ROOT / "VOQC-benchmarks" / "Nam-benchmarks").rglob("*.qasm")):
        try:
            txt = f.read_text()
            if "ccz" in txt:
                txt = txt.replace('include "qelib1.inc";', 'include "qelib1.inc";\n' + CCZ, 1)
            add("nam/" + f.stem, QuantumCircuit.from_qasm_str(txt))
        except Exception as e:  # noqa: BLE001
            print("skip", f.name, type(e).__name__)
    for d in sorted((ROOT / "QASMBench" / "small").iterdir()):
        f = d / f"{d.name}.qasm"
        if f.exists():
            try:
                add("qasmbench/" + d.name, QuantumCircuit.from_qasm_file(str(f)))
            except Exception as e:  # noqa: BLE001
                print("skip", d.name, type(e).__name__)
    rng = np.random.default_rng(0)
    for n in (4, 6, 8):
        add(f"alg/qft_{n}", QFT(n))
    for n in (6, 8):  # QAOA MaxCut, p=2, random 3-regular graph
        import networkx as nx

        G = nx.random_regular_graph(3, n, seed=n)
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(2):
            g, b = rng.uniform(0, pi, 2)
            for u, v in G.edges():
                qc.rzz(2 * g, u, v)
            for q in range(n):
                qc.rx(2 * b, q)
        add(f"alg/qaoa_{n}", qc)
    for n in (6, 8):  # Heisenberg chain, 3 Trotter steps
        qc = QuantumCircuit(n)
        for _ in range(3):
            for q in range(n - 1):
                qc.rxx(0.2, q, q + 1), qc.ryy(0.2, q, q + 1), qc.rzz(0.2, q, q + 1)
            for q in range(n):
                qc.rz(0.1 * (q + 1), q)
        add(f"alg/heisenberg_{n}", qc)
    add("alg/efficientsu2_6", EfficientSU2(6, reps=3).assign_parameters(rng.uniform(-pi, pi, 48)))
    # Steane [[7,1,3]] encoder (Clifford)
    qc = QuantumCircuit(7)
    for c in (4, 5, 6):
        qc.h(c)
    for c, ts in ((0, (1, 2)), (4, (0, 1, 3)), (5, (0, 2, 3)), (6, (1, 2, 3))):
        for t in ts:
            qc.cx(c, t)
    add("alg/steane_encoder", qc)
    # non-Clifford random circuits (pi/4 grid and arbitrary angles)
    for n in (4, 6):
        for kind in ("grid", "cont"):
            qc = QuantumCircuit(n)
            for _ in range(150):
                if rng.random() < 0.3:
                    a, b = rng.choice(n, 2, replace=False)
                    qc.cx(int(a), int(b))
                else:
                    th = rng.choice([pi / 4, -pi / 4, pi / 2]) if kind == "grid" else rng.uniform(-pi, pi)
                    getattr(qc, rng.choice(["rx", "ry", "rz"]))(th, int(rng.integers(n)))
            add(f"random/{kind}_{n}q", qc)
    return circs


# ------------------------------------------------------------------ conversion
def qk_to_gates(qc, layout_compact=False):
    out = []
    for inst in qc.data:
        nm = inst.operation.name
        if nm in ("barrier", "measure", "delay", "reset"):
            continue
        qs = tuple(qc.find_bit(q).index for q in inst.qubits)
        prm = [float(x) for x in inst.operation.params]
        if nm in ("cz", "rxx", "rzz"):
            qs = tuple(sorted(qs))
        out.append(Gate(nm, qs, prm[0] if prm else 0.0))
    return out


def compact(gates):
    act = sorted({q for g in gates for q in g.qubits})
    m = {q: i for i, q in enumerate(act)}
    return [g.remap(m) for g in gates], len(act), m


def depth(gates):
    lvl = {}
    d = 0
    for g in gates:
        t = max(lvl.get(q, 0) for q in g.qubits) + 1
        for q in g.qubits:
            lvl[q] = t
        d = max(d, t)
    return d


def metrics(gates, dev):
    n2 = sum(len(g.qubits) == 2 for g in gates)
    return dict(
        total=len(gates),
        twoq=n2,
        oneq=len(gates) - n2,
        rz=sum(g.name == "rz" for g in gates),
        depth=depth(gates),
        success=dev.success_probability(gates),
    )


def verify(a, b, mode, state_aware, eps_bound):
    """Check b against a on their active qubits (union)."""
    act = sorted({q for g in a + b for q in g.qubits})
    m = {q: i for i, q in enumerate(act)}
    A, B, n = [g.remap(m) for g in a], [g.remap(m) for g in b], len(act)
    if n > 16:
        return None
    if state_aware:
        z = np.zeros(2**n, complex)
        z[0] = 1
        pa, pb = np.abs(apply(z, A, n)) ** 2, np.abs(apply(z, B, n)) ** 2
        return float(0.5 * np.abs(pa - pb).sum())
    rng = np.random.default_rng(1)
    worst = 0.0
    ph = None
    for _ in range(3):
        v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        v /= np.linalg.norm(v)
        x, y = apply(v, A, n), apply(v, B, n)
        if mode == "exact":
            worst = max(worst, float(np.linalg.norm(x - y)))
        else:
            o = np.vdot(x, y)
            ph = o / abs(o) if ph is None else ph
            worst = max(worst, float(np.linalg.norm(x * ph - y)))
    return worst


_TORINO_NM = None


def _torino_noise():
    global _TORINO_NM
    if _TORINO_NM is None:
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        _TORINO_NM = NoiseModel.from_backend(FakeTorino())
    return _TORINO_NM


def noisy_tvd(gates, dev_name, shots=4000):
    """TVD between ideal and noisy output distributions (<= 10 active qubits)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from qcr.verify import to_qiskit

    g, n, m = compact(gates)
    if n > 10:
        return None
    qc = to_qiskit(g, n)
    act = list(range(n))
    if dev_name == "heron":
        # calibrated FakeTorino noise of the actual physical qubits/pairs, remapped
        # onto the compacted register (simulating all 133 physical qubits is infeasible)
        full = _torino_noise()
        nm = NoiseModel(basis_gates=["rz", "sx", "x", "cz"])
        for gname, per_q in full._local_quantum_errors.items():
            if gname not in ("sx", "x", "cz"):
                continue
            for qs, err in per_q.items():
                if all(q in m for q in qs):
                    nm.add_quantum_error(err, gname, [m[q] for q in qs])
    else:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(3e-5, 1), ["rx", "ry", "rz"])
        nm.add_all_qubit_quantum_error(depolarizing_error(1e-3, 2), ["rxx"])
    from qiskit import ClassicalRegister

    cr = ClassicalRegister(len(act))
    qc.add_register(cr)
    for i, q in enumerate(act):
        qc.measure(q, cr[i])
    qc = qc.remove_final_measurements(inplace=False)
    qc.save_probabilities(act)
    from qiskit import transpile as _tp

    sim_i = AerSimulator(method="statevector")
    sim_n = AerSimulator(noise_model=nm, method="density_matrix")
    pi_ = sim_i.run(_tp(qc, sim_i, optimization_level=0)).result().data()["probabilities"]
    pn = sim_n.run(_tp(qc, sim_n, optimization_level=0)).result().data()["probabilities"]
    return float(0.5 * np.abs(np.asarray(pi_) - np.asarray(pn)).sum())


# ------------------------------------------------------------------ baselines
def tket_opt(gates, dev):
    from pytket import OpType
    from pytket.passes import AutoRebase, FullPeepholeOptimise, PeepholeOptimise2Q, RemoveRedundancies
    from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str

    n = max(q for g in gates for q in g.qubits) + 1
    c = circuit_from_qasm_str(to_qasm(gates, n))
    if dev.coupling is None:
        FullPeepholeOptimise(allow_swaps=False).apply(c)
        AutoRebase({OpType.Rx, OpType.Ry, OpType.Rz, OpType.XXPhase}).apply(c)
    else:
        PeepholeOptimise2Q(allow_swaps=False).apply(c)
        AutoRebase({OpType.Rz, OpType.SX, OpType.X, OpType.CZ}).apply(c)
    RemoveRedundancies().apply(c)
    g, _ = parse_qasm(circuit_to_qasm_str(c))
    g = [Gate(x.name, tuple(sorted(x.qubits)) if len(x.qubits) == 2 else x.qubits, x.theta) for x in g]
    return g


# ------------------------------------------------------------------ main
DBS = {}


def load_dbs(devname):
    topo = "tri" if devname == "ion" else "line"
    for kind in ("phase", "out", "state"):
        p = CACHE / f"gdb_{devname}_{topo}_{kind}.npz"
        if p.exists():
            DBS[(devname, kind)] = Database.load(p, GRIDS[devname])
        else:
            DBS[(devname, kind)] = Database(GRIDS[devname], topo, kind, 4, verbose=False)
    return {(k, topo): v for (d, k), v in DBS.items() if d == devname}


CONFIGS = {  # name -> Config kwargs
    "exact_count": dict(mode="exact", cost="count"),
    "exact_w2": dict(mode="exact", cost="w2"),
    "exact_cal": dict(mode="exact", cost="calibrated"),
    "state_cal": dict(mode="exact", state_aware=True, cost="calibrated"),
    "approx_cal": dict(mode="exact", eps=1e-3, cost="calibrated"),
}


def job(args):
    name, devname, qasm_str = args
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit.from_qasm_str(qasm_str)
    dev = heron() if devname == "heron" else ion(qc.num_qubits)
    dbs = {(k, t): v for (k, t), v in GLOBAL_DBS[devname].items()}
    kw = dict(basis_gates=qiskit_basis(dev), seed_transpiler=0)
    if dev.coupling is not None:
        from qiskit.transpiler import CouplingMap

        kw["coupling_map"] = CouplingMap(coupling_list(dev))
    row = dict(name=name, device=devname, n=qc.num_qubits, input_size=qc.size())
    t = time.perf_counter()
    l1 = qk_to_gates(transpile(qc, optimization_level=1, **kw))
    row["qiskit_L1"] = metrics(l1, dev)
    t = time.perf_counter()
    l3 = qk_to_gates(transpile(qc, optimization_level=3, **kw))
    row["qiskit_L3"] = metrics(l3, dev) | dict(seconds=time.perf_counter() - t)
    try:
        t = time.perf_counter()
        tk = tket_opt(l1, dev)
        legal = all(dev.is_native(g) for g in tk)
        row["tket"] = metrics(tk, dev) | dict(seconds=time.perf_counter() - t, legal=legal)
    except Exception as e:  # noqa: BLE001
        row["tket"] = dict(error=type(e).__name__)
    for cname, ckw in CONFIGS.items():
        for start_name, start in (("L1", l1), ("L3", l3)):
            if start_name == "L3" and cname not in ("exact_cal",):
                continue
            opt = Optimizer(dev, dbs, Config(**ckw, patience=4))
            r = opt.run(start, seed=0)
            key = f"ours_{cname}" + ("" if start_name == "L1" else "_fromL3")
            row[key] = metrics(r.gates, dev) | dict(
                seconds=r.seconds, err_bound=r.error_bound, stats=r.stats, legal=all(dev.is_native(g) for g in r.gates)
            )
            row[key]["verify"] = verify(start, r.gates, ckw.get("mode"), ckw.get("state_aware", False), r.error_bound)
    # noisy simulation for small circuits (objective selection evidence)
    act = len({q for g in l1 for q in g.qubits})
    if act <= 8:
        row["tvd"] = {}
        for tag, gl in (("qiskit_L3", l3), ("L1", l1)):
            row["tvd"][tag] = noisy_tvd(gl, devname)
        for cname in ("exact_count", "exact_w2", "exact_cal"):
            opt = Optimizer(dev, dbs, Config(**CONFIGS[cname], patience=4))
            row["tvd"]["ours_" + cname] = noisy_tvd(opt.run(l1, seed=0).gates, devname)
    print(
        json.dumps(
            dict(
                name=name,
                dev=devname,
                L3=row["qiskit_L3"]["twoq"],
                ours=row["ours_exact_cal"]["twoq"],
                ok=row["ours_exact_cal"]["verify"],
            )
        ),
        flush=True,
    )
    return row


GLOBAL_DBS = {}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*", default=["heron", "ion"])
    ap.add_argument("--filter", default="")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    circs = {k: v for k, v in suite_circuits().items() if re.search(a.filter, k)}
    print(len(circs), "circuits:", ", ".join(circs), flush=True)
    from qiskit import qasm2

    for d in a.devices:
        GLOBAL_DBS[d] = load_dbs(d)
    # rows are appended as they finish, so an interrupted run resumes where it stopped
    part = OUT / f"bench_{a.tag}.jsonl"
    rows = [json.loads(l) for l in open(part)] if part.exists() else []
    done = {(r["name"], r["device"]) for r in rows}
    jobs = [(k, d, qasm2.dumps(v)) for d in a.devices for k, v in circs.items() if (k, d) not in done]
    print(f"{len(done)} rows already done, {len(jobs)} to run", flush=True)
    with mp.get_context("fork").Pool(a.procs) as p, open(part, "a") as f:
        for r in p.imap_unordered(job, jobs):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    json.dump(rows, open(OUT / f"bench_{a.tag}.json", "w"), indent=1)
