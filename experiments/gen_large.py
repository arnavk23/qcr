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


"""Large-circuit benchmark: the Nam/VOQC circuits with 13..133 qubits on Heron and ion.

Modes:
  ours    qiskit L1/L3, TKET, ours (exact_cal) on L1 and on L3, with wall-clock times (large_<tag>.jsonl)
  bqskit  BQSKit L2/L3 on the same qiskit_L1 input, one compile at a time with a timeout (large_bqskit_<tag>.jsonl)

Same pipelines and metrics as gen_bench, minus the noisy simulation. Outputs on more than
16 active qubits are checked with MQT QCEC (decision-diagram alternating checker) instead of
statevectors; the result field says which check ran and what it returned.
"""

import argparse
import json
import multiprocessing as mp
import re
import time

import numpy as np
import runstatus
from gen_bench import CONFIGS, OUT, ROOT, load_dbs, metrics, qk_to_gates, tket_opt

from qcr.certify import replay
from qcr.general import Config, Optimizer
from qcr.hardware import coupling_list, heron, ion, qiskit_basis
from qcr.verify import to_qiskit

QCEC_TIMEOUT = 900.0
BQ_QCEC_TIMEOUT = 300.0  # BQSKit outputs over 24 qubits: QCEC approx gave no_information even at 19 qubits
SV_MAX = 24  # dense statevector check up to here (256 MB per state)
SIMS = 32


def large_circuits(min_qubits=13, max_qubits=133, max_gates=6000):
    """Nam benchmarks in the size window, decomposed to u3/cx like suite_circuits."""
    from qiskit import QuantumCircuit, transpile

    CCZ = "gate ccz a,b,c { h c; ccx a,b,c; h c; }\n"
    circs = {}
    for f in sorted((ROOT / "VOQC-benchmarks" / "Nam-benchmarks").rglob("*.qasm")):
        txt = f.read_text()
        if "ccz" in txt:
            txt = txt.replace('include "qelib1.inc";', 'include "qelib1.inc";\n' + CCZ, 1)
        try:
            qc = QuantumCircuit.from_qasm_str(txt)
        except Exception as e:  # noqa: BLE001
            print("skip", f.name, type(e).__name__)
            continue
        if not min_qubits <= qc.num_qubits <= max_qubits:
            continue
        qc = transpile(qc, basis_gates=["u3", "cx"], optimization_level=0)
        if 0 < qc.size() <= max_gates:
            circs["nam/" + f.stem] = qc
    return dict(sorted(circs.items(), key=lambda kv: kv[1].size()))


def qcec_check(qa, qb, approx=False, timeout=QCEC_TIMEOUT):
    """_qcec_check in a child process, killed after its own timeouts plus a margin: QCEC's timeout is not
    always honoured (on a 30-qubit BQSKit output its approximate DD check ran for hours)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_qcec_child, args=(qa, qb, approx, q, timeout))
    t = time.perf_counter()
    p.start()
    p.join(2 * timeout + 300)
    if p.is_alive():
        p.kill()
        p.join()
        return dict(
            method="qcec_approx" if approx else "qcec",
            result="no_information",
            hard_timeout=True,
            seconds=time.perf_counter() - t,
        )
    try:
        return q.get(timeout=10)
    except Exception:  # noqa: BLE001  child crashed before reporting
        return dict(method="qcec", result="no_information", exitcode=p.exitcode)


def _qcec_child(qa, qb, approx, q, timeout):
    q.put(_qcec_check(qa, qb, approx, timeout))


def _qcec_check(qa, qb, approx=False, timeout=QCEC_TIMEOUT):
    """Exact DD check first (or process fidelity >= 1 - 1e-6 for BQSKit's numerically synthesised outputs).
    If that times out, fall back to QCEC's simulation checker with random stabilizer stimuli, which is a
    probabilistic check of the same kind as the statevector one (result 'probably_equivalent')."""
    from mqt import qcec

    kw = dict(check_approximate_equivalence=True, approximate_checking_threshold=1 - 1e-6) if approx else {}
    t = time.perf_counter()
    r = qcec.verify(qa, qb, run_zx_checker=False, run_simulation_checker=False, timeout=timeout, **kw)
    out = dict(
        method="qcec_approx" if approx else "qcec",
        result=str(r.equivalence).split(".")[-1],
        seconds=time.perf_counter() - t,
    )
    if out["result"] == "no_information" and not approx:  # QCEC's simulation checker has no approximate mode
        t = time.perf_counter()
        r = qcec.verify(
            qa,
            qb,
            run_zx_checker=False,
            run_alternating_checker=False,
            max_sims=SIMS,
            state_type=qcec.pyqcec.StateType.stabilizer,
            timeout=timeout,
            **kw,
        )
        out |= dict(sim_result=str(r.equivalence).split(".")[-1], sim_seconds=time.perf_counter() - t, sims=SIMS)
    return out


def check(a, b):
    """Exact check of b against a on their active qubits (same physical wires)."""
    act = sorted({q for g in a + b for q in g.qubits})
    if len(act) <= SV_MAX:
        return dict(method="statevector", err=verify_sv_wide(a, b))
    m = {q: i for i, q in enumerate(act)}
    n = len(act)
    return qcec_check(to_qiskit([g.remap(m) for g in a], n), to_qiskit([g.remap(m) for g in b], n))


def verify_sv_wide(a, b):
    """gen_bench.verify without its 16-qubit cap (3 random states, exact mode)."""
    from qcr.verify import apply

    act = sorted({q for g in a + b for q in g.qubits})
    m = {q: i for i, q in enumerate(act)}
    A, B, n = [g.remap(m) for g in a], [g.remap(m) for g in b], len(act)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(3):
        v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        v /= np.linalg.norm(v)
        worst = max(worst, float(np.linalg.norm(apply(v, A, n) - apply(v, B, n))))
    return worst


def check_mapped(a, b, act, pi, pf):
    """Check P_f A = B P_i up to phase (BQSKit output with its initial/final qubit permutations)."""
    from ext_bqskit import verify_mapped
    from qiskit import QuantumCircuit
    from qiskit.synthesis import synth_permutation_basic

    n = len(act)
    if n <= SV_MAX:  # QCEC's approximate DD check gives no_information here after its full timeout
        return dict(method="statevector", err=verify_mapped(a, b, act, pi, pf, max_n=SV_MAX))
    m = {q: i for i, q in enumerate(act)}
    qa = QuantumCircuit(n)
    qa.compose(to_qiskit([g.remap(m) for g in a], n), inplace=True)
    qa.compose(synth_permutation_basic(list(np.argsort(pf))), inplace=True)
    qb = QuantumCircuit(n)
    qb.compose(synth_permutation_basic(list(np.argsort(pi))), inplace=True)
    qb.compose(to_qiskit([g.remap(m) for g in b], n), inplace=True)
    return qcec_check(qa, qb, approx=True, timeout=BQ_QCEC_TIMEOUT)


GATES = OUT / "large_gates"


def gate_file(name, devname):
    return GATES / f"{name.replace('/', '_')}__{devname}.json.gz"


def save_gates(name, devname, d):
    import gzip

    GATES.mkdir(exist_ok=True)
    with gzip.open(gate_file(name, devname), "wt") as f:
        json.dump({k: [[g.name, list(g.qubits), g.theta] for g in v] for k, v in d.items()}, f)


def load_gates(name, devname):
    import gzip

    from qcr.gates import Gate

    with gzip.open(gate_file(name, devname), "rt") as f:
        return {k: [Gate(n, tuple(q), t) for n, q, t in v] for k, v in json.load(f).items()}


def device_and_kw(devname, n):
    dev = heron() if devname == "heron" else ion(n)
    kw = dict(basis_gates=qiskit_basis(dev), seed_transpiler=0)
    if dev.coupling is not None:
        from qiskit.transpiler import CouplingMap

        kw["coupling_map"] = CouplingMap(coupling_list(dev))
    return dev, kw


def l1_input(qc, devname):
    from qiskit import transpile

    dev, kw = device_and_kw(devname, qc.num_qubits)
    t = time.perf_counter()
    l1 = qk_to_gates(transpile(qc, optimization_level=1, **kw))
    return dev, kw, l1, time.perf_counter() - t


def job(args):
    name, devname, qasm_str = args
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit.from_qasm_str(qasm_str)
    dev, kw, l1, sec1 = l1_input(qc, devname)
    row = dict(name=name, device=devname, n=qc.num_qubits, input_size=qc.size())
    row["qiskit_L1"] = metrics(l1, dev) | dict(seconds=sec1)
    t = time.perf_counter()
    l3 = qk_to_gates(transpile(qc, optimization_level=3, **kw))
    row["qiskit_L3"] = metrics(l3, dev) | dict(seconds=time.perf_counter() - t)
    try:
        t = time.perf_counter()
        tk = tket_opt(l1, dev)
        row["tket"] = metrics(tk, dev) | dict(seconds=time.perf_counter() - t, legal=all(dev.is_native(g) for g in tk))
    except Exception as e:  # noqa: BLE001
        row["tket"] = dict(error=type(e).__name__)
    saved = dict(qiskit_L1=l1, qiskit_L3=l3)
    for key, start in (("ours_exact_cal", l1), ("ours_exact_cal_fromL3", l3)):
        r = Optimizer(dev, DBS[devname], Config(**CONFIGS["exact_cal"], patience=4, log=True)).run(start, seed=0)
        t = time.perf_counter()
        cert = replay(start, r.log, r.gates) | dict(seconds=time.perf_counter() - t)
        row[key] = metrics(r.gates, dev) | dict(
            seconds=r.seconds, stats=r.stats, legal=all(dev.is_native(g) for g in r.gates), certificate=cert
        )
        saved[key] = r.gates
    save_gates(name, devname, saved)  # verified afterwards, one circuit at a time (mode 'verify')
    print(
        json.dumps(
            dict(
                name=name,
                dev=devname,
                n=row["n"],
                L1=row["qiskit_L1"]["twoq"],
                L3=row["qiskit_L3"]["twoq"],
                ours=row["ours_exact_cal"]["twoq"],
                sec=round(row["ours_exact_cal"]["seconds"]),
                cert=row["ours_exact_cal"]["certificate"]["ok"],
            )
        ),
        flush=True,
    )
    return row


def safe_job(args):
    """job, but a failure comes back as a record instead of ending the whole pool run."""
    try:
        return job(args)
    except Exception:  # noqa: BLE001
        import traceback

        return dict(_error=traceback.format_exc()[-3000:], name=args[0], device=args[1])


def bqskit_job(name, devname, qc, levels, comp, timeout, cut=None):
    """cut: {level: input size of a smaller circuit that already timed out}; such levels are not attempted."""
    from ext_bqskit import bqskit_opt

    dev, _, l1, _ = l1_input(qc, devname)
    row = dict(name=name, device=devname, n=qc.num_qubits, input_size=qc.size(), qiskit_L1=metrics(l1, dev))
    for lv in levels:
        key = f"bqskit_L{lv}"
        if cut and lv in cut:
            row[key] = dict(error="skipped", reason=f"not attempted: a {cut[lv]}-gate input already timed out")
            continue
        runstatus.phase("compile", limit=timeout + 600, job=f"{name} {devname} L{lv}")
        t = time.perf_counter()
        try:
            g, pi, pf, act = bqskit_opt(l1, dev, lv, compiler=TimedCompiler(comp, timeout))
        except TimeoutError:
            row[key] = dict(error="timeout", seconds=time.perf_counter() - t)
            continue
        except Exception as e:  # noqa: BLE001
            row[key] = dict(error=f"{type(e).__name__}: {e}"[:300], seconds=time.perf_counter() - t)
            continue
        sec = time.perf_counter() - t
        runstatus.phase("check", limit=2 * BQ_QCEC_TIMEOUT + 1800, job=f"{name} {devname} L{lv}", active=len(act))
        try:
            ver = check_mapped(l1, g, act, pi, pf)
        except Exception as e:  # noqa: BLE001  a failed check is recorded, it must not end the run
            ver = dict(error=f"{type(e).__name__}: {e}"[:300])
        row[key] = metrics(g, dev) | dict(seconds=sec, legal=all(dev.is_native(x) for x in g), verify=ver)
    print(
        json.dumps(
            {"name": name, "dev": devname, "n": row["n"]}
            | {k: row[k].get("twoq", row[k].get("error")) for k in row if k.startswith(("qiskit", "bqskit"))}
        ),
        flush=True,
    )
    return row


def TimedCompiler(comp, timeout):
    """Make comp.compile give up after `timeout` seconds (bqskit.compile insists on a real Compiler)."""
    from bqskit.compiler.status import CompilationStatus

    def compile(circuit, workflow, request_data=False, *rest):
        tid = comp.submit(circuit, workflow, request_data)
        t0 = time.perf_counter()
        while comp.status(tid) != CompilationStatus.DONE:
            if time.perf_counter() - t0 > timeout:
                comp.cancel(tid)
                raise TimeoutError
            time.sleep(2)
        return comp.result(tid)

    comp.compile = compile
    return comp


DBS = {}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ours", "bqskit", "verify", "recert", "list"])
    ap.add_argument("--devices", nargs="*", default=["heron", "ion"])
    ap.add_argument("--filter", default="")
    ap.add_argument("--max-gates", type=int, default=6000)
    ap.add_argument("--bq-max-qubits", type=int, default=40)
    ap.add_argument("--bq-timeout", type=float, default=7200)
    ap.add_argument("--levels", nargs="*", type=int, default=[2, 3])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    if a.mode != "list":
        runstatus.start(f"large_{a.mode}_{a.tag}")
    circs = {k: v for k, v in large_circuits(max_gates=a.max_gates).items() if re.search(a.filter, k)}
    print(len(circs), "circuits:", ", ".join(f"{k}({v.num_qubits}q,{v.size()})" for k, v in circs.items()), flush=True)
    if a.mode == "list":
        raise SystemExit
    from qiskit import qasm2

    if a.mode == "recert":  # re-run (deterministic) and re-check the outputs whose certificate failed
        part = OUT / f"large_recert_{a.tag}.jsonl"
        done = {(r["name"], r["device"], r["key"]) for r in map(json.loads, open(part))} if part.exists() else set()
        rows = [json.loads(line) for line in open(OUT / f"large_{a.tag}.jsonl")]
        for d in a.devices:
            DBS[d] = load_dbs(d)
        with open(part, "a") as f:
            for r in rows:
                for key, start in (("ours_exact_cal", "qiskit_L1"), ("ours_exact_cal_fromL3", "qiskit_L3")):
                    if r[key]["certificate"]["ok"] or (r["name"], r["device"], key) in done:
                        continue
                    if not re.search(a.filter, r["name"]):
                        continue
                    runstatus.phase("recert", limit=8 * 3600, job=f"{r['name']} {r['device']} {key}")
                    gs = load_gates(r["name"], r["device"])
                    dev = heron() if r["device"] == "heron" else ion(r["n"])
                    res = Optimizer(dev, DBS[r["device"]], Config(**CONFIGS["exact_cal"], patience=4, log=True)).run(
                        gs[start], seed=0
                    )
                    t = time.perf_counter()
                    cert = replay(gs[start], res.log, gs[key]) | dict(seconds=time.perf_counter() - t)
                    v = dict(
                        name=r["name"], device=r["device"], key=key, same_output=res.gates == gs[key], certificate=cert
                    )
                    print(json.dumps(v), flush=True)
                    f.write(json.dumps(v) + "\n")
                    f.flush()
                    runstatus.row_done()
        raise SystemExit
    if a.mode == "verify":  # statevector re-check of the saved outputs, smallest first, one at a time
        part = OUT / f"large_verify_{a.tag}.jsonl"
        done = {(r["name"], r["device"], r["key"]) for r in map(json.loads, open(part))} if part.exists() else set()
        rows = [json.loads(l) for l in open(OUT / f"large_{a.tag}.jsonl")]
        with open(part, "a") as f:
            for r in sorted(rows, key=lambda r: r["n"]):
                if r["n"] > SV_MAX:
                    continue
                gs = load_gates(r["name"], r["device"])
                for key, start in (("ours_exact_cal", "qiskit_L1"), ("ours_exact_cal_fromL3", "qiskit_L3")):
                    if (r["name"], r["device"], key) in done:
                        continue
                    act = len({q for g in gs[start] + gs[key] for q in g.qubits})
                    if act > SV_MAX:  # routing can pull in more physical qubits than the circuit has
                        continue
                    runstatus.phase("statevector", limit=4 * 3600, job=f"{r['name']} {r['device']} {key}", active=act)
                    t = time.perf_counter()
                    v = dict(
                        name=r["name"],
                        device=r["device"],
                        key=key,
                        n=r["n"],
                        active=act,
                        method="statevector",
                        err=verify_sv_wide(gs[start], gs[key]),
                        seconds=0.0,
                    )
                    v["seconds"] = time.perf_counter() - t
                    print(json.dumps(v), flush=True)
                    f.write(json.dumps(v) + "\n")
                    f.flush()
                    runstatus.row_done()
        raise SystemExit
    if a.mode == "ours":
        for d in a.devices:
            DBS[d] = load_dbs(d)
        part = OUT / f"large_{a.tag}.jsonl"
        done = {(r["name"], r["device"]) for r in map(json.loads, open(part))} if part.exists() else set()
        jobs = [(k, d, qasm2.dumps(v)) for k, v in circs.items() for d in a.devices if (k, d) not in done]
        print(f"{len(done)} rows done, {len(jobs)} to run", flush=True)
        runstatus.progress(len(done), len(done) + len(jobs))
        runstatus.phase("pool", jobs=len(jobs), procs=a.procs)
        with mp.get_context("fork").Pool(a.procs, maxtasksperchild=1) as p, open(part, "a") as f:
            for r in p.imap_unordered(safe_job, jobs):
                if "_error" in r:  # recorded, not written as a row, so a resume retries it
                    print(f"FAILED {r['name']} {r['device']}\n{r['_error']}", flush=True)
                    runstatus.problem(f"{r['name']} {r['device']} failed:\n{r['_error'][-800:]}")
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                runstatus.row_done()
    else:
        from bqskit.compiler import Compiler

        part = OUT / f"large_bqskit_{a.tag}.jsonl"
        done = {(r["name"], r["device"]) for r in map(json.loads, open(part))} if part.exists() else set()
        jobs = [
            (k, d, v)
            for k, v in circs.items()
            for d in a.devices
            if v.num_qubits <= a.bq_max_qubits and (k, d) not in done
        ]
        print(f"{len(done)} rows done, {len(jobs)} to run", flush=True)
        runstatus.progress(len(done), len(done) + len(jobs))
        # once a (device, level) times out, larger inputs on it are not attempted (jobs run smallest first)
        cut = {}
        for r in map(json.loads, open(part)) if part.exists() else []:
            for lv in a.levels:
                if r.get(f"bqskit_L{lv}", {}).get("error") == "timeout" and r["name"] in circs:
                    size = circs[r["name"]].size()
                    cut[r["device"], lv] = min(size, cut.get((r["device"], lv), size))
        with Compiler(num_workers=a.procs) as comp, open(part, "a") as f:
            for k, d, v in jobs:
                skip = {lv: s for (dd, lv), s in cut.items() if dd == d and v.size() >= s}
                row = bqskit_job(k, d, v, a.levels, comp, a.bq_timeout, skip)
                for lv in a.levels:
                    if row[f"bqskit_L{lv}"].get("error") == "timeout":
                        cut[d, lv] = min(v.size(), cut.get((d, lv), v.size()))
                f.write(json.dumps(row) + "\n")
                f.flush()
                runstatus.row_done()
