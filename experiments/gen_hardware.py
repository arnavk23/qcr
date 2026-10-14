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


"""Real-hardware experiment on an IBM Heron backend (or a fake backend with its noise model, for a dry run).

  plan     Fetch the backend's current calibration, pick small circuits of the 60-circuit suite, compile each with
           Qiskit (noise-aware target, levels 1 and 3) and run ours (exact, calibrated cost from the SAME calibration)
           on top. Every ours output is verified against its input. Saves the ISA circuits and their predicted ESP.
  run      Submit every (circuit, pipeline) pair --repeats times in one Batch, each repeat in a freshly shuffled
           order (interleaved, so drift hits all pipelines alike), --shots shots each. Results are appended per job.
  analyze  Per circuit and pipeline: success probability (circuits whose ideal output is one bitstring) or Hellinger
           fidelity to the ideal distribution, mean and sd over repeats, and the paired comparison with Qiskit L3.

Output: results/hardware/<backend>_<tag>/{plan.json, circuits/*.qasm, runs.jsonl, analysis.json}.
Account: QiskitRuntimeService() with a saved account (save it yourself; this script never reads a token).
Dry run: --fake torino (FakeTorino calibration and noise model, local simulation, no account needed).
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import runstatus
from gen_bench import load_dbs, metrics, qk_to_gates, suite_circuits, verify

from qcr.general import Config, Optimizer
from qcr.hardware import coupled_target, device_from_backend, fake_backend

ROOT = Path(__file__).resolve().parents[1]
PIPES = ("L3", "L3_ours", "L1_ours")


def backend_of(a):
    if a.fake:
        return fake_backend(a.fake)
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService().backend(a.backend)


def outdir(a):
    return ROOT / "results" / "hardware" / f"{a.fake and 'fake_' + a.fake or a.backend}_{a.tag}"


def ideal_probs(qc):
    """Ideal output distribution of a measure-free logical circuit, keys as Qiskit count bitstrings."""
    from qiskit.quantum_info import Statevector

    return {k: float(v) for k, v in Statevector(qc).probabilities_dict().items() if v > 1e-12}


def to_isa(gates, nq, measures):
    """Our gates (physical qubits) plus the original measurements -> an ISA circuit for the backend."""
    from qiskit import ClassicalRegister, QuantumCircuit

    qc = QuantumCircuit(nq)
    creg = ClassicalRegister(len(measures), "meas")
    qc.add_register(creg)
    for g in gates:
        if g.name in ("sx", "x"):
            getattr(qc, g.name)(g.qubits[0])
        elif g.name == "cz":
            qc.cz(*g.qubits)
        else:
            qc.rz(g.theta, g.qubits[0])
    for q, c in measures:
        qc.measure(q, creg[c])
    return qc


def plan(a):
    from qiskit import qasm2, transpile

    b = backend_of(a)
    dev = device_from_backend(b)
    tgt = coupled_target(b)
    dbs = load_dbs("heron")
    out = outdir(a)
    (out / "circuits").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, qc in suite_circuits().items():
        if not a.min_qubits <= qc.num_qubits <= a.max_qubits:
            continue
        from qiskit import QuantumCircuit

        logical = QuantumCircuit(qc.num_qubits)  # drop any classical bits and measurements the source circuit has
        for ins in qc.data:
            if ins.operation.name not in ("measure", "barrier") and not ins.clbits:
                logical.append(ins.operation, [qc.find_bit(q).index for q in ins.qubits])
        ideal = ideal_probs(logical)
        m = logical.copy()
        m.measure_all()
        row = dict(name=name, n=qc.num_qubits, deterministic=max(ideal.values()) > 0.99, ideal=ideal, pipes={})
        starts = {}
        for lv in (1, 3):
            t = transpile(m, target=tgt, optimization_level=lv, seed_transpiler=a.seed)
            meas = [
                (t.find_bit(i.qubits[0]).index, t.find_bit(i.clbits[0]).index) for i in t.data if i.name == "measure"
            ]
            starts[lv] = (qk_to_gates(t), meas)
        for p in PIPES:
            lv = int(p[1])
            g, meas = starts[lv]
            if p.endswith("_ours"):
                r = Optimizer(dev, dbs, Config(mode="exact", cost="calibrated", patience=4)).run(g, seed=0)
                err = verify(g, r.gates, "exact", False, 0.0)
                assert err is not None and err < 1e-8, (name, p, err)
                g = r.gates
            isa = to_isa(g, b.num_qubits, meas)
            (out / "circuits" / f"{name.replace('/', '_')}__{p}.qasm").write_text(qasm2.dumps(isa))
            row["pipes"][p] = metrics(g, dev) | dict(esp_readout=dev.success_probability(g, readout=True))
        row["esp_L3"] = row["pipes"]["L3"]["esp_readout"]
        rows.append(row)
        print(name, row["n"], {p: (v["twoq"], v["total"], round(v["esp_readout"], 3)) for p, v in row["pipes"].items()})
    # keep circuits whose output is not pure noise at L3, most informative (deterministic, highest ESP) first
    rows = [r for r in rows if r["esp_L3"] >= a.min_esp and r["pipes"]["L3"]["twoq"] >= a.min_twoq]
    rows.sort(key=lambda r: (not r["deterministic"], -r["esp_L3"]))
    rows = rows[: a.max_circuits]
    json.dump(dict(backend=b.name, calibrated=str(getattr(b.target, "_calibrated", "")), seed=a.seed, rows=rows,
                   created=time.strftime("%Y-%m-%d %H:%M")), open(out / "plan.json", "w"), indent=1)  # fmt: skip
    print(f"{len(rows)} circuits x {len(PIPES)} pipelines planned in {out}")


def run(a):
    from qiskit import qasm2
    from qiskit_ibm_runtime import Batch, SamplerV2

    out = outdir(a)
    P = json.load(open(out / "plan.json"))
    b = backend_of(a)
    pubs = []
    for r in P["rows"]:
        for p in PIPES:
            qc = qasm2.load(str(out / "circuits" / f"{r['name'].replace('/', '_')}__{p}.qasm"),
                            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)  # fmt: skip
            pubs.append(((r["name"], p), qc))
    done = {json.loads(line)["repeat"] for line in open(out / "runs.jsonl")} if (out / "runs.jsonl").exists() else set()
    rng = random.Random(a.seed)
    orders = [rng.sample(range(len(pubs)), len(pubs)) for _ in range(a.repeats)]
    runstatus.progress(len(done), a.repeats)
    mode = b if a.fake else Batch(backend=b)
    try:
        for rep in range(a.repeats):
            if rep in done:
                continue
            order = orders[rep]
            runstatus.phase("job", limit=6 * 3600, repeat=rep, pubs=len(order))
            sampler = SamplerV2(mode=mode)
            job = sampler.run([pubs[i][1] for i in order], shots=a.shots)
            print(f"repeat {rep}: job {job.job_id()}", flush=True)
            res = job.result()
            counts = {f"{pubs[i][0][0]}|{pubs[i][0][1]}": res[k].data.meas.get_counts() for k, i in enumerate(order)}
            with open(out / "runs.jsonl", "a") as f:
                f.write(
                    json.dumps(dict(repeat=rep, job=job.job_id(), shots=a.shots, order=order, counts=counts)) + "\n"
                )
            runstatus.row_done()
    finally:
        if not a.fake:
            mode.close()


def hellinger_fidelity(p, q):
    keys = set(p) | set(q)
    return sum(np.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys) ** 2


def analyze(a):
    out = outdir(a)
    P = json.load(open(out / "plan.json"))
    runs = [json.loads(line) for line in open(out / "runs.jsonl")]
    res = []
    for r in P["rows"]:
        ideal = r["ideal"]
        best = max(ideal, key=ideal.get)
        row = dict(name=r["name"], n=r["n"], deterministic=r["deterministic"], pipes={})
        for p in PIPES:
            vals = []
            for run_ in runs:
                c = run_["counts"][f"{r['name']}|{p}"]
                tot = sum(c.values())
                emp = {k: v / tot for k, v in c.items()}
                vals.append(emp.get(best, 0.0) if r["deterministic"] else hellinger_fidelity(emp, ideal))
            row["pipes"][p] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                   values=vals, esp_pred=r["pipes"][p]["esp_readout"],
                                   twoq=r["pipes"][p]["twoq"], total=r["pipes"][p]["total"])  # fmt: skip
        res.append(row)
    summary = {}
    for p in PIPES[1:]:
        ratios = [x["pipes"][p]["mean"] / x["pipes"]["L3"]["mean"] for x in res if x["pipes"]["L3"]["mean"] > 0]
        wins = sum(x["pipes"][p]["mean"] > x["pipes"]["L3"]["mean"] for x in res)
        summary[p] = dict(geo_mean_ratio=float(np.exp(np.mean(np.log(ratios)))), wins=wins, circuits=len(res))
    json.dump(dict(backend=P["backend"], repeats=len(runs), rows=res, summary=summary),
              open(out / "analysis.json", "w"), indent=1)  # fmt: skip
    for x in res:
        print(f"{x['name']:28} {'det' if x['deterministic'] else 'hel'}",
              "  ".join(f"{p} {v['mean']:.3f}±{v['sd']:.3f} (pred {v['esp_pred']:.3f})" for p, v in x["pipes"].items()))  # fmt: skip
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "analyze"])
    ap.add_argument("--backend", default="ibm_fez", help="real backend name (needs a saved IBM Quantum account)")
    ap.add_argument("--fake", default="", help="dry run on a fake backend, e.g. torino")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-qubits", type=int, default=3)
    ap.add_argument("--max-qubits", type=int, default=8)
    ap.add_argument("--min-esp", type=float, default=0.2, help="drop circuits whose L3 ESP (with readout) is lower")
    ap.add_argument("--max-circuits", type=int, default=15)
    ap.add_argument("--min-twoq", type=int, default=4, help="drop circuits with fewer 2q gates at L3 (nothing to test)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4000)
    a = ap.parse_args()
    if a.mode == "run":
        runstatus.start(f"hardware_{a.fake or a.backend}_{a.tag}")
    dict(plan=plan, run=run, analyze=analyze)[a.mode](a)
