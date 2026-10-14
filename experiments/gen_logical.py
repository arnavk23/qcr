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


"""Logical-level comparison on the Nam gate set {H, X, Rz, CNOT}, all-to-all, gate count.

This is the setting in which VOQC, Quartz and QUESO report results: the 26 standard Nam/VOQC arithmetic and Toffoli
circuits, equivalence up to global phase, total gates and CNOTs. Every tool gets the same input file
(results/logical/inputs/<name>.qasm, the standard decomposition distributed with QUESO).

Tools (each skipped when not installed): qiskit (L3), tket (FullPeepholeOptimise + rebase), ours (qcr, phase mode,
gate count), voqc (optimize_nam), queso (time limit --queso-sec). Outputs go to results/logical/<tool>/<name>.qasm,
rows to results/general/logical_<tag>.jsonl (one row per circuit and tool, resumable). Each output is checked
against the input: statevector on random states up to global phase (<= 24 qubits), else MQT QCEC.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import runstatus
from gen_bench import OUT, qk_to_gates
from gen_large import _qcec_check

from qcr.gates import to_qasm
from qcr.general import GRIDS, Config, Database, Optimizer
from qcr.hardware import nam
from qcr.verify import apply

ROOT = Path(__file__).resolve().parents[1]
SUITE = [  # the 26 circuits of the VOQC, Quartz and QUESO evaluations
    "adder_8", "barenco_tof_3", "barenco_tof_4", "barenco_tof_5", "barenco_tof_10", "csla_mux_3", "csum_mux_9",
    "gf2^4_mult", "gf2^5_mult", "gf2^6_mult", "gf2^7_mult", "gf2^8_mult", "gf2^9_mult", "gf2^10_mult", "mod5_4",
    "mod_mult_55", "mod_red_21", "qcla_adder_10", "qcla_com_7", "qcla_mod_7", "rc_adder_6", "tof_3", "tof_4",
    "tof_5", "tof_10", "vbe_adder_3",
]  # fmt: skip
LOG = ROOT / "results" / "logical"
TOOLS = Path.home() / "tools"
SV_MAX = 20  # statevector checks of 21-24 qubits take 15-40 min each; QCEC is fast on these unrouted circuits
DB = {}


def inputs():
    """Copy (once) and return {name: input qasm path}. The inputs are the standard {h, x, rz, cx} decompositions
    distributed with QUESO (benchmarks/decomposed/decomp0/nam_rz), the files behind the published VOQC, Quartz and
    QUESO numbers, so every tool here and in those papers starts from the same circuits."""
    src = TOOLS / "src" / "queso" / "benchmarks" / "decomposed" / "decomp0" / "nam_rz"
    d = LOG / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in SUITE:
        p = d / f"{name}.qasm"
        if not p.exists():
            shutil.copy(src / f"decomp0_{name}.qasm", p)
        out[name] = p
    return out


def read(path):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_file(str(path))
    return qk_to_gates(qc), qc.num_qubits


def check(a, b, n):
    """max over random states of min_phi ||e^{i phi} A v - B v||, or QCEC above SV_MAX qubits."""
    if n <= SV_MAX:
        rng = np.random.default_rng(1)
        worst = 0.0
        for _ in range(3):
            v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
            v /= np.linalg.norm(v)
            x, y = apply(v, a, n), apply(v, b, n)
            o = np.vdot(x, y)
            worst = max(worst, float(np.linalg.norm(x * (o / abs(o)) - y)))
        return dict(method="statevector", err=worst, ok=worst < 1e-6)
    from qcr.verify import to_qiskit

    r = _qcec_check(to_qiskit(a, n), to_qiskit(b, n))  # in-process: pool workers may not start child processes
    return r | dict(ok=r["result"] in ("equivalent", "equivalent_up_to_global_phase"))


# ---------------------------------------------------------------- tools: (input path, n) -> gates


def t_qiskit(path, n, a):
    from qiskit import QuantumCircuit, transpile

    qc = transpile(QuantumCircuit.from_qasm_file(str(path)), basis_gates=["h", "x", "rz", "cx"], optimization_level=3)
    return qk_to_gates(qc)


def t_tket(path, n, a):
    from pytket import OpType
    from pytket.passes import AutoRebase, FullPeepholeOptimise, RemoveRedundancies
    from pytket.qasm import circuit_from_qasm, circuit_to_qasm_str

    c = circuit_from_qasm(str(path))
    FullPeepholeOptimise(allow_swaps=False).apply(c)
    AutoRebase({OpType.H, OpType.X, OpType.Rz, OpType.CX}).apply(c)
    RemoveRedundancies().apply(c)
    return _from_qasm_str(circuit_to_qasm_str(c))


def t_ours(path, n, a):
    if "nam" not in DB:
        p = ROOT / "cache" / "gdb_nam_tri_phase.npz"
        DB["nam"] = Database.load(p, GRIDS["nam"]) if p.exists() else Database(GRIDS["nam"], "tri", "phase", 6)
    g, _ = read(path)
    cfg = Config(mode="phase", cost=a.ours_cost, patience=4)
    return Optimizer(nam(n), {("phase", "tri"): DB["nam"]}, cfg).run(g, seed=0).gates


def t_voqc(path, n, a):
    exe = TOOLS / "bin" / "voqc_nam"
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "out.qasm"
        subprocess.run([str(exe), "-i", str(path), "-o", str(o)], check=True, capture_output=True, timeout=a.tool_sec)
        return _from_qasm_str(o.read_text())


def t_voqc_ours(path, n, a):
    """VOQC first, then ours on its output (the stacking we also use after BQSKit)."""
    with tempfile.TemporaryDirectory() as td:
        mid = Path(td) / "voqc.qasm"
        mid.write_text(to_qasm(t_voqc(path, n, a), n))
        return t_ours(mid, n, a)


def t_queso(path, n, a):
    jar = next((TOOLS / "src" / "queso" / "target").glob("*with-dependencies.jar"))
    env = dict(os.environ, JAVA_HOME=str(TOOLS / "jdk17"))
    java = str(TOOLS / "jdk17" / "bin" / "java")
    rules = TOOLS / "queso_rules"
    with tempfile.TemporaryDirectory() as td:  # QUESO reads rule files relative to its working directory
        cmd = [java, "-Xmx3g", "--enable-preview", "-cp", str(jar), "Optimizer", "-c", str(path), "-g", "nam",
               "-r", "rules_q3_s6_nam.txt", "-sr", "rules_q3_s3_nam_symb.txt",
               "-t", str(int(a.queso_sec)), "-o", td, "-j", "nam"]  # fmt: skip
        subprocess.run(
            cmd, check=True, capture_output=True, timeout=2 * a.queso_sec, env=env, cwd=rules
        )  # QUESO overruns -t on large inputs
        return _from_qasm_str(next(Path(td).glob("optimized_*.qasm")).read_text())


def _from_qasm_str(txt):
    from qiskit import QuantumCircuit, transpile

    # VOQC writes rotations as rzq(a,b) = Rz(a/b * pi)
    txt = re.sub(r"rzq\((-?\d+),\s*(-?\d+)\)", lambda m: f"rz(pi*{m[1]}/{m[2]})", txt)
    qc = QuantumCircuit.from_qasm_str(txt)
    if not set(qc.count_ops()) <= {"h", "x", "rz", "cx", "barrier"}:  # e.g. rzq / u1 spellings: rebase only, no opt
        qc = transpile(qc, basis_gates=["h", "x", "rz", "cx"], optimization_level=0)
    return qk_to_gates(qc)


TOOL_FNS = dict(
    qiskit=t_qiskit, tket=t_tket, ours=t_ours, voqc=t_voqc, voqc_ours=t_voqc_ours, queso=t_queso, queso5=t_queso
)  # queso5: --queso-sec 300


def available(tool):
    return {
        "voqc": (TOOLS / "bin" / "voqc_nam").exists(),
        "queso": (TOOLS / "queso_rules" / "rules_q3_s6_nam.txt").exists()
        and any((TOOLS / "src" / "queso" / "target").glob("*with-dependencies.jar")),
        "voqc_ours": (TOOLS / "bin" / "voqc_nam").exists(),
    }.get(tool.rstrip("5"), True)


def job(args):
    name, tool, a = args
    try:
        path = LOG / "inputs" / f"{name}.qasm"
        g0, n = read(path)
        t = time.perf_counter()
        g = TOOL_FNS[tool](path, n, a)
        sec = time.perf_counter() - t
        d = LOG / tool
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.qasm").write_text(to_qasm(g, n))
        return dict(
            name=name, tool=tool, n=n, input_total=len(g0), input_cx=sum(x.name == "cx" for x in g0),
            total=len(g), cx=sum(x.name == "cx" for x in g), seconds=sec,
            native=all(x.name in ("h", "x", "rz", "cx") for x in g), check=check(g0, g, n),
        )  # fmt: skip
    except Exception:  # noqa: BLE001  recorded, a resume retries it
        return dict(name=name, tool=tool, _error=traceback.format_exc()[-3000:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="*", default=list(TOOL_FNS))
    ap.add_argument("--filter", default="")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--ours-cost", default="count", choices=["count", "w2"])
    ap.add_argument("--queso-sec", type=float, default=600)
    ap.add_argument("--tool-sec", type=float, default=3600)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    runstatus.start(f"logical_{a.tag}")
    ins = inputs()
    part = OUT / f"logical_{a.tag}.jsonl"
    done = {(r["name"], r["tool"]) for r in map(json.loads, open(part))} if part.exists() else set()
    tools = [t for t in a.tools if available(t)]
    print("tools:", tools, "skipped:", sorted(set(a.tools) - set(tools)), flush=True)
    jobs = [(k, t, a) for k in ins if re.search(a.filter, k) for t in tools if (k, t) not in done]
    print(f"{len(done)} rows done, {len(jobs)} to run", flush=True)
    runstatus.progress(len(done), len(done) + len(jobs))
    runstatus.phase("pool", jobs=len(jobs), procs=a.procs)
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=1) as p, open(part, "a") as f:
        for r in p.imap_unordered(job, jobs):
            if "_error" in r:
                print(f"FAILED {r['name']} {r['tool']}\n{r['_error']}", flush=True)
                runstatus.problem(f"{r['name']} {r['tool']} failed:\n{r['_error'][-800:]}")
                continue
            if not r["check"]["ok"] or not r["native"]:
                runstatus.problem(f"{r['name']} {r['tool']}: check {r['check']} native {r['native']}")
            f.write(json.dumps(r) + "\n")
            f.flush()
            runstatus.row_done()
            print(json.dumps({k: r[k] for k in ("name", "tool", "total", "cx", "seconds")}), flush=True)
