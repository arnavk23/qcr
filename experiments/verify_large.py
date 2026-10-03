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


"""Re-run circuits whose routed layout was too wide for gen_bench.verify (> 16 active
qubits) and verify them on the actual active qubits (statevector, up to 26 qubits)."""

import json
import sys

import gen_bench as gb
from common import RESULTS
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.transpiler import CouplingMap

from qcr.general import Config, Optimizer
from qcr.hardware import coupling_list, heron, qiskit_basis


def verify_wide(a, b, mode, state_aware):
    act = sorted({q for g in a + b for q in g.qubits})
    if len(act) > 26:
        return None, len(act)
    import numpy as np

    m = {q: i for i, q in enumerate(act)}
    A, B, n = [g.remap(m) for g in a], [g.remap(m) for g in b], len(act)
    if state_aware:
        z = np.zeros(2**n, complex)
        z[0] = 1
        pa, pb = np.abs(gb.apply(z, A, n)) ** 2, np.abs(gb.apply(z, B, n)) ** 2
        return float(0.5 * np.abs(pa - pb).sum()), n
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2):
        v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        v /= np.linalg.norm(v)
        worst = max(worst, float(np.linalg.norm(gb.apply(v, A, n) - gb.apply(v, B, n))))
    return worst, n


name = sys.argv[1] if len(sys.argv) > 1 else "nam/pf2_10"
qc = QuantumCircuit.from_qasm_str(qasm2.dumps(gb.suite_circuits()[name]))
dev = heron()
dbs = gb.load_dbs("heron")
kw = dict(basis_gates=qiskit_basis(dev), seed_transpiler=0, coupling_map=CouplingMap(coupling_list(dev)))
l1 = gb.qk_to_gates(transpile(qc, optimization_level=1, **kw))
out = {}
for cname, ckw in gb.CONFIGS.items():
    r = Optimizer(dev, dbs, Config(**ckw, patience=4)).run(l1, seed=0)
    err, n = verify_wide(l1, r.gates, ckw.get("mode"), ckw.get("state_aware", False))
    out[cname] = dict(total=len(r.gates), verify=err, active=n, err_bound=r.error_bound)
    print(cname, out[cname], flush=True)
json.dump({name: out}, open(RESULTS / "general" / f"verify_large_{name.replace('/', '_')}.json", "w"), indent=1)
