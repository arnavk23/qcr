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


"""Equivalence checks that scale beyond dense unitaries."""

from __future__ import annotations

import numpy as np

from .gates import BASE, Gate


def apply(state: np.ndarray, gates, n: int) -> np.ndarray:
    psi = state.reshape([2] * n)
    for g in gates:
        k = len(g.qubits)
        M = BASE[g.name](g.theta).reshape([2] * (2 * k))
        psi = np.tensordot(M, psi, axes=(list(range(k, 2 * k)), list(g.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(g.qubits))
    return psi.reshape(-1)


def equivalent(a, b, n: int, trials: int = 3, seed: int = 0, tol: float = 1e-8) -> bool:
    """Probabilistic test of a == b up to one global phase (exact for unitaries
    with probability 1 over random inputs, checked on several inputs)."""
    rng = np.random.default_rng(seed)
    phase = None
    for _ in range(trials):
        v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        v /= np.linalg.norm(v)
        x, y = apply(v, a, n), apply(v, b, n)
        ov = np.vdot(x, y)
        if abs(abs(ov) - 1) > tol:
            return False
        if phase is None:
            phase = ov
        elif abs(ov - phase) > 1e-6:
            return False
    return True


def from_qiskit(qc) -> list[Gate]:
    out = []
    for inst in qc.data:
        name = inst.operation.name
        qs = tuple(qc.find_bit(q).index for q in inst.qubits)
        th = float(inst.operation.params[0]) if inst.operation.params else 0.0
        out.append(Gate(name, qs, th))
    return out


def to_qiskit(gates, n: int):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for g in gates:
        if g.name in ("cz", "cx", "sx", "x", "h"):
            getattr(qc, g.name)(*g.qubits)
        else:
            getattr(qc, g.name)(g.theta, *g.qubits)
    return qc
