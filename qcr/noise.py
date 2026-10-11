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

"""Noisy simulation of routed circuits with a fake backend's calibration.

Simulating all 133-156 physical qubits of a Heron chip is infeasible, and unnecessary: only the qubits a circuit
touches matter. :func:`noisy_probabilities` restricts the backend's noise model (gate errors with thermal
relaxation, and optionally readout errors) to the active physical qubits, remaps it onto a compact register and
returns exact output probabilities from a density-matrix simulation. :func:`sampled_tvd` adds shot noise for
error bars. Requires the ``hardware`` extra (``qiskit-aer``, ``qiskit-ibm-runtime``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .verify import to_qiskit


@lru_cache(maxsize=8)
def _backend_noise(name: str):
    from qiskit_aer.noise import NoiseModel

    from .hardware import fake_backend

    return NoiseModel.from_backend(fake_backend(name))


def _compact(gates):
    act = sorted({q for g in gates for q in g.qubits})
    m = {q: i for i, q in enumerate(act)}
    return [g.remap(m) for g in gates], m


def local_noise_model(backend: str, mapping: dict, readout: bool = True):
    """Noise model of ``backend`` restricted to the physical qubits in ``mapping`` (physical -> compact)."""
    from qiskit_aer.noise import NoiseModel

    full = _backend_noise(backend)
    nm = NoiseModel(basis_gates=["rz", "sx", "x", "cz"])
    for gname, per_q in full._local_quantum_errors.items():
        if gname not in ("sx", "x", "cz"):
            continue
        for qs, err in per_q.items():
            if all(q in mapping for q in qs):
                nm.add_quantum_error(err, gname, [mapping[q] for q in qs])
    if readout:
        for qs, err in full._local_readout_errors.items():
            if qs[0] in mapping:
                nm.add_readout_error(err, [mapping[qs[0]]])
    return nm


def ideal_probabilities(gates) -> np.ndarray:
    """Exact output distribution on the compact register (little-endian, as Qiskit)."""
    from qiskit.quantum_info import Statevector

    g, _ = _compact(gates)
    n = len({q for x in g for q in x.qubits})
    return Statevector(to_qiskit(g, n)).probabilities()


def noisy_probabilities(gates, backend: str = "torino", readout: bool = True) -> np.ndarray:
    """Exact noisy output distribution of a routed circuit (density matrix, active qubits only)."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    g, m = _compact(gates)
    n = len(m)
    qc = to_qiskit(g, n)
    qc.measure_all()
    # readout errors act on measurements, which are removed here; they are applied exactly afterwards
    sim = AerSimulator(method="density_matrix", noise_model=local_noise_model(backend, m, readout=False))
    qc = transpile(qc, sim, optimization_level=0)
    qc.remove_final_measurements()
    qc.save_probabilities()
    p = np.asarray(sim.run(qc).result().data()["probabilities"])
    return _apply_readout(p, backend, m) if readout else p


def _apply_readout(p, backend, mapping):
    full = _backend_noise(backend)
    n = len(mapping)
    t = p.reshape([2] * n)  # axis k <-> qubit n-1-k (little-endian)
    for phys, comp in mapping.items():
        err = full._local_readout_errors.get((phys,))
        if err is None:
            continue
        M = np.asarray(err.probabilities)  # M[true, measured]
        ax = n - 1 - comp
        t = np.moveaxis(np.tensordot(M.T, np.moveaxis(t, ax, 0), axes=1), 0, ax)
    return t.reshape(-1)


def tvd(p, q) -> float:
    """Total-variation distance between two distributions."""
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def hellinger_fidelity(p, q) -> float:
    """Hellinger fidelity :math:`(\\sum_x \\sqrt{p_x q_x})^2`."""
    return float(np.sum(np.sqrt(np.asarray(p) * np.asarray(q))) ** 2)


def sampled_tvd(p_ideal, p_noisy, shots: int, rng: np.random.Generator) -> float:
    """TVD between the ideal distribution and an empirical one drawn from ``p_noisy`` with ``shots`` samples."""
    counts = rng.multinomial(shots, np.clip(p_noisy, 0, None) / np.clip(p_noisy, 0, None).sum())
    return tvd(p_ideal, counts / shots)
