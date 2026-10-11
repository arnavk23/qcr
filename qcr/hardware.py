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


"""Hardware models: native gates, connectivity, cost models.

Two device families:
  * "heron": IBM Heron r1 (qiskit FakeTorino): rz(theta) [virtual], sx, x, cz,
    heavy-hex coupling, per-qubit / per-pair calibrated errors.
  * "ion":   trapped-ion (Quantinuum-like), all-to-all: rx/ry/rz(theta) and
    Moelmer-Soerensen rxx(theta) at any angle.  No public calibration ships with
    qiskit, so errors are model parameters (1q 3e-5, 2q 1e-3, rz virtual).

Cost models (per gate, physical qubits known):
  "count"      every gate costs 1
  "w2"         1q = 1, 2q = 10
  "calibrated" -log(1 - error)  (rz is virtual: error 0 -> tiny tie-breaker)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gates import Gate

EPS_VIRTUAL = 1e-7


@dataclass
class Device:
    name: str
    n: int
    coupling: set  # frozenset pairs; None = all-to-all
    native_1q: tuple[str, ...]
    native_2q: str
    continuous_1q: tuple[str, ...]  # axes accepting any angle
    continuous_2q: bool
    err1: dict = field(default_factory=dict)  # (gate, q) -> error
    err2: dict = field(default_factory=dict)  # (gate, frozenset) -> error
    default_err1: float = 3e-4
    default_err2: float = 5e-3
    angle_grid: tuple | None = None  # if set, rotations must use these angles (mod 2 pi)
    err_meas: dict = field(default_factory=dict)  # qubit -> readout error (used for placement and ESP)

    def coupled(self, a, b) -> bool:
        return self.coupling is None or frozenset((a, b)) in self.coupling

    def gate_error(self, g: Gate) -> float:
        if len(g.qubits) == 2:
            return self.err2.get((g.name, frozenset(g.qubits)), self.default_err2)
        if g.name == "rz":
            return 0.0
        return self.err1.get((g.name, g.qubits[0]), self.default_err1)

    def cost_fn(self, model: str):
        if model == "count":
            return lambda g: 1.0
        if model == "w2":
            return lambda g: 10.0 if len(g.qubits) == 2 else 1.0
        if model == "calibrated":
            return lambda g: max(-math.log1p(-min(self.gate_error(g), 0.5)), EPS_VIRTUAL)
        raise ValueError(model)

    def success_probability(self, gates, readout: bool = False) -> float:
        """Estimated success probability: product of gate fidelities (and of the readout fidelities of the
        active qubits if ``readout`` is true)."""
        s = sum(math.log1p(-min(self.gate_error(g), 0.5)) for g in gates)
        if readout:
            s += sum(math.log1p(-min(self.err_meas.get(q, 0.0), 0.5)) for q in {q for g in gates for q in g.qubits})
        return float(np.exp(s))

    def is_native(self, g: Gate) -> bool:
        if self.angle_grid is not None and g.name in ("rx", "ry", "rz", "rxx"):
            # compare modulo 2 pi (so that a rotation by pi matches a grid entry -pi and vice versa)
            if not any(abs((g.theta - a + math.pi) % (2 * math.pi) - math.pi) < 1e-9 for a in self.angle_grid):
                return False
        if len(g.qubits) == 2:
            return g.name == self.native_2q and self.coupled(*g.qubits)
        return g.name in self.native_1q


HERON_BACKENDS = ("torino", "fez", "marrakesh", "kingston", "aachen")


def fake_backend(name: str = "torino"):
    """Return Qiskit's fake IBM backend ``Fake<Name>`` (needs the ``hardware`` extra)."""
    from qiskit_ibm_runtime import fake_provider

    return getattr(fake_provider, f"Fake{name.capitalize()}")()


def dead_couplers(backend) -> set:
    """Coupler pairs with a calibrated CZ error of 1.0 in either direction (unusable)."""
    t = backend.target
    return {frozenset(q) for q, p in t["cz"].items() if p is not None and p.error is not None and p.error >= 1.0}


def heron(n_active: int | None = None, backend: str = "torino") -> Device:
    """IBM Heron device model from a fake backend's calibration (default: FakeTorino, Heron r1).

    Dead couplers (calibrated CZ error 1.0) are removed from the coupling map, so no pipeline can use them.
    """
    name = "heron" if backend == "torino" else f"heron-{backend}"
    return device_from_backend(fake_backend(backend), name)


def device_from_backend(b, name: str | None = None) -> Device:
    """Device model (rz/sx/x/cz, coupling map, calibrated gate and readout errors) of any CZ-based IBM backend,
    fake or real. Dead couplers are removed."""
    t = b.target
    dead = dead_couplers(b)
    err1, err2 = {}, {}
    for gname in ("sx", "x"):
        for qargs, p in t[gname].items():
            if p is not None and p.error is not None:
                err1[(gname, qargs[0])] = p.error
    err_meas = {q[0]: p.error for q, p in t["measure"].items() if p is not None and p.error is not None}
    coupling = set()
    for qargs, p in t["cz"].items():
        pair = frozenset(qargs)
        if pair in dead:
            continue
        if p is not None and p.error is not None:
            err2[("cz", pair)] = p.error
        coupling.add(pair)
    return Device(
        name or b.name,
        b.num_qubits,
        coupling,
        ("rz", "sx", "x"),
        "cz",
        ("rz",),
        False,
        err1,
        err2,
        err_meas=err_meas,
    )


def coupled_target(backend):
    """Copy of ``backend.target`` without its dead couplers, for noise-aware Qiskit transpilation."""
    from qiskit.transpiler import Target

    src = backend.target
    dead = dead_couplers(backend)
    tgt = Target(num_qubits=src.num_qubits, dt=src.dt)
    for name in src.operation_names:
        if name not in ("rz", "sx", "x", "cz", "measure", "id", "delay", "reset"):
            continue
        props = {q: p for q, p in src[name].items() if q is None or frozenset(q) not in dead or len(q) < 2}
        tgt.add_instruction(src.operation_from_name(name), props)
    return tgt


def ion(n: int) -> Device:
    return Device(
        "ion", n, None, ("rx", "ry", "rz"), "rxx", ("rx", "ry", "rz"), True, default_err1=3e-5, default_err2=1e-3
    )


def qiskit_basis(dev: Device):
    return list(dev.native_1q) + [dev.native_2q]


def coupling_list(dev: Device):
    if dev.coupling is None:
        return None
    out = []
    for p in dev.coupling:
        a, b = tuple(p)
        out += [[a, b], [b, a]]
    return out


def paper_device(kind: str, n: int) -> Device:
    """The paper's discrete alphabets: rotations by +-pi/2 only, all-to-all."""
    half = (math.pi / 2, -math.pi / 2)
    if kind == "ion":
        return Device("ion_paper", n, None, ("rx", "ry", "rz"), "rxx", (), False, angle_grid=half)
    return Device("ibm_paper", n, None, ("rx", "rz"), "cz", (), False, angle_grid=half)
