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

    def success_probability(self, gates) -> float:
        return float(np.exp(sum(math.log1p(-min(self.gate_error(g), 0.5)) for g in gates)))

    def is_native(self, g: Gate) -> bool:
        if self.angle_grid is not None and g.name in ("rx", "ry", "rz", "rxx"):
            t = (g.theta + math.pi) % (2 * math.pi) - math.pi
            if not any(abs(t - a) < 1e-9 for a in self.angle_grid):
                return False
        if len(g.qubits) == 2:
            return g.name == self.native_2q and self.coupled(*g.qubits)
        return g.name in self.native_1q


def heron(n_active: int | None = None) -> Device:
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    b = FakeTorino()
    t = b.target
    err1, err2 = {}, {}
    for gname in ("sx", "x"):
        for qargs, p in t[gname].items():
            if p is not None and p.error is not None:
                err1[(gname, qargs[0])] = p.error
    coupling = set()
    for qargs, p in t["cz"].items():
        pair = frozenset(qargs)
        if p is not None and p.error is not None:
            if p.error >= 1.0:  # dead coupler (22 of them in FakeTorino): unusable by any pipeline
                continue
            err2[("cz", pair)] = p.error
        coupling.add(pair)
    return Device("heron", b.num_qubits, coupling, ("rz", "sx", "x"), "cz", ("rz",), False, err1, err2)


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
