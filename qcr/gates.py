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


"""Gate definitions, token pools and dense unitaries.

Conventions follow MATLAB's Quantum Computing Support Package, which the
authors' reference code (matlab_demo/QCOptimDemo) is written against:

* rx(t) = exp(-i t X / 2), ry, rz likewise; rxx(t) = exp(-i t X(x)X / 2)
* qubit 0 is the most significant tensor factor
* a circuit (O_1, ..., O_L) implements U = O_L ... O_1
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import combinations
from math import pi

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _rot(P, t):
    return np.cos(t / 2) * np.eye(P.shape[0]) - 1j * np.sin(t / 2) * P


BASE = {
    "rx": lambda t: _rot(X, t),
    "ry": lambda t: _rot(Y, t),
    "rz": lambda t: _rot(Z, t),
    "rxx": lambda t: _rot(np.kron(X, X), t),
    "cz": lambda t: np.diag([1, 1, 1, -1]).astype(complex),
    "cx": lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    # IBM natives with their exact (phase-carrying) definitions
    "sx": lambda t: 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
    "x": lambda t: np.array([[0, 1], [1, 0]], dtype=complex),
    "h": lambda t: np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "rzz": lambda t: np.diag(np.exp(-0.5j * t * np.array([1, -1, -1, 1]))),
}
TWO_QUBIT = {"rxx", "cz", "cx", "rzz"}
SYMMETRIC = {"rxx", "cz", "rzz"}  # invariant under swapping their two qubits


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    theta: float = 0.0

    def __str__(self) -> str:
        q = ",".join(str(x) for x in self.qubits)
        if self.name in ("cz", "cx", "sx", "x", "h"):
            return f"{self.name}({q})"
        return f"{self.name}({q},{self.theta:+.4f})"

    def remap(self, mapping: dict[int, int]) -> Gate:
        qs = tuple(mapping[q] for q in self.qubits)
        if self.name in SYMMETRIC:
            qs = tuple(sorted(qs))
        return Gate(self.name, qs, self.theta)


@cache
def embed(name: str, qubits: tuple[int, ...], theta: float, n: int) -> np.ndarray:
    """Dense 2^n x 2^n matrix of a gate acting on `qubits` of an n-qubit register."""
    g = BASE[name](theta)
    k = len(qubits)
    # Build by permuting a (g (x) I) operator: put target qubits first.
    full = np.kron(g, np.eye(2 ** (n - k)))
    order = list(qubits) + [q for q in range(n) if q not in qubits]
    # full acts on tensor legs in `order`; transpose to natural order.
    t = full.reshape([2] * (2 * n))
    inv = np.argsort(order)
    perm = list(inv) + [n + i for i in inv]
    return np.ascontiguousarray(t.transpose(perm).reshape(2**n, 2**n))


def gate_matrix(g: Gate, n: int) -> np.ndarray:
    return embed(g.name, g.qubits, round(g.theta, 12), n)


def circuit_unitary(gates, n: int) -> np.ndarray:
    U = np.eye(2**n, dtype=complex)
    for g in gates:
        U = gate_matrix(g, n) @ U
    return U


def ion_pool(n: int, rxx_angles=(pi / 2,), angles=(-pi / 2, pi / 2)) -> list[Gate]:
    """QoperatorsIon1.m ordering: rxx pairs first, then rx, ry, rz per qubit."""
    pool = [Gate("rxx", (i, j), t) for i, j in combinations(range(n), 2) for t in rxx_angles]
    for name in ("rx", "ry", "rz"):
        pool += [Gate(name, (i,), t) for i in range(n) for t in angles]
    return pool


def nisq_pool(n: int, angles=(-pi / 2, pi / 2)) -> list[Gate]:
    """IBM-style pool: rx, rz rotations and cz (Table IV/V of the paper)."""
    pool = [Gate("cz", (i, j)) for i, j in combinations(range(n), 2)]
    for name in ("rx", "rz"):
        pool += [Gate(name, (i,), t) for i in range(n) for t in angles]
    return pool


def make_pool(kind: str, n: int, **kw) -> list[Gate]:
    if kind == "ion":
        return ion_pool(n, **kw)
    if kind == "ion_pm":  # rxx(+-pi/2): the 14/24-operator pool of Table I
        return ion_pool(n, rxx_angles=(-pi / 2, pi / 2), **kw)
    if kind == "nisq":
        return nisq_pool(n, **kw)
    raise ValueError(kind)


def random_circuit(pool: list[Gate], length: int, rng: np.random.Generator) -> list[Gate]:
    return [pool[i] for i in rng.integers(len(pool), size=length)]


def counts(gates) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in gates:
        out[g.name] = out.get(g.name, 0) + 1
    return out


def to_qasm(gates, n: int) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    for g in gates:
        qs = ", ".join(f"q[{q}]" for q in g.qubits)
        if g.name in ("cz", "cx", "sx", "x", "h"):
            lines.append(f"{g.name} {qs};")
        else:
            lines.append(f"{g.name}({g.theta!r}) {qs};")
    return "\n".join(lines) + "\n"


def parse_qasm(text: str) -> tuple[list[Gate], int]:
    """Minimal parser for the demo's QASM files (longcode10.txt etc.)."""
    import re

    n = 0
    gates = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"qreg q\[(\d+)\]", line)
        if m:
            n = int(m.group(1))
            continue
        m = re.match(r"(\w+)(?:\(([^)]*)\))?\s+(.*);", line)
        if not m or m.group(1) in ("OPENQASM", "include", "creg", "qreg", "measure"):
            continue
        name, arg, qs = m.groups()
        qubits = tuple(int(x) for x in re.findall(r"q\[(\d+)\]", qs))
        theta = float(eval(arg, {"pi": pi})) if arg else 0.0
        snapped = round(theta / (pi / 4)) * (pi / 4)
        if abs(snapped - theta) < 1e-3:  # demo files print angles with 4 decimals
            theta = snapped
        gates.append(Gate(name, qubits, theta))
    return gates, n


def to_quarter_turns(gates):
    """Rewrite rotations by k*pi/2 as ``|k|`` rotations by +-pi/2 (pool alphabet);
    drop identities.  Raises ValueError on other angles."""
    out = []
    for g in gates:
        if g.name in ("rx", "ry", "rz", "rxx"):
            k = g.theta / (pi / 2)
            kr = round(k)
            if abs(k - kr) > 1e-9:
                raise ValueError(f"angle not a multiple of pi/2: {g}")
            kr %= 4
            if kr == 3:
                kr = -1
            for _ in range(abs(kr)):
                out.append(Gate(g.name, g.qubits, pi / 2 if kr > 0 else -pi / 2))
        elif g.name in ("cz", "cx"):
            out.append(g)
        else:
            raise ValueError(g)
    return out
