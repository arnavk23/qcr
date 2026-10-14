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

"""The general optimiser in every equivalence mode, on both device families."""

from math import pi

import numpy as np

from qcr.gates import Gate, circuit_unitary
from qcr.general import GRIDS, Config, Database, Optimizer, commutes_exact, equivalent_local, repair_phase
from qcr.hardware import Device, ion, nam

from .base import QCRTestCase


def line(n):
    return Device("line", n, {frozenset((i, i + 1)) for i in range(n - 1)}, ("rz", "sx", "x"), "cz", ("rz",), False)


class TestPrimitives(QCRTestCase):
    """Exactness of the low-level tests."""

    def test_near_commuting_gates_do_not_commute(self):
        """Regression: np.allclose's relative tolerance once let these be swapped."""
        a, b = Gate("rx", (0,), 1e-7), Gate("rz", (0,), 1.0)
        self.assertFalse(commutes_exact(a, b))
        self.assertTrue(commutes_exact(Gate("rz", (0,), 1e-7), b))

    def test_equivalence_test_is_linear_in_the_error(self):
        """Regression: a fidelity test accepts errors of order sqrt(tol)."""
        U = np.eye(8, dtype=complex)
        V = circuit_unitary([Gate("rz", (0,), 1e-6)], 3)
        self.assertFalse(equivalent_local(U, V, "phase"))

    def test_phase_repair_tolerates_float_drift(self):
        """Regression: on QFTAdd32 (Heron) the phase product drifted 1.2e-9 from i and no gadget matched."""
        circ = [Gate("rz", (0,), 0.3), Gate("cz", (0, 1)), Gate("sx", (1,))]
        out, _ = repair_phase(circ, 1j * (1 - 1.24e-9), line(2))
        np.testing.assert_allclose(circuit_unitary(out, 2), -1j * circuit_unitary(circ, 2), atol=1e-12)


class _OptimizerModes:
    """Mixin: every mode must return a legal circuit within its error bound."""

    dev = None
    topo = None
    grid = None
    n = 4

    @classmethod
    def setUpClass(cls):
        cls.dbs = {
            (k, cls.topo): Database(GRIDS[cls.grid], cls.topo, k, 3, verbose=False) for k in ("phase", "out", "state")
        }

    def _run(self, mode, state_aware=False, eps=0.0, continuous=True, seeds=2):
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            circ = self.random_native(self.dev, self.n, 50, rng, continuous)
            cfg = Config(mode=mode, state_aware=state_aware, eps=eps, cost="calibrated", patience=2)
            opt = Optimizer(self.dev, self.dbs, cfg)
            res = opt.run(circ, seed=seed)
            self.assertTrue(all(self.dev.is_native(g) for g in res.gates))
            # the guarantee is on the objective; on ions the gate count may rise (pulses -> virtual rz)
            self.assertLessEqual(opt.cost(res.gates), opt.cost(circ) + 1e-12)
            U, V = circuit_unitary(circ, self.n), circuit_unitary(res.gates, self.n)
            if state_aware:
                p, q = np.abs(U[:, 0]) ** 2, np.abs(V[:, 0]) ** 2
                self.assertLess(np.abs(p - q).max(), 1e-9 + 2 * res.error_bound)
            elif mode == "exact":
                self.assertLessEqual(np.linalg.norm(U - V, 2), 1e-9 + res.error_bound)
            else:
                ov = np.trace(U.conj().T @ V)
                self.assertLessEqual(np.linalg.norm(U * (ov / abs(ov)) - V, 2), 1e-9 + res.error_bound)

    def test_exact_including_phase(self):
        self._run("exact")

    def test_up_to_phase(self):
        self._run("phase")

    def test_state_and_measurement_aware(self):
        self._run("exact", state_aware=True)

    def test_approximate_error_within_bound(self):
        self._run("exact", eps=1e-2)

    def test_discrete_angles(self):
        self._run("exact", continuous=False)

    def test_ablation_switches_stay_exact(self):
        """Every ablation setting still returns an exact, legal, never-costlier circuit."""
        rng = np.random.default_rng(3)
        circ = self.random_native(self.dev, self.n, 40, rng)
        U = circuit_unitary(circ, self.n)
        for kw in (
            dict(use_db=False),
            dict(use_mitm=False),
            dict(use_resynth=False),
            dict(db_radius=2),
            dict(max_jump=1),
            dict(w=4),
            dict(patience=0),
        ):
            opt = Optimizer(self.dev, self.dbs, Config(mode="exact", cost="calibrated", **{"patience": 2} | kw))
            res = opt.run(circ, seed=0)
            self.assertTrue(all(self.dev.is_native(g) for g in res.gates), kw)
            self.assertLessEqual(opt.cost(res.gates), opt.cost(circ) + 1e-12, kw)
            self.assertLessEqual(np.linalg.norm(U - circuit_unitary(res.gates, self.n), 2), 1e-9, kw)

    def test_never_more_expensive_than_input(self):
        rng = np.random.default_rng(7)
        circ = self.random_native(self.dev, self.n, 40, rng)
        opt = Optimizer(self.dev, self.dbs, Config(mode="exact", cost="calibrated", patience=2))
        self.assertLessEqual(opt.cost(opt.run(circ).gates), opt.cost(circ) + 1e-12)


class TestHeavyHexLike(_OptimizerModes, QCRTestCase):
    """IBM-style native set on a line coupling map."""

    dev, topo, grid = line(4), "line", "heron"


class TestTrappedIon(_OptimizerModes, QCRTestCase):
    """All-to-all ion model with arbitrary-angle MS gates."""

    dev, topo, grid = ion(4), "tri", "ion"

    def test_ms_resynthesis_reduces_entanglers(self):
        """Three MS gates on one pair compress to at most three, usually fewer, exactly."""
        circ = [
            Gate("rxx", (0, 1), 0.3),
            Gate("ry", (0,), 0.4),
            Gate("rxx", (0, 1), 0.5),
            Gate("rx", (1,), 0.6),
            Gate("rxx", (0, 1), pi / 7),
        ]
        res = Optimizer(ion(2), {}, Config(mode="exact", cost="calibrated")).run(circ)
        self.assertLessEqual(sum(g.name == "rxx" for g in res.gates), 3)
        self.assertUnitaryEqual(res.gates, circ, 2)


class TestNamGateSet(QCRTestCase):
    """Logical-level Nam gate set {H, X, Rz, CNOT}, all-to-all (the VOQC/Quartz/QUESO setting)."""

    def test_equal_up_to_phase_native_and_shorter(self):
        dbs = {("phase", "tri"): Database(GRIDS["nam"], "tri", "phase", 4, verbose=False)}
        for seed in range(2):
            rng = np.random.default_rng(seed)
            circ = self.random_native(nam(4), 4, 60, rng, continuous=seed == 1)
            opt = Optimizer(nam(4), dbs, Config(mode="phase", cost="count", patience=2))
            out = opt.run(circ, seed=seed).gates
            self.assertTrue(all(nam(4).is_native(g) for g in out))
            self.assertLessEqual(len(out), len(circ))
            U, V = circuit_unitary(circ, 4), circuit_unitary(out, 4)
            ov = np.trace(U.conj().T @ V)
            self.assertLessEqual(np.linalg.norm(U * (ov / abs(ov)) - V, 2), 1e-9)
