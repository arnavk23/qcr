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

"""The faithful port of the reference method (V1/V2)."""

from math import pi

import numpy as np

from qcr.compute_graph import build
from qcr.gates import Gate, circuit_unitary, make_pool, random_circuit
from qcr.reducer import DBEngine, RandomSamplingEngine, reduce_circuit, simplereduce
from qcr.unitary import equal_up_to_phase, phase_key

from .base import QCRTestCase


class TestUnitaryKeys(QCRTestCase):
    """O(1) database keys modulo global phase."""

    def test_phase_key_ignores_global_phase(self):
        U = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)) + 0j)[0]
        self.assertEqual(phase_key(U), phase_key(np.exp(0.7j) * U))
        self.assertTrue(equal_up_to_phase(U, -1j * U))


class TestSimpleReduce(QCRTestCase):
    """simplereduce.m, including its documented loop-bound bug."""

    def test_removes_inner_inverse_pair(self):
        a, ainv, b = Gate("rx", (0,), pi / 2), Gate("rx", (0,), -pi / 2), Gate("rz", (1,), pi / 2)
        self.assertEqual(simplereduce([b, a, ainv, b, b]), [b, b, b])

    def test_never_checks_last_pair(self):
        """Faithful to the MATLAB loop bounds: the final adjacent pair is skipped."""
        a, ainv, b = Gate("rx", (0,), pi / 2), Gate("rx", (0,), -pi / 2), Gate("rz", (1,), pi / 2)
        self.assertEqual(simplereduce([b, a, ainv]), [b, a, ainv])


class TestReduceCircuit(QCRTestCase):
    """V1 and V2 return equivalent, shorter circuits."""

    @classmethod
    def setUpClass(cls):
        cls.pool = make_pool("ion_pm", 2)
        cls.cg = build(2, cls.pool, 4)

    def _check(self, engine, seed):
        rng = np.random.default_rng(seed)
        circ = random_circuit(make_pool("ion_pm", 3), 60, rng)
        out, trace = reduce_circuit(circ, engine, self.pool, rng, time_budget=20, stall_attempts=400)
        self.assertLess(len(out), len(circ))
        self.assertUnitaryEqual(out, circ, 3, up_to_phase=True)
        return trace

    def test_database_engine(self):
        self._check(DBEngine(self.cg, "hash"), seed=3)

    def test_scan_and_hash_lookups_agree(self):
        scan, hashed = DBEngine(self.cg, "scan"), DBEngine(self.cg, "hash")
        rng = np.random.default_rng(4)
        for _ in range(30):
            blen = int(rng.integers(2, 7))
            U = circuit_unitary(random_circuit(self.pool, blen, rng), 2)
            a, b = scan.find(U, blen), hashed.find(U, blen)
            self.assertEqual(a is None, b is None)
            if a is not None:
                self.assertEqual(len(a), len(b))

    def test_random_sampling_engine(self):
        self._check(RandomSamplingEngine(self.pool, 2, rng=np.random.default_rng(5)), seed=5)
