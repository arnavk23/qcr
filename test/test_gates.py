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

"""Tests for gate conventions, pools and QASM I/O."""

from math import pi

import numpy as np

from qcr.gates import Gate, circuit_unitary, counts, make_pool, parse_qasm, to_qasm, to_quarter_turns
from qcr.verify import from_qiskit, to_qiskit

from .base import QCRTestCase


class TestConventions(QCRTestCase):
    """Rotation and ordering conventions (MATLAB Quantum Computing Support Package)."""

    def test_rx_pi_is_minus_i_x(self):
        U = circuit_unitary([Gate("rx", (0,), pi)], 1)
        np.testing.assert_allclose(U, -1j * np.array([[0, 1], [1, 0]]), atol=1e-12)

    def test_circuit_order_is_left_to_right_in_time(self):
        a, b = Gate("rx", (0,), pi / 2), Gate("rz", (0,), pi / 2)
        expected = circuit_unitary([b], 1) @ circuit_unitary([a], 1)
        np.testing.assert_allclose(circuit_unitary([a, b], 1), expected, atol=1e-12)

    def test_qubit_zero_is_most_significant(self):
        U = circuit_unitary([Gate("x", (0,))], 2)
        self.assertAlmostEqual(abs(U[2, 0]), 1.0)  # |00> -> |10>

    def test_symmetric_gates_sort_qubits_on_remap(self):
        g = Gate("cz", (0, 1)).remap({0: 2, 1: 1})
        self.assertEqual(g.qubits, (1, 2))


class TestPools(QCRTestCase):
    """The paper's operator pools."""

    def test_pool_sizes_match_table_one(self):
        self.assertEqual(len(make_pool("ion_pm", 2)), 14)
        self.assertEqual(len(make_pool("ion_pm", 3)), 24)
        self.assertEqual(len(make_pool("nisq", 3)), 15)

    def test_unknown_pool_raises(self):
        with self.assertRaises(ValueError):
            make_pool("photonic", 2)

    def test_quarter_turns(self):
        out = to_quarter_turns([Gate("rz", (0,), pi), Gate("rx", (1,), -pi / 2), Gate("ry", (0,), 0.0)])
        self.assertEqual(counts(out), {"rz": 2, "rx": 1})
        with self.assertRaises(ValueError):
            to_quarter_turns([Gate("rz", (0,), 0.3)])


class TestInterop(QCRTestCase):
    """QASM and Qiskit round trips."""

    def setUp(self):
        self.gates = [
            Gate("rx", (0,), pi / 2),
            Gate("rxx", (0, 2), -pi / 2),
            Gate("rz", (1,), 0.25),
            Gate("cz", (1, 2)),
        ]

    def test_qasm_round_trip(self):
        back, n = parse_qasm(to_qasm(self.gates, 3))
        self.assertEqual(n, 3)
        self.assertUnitaryEqual(back, self.gates, 3)

    def test_qiskit_round_trip(self):
        back = from_qiskit(to_qiskit(self.gates, 3))
        self.assertUnitaryEqual(back, self.gates, 3)

    def test_qiskit_unitary_agrees_up_to_endianness(self):
        from qiskit.quantum_info import Operator

        U = Operator(to_qiskit(self.gates, 3).reverse_bits()).data
        np.testing.assert_allclose(U, circuit_unitary(self.gates, 3), atol=1e-12)
