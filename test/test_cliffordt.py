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

"""Exact Clifford+T arithmetic, certified distances and planted circuits."""

from math import pi

import numpy as np

from qcr.cliffordt import (
    CliffordTBall,
    alphabet,
    exact_equal_up_to_phase,
    exact_gate,
    planted_circuit,
    ring_conj,
    ring_mul,
    to_exact,
)
from qcr.gates import Gate, gate_matrix

from .base import QCRTestCase

OMEGA_POW = np.exp(1j * pi / 4) ** np.arange(4)


def value(M, k):
    return (M @ OMEGA_POW) / np.sqrt(2.0) ** k


class TestRing(QCRTestCase):
    """Arithmetic in Z[w], w = exp(i pi/4)."""

    def test_multiplication_and_conjugation_match_complex_numbers(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            x, y = rng.integers(-5, 6, size=4), rng.integers(-5, 6, size=4)
            self.assertAlmostEqual(complex(ring_mul(x, y) @ OMEGA_POW), complex(x @ OMEGA_POW) * complex(y @ OMEGA_POW))
            self.assertAlmostEqual(complex(ring_conj(x) @ OMEGA_POW), np.conj(complex(x @ OMEGA_POW)))

    def test_generators_are_exact_up_to_phase(self):
        for g in alphabet():
            M, k = exact_gate(g)
            U, V = value(M, k), gate_matrix(g, 2)
            ov = np.trace(U.conj().T @ V) / 4
            self.assertAlmostEqual(abs(ov), 1.0)
            np.testing.assert_allclose(U * ov, V, atol=1e-12)

    def test_non_grid_angle_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_gate(Gate("rz", (0,), 0.1))

    def test_exact_equality_is_projective(self):
        a = [Gate("rz", (0,), pi / 4)] * 8  # rz(2 pi) = -I
        self.assertTrue(exact_equal_up_to_phase(a, []))
        self.assertFalse(exact_equal_up_to_phase(a[:7], []))
        M, k = to_exact([Gate("rxx", (0, 1), pi / 4), Gate("rx", (1,), -pi / 2)])
        self.assertEqual(M.shape, (4, 4, 4))
        self.assertGreaterEqual(k, 0)


class TestBall(QCRTestCase):
    """Breadth-first ball, collision certificate and meet-in-the-middle distances."""

    @classmethod
    def setUpClass(cls):
        cls.ball = CliffordTBall(3).build()

    def test_level_sizes_and_certificate(self):
        self.assertEqual(self.ball.levels, [1, 34, 569, 6744])
        self.assertTrue(self.ball.check_collisions())

    def test_distance_returns_an_exactly_equal_optimal_word(self):
        rng = np.random.default_rng(1)
        pool = alphabet()
        for length in (1, 2, 3, 4, 5):
            for _ in range(5):
                circ = [pool[i] for i in rng.integers(len(pool), size=length)]
                d, word, exact = self.ball.distance(circ)
                self.assertTrue(exact)
                self.assertLessEqual(d, length)
                self.assertEqual(len(word), d)
                self.assertTrue(exact_equal_up_to_phase(circ, word))

    def test_distance_is_a_lower_bound_beyond_reach(self):
        rng = np.random.default_rng(2)
        pool = alphabet()
        for _ in range(5):
            circ = [pool[i] for i in rng.integers(len(pool), size=12)]
            d, word, exact = self.ball.distance(circ)
            if not exact:
                self.assertIsNone(word)
                self.assertEqual(d, 6)  # certified: farther than radius 3 + spheres up to 2

    def test_planted_circuit_keeps_the_unitary(self):
        rng = np.random.default_rng(3)
        pool = alphabet()
        core = [pool[i] for i in rng.integers(len(pool), size=3)]
        circ = planted_circuit(core, 30, self.ball, rng)
        self.assertGreaterEqual(len(circ), 30)
        self.assertTrue(exact_equal_up_to_phase(core, circ))
