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

"""Device models and cost functions."""

import math
import unittest
from math import pi

from qcr.gates import Gate
from qcr.hardware import EPS_VIRTUAL, Device, ion, paper_device

from .base import QCRTestCase


def line(n):
    return Device("line", n, {frozenset((i, i + 1)) for i in range(n - 1)}, ("rz", "sx", "x"), "cz", ("rz",), False)


class TestDevice(QCRTestCase):
    """Nativeness, connectivity and costs."""

    def test_connectivity(self):
        dev = line(3)
        self.assertTrue(dev.is_native(Gate("cz", (0, 1))))
        self.assertFalse(dev.is_native(Gate("cz", (0, 2))))
        self.assertFalse(dev.is_native(Gate("rx", (0,), 0.1)))

    def test_ion_is_all_to_all_with_continuous_ms(self):
        dev = ion(4)
        self.assertTrue(dev.is_native(Gate("rxx", (0, 3), 0.123)))

    def test_paper_device_enforces_angle_grid(self):
        dev = paper_device("ion", 3)
        self.assertTrue(dev.is_native(Gate("rx", (0,), -pi / 2)))
        self.assertFalse(dev.is_native(Gate("rx", (0,), pi / 4)))

    def test_cost_models(self):
        dev = ion(2)
        g1, g2, rz = Gate("rx", (0,), 1.0), Gate("rxx", (0, 1), 1.0), Gate("rz", (0,), 1.0)
        self.assertEqual(dev.cost_fn("w2")(g2), 10.0)
        cal = dev.cost_fn("calibrated")
        self.assertAlmostEqual(cal(g2), -math.log1p(-1e-3))
        self.assertEqual(cal(rz), EPS_VIRTUAL)  # virtual gate: tie-breaker only
        self.assertLess(cal(g1), cal(g2))
        with self.assertRaises(ValueError):
            dev.cost_fn("depth")

    def test_success_probability_is_exp_of_minus_calibrated_cost(self):
        dev = ion(2)
        gates = [Gate("rx", (0,), 1.0), Gate("rxx", (0, 1), 1.0)] * 3
        cal = dev.cost_fn("calibrated")
        self.assertAlmostEqual(dev.success_probability(gates), math.exp(-sum(cal(g) for g in gates)))


class TestHeron(QCRTestCase):
    """The FakeTorino-based model (needs the ``hardware`` extra)."""

    def test_dead_couplers_are_removed(self):
        try:
            from qcr.hardware import heron

            dev = heron()
        except ImportError:
            raise unittest.SkipTest("qiskit-ibm-runtime not installed") from None
        self.assertEqual(dev.n, 133)
        self.assertTrue(all(dev.err2[("cz", p)] < 1.0 for p in dev.coupling if ("cz", p) in dev.err2))


class TestAngleGrid(QCRTestCase):
    """Angle-grid membership is decided modulo 2 pi."""

    def test_pi_and_minus_pi_are_the_same_rotation(self):
        dev = Device("grid", 1, None, ("rx",), "rxx", (), False, angle_grid=(pi,))
        self.assertTrue(dev.is_native(Gate("rx", (0,), pi)))
        self.assertTrue(dev.is_native(Gate("rx", (0,), -pi)))
        self.assertTrue(dev.is_native(Gate("rx", (0,), 3 * pi)))
        self.assertFalse(dev.is_native(Gate("rx", (0,), pi / 2)))
