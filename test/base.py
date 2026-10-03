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

"""Shared test fixtures."""

import unittest
from math import pi

import numpy as np

from qcr.gates import Gate, circuit_unitary


class QCRTestCase(unittest.TestCase):
    """Base class with circuit-level assertions."""

    def assertUnitaryEqual(self, a, b, n, atol=1e-9, up_to_phase=False):  # noqa: N802
        """Assert two gate lists implement the same unitary."""
        U, V = circuit_unitary(a, n), circuit_unitary(b, n)
        if up_to_phase:
            ov = np.trace(U.conj().T @ V)
            U = U * (ov / abs(ov))
        np.testing.assert_allclose(U, V, atol=atol)

    @staticmethod
    def random_native(dev, n, length, rng, continuous=True):
        """Random circuit over a device's native gates and coupling map."""
        if dev.coupling:
            pairs = [tuple(sorted(p)) for p in dev.coupling]
        else:
            pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        out = []
        for _ in range(length):
            if rng.random() < 0.3:
                theta = rng.uniform(-pi, pi) if dev.native_2q == "rxx" else 0.0
                out.append(Gate(dev.native_2q, pairs[rng.integers(len(pairs))], theta))
                continue
            q = int(rng.integers(n))
            name = dev.native_1q[rng.integers(len(dev.native_1q))]
            if name in ("rx", "ry", "rz"):
                theta = rng.uniform(-pi, pi) if continuous else float(rng.choice([-pi / 2, pi / 2, pi / 4]))
            else:
                theta = 0.0
            out.append(Gate(name, (q,), theta))
        return out
