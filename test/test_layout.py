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

"""Layout-aware rewriting: noise-aware relocation."""

from qcr.gates import Gate
from qcr.hardware import Device, ion
from qcr.layout import relocate

from .base import QCRTestCase


def grid(errs2, errs1=None, meas=None):
    """A 2x3 grid device (qubits 0-1-2 / 3-4-5) with given CZ, SX and readout errors."""
    coupling = {frozenset(p) for p in [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]}
    err2 = {("cz", frozenset(p)): e for p, e in errs2.items()}
    err1 = {("sx", q): e for q, e in (errs1 or {}).items()}
    return Device(
        "grid", 6, coupling, ("rz", "sx", "x"), "cz", ("rz",), False, err1, err2, default_err2=1e-2, err_meas=meas or {}
    )


class TestRelocate(QCRTestCase):
    """Relocation is an exact relabelling that never increases the calibrated cost."""

    def setUp(self):
        self.circ = [Gate("sx", (0,)), Gate("cz", (0, 1)), Gate("sx", (1,)), Gate("cz", (1, 2)), Gate("rz", (2,), 0.3)]

    def test_moves_to_better_couplers(self):
        dev = grid({(3, 4): 1e-3, (4, 5): 1e-3})
        pl = relocate(self.circ, dev)
        self.assertLess(pl.cost_after, pl.cost_before)
        self.assertEqual(
            {frozenset(g.qubits) for g in pl.gates if len(g.qubits) == 2}, {frozenset((3, 4)), frozenset((4, 5))}
        )
        self.assertTrue(all(dev.is_native(g) for g in pl.gates))
        # same gates in the same order, relabelled by the mapping
        for a, b in zip(self.circ, pl.gates):
            self.assertEqual((a.name, a.theta), (b.name, b.theta))
            self.assertEqual({pl.mapping[q] for q in a.qubits}, set(b.qubits))

    def test_readout_errors_steer_the_placement(self):
        meas = {q: (0.2 if q < 3 else 0.001) for q in range(6)}
        pl = relocate(self.circ, grid({}, meas=meas))
        self.assertTrue(all(q >= 3 for g in pl.gates for q in g.qubits))
        pl_no = relocate(self.circ, grid({}, meas=meas), readout=False)
        self.assertEqual(pl_no.cost_after, pl_no.cost_before)  # uniform gate errors: nothing to gain

    def test_never_worse_and_identity_when_optimal(self):
        dev = grid({(0, 1): 1e-4, (1, 2): 1e-4})
        pl = relocate(self.circ, dev)
        self.assertEqual(pl.gates, self.circ)
        self.assertEqual(pl.cost_after, pl.cost_before)

    def test_all_to_all_devices_are_unchanged(self):
        circ = [Gate("rxx", (0, 1), 0.2)]
        self.assertEqual(relocate(circ, ion(3)).gates, circ)
