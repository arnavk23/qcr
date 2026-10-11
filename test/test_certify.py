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

"""Replay certificates: accepted for every exact run, rejected for any tampered output or log."""

from math import pi

import numpy as np

from qcr.certify import replay
from qcr.gates import Gate
from qcr.general import GRIDS, Config, Database, Optimizer
from qcr.hardware import ion

from .base import QCRTestCase
from .test_general import line


class _Certify:
    dev = topo = grid = None
    n = 4

    @classmethod
    def setUpClass(cls):
        cls.dbs = {("phase", cls.topo): Database(GRIDS[cls.grid], cls.topo, "phase", 3, verbose=False)}

    def _runs(self, seeds=3):
        for seed in range(seeds):
            circ = self.random_native(self.dev, self.n, 60, np.random.default_rng(seed))
            cfg = Config(mode="exact", cost="calibrated", patience=2)
            res = Optimizer(self.dev, self.dbs, Config(**{**cfg.__dict__, "log": True})).run(circ, seed=seed)
            self.assertEqual(res.gates, Optimizer(self.dev, self.dbs, cfg).run(circ, seed=seed).gates)
            yield circ, res

    def test_accepts_exact_runs(self):
        for circ, res in self._runs():
            cert = replay(circ, res.log, res.gates)
            self.assertTrue(cert["ok"], cert)
            self.assertUnitaryEqual(res.gates, circ, self.n)

    def test_rejects_changed_angle_and_global_sign(self):
        for circ, res in self._runs(1):
            k = next(i for i, g in enumerate(res.gates) if g.name in ("rz", "rx", "ry"))
            for d in (1e-4, 2 * pi):  # a small error, and an invisible-to-fidelity global -1
                bad = list(res.gates)
                bad[k] = Gate(bad[k].name, bad[k].qubits, bad[k].theta + d)
                self.assertFalse(replay(circ, res.log, bad)["ok"])

    def test_rejects_tampered_log(self):
        for circ, res in self._runs(1):
            log = {**res.log, "chain": [list(x) for x in res.log["chain"]]}
            kind, events = log["chain"][0]
            i, S, word = events[0]
            log["chain"][0][1] = [(i, S, word[:-1] if word else [Gate("x", (S[0] and 0,))])] + events[1:]
            self.assertFalse(replay(circ, log, res.gates)["ok"])


class TestCertifyHeavyHexLike(_Certify, QCRTestCase):
    dev, topo, grid = line(4), "line", "heron"


class TestCertifyTrappedIon(_Certify, QCRTestCase):
    dev, topo, grid = ion(4), "tri", "ion"
