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

"""Exact Clifford ground truth: group orders, diameters, parity, optimal words."""

import numpy as np

from qcr.compute_graph import build
from qcr.exact import Oracle, exact_window_reduce
from qcr.gates import make_pool, random_circuit
from qcr.tableau import cayley_bfs

from .base import QCRTestCase


class TestCayleyGraph(QCRTestCase):
    """Complete enumeration of small Clifford groups."""

    @classmethod
    def setUpClass(cls):
        cls.pool = make_pool("ion_pm", 2)
        cls.codes, cls.dists, cls.levels, cls.tb, cls.tables = cayley_bfs(2, cls.pool, verbose=False)

    def test_one_qubit_group_order(self):
        codes, *_ = cayley_bfs(1, make_pool("ion_pm", 1), verbose=False)
        self.assertEqual(len(codes), 24)

    def test_two_qubit_group_order_and_diameter(self):
        self.assertEqual(len(self.codes), 11_520)
        self.assertEqual(int(self.dists.max()), 10)

    def test_two_qubit_cayley_graph_is_bipartite(self):
        """Theorem (parity): every generator changes the distance by exactly one for n = 2."""
        dist = dict(zip(self.codes.tolist(), self.dists.tolist()))
        for T in self.tables.values():
            images = self.tb.apply(T, self.codes)
            delta = np.abs(np.array([dist[int(c)] for c in images]) - self.dists.astype(int))
            self.assertTrue(np.all(delta == 1))

    def test_compute_graph_matches_table_one(self):
        """The paper's compute graph is a ball of the Cayley graph (Table I node counts)."""
        expected = {1: 15, 2: 114, 3: 584, 4: 2024}
        cg = build(2, self.pool, 4)
        for d, n in expected.items():
            self.assertEqual(int(np.sum(cg.dist <= d)), n)
        self.assertEqual(list(np.cumsum(self.levels[:5])), [1, *expected.values()])


class TestExactOracle(QCRTestCase):
    """Provably optimal windows."""

    @classmethod
    def setUpClass(cls):
        cls.oracle = Oracle("ion_pm", 2)

    def test_optimal_word_has_optimal_length(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            circ = random_circuit(self.oracle.pool, 30, rng)
            code = self.oracle.code_of(circ)
            word = self.oracle.optimal_word(code)
            self.assertEqual(len(word), self.oracle.dist(code))
            self.assertUnitaryEqual(word, circ, 2, up_to_phase=True)

    def test_window_above_diameter_is_globally_optimal(self):
        """Theorem (exactness threshold): w > D gives a geodesic."""
        rng = np.random.default_rng(2)
        for _ in range(5):
            circ = random_circuit(self.oracle.pool, 60, rng)
            out = exact_window_reduce(circ, self.oracle, w=11, rng=rng, patience=5)
            self.assertEqual(len(out), self.oracle.dist(self.oracle.code_of(circ)))
            self.assertUnitaryEqual(out, circ, 2, up_to_phase=True)
