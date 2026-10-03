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


"""Local term-replacement reducers V1-RS, V2-DR, V3-RF.

The control flow is a port of matlab_demo/QCOptimDemo/optimCodeGMode1DComp.m::

  loop:
    with prob 0.7: randomly swap commuting neighbours (each pair w.p. 0.6)
    sample a block: start s ~ U{0..L-2}, extra e = min(U{1..L-1-s}, 7)
                    -> block of length e+1 in [2, 8]; resample until the
                    block touches <= dimD wires
    map the block's wires onto 0..k-1 (wire reduction, Fig. 7)
    V1: sample random words of length p < len(block) from the dimD pool
        and accept the first one with the same unitary (up to phase)
    V2: look the block unitary up in the compute-graph database restricted
        to entries strictly shorter than the block
    V3: as V2, but only after a random forest predicts "reducible"
    on success: splice the (wire-lifted) replacement in, run simplereduce,
                recompute the commutation table
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .compute_graph import ComputeGraph
from .gates import Gate, circuit_unitary
from .unitary import matlab_compare, phase_key

TOL = 1e-3  # compare.m threshold used throughout optimCodeGMode1DComp.m


# ----------------------------------------------------------------- helpers
def wire_map(gates):
    W = sorted({q for g in gates for q in g.qubits})
    return W, {w: i for i, w in enumerate(W)}


_commute_cache: dict = {}


def commutes(a: Gate, b: Gate) -> bool:
    key = (a, b)
    r = _commute_cache.get(key)
    if r is None:
        W, m = wire_map([a, b])
        A, B = a.remap(m), b.remap(m)
        k = len(W)
        r = matlab_compare(circuit_unitary([A, B], k), circuit_unitary([B, A], k)) < TOL
        _commute_cache[key] = r
    return r


_identity_cache: dict = {}


def pair_is_identity(a: Gate, b: Gate) -> bool:
    key = (a, b)
    r = _identity_cache.get(key)
    if r is None:
        W, m = wire_map([a, b])
        U = circuit_unitary([a.remap(m), b.remap(m)], len(W))
        r = matlab_compare(U, np.eye(len(U))) < TOL
        _identity_cache[key] = r
    return r


def simplereduce(gates: list[Gate]) -> list[Gate]:
    """Port of simplereduce.m (including its loop bounds)."""
    g = list(gates)
    i = 0
    while i < len(g) - 2:  # MATLAB: while(i<size-1) with 1-based i
        if pair_is_identity(g[i], g[i + 1]):
            g = g[:i] + g[i + 2 :]
        i += 1
    return g


def commute_table(gates):
    return [commutes(gates[i], gates[i + 1]) for i in range(len(gates) - 1)]


def shuffle_commuting(gates, rng, p=0.6):
    g = list(gates)
    com = commute_table(g)
    shuffled = np.zeros(len(g), bool)
    for i, c in enumerate(com):
        if c and not shuffled[i] and rng.random() < p:
            g[i], g[i + 1] = g[i + 1], g[i]
            shuffled[i + 1] = True
    return g


def sample_block(gates, rng, dimD, max_extra=7):
    L = len(gates)
    while True:
        s = int(rng.integers(0, L - 1))
        e = min(int(rng.integers(1, L - s)), max_extra)
        block = gates[s : s + e + 1]
        W, m = wire_map(block)
        if len(W) <= dimD:
            return s, e + 1, W, m


# ----------------------------------------------------------------- engines
class DBEngine:
    """Database retrieval.  mode='scan' mimics the MATLAB linear scan
    (entries with OptLength < block length, tolerance compare);
    mode='hash' is an O(1) dictionary lookup with identical results."""

    def __init__(self, cg: ComputeGraph, mode="hash", include_identity=False):
        self.cg = cg
        self.mode = mode
        self.dimD = cg.n
        # MATLAB's database omits the root (Dists>0), so a block equal to the
        # identity can never be deleted by a lookup.  Faithful by default.
        self.include_identity = include_identity
        order = np.argsort(cg.dist, kind="stable")
        self.sorted_U = cg.unitaries[order]
        self.sorted_ids = order
        self.cum = np.searchsorted(cg.dist[order], np.arange(cg.depth + 2))
        self._words: dict[int, list[int]] = {}

    def word(self, i):
        w = self._words.get(i)
        if w is None:
            w = self._words[i] = self.cg.word(i)
        return w

    def find(self, U: np.ndarray, blocklen: int):
        lo = 0 if self.include_identity else 1
        if self.mode == "hash":
            i = self.cg.index.get(phase_key(U))
            if i is None or not (lo <= self.cg.dist[i] < blocklen):
                return None
            return self.word(i)
        hi = self.cum[min(blocklen, self.cg.depth + 1)]
        start = self.cum[lo]
        C = self.sorted_U[start:hi]
        if len(C) == 0:
            return None
        num = np.einsum("ij,nij->n", U.conj(), C)
        s = num / np.vdot(U, U)
        res = np.linalg.norm((U[None] * s[:, None, None] - C).reshape(len(C), -1), axis=1)
        for j in np.flatnonzero(res < TOL * np.sqrt(U.shape[0])):
            if matlab_compare(U, C[j]) < TOL:
                return self.word(int(self.sorted_ids[start + j]))
        return None


class RandomSamplingEngine:
    """V1: sample random words of length p < block length from the dimD pool."""

    def __init__(self, pool, dimD, samples_per_block=300, rng=None):
        self.pool = pool
        self.dimD = dimD
        self.k = samples_per_block
        self.rng = rng or np.random.default_rng()
        from .gates import gate_matrix

        self.mats = np.stack([gate_matrix(g, dimD) for g in pool])
        self.I = np.eye(2**dimD, dtype=complex)

    def find(self, U, blocklen):
        p = self.rng.integers(1, blocklen, size=self.k)  # word lengths 1..blocklen-1
        maxp = int(p.max())
        idx = self.rng.integers(len(self.pool), size=(self.k, maxp))
        V = np.broadcast_to(self.I, (self.k,) + self.I.shape).copy()
        for t in range(maxp):
            act = p > t
            V[act] = np.einsum("nij,njk->nik", self.mats[idx[act, t]], V[act])
        num = np.einsum("ij,nij->n", U.conj(), V)
        s = num / np.vdot(U, U)
        res = np.linalg.norm((U[None] * s[:, None, None] - V).reshape(self.k, -1), axis=1)
        for j in np.flatnonzero(res < TOL * np.sqrt(U.shape[0])):
            if matlab_compare(U, V[j]) < TOL:
                return [int(x) for x in idx[j, : p[j]]]
        return None


# ----------------------------------------------------------------- driver
@dataclass
class Trace:
    times: list = field(default_factory=list)
    lengths: list = field(default_factory=list)
    lookups: int = 0
    skipped: int = 0
    attempts: int = 0


def reduce_circuit(
    gates: list[Gate],
    engine,
    db_pool: list[Gate],
    rng: np.random.Generator,
    target_len: int = 0,
    time_budget: float = 60.0,
    max_attempts: int | None = None,
    stall_attempts: int | None = None,
    classifier=None,
    shuffle_prob: float = 0.7,
):
    dimD = engine.dimD
    g = list(gates)
    tr = Trace()
    t0 = time.perf_counter()
    tr.times.append(0.0)
    tr.lengths.append(len(g))
    last_improve = 0
    while len(g) > max(target_len, 4):
        if time.perf_counter() - t0 > time_budget:
            break
        if max_attempts and tr.attempts >= max_attempts:
            break
        if stall_attempts and tr.attempts - last_improve >= stall_attempts:
            break
        tr.attempts += 1
        if rng.random() < shuffle_prob:
            g = shuffle_commuting(g, rng)
        s, blen, W, m = sample_block(g, rng, dimD)
        block = [x.remap(m) for x in g[s : s + blen]]
        if classifier is not None and not classifier.predict_reducible(block):
            tr.skipped += 1
            continue
        U = circuit_unitary(block, dimD)
        tr.lookups += 1
        w = engine.find(U, blen)
        if w is None:
            continue
        inv = {i: q for q, i in m.items()}
        repl = []
        for k in w:
            gd = db_pool[k]
            if any(q >= len(W) for q in gd.qubits):
                break  # replacement uses a padding wire -> not liftable
            repl.append(gd.remap(inv))
        else:
            g = simplereduce(g[:s] + repl + g[s + blen :])
            tr.times.append(time.perf_counter() - t0)
            tr.lengths.append(len(g))
            last_improve = tr.attempts
    tr.times.append(time.perf_counter() - t0)
    tr.lengths.append(len(g))
    return g, tr
