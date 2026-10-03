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


"""Exact-window local optimisation (extension).

`Oracle` holds the full distance table of a <=3-qubit Clifford group (from
`tableau.cayley_bfs`) and returns provably optimal words.  `exact_window_reduce`
generalises V2: every contiguous window touching <= k wires (k = oracle
qubits) and of length <= w is replaced by an optimal word for its unitary,
so the output is w-locally optimal w.r.t. the *global* group metric, not
w.r.t. a depth-r ball.
"""

from __future__ import annotations

import numpy as np

from .gates import Gate, make_pool
from .reducer import shuffle_commuting
from .tableau import Tableau, cayley_bfs


class Oracle:
    def __init__(self, kind: str, n: int, codes=None, dists=None):
        self.kind, self.n = kind, n
        self.pool = make_pool(kind, n)
        if codes is None:
            codes, dists, _, _, _ = cayley_bfs(n, self.pool, verbose=False)
        self.codes, self.dists = codes, dists
        self.tb = Tableau(n)
        from .gates import gate_matrix

        self.tables = {g: [int(x) for x in self.tb.row_table(gate_matrix(g, n))] for g in self.pool}
        self.inv = {g: [t[t[t[x]]] for x in range(len(t))] for g, t in self.tables.items()}
        self.R, self.rows, self.mask = self.tb.R, self.tb.rows, self.tb.mask
        self.ident = self.tb.identity_code()

    def step(self, code: int, g: Gate, inverse=False) -> int:
        T = self.inv[g] if inverse else self.tables[g]
        out, R, m = 0, self.R, self.mask
        for r in range(self.rows):
            out |= T[(code >> (R * r)) & m] << (R * r)
        return out

    def dist(self, code: int) -> int:
        i = int(np.searchsorted(self.codes, np.uint64(code)))
        return int(self.dists[i])

    def code_of(self, gates) -> int:
        c = self.ident
        for g in gates:
            c = self.step(c, g)
        return c

    def optimal_word(self, code: int) -> list[Gate]:
        d = self.dist(code)
        word = []
        while d > 0:
            for g in self.pool:
                prev = self.step(code, g, inverse=True)
                if self.dist(prev) == d - 1:
                    word.append(g)
                    code, d = prev, d - 1
                    break
            else:
                raise RuntimeError("descent failed")
        return word[::-1]


def _best_window(gates, i, oracle: Oracle, w: int, r: int = 10**9):
    """Scan windows starting at i; return (saving, j, mapping, code)."""
    k = oracle.n
    m: dict[int, int] = {}
    code = oracle.ident
    best = (0, None, None, None)
    for j in range(i, min(len(gates), i + w)):
        g = gates[j]
        for q in g.qubits:
            if q not in m:
                if len(m) == k:
                    return best
                m[q] = len(m)
        code = oracle.step(code, g.remap(m))
        L = j - i + 1
        if L >= 2:
            dd = oracle.dist(code)
            sav = L - dd if dd <= r else 0
            if sav > best[0]:
                best = (sav, j, dict(m), code)
    return best


def sweep(gates, oracle: Oracle, w: int, r: int = 10**9):
    g = list(gates)
    changed = False
    i = 0
    while i < len(g) - 1:
        sav, j, m, code = _best_window(g, i, oracle, w, r)
        if sav > 0:
            inv = {v: q for q, v in m.items()}
            word = oracle.optimal_word(code)
            if all(q in inv for x in word for q in x.qubits):
                g = g[:i] + [x.remap(inv) for x in word] + g[j + 1 :]
                changed = True
                i = max(0, i - w)  # neighbours may now reduce
                continue
        i += 1
    return g, changed


def exact_window_reduce(gates, oracle: Oracle, w: int = 20, rng=None, patience: int = 30, r: int = 10**9):
    """Sweep to a fixed point, then perturb by random commuting swaps; stop
    after `patience` perturbations without improvement."""
    rng = rng or np.random.default_rng(0)
    g, _ = sweep(gates, oracle, w, r)
    best = g
    fails = 0
    while fails < patience:
        h, _ = sweep(shuffle_commuting(best, rng), oracle, w, r)
        if len(h) < len(best):
            best, fails = h, 0
        else:
            fails += 1
    return best


# ------------------------------------------------------------ convex windows
def _convex_window(gates, i, oracle: Oracle, w: int, W: tuple, r: int = 10**9):
    """Grow a convex set S of gates starting at i whose qubits lie in the wire
    set W.  Gates disjoint from W commute with S and are skipped; a gate that
    touches W but leaves it (or touches a blocked wire) blocks its W-wires.
    All gates of S commute past the skipped/blocked ones, so S can be moved
    to position i and replaced by an optimal word.  Returns the best prefix."""
    Wset = set(W)
    m = {q: k for k, q in enumerate(W)}
    blocked: set = set()
    code = oracle.ident
    S: list[int] = []
    best = (0, None, None)
    j = i
    while j < len(gates) and len(S) < w:
        g = gates[j]
        qs = set(g.qubits)
        if not (qs & Wset):
            j += 1
            continue
        if qs <= Wset and not (qs & blocked):
            S.append(j)
            code = oracle.step(code, g.remap(m))
            if len(S) >= 2:
                dd = oracle.dist(code)
                sav = len(S) - dd if dd <= r else 0
                if sav > best[0]:
                    best = (sav, list(S), code)
        else:
            blocked |= qs & Wset
            if blocked == Wset:
                break
        j += 1
    return best


def convex_sweep(gates, oracle: Oracle, w: int, n: int, r: int = 10**9):
    from itertools import combinations

    k = oracle.n
    subsets = [W for W in combinations(range(n), min(k, n))]
    g = list(gates)
    changed = False
    i = 0
    while i < len(g) - 1:
        cands = [W for W in subsets if set(g[i].qubits) <= set(W)]
        best = (0, None, None, None)
        for W in cands:
            sav, S, code = _convex_window(g, i, oracle, w, W, r)
            if sav > best[0]:
                best = (sav, S, code, W)
        sav, S, code, W = best
        if sav > 0:
            inv = dict(enumerate(W))
            word = [x.remap(inv) for x in oracle.optimal_word(code)]
            Sset = set(S)
            rest = [g[t] for t in range(i, S[-1] + 1) if t not in Sset]
            g = g[:i] + word + rest + g[S[-1] + 1 :]
            changed = True
            i = max(0, i - w)
            continue
        i += 1
    return g, changed


def convex_window_reduce(gates, oracle: Oracle, n: int, w: int = 24, rng=None, patience: int = 30, r: int = 10**9):
    rng = rng or np.random.default_rng(0)
    best, _ = convex_sweep(gates, oracle, w, n, r)
    fails = 0
    while fails < patience:
        h, _ = convex_sweep(shuffle_commuting(best, rng), oracle, w, n, r)
        if len(h) < len(best):
            best, fails = h, 0
        else:
            fails += 1
    return best


# ------------------------------------------------- adaptive convex windows
def adaptive_convex_window(gates, i, k: int, w: int, score, max_jump: int | None = None):
    """Grow a set S starting at gate i on at most k wires, choosing the wires
    adaptively (no enumeration of wire subsets, so it scales to any n).

    Scanning forward, a gate joins S if (a) its wires fit into W (``|W| <= k``)
    and (b) it commutes with every gate already jumped over that shares a
    wire with it.  Otherwise it is jumped over.  Then every gate of S can be
    moved left past all jumped gates (it commutes with each of them), so
    S can be gathered at position i and replaced: this is convexity up to
    *semantic* commutation (e.g. diagonal gates CZ/RZ commute through each
    other), the circuit-DAG generalisation of the paper's random swaps.

    `score(mapped_gates, W)` -> (saving, payload); returns the best prefix as
    (saving, S_indices, W, payload)."""
    from .reducer import commutes

    W: dict[int, int] = {}
    jumped: dict[int, list] = {}  # wire -> jumped gates touching it
    S: list[int] = []
    mapped = []
    best = (0, None, None, None)
    j = i
    misses = 0
    max_jump = 4 * w if max_jump is None else max_jump
    while j < len(gates) and len(S) < w and misses < max_jump:
        g = gates[j]
        new = [q for q in g.qubits if q not in W]
        ok = len(W) + len(new) <= k and all(commutes(t, g) for q in g.qubits for t in jumped.get(q, ()))
        if ok:
            for q in new:
                W[q] = len(W)
            S.append(j)
            mapped.append(g.remap(W))
            if len(S) >= 2:
                sav, payload = score(mapped, W)
                if sav > best[0]:
                    best = (sav, list(S), dict(W), payload)
        else:
            misses += 1
            for q in g.qubits:
                jumped.setdefault(q, []).append(g)
        j += 1
    return best


def oracle_score(oracle: Oracle):
    def score(mapped, W):
        code = oracle.code_of(mapped)
        return len(mapped) - oracle.dist(code), code

    return score


def adaptive_sweep(gates, w: int, k: int, score, rewrite, max_jump=None):
    """rewrite(payload, W) -> replacement gates in the window's local wires."""
    g = list(gates)
    i = 0
    while i < len(g) - 1:
        sav, S, W, payload = adaptive_convex_window(g, i, k, w, score, max_jump)
        if sav > 0:
            inv = {v: q for q, v in W.items()}
            word = [x.remap(inv) for x in rewrite(payload, W)]
            Sset = set(S)
            rest = [g[t] for t in range(i, S[-1] + 1) if t not in Sset]
            g = g[:i] + word + rest + g[S[-1] + 1 :]
            i = max(0, i - w)
            continue
        i += 1
    return g


def adaptive_reduce(gates, w, k, score, rewrite, rng=None, patience=30, restarts=1, max_jump=None):
    rng = rng or np.random.default_rng(0)
    best_all = None
    for _ in range(restarts):
        best = adaptive_sweep(gates, w, k, score, rewrite, max_jump)
        fails = 0
        while fails < patience:
            h = adaptive_sweep(shuffle_commuting(best, rng), w, k, score, rewrite, max_jump)
            if len(h) < len(best):
                best, fails = h, 0
            else:
                fails += 1
        if best_all is None or len(best) < len(best_all):
            best_all = best
    return best_all
