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


"""Full compute graph (Sec. II.B of the paper; FullQCGraphIOn3_4.m).

Nodes are unitaries modulo global phase reachable from the identity by
words of length <= depth over a gate pool; edges are labelled by gates.
Because every node is reached first along a shortest path when the graph
is grown breadth-first, storing a BFS parent pointer per node yields the
"perfect factorization" (shortest gate word) the paper stores in its
database.  Edge counting follows the MATLAB code: every node at distance
< depth gets one outgoing edge per pool operator (self-loops/duplicates
included), so ``#edges = |ball(depth-1)| * |pool|``.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gates import Gate, gate_matrix
from .unitary import phase_key, phase_normalize


@dataclass
class ComputeGraph:
    n: int
    pool: list[Gate]
    depth: int
    unitaries: np.ndarray = None  # (N, d, d), phase-normalized
    dist: np.ndarray = None  # (N,)
    parent: np.ndarray = None
    op: np.ndarray = None
    index: dict = field(default_factory=dict)
    level_sizes: list = field(default_factory=list)
    build_seconds: float = 0.0

    @property
    def num_nodes(self) -> int:
        return len(self.dist)

    @property
    def num_edges(self) -> int:
        return int(np.sum(self.dist < self.depth)) * len(self.pool)

    def word(self, i: int) -> list[int]:
        """Shortest op-index word (time order) for node i."""
        w = []
        while self.parent[i] >= 0:
            w.append(int(self.op[i]))
            i = int(self.parent[i])
        return w[::-1]

    def lookup(self, U: np.ndarray):
        return self.index.get(phase_key(U))


def build(n: int, pool: list[Gate], depth: int, max_nodes: int | None = None, verbose=False) -> ComputeGraph:
    t0 = time.time()
    d = 2**n
    mats = np.stack([gate_matrix(g, n) for g in pool])
    I = np.eye(d, dtype=complex)
    U = [I[None]]
    dist = [0]
    parent = [-1]
    op = [-1]
    index = {phase_key(I): 0}
    frontier = I[None]
    frontier_ids = np.array([0])
    level_sizes = [1]
    for lvl in range(1, depth + 1):
        new_U, new_par, new_op = [], [], []
        for k, G in enumerate(mats):
            cand = phase_normalize(np.einsum("ij,njk->nik", G, frontier))
            keys = phase_key(cand)
            for m, key in enumerate(keys):
                if key not in index:
                    index[key] = len(dist) + len(new_par)
                    new_U.append(cand[m])
                    new_par.append(frontier_ids[m])
                    new_op.append(k)
        if not new_U:
            break
        frontier = np.stack(new_U)
        frontier_ids = np.arange(len(dist), len(dist) + len(new_U))
        U.append(frontier)
        dist += [lvl] * len(new_U)
        parent += new_par
        op += new_op
        level_sizes.append(len(new_U))
        if verbose:
            print(f"  depth {lvl}: +{len(new_U)} nodes (total {len(dist)}) {time.time() - t0:.1f}s", flush=True)
        if max_nodes and len(dist) > max_nodes:
            break
    cg = ComputeGraph(n=n, pool=pool, depth=depth)
    cg.unitaries = np.concatenate(U)
    cg.dist = np.array(dist)
    cg.parent = np.array(parent)
    cg.op = np.array(op)
    cg.index = index
    cg.level_sizes = level_sizes
    cg.build_seconds = time.time() - t0
    return cg


def cached_build(n, pool, depth, tag, cache_dir="cache", **kw) -> ComputeGraph:
    p = Path(cache_dir) / f"cg_{tag}_n{n}_d{depth}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    cg = build(n, pool, depth, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(cg, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cg
