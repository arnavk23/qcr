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

"""Layout-aware rewriting.

Local rewriting cannot change *where* a circuit runs, yet on calibrated hardware the placement matters as
much as the gate count: on IBM Heron the layout chosen by one Qiskit optimisation level can have 10% higher
estimated success probability than another's. This module adds the missing degree of freedom.

* :func:`relocate` moves a routed circuit rigidly onto the cheapest region of the chip. It enumerates
  subgraph monomorphisms of the circuit's interaction graph into the coupling graph (anchored at every
  physical qubit, so the search covers the whole chip) and scores each placement with the device's cost
  model. Gates and their order are unchanged, so the circuit stays exactly equivalent up to the qubit
  relabelling.
* :func:`joint_layout` searches over candidate layouts (Qiskit transpilations with different seeds, with and
  without noise awareness) and scores every candidate *after* rewriting and relocation, because the
  layout that is best before optimisation need not be best after it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .gates import Gate
from .hardware import Device


@dataclass
class Placement:
    """Result of :func:`relocate`."""

    gates: list
    mapping: dict  # old physical qubit -> new physical qubit
    cost_before: float
    cost_after: float
    candidates: int = 0


def _cost_tables(gates):
    """Per-logical-qubit 1q gate counts and per-edge 2q gate counts, plus the cost of one gate anywhere."""
    one = {q: Counter() for g in gates for q in g.qubits}
    two = Counter()
    for g in gates:
        if len(g.qubits) == 1:
            one[g.qubits[0]][g.name] += 1
        else:
            two[(g.name, tuple(sorted(g.qubits)))] += 1
    return one, two


def relocate(
    gates,
    dev: Device,
    cost: str = "calibrated",
    per_anchor: int = 300,
    max_anchors: int | None = None,
    readout: bool = True,
):
    """Move a circuit to the cheapest equivalent placement on ``dev``.

    Args:
        gates: circuit on physical qubits of ``dev`` (all two-qubit gates on coupled pairs).
        dev: device with a coupling map (all-to-all devices are returned unchanged).
        cost: cost model of :meth:`qcr.Device.cost_fn` used to score placements.
        per_anchor: number of placements examined per anchor qubit.
        max_anchors: optionally limit the number of anchor qubits (all by default).
        readout: add ``-log(1 - readout error)`` of every active qubit to the score (calibrated cost only), since
            a complete program measures all of them.

    Returns:
        A :class:`Placement`; ``placement.gates`` never costs more than ``gates``.
    """
    import networkx as nx
    from networkx.algorithms import isomorphism

    costf = dev.cost_fn(cost)
    base = sum(costf(g) for g in gates)
    active = sorted({q for g in gates for q in g.qubits})
    ident = {q: q for q in active}
    if dev.coupling is None or not gates:
        return Placement(list(gates), ident, base, base)

    one, two = _cost_tables(gates)
    phys = sorted({q for p in dev.coupling for q in p} | set(range(dev.n)))
    use_meas = readout and cost == "calibrated" and bool(dev.err_meas)

    def meas(p):
        return -np.log1p(-min(dev.err_meas.get(p, 0.0), 0.5)) if use_meas else 0.0

    c1 = {
        q: {p: sum(k * costf(Gate(name, (p,), 0.0)) for name, k in cnt.items()) + meas(p) for p in phys}
        for q, cnt in one.items()
    }
    base += sum(meas(q) for q in active)
    pattern = nx.Graph()
    pattern.add_nodes_from(q for q in active if any(q in e for _, e in two))
    pattern.add_edges_from(e for _, e in two)
    chip = nx.Graph()
    chip.add_nodes_from(phys)
    chip.add_edges_from(tuple(p) for p in dev.coupling)

    def score(m):  # m: logical -> physical
        s = sum(c1[q][m[q]] for q in m if q in c1)
        for (name, (a, b)), k in two.items():
            s += k * costf(Gate(name, tuple(sorted((m[a], m[b]))), 0.0))
        return s

    best_map, best = None, np.inf
    n_cand = 0
    comps = sorted(nx.connected_components(pattern), key=len, reverse=True)
    if comps:
        # place the largest component by anchored enumeration over the whole chip
        comp = pattern.subgraph(comps[0]).copy()
        root = max(comp.degree, key=lambda t: t[1])[0]
        radius = nx.eccentricity(comp, root)
        nx.set_node_attributes(comp, {v: v == root for v in comp}, "anchor")
        anchors = [v for v in phys if chip.degree(v) >= comp.degree(root)]
        if max_anchors is not None:
            anchors = anchors[:max_anchors]
        for v in anchors:
            ball = nx.ego_graph(chip, v, radius=radius)
            nx.set_node_attributes(ball, {u: u == v for u in ball}, "anchor")
            gm = isomorphism.GraphMatcher(ball, comp, node_match=lambda a, b: a["anchor"] == b["anchor"])
            for k, mono in enumerate(gm.subgraph_monomorphisms_iter()):
                if k >= per_anchor:
                    break
                m = {lq: pq for pq, lq in mono.items()}
                # remaining components and 1q-only qubits are placed greedily on free qubits
                m = _complete(m, pattern, comps[1:], active, chip, c1, two, costf)
                if m is None:
                    continue
                n_cand += 1
                s = score(m)
                if s < best - 1e-15:
                    best, best_map = s, m
    else:  # no two-qubit gates: greedy placement only
        best_map = _complete({}, pattern, [], active, chip, c1, two, costf)
        best = score(best_map) if best_map else np.inf

    if best_map is None or best >= base - 1e-15:
        return Placement(list(gates), ident, float(base), float(base), n_cand)
    out = [
        Gate(
            g.name,
            tuple(sorted(best_map[q] for q in g.qubits)) if len(g.qubits) == 2 else (best_map[g.qubits[0]],),
            g.theta,
        )
        for g in gates
    ]
    return Placement(out, best_map, float(base), float(best), n_cand)


def _complete(m, pattern, comps, active, chip, c1, two, costf):
    """Greedily extend a partial placement to all active qubits; ``None`` if impossible."""
    from networkx.algorithms import isomorphism

    m = dict(m)
    used = set(m.values())
    for comp_nodes in comps:
        comp = pattern.subgraph(comp_nodes)
        free = chip.subgraph([v for v in chip if v not in used])
        gm = isomorphism.GraphMatcher(free, comp)
        best, best_m = np.inf, None
        for k, mono in enumerate(gm.subgraph_monomorphisms_iter()):
            if k >= 200:
                break
            cand = {lq: pq for pq, lq in mono.items()}
            s = sum(c1[q][cand[q]] for q in cand if q in c1)
            if s < best:
                best, best_m = s, cand
        if best_m is None:
            return None
        m.update(best_m)
        used |= set(best_m.values())
    for q in active:
        if q in m:
            continue
        free = [v for v in chip if v not in used]
        if not free:
            return None
        p = min(free, key=lambda v: c1[q][v] if q in c1 else 0.0)
        m[q] = p
        used.add(p)
    return m


@dataclass
class LayoutCandidate:
    """One candidate of :func:`joint_layout`."""

    label: str
    gates: list
    cost: float
    stats: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict, repr=False)  # routed / optimised / mapping / relocated circuits


def joint_layout(qc, dev: Device, optimize, backend=None, seeds=(0, 1, 2, 3), level: int = 3, cost="calibrated"):
    """Choose the layout by its cost *after* rewriting.

    Args:
        qc: logical :class:`~qiskit.circuit.QuantumCircuit`.
        dev: device model (coupling map and calibration).
        optimize: callable ``gates -> gates`` (e.g. ``lambda g: Optimizer(...).run(g).gates``).
        backend: optional fake backend; when given, noise-aware candidates are added
            (Qiskit transpilation against :func:`qcr.hardware.coupled_target`).
        seeds: transpiler seeds; each seed gives one candidate layout per mode.
        level: Qiskit optimisation level used to produce candidates.
        cost: cost model used to rank candidates.

    Returns:
        ``(best, candidates)`` where ``best`` is the cheapest :class:`LayoutCandidate`.
    """
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    from .hardware import coupled_target, coupling_list, qiskit_basis
    from .verify import from_qiskit

    costf = dev.cost_fn(cost)
    modes = [("coupling", dict(basis_gates=qiskit_basis(dev), coupling_map=CouplingMap(coupling_list(dev))))]
    if backend is not None:
        modes.append(("noise-aware", dict(target=coupled_target(backend))))
    cands = []
    for mode, kw in modes:
        for s in seeds:
            routed = transpile(qc, optimization_level=level, seed_transpiler=s, **kw)
            g0 = _sorted_2q(g for g in from_qiskit(routed) if g.name not in ("barrier", "delay", "id"))
            g1 = optimize(g0)
            pl = relocate(g1, dev, cost)
            g2 = optimize(pl.gates) if pl.mapping and pl.cost_after < pl.cost_before else pl.gates
            c = sum(costf(g) for g in g2)
            if cost == "calibrated" and dev.err_meas:
                act = {q for g in g2 for q in g.qubits}
                c += sum(-np.log1p(-min(dev.err_meas.get(q, 0.0), 0.5)) for q in act)
            stats = dict(routed=len(g0), optimised=len(g1), relocated=pl.cost_after < pl.cost_before)
            trace = dict(routed=g0, optimised=g1, mapping=pl.mapping, relocated=pl.gates)
            cands.append(LayoutCandidate(f"{mode}/seed{s}", g2, c, stats, trace))
    return min(cands, key=lambda c: c.cost), cands


def _sorted_2q(gates):
    return [Gate(g.name, tuple(sorted(g.qubits)), g.theta) if len(g.qubits) == 2 else g for g in gates]
