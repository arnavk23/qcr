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


"""Replay certificates for the exact optimiser (any circuit width).

``Optimizer(..., Config(log=True)).run`` records every accepted step: the commuting
swaps of each shuffle and every window rewrite ``(i, S, word)``, which replaces
``g[i..S[-1]]`` by ``word`` followed by the gates of that range not in ``S``.
:func:`replay` re-executes the log from the input and checks each step on its own
wires only (at most a few qubits), without trusting the optimiser's databases,
canonical forms or commutation rules:

* a swap is valid if the two gates commute (their unitaries, on their joint wires);
* a rewrite is valid if every skipped gate commutes with every later window gate
  it is moved past, and ``U(word) = c U(S)`` for a scalar ``c`` (tracked);
* the final global-phase repair is checked as a local difference between the
  replayed circuit and the output, on the closure of the wires it touches.

Exact equivalence follows when the replayed circuit equals the output gate for gate,
and the product of all scalars is 1. ``err`` sums the Frobenius residuals
``||U(word) - c U(S)||``, a bound on the operator-norm distance caused by rounding.
"""

from __future__ import annotations

import numpy as np

from .gates import Gate, circuit_unitary

TOL = 1e-8
MAX_LOCAL = 12  # widest local check (phase repair closure)


def _local(gates, wires):
    m = {q: i for i, q in enumerate(wires)}
    return circuit_unitary([g.remap(m) for g in gates], len(wires))


def _wires(gates):
    return sorted({q for g in gates for q in g.qubits})


class _Checker:
    def __init__(self):
        self.memo = {}
        self.err = 0.0

    def commute(self, a: Gate, b: Gate) -> bool:
        if not set(a.qubits) & set(b.qubits):
            return True
        k = (a, b)
        if k not in self.memo:
            w = _wires([a, b])
            self.memo[k] = bool(np.allclose(_local([a, b], w), _local([b, a], w), atol=TOL, rtol=0))
        return self.memo[k]

    def scalar(self, old, new):
        """c with U(new) = c U(old) on the joint wires; raises if there is none."""
        w = _wires(old + new)
        if len(w) > MAX_LOCAL:
            raise ValueError(f"local check on {len(w)} wires")
        A, B = _local(old, w), _local(new, w)
        c = np.trace(A.conj().T @ B) / A.shape[0]
        if abs(abs(c) - 1) > TOL:
            raise ValueError(f"not equal up to a scalar (|c| = {abs(c)})")
        c /= abs(c)
        r = float(np.linalg.norm(B - c * A))
        if r > TOL * A.shape[0]:
            raise ValueError(f"residual {r}")
        self.err += r
        return c


def _closure(seg, wires):
    """Wires reachable from `wires` through the gates of `seg` (multi-qubit gates link wires)."""
    W = set(wires)
    grow = True
    while grow:
        grow = False
        for g in seg:
            if set(g.qubits) & W and not set(g.qubits) <= W:
                W |= set(g.qubits)
                grow = True
    return sorted(W)


def _phase_diff(chk, a, b):
    """Scalar c with circuit(b) = c circuit(a), where a and b differ only in a local patch."""
    lo = 0
    while lo < min(len(a), len(b)) and a[lo] == b[lo]:
        lo += 1
    ha, hb = len(a), len(b)
    while ha > lo and hb > lo and a[ha - 1] == b[hb - 1]:
        ha, hb = ha - 1, hb - 1
    sa, sb = a[lo:ha], b[lo:hb]
    if not sa and not sb:
        return 1.0 + 0j
    changed = set(_wires(sa) if not sb else _wires(sb)) | set(_wires(sa))
    # gates of the two segments outside the closure must agree: compare them in order
    W = _closure(sa + sb, changed)
    outa = [g for g in sa if not set(g.qubits) & set(W)]
    outb = [g for g in sb if not set(g.qubits) & set(W)]
    if outa != outb:
        raise ValueError("phase repair changed gates outside its wires")
    return chk.scalar([g for g in sa if set(g.qubits) & set(W)], [g for g in sb if set(g.qubits) & set(W)])


def replay(inp, log, out):
    """Check `out` against `inp` with the optimiser's log. Returns a dict (ok, steps, err, reason)."""
    chk = _Checker()
    g = list(inp)
    C = 1.0 + 0j
    steps = 0
    try:
        if log["kept_input"]:
            if list(out) != g:
                raise ValueError("kept_input but output differs from input")
            return dict(ok=True, steps=0, err=0.0, kept_input=True)
        for kind, data in log["chain"]:
            if kind == "shuffle":
                for i in data:
                    if not chk.commute(g[i], g[i + 1]):
                        raise ValueError(f"swap of non-commuting gates at {i}")
                    g[i], g[i + 1] = g[i + 1], g[i]
                    steps += 1
                continue
            for i, S, word in data:
                if S != sorted(set(S)) or S[0] < i or S[-1] >= len(g):
                    raise ValueError(f"bad window indices at {i}")
                Sset = set(S)
                rest = [t for t in range(i, S[-1] + 1) if t not in Sset]
                for r in rest:  # r moves after every window gate s that follows it
                    for s in S:
                        if s > r and not chk.commute(g[r], g[s]):
                            raise ValueError(f"window gate {s} moved past non-commuting gate {r}")
                C *= chk.scalar([g[t] for t in S], list(word))
                g = g[:i] + list(word) + [g[t] for t in rest] + g[S[-1] + 1 :]
                steps += 1
        C *= _phase_diff(chk, g, list(out))
        if abs(C - 1) > 1e-6:
            raise ValueError(f"global phase {C} != 1")
    except (ValueError, IndexError) as e:
        return dict(ok=False, steps=steps, err=chk.err, reason=str(e))
    return dict(ok=True, steps=steps, err=chk.err, phase=float(np.angle(C)))
