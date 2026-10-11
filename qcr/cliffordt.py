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

r"""Exact ground truth beyond Clifford: two-qubit Clifford+T circuits in exact arithmetic.

With rotation angles in :math:`\{\pm\pi/4, \pm\pi/2, \pi\}` the gates no longer generate a finite group, so the
Clifford tables of :mod:`qcr.tableau` do not apply. Up to a global phase, however, every such gate has entries in
the ring :math:`\mathbb D[\omega] = \mathbb Z[1/\sqrt2, \omega]`, :math:`\omega = e^{i\pi/4}`. We represent a
unitary exactly as :math:`U = M/\sqrt2^{\,k}` with :math:`M` a matrix over :math:`\mathbb Z[\omega]` (four integer
coefficients per entry in the basis :math:`1, \omega, \omega^2, \omega^3`), so equality is decided without
tolerances.

* Projective canonical form: :math:`U_{ij}\,\overline{U_p}` for the first non-zero entry :math:`U_p` is invariant
  under global phase and determines :math:`U` up to phase; it is reduced to the smallest power of :math:`\sqrt2`.
  Its 64-bit hash keys a breadth-first ball; :meth:`CliffordTBall.check_collisions` certifies that no two
  distinct canonical forms share a key.
* :meth:`CliffordTBall.distance` returns the exact word length of any element whose length is at most
  :math:`R + R'` (meet in the middle with the stored spheres), or a certified lower bound otherwise.
* :func:`planted_circuit` hides a known short circuit inside a long one, which gives long benchmark circuits with
  certified optimal length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np

from .gates import Gate, circuit_unitary

N_QUBITS = 2
DIM = 2**N_QUBITS
ANGLES = (pi / 4, -pi / 4, pi / 2, -pi / 2, pi)
MS_ANGLES = (pi / 4, -pi / 4, pi / 2, -pi / 2)
OMEGA = np.exp(1j * pi / 4)
_POW = OMEGA ** np.arange(4)

# ring multiplication in Z[w], w^4 = -1:  (x*y)[c] = sum_ab x[a] y[b] T[a,b,c]
_T = np.zeros((4, 4, 4), dtype=np.int64)
for _a in range(4):
    for _b in range(4):
        _s = _a + _b
        _T[_a, _b, _s % 4] = 1 if _s < 4 else -1


def ring_mul(x, y):
    """Elementwise product of arrays of Z[w] elements (last axis: 4 coefficients)."""
    return np.einsum("...a,...b,abc->...c", x, y, _T)


def ring_conj(x):
    """Complex conjugate: w -> w^7 = -w^3."""
    return np.stack([x[..., 0], -x[..., 3], -x[..., 2], -x[..., 1]], axis=-1)


def ring_value(x, k=0):
    """Complex value of x / sqrt(2)^k."""
    return (x @ _POW) / np.sqrt(2.0) ** np.asarray(k)[..., None, None] if np.ndim(k) else (x @ _POW) / np.sqrt(2.0) ** k


_SQRT2 = np.array([0, 1, 0, -1], dtype=np.int64)  # w - w^3 = sqrt(2)


def _reduce(M, k):
    """Divide M by sqrt(2) as long as every entry stays in Z[w] (batch over the first axis)."""
    M, k = M.copy(), k.copy()
    while True:
        Y = ring_mul(M, np.broadcast_to(_SQRT2, M.shape))  # M * sqrt2; divisible by 2 <=> M divisible by sqrt2
        ok = (k > 0) & np.all((Y % 2 == 0).reshape(len(M), -1), axis=1)
        if not ok.any():
            return M, k
        M[ok] = Y[ok] // 2
        k[ok] -= 1


def matmul(A, ka, B, kb):
    """(A / sqrt2^ka) @ (B / sqrt2^kb) for a single A and a batch B; result reduced."""
    C = np.einsum("ila,nljb,abc->nijc", A, B, _T)
    return _reduce(C, kb + ka)


def canonical(M, k):
    """Projective canonical forms (batch): pivot-normalised, reduced. Returns (forms, exponents)."""
    flat = M.reshape(len(M), DIM * DIM, 4)
    nz = np.any(flat != 0, axis=2)
    piv = flat[np.arange(len(M)), np.argmax(nz, axis=1)]
    P = ring_mul(flat, ring_conj(piv)[:, None, :]).reshape(M.shape)
    return _reduce(P, 2 * k)


_HC = (np.random.default_rng(20251003).integers(1, 2**62, size=DIM * DIM * 4 + 1, dtype=np.int64) * 2 + 1).astype(
    np.uint64
)


def keys_of(P, kp):
    """64-bit keys of canonical forms."""
    v = np.concatenate([P.reshape(len(P), -1), kp[:, None]], axis=1).astype(np.int64).view(np.uint64)
    return (v * _HC).sum(axis=1, dtype=np.uint64)


# ------------------------------------------------------------------ exact generators
def _kron(A, B):
    """Kronecker product of two matrices over Z[w]."""
    n, m = A.shape[0], B.shape[0]
    out = np.zeros((n * m, n * m, 4), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for r in range(m):
                for s in range(m):
                    out[i * m + r, j * m + s] = ring_mul(A[i, j], B[r, s])
    return out


def _e(c0=0, c1=0, c2=0, c3=0):
    return np.array([c0, c1, c2, c3], dtype=np.int64)


def _phase_diag(steps):
    """diag(w^s) for integer steps s (exact)."""
    n = len(steps)
    D = np.zeros((n, n, 4), dtype=np.int64)
    for i, s in enumerate(steps):
        s %= 8
        D[i, i] = _e(*(np.eye(4, dtype=np.int64)[s % 4] * (1 if s < 4 else -1)))
    return D


_I2 = _phase_diag([0, 0])
_H = np.array([[_e(1), _e(1)], [_e(1), _e(-1)]])  # times 1/sqrt2
_S = _phase_diag([0, 2])
_SD = _phase_diag([0, 6])


def _mm(A, B):
    return np.einsum("ila,ljb,abc->ijc", A, B, _T)


def _rot1(axis, steps):
    """Single-qubit rotation by steps * pi/4 about axis, up to phase, as (M, k)."""
    Z = _phase_diag([0, steps])  # rz(steps*pi/4) ~ diag(1, w^steps)
    if axis == "z":
        return Z, 0
    X = _mm(_mm(_H, Z), _H)  # H Z H, exponent 2
    if axis == "x":
        return X, 2
    return _mm(_mm(_S, X), _SD), 2  # ry = S rx S^dag


def exact_gate(g: Gate):
    """Exact (M, k) of a 2-qubit gate (up to global phase); qubit 0 is the most significant factor."""
    steps = int(round(g.theta / (pi / 4)))
    if abs(steps * pi / 4 - g.theta) > 1e-9:
        raise ValueError(f"angle not a multiple of pi/4: {g}")
    if g.name in ("rx", "ry", "rz"):
        R, k = _rot1(g.name[1], steps)
        M = _kron(R, _I2) if g.qubits[0] == 0 else _kron(_I2, R)
        return M, k
    if g.name == "rxx":  # exp(-i t XX/2) ~ (H x H) diag(1, w^s, w^s, 1) (H x H)
        HH = _kron(_H, _H)
        return _mm(_mm(HH, _phase_diag([0, steps, steps, 0])), HH), 4
    raise ValueError(g)


def alphabet():
    """The 2-qubit Clifford+T alphabet of the ion-trap grid: rx/ry/rz on each qubit and rxx."""
    pool = [Gate(a, (q,), t) for a in ("rx", "ry", "rz") for q in range(N_QUBITS) for t in ANGLES]
    pool += [Gate("rxx", (0, 1), t) for t in MS_ANGLES]
    return pool


def to_exact(gates):
    """Exact (M, k) of a 2-qubit circuit."""
    M = np.zeros((1, DIM, DIM, 4), dtype=np.int64)
    M[0, np.arange(DIM), np.arange(DIM), 0] = 1
    k = np.zeros(1, dtype=np.int64)
    for g in gates:
        G, kg = exact_gate(g)
        M, k = matmul(G, kg, M, k)
    return M[0], int(k[0])


def exact_equal_up_to_phase(a, b) -> bool:
    """Exact projective equality of two 2-qubit Clifford+T circuits."""
    Ma, ka = to_exact(a)
    Mb, kb = to_exact(b)
    Pa, pa = canonical(Ma[None], np.array([ka]))
    Pb, pb = canonical(Mb[None], np.array([kb]))
    return bool(pa[0] == pb[0] and np.array_equal(Pa, Pb))


# ------------------------------------------------------------------ ball
@dataclass
class CliffordTBall:
    """Breadth-first ball of radius ``radius`` in the projective group generated by :func:`alphabet`."""

    radius: int
    pool: list = field(default_factory=alphabet)
    keys: np.ndarray = None  # sorted
    dist: np.ndarray = None
    parent: np.ndarray = None  # index into the unsorted arrays
    gate: np.ndarray = None
    order: np.ndarray = None  # sorted position -> unsorted index
    spheres: list = field(default_factory=list)  # [(M, k, unsorted indices)] for d <= radius - 1
    levels: list = field(default_factory=list)
    forms: list = field(default_factory=list)  # canonical forms per level (for collision certification)

    def build(self, verbose=False, keep_matrices_up_to=None):
        keep = self.radius - 1 if keep_matrices_up_to is None else keep_matrices_up_to
        gens = [exact_gate(g) for g in self.pool]
        M = np.zeros((1, DIM, DIM, 4), dtype=np.int64)
        M[0, np.arange(DIM), np.arange(DIM), 0] = 1
        k = np.zeros(1, dtype=np.int64)
        P, kp = canonical(M, k)
        keys = [keys_of(P, kp)]
        self.forms = [(P, kp)]
        dist, parent, gate = [np.zeros(1, np.int64)], [np.array([-1])], [np.array([-1])]
        self.spheres = [(M, k, np.array([0]))]
        seen = np.sort(keys[0])
        frontier, fk, fidx = M, k, np.array([0])
        total = 1
        self.levels = [1]
        for d in range(1, self.radius + 1):
            newM, newk, newkey, par, gat, newP, newkp = [], [], [], [], [], [], []
            for gi, (G, kg) in enumerate(gens):
                C, kc = matmul(G, kg, frontier, fk)
                Pc, kpc = canonical(C, kc)
                kc_keys = keys_of(Pc, kpc)
                pos = np.searchsorted(seen, kc_keys)
                fresh = (pos >= len(seen)) | (seen[np.minimum(pos, len(seen) - 1)] != kc_keys)
                newM.append(C[fresh])
                newk.append(kc[fresh])
                newkey.append(kc_keys[fresh])
                par.append(fidx[fresh])
                gat.append(np.full(fresh.sum(), gi))
                newP.append(Pc[fresh])
                newkp.append(kpc[fresh])
            newkey = np.concatenate(newkey)
            _, first = np.unique(newkey, return_index=True)
            first = np.sort(first)
            M_d = np.concatenate(newM)[first]
            k_d = np.concatenate(newk)[first]
            idx = np.arange(total, total + len(first))
            keys.append(newkey[first])
            dist.append(np.full(len(first), d))
            parent.append(np.concatenate(par)[first])
            gate.append(np.concatenate(gat)[first])
            self.forms.append((np.concatenate(newP)[first], np.concatenate(newkp)[first]))
            if d <= keep:
                self.spheres.append((M_d, k_d, idx))
            seen = np.sort(np.concatenate([seen, newkey[first]]))
            frontier, fk, fidx = M_d, k_d, idx
            total += len(first)
            self.levels.append(len(first))
            if verbose:
                print(f"  depth {d}: +{len(first):,} (total {total:,})", flush=True)
        allkeys = np.concatenate(keys)
        self.order = np.argsort(allkeys, kind="stable")
        self.keys = allkeys[self.order]
        self.dist = np.concatenate(dist)
        self.parent = np.concatenate(parent)
        self.gate = np.concatenate(gate)
        return self

    def check_collisions(self) -> bool:
        """Certify that distinct canonical forms never share a key (the ball is then exact)."""
        P = np.concatenate([f[0].reshape(len(f[0]), -1) for f in self.forms])
        kp = np.concatenate([f[1] for f in self.forms])
        rows = np.ascontiguousarray(np.concatenate([P, kp[:, None]], axis=1))
        n_forms = len(np.unique(rows.view([("", rows.dtype)] * rows.shape[1])))
        return n_forms == len(self.keys) == len(np.unique(self.keys))

    def _lookup(self, keys):
        pos = np.searchsorted(self.keys, keys)
        pos = np.minimum(pos, len(self.keys) - 1)
        hit = self.keys[pos] == keys
        return np.where(hit, self.order[pos], -1)

    def word(self, i: int):
        """Geodesic word of the element with (unsorted) index i."""
        out = []
        while self.parent[i] >= 0:
            out.append(self.pool[self.gate[i]])
            i = self.parent[i]
        return out[::-1]

    def distance(self, gates):
        """Exact optimal length of a 2-qubit Clifford+T circuit, with an optimal word.

        Returns ``(length, word, exact)``. If ``exact`` is false, ``length`` is a certified lower bound
        (the element is farther than ``radius + len(spheres) - 1``) and ``word`` is ``None``.
        """
        M, k = to_exact(gates)
        best, best_word = None, None
        for j, (S, kS, idx) in enumerate(self.spheres):
            if best is not None and j >= best:
                break
                # U = X Y with |Y| = j  =>  X = U Y^dag; Y^dag is the conjugate transpose (exact)
            Yd = ring_conj(np.swapaxes(S, 1, 2))
            C = np.einsum("ila,nljb,abc->nijc", M, Yd, _T)  # (U Y^dag)_ij = sum_l U_il (Y^dag)_lj
            C, kc = _reduce(C, kS + k)
            P, kp = canonical(C, kc)
            hit = self._lookup(keys_of(P, kp))
            ok = np.flatnonzero(hit >= 0)
            if len(ok):
                cand = self.dist[hit[ok]] + j
                t = int(np.argmin(cand))
                if best is None or cand[t] < best:
                    best = int(cand[t])
                    # Y acts first in time, then X
                    best_word = self.word(int(idx[ok[t]])) + self.word(int(hit[ok[t]]))
        if best is None:
            return self.radius + len(self.spheres), None, False
        return best, best_word, True


def _inverse(word):
    return [Gate(g.name, g.qubits, pi if abs(abs(g.theta) - pi) < 1e-12 else -g.theta) for g in reversed(word)]


def planted_circuit(
    core, length: int, ball: CliffordTBall, rng: np.random.Generator, max_window: int = 4, max_prefix: int = 3
):
    """Hide ``core`` in a circuit of at least ``length`` gates with exactly the same unitary.

    Scrambling step: pick a random window ``W`` of the current circuit (up to ``max_window`` gates) and a random
    word ``r`` (up to ``max_prefix`` gates), and replace ``W`` by ``r`` followed by a geodesic ``s`` of
    ``W r^{-1}`` (computed exactly in ``ball``), so that ``s r = W``. Windows overlap as the circuit grows, so the
    planted structure is not local. The optimal length of the result equals that of ``core``.
    """
    out = list(core)
    pool = ball.pool
    while len(out) < length:
        m = int(rng.integers(1, max_window + 1))
        i = int(rng.integers(0, max(1, len(out) - m + 1)))
        W = out[i : i + m]
        r = [pool[t] for t in rng.integers(len(pool), size=int(rng.integers(1, max_prefix + 1)))]
        _, s_word, exact = ball.distance(_inverse(r) + W)
        if not exact:
            continue
        out = out[:i] + r + s_word + out[i + m :]
    return out


def numeric(gates) -> np.ndarray:
    """Floating-point unitary (for cross-checks)."""
    return circuit_unitary(gates, N_QUBITS)
