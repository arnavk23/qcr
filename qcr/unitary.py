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


"""Unitary comparison up to global phase.

`matlab_compare` is a line-by-line port of compare.m.  `phase_key` is a
canonical, hashable fingerprint of a unitary modulo global phase, used to
replace the O(N) linear database scan with an O(1) dictionary lookup (the
two give identical answers on the discrete pools used in the paper).
"""

from __future__ import annotations

import numpy as np


def matlab_compare(R: np.ndarray, F: np.ndarray) -> float:
    """compare.m: least-squares global scale, then spectral norm of the residual."""
    Rv = R.reshape(-1, order="F")
    Fv = F.reshape(-1, order="F")
    scale = np.vdot(Rv, Fv) / np.vdot(Rv, Rv)
    return float(np.linalg.norm(R * scale - F, 2))


def equal_up_to_phase(A: np.ndarray, B: np.ndarray, tol: float = 1e-3) -> bool:
    return matlab_compare(A, B) < tol


def phase_normalize(M: np.ndarray) -> np.ndarray:
    """Batch-capable: divide each matrix by the phase of its first non-negligible entry."""
    flat = M.reshape(M.shape[:-2] + (-1,))
    idx = np.argmax(np.abs(flat) > 1e-6, axis=-1)
    pivot = np.take_along_axis(flat, idx[..., None], axis=-1)
    return M * (np.abs(pivot) / pivot)[..., None]


def phase_key(M: np.ndarray, decimals: int = 6):
    """Hashable key(s) of unitary/unitaries modulo global phase."""
    N = phase_normalize(M)
    flat = N.reshape(N.shape[:-2] + (-1,))
    r = np.round(np.concatenate([flat.real, flat.imag], axis=-1), decimals) + 0.0
    if r.ndim == 1:
        return r.tobytes()
    return [row.tobytes() for row in r]
