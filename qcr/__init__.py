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

"""
=========================================
Quantum circuit reduction (:mod:`qcr`)
=========================================

Exact, hardware-aware local rewriting of quantum circuits.

The package has three layers:

* :mod:`qcr.general` -- the gate-set-agnostic optimiser (exact, state-aware and
  approximate equivalence; calibrated costs; connectivity-aware windows).
* :mod:`qcr.tableau`, :mod:`qcr.exact` -- exact Clifford ground truth: complete
  Cayley-graph enumeration for up to three qubits and provably optimal windows.
* :mod:`qcr.compute_graph`, :mod:`qcr.reducer`, :mod:`qcr.rf` -- a faithful port of
  the reference method of Rosenhahn, Osborne & Hirche (V1/V2/V3).

Most users only need the names re-exported here: :class:`Gate`, :class:`Device`,
:class:`Optimizer`, :class:`Config`, :class:`Result` and :class:`Database`.
"""

__version__ = "0.1.0"

from .gates import Gate, circuit_unitary
from .general import GRIDS, Config, Database, Optimizer, Result
from .hardware import Device, heron, ion, paper_device
from .verify import equivalent, from_qiskit, to_qiskit

__all__ = [
    "GRIDS",
    "Config",
    "Database",
    "Device",
    "Gate",
    "Optimizer",
    "Result",
    "__version__",
    "circuit_unitary",
    "equivalent",
    "from_qiskit",
    "heron",
    "ion",
    "paper_device",
    "to_qiskit",
]
