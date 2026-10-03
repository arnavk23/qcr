Getting started
===============

Installation
------------

.. code-block:: bash

   pip install -e .                   # library
   pip install -e ".[experiments]"    # + FakeTorino calibration, Aer, TKET, plotting

Optimising a circuit
--------------------

Circuits are lists of :class:`qcr.Gate` over a device's native gates. Qiskit circuits convert with
:func:`qcr.from_qiskit` and :func:`qcr.to_qiskit`.

.. code-block:: python

   import numpy as np
   from qiskit import QuantumCircuit, transpile
   from qiskit.circuit.library import QFTGate

   import qcr

   qc = QuantumCircuit(3)
   qc.append(QFTGate(3), range(3))
   dev = qcr.ion(3)
   native = transpile(qc, basis_gates=["rx", "ry", "rz", "rxx"], optimization_level=1)
   gates = qcr.from_qiskit(native)

   dbs = {("phase", "tri"): qcr.Database(qcr.GRIDS["ion"], "tri", "phase", depth=3, verbose=False)}
   result = qcr.Optimizer(dev, dbs, qcr.Config(mode="exact", cost="calibrated")).run(gates)

   U, V = qcr.circuit_unitary(gates, 3), qcr.circuit_unitary(result.gates, 3)
   assert np.allclose(U, V, atol=1e-9)        # equal including global phase
   print(len(gates), "->", len(result.gates))  # 31 -> 23

Databases
---------

A :class:`qcr.Database` enumerates words over a discrete grid of native gates on a three-qubit local topology
(``"tri"``: all pairs coupled, ``"line"``: 0-1-2) and stores one 64-bit key per equivalence class. ``depth=3``
builds in seconds; the experiments use depths 6-14 (up to 3.2e7 classes) and cache them with
:meth:`qcr.Database.save` / :meth:`qcr.Database.load`. Pass one database per equivalence notion you want to use:
``"phase"`` always, plus ``"state"`` and ``"out"`` for ``state_aware=True``.

Devices
-------

* :func:`qcr.heron` -- IBM Heron r1 from Qiskit's ``FakeTorino`` (needs the ``hardware`` extra): ``rz``, ``sx``,
  ``x``, ``cz`` on heavy-hex, calibrated per-qubit and per-pair errors, dead couplers removed.
* :func:`qcr.ion` -- all-to-all trapped ion with arbitrary-angle ``rx``, ``ry``, ``rz`` and ``rxx``.
* :func:`qcr.paper_device` -- the discrete :math:`\pm\pi/2` alphabets of the original paper.
