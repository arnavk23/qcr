qcr: quantum circuit reduction, made exact
==========================================

**qcr** shortens quantum circuits by local rewriting: it repeatedly replaces a small sub-circuit with a cheaper
equivalent one from a precomputed database. It provides

* a general, hardware-aware optimiser with exact, state-aware and approximate semantics and calibrated costs;
* exact Clifford ground truth (all 92,897,280 three-qubit Clifford elements) to measure optimality gaps;
* a faithful port of the reference method of Rosenhahn, Osborne & Hirche (`arXiv:2502.14715
  <https://arxiv.org/abs/2502.14715>`_).

.. toctree::
   :maxdepth: 2

   getting_started
   explanation
   reproduction
   apidocs/index
   release_notes
