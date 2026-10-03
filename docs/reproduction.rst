Reproduction and results
========================

The ``experiments/`` directory contains one script per table, figure or benchmark of the original paper and of the
extension; raw outputs are in ``results/`` and every number in the manuscripts is generated from them.

.. list-table::
   :header-rows: 1

   * - document
     - contents
   * - ``manuscripts/notes/notes.pdf``
     - the original paper explained from scratch
   * - ``manuscripts/reproduction/reproduction_report.pdf``
     - every table and figure re-run, with verdicts
   * - ``manuscripts/extension/extension_paper.pdf``
     - framework, theory, Clifford ground truth, hardware results and limitations
   * - ``manuscripts/talk/talk.pdf``
     - presentation

Summary of findings:

* Table I reproduces exactly; Tables III/IV and the journal statistics reproduce only with the larger (24-operator,
  depth-5) database; the V1 > V2 > V3 timing order needs a tuned random-forest threshold; Table V no longer beats a
  current Qiskit; the hardware effect disappears at Qiskit optimisation level 3.
* On its own 3-qubit benchmark the original method is 4.5x (ion trap) and 3.0x (IBM-style) longer than optimal.
* The general optimiser reaches the proven optimum on 35 of 40 such circuits without using Clifford structure.
* On 58 algorithm circuits compiled by Qiskit level 3, it never adds a two-qubit gate and removes 13% of all gates on
  IBM Heron; with the calibrated objective its noisy-simulation error is below Qiskit level 3 on 39 of 49 circuits.
