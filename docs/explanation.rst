How it works
============

Equivalence as a group action
-----------------------------

A replacement :math:`U \to V` of a window is sound if the program's output does not change. Each useful notion
of "equivalent" is the orbit relation of a group acting on unitaries, and the database key is a canonical
representative of the orbit, so one hash lookup finds the whole class.

.. list-table::
   :header-rows: 1

   * - notion
     - :math:`U \sim V` iff
     - canonical key
   * - exact
     - :math:`V = U`
     - :math:`U`
   * - up to phase
     - :math:`V = e^{i\varphi} U`
     - phase-normalised :math:`U`
   * - input-aware
     - :math:`V|0\rangle = e^{i\varphi} U|0\rangle`
     - phase-normalised :math:`U|0\rangle`
   * - output-aware
     - :math:`V = DU`, :math:`D` diagonal
     - rows phase-normalised
   * - symmetry
     - :math:`V = PUP^\dagger`, :math:`P` a coupling-graph automorphism
     - minimum over the orbit

Input-aware rewriting is sound for windows with nothing before them on their wires, output-aware rewriting for
windows with nothing after them, when the circuit is a complete program (:math:`|0\cdots0\rangle` in,
computational-basis measurement out). Approximate rewriting with per-replacement operator-norm errors
:math:`\varepsilon_i` changes the whole circuit by at most :math:`\sum_i \varepsilon_i`, which the optimiser reports
as :attr:`qcr.Result.error_bound`.

The optimiser loop
------------------

1. Grow a convex window of up to 16 gates on at most three coupled wires, jumping over gates that commute exactly.
2. Look it up in the databases; if absent, try meet-in-the-middle :math:`U = RA` with :math:`A` from a radius-2 ball.
3. On at most two wires, also resynthesise analytically (KAK with the minimal number of CZ; Weyl decomposition
   into arbitrary-angle MS gates; Euler angles).
4. Apply the cheapest legal replacement; repeat with random exact-commutation shuffles until no progress.
5. In exact mode, repair the accumulated global phase with the cheapest gadget (absorbed into an MS gate on ions);
   never return a circuit more expensive than the input.

When is local rewriting optimal?
--------------------------------

For a finite gate group of diameter :math:`D`, windows longer than :math:`D` with a database of radius
:math:`D` make every locally reduced circuit globally optimal. The paper's discrete alphabets generate the
Clifford group, so this is checkable: for three qubits :math:`D = 17` (ion trap) and :math:`19` (IBM-style), and in
practice windows of 12-16 gates already reach the proven optimum. The extension paper in
``manuscripts/extension`` proves this and further results (gate-count parity, mixing of random-circuit benchmarks,
counting lower bounds).
