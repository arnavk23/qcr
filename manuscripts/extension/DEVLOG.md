# General local-rewriting optimiser: plan (agreed decisions)

## Decisions from the user (2026-10-02)
- **No Clifford assumption in the method.** Clifford tables are kept only as exact ground truth for evaluation.
- **Equivalence modes:** (a) exact *including* global phase (safe for controlled use); (b) state/measurement-aware
  (|0..0> input, computational-basis measurement at the end); (c) approximate, with a per-replacement error ε and a
  tracked total error bound. Up-to-phase is kept only for comparison with the paper.
- **Objective:** try calibrated fidelity (−log F per gate, per qubit/pair, from backend calibration) and a 2q-weighted
  count. Keep whichever actually improves hardware-relevant outcomes (estimated success probability and noisy
  simulation). If neither beats plain total count, fall back to total gate count.
- **Benchmarks:** real algorithms, the paper's random circuits, non-Clifford random circuits, standard suites
  (Nam/VOQC arithmetic, QASMBench) if downloadable.
- **Hardware:** native gate sets *with connectivity* (heavy-hex / line / all-to-all for ions).

## Unifying theory: equivalence = group action, database key = orbit canonical form
| mode | equivalence U ~ V iff | canonical key |
|---|---|---|
| paper | V = e^{iφ}U | phase-normalised matrix |
| exact | V = U | raw matrix |
| input-aware (window at circuit start) | V|0> = U|0> (up to phase if allowed) | first column |
| output-aware (window before final measurement) | V = D U, D diagonal | row-phase-normalised matrix |
| symmetry | conjugation by wire permutations that are automorphisms of the coupling graph | min over automorphism group |
| approximate | ‖U − V‖ ≤ ε | ε-net / exact resynthesis with fewer entanglers |

## Components
1. `qcr/hardware.py`: native gate sets (IBM Heron: rz(θ), sx, x, cz; ions: rx/ry/rz(θ), MS = rxx(θ) any angle),
   coupling maps, cost models (count, 2q-weighted, calibrated from FakeTorino).
2. `qcr/general.py`: big hashed databases (keys-only deepest level, automorphism-reduced, per topology {all-to-all
   triangle, line}), meet-in-the-middle, exact KAK/XX resynthesis (phase-exact), approximate synthesis, input- and
   output-aware windows, commutation-aware adaptive convex windows restricted to coupled pairs.
3. Baselines: Qiskit L3 (same coupling map), TKET FullPeepholeOptimise, BQSKit (small circuits), the paper's method
   (where its discrete alphabet applies).
4. Benchmarks + metrics: total, 2q count, depth, estimated success probability (calibrated), noisy-simulation
   fidelity/TVD for ≤ 10 qubits, verification (exact unitary ≤ 10 q, random states otherwise; state mode via
   output distribution).
5. Rewrite the extension paper around the general framework. Clifford results move to the "ground truth" section.

## Status
- [x] Reproduction, Clifford ground truth, CW-sem on paper benchmarks (running)
- [x] hardware.py  - [x] general.py v2  - [x] baselines (Qiskit L1/L3, TKET; BQSKit not installed)
- [x] benchmarks (run 2 `bench_v2.json`, paper benchmarks `general_paper_bench_*.json`)
- [x] objective selection (calibrated)  - [x] paper rewrite (2026-10-03, `extension_paper.tex`; v1 kept as
  superseded Clifford-only draft removed; numbers/tables/figs from `experiments/make_general_assets.py`)
- Note (2026-10-02): Heron noisy TVD now simulates only the active qubits with FakeTorino's calibrated errors
  remapped onto them (the full 133-qubit register exceeded Aer's width limit and crashed the first full run).
- Exactness fixes (2026-10-02), found on Nam pf1_10/pf2_10 (rotations ~1e-7 rad; outputs verified only to ~1e-6):
  `np.allclose` default rtol=1e-5 in `commutes_exact` let near-commuting gates be swapped (the actual cause);
  also made the phase test linear (operator distance, not fidelity), one tolerance `TOL = 1e-10`, phases snapped
  to the device-reachable set (pi/4 Heron, pi/2 ion) with exact gadget matching. Self-test threshold 1e-7 -> 1e-9.
- Run 1 (20:18; raw output removed, superseded by run 2) findings -> fixes, rerun as `bench_v2`:
  * FakeTorino has 11 dead couplers (cz error 1.0); the non-noise-aware layout put Qiskit L3 on them
    (success ~1e-22 on gf2^4_mult, vbe_adder_3, ising_n10, heisenberg_6/8, efficientsu2_6). `heron()` now drops them.
  * Strict global-phase repair cost 221 gates (40 Heron circuits) / 87 (29 ion); every ion regression vs L3 was
    the +1 MS gate of the gadget. Ion +-i phases are now absorbed into an existing MS gate (0-2 extra 1q gates);
    the optimiser never returns more than its input. Qiskit/TKET keep global phase as free metadata, so report
    both conventions (strict vs phase-as-metadata) in the write-up.
  * Verified: exact modes <= 1e-11; state-aware output TVD <= 6e-12; approx error 6.4e-4 within its bound.
  * Objective selection evidence (noisy TVD): calibrated ~= w2 > count on both devices (Heron mean TVD 0.105 /
    0.105 / 0.113; ion 0.0086 / 0.0087 / 0.0093) -> keep `calibrated` as default; confirm on v2.
- Paper benchmarks with the general engine (no Clifford knowledge; `general_paper_bench_*.json`, DB caps 9/10 ->
  2.9M / 1.8M entries). All 250 verified + legal. Mean totals (L3 / paper method / Clifford CW / optimum / general):
  T3 127.2 / 47.1 / - / 10.4 / 10.8 (optimum on 17/20); T4 104.0 / 36.8 / - / 12.2 / 12.5 (18/20);
  NJP_ion -- / 78.8* / 45.6 / - / 46.1; NJP_nisq -- / 76.0* / 49.3 / - / 51.4; T5 332.1 / 335.9 / 324.6 / - / 327.7.
  Paired general/paper ratio: 0.25, 0.42, 0.57, 0.70, 0.98. ~3-9 s per circuit. (*30-seed subset)
- Run 2 (`bench_v2.json`, 21:37; dead couplers removed, ion phase absorbed): all verified (exact <= 8.2e-11).
  vs Qiskit L3 (2q / all / Heron success): ours from L3 Heron x0.980 / x0.870 / x1.03, never worse (ion x1.00,
  never worse); ours from L1 cal Heron x1.004 / x0.838 / x1.21, ion x0.947 / x0.755; state-aware Heron x0.77 / x0.61,
  ion x0.62 / x0.50; TKET Heron x1.02 / x1.72. Noisy TVD Heron: L3 0.137, count 0.096, w2 0.089, cal 0.088
  (cal beats L3 on 39/49); ion: L3 0.0091, count 0.0092, w2 = cal 0.0084. -> calibrated is the default objective.
  Heron strict-phase gadgets ~200 1q gates (phase-as-metadata: all-gates x0.838 -> x0.825).
- Talk: `manuscripts/talk/talk.pdf` (beamer, Indico style), hardware slides use run 2.
- Circuits with classical feed-forward (QASMBench inverseqft_n4, ipea_n2, qec_sm_n5, shor_n5) are skipped.
- Caveat for the write-up: Qiskit L3 reports 0 two-qubit gates on pf1_10/pf2_10 because it drops the ~1e-7 rad
  rotations (approximate, not exact); compare those two circuits separately.
- Corrections found while writing the paper (2026-10-03):
  * "Never worse vs L3" holds for the calibrated objective, 2q count and ESP, not for total count: on ion the
    L3-stacked run raises the total on 28/58 circuits (rx/ry pulses 3824 -> 2422, virtual rz 2223 -> 3622).
  * The Heron ESP gain from L1 (x1.21 vs L3) is x1.11 from Qiskit L1's layout already beating L3's, times x1.09
    from our rewriting; the optimiser never changes the layout.
  * Heron L3-stacked savings are split between sx/x (7218 -> 6139) and rz (5297 -> 4436), not "mostly sx".
  * Nam pf2_10 on Heron spans > 16 physical qubits, so its 5 L1-based outputs were unverified in run 2;
    re-verified with `scripts/verify_large.py` (`results/general/verify_large_*.json`).
  * Suite is 2-12 qubits, <= 1008 gates. Noisy TVD uses exact density-matrix probabilities (no shot noise),
    gate errors only (no readout/idle/crosstalk); ion sim charges rz 3e-5 while the cost model treats it as free.
