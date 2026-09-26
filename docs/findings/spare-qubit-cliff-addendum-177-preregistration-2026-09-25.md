# Addendum 177 -- Pre-registration: as circuits get deeper, does PSF-Zero's synthesis beat Qiskit's default compilation (optimization level 3) under realistic device noise? A test of "PSF-Zero has a strength beyond the cliff" (2026-09-25)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

The project owner's hypothesis: even where the layout cliff does not
occur, current compilers accumulate error and do not improve, so
PSF-Zero's exact closed-form synthesis should give better results on
realistic noise, and the advantage should appear as circuits get deeper.
If true, this would be a strength of PSF-Zero other than the cliff, and a
candidate centre for a third paper.

The evidence so far points the other way, and is stated here so the
prediction below is not mistaken for a neutral guess: on small circuits,
PSF-Zero's output was structurally identical to Qiskit's decomposer with
`euler_basis="ZSX"`, with no noisy-score difference, and slower to
synthesize (Addenda 157, 159); against Qiskit's default compilation it was
smaller in both size and depth in only 40 of 105 cells, with a mean noisy
TVD difference of +0.00003 (Addendum 172, exploratory). None of those
tests varied depth systematically, which is what this one does.

## 2. Design

- **Circuits**: 4 logical qubits; L layers of Haar-random 2-qubit blocks
  (`UnitaryGate`) in a brickwork pattern -- even layers on (0,1) and
  (2,3), odd layers on (1,2). L in {1, 2, 4, 8, 16} (2 to 24 blocks). 10
  random circuits per L (fixed seeds).
- **Backends**: FakeFez, FakeMarrakesh, FakeKingston (Heron, CZ) and
  FakeBrisbane (Eagle, ECR) from `qiskit_ibm_runtime.fake_provider`.
- **Arms** (all measure every logical qubit, `measure_all()` on the logical
  circuit BEFORE compilation, so layout selection can see readout error --
  the Addendum 170 lesson):
  - **Q3 -- Qiskit default**: `transpile(..., optimization_level=3,
    seed_transpiler=0)` on the whole circuit. Qiskit chooses the layout and
    synthesizes every block itself.
  - **P -- PSF-Zero**: each block synthesized by
    `SU4GeodesicPSFSynthesizer(GeodesicPSFHyper(entangling_basis="cx",
    on_unsupported="raise"))`, the gates composed into the circuit, then
    `transpile(..., optimization_level=1, initial_layout=<Q3's own layout
    for that circuit>)`.
  - **Z -- control**: identical to P, but each block synthesized by
    `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")`.
- **Why P and Z use level 1 and Q3's layout**: at level 3 Qiskit
  re-consolidates and re-synthesizes every 2-qubit block, which would erase
  PSF-Zero's output (the Addendum 156 Section 1a problem). Pinning P and Z
  to the layout Q3 chose isolates the synthesis method from layout choice.
- **Known asymmetry, stated in advance**: PSF-Zero can emit only
  `canonical` or `cx` gates, not CZ or ECR. P and Z therefore rely on
  Qiskit's level-1 translation from CX to each device's native gate, while
  Q3 synthesizes directly in the native gate. This is how PSF-Zero would
  actually be used today; it may cost P and Z extra single-qubit gates.
- **Execution**: `SamplerV2` in local testing mode (`psf_ibm_real_submit`),
  4000 shots, simulator seed 42, to each fake backend (noisy) and to a plain
  `AerSimulator` (noiseless).
- **Recorded per (backend, L, circuit, arm)**: routed two-qubit gate count,
  depth, size, the physical layout used, noisy and noiseless TVD against
  the logical circuit's exact distribution, PSF-Zero fallback count.

## 3. Pre-registered decision rule for the hypothesis

For each (backend, L): d = mean over the 10 circuits of (TVD_Q3 - TVD_P),
paired by circuit, and SE = sample standard deviation of the paired
differences / sqrt(10).

- **P wins at (backend, L)** if d >= 0.02 and d >= 3 SE.
- **Q3 wins at (backend, L)** if -d >= 0.02 and -d >= 3 SE.

**H (the owner's hypothesis) is CONFIRMED** if P wins on at least 3 of the
4 backends at both L = 8 and L = 16.
**H is REFUTED** if P wins on at most 1 backend at L = 16.
Anything in between is **INCONCLUSIVE** and reported as such.

**The assistant's expectation, stated before running: H refuted** -- no
consistent difference at any depth, on the evidence in Section 1.

## 4. Secondary predictions

- **S1.** P and Z have identical routed two-qubit count, depth and size on
  every circuit (as in Addendum 159), and P has zero fallbacks.
- **S2.** Q3 routes to the same two-qubit gate count as P on every circuit
  (3 per generic block, no SWAPs needed on a 4-qubit path).
- **S3.** For every arm and backend, mean noisy TVD is non-decreasing from
  L = 1 to L = 16 (noise accumulates with depth).
- **S4.** Noiseless TVD < 0.06 for every circuit and arm (sampling noise
  only; confirms the three compilations are correct).

If S1, S2 or S4 fails, the comparison itself is compromised and that is
reported before any reading of H.

## 5. What this cannot establish

- Real hardware; fake backends are past snapshots.
- Larger circuits, other structures, or approximate synthesis
  (`approximation_degree` left at Qiskit's default).
- Timing (not recorded as a result).

## 6. Script lock

`psf_vs_qiskit_depth_sweep.py`, normalized SHA-256 (trailing whitespace
stripped per line, surrounding blank lines removed):
`cbcb93fa60b920d5651e5fcf689aeacf0b4e91dbd3b5b4f2b8a60808e4bb009c`.
Re-check it on the machine that runs the experiment BEFORE running, and
record the check in the results. Output:
`psf_vs_qiskit_depth_sweep_2026-09-25.csv` (600 rows: 4 backends x 5 depths
x 10 circuits x 3 arms).
