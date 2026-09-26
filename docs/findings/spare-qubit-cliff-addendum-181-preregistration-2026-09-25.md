# Addendum 181 -- Pre-registration: a compound ("interest-on-interest") test of the whole PennyLane -> synthesis -> IBM-topology pipeline, 20,000 laps, with and without PSF-Zero -- does any small error accumulate once the known bugs are fixed? (2026-09-25)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

The PennyLane <-> Qiskit <-> IBM connection had four bugs found and fixed
on 2026-09-24 (Addenda 154, 160-161, 164-166); 36 tests pass. Passing tests
do not prove that no bug remains. A small, biased error -- 1e-15 per pass,
say -- is invisible in one pass but grows if the output of each pass is fed
back as the next pass's input. The workplace round-trip chain (Addenda
175-176) did this for the PennyLane <-> Qiskit conversion alone, 100 times,
and found nothing compounding. This test runs the WHOLE pipeline around the
loop, 20,000 times, for four compilation methods, including PSF-Zero and
Qiskit's own default.

## 2. Design

**One lap** (lap k turns tape_{k-1} into tape_k):
1. `tape_to_qiskit(tape, wire_order=[0,1,2,3])` (the fixed converter;
   the control arm C uses the pre-fix conversion, Section 2 below).
2. Compile, per arm:
   - **A -- without PSF-Zero**: `collect_and_consolidate(block_gate_floor=0)`,
     each 2-qubit block synthesized by `TwoQubitBasisDecomposer(CXGate(),
     euler_basis="ZSX")`, gates composed in, then `transpile(...,
     optimization_level=1, initial_layout=L, seed_transpiler=0)`.
   - **P -- with PSF-Zero**: as A, synthesizer `SU4GeodesicPSFSynthesizer(
     GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"),
     verify=True)`.
   - **Q3 -- Qiskit default**: `transpile(qc, optimization_level=3,
     initial_layout=L, seed_transpiler=0)` on the unsynthesized circuit.
   - **C -- control, deliberately broken**: as A, but step 1 uses the
     pre-fix conversion `qc.unitary(mat, qubits)` (no qubit-order
     reversal), while step 4 uses the fixed one. This reintroduces the
     Addendum 164 bug on one side of the loop.
3. Map the routed circuit back to logical qubits using its own layout.
   The fixed layout L is a 4-qubit path on the device, so no SWAP is
   needed; a lap that needs one, or touches a qubit outside L, raises.
4. Every two-qubit gate is wrapped as a `unitary` and the circuit is
   converted back with `qiskit_to_tape(..., [0,1,2,3])` -> tape_k.
5. **Meaning check** against PennyLane's own matrix:
   U_k = `qml.matrix(tape_k, wire_order=[0,1,2,3])`.

**Device**: FakeNighthawk (square lattice; Addendum 180). L = the first
4-qubit simple path found by a deterministic depth-first search from qubit 0.

**Circuit**: tape_0 = three Haar-random 2-qubit `QubitUnitary` blocks on
wires (0,1), (2,3), (1,2) (fixed seed).

**Recorded**: delta_k = phase-aligned Frobenius distance ||U_k - e^{i phi}
U_0||_F at k = 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
20000; the per-lap distance e_k = d(U_{k-1}, U_k) (median and max over all
laps); the number of operations in tape_k at each checkpoint; the growth
exponent alpha (least-squares slope of log delta_k on log k over
checkpoints with k >= 10 and delta_k > 0); wall time per arm; fallbacks.
Each arm runs as its own process.

Reference scales for 20,000 laps at ~1e-15 per lap: a random walk grows as
sqrt(k) to ~1e-13 (alpha ~ 0.5); a systematic bias grows as k to ~2e-11
(alpha ~ 1); a plateau stays flat (alpha ~ 0).

## 3. Pre-registered predictions

**R1 (no compounding -- the main prediction).** For A, P and Q3: alpha <
0.75 and delta_20000 < 1e-10. **An arm with alpha >= 0.75 has an error that
accumulates systematically -- a hidden bug or bias -- and that is reported
as the finding.**

**R2 (the test can see a real bug).** For C: delta_1 > 0.1. If C does not
fail at lap 1, the meaning check is not sensitive enough and R1 means
nothing.

**R3 (no growth in circuit size).** For A, P and Q3: the operation count of
tape_20000 is at most 1.5 times that of tape_1.

**R4 (the pipeline stays healthy).** For A, P and Q3: every lap completes
with no exception, no SWAP, and (for P) no fallback.

**Descriptive, no prediction**: A versus P versus Q3 -- delta trajectories,
alpha, per-lap error, fixed-point behaviour.

## 4. What this cannot establish

- Anything about larger circuits, where the per-lap check cannot be exact.
- Real hardware or noise (this is arithmetic only).
- That no bug remains -- only that none accumulates in this loop.

## 5. Script lock

`compound_pipeline_chain.py`, normalized SHA-256:
`3b73c64139cb034aaa6a78a6b756d61f19e79e5bbe13fe1996421fabc38ea152`.
Re-check on the machine BEFORE running and record the check. Requires the
fixed prototype (`psf_pennylane_gpu_prototype.py` with
`tape_to_qiskit(..., wire_order=...)`, Addendum 166). Outputs, one per arm:
`compound_chain_{A,P,Q3,C}_checkpoints_2026-09-25.csv`, written at every
checkpoint.
