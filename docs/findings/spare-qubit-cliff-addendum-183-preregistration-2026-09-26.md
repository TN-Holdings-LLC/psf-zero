# Addendum 183 -- Pre-registration: the timed exam, repeated with compounding -- on FakeNighthawk's cliff, does every lap of a PennyLane -> compile -> PennyLane loop hit the cliff again, and does each compiler keep meeting (or missing) a 1-second deadline while the circuit's meaning is preserved? Pilot: 10 laps (2026-09-26)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

Two results from 2026-09-25 are combined here.
- **Timed (Addendum 180)**: on FakeNighthawk at spare = 0, Qiskit's default
  compilation took ~12.9 s and missed a 1 s deadline 0/5; PSF-Zero took
  ~0.15 s and met it 5/5, with identical output.
- **Compound (Addendum 182)**: 20,000 laps of the fixed PennyLane ->
  synthesis -> IBM-topology loop on 4 qubits never changed the circuit's
  meaning; ordinary floating-point drift accumulated linearly.

A training loop recompiles the same circuit shape again and again. This
test asks what that looks like on the cliff: each lap's compiled output
becomes the next lap's input, every lap is timed against the deadlines,
and every lap's meaning is checked. Unlike Addendum 180, a lap here starts
from a PennyLane tape and returns to one.

**Prior knowledge, disclosed**: the first lap is essentially Addendum
180's measurement (same device and circuit family), so predictions about
lap 1 are not blind. The new questions are whether the cliff recurs on
every later lap, where the input is the previous lap's compiled output
rather than the original circuit, and whether meaning is preserved
throughout.

## 2. Design

- **Device**: FakeNighthawk (120 qubits).
- **tape_0**: a PennyLane tape on n = 120 - spare wires: for each pair
  (0,1), (2,3), ..., 20 Haar-random `qml.QubitUnitary` 2-qubit gates (the
  cliff circuit family of Addenda 178-180, written in PennyLane).
- **One lap**:
  1. `tape_to_qiskit(tape, wire_order=range(n))` (the fixed converter).
  2. Compile, timed around this call only:
     - **Q3**: `transpile(qc, FakeNighthawk(), optimization_level=3,
       seed_transpiler=0)`;
     - **P**: `compile_for_hardware(qc, coupling_map, native basis,
       entangling_basis="cx", layout_search=True, on_unsupported="raise",
       seed_transpiler=0)`.
  3. Map back to logical qubits with the compiled circuit's own layout. A
     lap whose routing permuted qubits (a SWAP) or placed a two-qubit gate
     outside the layout stops that run and is reported.
  4. Wrap every two-qubit gate as a `unitary`; `qiskit_to_tape` -> tape_k.
  5. **Meaning check, per pair** (exact at this size because pairs are
     independent): for each pair, PennyLane's own matrix of that pair's
     operations in tape_k versus in tape_0, phase-aligned Frobenius
     distance; the maximum over pairs is recorded.
- **Conditions**: spare in {0, 8}; arms Q3 and P; **10 laps** each. The
  four runs execute one after another, never in parallel (timing).
- **Recorded per lap**: compile time; whole-lap time; whether compile time
  is within 0.01 / 0.1 / 1 / 10 s; maximum per-pair distance from tape_0;
  operation count; routed two-qubit count; whether the physical layout
  changed from the previous lap.

## 3. Pre-registered predictions

**D1 (the cliff recurs every lap).** Q3 at spare = 0: compile time > 1 s
on 10 of 10 laps, median >= 5 s. **If the cliff disappears after lap 1**
(the recompiled circuit no longer triggers it), that is the finding: a
training loop would pay it only once.

**D2 (PSF-Zero keeps meeting the deadline).** P at spare = 0: compile time
<= 1 s on at least 9 of 10 laps.

**D3 (no difference away from the cliff).** At spare = 8, both arms <= 1 s
on at least 9 of 10 laps.

**D4 (meaning preserved).** For every arm and spare, every lap completes
(no SWAP, no failure) and the maximum per-pair distance after lap 10 is
below 1e-10.

**Descriptive**: cumulative compile time over 10 laps per arm and spare;
whether Q3's compile time on later laps differs from lap 1; layout
stability across laps.

## 4. What this cannot establish

- Behaviour beyond 10 laps (a longer run is a separate decision after
  this pilot).
- Real hardware; FakeNighthawk's error values are not representative.
- Typical training circuits (this is the saturated cliff family).

## 5. Script lock

`deadline_compound_chain.py`, normalized SHA-256:
`8ffabd4e1b34e00a30c54dc181d48cd791a2362d49357612fbfca6833d359ccf`.
Re-check on the machine BEFORE running and keep the check in the saved
log. Output: `deadline_compound_chain_2026-09-26.csv` (up to 40 rows for
10 laps), rewritten after every lap.
