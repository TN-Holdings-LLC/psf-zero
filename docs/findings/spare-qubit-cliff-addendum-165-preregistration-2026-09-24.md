# Addendum 165 -- Pre-registration: fix the qubit-order bug in the prototype's tape <-> Qiskit conversion (Addendum 164, P5), checked against PennyLane's OWN meaning (qml.matrix), not against another conversion (2026-09-24 night)

**Status: pre-registration only. No measurement has been run.**

## 1. The fix

In `psf_pennylane_gpu_prototype.py`:
- `tape_to_qiskit`: `qc.unitary(mat, qubits)` -> `qc.unitary(mat, qubits[::-1])`.
- `qiskit_to_tape` (`unitary` branch): `qml.QubitUnitary(mat, wires=qubits)`
  -> `qml.QubitUnitary(mat, wires=qubits[::-1])`.

PennyLane reads a matrix with the first listed wire as the most
significant index; Qiskit reads it with the first listed qubit as the
least significant. Reversing the list in both directions makes each side
mean what the other meant. The single-qubit fallback is unaffected.

## 2. New tests (`test_tape_conversion_fidelity.py`)

Every reference is PennyLane's own `qml.matrix(tape, wire_order=...)`,
never another `tape_to_qiskit` output -- the weakness Addendum 164 found in
every earlier test.

- T1: CNOT as a 2-wire `QubitUnitary` -> Qiskit operator equals
  PennyLane's matrix in Qiskit bit order (the exact Addendum 164 P5 case).
- T2: 4 wires, random 2-qubit unitaries on pairs including non-adjacent
  and reversed ones ([2, 0], [3, 1]), with named single-qubit gates -- same
  check.
- T3: round trip `qiskit_to_tape(tape_to_qiskit(tape))` has the same
  `qml.matrix` as the tape.
- T4: `psf_pennylane_gpu_transform` (CPU stand-in synthesizer, no GPU
  needed) returns a tape whose `qml.matrix` equals the input tape's, up to
  global phase.

## 3. Pre-registered predictions

**P1.** T1-T4 all pass, with infidelities below 1e-9.
**P2.** All 32 earlier tests still pass. Rationale: they compare
conversions against conversions; flipping both directions consistently
preserves their internal agreement.
**P3.** Stated in advance: T1 and T2 would FAIL on the unfixed file (T1 is
exactly Addendum 164's P5 case, observed as False there). Not re-run here.

## 4. What this cannot establish

Real hardware; timing; conversion of gates outside the prototype's small
op set.
