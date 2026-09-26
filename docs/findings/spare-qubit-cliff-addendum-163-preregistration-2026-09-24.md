# Addendum 163 -- Pre-registration: 2026-09-28 rehearsal -- the trained XOR circuit (Addendum 148, seed 0) through the verified routing / logical-measurement / SamplerV2 path, on a 127-qubit fake heavy-hex device (2026-09-24 night)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

Addenda 155-162 verified the submission path (real `SamplerV2` in local
testing mode; measurement of logical qubits only, at final routed
positions, in the original wire order) on a 5-qubit fake device. The
actual 2026-09-28 target will be a ~127-qubit heavy-hex device, and the
circuits to submit are the trained XOR classifier's (Addendum 148, seed 0).
Addendum 151 prepared those circuits for a generic heavy-hex lattice before
Addenda 160-162 existed. This rehearses the real submission as closely as
possible without credentials.

## 2. Design

- Retrain Addendum 148's seed 0 (ideal condition; deterministic --
  Addendum 151 reproduced its exact loss 0.000005).
- For each XOR input (00, 01, 10, 11): build the trained circuit DIRECTLY
  in Qiskit, gate by gate (as Addendum 151 did), NOT via the prototype's
  `tape_to_qiskit` -- see Section 3, P5.
- Route with `route_for_backend` (pinned initial layout, optimization
  level 1) to `FakeBrisbane` (127-qubit Eagle heavy-hex snapshot); check
  `is_isa_compliant`; measure with `logical_measurement` (4 classical bits);
  submit via `make_sampler_submit_fn` to FakeBrisbane (noisy) and to a
  noiseless `AerSimulator`, 4000 shots, simulator seed 42.
- Estimate <Z0> from classical bit 0 (Qiskit convention: rightmost), and
  predict the XOR label from its sign, as the trained model does.

## 3. Pre-registered predictions

**P1 (the Qiskit circuit is the trained model).** For every input, the
Qiskit circuit's exact <Z0> equals the retrained PennyLane model's own
<Z0> to within 1e-9.

**P2 (PSF-Zero has nothing to do here).** `collect_and_consolidate` at
this project's default floor (runs of more than 12 gates on one pair)
finds 0 blocks in every XOR circuit -- so neither PSF-Zero synthesis nor
the real-GPU block check is exercised by these circuits. Recorded so
nothing in this rehearsal is later read as a PSF-Zero result.

**P3 (noiseless).** Correct XOR label on 4/4 inputs, |<Z0>| > 0.9.

**P4 (FakeBrisbane noise).** Correct XOR label on 4/4 inputs, |<Z0>| > 0.5
(Addendum 150 found the sign robust far above this project's standard
noise level; the magnitude is expected to shrink).

**P5 (diagnostic: a suspected latent bug in the prototype, stated before
checking).** The prototype's `tape_to_qiskit` converts a 2-wire
`qml.QubitUnitary(M, wires=[a, b])` to `qc.unitary(M, [a, b])`. PennyLane
reads M with wire a as the most significant index; Qiskit reads it with
qubit a as the least significant -- the same mismatch Addendum 154 found in
the GPU check's reference. Prediction: for M = CNOT, the Qiskit operator
from `tape_to_qiskit` does NOT equal the operator PennyLane itself assigns
to the tape (`qml.matrix(tape, wire_order=[1, 0])`, i.e. PennyLane's own
matrix expressed in Qiskit's bit order). **If they are equal, this
suspicion is wrong and is recorded as such.** If they differ, every
earlier connection test compared Qiskit-side results against Qiskit-side
references built by the same conversion, and could not have detected it.

## 4. What this cannot establish

- Real hardware, queueing, authentication (2026-09-28).
- Whether FakeBrisbane's snapshot matches the device actually chosen, or
  that device's current calibration.
- Anything about PSF-Zero (see P2), or timing.
