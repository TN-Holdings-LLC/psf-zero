# Addendum 164 -- Rehearsal on a 127-qubit fake device: the trained XOR classifier keeps 4/4 correct under FakeBrisbane noise (|<Z0>| 0.90-0.92); and the suspected qubit-order bug in the prototype's tape_to_qiskit is confirmed (2026-09-24 night)

**Pre-registered in**:
`spare-qubit-cliff-addendum-163-preregistration-2026-09-24.md`. Raw
output observed as text in the conversation.

## 0. In one line

All five predictions hold. Built gate by gate in Qiskit, the retrained
XOR circuits reproduce the PennyLane model to 3.3e-16; routed to
FakeBrisbane (127 qubits), measured on the 4 logical qubits only and
submitted through the real SamplerV2 in local testing mode, they give the
correct XOR label on all 4 inputs both noiselessly (|<Z0>| 0.9955-0.9995)
and under the device snapshot's noise (0.9040-0.9185). No PSF-Zero code
runs in these circuits (0 consolidatable blocks). The diagnostic confirms
that the prototype's `tape_to_qiskit` gives a 2-wire `QubitUnitary` the
opposite qubit order from PennyLane's own meaning.

## 1. Results

Retrained seed 0: final loss 0.000005 (matches Addenda 148, 151).

| input | label | PennyLane <Z0> | Qiskit <Z0> | diff | blocks | routed 2q | bits | noiseless <Z0> | noisy <Z0> | correct (noiseless / noisy) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 00 | -1 | -0.99776 | -0.99776 | 1.1e-16 | 0 | 9 | {4} | -0.9985 | -0.9040 | yes / yes |
| 01 | +1 | 0.99776 | 0.99776 | 3.3e-16 | 0 | 9 | {4} | 0.9995 | 0.9185 | yes / yes |
| 10 | +1 | 0.99776 | 0.99776 | 0.0 | 0 | 9 | {4} | 0.9980 | 0.9085 | yes / yes |
| 11 | -1 | -0.99776 | -0.99776 | 1.1e-16 | 0 | 9 | {4} | -0.9955 | -0.9130 | yes / yes |

FakeBrisbane, pinned layout on physical qubits 0-3 (a straight run of the
heavy-hex lattice, so no SWAPs: 9 two-qubit gates = 3 layers x 3), 4000
shots, simulator seed 42. The 127-qubit noisy local simulation completed;
only the 4 measured qubits carried any gates.

## 2. Scoring

- **P1 -- CONFIRMED.** Worst difference 3.3e-16.
- **P2 -- CONFIRMED.** 0 blocks in every circuit. Nothing here is a
  PSF-Zero result.
- **P3 -- CONFIRMED.** 4/4 correct, |<Z0>| > 0.99.
- **P4 -- CONFIRMED.** 4/4 correct, |<Z0>| > 0.90 -- well above the 0.5 bar,
  consistent with Addendum 150's finding that the sign is robust.
- **P5 -- CONFIRMED (the suspicion was right).** `tape_to_qiskit` turns
  `qml.QubitUnitary(CNOT, wires=[0, 1])` into a Qiskit operator that does
  NOT equal PennyLane's own matrix for that tape in Qiskit's bit order.

## 3. What P5 means

- `tape_to_qiskit` passes a PennyLane matrix (first listed wire = most
  significant index) to `qc.unitary(M, [a, b])`, which Qiskit reads with
  qubit a as the LEAST significant index. The result is the qubit-reversed
  operation -- the same class of mismatch as Addendum 154's GPU-reference
  bug.
- Every earlier connection test built both the pipeline's circuit and its
  correctness reference with this same conversion (Qiskit side against
  Qiskit side), so none of them could detect it. Their passes remain valid
  statements about the pipeline's internal consistency, not about
  faithfulness to a tape's PennyLane meaning.
- **Unaffected**: this rehearsal's XOR circuits (built directly in Qiskit
  with named gates) and therefore the 2026-09-28 submission plan as
  prepared here.
- **Affected**: any use of the prototype connection on a PennyLane tape
  containing a multi-qubit `QubitUnitary`. Likely fix (not yet made or
  tested): reverse the qubit list in both directions --
  `qc.unitary(M, qubits[::-1])` in `tape_to_qiskit` and
  `qml.QubitUnitary(M, wires=qubits[::-1])` in `qiskit_to_tape` -- then add
  a test comparing against `qml.matrix` (PennyLane's own meaning), not
  against another `tape_to_qiskit` output.
- The prototype's own comment attributes a past "0.94 round-trip
  infidelity" to named multi-qubit gates having different conventions in
  the two libraries. Named gates such as CNOT take (control, target) in
  both; this matrix-order mismatch may be what was actually observed then.
  Not verified.

## 4. What this does not establish

Real hardware, queueing, authentication; the chosen device's current
calibration; anything about PSF-Zero; timing.

## 5. Files

| File | What it is |
|---|---|
| `rehearse_xor_fake127.py` | this run's script |
