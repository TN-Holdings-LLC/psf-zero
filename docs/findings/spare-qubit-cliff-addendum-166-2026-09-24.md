# Addendum 166 -- Addendum 165 scored: the conversion fix is correct against PennyLane's own meaning (T1-T4 pass, worst 5.55e-16), but P2 failed -- two "physics survives" tests broke because the fix changed the synthesized tape's first-appearance wire order; fixed at the root by building circuits in the input tape's own wire order (2026-09-24 night)

**Pre-registered in**:
`spare-qubit-cliff-addendum-165-preregistration-2026-09-24.md`. Raw
output observed as text in the conversation.

## 0. In one line

34 passed, 2 failed. The four new tests (reference = `qml.matrix`) all
passed: T2 4.44e-16, T3 5.55e-16, T4 0.0. P2 ("all 32 earlier tests still
pass") failed on `test_physics_survives_gpu_mock_and_routing` (TVD 0.262)
and `test_physics_survives_the_full_chain` (TVD 0.940). Every test that
reads RESULTS (counts) still passed.

## 1. Scoring (Addendum 165)

| Prediction | Result |
|---|---|
| P1: T1-T4 pass, infidelity < 1e-9 | CONFIRMED |
| P2: the 32 earlier tests still pass | FAILED -- 30/32; two exact-statevector "physics" tests failed |
| P3: T1, T2 would fail on the unfixed file | not re-run, as registered |

## 2. Diagnosis

- `qiskit_to_tape` now emits `qml.QubitUnitary(M, wires=[b, a])` for a
  Qiskit block on qubits (a, b) -- correct, but it changes the order in
  which wires FIRST APPEAR in the synthesized tape (for the test tapes:
  1, 0, 3, 2 instead of 0, 1, 2, 3).
- `tape_to_qiskit` numbered wires by first appearance, so the circuit the
  transform routed had its qubits permuted relative to the input tape.
- Tests reading counts still passed: Addendum 161's explicit
  logical-to-circuit mapping restores the order at measurement.
- The two failing tests compare the routed circuit's statevector directly
  against the input tape's, assuming routed qubit i is input wire i -- true
  before only because first-appearance order happened to be natural. Same
  root cause as Addendum 161, surfacing in a different place.

## 3. The fix (root cause, not the tests)

- `tape_to_qiskit(tape, wire_order=None)`: an explicit `wire_order` pins
  Qiskit qubit i to `wire_order[i]`; a tape wire missing from it raises.
  Default behaviour (first appearance) is unchanged.
- `psf_pennylane_gpu_ibm_transform` builds the synthesized circuit with
  `wire_order=list(tape.wires)` -- the INPUT tape's own order -- so routed
  qubit i is always input wire i. Addendum 161's mapping is kept as a
  second guard; with this order it is the identity.
- No test was modified.

## 4. Predictions for the re-run (registered before running)

**R1.** All 36 tests pass, including both "physics survives" tests (TVD
< 1e-6, as those tests require).
**R2.** T1-T4 unchanged (they do not go through the IBM transform).

## 5. Re-run result (raw output observed as text)

**36 passed, 0 failed.** R1 and R2 confirmed.

- Both "physics survives" tests pass with no test modified.
- T2 4.44e-16, T3 5.55e-16, T4 0.0 -- unchanged, as R2 predicted.
- Real-GPU block check unchanged (worst 8.771e-15).
- Local testing mode: noiseless TVD 0.0183; FakeManilaV2 TVD 0.1000;
  widths {4}; triangle tape final positions [0, 2, 1], 12 routed two-qubit
  gates, TVD 0.0134.
- Side effect worth recording: the renumbering tape now routes to 9
  two-qubit gates (was 6). Built in the input tape's own wire order, its
  blocks on (0, 2) and (1, 3) are no longer renumbered into adjacent pairs,
  so FakeManilaV2's linear chain needs a SWAP -- that test now exercises
  renumbering and routing together (TVD 0.0175).

## 6. State at the end of this session

The prototype connection now: synthesizes blocks (stand-in or PSF-Zero),
verifies each on real `lightning.gpu` against a correctly ordered
reference, converts between PennyLane and Qiskit with the correct qubit
order in both directions (checked against `qml.matrix`), builds circuits in
the input tape's own wire order, routes, measures only logical qubits at
their final positions, and submits through the real `SamplerV2` -- 36 tests,
all observed passing as raw output.

## 7. Regression check: the 2026-09-28 rehearsal re-run after all fixes

`rehearse_xor_fake127.py` (Addendum 163) re-run after Addenda 165-166,
with two predictions stated before running: the XOR table identical to
Addendum 164's, and the P5 diagnostic flipping from False to True. Raw
output observed as text. Both held:

- All four inputs reproduce Addendum 164 to the printed precision
  (noisy <Z0>: -0.9040, 0.9185, 0.9085, -0.9130; noiseless: -0.9985, 0.9995,
  0.9980, -0.9955; 9 routed two-qubit gates, widths {4}, 4/4 correct both
  ways). The fixes did not change the path the 2026-09-28 submission uses.
- P5 diagnostic: `tape_to_qiskit preserves 2-wire QubitUnitary meaning?
  True` (was False in Addendum 164).

