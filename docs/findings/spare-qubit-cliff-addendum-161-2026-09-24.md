# Addendum 161 -- Addendum 160 scored: the width fix works (P1, P3, P4 hold), but the new SWAP test failed with NO swaps inserted -- it caught a different, pre-existing bug: classical bits came back in the synthesized tape's renumbered wire order, not the original tape's (2026-09-24 night)

**Pre-registered in**:
`spare-qubit-cliff-addendum-160-preregistration-2026-09-24.md`. Raw
output received as an uploaded text file and read from disk.

## 0. In one line

30 passed, 1 failed. Bitstrings are now 4 bits wide (P1), noisy TVD fell
from 0.1297 to 0.1198 (P3), and every earlier suite still passes (P4). P2
failed: the "SWAP" tape gave noiseless TVD 0.1598 -- but with final
positions [0, 1, 2, 3] and 6 two-qubit gates, i.e. no SWAP at all. The
cause is not routing: `tape_to_qiskit` numbers wires by first appearance,
consolidation absorbed the single-qubit ops into the blocks, so the
synthesized tape's wires first appear as 0, 2, 1, 3 -- and the transform
discarded that mapping, measuring in the renumbered order. Fixed; a
genuinely SWAP-forcing test was added.

## 1. Scoring (Addendum 160)

| Prediction | Result |
|---|---|
| P1: bitstrings 4 bits wide | CONFIRMED -- widths {4} |
| P2: noiseless TVD < 0.1 on both tapes | FAILED on the second tape -- 0.1598 (original tape: 0.0178) |
| P3: noisy TVD in (0.01, 0.3) and below 0.1297 | CONFIRMED -- 0.1198 |
| P4: all earlier suites still pass | CONFIRMED -- 10 + 7 + 3 + 3 prior tests, plus 5 unchanged tests of the updated suite |

## 2. Diagnosis

- The failing tape: single-qubit RX on wires 0-3, then 15 random 2-qubit
  unitaries on (0,2), then 15 on (1,3). Intended to force SWAPs on
  FakeManilaV2's linear chain.
- `collect_and_consolidate` absorbed each wire's RX into its block. In the
  synthesized tape the first ops are the (0,2) block then the (1,3) block,
  so `tape_to_qiskit` numbered wires 0->0, 2->1, 1->2, 3->3. The blocks
  became adjacent pairs (0,1), (2,3) -- hence no SWAPs -- and the transform
  discarded the returned wire order (`qc_synth, _wire_order = ...`).
- `logical_measurement` then measured circuit qubit i into classical bit
  i, so bits 1 and 2 held wires 2 and 1: a correct distribution with two
  bits swapped, against a reference in the original order.
- Earlier tapes never exposed this: their wires first appeared in natural
  order (0, 1, then 2, 3).

Not independently re-derived here by running code (this environment has
no Qiskit); the diagnosis rests on the logged final positions and gate
count (no SWAP) together with the numbering rule in `tape_to_qiskit`'s own
code. The re-run below tests it.

## 3. The fix

`psf_pennylane_gpu_ibm_transform` now keeps the synthesized tape's wire
order and builds `logical_to_circuit[i]` = the circuit qubit holding the
ORIGINAL tape's wire i; `logical_measurement(qc_routed,
logical_to_circuit)` measures that qubit's final routed position into
classical bit i. A wire missing from the synthesized tape, or a mapping
that is not a permutation, raises rather than being guessed.

Tests (`test_real_submit_local_mode.py`, now 9):
- the failing tape is kept as `test_wire_renumbering_keeps_logical_bit_order`;
- `test_swap_routed_circuit_maps_logical_qubits_correctly` now uses a
  triangle of blocks, (0,1), (1,2), (0,2), which cannot be embedded in a
  linear chain, and asserts that more than 9 two-qubit gates were routed
  (i.e. a SWAP really was inserted) before checking the distribution.

## 4. Predictions for the re-run (registered before running)

**R1.** Renumbering tape: noiseless TVD < 0.1, widths {4}.
**R2.** Triangle tape: more than 9 routed two-qubit gates, widths {3},
noiseless TVD < 0.1.
**R3.** All other tests (30 in this run) still pass.

## 5. What this does not establish

Real hardware; timing; devices other than FakeManilaV2's snapshot.
