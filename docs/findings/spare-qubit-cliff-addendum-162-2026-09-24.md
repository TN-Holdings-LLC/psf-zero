# Addendum 162 -- Re-run after the wire-order fix: 32/32 pass; a triangle of blocks really did force a SWAP (final positions [0, 2, 1]) and results still came back in the original wire order (2026-09-24 night)

**Predictions registered in**: Addendum 161, Section 4 (R1-R3), before
this run. Raw output observed as text in the conversation.

## 0. In one line

All three re-run predictions hold. The renumbering tape that failed in
Addendum 161 now gives noiseless TVD 0.0176 (was 0.1598). The new triangle
tape routed to 12 two-qubit gates -- 9 for three generic blocks plus one
SWAP's 3 -- with final positions [0, 2, 1], i.e. logical qubits 1 and 2
genuinely ended on each other's physical qubit, and the noiseless TVD
against the logical circuit was 0.0128. The measurement path now handles
both the tape's own wire renumbering and routing SWAPs.

## 1. Results (the tests that changed; all 32 passed)

| Test | Output |
|---|---|
| bitstring width (Addendum 160 P1) | widths {4} |
| noiseless, original tape | TVD 0.0178 |
| noisy FakeManilaV2, original tape | TVD 0.1198 |
| renumbering tape (Addendum 161 failure) | 6 routed 2q gates, TVD 0.0176 |
| triangle tape | final positions [0, 2, 1], 12 routed 2q gates, TVD 0.0128 |

Unchanged suites: `test_weakness_probes.py` 10/10,
`test_pennylane_gpu_ibm_pipeline_mock.py` 7/7,
`test_gpu_real_verification.py` 3/3 (worst GPU difference 8.771e-15),
`test_full_chain_gpu.py` 3/3.

## 2. Scoring (Addendum 161, Section 4)

- **R1 -- CONFIRMED.** Renumbering tape: TVD 0.0176 < 0.1, widths {4}.
- **R2 -- CONFIRMED.** Triangle tape: 12 > 9 routed two-qubit gates (a SWAP
  was inserted), final layout [0, 2, 1] (a real logical-to-physical
  permutation), widths {3}, TVD 0.0128 < 0.1.
- **R3 -- CONFIRMED.** The other 30 tests pass.

## 3. Status before 2026-09-28

Done:
- real SamplerV2 submission path, verified in local testing mode
  (Addendum 155);
- measurement restricted to logical qubits at final routed positions,
  in the original tape's wire order, verified with and without SWAPs
  (Addenda 160-162).

Remaining:
1. Save an IBM Quantum account by typing it into a terminal (never in
   code, a repository, or a chat).
2. Choose the target device, and re-run the local-mode suite against a
   fake backend of the same size and topology before submitting.
3. Decide which circuits to submit (Addendum 151's four XOR-input circuits
   were prepared for a generic heavy-hex lattice, not a specific device,
   and predate Addenda 160-162).

## 4. Files

| File | What it is |
|---|---|
| `psf_pennylane_gpu_ibm_prototype.py` | logical_measurement with the logical-to-circuit mapping |
| `test_real_submit_local_mode.py` | 9 tests, including the renumbering and triangle tapes |
