# Addendum 155 -- After the qubit-order fix, the real-GPU check and the real SamplerV2 submission path both pass (13/13); all four Addendum 153 predictions hold, including the known measure_all() width issue (2026-09-24 night)

**Pre-registered in**:
`spare-qubit-cliff-addendum-153-preregistration-2026-09-24.md`. The GPU
fix itself is described in Addendum 154. Every result below was observed as
raw text output in the conversation, per Addendum 154's standing rule.

## 0. In one line

With the corrected `psf_pennylane_gpu_real.py` in place (confirmed by
`grep` for `wires=[1, 0]` and file size before running), all 13 tests
across three suites passed: the real-`lightning.gpu` block check now agrees
with the CPU reference to 8.771e-15 over five random unitaries, the full
chain passes on real GPU hardware, and the real `qiskit_ibm_runtime.SamplerV2`
submission path, run in IBM's own local testing mode, satisfies every
contract the mock satisfied. All four pre-registered predictions hold.

## 1. Results (raw output, WSL2, RTX 4070, Qiskit 2.5.2)

| Suite | Tests | Result |
|---|---:|---|
| `test_gpu_real_verification.py` | 3 | 3 passed (worst GPU/CPU expectation difference 8.771e-15) |
| `test_full_chain_gpu.py` | 3 | 3 passed |
| `test_real_submit_local_mode.py` | 7 | 7 passed |
| **Total** | **13** | **13 passed** |

Together with the 17 mocked-connection tests observed earlier the same
night (Addendum 152, Section 5, the two rows Addendum 154 did NOT
invalidate), 30 tests have now been observed passing as raw output.

## 2. Scoring (Addendum 153)

**P1 (contracts survive the swap from mock to real SamplerV2) --
CONFIRMED.** Submission called exactly once; counts sum to the requested
4000 shots; invalid shots (0, -5, 100.7, "100", None) rejected before any
submission; an unmeasured circuit rejected; a submission-stage failure
propagated rather than swallowed.

**P2 (noiseless target, TVD < 0.1) -- CONFIRMED.** TVD = 0.0178 at 4000
shots, submitting to a plain `AerSimulator` with routing against
`FakeManilaV2`.

**P3 (device-snapshot noise, 0.01 < TVD < 0.3) -- CONFIRMED.** TVD = 0.1297
submitting to `FakeManilaV2` -- noise clearly applied, the distribution not
destroyed.

**P4 (known issue, bitstring width = backend width) -- CONFIRMED as
predicted.** Widths {5} for a 5-qubit backend and a 4-qubit circuit.
**Must be fixed before the 2026-09-28 submission** (measure only the
logical qubits): on a 127-qubit device this yields 127-bit results and makes
local simulation infeasible.

## 3. What this establishes and does not

- **Establishes**: the replacement for `mock_ibm_submit` uses the same
  SamplerV2 call real hardware uses, and behaves correctly end to end in
  IBM's own local testing mode, behind a GPU-verified synthesis step that
  is now itself verified to machine precision.
- **Does not establish**: anything on real hardware, queueing or
  authentication (deferred to 2026-09-28); whether `FakeManilaV2`'s
  snapshot matches any current device calibration; any timing.

## 4. Before 2026-09-28

1. Fix P4: measure only the logical qubits in
   `psf_pennylane_gpu_ibm_transform` (currently `measure_all()` on the full
   routed width).
2. Save an IBM Quantum account in a terminal (never in code or chat), then
   select a real backend via `get_saved_account_backend(name)`.
3. Re-run this suite's local-mode tests against a fake backend matching the
   chosen device's size, once P4 is fixed.

## 5. Files

| File | What it is |
|---|---|
| `psf_pennylane_gpu_real.py` | fixed (Addendum 154) |
| `psf_ibm_real_submit.py` | real SamplerV2 submission function |
| `test_gpu_real_verification.py`, `test_full_chain_gpu.py`, `test_real_submit_local_mode.py` | the three suites above |
