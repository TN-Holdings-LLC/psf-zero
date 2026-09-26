# Addendum 153 -- Pre-registration: a REAL IBM submission function (qiskit-ibm-runtime SamplerV2), verified in local testing mode against the same connection contracts the mock satisfied -- no credentials, no network (2026-09-24 night)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

Addendum 152's connection tests used `mock_ibm_submit`, which samples an
exact statevector and never touches IBM's own software stack. IBM
Quantum's free device time resets 2026-09-28 (Addendum 147-151). This
experiment replaces the mock with a function that uses the SAME call real
hardware submission uses -- `qiskit_ibm_runtime.SamplerV2(mode=...)` -- and
exercises it in IBM's own documented "local testing mode": passing a fake
backend from `qiskit_ibm_runtime.fake_provider` (a snapshot of a real QPU's
coupling map, basis gates and noise) or a Qiskit Aer simulator as `mode`
runs the job locally, with no credentials and no network call. Per IBM's own
documentation, moving from this to a real QPU should require changing only
the backend object.

## 2. Design

**New file** `psf_ibm_real_submit.py`:
- `make_sampler_submit_fn(mode, seed_simulator=None)` returns a callable
  with the SAME signature the connection already injects,
  `submit(circuit, shots) -> dict[str, int]`, backed by a real
  `SamplerV2(mode=mode)` call.
- Validates shots with the prototype's own `_validate_shots` (same rule as
  the mock, not a second, divergent copy), and rejects a circuit with no
  measurements before submitting (SamplerV2 requires measurements).
- Counts come from `pub_result.join_data().get_counts()`, not from assuming
  a register name.
- `get_saved_account_backend(name)` (for 2026-09-28 only; NOT called by any
  test here) loads a backend from an account saved beforehand with
  `QiskitRuntimeService.save_account(...)` typed directly in a terminal.
  It takes no token argument by design: credentials never appear in code,
  in a repository, or in a chat.

**Routing backend**: `FakeManilaV2` (5 qubits, a real IBM device snapshot),
used for both routing and noisy submission; the 4-qubit, two-block test
tape from Addendum 152's suites.

**Submission targets**: (a) `FakeManilaV2` (noisy, device snapshot);
(b) a plain `AerSimulator()` (noiseless), with routing still against
`FakeManilaV2` -- isolates "does the submission path return the right
distribution" from "how much does device noise change it".

**Measured**: call count, shots accounting, bitstring width, and total
variation distance (TVD) between sampled counts and the routed circuit's
own exact statevector distribution. `seed_simulator` fixed for
reproducibility.

## 3. Pre-registered predictions

**P1 (contracts survive the swap).** With the real SamplerV2 submission
function in place of the mock: submission is called exactly once; counts
sum to the requested shots; invalid shots are rejected BEFORE any
submission; a circuit without measurements is rejected; a submission-stage
failure propagates to the caller rather than being swallowed.

**P2 (noiseless target reproduces the exact distribution).** Submitting to
a noiseless `AerSimulator`, 4000 shots: TVD against the exact distribution
< 0.1 (sampling noise only).

**P3 (device-snapshot noise is visible but not destructive).** Submitting
to `FakeManilaV2`, 4000 shots: 0.01 < TVD < 0.3. Below 0.01 would suggest
the noise model is not actually being applied; above 0.3 would suggest
something is wrong beyond ordinary device noise.

**P4 (a known issue, stated before running, documented not fixed).** The
returned bitstrings have width equal to `backend.num_qubits` (5), not the
circuit's 4 logical qubits, because `psf_pennylane_gpu_ibm_transform` calls
`measure_all()` on the circuit routed to the FULL backend. On a
127-qubit device this means 127-bit strings on real hardware and an
infeasible local simulation. **This must be fixed before the 2026-09-28
submission** (measure only the logical qubits); this addendum records the
behavior rather than changing the transform in the same step.

## 4. What this cannot establish

- Anything about real hardware, queueing, or authentication -- deferred to
  2026-09-28.
- Whether the fake backend's snapshot matches any specific device's CURRENT
  calibration.
- Timing of any kind.
