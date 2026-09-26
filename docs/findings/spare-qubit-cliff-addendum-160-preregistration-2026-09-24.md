# Addendum 160 -- Pre-registration: fix Addendum 153's P4 -- measure only the logical qubits, at their final routed positions, before the 2026-09-28 submission (2026-09-24 night)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

Addendum 155 confirmed the known issue Addendum 153 flagged in advance:
`psf_pennylane_gpu_ibm_transform` calls `measure_all()` on the circuit
routed to the FULL backend, so every physical qubit is measured (5-bit
results for a 4-qubit circuit on FakeManilaV2; 127-bit results on a
127-qubit device, and an infeasible local simulation). This must be fixed
before real submission.

## 2. The fix

- `logical_measurement(qc_routed)` (new, in
  `psf_pennylane_gpu_ibm_prototype.py`): reads the routed circuit's own
  `layout.final_index_layout(filter_ancillas=True)` -- where each logical
  qubit actually ends up after routing, including any SWAPs -- and measures
  logical qubit i at that physical position into classical bit i. Raises
  (no silent fallback) if the routed circuit carries no layout.
- The transform uses it instead of `measure_all()`.
- `reference_local_counts` (the mock sampler) previously ignored which
  qubits were measured and always returned all qubits; it now returns the
  distribution over the measured qubits in classical-bit order, so the
  mock and the real SamplerV2 path agree on what a result means. For
  circuits measured with `measure_all()` (as in the existing weakness
  tests) this is unchanged.
- `test_real_submit_local_mode.py`: TVD is now computed against the
  LOGICAL circuit's own exact distribution (the tape converted to Qiskit,
  before any routing), not the routed full-width circuit -- a stronger
  check, since a wrong logical-to-physical mapping would now show up as a
  large TVD. The known-issue test is replaced by one asserting the fix. A
  new test uses a tape whose two blocks act on non-adjacent qubits of
  FakeManilaV2's linear chain, so routing must insert SWAPs.

## 3. Pre-registered predictions

**P1.** Result bitstrings are 4 bits wide (the logical qubit count), not 5.

**P2.** Noiseless TVD against the logical circuit's exact distribution
< 0.1, both for the original tape and for the SWAP-requiring tape. **If
the SWAP tape fails this while the original passes, the logical-to-physical
mapping is wrong.**

**P3.** Noisy (FakeManilaV2) TVD stays within 0.01-0.3, and is LOWER than
Addendum 155's 0.1297 for the same tape and seed: the idle fifth qubit's
readout error no longer enters the result. (Weak prediction: the change is
expected to be small.)

**P4.** Every previously passing suite still passes unchanged
(`test_weakness_probes.py`, `test_pennylane_gpu_ibm_pipeline_mock.py`,
`test_gpu_real_verification.py`, `test_full_chain_gpu.py`), plus the
updated `test_real_submit_local_mode.py`.

## 4. What this cannot establish

Real hardware (2026-09-28); devices other than FakeManilaV2's snapshot;
timing.
