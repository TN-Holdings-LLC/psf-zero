# psfz r5 templates (prototype): check of 2026-10-19 (exploratory; not pre-registered)

Run at home on 2026-10-19, about 14:20 CEST, in the cloud workspace (CPU), with
`python rust/psfz/harness/template_check.py OUT` from the repository root (psfz built from this commit). Seed
20261019. Raw rows: `template_check.json`.

**What was checked,** on FakeFez and FakeTorino (CZ), for TOQB's ring classifier (n = 12, L = 1, 2, 8, 16; every
ry and rz a slot) and TFIM chains (n = 6, 10, 16; rx and u slots, rzz fixed):

| check | result |
|---|---|
| S1, fixed shape: templates built at three random value sets have the same gate list | 14 of 14 |
| exactness: template built at values A, bound to values C, against the input at C (largest difference of an outcome's probability, state vector on the active qubits) | at most 2.4e-16 |
| the bound angles against the template built directly at C | at most 6.9e-18 (mod 2 pi) |
| `compile` (r4) unchanged: its QASM and report on 32 inputs x 3 devices (FakeFez, FakeTorino, FakeBrisbane) against the r4 binary | 96 of 96 identical (timings aside) |
| unit tests (`cargo test --release`) | 14 passed |

**Cost and speed (bound template against r4's `compile` at the same values; one measurement each):**

| family | sx / cz, template | sx / cz, r4 | bind in psfz (in process) | build |
|---|---|---|---|---|
| ring classifier n 12, L 8 | 216 / 96 | 216 / 96 | 0.15-0.17 ms | 2.9 ms |
| ring classifier n 12, L 16 | 408 / 192 | 408 / 192 | 0.28-0.29 ms | 8.8 ms |
| TFIM n 6 / 10 / 16 | 81 / 30, 189 / 72, 236 / 90 | 69 / 30, 159 / 72, 204 / 90 | 0.04-0.16 ms | - |

- **Vanity figure:** binding a 12-qubit, 16-layer template takes 0.28 ms in Rust against 4.7 ms for psf_compile's
  Python template (measured the same day on the same cloud CPU; the Python figure includes building the Qiskit
  circuit, the Rust figure does not include starting the process).
- **Essential figure:** on the ring classifier the template costs nothing (same sx and cz as r4 at generic values);
  on TFIM chains it writes 16-19 % more sx than r4, because every run with a slot is written in the general form,
  while r4 shortens runs whose values allow it (for example a run that is diagonal at those values). Removing this
  needs a structural rule for runs that are diagonal for every value (not done).
- Two-qubit gates with parameters are fixed in a template (not slots); ECR devices are refused.
