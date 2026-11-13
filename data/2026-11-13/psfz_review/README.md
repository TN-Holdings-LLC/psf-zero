# psfz: review of an outside proposal for all ten source files (2026-11-13, home)

An outside AI proposed new versions of every file in `rust/psfz/src`. Each file was built alone on top of the current
code, its tests run, and its outputs compared bit for bit with the current binary:

- compile, template and bind: 484 cases (44 inputs: chains and rings, every qubit measured, some measured, one
  measured, an empty and a one-qubit circuit; on 11 devices: FakeFez, FakeMarrakesh, FakeTorino, FakeSherbrooke,
  FakeBrisbane and six damaged copies of FakeFez and FakeSherbrooke). `compare_bins.py`.
- gate: 629 records (N2's refusals, controls, requests and fault injection, and 600 mutated records). `compare_gate.py`.
- twin: the harness `rust/psfz/harness/twin_check.py` (exactness against Qiskit's state vector, n up to 20; size and
  time at n = 50, 80).

| file | finding | taken |
|---|---|---|
| kak.rs | two of the three sign constants are wrong (S_XX, S_YY in the magic basis); every decomposition fails, its own tests too | no |
| linalg.rs | changes 337 of 484 compile outputs, also with every qubit measured (constants and division in other bits) | no |
| compile.rs | identical when every qubit is measured; changes 267 outputs when some are not (keeps the final rz on unmeasured qubits, which psfz drops by its contract: outcome distribution) and N2 controls K1/K2 (rz 408 -> 419). Its guard for a circuit with no qubits fixes a real panic | the empty-circuit guard only (written here, 4 lines) |
| mps.rs | "SVD bug for m < k" not a bug: the extra singular values are about 1e-155 of the largest and the 1e-12 cut drops them; the kept U is orthonormal to 1e-15. "z_all O(chi^2)" not so: still chi^3. But the twin is 1.8-10x faster, same bond dimensions, exact to 3.5e-15 (before: 4.2e-15) | yes |
| main.rs | identical outputs on good input; bind no longer writes `rz(NaN)` QASM for a value that is not a number (checked: the current code did); twin on an ECR device now writes a report and exits 2 instead of exiting 1 | yes |
| template.rs | does not compile (one type error); with it fixed, identical outputs; checks of a malformed TEMPLATE.json | no (small gain) |
| place.rs | "failed qubits not checked in ring_on" not so: `usable` calls `working`, which checks both qubits; identical outputs | no (small gain) |
| gate.rs, device.rs, qasm.rs, json.rs, sha256.rs | identical outputs on every case | no (small gain) |

The twin report's version reads `psfz twin prototype (2026-11-13)`: its numbers now differ from the 2026-10-20 build in
the last bits (largest seen 1.0e-14 on 40 qubits). The compiler (compile, template, bind) and gate outputs are
bit-identical to the current build on every case above, except the circuit with no qubits: a refusal (exit 2) instead
of a panic (exit 101).

Twin times (simulate, seconds; `twin_check_before.json`, `twin_check_after.json`):

| n, steps | chi | before | after |
|---|---|---|---|
| 20, 4 | 16 | 0.081 | 0.016 |
| 50, 6 | 64 | 11.5 | 4.3 |
| 80, 6 | 64 | 14.5 | 8.2 |
